#pragma once

#include "v2_efa/descriptor.hpp"
#include "v2_efa/proxy_queue.cuh"
#include "v2_efa/transfer_cmd.hpp"

namespace uccl::v2_efa {

namespace detail {

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t make_dispatch_flags(int scale_bytes,
                                                        bool has_topk_weight) {
  uint32_t flags = 0;
  if (scale_bytes > 0) {
    flags |= static_cast<uint32_t>(DescriptorFlags::kHasScale);
  }
  if (has_topk_weight) {
    flags |= static_cast<uint32_t>(DescriptorFlags::kHasTopkWeight);
  }
  return flags;
}

#endif

}  // namespace detail

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHiddenBytes>
__global__ void v2_efa_dispatch_descriptor_kernel(
    const int64_t* topk_idx, DispatchSegmentDescriptor* segments,
    DispatchExpertBatch* batches, uint32_t* counters, int num_tokens,
    int scaleout_rank, int scaleup_rank, int scale_bytes,
    bool has_topk_weight, int max_segments, int max_batches) {
  // Device-side reference generator. It intentionally mirrors the CPU
  // reference planner before we parallelize descriptor construction inside the
  // real DeepEP V2 JIT dispatch path.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)scaleout_rank;
  (void)scaleup_rank;

  constexpr int kWorldSize = kNumScaleoutRanks * kNumScaleupRanks;
  static_assert(kWorldSize > 0, "invalid V2 EFA topology");
  static_assert(kNumExperts % kWorldSize == 0,
                "num experts must be divisible by world size");
  constexpr int kExpertsPerRank = kNumExperts / kWorldSize;

  const auto flags = detail::make_dispatch_flags(scale_bytes, has_topk_weight);
  int num_segments = 0;
  int num_batches = 0;

  for (int expert = 0; expert < kNumExperts; ++expert) {
    const int owner_rank = expert / kExpertsPerRank;
    const int dst_scaleout_rank = owner_rank / kNumScaleupRanks;
    const int dst_scaleup_lane = owner_rank % kNumScaleupRanks;

    int first_segment = num_segments;
    int total_tokens = 0;
    int current_segment = -1;

    for (int token = 0; token < num_tokens; ++token) {
      for (int topk_slot = 0; topk_slot < kNumTopk; ++topk_slot) {
        const auto routed_expert = static_cast<int>(
            topk_idx[token * kNumTopk + topk_slot]);
        if (routed_expert != expert) {
          continue;
        }

        const int expanded_slot = total_tokens++;
        bool can_extend = false;
        if (current_segment >= first_segment) {
          auto& last = segments[current_segment];
          can_extend = last.src_token_index_offset < 0 &&
                       last.topk_slot == topk_slot &&
                       last.src_token_begin + last.count == token &&
                       last.expanded_slot_begin + last.count == expanded_slot &&
                       last.payload_bytes == kHiddenBytes &&
                       last.scale_bytes == scale_bytes && last.flags == flags;
        }

        if (can_extend) {
          segments[current_segment].count += 1;
          continue;
        }

        if (num_segments >= max_segments) {
          counters[kDescriptorCounterSegments] = num_segments;
          counters[kDescriptorCounterBatches] = num_batches;
          counters[kDescriptorCounterOverflow] = 1;
          return;
        }

        current_segment = num_segments++;
        auto& segment = segments[current_segment];
        segment.dst_scaleout_rank = dst_scaleout_rank;
        segment.dst_scaleup_lane = dst_scaleup_lane;
        segment.expert_id = expert;
        segment.count = 1;
        segment.src_token_begin = token;
        segment.src_token_index_offset = -1;
        segment.topk_slot = topk_slot;
        segment.expanded_slot_begin = expanded_slot;
        segment.payload_bytes = kHiddenBytes;
        segment.scale_bytes = scale_bytes;
        segment.flags = flags;
      }
    }

    if (total_tokens == 0) {
      continue;
    }
    if (num_batches >= max_batches) {
      counters[kDescriptorCounterSegments] = num_segments;
      counters[kDescriptorCounterBatches] = num_batches;
      counters[kDescriptorCounterOverflow] = 1;
      return;
    }

    auto& batch = batches[num_batches++];
    batch.dst_scaleout_rank = dst_scaleout_rank;
    batch.dst_scaleup_lane = dst_scaleup_lane;
    batch.expert_id = expert;
    batch.first_segment = first_segment;
    batch.num_segments = num_segments - first_segment;
    batch.total_tokens = total_tokens;
    batch.reserved = 0;
  }

  counters[kDescriptorCounterSegments] = num_segments;
  counters[kDescriptorCounterBatches] = num_batches;
  counters[kDescriptorCounterOverflow] = 0;
}

__global__ void v2_efa_dispatch_enqueue_proxy_kernel(
    const DispatchSegmentDescriptor* segments,
    const DispatchExpertBatch* batches, int num_batches,
    ProxyQueueView queue, DispatchProxyLayout layout) {
  // Device-side reference enqueue. The high-throughput version will split
  // batches across CTAs/warps, but it must preserve this command ordering:
  // all payload writes for a batch first, then one signal command.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& batch = batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx =
          static_cast<uint32_t>(batch.first_segment + i);
      enqueue_proxy_command(
          queue, make_dispatch_payload_command(
                     segments[segment_idx], segment_idx,
                     static_cast<uint32_t>(batch_idx), layout));
    }
    enqueue_proxy_command(
        queue, make_dispatch_signal_command(
                   batch, static_cast<uint32_t>(batch_idx), layout));
  }
}

__global__ void v2_efa_dispatch_enqueue_transfer_kernel(
    const DispatchSegmentDescriptor* segments,
    const DispatchExpertBatch* batches, int num_batches,
    V2TransferQueueView queue, DispatchProxyLayout layout) {
  // Preferred native V2 command-ring path. It writes 64-byte V2TransferCmd
  // entries directly instead of round-tripping through the legacy command
  // format or the temporary ProxyCommand host representation.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& batch = batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx =
          static_cast<uint32_t>(batch.first_segment + i);
      enqueue_v2_transfer_cmd(
          queue, make_v2_dispatch_payload_cmd(
                     segments[segment_idx], segment_idx,
                     static_cast<uint32_t>(batch_idx), layout));
    }
    enqueue_v2_transfer_cmd(
        queue, make_v2_dispatch_signal_cmd(
                   batch, static_cast<uint32_t>(batch_idx), layout));
  }
}

}  // namespace uccl::v2_efa
