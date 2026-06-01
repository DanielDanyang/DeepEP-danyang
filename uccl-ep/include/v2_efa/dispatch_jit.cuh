#pragma once

#include "descriptor.hpp"
#include "transfer_cmd.hpp"
#include "transfer_d2h_queue.cuh"

namespace uccl::v2_efa {

namespace detail {

#if defined(__CUDA_ARCH__)
struct DispatchDescriptorBuildResult {
  int num_segments = 0;
  int num_batches = 0;
  int overflow = 0;
};

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

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHiddenBytes>
__device__ __forceinline__ DispatchDescriptorBuildResult build_dispatch_descriptors(
    const int64_t* topk_idx, DispatchSegmentDescriptor* segments,
    DispatchExpertBatch* batches, uint32_t* counters, int num_tokens,
    int scale_bytes, bool has_topk_weight, int max_segments,
    int max_batches) {
  constexpr int kWorldSize = kNumScaleoutRanks * kNumScaleupRanks;
  static_assert(kWorldSize > 0, "invalid V2 EFA topology");
  static_assert(kNumExperts % kWorldSize == 0,
                "num experts must be divisible by world size");
  constexpr int kExpertsPerRank = kNumExperts / kWorldSize;

  const auto flags = make_dispatch_flags(scale_bytes, has_topk_weight);
  int num_segments = 0;
  int num_batches = 0;
  int compact_slot_cursor = 0;
  DispatchDescriptorBuildResult result;

  for (int i = 0; i < max_batches; ++i) {
    batches[i] = DispatchExpertBatch{};
  }

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
          result.num_segments = num_segments;
          result.num_batches = num_batches;
          result.overflow = 1;
          return result;
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
      result.num_segments = num_segments;
      result.num_batches = num_batches;
      result.overflow = 1;
      return result;
    }

    auto& batch = batches[num_batches++];
    batch.dst_scaleout_rank = dst_scaleout_rank;
    batch.dst_scaleup_lane = dst_scaleup_lane;
    batch.expert_id = expert;
    batch.first_segment = first_segment;
    batch.num_segments = num_segments - first_segment;
    batch.total_tokens = total_tokens;
    batch.reserved = compact_slot_cursor;
    compact_slot_cursor += total_tokens;
  }

  counters[kDescriptorCounterSegments] = num_segments;
  counters[kDescriptorCounterBatches] = num_batches;
  counters[kDescriptorCounterOverflow] = 0;
  result.num_segments = num_segments;
  result.num_batches = num_batches;
  result.overflow = 0;
  return result;
}

__device__ __forceinline__ void enqueue_dispatch_d2h(
    const DispatchSegmentDescriptor* segments,
    const DispatchExpertBatch* batches, int num_batches,
    V2TransferD2HQueueView queue, DispatchTransferLayout layout) {
  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& batch = batches[batch_idx];
    if (batch.num_segments <= 0 || batch.total_tokens <= 0) {
      continue;
    }
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx =
          static_cast<uint32_t>(batch.first_segment + i);
      auto segment = segments[segment_idx];
      segment.expanded_slot_begin += batch.reserved;
      enqueue_v2_transfer_d2h(
          queue, make_v2_dispatch_payload_cmd(
                     segment, segment_idx,
                     static_cast<uint32_t>(batch_idx), layout));
    }
    enqueue_v2_transfer_d2h(
        queue, make_v2_dispatch_signal_cmd(
                   batch, static_cast<uint32_t>(batch_idx), layout));
  }
  for (uint32_t target_rank = 0; target_rank < layout.num_ranks;
       ++target_rank) {
    enqueue_v2_transfer_d2h(
        queue, make_v2_dispatch_done_cmd(target_rank, layout));
  }
}

#endif

}  // namespace detail

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHiddenBytes>
__global__ void v2_efa_dispatch_descriptor_enqueue_d2h_kernel(
    const int64_t* topk_idx, DispatchSegmentDescriptor* segments,
    DispatchExpertBatch* batches, uint32_t* counters, int num_tokens,
    int scaleout_rank, int scaleup_rank, int scale_bytes,
    bool has_topk_weight, int max_segments, int max_batches,
    V2TransferD2HQueueView queue, DispatchTransferLayout layout) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)scaleout_rank;
  (void)scaleup_rank;

  const auto result = detail::build_dispatch_descriptors<
      kNumScaleoutRanks, kNumScaleupRanks, kNumExperts, kNumTopk, kHiddenBytes>(
      topk_idx, segments, batches, counters, num_tokens, scale_bytes,
      has_topk_weight, max_segments, max_batches);
  if (result.overflow != 0) {
    return;
  }
  detail::enqueue_dispatch_d2h(segments, batches, max_batches, queue, layout);
}

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumTopk,
          int kNumChannels>
__global__ void v2_efa_dispatch_forward_metadata_kernel(
    const int64_t* recv_topk_idx, const int32_t* recv_src_metadata,
    int32_t* token_metadata_at_forward, int32_t* channel_linked_list,
    int num_recv_tokens, int rows_per_channel, int scaleup_rank,
    bool do_expand) {
  constexpr int kMetadataDims = 2 + kNumTopk * 2;
  constexpr int kRecvMetadataDims = 2 + kNumTopk;
  static_assert(kNumChannels > 0, "invalid V2 dispatch channel count");
  static_assert(kNumScaleupRanks > 0, "invalid V2 scale-up rank count");
  (void)kNumScaleoutRanks;

  const int total_rows = kNumChannels * rows_per_channel;
  const int global_thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int global_stride = static_cast<int>(gridDim.x * blockDim.x);

  for (int flat_row = global_thread; flat_row < total_rows;
       flat_row += global_stride) {
    const int channel_idx = flat_row / rows_per_channel;
    const int channel_row = flat_row - channel_idx * rows_per_channel;
    const int token_row = channel_row * kNumChannels + channel_idx;

    auto* metadata = token_metadata_at_forward + flat_row * kMetadataDims;
    for (int i = 0; i < kMetadataDims; ++i) {
      metadata[i] = -1;
    }

    auto* linked =
        channel_linked_list + flat_row * kNumScaleupRanks;
    for (int lane = 0; lane < kNumScaleupRanks; ++lane) {
      linked[lane] = -1;
    }

    if (token_row >= num_recv_tokens || channel_row >= rows_per_channel - 1) {
      continue;
    }

    const auto* src_metadata =
        recv_src_metadata + token_row * kRecvMetadataDims;
    metadata[0] = src_metadata[0];
    metadata[1] = (token_row == num_recv_tokens - 1) ? 1 : 0;
    for (int topk_slot = 0; topk_slot < kNumTopk; ++topk_slot) {
      const int64_t local_expert =
          __ldg(recv_topk_idx + token_row * kNumTopk + topk_slot);
      if (local_expert < 0) {
        continue;
      }
      metadata[2 + topk_slot] = scaleup_rank;
      metadata[2 + kNumTopk + topk_slot] =
          do_expand ? src_metadata[2 + topk_slot] : token_row;
    }
    linked[scaleup_rank] = flat_row;
  }
}

template <int kWorldSize, int kNumScaleupRanks, int kNumExperts, int kNumTopk>
__global__ void v2_efa_dispatch_receiver_metadata_kernel(
    const int64_t* recv_topk_idx, const int32_t* recv_src_global,
    const int32_t* recv_counts_per_rank, int32_t* recv_src_metadata,
    int32_t* dst_buffer_slot_idx, int32_t* psum_num_recv_tokens_per_scaleup_rank,
    int32_t* psum_num_recv_tokens_per_expert, int32_t* expert_counts_aligned,
    int32_t* expert_counts_scratch, int32_t* next_expanded_scratch,
    int num_recv_tokens, int num_source_tokens, int num_max_tokens_per_rank,
    int rank, int expert_alignment, bool do_expand) {
  constexpr int kMetadataDims = 2 + kNumTopk;
  static_assert(kWorldSize > 0, "invalid V2 receiver world size");
  static_assert(kNumScaleupRanks > 0, "invalid V2 receiver scaleup ranks");
  static_assert(kNumExperts % kWorldSize == 0,
                "num experts must be divisible by world size");
  constexpr int kNumLocalExperts = kNumExperts / kWorldSize;

  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);

  for (int expert = tid; expert < kNumLocalExperts; expert += stride) {
    expert_counts_aligned[expert] = 0;
    expert_counts_scratch[expert] = 0;
    next_expanded_scratch[expert] = 0;
    psum_num_recv_tokens_per_expert[expert] = 0;
  }
  for (int lane = tid; lane < kNumScaleupRanks; lane += stride) {
    psum_num_recv_tokens_per_scaleup_rank[lane] = 0;
  }
  for (int idx = tid; idx < num_source_tokens * kNumTopk; idx += stride) {
    dst_buffer_slot_idx[idx] = -1;
  }
  for (int row = tid; row < num_recv_tokens; row += stride) {
    auto* metadata = recv_src_metadata + row * kMetadataDims;
    metadata[0] = recv_src_global[row];
    metadata[1] = 0;
    for (int slot = 0; slot < kNumTopk; ++slot) {
      metadata[2 + slot] = -1;
    }
  }
  __syncthreads();

  for (int row = tid; row < num_recv_tokens; row += stride) {
    for (int slot = 0; slot < kNumTopk; ++slot) {
      const int expert =
          static_cast<int>(__ldg(recv_topk_idx + row * kNumTopk + slot));
      if (expert >= 0 && expert < kNumLocalExperts) {
        atomicAdd(expert_counts_scratch + expert, 1);
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    int running = 0;
    for (int lane = 0; lane < kNumScaleupRanks; ++lane) {
      for (int peer = lane; peer < kWorldSize; peer += kNumScaleupRanks) {
        running += recv_counts_per_rank[peer];
      }
      psum_num_recv_tokens_per_scaleup_rank[lane] = running;
    }

    running = 0;
    for (int expert = 0; expert < kNumLocalExperts; ++expert) {
      const int raw = expert_counts_scratch[expert];
      const int aligned =
          ((raw + expert_alignment - 1) / expert_alignment) * expert_alignment;
      const int expanded_begin =
          ((running + expert_alignment - 1) / expert_alignment) *
          expert_alignment;
      expert_counts_aligned[expert] = aligned;
      next_expanded_scratch[expert] = expanded_begin;
      running = do_expand ? expanded_begin + raw : running + aligned;
      psum_num_recv_tokens_per_expert[expert] = running;
    }
  }
  __syncthreads();

  for (int row = tid; row < num_recv_tokens; row += stride) {
    auto* metadata = recv_src_metadata + row * kMetadataDims;
    const int src_global = metadata[0];
    const int src_rank = src_global / num_max_tokens_per_rank;
    const int src_token = src_global - src_rank * num_max_tokens_per_rank;
    int linked_slot = 0;

    for (int slot = 0; slot < kNumTopk; ++slot) {
      const int expert =
          static_cast<int>(__ldg(recv_topk_idx + row * kNumTopk + slot));
      if (expert < 0 || expert >= kNumLocalExperts) {
        continue;
      }
      linked_slot = row * kNumTopk + slot;
      if (do_expand) {
        metadata[2 + slot] = atomicAdd(next_expanded_scratch + expert, 1);
      }
      if (src_rank == rank && src_token >= 0 && src_token < num_source_tokens) {
        dst_buffer_slot_idx[src_token * kNumTopk + slot] =
            rank * num_max_tokens_per_rank + row;
      }
    }
    metadata[1] = linked_slot;
  }
}

template <int kNumTopk, int kHiddenBytes>
__global__ void v2_efa_dispatch_materialize_records_kernel(
    const uint8_t* window, const int32_t* batch_counts,
    const int32_t* batch_offsets, uint8_t* recv_x, int64_t* recv_topk_idx,
    float* recv_topk_weights, int32_t* recv_src_global, int num_sources,
    int max_batches, int num_max_tokens_per_rank, int num_experts, int rank,
    DispatchTransferLayout layout, bool has_topk_weight) {
  const int global_thread =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int global_stride = static_cast<int>(gridDim.x * blockDim.x);
  const int total_slots =
      num_sources * max_batches * num_max_tokens_per_rank;
  const int world_size = num_sources > 0 ? num_sources : 1;
  const int experts_per_rank = num_experts / world_size;
  const int num_local_experts = experts_per_rank > 0 ? experts_per_rank : 1;
  const int local_expert_begin = rank * num_local_experts;
  const int local_expert_end = local_expert_begin + num_local_experts;

  for (int linear = global_thread; linear < total_slots;
       linear += global_stride) {
    const int slot = linear % num_max_tokens_per_rank;
    const int batch_linear = linear / num_max_tokens_per_rank;
    const int batch = batch_linear % max_batches;
    const int source = batch_linear / max_batches;
    const int count = __ldg(batch_counts + source * max_batches + batch);
    const int out_begin = __ldg(batch_offsets + source * max_batches + batch);
    if (count <= 0 || out_begin < 0 || slot >= count) {
      continue;
    }

    const int out_row = out_begin + slot;
    int source_local_slot = slot;
    for (int prev_batch = 0; prev_batch < batch; ++prev_batch) {
      const int prev_count =
          __ldg(batch_counts + source * max_batches + prev_batch);
      source_local_slot += prev_count > 0 ? prev_count : 0;
    }
    const uint64_t record_offset =
        layout.remote_payload_base +
        static_cast<uint64_t>(source) * layout.source_rank_stride +
        static_cast<uint64_t>(source_local_slot) * layout.expanded_slot_stride;
    const auto* record = window + record_offset;
    auto* dst_payload = recv_x + static_cast<uint64_t>(out_row) * kHiddenBytes;
    const auto* src_payload = record + layout.record_payload_offset;
    for (int byte = 0; byte < kHiddenBytes; ++byte) {
      dst_payload[byte] = src_payload[byte];
    }

    const auto src_global_ptr = reinterpret_cast<const int32_t*>(
        record + layout.record_src_global_offset);
    recv_src_global[out_row] = *src_global_ptr;

    const auto topk_ptr = reinterpret_cast<const int64_t*>(
        record + layout.record_topk_idx_offset);
    for (int topk_slot = 0; topk_slot < kNumTopk; ++topk_slot) {
      const int64_t raw_expert = topk_ptr[topk_slot];
      int64_t local_expert = -1;
      if (raw_expert >= local_expert_begin && raw_expert < local_expert_end) {
        local_expert = raw_expert - local_expert_begin;
      }
      recv_topk_idx[out_row * kNumTopk + topk_slot] = local_expert;
    }

    if (has_topk_weight && recv_topk_weights != nullptr &&
        layout.record_topk_weight_bytes != 0) {
      const auto weight_ptr = reinterpret_cast<const float*>(
          record + layout.record_topk_weight_offset);
      for (int topk_slot = 0; topk_slot < kNumTopk; ++topk_slot) {
        recv_topk_weights[out_row * kNumTopk + topk_slot] =
            weight_ptr[topk_slot];
      }
    }
  }
}

template <int kInstance>
__global__ void v2_efa_dispatch_signal_offsets_kernel(
    const uint8_t* window, int32_t* batch_counts, int32_t* batch_offsets,
    int32_t* recv_counts_per_rank, int32_t* total_recv_tokens,
    int num_sources, int max_batches, DispatchTransferLayout layout) {
  if (blockIdx.x != 0) {
    return;
  }
  (void)kInstance;

  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);

  for (int source = tid; source < num_sources; source += stride) {
    const uint64_t done_offset =
        layout.remote_signal_base +
        static_cast<uint64_t>(source) * layout.source_signal_stride +
        static_cast<uint64_t>(max_batches) * layout.signal_stride;
    const auto* done_ptr =
        reinterpret_cast<const volatile uint32_t*>(window + done_offset);
    while (*done_ptr == 0u) {
#if defined(__CUDA_ARCH__)
      __nanosleep(64);
#endif
    }
  }
  __syncthreads();
#if defined(__CUDA_ARCH__)
  __threadfence_system();
#endif

  for (int linear = tid; linear < num_sources * max_batches;
       linear += stride) {
    const int batch = linear % max_batches;
    const int source = linear / max_batches;
    const uint64_t signal_offset =
        layout.remote_signal_base +
        static_cast<uint64_t>(source) * layout.source_signal_stride +
        static_cast<uint64_t>(batch) * layout.signal_stride;
    const auto* count_ptr =
        reinterpret_cast<const volatile uint32_t*>(window + signal_offset);
    batch_counts[linear] = static_cast<int>(*count_ptr);
    batch_offsets[linear] = -1;
  }
  __syncthreads();

  if (tid == 0) {
    int running = 0;
    for (int source = 0; source < num_sources; ++source) {
      int source_total = 0;
      for (int batch = 0; batch < max_batches; ++batch) {
        const int idx = source * max_batches + batch;
        const int count = batch_counts[idx];
        if (count > 0) {
          batch_offsets[idx] = running;
          running += count;
          source_total += count;
        }
      }
      recv_counts_per_rank[source] = source_total;
    }
    total_recv_tokens[0] = running;
  }
}

template <int kNumTopk, int kHiddenBytes>
__global__ void v2_efa_dispatch_expand_records_kernel(
    const uint8_t* recv_x, const float* recv_topk_weights,
    const int32_t* recv_src_metadata, uint8_t* expanded_x,
    float* expanded_topk_weights, int num_recv_tokens,
    int num_expanded_tokens, bool has_topk_weight) {
  const int global_thread =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int global_stride = static_cast<int>(gridDim.x * blockDim.x);
  const int total_routes = num_recv_tokens * kNumTopk;

  for (int linear = global_thread; linear < total_routes;
       linear += global_stride) {
    const int row = linear / kNumTopk;
    const int slot = linear - row * kNumTopk;
    const int expanded_idx =
        __ldg(recv_src_metadata + row * (2 + kNumTopk) + 2 + slot);
    if (expanded_idx < 0 || expanded_idx >= num_expanded_tokens) {
      continue;
    }
    const auto* src_payload =
        recv_x + static_cast<uint64_t>(row) * kHiddenBytes;
    auto* dst_payload =
        expanded_x + static_cast<uint64_t>(expanded_idx) * kHiddenBytes;
    for (int byte = 0; byte < kHiddenBytes; ++byte) {
      dst_payload[byte] = src_payload[byte];
    }
    if (has_topk_weight && recv_topk_weights != nullptr &&
        expanded_topk_weights != nullptr) {
      expanded_topk_weights[expanded_idx * kNumTopk + slot] =
          recv_topk_weights[row * kNumTopk + slot];
    }
  }
}

}  // namespace uccl::v2_efa
