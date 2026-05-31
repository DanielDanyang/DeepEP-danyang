#pragma once

#include "descriptor.hpp"
#include "transfer_cmd.hpp"
#include "transfer_d2h_queue.cuh"

namespace uccl::v2_efa {

namespace detail {

#if defined(__CUDA_ARCH__)
struct CombineDescriptorBuildResult {
  int num_segments = 0;
  int num_batches = 0;
  int overflow = 0;
};

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHidden>
__device__ __forceinline__ CombineDescriptorBuildResult build_combine_descriptors(
    const DispatchSegmentDescriptor* dispatch_segments,
    const DispatchExpertBatch* dispatch_batches, int num_dispatch_batches,
    CombineSegmentDescriptor* segments, CombineExpertBatch* batches,
    uint32_t* counters, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches) {
  (void)kNumScaleoutRanks;
  (void)kNumExperts;
  (void)kNumTopk;
  (void)kHidden;

  int num_segments = 0;
  int num_batches = 0;
  CombineDescriptorBuildResult result;
  const int dst_scaleout_rank = dst_original_rank / kNumScaleupRanks;
  const int dst_scaleup_lane = dst_original_rank % kNumScaleupRanks;

  for (int i = 0; i < max_batches; ++i) {
    batches[i] = CombineExpertBatch{};
  }

  for (int dispatch_batch_idx = 0; dispatch_batch_idx < num_dispatch_batches;
       ++dispatch_batch_idx) {
    const auto& dispatch_batch = dispatch_batches[dispatch_batch_idx];
    if (dispatch_batch.total_tokens == 0) {
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

    const int first_segment = num_segments;
    auto& combine_batch = batches[num_batches++];
    combine_batch.dst_original_rank = dst_original_rank;
    combine_batch.dst_scaleout_rank = dst_scaleout_rank;
    combine_batch.dst_scaleup_lane = dst_scaleup_lane;
    combine_batch.src_scaleout_rank = dispatch_batch.dst_scaleout_rank;
    combine_batch.expert_id = dispatch_batch.expert_id;
    combine_batch.first_segment = first_segment;
    combine_batch.total_tokens = 0;
    combine_batch.reserved = 0;

    for (int i = 0; i < dispatch_batch.num_segments; ++i) {
      if (num_segments >= max_segments) {
        counters[kDescriptorCounterSegments] = num_segments;
        counters[kDescriptorCounterBatches] = num_batches;
        counters[kDescriptorCounterOverflow] = 1;
        result.num_segments = num_segments;
        result.num_batches = num_batches;
        result.overflow = 1;
        return result;
      }

      const auto& dispatch_segment =
          dispatch_segments[dispatch_batch.first_segment + i];
      auto& segment = segments[num_segments++];
      segment.dst_original_rank = dst_original_rank;
      segment.dst_scaleout_rank = dst_scaleout_rank;
      segment.dst_scaleup_lane = dst_scaleup_lane;
      segment.src_scaleout_rank = dispatch_batch.dst_scaleout_rank;
      segment.expert_id = dispatch_batch.expert_id;
      segment.count = dispatch_segment.count;
      segment.expanded_slot_begin = dispatch_segment.expanded_slot_begin;
      segment.expanded_slot_index_offset = -1;
      segment.topk_slot = dispatch_segment.topk_slot;
      segment.reduced_token_slot = dispatch_segment.src_token_begin;
      segment.payload_bytes = payload_bytes;
      segment.flags = static_cast<uint32_t>(DescriptorFlags::kReduce);
      combine_batch.total_tokens += dispatch_segment.count;
    }

    combine_batch.num_segments = num_segments - first_segment;
  }

  counters[kDescriptorCounterSegments] = num_segments;
  counters[kDescriptorCounterBatches] = num_batches;
  counters[kDescriptorCounterOverflow] = 0;
  result.num_segments = num_segments;
  result.num_batches = num_batches;
  result.overflow = 0;
  return result;
}

__device__ __forceinline__ void enqueue_combine_d2h(
    const CombineSegmentDescriptor* segments, const CombineExpertBatch* batches,
    int num_batches, V2TransferD2HQueueView queue,
    CombineTransferLayout layout) {
  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& batch = batches[batch_idx];
    if (batch.num_segments <= 0 || batch.total_tokens <= 0) {
      continue;
    }
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx =
          static_cast<uint32_t>(batch.first_segment + i);
      enqueue_v2_transfer_d2h(
          queue, make_v2_combine_payload_cmd(
                     segments[segment_idx], segment_idx,
                     static_cast<uint32_t>(batch_idx), layout));
    }
    enqueue_v2_transfer_d2h(
        queue, make_v2_combine_signal_cmd(
                   batch, static_cast<uint32_t>(batch_idx), layout));
  }
}

#endif

}  // namespace detail

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHidden>
__global__ void v2_efa_combine_descriptor_kernel(
    const DispatchSegmentDescriptor* dispatch_segments,
    const DispatchExpertBatch* dispatch_batches, int num_dispatch_batches,
    CombineSegmentDescriptor* segments, CombineExpertBatch* batches,
    uint32_t* counters, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches) {
  // Device-side reference generator. The production V2 kernel will derive
  // equivalent descriptors from token_metadata_at_forward/channel_linked_list;
  // this form is useful for roundtrip validation against dispatch descriptors.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)kNumScaleoutRanks;
  (void)kNumScaleupRanks;
  (void)kNumExperts;
  (void)kNumTopk;
  (void)kHidden;

  (void)detail::build_combine_descriptors<kNumScaleoutRanks, kNumScaleupRanks,
                                          kNumExperts, kNumTopk, kHidden>(
      dispatch_segments, dispatch_batches, num_dispatch_batches, segments,
      batches, counters, dst_original_rank, payload_bytes, max_segments,
      max_batches);
}

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHidden>
__global__ void v2_efa_combine_descriptor_enqueue_d2h_kernel(
    const DispatchSegmentDescriptor* dispatch_segments,
    const DispatchExpertBatch* dispatch_batches, int num_dispatch_batches,
    CombineSegmentDescriptor* segments, CombineExpertBatch* batches,
    uint32_t* counters, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches, V2TransferD2HQueueView queue,
    CombineTransferLayout layout) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const auto result = detail::build_combine_descriptors<
      kNumScaleoutRanks, kNumScaleupRanks, kNumExperts, kNumTopk, kHidden>(
      dispatch_segments, dispatch_batches, num_dispatch_batches, segments,
      batches, counters, dst_original_rank, payload_bytes, max_segments,
      max_batches);
  if (result.overflow != 0) {
    return;
  }
  detail::enqueue_combine_d2h(segments, batches, max_batches, queue, layout);
}

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHidden>
__global__ void v2_efa_combine_forward_metadata_enqueue_d2h_kernel(
    const int32_t* token_metadata_at_forward,
    CombineSegmentDescriptor* segments, CombineExpertBatch* batches,
    uint32_t* counters, int num_forward_rows, int scaleout_rank,
    int num_max_tokens_per_rank, int payload_bytes, int max_segments,
    int max_batches, V2TransferD2HQueueView queue,
    CombineTransferLayout layout) {
  // Transitional native V2 combine path: parse the V2-like multi-channel
  // forward metadata carried in the handle and generate V2TransferCmd directly
  // on device. The final version should also walk channel_linked_list to match
  // the official V2 scheduling order.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)kNumScaleoutRanks;
  (void)kNumExperts;
  (void)kHidden;

  constexpr int kMetadataDims = 2 + kNumTopk * 2;
  int num_segments = 0;
  int num_batches = 0;

  for (int i = 0; i < max_batches; ++i) {
    batches[i] = CombineExpertBatch{};
  }

  for (int row = 0; row < num_forward_rows; ++row) {
    const auto metadata = token_metadata_at_forward + row * kMetadataDims;
    const int src_global = metadata[0];
    if (src_global < 0) {
      continue;
    }
    const int dst_original_rank = src_global / num_max_tokens_per_rank;
    const int dst_scaleout_rank = dst_original_rank / kNumScaleupRanks;
    const int dst_scaleup_lane = dst_original_rank % kNumScaleupRanks;
    if (dst_scaleout_rank == scaleout_rank) {
      continue;
    }

    const int first_segment = num_segments;
    if (num_batches >= max_batches) {
      counters[kDescriptorCounterSegments] = num_segments;
      counters[kDescriptorCounterBatches] = num_batches;
      counters[kDescriptorCounterOverflow] = 1;
      return;
    }
    auto& batch = batches[num_batches];
    batch.dst_original_rank = dst_original_rank;
    batch.dst_scaleout_rank = dst_scaleout_rank;
    batch.dst_scaleup_lane = dst_scaleup_lane;
    batch.src_scaleout_rank = scaleout_rank;
    batch.expert_id = 0;
    batch.first_segment = first_segment;
    batch.num_segments = 0;
    batch.total_tokens = 0;
    batch.reserved = 0;

    for (int topk_slot = 0; topk_slot < kNumTopk; ++topk_slot) {
      const int source_slot = metadata[2 + kNumTopk + topk_slot];
      if (source_slot < 0) {
        continue;
      }
      if (num_segments >= max_segments) {
        counters[kDescriptorCounterSegments] = num_segments;
        counters[kDescriptorCounterBatches] = num_batches;
        counters[kDescriptorCounterOverflow] = 1;
        return;
      }
      auto& segment = segments[num_segments++];
      segment.dst_original_rank = dst_original_rank;
      segment.dst_scaleout_rank = dst_scaleout_rank;
      segment.dst_scaleup_lane = dst_scaleup_lane;
      segment.src_scaleout_rank = scaleout_rank;
      segment.expert_id = 0;
      segment.count = 1;
      segment.expanded_slot_begin = source_slot;
      segment.expanded_slot_index_offset = -1;
      segment.topk_slot = topk_slot;
      segment.reduced_token_slot = src_global % num_max_tokens_per_rank;
      segment.payload_bytes = payload_bytes;
      segment.flags = static_cast<uint32_t>(DescriptorFlags::kReduce);
      batch.num_segments += 1;
      batch.total_tokens += 1;
    }

    if (batch.num_segments > 0) {
      num_batches += 1;
    }
  }

  counters[kDescriptorCounterSegments] = num_segments;
  counters[kDescriptorCounterBatches] = num_batches;
  counters[kDescriptorCounterOverflow] = 0;
  detail::enqueue_combine_d2h(segments, batches, num_batches, queue, layout);
}

template <int kInstance>
__global__ void v2_efa_combine_enqueue_transfer_kernel(
    const CombineSegmentDescriptor* segments, const CombineExpertBatch* batches,
    int num_batches, V2TransferQueueView queue, CombineTransferLayout layout) {
  // Preferred native V2 command-ring path. Signal commands are emitted after
  // all payload commands for the same semantic batch.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)kInstance;

  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& batch = batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx =
          static_cast<uint32_t>(batch.first_segment + i);
      enqueue_v2_transfer_cmd(
          queue, make_v2_combine_payload_cmd(
                     segments[segment_idx], segment_idx,
                     static_cast<uint32_t>(batch_idx), layout));
    }
    enqueue_v2_transfer_cmd(
        queue, make_v2_combine_signal_cmd(
                   batch, static_cast<uint32_t>(batch_idx), layout));
  }
}

template <int kInstance>
__global__ void v2_efa_combine_enqueue_d2h_kernel(
    const CombineSegmentDescriptor* segments, const CombineExpertBatch* batches,
    int num_batches, V2TransferD2HQueueView queue,
    CombineTransferLayout layout) {
  // Production-facing scaffold: write compact V2 commands directly into a D2H
  // ring compatible with the retained CPU proxy model.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)kInstance;

  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& batch = batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx =
          static_cast<uint32_t>(batch.first_segment + i);
      enqueue_v2_transfer_d2h(
          queue, make_v2_combine_payload_cmd(
                     segments[segment_idx], segment_idx,
                     static_cast<uint32_t>(batch_idx), layout));
    }
    enqueue_v2_transfer_d2h(
        queue, make_v2_combine_signal_cmd(
                   batch, static_cast<uint32_t>(batch_idx), layout));
  }
}

}  // namespace uccl::v2_efa
