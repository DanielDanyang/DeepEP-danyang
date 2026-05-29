#pragma once

#include "v2_efa/descriptor.hpp"
#include "v2_efa/transfer_cmd.hpp"

namespace uccl::v2_efa {

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

  int num_segments = 0;
  int num_batches = 0;

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
      return;
    }

    const int first_segment = num_segments;
    auto& combine_batch = batches[num_batches++];
    combine_batch.dst_original_rank = dst_original_rank;
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
        return;
      }

      const auto& dispatch_segment =
          dispatch_segments[dispatch_batch.first_segment + i];
      auto& segment = segments[num_segments++];
      segment.dst_original_rank = dst_original_rank;
      segment.src_scaleout_rank = dispatch_batch.dst_scaleout_rank;
      segment.expert_id = dispatch_batch.expert_id;
      segment.count = dispatch_segment.count;
      segment.expanded_slot_begin = dispatch_segment.expanded_slot_begin;
      segment.expanded_slot_index_offset = -1;
      segment.topk_slot = dispatch_segment.topk_slot;
      segment.reduced_token_slot = dispatch_segment.src_token_begin;
      segment.payload_bytes = payload_bytes;
      segment.flags = static_cast<uint32_t>(DescriptorFlags::kReduce);
      segment.reserved = 0;
      combine_batch.total_tokens += dispatch_segment.count;
    }

    combine_batch.num_segments = num_segments - first_segment;
  }

  counters[kDescriptorCounterSegments] = num_segments;
  counters[kDescriptorCounterBatches] = num_batches;
  counters[kDescriptorCounterOverflow] = 0;
}

__global__ void v2_efa_combine_enqueue_transfer_kernel(
    const CombineSegmentDescriptor* segments, const CombineExpertBatch* batches,
    int num_batches, V2TransferQueueView queue, CombineTransferLayout layout) {
  // Preferred native V2 command-ring path. Signal commands are emitted after
  // all payload commands for the same semantic batch.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

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

}  // namespace uccl::v2_efa
