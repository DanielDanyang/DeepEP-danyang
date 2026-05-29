#pragma once

#include "v2_efa/descriptor.hpp"
#include "v2_efa/proxy_queue.cuh"

namespace uccl::v2_efa {

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHidden>
__global__ void v2_efa_combine_descriptor_kernel(
    const int* token_metadata_at_forward, const int* channel_linked_list,
    CombineSegmentDescriptor* segments, CombineExpertBatch* batches,
    uint32_t* counters, int num_reduced_tokens, int scaleout_rank,
    int scaleup_rank) {
  // This is a scaffold for the real DeepEP V2 JIT port. The production kernel
  // will generate reverse descriptors from V2 forward metadata and send
  // reduced-combine payloads directly to owner-rank reduced layout.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counters[0] = 0;
    counters[1] = 0;
  }
}

}  // namespace uccl::v2_efa
