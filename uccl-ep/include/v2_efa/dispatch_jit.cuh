#pragma once

#include "v2_efa/descriptor.hpp"
#include "v2_efa/proxy_queue.cuh"

namespace uccl::v2_efa {

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts,
          int kNumTopk, int kHiddenBytes>
__global__ void v2_efa_dispatch_descriptor_kernel(
    const int64_t* topk_idx, DispatchSegmentDescriptor* segments,
    DispatchExpertBatch* batches, uint32_t* counters, int num_tokens,
    int scaleout_rank, int scaleup_rank) {
  // This is a scaffold for the real DeepEP V2 JIT port. The production kernel
  // will derive segments from V2 token layout and write receiver expanded slots
  // directly, replacing only the NCCL Gin scaleout transfer section.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counters[0] = 0;
    counters[1] = 0;
  }
}

}  // namespace uccl::v2_efa
