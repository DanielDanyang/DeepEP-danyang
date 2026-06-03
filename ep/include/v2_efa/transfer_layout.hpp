#pragma once

#include <cstdint>

// Native V2 EFA transfer layout descriptors.
//
// These two POD structs were salvaged from the now-removed transfer_cmd.hpp.
// They describe how the JIT kernel maps V2 token records to RDMA payload/signal
// offsets; the obsolete V2TransferCmd / encode / sink machinery is gone — the
// native path emits the old 16B uccl TransferCmd (ring_buffer.cuh) instead.

namespace uccl::v2_efa {

struct DispatchTransferLayout {
  uint64_t local_payload_base = 0;
  uint64_t remote_payload_base = 0;
  uint64_t remote_signal_base = 0;
  uint32_t src_token_stride = 0;
  uint32_t expanded_slot_stride = 0;
  uint32_t batch_payload_stride = 0;
  uint32_t source_rank_stride = 0;
  uint32_t source_signal_stride = 0;
  uint32_t signal_stride = sizeof(uint32_t);
  uint32_t token_record_bytes = 0;
  uint32_t record_payload_offset = 0;
  uint32_t record_src_global_offset = 0;
  uint32_t record_topk_idx_offset = 0;
  uint32_t record_topk_weight_offset = 0;
  uint32_t record_topk_weight_bytes = 0;
  uint32_t num_scaleup_ranks = 0;
  uint32_t num_ranks = 0;
  uint32_t source_rank = 0;
  uint32_t efa_lane = 0;
  uint32_t num_efa_lanes = 1;
  uint32_t max_batches = 0;
  int32_t skip_scaleout_rank = -1;
};

struct CombineTransferLayout {
  uint64_t local_payload_base = 0;
  uint64_t remote_payload_base = 0;
  uint64_t remote_signal_base = 0;
  uint32_t expanded_slot_stride = 0;
  uint32_t reduced_token_stride = 0;
  uint32_t batch_payload_stride = 0;
  uint32_t signal_stride = sizeof(uint32_t);
};

}  // namespace uccl::v2_efa
