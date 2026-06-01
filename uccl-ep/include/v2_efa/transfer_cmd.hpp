#pragma once

#include <cstdint>
#include <stdexcept>

#include "descriptor.hpp"

namespace uccl::v2_efa {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define V2_EFA_HOST_DEVICE __host__ __device__
#else
#define V2_EFA_HOST_DEVICE
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define V2_EFA_DEVICE_CODE 1
#else
#define V2_EFA_DEVICE_CODE 0
#endif

constexpr int kV2TransferOffsetShift = 2;

enum class V2TransferCmdKind : uint8_t {
  kDispatchPayload = 1,
  kDispatchSignal = 2,
  kCombinePayload = 3,
  kCombineSignal = 4,
};

enum class V2TransferCmdFlags : uint8_t {
  kNone = 0,
  kSignal = 1u << 0,
  kPayload = 1u << 1,
  kDispatch = 1u << 2,
  kCombine = 1u << 3,
};

V2_EFA_HOST_DEVICE inline constexpr V2TransferCmdFlags operator|(
    V2TransferCmdFlags a, V2TransferCmdFlags b) {
  return static_cast<V2TransferCmdFlags>(static_cast<uint8_t>(a) |
                                         static_cast<uint8_t>(b));
}

#pragma pack(push, 1)
struct V2TransferCmd {
  uint8_t kind = 0;
  uint8_t target_rank = 0;
  uint8_t target_lane = 0;
  uint8_t flags = 0;
  uint32_t bytes = 0;
  uint32_t remote_offset_shifted = 0;
  union {
    uint32_t local_offset_shifted;
    uint32_t signal_value;
  };
};
#pragma pack(pop)

static_assert(sizeof(V2TransferCmd) == 16,
              "V2TransferCmd must stay one 128-bit FIFO command");

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

V2_EFA_HOST_DEVICE inline bool is_v2_transfer_cmd(const V2TransferCmd& cmd) {
  return cmd.kind >= static_cast<uint8_t>(V2TransferCmdKind::kDispatchPayload) &&
         cmd.kind <= static_cast<uint8_t>(V2TransferCmdKind::kCombineSignal);
}

V2_EFA_HOST_DEVICE inline uint32_t v2_transfer_cmd_header(
    const V2TransferCmd& cmd) {
  return static_cast<uint32_t>(cmd.kind) |
         (static_cast<uint32_t>(cmd.target_rank) << 8) |
         (static_cast<uint32_t>(cmd.target_lane) << 16) |
         (static_cast<uint32_t>(cmd.flags) << 24);
}

V2_EFA_HOST_DEVICE inline uint8_t v2_transfer_kind_from_header(
    uint32_t header) {
  return static_cast<uint8_t>(header & 0xffu);
}

V2_EFA_HOST_DEVICE inline uint8_t v2_transfer_flags(V2TransferCmdKind kind) {
  switch (kind) {
    case V2TransferCmdKind::kDispatchPayload:
      return static_cast<uint8_t>(V2TransferCmdFlags::kDispatch) |
             static_cast<uint8_t>(V2TransferCmdFlags::kPayload);
    case V2TransferCmdKind::kDispatchSignal:
      return static_cast<uint8_t>(V2TransferCmdFlags::kDispatch) |
             static_cast<uint8_t>(V2TransferCmdFlags::kSignal);
    case V2TransferCmdKind::kCombinePayload:
      return static_cast<uint8_t>(V2TransferCmdFlags::kCombine) |
             static_cast<uint8_t>(V2TransferCmdFlags::kPayload);
    case V2TransferCmdKind::kCombineSignal:
      return static_cast<uint8_t>(V2TransferCmdFlags::kCombine) |
             static_cast<uint8_t>(V2TransferCmdFlags::kSignal);
  }
  return 0;
}

V2_EFA_HOST_DEVICE inline uint32_t encode_v2_transfer_offset(uint64_t offset) {
  const auto align = uint64_t{1} << kV2TransferOffsetShift;
#if !V2_EFA_DEVICE_CODE
  if ((offset & (align - 1)) != 0) {
    throw std::invalid_argument("V2 transfer offset is not aligned");
  }
#endif
  const auto shifted = offset >> kV2TransferOffsetShift;
#if !V2_EFA_DEVICE_CODE
  if (shifted > UINT32_MAX) {
    throw std::out_of_range("V2 transfer offset exceeds command range");
  }
#endif
  return static_cast<uint32_t>(shifted);
}

V2_EFA_HOST_DEVICE inline uint64_t decode_v2_transfer_offset(uint32_t shifted) {
  return static_cast<uint64_t>(shifted) << kV2TransferOffsetShift;
}

V2_EFA_HOST_DEVICE inline bool is_v2_transfer_payload(
    const V2TransferCmd& command) {
  return (command.flags & static_cast<uint8_t>(V2TransferCmdFlags::kPayload)) !=
         0;
}

V2_EFA_HOST_DEVICE inline bool is_v2_transfer_signal(
    const V2TransferCmd& command) {
  return (command.flags & static_cast<uint8_t>(V2TransferCmdFlags::kSignal)) !=
         0;
}

V2_EFA_HOST_DEVICE inline uint64_t v2_transfer_remote_offset(
    const V2TransferCmd& command) {
  return decode_v2_transfer_offset(command.remote_offset_shifted);
}

V2_EFA_HOST_DEVICE inline uint64_t v2_transfer_local_offset(
    const V2TransferCmd& command) {
  return decode_v2_transfer_offset(command.local_offset_shifted);
}

V2_EFA_HOST_DEVICE inline void pack_v2_transfer_cmd(
    const V2TransferCmd& command, uint64_t* first, uint64_t* second) {
#if !V2_EFA_DEVICE_CODE
  if (first == nullptr || second == nullptr) {
    throw std::invalid_argument("V2 transfer command pack output is null");
  }
#endif
  uint64_t a = 0;
  uint64_t b = 0;
  a |= static_cast<uint64_t>(command.kind);
  a |= static_cast<uint64_t>(command.target_rank) << 8;
  a |= static_cast<uint64_t>(command.target_lane) << 16;
  a |= static_cast<uint64_t>(command.flags) << 24;
  a |= static_cast<uint64_t>(command.bytes) << 32;
  b |= static_cast<uint64_t>(command.remote_offset_shifted);
  b |= static_cast<uint64_t>(command.local_offset_shifted) << 32;
  *first = a;
  *second = b;
}

V2_EFA_HOST_DEVICE inline V2TransferCmd unpack_v2_transfer_cmd(
    uint64_t first, uint64_t second) {
  V2TransferCmd command;
  command.kind = static_cast<uint8_t>(first & 0xFFu);
  command.target_rank = static_cast<uint8_t>((first >> 8) & 0xFFu);
  command.target_lane = static_cast<uint8_t>((first >> 16) & 0xFFu);
  command.flags = static_cast<uint8_t>((first >> 24) & 0xFFu);
  command.bytes = static_cast<uint32_t>((first >> 32) & 0xFFFFFFFFu);
  command.remote_offset_shifted = static_cast<uint32_t>(second & 0xFFFFFFFFu);
  command.local_offset_shifted =
      static_cast<uint32_t>((second >> 32) & 0xFFFFFFFFu);
  return command;
}

V2_EFA_HOST_DEVICE inline V2TransferCmd make_v2_transfer_cmd(
    V2TransferCmdKind kind, uint32_t target_rank, uint32_t target_lane,
    uint32_t descriptor_index, uint32_t batch_index, uint32_t bytes,
    uint32_t signal_value, uint64_t local_offset, uint64_t remote_offset) {
#if !V2_EFA_DEVICE_CODE
  if (target_rank > UINT8_MAX || target_lane > UINT8_MAX) {
    throw std::out_of_range("V2 transfer target exceeds command range");
  }
#endif
  V2TransferCmd out;
  out.kind = static_cast<uint8_t>(kind);
  out.target_rank = static_cast<uint8_t>(target_rank);
  out.target_lane = static_cast<uint8_t>(target_lane);
  out.flags = v2_transfer_flags(kind);
  out.bytes = bytes;
  out.remote_offset_shifted = encode_v2_transfer_offset(remote_offset);
  if ((out.flags & static_cast<uint8_t>(V2TransferCmdFlags::kSignal)) != 0) {
    out.signal_value = signal_value;
  } else {
    out.local_offset_shifted = encode_v2_transfer_offset(local_offset);
  }
  (void)descriptor_index;
  (void)batch_index;
  return out;
}

V2_EFA_HOST_DEVICE inline V2TransferCmd make_v2_dispatch_payload_cmd(
    const DispatchSegmentDescriptor& segment, uint32_t segment_idx,
    uint32_t batch_idx, const DispatchTransferLayout& layout) {
  const uint32_t token_bytes =
      layout.token_record_bytes != 0
          ? layout.token_record_bytes
          : static_cast<uint32_t>(segment.payload_bytes);
  const uint32_t src_stride =
      layout.src_token_stride != 0 ? layout.src_token_stride : token_bytes;
  const uint32_t expanded_stride =
      layout.expanded_slot_stride != 0 ? layout.expanded_slot_stride
                                       : token_bytes;
  const uint64_t source_payload_base =
      layout.remote_payload_base +
      static_cast<uint64_t>(layout.source_rank) * layout.source_rank_stride;
  const uint64_t batch_payload_base =
      source_payload_base +
      (layout.token_record_bytes != 0
           ? 0
           : static_cast<uint64_t>(batch_idx) * layout.batch_payload_stride);
  const bool use_global_rank = layout.num_scaleup_ranks != 0;
  const uint32_t dst_global_rank =
      use_global_rank
          ? static_cast<uint32_t>(segment.dst_scaleout_rank) *
                    layout.num_scaleup_ranks +
                static_cast<uint32_t>(segment.dst_scaleup_lane)
          : static_cast<uint32_t>(segment.dst_scaleout_rank);
  const uint32_t num_efa_lanes =
      layout.num_efa_lanes == 0 ? 1u : layout.num_efa_lanes;
  const uint32_t efa_lane =
      use_global_rank ? (static_cast<uint32_t>(segment.expert_id) %
                         num_efa_lanes)
                      : static_cast<uint32_t>(segment.dst_scaleup_lane);
  return make_v2_transfer_cmd(
      V2TransferCmdKind::kDispatchPayload,
      dst_global_rank,
      efa_lane,
      segment_idx, batch_idx,
      static_cast<uint32_t>(segment.count) * token_bytes,
      /*signal_value=*/0,
      layout.local_payload_base +
          static_cast<uint64_t>(segment.src_token_begin) * src_stride,
      batch_payload_base + static_cast<uint64_t>(segment.expanded_slot_begin) *
                               expanded_stride);
}

V2_EFA_HOST_DEVICE inline V2TransferCmd make_v2_dispatch_signal_cmd(
    const DispatchExpertBatch& batch, uint32_t batch_idx,
    const DispatchTransferLayout& layout) {
  const uint64_t source_signal_base =
      layout.remote_signal_base +
      static_cast<uint64_t>(layout.source_rank) * layout.source_signal_stride;
  const bool use_global_rank = layout.num_scaleup_ranks != 0;
  const uint32_t dst_global_rank =
      use_global_rank
          ? static_cast<uint32_t>(batch.dst_scaleout_rank) *
                    layout.num_scaleup_ranks +
                static_cast<uint32_t>(batch.dst_scaleup_lane)
          : static_cast<uint32_t>(batch.dst_scaleout_rank);
  const uint32_t num_efa_lanes =
      layout.num_efa_lanes == 0 ? 1u : layout.num_efa_lanes;
  const uint32_t efa_lane =
      use_global_rank ? (static_cast<uint32_t>(batch.expert_id) %
                         num_efa_lanes)
                      : static_cast<uint32_t>(batch.dst_scaleup_lane);
  return make_v2_transfer_cmd(
      V2TransferCmdKind::kDispatchSignal,
      dst_global_rank,
      efa_lane,
      static_cast<uint32_t>(batch.first_segment), batch_idx,
      sizeof(uint32_t),
      static_cast<uint32_t>(batch.total_tokens),
      /*local_offset=*/0,
      source_signal_base +
          static_cast<uint64_t>(batch_idx) * layout.signal_stride);
}

V2_EFA_HOST_DEVICE inline V2TransferCmd make_v2_dispatch_done_cmd(
    uint32_t target_rank, uint32_t expected_count,
    const DispatchTransferLayout& layout) {
  const uint64_t source_signal_base =
      layout.remote_signal_base +
      static_cast<uint64_t>(layout.source_rank) * layout.source_signal_stride;
  return make_v2_transfer_cmd(
      V2TransferCmdKind::kDispatchSignal,
      target_rank,
      layout.efa_lane,
      /*descriptor_index=*/0,
      layout.max_batches,
      sizeof(uint32_t),
      /*signal_value=*/expected_count + 1u,
      /*local_offset=*/0,
      source_signal_base +
          static_cast<uint64_t>(layout.max_batches) * layout.signal_stride);
}

V2_EFA_HOST_DEVICE inline V2TransferCmd make_v2_combine_payload_cmd(
    const CombineSegmentDescriptor& segment, uint32_t segment_idx,
    uint32_t batch_idx, const CombineTransferLayout& layout) {
  return make_v2_transfer_cmd(
      V2TransferCmdKind::kCombinePayload,
      static_cast<uint32_t>(segment.dst_scaleout_rank),
      static_cast<uint32_t>(segment.dst_scaleup_lane),
      segment_idx, batch_idx,
      static_cast<uint32_t>(segment.count * segment.payload_bytes),
      /*signal_value=*/0,
      layout.local_payload_base +
          static_cast<uint64_t>(segment.expanded_slot_begin) *
              layout.expanded_slot_stride,
      layout.remote_payload_base +
          static_cast<uint64_t>(batch_idx) * layout.batch_payload_stride +
          static_cast<uint64_t>(segment.reduced_token_slot) *
              layout.reduced_token_stride);
}

V2_EFA_HOST_DEVICE inline V2TransferCmd make_v2_combine_signal_cmd(
    const CombineExpertBatch& batch, uint32_t batch_idx,
    const CombineTransferLayout& layout) {
  return make_v2_transfer_cmd(
      V2TransferCmdKind::kCombineSignal,
      static_cast<uint32_t>(batch.dst_scaleout_rank),
      static_cast<uint32_t>(batch.dst_scaleup_lane),
      static_cast<uint32_t>(batch.first_segment), batch_idx,
      sizeof(uint32_t),
      static_cast<uint32_t>(batch.total_tokens),
      /*local_offset=*/0,
      layout.remote_signal_base +
          static_cast<uint64_t>(batch_idx) * layout.signal_stride);
}

struct V2TransferQueueView {
  V2TransferCmd* commands = nullptr;
  uint32_t* tail = nullptr;
  uint32_t capacity = 0;
};

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t reserve_v2_transfer_cmd(
    V2TransferQueueView queue) {
  while (true) {
    const auto tail = *reinterpret_cast<volatile uint32_t*>(queue.tail);
    if (tail >= queue.capacity) {
      return queue.capacity;
    }
    const auto claimed = atomicCAS(queue.tail, tail, tail + 1u);
    if (claimed == tail) {
      return tail;
    }
    __nanosleep(64);
  }
}

__device__ __forceinline__ bool enqueue_v2_transfer_cmd(
    V2TransferQueueView queue, V2TransferCmd command) {
  if (!is_v2_transfer_cmd(command)) {
    return false;
  }
  const auto slot = reserve_v2_transfer_cmd(queue);
  if (slot < queue.capacity) {
    queue.commands[slot] = command;
    return true;
  }
  return false;
}
#endif

}  // namespace uccl::v2_efa

#undef V2_EFA_HOST_DEVICE
#undef V2_EFA_DEVICE_CODE
