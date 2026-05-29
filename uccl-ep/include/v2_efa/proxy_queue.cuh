#pragma once

#include <cstdint>

#include "v2_efa/descriptor.hpp"

namespace uccl::v2_efa {

enum class ProxyCommandKind : uint32_t {
  kDispatchPayload = 1,
  kDispatchSignal = 2,
  kCombinePayload = 3,
  kCombineSignal = 4,
};

struct ProxyCommand {
  uint32_t kind = 0;
  uint32_t descriptor_index = 0;
  uint32_t batch_index = 0;
  uint32_t bytes = 0;
  uint64_t local_offset = 0;
  uint64_t remote_offset = 0;
};

struct ProxyQueueView {
  ProxyCommand* commands = nullptr;
  uint32_t* tail = nullptr;
  uint32_t capacity = 0;
};

struct DispatchProxyLayout {
  uint64_t local_payload_base = 0;
  uint64_t remote_payload_base = 0;
  uint64_t remote_signal_base = 0;
  uint32_t src_token_stride = 0;
  uint32_t expanded_slot_stride = 0;
  uint32_t signal_stride = sizeof(uint32_t);
};

struct CombineProxyLayout {
  uint64_t local_payload_base = 0;
  uint64_t remote_payload_base = 0;
  uint64_t remote_signal_base = 0;
  uint32_t expanded_slot_stride = 0;
  uint32_t reduced_token_stride = 0;
  uint32_t signal_stride = sizeof(uint32_t);
};

inline ProxyCommand make_dispatch_payload_command(
    const DispatchSegmentDescriptor& segment, uint32_t segment_idx,
    uint32_t batch_idx, const DispatchProxyLayout& layout) {
  ProxyCommand command;
  command.kind = static_cast<uint32_t>(ProxyCommandKind::kDispatchPayload);
  command.descriptor_index = segment_idx;
  command.batch_index = batch_idx;
  command.bytes = static_cast<uint32_t>(segment.count * segment.payload_bytes);
  command.local_offset =
      layout.local_payload_base +
      static_cast<uint64_t>(segment.src_token_begin) * layout.src_token_stride;
  command.remote_offset =
      layout.remote_payload_base +
      static_cast<uint64_t>(segment.expanded_slot_begin) *
          layout.expanded_slot_stride;
  return command;
}

inline ProxyCommand make_dispatch_signal_command(
    const DispatchExpertBatch& batch, uint32_t batch_idx,
    const DispatchProxyLayout& layout) {
  ProxyCommand command;
  command.kind = static_cast<uint32_t>(ProxyCommandKind::kDispatchSignal);
  command.descriptor_index = static_cast<uint32_t>(batch.first_segment);
  command.batch_index = batch_idx;
  command.bytes = sizeof(uint32_t);
  command.local_offset = 0;
  command.remote_offset =
      layout.remote_signal_base +
      static_cast<uint64_t>(batch_idx) * layout.signal_stride;
  return command;
}

inline ProxyCommand make_combine_payload_command(
    const CombineSegmentDescriptor& segment, uint32_t segment_idx,
    uint32_t batch_idx, const CombineProxyLayout& layout) {
  ProxyCommand command;
  command.kind = static_cast<uint32_t>(ProxyCommandKind::kCombinePayload);
  command.descriptor_index = segment_idx;
  command.batch_index = batch_idx;
  command.bytes = static_cast<uint32_t>(segment.count * segment.payload_bytes);
  command.local_offset =
      layout.local_payload_base +
      static_cast<uint64_t>(segment.expanded_slot_begin) *
          layout.expanded_slot_stride;
  command.remote_offset =
      layout.remote_payload_base +
      static_cast<uint64_t>(segment.reduced_token_slot) *
          layout.reduced_token_stride;
  return command;
}

inline ProxyCommand make_combine_signal_command(
    const CombineExpertBatch& batch, uint32_t batch_idx,
    const CombineProxyLayout& layout) {
  ProxyCommand command;
  command.kind = static_cast<uint32_t>(ProxyCommandKind::kCombineSignal);
  command.descriptor_index = static_cast<uint32_t>(batch.first_segment);
  command.batch_index = batch_idx;
  command.bytes = sizeof(uint32_t);
  command.local_offset = 0;
  command.remote_offset =
      layout.remote_signal_base +
      static_cast<uint64_t>(batch_idx) * layout.signal_stride;
  return command;
}

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t reserve_proxy_command(ProxyQueueView queue) {
  return atomicAdd(queue.tail, 1u);
}

__device__ __forceinline__ void enqueue_proxy_command(ProxyQueueView queue,
                                                      ProxyCommand command) {
  const auto slot = reserve_proxy_command(queue);
  if (slot < queue.capacity) {
    queue.commands[slot] = command;
  }
}
#endif

}  // namespace uccl::v2_efa
