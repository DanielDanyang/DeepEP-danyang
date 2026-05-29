#pragma once

#include <cstdint>
#include <vector>

#include "v2_efa/combine_plan.hpp"
#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/proxy_queue.cuh"

namespace uccl::v2_efa {

struct ProxyCommandPlan {
  std::vector<ProxyCommand> commands;
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

inline ProxyCommandPlan build_dispatch_proxy_command_plan(
    const DispatchPlan& plan, const DispatchProxyLayout& layout) {
  ProxyCommandPlan out;
  for (size_t batch_idx = 0; batch_idx < plan.batches.size(); ++batch_idx) {
    const auto& batch = plan.batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx = static_cast<uint32_t>(batch.first_segment + i);
      const auto& segment = plan.segments[segment_idx];
      ProxyCommand command;
      command.kind = static_cast<uint32_t>(ProxyCommandKind::kDispatchPayload);
      command.descriptor_index = segment_idx;
      command.batch_index = static_cast<uint32_t>(batch_idx);
      command.bytes = static_cast<uint32_t>(segment.count * segment.payload_bytes);
      command.local_offset =
          layout.local_payload_base +
          static_cast<uint64_t>(segment.src_token_begin) * layout.src_token_stride;
      command.remote_offset =
          layout.remote_payload_base +
          static_cast<uint64_t>(segment.expanded_slot_begin) *
              layout.expanded_slot_stride;
      out.commands.push_back(command);
    }

    ProxyCommand signal;
    signal.kind = static_cast<uint32_t>(ProxyCommandKind::kDispatchSignal);
    signal.descriptor_index = static_cast<uint32_t>(batch.first_segment);
    signal.batch_index = static_cast<uint32_t>(batch_idx);
    signal.bytes = sizeof(uint32_t);
    signal.local_offset = 0;
    signal.remote_offset =
        layout.remote_signal_base + batch_idx * layout.signal_stride;
    out.commands.push_back(signal);
  }
  return out;
}

inline ProxyCommandPlan build_combine_proxy_command_plan(
    const CombinePlan& plan, const CombineProxyLayout& layout) {
  ProxyCommandPlan out;
  for (size_t batch_idx = 0; batch_idx < plan.batches.size(); ++batch_idx) {
    const auto& batch = plan.batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx = static_cast<uint32_t>(batch.first_segment + i);
      const auto& segment = plan.segments[segment_idx];
      ProxyCommand command;
      command.kind = static_cast<uint32_t>(ProxyCommandKind::kCombinePayload);
      command.descriptor_index = segment_idx;
      command.batch_index = static_cast<uint32_t>(batch_idx);
      command.bytes = static_cast<uint32_t>(segment.count * segment.payload_bytes);
      command.local_offset =
          layout.local_payload_base +
          static_cast<uint64_t>(segment.expanded_slot_begin) *
              layout.expanded_slot_stride;
      command.remote_offset =
          layout.remote_payload_base +
          static_cast<uint64_t>(segment.reduced_token_slot) *
              layout.reduced_token_stride;
      out.commands.push_back(command);
    }

    ProxyCommand signal;
    signal.kind = static_cast<uint32_t>(ProxyCommandKind::kCombineSignal);
    signal.descriptor_index = static_cast<uint32_t>(batch.first_segment);
    signal.batch_index = static_cast<uint32_t>(batch_idx);
    signal.bytes = sizeof(uint32_t);
    signal.local_offset = 0;
    signal.remote_offset =
        layout.remote_signal_base + batch_idx * layout.signal_stride;
    out.commands.push_back(signal);
  }
  return out;
}

}  // namespace uccl::v2_efa
