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

inline ProxyCommandPlan build_dispatch_proxy_command_plan(
    const DispatchPlan& plan, const DispatchProxyLayout& layout) {
  ProxyCommandPlan out;
  for (size_t batch_idx = 0; batch_idx < plan.batches.size(); ++batch_idx) {
    const auto& batch = plan.batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx = static_cast<uint32_t>(batch.first_segment + i);
      const auto& segment = plan.segments[segment_idx];
      out.commands.push_back(make_dispatch_payload_command(
          segment, segment_idx, static_cast<uint32_t>(batch_idx), layout));
    }

    out.commands.push_back(make_dispatch_signal_command(
        batch, static_cast<uint32_t>(batch_idx), layout));
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
      out.commands.push_back(make_combine_payload_command(
          segment, segment_idx, static_cast<uint32_t>(batch_idx), layout));
    }

    out.commands.push_back(make_combine_signal_command(
        batch, static_cast<uint32_t>(batch_idx), layout));
  }
  return out;
}

}  // namespace uccl::v2_efa
