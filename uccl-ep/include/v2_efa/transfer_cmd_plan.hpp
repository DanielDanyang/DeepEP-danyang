#pragma once

#include <vector>

#include "v2_efa/combine_plan.hpp"
#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/transfer_cmd.hpp"

namespace uccl::v2_efa {

struct V2TransferCmdPlan {
  std::vector<V2TransferCmd> commands;
};

inline V2TransferCmdPlan build_dispatch_transfer_cmd_plan(
    const DispatchPlan& plan, const DispatchTransferLayout& layout) {
  V2TransferCmdPlan out;
  for (size_t batch_idx = 0; batch_idx < plan.batches.size(); ++batch_idx) {
    const auto& batch = plan.batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx = static_cast<uint32_t>(batch.first_segment + i);
      out.commands.push_back(make_v2_dispatch_payload_cmd(
          plan.segments[segment_idx], segment_idx,
          static_cast<uint32_t>(batch_idx), layout));
    }
    out.commands.push_back(make_v2_dispatch_signal_cmd(
        batch, static_cast<uint32_t>(batch_idx), layout));
  }
  return out;
}

inline V2TransferCmdPlan build_combine_transfer_cmd_plan(
    const CombinePlan& plan, const CombineTransferLayout& layout) {
  V2TransferCmdPlan out;
  for (size_t batch_idx = 0; batch_idx < plan.batches.size(); ++batch_idx) {
    const auto& batch = plan.batches[batch_idx];
    for (int i = 0; i < batch.num_segments; ++i) {
      const auto segment_idx = static_cast<uint32_t>(batch.first_segment + i);
      out.commands.push_back(make_v2_combine_payload_cmd(
          plan.segments[segment_idx], segment_idx,
          static_cast<uint32_t>(batch_idx), layout));
    }
    out.commands.push_back(make_v2_combine_signal_cmd(
        batch, static_cast<uint32_t>(batch_idx), layout));
  }
  return out;
}

}  // namespace uccl::v2_efa
