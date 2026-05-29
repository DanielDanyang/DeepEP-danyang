#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/runtime.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

using namespace uccl::v2_efa;

int main() {
  RuntimeConfig runtime_config;
  runtime_config.world_size = 4;
  runtime_config.num_scaleout_ranks = 2;
  runtime_config.num_scaleup_ranks = 2;
  runtime_config.scaleout_rank = 0;
  runtime_config.num_experts = 8;
  runtime_config.num_topk = 2;
  runtime_config.hidden = 16;
  runtime_config.elem_bytes = 2;

  V2EfaRuntime runtime(runtime_config);
  const auto route = runtime.route_expert(6);
  assert(route.owner_rank == 3);
  assert(route.dst_scaleout_rank == 1);
  assert(route.dst_scaleup_lane == 1);
  assert(route.is_remote_scaleout == 1);

  const std::vector<int64_t> topk = {
      0, 4,
      0, 4,
      2, 6,
      2, 6,
  };
  const auto plan =
      runtime.build_reference_dispatch_plan(topk.data(), 4, 32, 0, true);

  assert(plan.batches.size() == 4);
  assert(plan.segments.size() == 4);

  assert(plan.batches[0].dst_scaleout_rank == 0);
  assert(plan.batches[0].dst_scaleup_lane == 0);
  assert(plan.batches[0].expert_id == 0);
  assert(plan.batches[0].first_segment == 0);
  assert(plan.batches[0].num_segments == 1);
  assert(plan.batches[0].total_tokens == 2);
  assert(plan.segments[0].src_token_begin == 0);
  assert(plan.segments[0].topk_slot == 0);
  assert(plan.segments[0].expanded_slot_begin == 0);
  assert(plan.segments[0].count == 2);

  assert(plan.batches[2].dst_scaleout_rank == 1);
  assert(plan.batches[2].dst_scaleup_lane == 0);
  assert(plan.batches[2].expert_id == 4);
  assert(plan.segments[2].topk_slot == 1);
  assert(plan.segments[2].count == 2);

  const std::vector<int64_t> sparse_topk = {
      0, -1,
      0, -1,
  };
  const auto sparse_plan =
      runtime.build_reference_dispatch_plan(sparse_topk.data(), 2, 32, 8, true);
  assert(sparse_plan.batches.size() == 1);
  assert(sparse_plan.segments.size() == 1);
  assert(sparse_plan.segments[0].scale_bytes == 8);
  assert((sparse_plan.segments[0].flags &
          static_cast<uint32_t>(DescriptorFlags::kHasScale)) != 0);

  const auto combine_plan =
      runtime.build_reference_combine_plan_from_dispatch(plan, 32);
  assert(combine_plan.batches.size() == 4);
  assert(combine_plan.segments.size() == 4);
  assert(combine_plan.batches[2].src_scaleout_rank == 1);
  assert(combine_plan.batches[2].expert_id == 4);
  assert(combine_plan.segments[2].expanded_slot_begin == 0);
  assert(combine_plan.segments[2].topk_slot == 1);
  assert(combine_plan.segments[2].reduced_token_slot == 0);
  assert(combine_plan.segments[2].count == 2);

  const auto workspace = runtime.workspace_plan(4);
  assert(workspace.dispatch_counters.bytes ==
         sizeof(uint32_t) * kDescriptorCounterWords);
  assert(workspace.combine_counters.bytes ==
         sizeof(uint32_t) * kDescriptorCounterWords);

  return 0;
}
