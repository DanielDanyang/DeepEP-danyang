#pragma once

#include <cstdint>
#include <string>

#include "v2_efa/combine_plan.hpp"
#include "v2_efa/descriptor.hpp"
#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/topology.hpp"
#include "v2_efa/workspace.hpp"

namespace uccl::v2_efa {

struct RuntimeConfig {
  int rank = 0;
  int world_size = 1;
  int scaleout_rank = 0;
  int scaleup_rank = 0;
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int num_experts = 1;
  int num_topk = 1;
  int hidden = 0;
  int elem_bytes = 2;
  int num_sms = 0;
};

class V2EfaRuntime {
 public:
  explicit V2EfaRuntime(RuntimeConfig config);

  const RuntimeConfig& config() const { return config_; }
  bool is_ready() const { return false; }
  std::string status() const;

  WorkspacePlan workspace_plan(int num_max_tokens_per_rank) const;
  DescriptorPlanStats worst_case_stats(int num_max_tokens_per_rank) const;
  ExpertRoute route_expert(int expert_id) const;
  DispatchPlan build_reference_dispatch_plan(const int64_t* topk_idx,
                                             int num_tokens,
                                             int payload_bytes,
                                             int scale_bytes,
                                             bool has_topk_weight) const;
  CombinePlan build_reference_combine_plan_from_dispatch(
      const DispatchPlan& dispatch_plan, int payload_bytes) const;

  [[noreturn]] void launch_dispatch() const;
  [[noreturn]] void launch_combine() const;

 private:
  RuntimeConfig config_;
};

}  // namespace uccl::v2_efa
