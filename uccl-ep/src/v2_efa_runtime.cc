#include "v2_efa/runtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace uccl::v2_efa {

namespace {

void validate_config(const RuntimeConfig& config) {
  validate_non_negative("rank", config.rank);
  validate_non_negative("world_size", config.world_size);
  validate_non_negative("scaleout_rank", config.scaleout_rank);
  validate_non_negative("scaleup_rank", config.scaleup_rank);
  validate_non_negative("num_scaleout_ranks", config.num_scaleout_ranks);
  validate_non_negative("num_scaleup_ranks", config.num_scaleup_ranks);
  validate_non_negative("num_experts", config.num_experts);
  validate_non_negative("num_topk", config.num_topk);
  validate_non_negative("hidden", config.hidden);
  validate_non_negative("elem_bytes", config.elem_bytes);
  validate_non_negative("num_sms", config.num_sms);
  if (config.world_size == 0 || config.num_scaleout_ranks == 0 ||
      config.num_scaleup_ranks == 0 || config.num_experts == 0 ||
      config.num_topk == 0 || config.elem_bytes == 0) {
    throw std::invalid_argument("V2EfaRuntimeConfig contains a zero dimension");
  }
  if (config.num_scaleout_ranks * config.num_scaleup_ranks !=
      config.world_size) {
    throw std::invalid_argument(
        "num_scaleout_ranks * num_scaleup_ranks must equal world_size");
  }
  (void)experts_per_rank(config.num_experts, config.world_size);
}

[[noreturn]] void fail_not_ready() {
  throw std::runtime_error(
      "uccl-ep native V2 AWS EFA runtime scaffold is present, but "
      "dispatch/combine JIT kernels are not implemented yet.");
}

}  // namespace

V2EfaRuntime::V2EfaRuntime(RuntimeConfig config) : config_(config) {
  validate_config(config_);
}

std::string V2EfaRuntime::status() const {
  return "native V2 AWS EFA runtime scaffold: descriptors/workspace ready, "
         "JIT dispatch/combine not implemented";
}

DescriptorPlanStats V2EfaRuntime::worst_case_stats(
    int num_max_tokens_per_rank) const {
  validate_non_negative("num_max_tokens_per_rank", num_max_tokens_per_rank);
  DescriptorPlanStats stats;
  stats.num_dispatch_segments = static_cast<int32_t>(
      max_dispatch_segments(num_max_tokens_per_rank, config_.num_topk));
  stats.num_dispatch_batches = static_cast<int32_t>(max_expert_batches(
      config_.num_experts, config_.num_scaleout_ranks,
      config_.num_scaleup_ranks));
  stats.num_combine_segments = stats.num_dispatch_segments;
  stats.num_combine_batches = stats.num_dispatch_batches;
  stats.max_tokens_per_segment = std::max(1, num_max_tokens_per_rank);
  stats.max_payload_bytes_per_segment =
      std::max(1, config_.hidden * config_.elem_bytes);
  return stats;
}

WorkspacePlan V2EfaRuntime::workspace_plan(
    int num_max_tokens_per_rank) const {
  const auto stats = worst_case_stats(num_max_tokens_per_rank);
  return build_workspace_plan(stats.num_dispatch_segments,
                              stats.num_dispatch_batches,
                              stats.num_combine_segments,
                              stats.num_combine_batches);
}

ExpertRoute V2EfaRuntime::route_expert(int expert_id) const {
  return uccl::v2_efa::route_expert(expert_id, config_.num_experts,
                                    config_.world_size,
                                    config_.num_scaleup_ranks,
                                    config_.scaleout_rank);
}

void V2EfaRuntime::launch_dispatch() const { fail_not_ready(); }

void V2EfaRuntime::launch_combine() const { fail_not_ready(); }

}  // namespace uccl::v2_efa
