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
  if (config.scaleout_rank * config.num_scaleup_ranks + config.scaleup_rank !=
      config.rank) {
    throw std::invalid_argument(
        "rank must equal scaleout_rank * num_scaleup_ranks + scaleup_rank");
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

DispatchPlan V2EfaRuntime::build_reference_dispatch_plan(
    const int64_t* topk_idx, int num_tokens, int payload_bytes, int scale_bytes,
    bool has_topk_weight) const {
  DispatchPlanConfig plan_config;
  plan_config.world_size = config_.world_size;
  plan_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  plan_config.local_scaleout_rank = config_.scaleout_rank;
  plan_config.num_experts = config_.num_experts;
  plan_config.num_topk = config_.num_topk;
  plan_config.payload_bytes = payload_bytes;
  plan_config.scale_bytes = scale_bytes;
  plan_config.has_topk_weight = has_topk_weight;
  return uccl::v2_efa::build_reference_dispatch_plan(topk_idx, num_tokens,
                                                     plan_config);
}

CombinePlan V2EfaRuntime::build_reference_combine_plan_from_dispatch(
    const DispatchPlan& dispatch_plan, int payload_bytes) const {
  CombinePlanConfig plan_config;
  plan_config.dst_original_rank = config_.rank;
  plan_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  plan_config.payload_bytes = payload_bytes;
  return uccl::v2_efa::build_reference_combine_plan_from_dispatch(
      dispatch_plan, plan_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_jit_plan(
    int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
    bool has_topk_weight, bool cached_mode, bool deterministic,
    bool do_cpu_sync, int smem_bytes,
    const std::string& uccl_include_path) const {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = config_.hidden;
  jit_config.elem_bytes = config_.elem_bytes;
  jit_config.num_sms = std::max(1, config_.num_sms);
  jit_config.num_channels_per_sm = num_channels_per_sm;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.scaleout_rank = config_.scaleout_rank;
  jit_config.scaleup_rank = config_.scaleup_rank;
  jit_config.scale_bytes = scale_bytes;
  jit_config.has_topk_weight = has_topk_weight;
  jit_config.cached_mode = cached_mode;
  jit_config.deterministic = deterministic;
  jit_config.do_cpu_sync = do_cpu_sync;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_combine_jit_plan(
    int num_max_tokens_per_rank, int num_channels, int payload_bytes,
    bool use_expanded_layout, bool allow_multiple_reduction, int smem_bytes,
    const std::string& uccl_include_path) const {
  V2EfaCombineJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = config_.hidden;
  jit_config.num_sms = std::max(1, config_.num_sms);
  jit_config.num_channels = num_channels;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.dst_original_rank = config_.rank;
  jit_config.payload_bytes = payload_bytes;
  jit_config.use_expanded_layout = use_expanded_layout;
  jit_config.allow_multiple_reduction = allow_multiple_reduction;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_combine_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_enqueue_d2h_jit_plan(
    const std::string& uccl_include_path) const {
  return build_v2_efa_dispatch_enqueue_d2h_jit_plan(uccl_include_path);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_combine_enqueue_d2h_jit_plan(
    const std::string& uccl_include_path) const {
  return build_v2_efa_combine_enqueue_d2h_jit_plan(uccl_include_path);
}

V2EfaJitLaunchPlan
V2EfaRuntime::build_dispatch_descriptor_enqueue_d2h_jit_plan(
    int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
    bool has_topk_weight, bool cached_mode, bool deterministic,
    bool do_cpu_sync, int smem_bytes,
    const std::string& uccl_include_path) const {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = config_.hidden;
  jit_config.elem_bytes = config_.elem_bytes;
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels_per_sm = num_channels_per_sm;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.scaleout_rank = config_.scaleout_rank;
  jit_config.scaleup_rank = config_.scaleup_rank;
  jit_config.scale_bytes = scale_bytes;
  jit_config.has_topk_weight = has_topk_weight;
  jit_config.cached_mode = cached_mode;
  jit_config.deterministic = deterministic;
  jit_config.do_cpu_sync = do_cpu_sync;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_descriptor_enqueue_d2h_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_direct_enqueue_d2h_jit_plan(
    int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
    bool has_topk_weight, bool cached_mode, bool deterministic,
    bool do_cpu_sync, int smem_bytes,
    const std::string& uccl_include_path) const {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = config_.hidden;
  jit_config.elem_bytes = config_.elem_bytes;
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels_per_sm = num_channels_per_sm;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.scaleout_rank = config_.scaleout_rank;
  jit_config.scaleup_rank = config_.scaleup_rank;
  jit_config.scale_bytes = scale_bytes;
  jit_config.has_topk_weight = has_topk_weight;
  jit_config.cached_mode = cached_mode;
  jit_config.deterministic = deterministic;
  jit_config.do_cpu_sync = do_cpu_sync;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_direct_enqueue_d2h_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_forward_metadata_jit_plan(
    int num_max_tokens_per_rank, int num_channels_per_sm,
    const std::string& uccl_include_path) const {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = std::max(1, config_.hidden);
  jit_config.elem_bytes = std::max(1, config_.elem_bytes);
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels_per_sm = num_channels_per_sm;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.scaleout_rank = config_.scaleout_rank;
  jit_config.scaleup_rank = config_.scaleup_rank;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_forward_metadata_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_receiver_metadata_jit_plan(
    int num_max_tokens_per_rank, const std::string& uccl_include_path) const {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = std::max(1, config_.hidden);
  jit_config.elem_bytes = std::max(1, config_.elem_bytes);
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels_per_sm = 1;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.scaleout_rank = config_.scaleout_rank;
  jit_config.scaleup_rank = config_.scaleup_rank;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_receiver_metadata_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_materialize_records_jit_plan(
    int num_max_tokens_per_rank, const std::string& uccl_include_path) const {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = std::max(1, config_.hidden);
  jit_config.elem_bytes = std::max(1, config_.elem_bytes);
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels_per_sm = 1;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.scaleout_rank = config_.scaleout_rank;
  jit_config.scaleup_rank = config_.scaleup_rank;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_materialize_records_jit_plan(jit_config);
}

V2EfaJitLaunchPlan
V2EfaRuntime::build_combine_descriptor_enqueue_d2h_jit_plan(
    int num_max_tokens_per_rank, int num_channels, int payload_bytes,
    bool use_expanded_layout, bool allow_multiple_reduction, int smem_bytes,
    const std::string& uccl_include_path) const {
  V2EfaCombineJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = config_.hidden;
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels = num_channels;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.dst_original_rank = config_.rank;
  jit_config.payload_bytes = payload_bytes;
  jit_config.use_expanded_layout = use_expanded_layout;
  jit_config.allow_multiple_reduction = allow_multiple_reduction;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_combine_descriptor_enqueue_d2h_jit_plan(jit_config);
}

V2EfaJitLaunchPlan
V2EfaRuntime::build_combine_forward_metadata_enqueue_d2h_jit_plan(
    int num_max_tokens_per_rank, int num_channels, int payload_bytes,
    bool use_expanded_layout, bool allow_multiple_reduction, int smem_bytes,
    const std::string& uccl_include_path) const {
  V2EfaCombineJitConfig jit_config;
  jit_config.num_scaleout_ranks = config_.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config_.num_scaleup_ranks;
  jit_config.num_experts = config_.num_experts;
  jit_config.num_topk = config_.num_topk;
  jit_config.hidden = config_.hidden;
  jit_config.num_sms = config_.num_sms > 0 ? config_.num_sms : 1;
  jit_config.num_channels = num_channels;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.dst_original_rank = config_.rank;
  jit_config.payload_bytes = payload_bytes;
  jit_config.use_expanded_layout = use_expanded_layout;
  jit_config.allow_multiple_reduction = allow_multiple_reduction;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_combine_forward_metadata_enqueue_d2h_jit_plan(jit_config);
}

void V2EfaRuntime::launch_dispatch() const { fail_not_ready(); }

void V2EfaRuntime::launch_combine() const { fail_not_ready(); }

}  // namespace uccl::v2_efa
