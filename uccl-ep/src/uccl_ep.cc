#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "v2_efa/runtime.hpp"
#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_cmd_plan.hpp"
#include "v2_efa/transfer_layout.hpp"

namespace nb = nanobind;
namespace v2 = uccl::v2_efa;

namespace {

constexpr const char* kRewriteMessage =
    "uccl-ep is being rewritten as a native DeepEP V2 AWS EFA backend. "
    "The old V1 static internode/intranode kernels have been removed from "
    "the public runtime. Implement V2EfaRuntime dispatch/combine through "
    "JIT .cuh kernels before running benchmarks.";

struct Config {
  int num_sms;
  int num_max_nvl_chunked_send_tokens;
  int num_max_nvl_chunked_recv_tokens;
  int num_max_rdma_chunked_send_tokens;
  int num_max_rdma_chunked_recv_tokens;

  Config(int num_sms = 0, int num_max_nvl_chunked_send_tokens = 0,
         int num_max_nvl_chunked_recv_tokens = 0,
         int num_max_rdma_chunked_send_tokens = 0,
         int num_max_rdma_chunked_recv_tokens = 0)
      : num_sms(num_sms),
        num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {}
};

struct EventHandle {
  void current_stream_wait() const {}
};

nb::dict region_to_dict(const v2::WorkspaceRegion& region) {
  nb::dict out;
  out["offset"] = region.offset;
  out["bytes"] = region.bytes;
  return out;
}

nb::dict workspace_to_dict(const v2::WorkspacePlan& plan) {
  nb::dict out;
  out["dispatch_segments"] = region_to_dict(plan.dispatch_segments);
  out["dispatch_batches"] = region_to_dict(plan.dispatch_batches);
  out["combine_segments"] = region_to_dict(plan.combine_segments);
  out["combine_batches"] = region_to_dict(plan.combine_batches);
  out["dispatch_counters"] = region_to_dict(plan.dispatch_counters);
  out["combine_counters"] = region_to_dict(plan.combine_counters);
  out["total_bytes"] = plan.total_bytes;
  return out;
}

nb::dict stats_to_dict(const v2::DescriptorPlanStats& stats) {
  nb::dict out;
  out["num_dispatch_segments"] = stats.num_dispatch_segments;
  out["num_dispatch_batches"] = stats.num_dispatch_batches;
  out["num_combine_segments"] = stats.num_combine_segments;
  out["num_combine_batches"] = stats.num_combine_batches;
  out["max_tokens_per_segment"] = stats.max_tokens_per_segment;
  out["max_payload_bytes_per_segment"] = stats.max_payload_bytes_per_segment;
  return out;
}

nb::dict route_to_dict(const v2::ExpertRoute& route) {
  nb::dict out;
  out["expert_id"] = route.expert_id;
  out["owner_rank"] = route.owner_rank;
  out["dst_scaleout_rank"] = route.dst_scaleout_rank;
  out["dst_scaleup_lane"] = route.dst_scaleup_lane;
  out["is_remote_scaleout"] = static_cast<bool>(route.is_remote_scaleout);
  return out;
}

nb::dict dispatch_segment_to_dict(const v2::DispatchSegmentDescriptor& segment) {
  nb::dict out;
  out["dst_scaleout_rank"] = segment.dst_scaleout_rank;
  out["dst_scaleup_lane"] = segment.dst_scaleup_lane;
  out["expert_id"] = segment.expert_id;
  out["count"] = segment.count;
  out["src_token_begin"] = segment.src_token_begin;
  out["src_token_index_offset"] = segment.src_token_index_offset;
  out["topk_slot"] = segment.topk_slot;
  out["expanded_slot_begin"] = segment.expanded_slot_begin;
  out["payload_bytes"] = segment.payload_bytes;
  out["scale_bytes"] = segment.scale_bytes;
  out["flags"] = segment.flags;
  return out;
}

nb::dict dispatch_batch_to_dict(const v2::DispatchExpertBatch& batch) {
  nb::dict out;
  out["dst_scaleout_rank"] = batch.dst_scaleout_rank;
  out["dst_scaleup_lane"] = batch.dst_scaleup_lane;
  out["expert_id"] = batch.expert_id;
  out["first_segment"] = batch.first_segment;
  out["num_segments"] = batch.num_segments;
  out["total_tokens"] = batch.total_tokens;
  return out;
}

nb::dict dispatch_plan_to_dict(const v2::DispatchPlan& plan) {
  nb::list segments;
  for (const auto& segment : plan.segments) {
    segments.append(dispatch_segment_to_dict(segment));
  }
  nb::list batches;
  for (const auto& batch : plan.batches) {
    batches.append(dispatch_batch_to_dict(batch));
  }
  nb::dict out;
  out["segments"] = segments;
  out["batches"] = batches;
  return out;
}

nb::dict combine_segment_to_dict(const v2::CombineSegmentDescriptor& segment) {
  nb::dict out;
  out["dst_original_rank"] = segment.dst_original_rank;
  out["dst_scaleout_rank"] = segment.dst_scaleout_rank;
  out["dst_scaleup_lane"] = segment.dst_scaleup_lane;
  out["src_scaleout_rank"] = segment.src_scaleout_rank;
  out["expert_id"] = segment.expert_id;
  out["count"] = segment.count;
  out["expanded_slot_begin"] = segment.expanded_slot_begin;
  out["expanded_slot_index_offset"] = segment.expanded_slot_index_offset;
  out["topk_slot"] = segment.topk_slot;
  out["reduced_token_slot"] = segment.reduced_token_slot;
  out["payload_bytes"] = segment.payload_bytes;
  out["flags"] = segment.flags;
  return out;
}

nb::dict combine_batch_to_dict(const v2::CombineExpertBatch& batch) {
  nb::dict out;
  out["dst_original_rank"] = batch.dst_original_rank;
  out["dst_scaleout_rank"] = batch.dst_scaleout_rank;
  out["dst_scaleup_lane"] = batch.dst_scaleup_lane;
  out["src_scaleout_rank"] = batch.src_scaleout_rank;
  out["expert_id"] = batch.expert_id;
  out["first_segment"] = batch.first_segment;
  out["num_segments"] = batch.num_segments;
  out["total_tokens"] = batch.total_tokens;
  return out;
}

nb::dict combine_plan_to_dict(const v2::CombinePlan& plan) {
  nb::list segments;
  for (const auto& segment : plan.segments) {
    segments.append(combine_segment_to_dict(segment));
  }
  nb::list batches;
  for (const auto& batch : plan.batches) {
    batches.append(combine_batch_to_dict(batch));
  }
  nb::dict out;
  out["segments"] = segments;
  out["batches"] = batches;
  return out;
}

nb::dict dispatch_transfer_layout_to_dict(
    const v2::DispatchTransferLayout& layout) {
  nb::dict out;
  out["local_payload_base"] = layout.local_payload_base;
  out["remote_payload_base"] = layout.remote_payload_base;
  out["remote_signal_base"] = layout.remote_signal_base;
  out["src_token_stride"] = layout.src_token_stride;
  out["expanded_slot_stride"] = layout.expanded_slot_stride;
  out["batch_payload_stride"] = layout.batch_payload_stride;
  out["signal_stride"] = layout.signal_stride;
  return out;
}

nb::dict combine_transfer_layout_to_dict(
    const v2::CombineTransferLayout& layout) {
  nb::dict out;
  out["local_payload_base"] = layout.local_payload_base;
  out["remote_payload_base"] = layout.remote_payload_base;
  out["remote_signal_base"] = layout.remote_signal_base;
  out["expanded_slot_stride"] = layout.expanded_slot_stride;
  out["reduced_token_stride"] = layout.reduced_token_stride;
  out["batch_payload_stride"] = layout.batch_payload_stride;
  out["signal_stride"] = layout.signal_stride;
  return out;
}

nb::dict transfer_cmd_to_dict(const v2::V2TransferCmd& cmd) {
  nb::dict out;
  out["kind"] = cmd.kind;
  out["target_rank"] = cmd.target_rank;
  out["target_lane"] = cmd.target_lane;
  out["flags"] = cmd.flags;
  out["bytes"] = cmd.bytes;
  out["remote_offset"] = v2::v2_transfer_remote_offset(cmd);
  if (v2::is_v2_transfer_signal(cmd)) {
    out["signal_value"] = cmd.signal_value;
  } else {
    out["local_offset"] = v2::v2_transfer_local_offset(cmd);
  }
  return out;
}

nb::dict jit_launch_plan_to_dict(const v2::V2EfaJitLaunchPlan& plan) {
  nb::dict out;
  out["name"] = plan.name;
  out["source"] = plan.source;
  out["grid_dim_x"] = plan.grid_dim_x;
  out["grid_dim_y"] = plan.grid_dim_y;
  out["num_threads"] = plan.num_threads;
  out["smem_bytes"] = plan.smem_bytes;
  out["cluster_dim"] = plan.cluster_dim;
  out["cooperative"] = plan.cooperative;
  out["pdl_enabled"] = plan.pdl_enabled;
  out["num_notify_warps"] = plan.num_notify_warps;
  out["num_scaleout_warps"] = plan.num_scaleout_warps;
  out["num_forward_warps"] = plan.num_forward_warps;
  out["num_payload_warps"] = plan.num_payload_warps;
  return out;
}

nb::list transfer_cmds_to_list(const v2::V2TransferCmdPlan& plan) {
  nb::list out;
  for (const auto& cmd : plan.commands) {
    out.append(transfer_cmd_to_dict(cmd));
  }
  return out;
}

std::vector<int64_t> sequence_to_i64_vector(const nb::sequence& values) {
  std::vector<int64_t> out;
  out.reserve(static_cast<size_t>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    out.push_back(nb::cast<int64_t>(values[i]));
  }
  return out;
}

}  // namespace

NB_MODULE(ep, m) {
  m.doc() = "DeepEP V2 AWS EFA native backend skeleton";
  m.attr("__native_v2_rewrite__") = true;
  m.attr("__native_v2_ready__") = false;

  nb::class_<Config>(m, "Config")
      .def(nb::init<int, int, int, int, int>(), nb::arg("num_sms") = 0,
           nb::arg("num_max_nvl_chunked_send_tokens") = 0,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 0,
           nb::arg("num_max_rdma_chunked_send_tokens") = 0,
           nb::arg("num_max_rdma_chunked_recv_tokens") = 0)
      .def_rw("num_sms", &Config::num_sms)
      .def_rw("num_max_nvl_chunked_send_tokens",
              &Config::num_max_nvl_chunked_send_tokens)
      .def_rw("num_max_nvl_chunked_recv_tokens",
              &Config::num_max_nvl_chunked_recv_tokens)
      .def_rw("num_max_rdma_chunked_send_tokens",
              &Config::num_max_rdma_chunked_send_tokens)
      .def_rw("num_max_rdma_chunked_recv_tokens",
              &Config::num_max_rdma_chunked_recv_tokens);

  nb::class_<EventHandle>(m, "EventHandle")
      .def(nb::init<>())
      .def("current_stream_wait", &EventHandle::current_stream_wait);

  nb::class_<v2::RuntimeConfig>(m, "V2EfaRuntimeConfig")
      .def(nb::init<>())
      .def_rw("rank", &v2::RuntimeConfig::rank)
      .def_rw("world_size", &v2::RuntimeConfig::world_size)
      .def_rw("scaleout_rank", &v2::RuntimeConfig::scaleout_rank)
      .def_rw("scaleup_rank", &v2::RuntimeConfig::scaleup_rank)
      .def_rw("num_scaleout_ranks", &v2::RuntimeConfig::num_scaleout_ranks)
      .def_rw("num_scaleup_ranks", &v2::RuntimeConfig::num_scaleup_ranks)
      .def_rw("num_experts", &v2::RuntimeConfig::num_experts)
      .def_rw("num_topk", &v2::RuntimeConfig::num_topk)
      .def_rw("hidden", &v2::RuntimeConfig::hidden)
      .def_rw("elem_bytes", &v2::RuntimeConfig::elem_bytes)
      .def_rw("num_sms", &v2::RuntimeConfig::num_sms);

  nb::class_<v2::V2EfaRuntime>(m, "V2EfaRuntime")
      .def(nb::init<v2::RuntimeConfig>())
      .def("is_ready", &v2::V2EfaRuntime::is_ready)
      .def("status", &v2::V2EfaRuntime::status)
      .def("workspace_plan",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank) {
             return workspace_to_dict(
                 self.workspace_plan(num_max_tokens_per_rank));
           })
      .def("worst_case_stats",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank) {
             return stats_to_dict(
                 self.worst_case_stats(num_max_tokens_per_rank));
           })
      .def("route_expert",
           [](const v2::V2EfaRuntime& self, int expert_id) {
             return route_to_dict(self.route_expert(expert_id));
           })
      .def("build_reference_dispatch_plan",
           [](const v2::V2EfaRuntime& self, nb::sequence topk_idx_flat,
              int num_tokens, int payload_bytes, int scale_bytes,
              bool has_topk_weight) {
             if (topk_idx_flat.size() !=
                 static_cast<size_t>(num_tokens * self.config().num_topk)) {
               throw std::invalid_argument(
                   "topk_idx_flat length must equal num_tokens * num_topk");
             }
             auto topk = sequence_to_i64_vector(topk_idx_flat);
             return dispatch_plan_to_dict(self.build_reference_dispatch_plan(
                 topk.data(), num_tokens, payload_bytes, scale_bytes,
                 has_topk_weight));
           })
      .def("build_reference_roundtrip_plan",
           [](const v2::V2EfaRuntime& self, nb::sequence topk_idx_flat,
              int num_tokens, int dispatch_payload_bytes, int scale_bytes,
              int combine_payload_bytes, bool has_topk_weight) {
             if (topk_idx_flat.size() !=
                 static_cast<size_t>(num_tokens * self.config().num_topk)) {
               throw std::invalid_argument(
                   "topk_idx_flat length must equal num_tokens * num_topk");
             }
             auto topk = sequence_to_i64_vector(topk_idx_flat);
             const auto dispatch_plan = self.build_reference_dispatch_plan(
                 topk.data(), num_tokens, dispatch_payload_bytes, scale_bytes,
                 has_topk_weight);
             const auto combine_plan =
                 self.build_reference_combine_plan_from_dispatch(
                     dispatch_plan, combine_payload_bytes);
             nb::dict out;
             out["dispatch"] = dispatch_plan_to_dict(dispatch_plan);
             out["combine"] = combine_plan_to_dict(combine_plan);
             return out;
           })
      .def("build_reference_transfer_roundtrip_plan",
           [](const v2::V2EfaRuntime& self, nb::sequence topk_idx_flat,
              int num_tokens, int dispatch_payload_bytes, int scale_bytes,
              int combine_payload_bytes, bool has_topk_weight) {
             if (topk_idx_flat.size() !=
                 static_cast<size_t>(num_tokens * self.config().num_topk)) {
               throw std::invalid_argument(
                   "topk_idx_flat length must equal num_tokens * num_topk");
             }
             auto topk = sequence_to_i64_vector(topk_idx_flat);
             const auto dispatch_plan = self.build_reference_dispatch_plan(
                 topk.data(), num_tokens, dispatch_payload_bytes, scale_bytes,
                 has_topk_weight);
             const auto combine_plan =
                 self.build_reference_combine_plan_from_dispatch(
                     dispatch_plan, combine_payload_bytes);
             const auto dispatch_layout =
                 v2::make_contiguous_dispatch_transfer_layout(
                     dispatch_plan,
                     static_cast<uint32_t>(dispatch_payload_bytes),
                     static_cast<uint32_t>(dispatch_payload_bytes));
             const auto combine_layout =
                 v2::make_contiguous_combine_transfer_layout(
                     combine_plan,
                     static_cast<uint32_t>(combine_payload_bytes),
                     static_cast<uint32_t>(combine_payload_bytes));
             const auto dispatch_commands =
                 v2::build_dispatch_transfer_cmd_plan(dispatch_plan,
                                                      dispatch_layout);
             const auto combine_commands =
                 v2::build_combine_transfer_cmd_plan(combine_plan,
                                                     combine_layout);
             nb::dict out;
             out["dispatch"] = dispatch_plan_to_dict(dispatch_plan);
             out["combine"] = combine_plan_to_dict(combine_plan);
             out["dispatch_layout"] =
                 dispatch_transfer_layout_to_dict(dispatch_layout);
             out["combine_layout"] =
                 combine_transfer_layout_to_dict(combine_layout);
             out["dispatch_commands"] = transfer_cmds_to_list(dispatch_commands);
             out["combine_commands"] = transfer_cmds_to_list(combine_commands);
             return out;
           })
      .def("build_dispatch_jit_plan",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
              bool cached_mode, bool deterministic, bool do_cpu_sync,
              int smem_bytes, const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(self.build_dispatch_jit_plan(
                 num_max_tokens_per_rank, num_channels_per_sm, scale_bytes,
                 has_topk_weight, cached_mode, deterministic, do_cpu_sync,
                 smem_bytes, uccl_include_path));
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("scale_bytes") = 0,
           nb::arg("has_topk_weight") = true,
           nb::arg("cached_mode") = false,
           nb::arg("deterministic") = false,
           nb::arg("do_cpu_sync") = false,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "")
      .def("compile_dispatch_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
              bool cached_mode, bool deterministic, bool do_cpu_sync,
              int smem_bytes, const std::string& uccl_include_path) {
             const auto plan = self.build_dispatch_jit_plan(
                 num_max_tokens_per_rank, num_channels_per_sm, scale_bytes,
                 has_topk_weight, cached_mode, deterministic, do_cpu_sync,
                 smem_bytes, uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("scale_bytes") = 0,
           nb::arg("has_topk_weight") = true,
           nb::arg("cached_mode") = false,
           nb::arg("deterministic") = false,
           nb::arg("do_cpu_sync") = false,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "")
      .def("build_combine_jit_plan",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels, int payload_bytes, bool use_expanded_layout,
              bool allow_multiple_reduction, int smem_bytes,
              const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(self.build_combine_jit_plan(
                 num_max_tokens_per_rank, num_channels, payload_bytes,
                 use_expanded_layout, allow_multiple_reduction, smem_bytes,
                 uccl_include_path));
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels") = 1,
           nb::arg("payload_bytes") = 0,
           nb::arg("use_expanded_layout") = true,
           nb::arg("allow_multiple_reduction") = true,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "")
      .def("compile_combine_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels, int payload_bytes, bool use_expanded_layout,
              bool allow_multiple_reduction, int smem_bytes,
              const std::string& uccl_include_path) {
             const auto plan = self.build_combine_jit_plan(
                 num_max_tokens_per_rank, num_channels, payload_bytes,
                 use_expanded_layout, allow_multiple_reduction, smem_bytes,
                 uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels") = 1,
           nb::arg("payload_bytes") = 0,
           nb::arg("use_expanded_layout") = true,
           nb::arg("allow_multiple_reduction") = true,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "")
      .def("launch_dispatch_descriptors",
           [](const v2::V2EfaRuntime& self, std::uintptr_t topk_idx_ptr,
              std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
              std::uintptr_t counters_ptr, int num_tokens,
              int num_max_tokens_per_rank, int num_channels_per_sm,
              int scale_bytes, bool has_topk_weight, bool cached_mode,
              bool deterministic, bool do_cpu_sync, int smem_bytes,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             self.launch_dispatch_descriptors(
                 topk_idx_ptr, segments_ptr, batches_ptr, counters_ptr,
                 num_tokens, num_max_tokens_per_rank, num_channels_per_sm,
                 scale_bytes, has_topk_weight, cached_mode, deterministic,
                 do_cpu_sync, smem_bytes, uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("topk_idx_ptr"),
           nb::arg("segments_ptr"),
           nb::arg("batches_ptr"),
           nb::arg("counters_ptr"),
           nb::arg("num_tokens"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("scale_bytes") = 0,
           nb::arg("has_topk_weight") = true,
           nb::arg("cached_mode") = false,
           nb::arg("deterministic") = false,
           nb::arg("do_cpu_sync") = false,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_combine_descriptors",
           [](const v2::V2EfaRuntime& self,
              std::uintptr_t dispatch_segments_ptr,
              std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
              std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
              std::uintptr_t counters_ptr, int num_max_tokens_per_rank,
              int num_channels, int payload_bytes, bool use_expanded_layout,
              bool allow_multiple_reduction, int smem_bytes,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             self.launch_combine_descriptors(
                 dispatch_segments_ptr, dispatch_batches_ptr,
                 num_dispatch_batches, segments_ptr, batches_ptr, counters_ptr,
                 num_max_tokens_per_rank, num_channels, payload_bytes,
                 use_expanded_layout, allow_multiple_reduction, smem_bytes,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("dispatch_segments_ptr"),
           nb::arg("dispatch_batches_ptr"),
           nb::arg("num_dispatch_batches"),
           nb::arg("segments_ptr"),
           nb::arg("batches_ptr"),
           nb::arg("counters_ptr"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels") = 1,
           nb::arg("payload_bytes") = 0,
           nb::arg("use_expanded_layout") = true,
           nb::arg("allow_multiple_reduction") = true,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch", &v2::V2EfaRuntime::launch_dispatch)
      .def("launch_combine", &v2::V2EfaRuntime::launch_combine);

  m.def("v2_descriptor_sizes", []() {
    nb::dict out;
    out["version"] = v2::kDescriptorVersion;
    out["dispatch_segment"] = sizeof(v2::DispatchSegmentDescriptor);
    out["dispatch_batch"] = sizeof(v2::DispatchExpertBatch);
    out["combine_segment"] = sizeof(v2::CombineSegmentDescriptor);
    out["combine_batch"] = sizeof(v2::CombineExpertBatch);
    out["transfer_cmd"] = sizeof(v2::V2TransferCmd);
    return out;
  });

  m.def("is_native_v2_ready", []() { return false; });
  m.def("native_v2_rewrite_message", []() { return std::string(kRewriteMessage); });
  m.def("init_deep_ep_jit", &v2::init_deep_ep_jit_bridge,
        nb::arg("library_root_path"), nb::arg("cuda_home_path"),
        nb::arg("nccl_root_path"));
  m.def("is_deep_ep_jit_initialized", &v2::is_deep_ep_jit_bridge_initialized);
}
