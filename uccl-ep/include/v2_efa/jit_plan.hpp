#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace uccl::v2_efa {

struct V2EfaJitLaunchPlan {
  std::string name;
  std::string source;
  int grid_dim_x = 0;
  int grid_dim_y = 1;
  int num_threads = 0;
  int smem_bytes = 0;
  int cluster_dim = 1;
  bool cooperative = false;
  bool pdl_enabled = false;
  int num_notify_warps = 0;
  int num_scaleout_warps = 0;
  int num_forward_warps = 0;
  int num_payload_warps = 0;
};

struct V2EfaDispatchJitConfig {
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int num_experts = 1;
  int num_topk = 1;
  int hidden = 1;
  int elem_bytes = 2;
  int num_sms = 1;
  int num_channels_per_sm = 1;
  int num_max_tokens_per_rank = 1;
  int scaleout_rank = 0;
  int scaleup_rank = 0;
  int scale_bytes = 0;
  bool has_topk_weight = true;
  bool cached_mode = false;
  bool deterministic = false;
  bool do_cpu_sync = false;
  int smem_bytes = 228 * 1024;
  std::string uccl_include_path;
};

struct V2EfaCombineJitConfig {
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int num_experts = 1;
  int num_topk = 1;
  int hidden = 1;
  int num_sms = 1;
  int num_channels = 1;
  int num_max_tokens_per_rank = 1;
  int dst_original_rank = 0;
  int payload_bytes = 0;
  bool use_expanded_layout = true;
  bool allow_multiple_reduction = true;
  int smem_bytes = 228 * 1024;
  std::string uccl_include_path;
};

inline void validate_v2_efa_jit_common(int num_scaleout_ranks,
                                       int num_scaleup_ranks,
                                       int num_experts, int num_topk,
                                       int num_sms,
                                       int num_max_tokens_per_rank) {
  if (num_scaleout_ranks <= 0 || num_scaleup_ranks <= 0 || num_experts <= 0 ||
      num_topk <= 0 || num_sms <= 0 || num_max_tokens_per_rank <= 0) {
    throw std::invalid_argument("invalid V2 EFA JIT dimensions");
  }
  const int world_size = num_scaleout_ranks * num_scaleup_ranks;
  if (num_experts % world_size != 0) {
    throw std::invalid_argument(
        "num_experts must be divisible by V2 EFA world size");
  }
}

inline int align_up_int(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline int v2_token_layout_bytes(int hidden_bytes, int sf_bytes, int num_topk,
                                 bool with_metadata, bool with_mbarrier) {
  constexpr int kTmaAlignBytes = 32;
  constexpr int kMBarrierBytes = 8;
  const int metadata_bytes =
      num_topk * (static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float))) +
      (with_metadata ? (1 + num_topk) * static_cast<int>(sizeof(int)) : 0);
  return align_up_int(hidden_bytes, kTmaAlignBytes) +
         align_up_int(sf_bytes, kTmaAlignBytes) +
         align_up_int(metadata_bytes, kTmaAlignBytes) +
         align_up_int(with_mbarrier ? kMBarrierBytes : 0, kTmaAlignBytes);
}

inline std::string quote_include(const std::string& include_root,
                                 const char* header) {
  if (include_root.empty()) {
    return std::string("<") + header + ">";
  }
  auto root = include_root;
  while (!root.empty() && root.back() == '/') {
    root.pop_back();
  }
  return std::string("\"") + root + "/" + header + "\"";
}

inline int default_dispatch_num_sms(const V2EfaDispatchJitConfig& config) {
  return config.num_sms > 0 ? config.num_sms : 1;
}

inline int default_combine_num_sms(const V2EfaCombineJitConfig& config) {
  return config.num_sms > 0 ? config.num_sms : 1;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_jit_plan(
    V2EfaDispatchJitConfig config) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0 || config.elem_bytes <= 0 ||
      config.num_channels_per_sm <= 0 || config.smem_bytes <= 0) {
    throw std::invalid_argument("invalid V2 EFA dispatch JIT config");
  }
  if (config.deterministic && config.num_scaleout_ranks > 1) {
    throw std::invalid_argument(
        "hybrid V2 dispatch does not support deterministic mode");
  }

  constexpr int kNumNotifyWarps = 4;
  const int hidden_bytes = config.hidden * config.elem_bytes;
  const int num_notify_warps = config.cached_mode ? 0 : kNumNotifyWarps;
  const int notify_smem_bytes =
      config.cached_mode
          ? 0
          : align_up_int(config.num_scaleout_ranks * config.num_scaleup_ranks +
                             config.num_experts,
                         kNumNotifyWarps * 32) *
                static_cast<int>(sizeof(int));

  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch";
  plan.grid_dim_x = config.num_sms;
  plan.grid_dim_y = 1;
  // The current native V2 EFA descriptor scaffold does not use dynamic shared
  // memory. Keep this at zero so the reference descriptor kernel can launch on
  // all devices while the production V2 payload path is still being wired in.
  plan.smem_bytes = 0;
  plan.cluster_dim = 2 - (config.num_sms % 2);
  plan.cooperative = true;
  plan.pdl_enabled = false;
  plan.num_notify_warps = num_notify_warps;

  if (config.num_scaleout_ranks == 1) {
    const int token_bytes =
        v2_token_layout_bytes(hidden_bytes, config.scale_bytes,
                              config.num_topk, true, true);
    plan.num_payload_warps = std::min<int>(
        std::min<int>((config.smem_bytes - notify_smem_bytes) / token_bytes,
                      32 - num_notify_warps),
        (512 + config.num_sms - 1) / config.num_sms);
    plan.num_threads = (num_notify_warps + plan.num_payload_warps) * 32;
  } else {
    plan.num_scaleout_warps = config.num_channels_per_sm;
    plan.num_forward_warps = config.num_channels_per_sm;
    plan.num_threads =
        (num_notify_warps + plan.num_scaleout_warps + plan.num_forward_warps) *
        32;
  }

  std::ostringstream source;
  source
      << "#include <deep_ep/impls/"
      << (config.num_scaleout_ranks == 1 ? "dispatch" : "hybrid_dispatch")
      << ".cuh>\n"
      << "#include "
      << quote_include(config.uccl_include_path, "v2_efa/dispatch_jit.cuh")
      << "\n\n"
      << "using namespace uccl::v2_efa;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&"
      << "v2_efa_dispatch_descriptor_kernel<" << config.num_scaleout_ranks
      << ", " << config.num_scaleup_ranks << ", " << config.num_experts
      << ", " << config.num_topk << ", " << hidden_bytes << ">);\n"
      << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_combine_jit_plan(
    V2EfaCombineJitConfig config) {
  config.num_sms = default_combine_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0 || config.num_channels <= 0 ||
      config.smem_bytes <= 0) {
    throw std::invalid_argument("invalid V2 EFA combine JIT config");
  }
  if (config.payload_bytes == 0) {
    config.payload_bytes = config.hidden * static_cast<int>(sizeof(uint16_t));
  }

  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_combine";
  plan.grid_dim_x = config.num_sms;
  plan.grid_dim_y = 1;
  plan.smem_bytes = 0;
  plan.cluster_dim = 2 - (config.num_sms % 2);
  plan.cooperative = true;
  plan.pdl_enabled = false;

  const int token_bytes =
      v2_token_layout_bytes(config.hidden * static_cast<int>(sizeof(uint16_t)),
                            0, config.num_topk, false, true);
  int num_warps = std::min(config.smem_bytes / token_bytes, 32);
  if (config.num_scaleout_ranks > 1) {
    if (config.num_channels % config.num_sms != 0 ||
        config.num_channels / config.num_sms > 16) {
      throw std::invalid_argument(
          "invalid V2 EFA combine channel/SM configuration");
    }
    plan.num_scaleout_warps = config.num_channels / config.num_sms;
    plan.num_forward_warps = plan.num_scaleout_warps;
    num_warps = plan.num_scaleout_warps + plan.num_forward_warps;
    if (num_warps * token_bytes > config.smem_bytes) {
      throw std::invalid_argument("invalid V2 EFA combine SM count");
    }
  }
  plan.num_payload_warps = num_warps;
  plan.num_threads = num_warps * 32;

  std::ostringstream source;
  source
      << "#include <deep_ep/impls/"
      << (config.num_scaleout_ranks == 1 ? "combine" : "hybrid_combine")
      << ".cuh>\n"
      << "#include "
      << quote_include(config.uccl_include_path, "v2_efa/combine_jit.cuh")
      << "\n\n"
      << "using namespace uccl::v2_efa;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&"
      << "v2_efa_combine_descriptor_kernel<" << config.num_scaleout_ranks
      << ", " << config.num_scaleup_ranks << ", " << config.num_experts
      << ", " << config.num_topk << ", " << config.hidden << ">);\n"
      << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_enqueue_d2h_jit_plan(
    const std::string& uccl_include_path = "") {
  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch_enqueue_d2h";
  plan.grid_dim_x = 1;
  plan.grid_dim_y = 1;
  plan.num_threads = 32;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;

  std::ostringstream source;
  source << "#include "
         << quote_include(uccl_include_path, "v2_efa/dispatch_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_dispatch_enqueue_d2h_kernel<0>);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_descriptor_enqueue_d2h_jit_plan(
    V2EfaDispatchJitConfig config) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0 || config.elem_bytes <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA fused dispatch descriptor/enqueue JIT config");
  }

  const int hidden_bytes = config.hidden * config.elem_bytes;
  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch_descriptor_enqueue_d2h";
  plan.grid_dim_x = 1;
  plan.grid_dim_y = 1;
  plan.num_threads = 32;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;

  std::ostringstream source;
  source << "#include <deep_ep/impls/"
         << (config.num_scaleout_ranks == 1 ? "dispatch" : "hybrid_dispatch")
         << ".cuh>\n"
         << "#include "
         << quote_include(config.uccl_include_path, "v2_efa/dispatch_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_dispatch_descriptor_enqueue_d2h_kernel<"
         << config.num_scaleout_ranks << ", " << config.num_scaleup_ranks
         << ", " << config.num_experts << ", " << config.num_topk << ", "
         << hidden_bytes << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_direct_enqueue_d2h_jit_plan(
    V2EfaDispatchJitConfig config) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0 || config.elem_bytes <= 0 ||
      config.num_channels_per_sm <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA direct dispatch enqueue JIT config");
  }

  const int hidden_bytes = config.hidden * config.elem_bytes;
  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch_direct_enqueue_d2h";
  plan.grid_dim_x = config.num_sms;
  plan.grid_dim_y = 1;
  plan.num_threads = std::max(32, config.num_channels_per_sm * 32);
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;
  plan.num_scaleout_warps = config.num_channels_per_sm;

  std::ostringstream source;
  source << "#include <deep_ep/impls/"
         << (config.num_scaleout_ranks == 1 ? "dispatch" : "hybrid_dispatch")
         << ".cuh>\n"
         << "#include "
         << quote_include(config.uccl_include_path, "v2_efa/dispatch_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_dispatch_direct_enqueue_d2h_kernel<"
         << config.num_scaleout_ranks << ", " << config.num_scaleup_ranks
         << ", " << config.num_experts << ", " << config.num_topk << ", "
         << hidden_bytes << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_forward_metadata_jit_plan(
    V2EfaDispatchJitConfig config) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.num_channels_per_sm <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA dispatch forward metadata JIT config");
  }

  const int num_channels = config.num_sms * config.num_channels_per_sm;
  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch_forward_metadata";
  plan.grid_dim_x = config.num_sms;
  plan.grid_dim_y = 1;
  plan.num_threads = 256;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;
  plan.num_forward_warps = config.num_channels_per_sm;

  std::ostringstream source;
  source << "#include "
         << quote_include(config.uccl_include_path, "v2_efa/dispatch_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_dispatch_forward_metadata_kernel<"
         << config.num_scaleout_ranks << ", " << config.num_scaleup_ranks
         << ", " << config.num_topk << ", " << num_channels << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_receiver_metadata_jit_plan(
    V2EfaDispatchJitConfig config) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);

  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch_receiver_metadata";
  plan.grid_dim_x = 1;
  plan.grid_dim_y = 1;
  plan.num_threads = 256;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;

  const int world_size = config.num_scaleout_ranks * config.num_scaleup_ranks;
  std::ostringstream source;
  source << "#include "
         << quote_include(config.uccl_include_path, "v2_efa/dispatch_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_dispatch_receiver_metadata_kernel<"
         << world_size << ", " << config.num_scaleup_ranks << ", "
         << config.num_experts << ", " << config.num_topk << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_combine_enqueue_d2h_jit_plan(
    const std::string& uccl_include_path = "") {
  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_combine_enqueue_d2h";
  plan.grid_dim_x = 1;
  plan.grid_dim_y = 1;
  plan.num_threads = 32;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;

  std::ostringstream source;
  source << "#include "
         << quote_include(uccl_include_path, "v2_efa/combine_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_combine_enqueue_d2h_kernel<0>);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_combine_descriptor_enqueue_d2h_jit_plan(
    V2EfaCombineJitConfig config) {
  config.num_sms = default_combine_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA fused combine descriptor/enqueue JIT config");
  }
  if (config.payload_bytes == 0) {
    config.payload_bytes = config.hidden * static_cast<int>(sizeof(uint16_t));
  }

  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_combine_descriptor_enqueue_d2h";
  plan.grid_dim_x = 1;
  plan.grid_dim_y = 1;
  plan.num_threads = 32;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;

  std::ostringstream source;
  source << "#include <deep_ep/impls/"
         << (config.num_scaleout_ranks == 1 ? "combine" : "hybrid_combine")
         << ".cuh>\n"
         << "#include "
         << quote_include(config.uccl_include_path, "v2_efa/combine_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_combine_descriptor_enqueue_d2h_kernel<"
         << config.num_scaleout_ranks << ", " << config.num_scaleup_ranks
         << ", " << config.num_experts << ", " << config.num_topk << ", "
         << config.hidden << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan
build_v2_efa_combine_forward_metadata_enqueue_d2h_jit_plan(
    V2EfaCombineJitConfig config) {
  config.num_sms = default_combine_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA combine forward metadata/enqueue JIT config");
  }
  if (config.payload_bytes == 0) {
    config.payload_bytes = config.hidden * static_cast<int>(sizeof(uint16_t));
  }

  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_combine_forward_metadata_linked_enqueue_d2h";
  plan.grid_dim_x = 1;
  plan.grid_dim_y = 1;
  plan.num_threads = 32;
  plan.smem_bytes = 0;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = false;

  std::ostringstream source;
  source << "#include <deep_ep/impls/"
         << (config.num_scaleout_ranks == 1 ? "combine" : "hybrid_combine")
         << ".cuh>\n"
         << "// UCCL V2 EFA combine forward metadata ABI: linked-list input v1\n"
         << "#include "
         << quote_include(config.uccl_include_path, "v2_efa/combine_jit.cuh")
         << "\n\n"
         << "using namespace uccl::v2_efa;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "v2_efa_combine_forward_metadata_enqueue_d2h_kernel<"
         << config.num_scaleout_ranks << ", " << config.num_scaleup_ranks
         << ", " << config.num_experts << ", " << config.num_topk << ", "
         << config.hidden << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

}  // namespace uccl::v2_efa
