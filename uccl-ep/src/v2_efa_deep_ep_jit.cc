#include "v2_efa/runtime.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <stdexcept>
#include <string>

#include "../../csrc/jit/compiler.hpp"
#include "../../csrc/jit/handle.hpp"
#include "../../csrc/jit/include_parser.hpp"
#include "../../csrc/jit/kernel_runtime.hpp"

namespace uccl::v2_efa {

namespace {

std::once_flag g_jit_init_once;
bool g_jit_initialized = false;

}  // namespace

void init_deep_ep_jit_bridge(const std::string& library_root_path,
                             const std::string& cuda_home_path,
                             const std::string& nccl_root_path) {
  std::call_once(g_jit_init_once, [&] {
    deep_ep::jit::Compiler::prepare_init(library_root_path, cuda_home_path,
                                         nccl_root_path);
    deep_ep::jit::KernelRuntime::prepare_init(cuda_home_path);
    deep_ep::jit::IncludeParser::prepare_init(library_root_path);
    g_jit_initialized = true;
  });
}

bool is_deep_ep_jit_bridge_initialized() { return g_jit_initialized; }

std::shared_ptr<deep_ep::jit::KernelRuntime> build_v2_efa_jit_runtime(
    const V2EfaJitLaunchPlan& plan) {
  if (!is_deep_ep_jit_bridge_initialized()) {
    throw std::runtime_error(
        "DeepEP JIT bridge is not initialized; call init_deep_ep_jit first");
  }
  if (plan.name.empty() || plan.source.empty()) {
    throw std::invalid_argument("empty V2 EFA JIT plan");
  }
  const auto runtime = deep_ep::jit::compiler->build(plan.name, plan.source);
  if (runtime == nullptr) {
    throw std::runtime_error("DeepEP JIT compiler returned null runtime");
  }
  return runtime;
}

deep_ep::jit::LaunchConfigHandle make_launch_config(
    const V2EfaJitLaunchPlan& plan,
    const deep_ep::jit::KernelHandle& kernel,
    std::uintptr_t cuda_stream_ptr) {
  if (plan.grid_dim_x <= 0 || plan.grid_dim_y <= 0 ||
      plan.num_threads <= 0) {
    throw std::invalid_argument("invalid V2 EFA JIT launch dimensions");
  }

  auto stream = cuda_stream_ptr == 0
                    ? at::cuda::getCurrentCUDAStream().stream()
                    : reinterpret_cast<cudaStream_t>(cuda_stream_ptr);
  const dim3 grid_dim{static_cast<unsigned>(plan.grid_dim_x),
                      static_cast<unsigned>(plan.grid_dim_y), 1};
  const dim3 block_dim{static_cast<unsigned>(plan.num_threads), 1, 1};
  return deep_ep::jit::construct_launch_config(
      kernel, stream, plan.smem_bytes, grid_dim, block_dim, plan.cluster_dim,
      plan.cooperative, plan.pdl_enabled);
}

template <typename T>
T* checked_ptr(std::uintptr_t ptr, const char* name) {
  if (ptr == 0) {
    throw std::invalid_argument(std::string(name) + " pointer is null");
  }
  return reinterpret_cast<T*>(ptr);
}

void compile_v2_efa_jit_plan(const V2EfaJitLaunchPlan& plan) {
  (void)build_v2_efa_jit_runtime(plan);
}

void launch_v2_efa_dispatch_descriptor_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t topk_idx_ptr,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_tokens, int scaleout_rank,
    int scaleup_rank, int scale_bytes, bool has_topk_weight, int max_segments,
    int max_batches, std::uintptr_t cuda_stream_ptr) {
  if (num_tokens < 0 || max_segments <= 0 || max_batches <= 0) {
    throw std::invalid_argument("invalid V2 EFA dispatch descriptor launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config =
      make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  EP_CUDA_UNIFIED_CHECK(deep_ep::jit::launch_kernel(
      runtime->kernel, config, checked_ptr<const int64_t>(topk_idx_ptr, "topk_idx"),
      checked_ptr<DispatchSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<DispatchExpertBatch>(batches_ptr, "batches"),
      checked_ptr<uint32_t>(counters_ptr, "counters"), num_tokens,
      scaleout_rank, scaleup_rank, scale_bytes, has_topk_weight, max_segments,
      max_batches));
}

void launch_v2_efa_combine_descriptor_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t dispatch_segments_ptr,
    std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches, std::uintptr_t cuda_stream_ptr) {
  if (num_dispatch_batches < 0 || payload_bytes <= 0 || max_segments <= 0 ||
      max_batches <= 0) {
    throw std::invalid_argument("invalid V2 EFA combine descriptor launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config =
      make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  EP_CUDA_UNIFIED_CHECK(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const DispatchSegmentDescriptor>(dispatch_segments_ptr,
                                                  "dispatch_segments"),
      checked_ptr<const DispatchExpertBatch>(dispatch_batches_ptr,
                                             "dispatch_batches"),
      num_dispatch_batches,
      checked_ptr<CombineSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<CombineExpertBatch>(batches_ptr, "batches"),
      checked_ptr<uint32_t>(counters_ptr, "counters"), dst_original_rank,
      payload_bytes, max_segments, max_batches));
}

void V2EfaRuntime::launch_dispatch_descriptors(
    std::uintptr_t topk_idx_ptr, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, std::uintptr_t counters_ptr, int num_tokens,
    int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
    bool has_topk_weight, bool cached_mode, bool deterministic,
    bool do_cpu_sync, int smem_bytes, const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_dispatch_jit_plan(
      num_max_tokens_per_rank, num_channels_per_sm, scale_bytes,
      has_topk_weight, cached_mode, deterministic, do_cpu_sync, smem_bytes,
      uccl_include_path);
  launch_v2_efa_dispatch_descriptor_plan(
      plan, topk_idx_ptr, segments_ptr, batches_ptr, counters_ptr, num_tokens,
      cfg.scaleout_rank, cfg.scaleup_rank, scale_bytes, has_topk_weight,
      static_cast<int>(max_dispatch_segments(num_max_tokens_per_rank,
                                             cfg.num_topk)),
      static_cast<int>(max_expert_batches(cfg.num_experts,
                                          cfg.num_scaleout_ranks,
                                          cfg.num_scaleup_ranks)),
      cuda_stream_ptr);
}

void V2EfaRuntime::launch_combine_descriptors(
    std::uintptr_t dispatch_segments_ptr,
    std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_max_tokens_per_rank, int num_channels,
    int payload_bytes, bool use_expanded_layout,
    bool allow_multiple_reduction, int smem_bytes,
    const std::string& uccl_include_path, std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_combine_jit_plan(
      num_max_tokens_per_rank, num_channels, payload_bytes,
      use_expanded_layout, allow_multiple_reduction, smem_bytes,
      uccl_include_path);
  launch_v2_efa_combine_descriptor_plan(
      plan, dispatch_segments_ptr, dispatch_batches_ptr, num_dispatch_batches,
      segments_ptr, batches_ptr, counters_ptr, cfg.rank, payload_bytes,
      static_cast<int>(max_dispatch_segments(num_max_tokens_per_rank,
                                             cfg.num_topk)),
      static_cast<int>(max_expert_batches(cfg.num_experts,
                                          cfg.num_scaleout_ranks,
                                          cfg.num_scaleup_ranks)),
      cuda_stream_ptr);
}

}  // namespace uccl::v2_efa
