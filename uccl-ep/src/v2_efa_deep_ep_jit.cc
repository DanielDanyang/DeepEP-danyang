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
#include "v2_efa/transfer_d2h_queue.cuh"

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

int checked_queue_capacity(int capacity) {
  if (capacity <= 0 || (capacity & (capacity - 1)) != 0) {
    throw std::invalid_argument(
        "V2 EFA D2H queue capacity must be a positive power of two");
  }
  return capacity;
}

template <typename Result>
void check_jit_launch_result(Result result) {
  using deep_ep::lazy_cuGetErrorName;
  using deep_ep::lazy_cuGetErrorString;
  EP_CUDA_UNIFIED_CHECK(result);
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
  check_jit_launch_result(deep_ep::jit::launch_kernel(
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
  check_jit_launch_result(deep_ep::jit::launch_kernel(
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

void launch_v2_efa_dispatch_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, int num_batches, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    DispatchTransferLayout layout, std::uintptr_t cuda_stream_ptr) {
  if (num_batches < 0) {
    throw std::invalid_argument("invalid V2 EFA dispatch enqueue launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  V2TransferD2HQueueView queue{
      checked_ptr<V2TransferCmd>(commands_ptr, "commands"),
      checked_ptr<uint64_t>(head_ptr, "head"),
      checked_ptr<uint64_t>(tail_ptr, "tail"),
      static_cast<uint32_t>(checked_queue_capacity(queue_capacity))};
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const DispatchSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<const DispatchExpertBatch>(batches_ptr, "batches"),
      num_batches, queue, layout));
}

void launch_v2_efa_combine_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, int num_batches, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    CombineTransferLayout layout, std::uintptr_t cuda_stream_ptr) {
  if (num_batches < 0) {
    throw std::invalid_argument("invalid V2 EFA combine enqueue launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  V2TransferD2HQueueView queue{
      checked_ptr<V2TransferCmd>(commands_ptr, "commands"),
      checked_ptr<uint64_t>(head_ptr, "head"),
      checked_ptr<uint64_t>(tail_ptr, "tail"),
      static_cast<uint32_t>(checked_queue_capacity(queue_capacity))};
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const CombineSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<const CombineExpertBatch>(batches_ptr, "batches"),
      num_batches, queue, layout));
}

void launch_v2_efa_dispatch_descriptor_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t topk_idx_ptr,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_tokens, int scaleout_rank,
    int scaleup_rank, int scale_bytes, bool has_topk_weight, int max_segments,
    int max_batches, std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, DispatchTransferLayout layout,
    std::uintptr_t cuda_stream_ptr) {
  if (num_tokens < 0 || max_segments <= 0 || max_batches <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA fused dispatch descriptor/enqueue launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  V2TransferD2HQueueView queue{
      checked_ptr<V2TransferCmd>(commands_ptr, "commands"),
      checked_ptr<uint64_t>(head_ptr, "head"),
      checked_ptr<uint64_t>(tail_ptr, "tail"),
      static_cast<uint32_t>(checked_queue_capacity(queue_capacity))};
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config, checked_ptr<const int64_t>(topk_idx_ptr, "topk_idx"),
      checked_ptr<DispatchSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<DispatchExpertBatch>(batches_ptr, "batches"),
      checked_ptr<uint32_t>(counters_ptr, "counters"), num_tokens,
      scaleout_rank, scaleup_rank, scale_bytes, has_topk_weight, max_segments,
      max_batches, queue, layout));
}

void launch_v2_efa_dispatch_direct_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t topk_idx_ptr,
    int num_tokens, int scaleout_rank, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    DispatchTransferLayout layout, std::uintptr_t cuda_stream_ptr) {
  if (num_tokens < 0) {
    throw std::invalid_argument("invalid V2 EFA direct dispatch launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  V2TransferD2HQueueView queue{
      checked_ptr<V2TransferCmd>(commands_ptr, "commands"),
      checked_ptr<uint64_t>(head_ptr, "head"),
      checked_ptr<uint64_t>(tail_ptr, "tail"),
      static_cast<uint32_t>(checked_queue_capacity(queue_capacity))};
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const int64_t>(topk_idx_ptr, "topk_idx"), num_tokens,
      scaleout_rank, queue, layout));
}

void launch_v2_efa_dispatch_forward_metadata_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t recv_topk_idx_ptr,
    std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t token_metadata_at_forward_ptr,
    std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
    int rows_per_channel, int scaleup_rank, bool do_expand,
    std::uintptr_t cuda_stream_ptr) {
  if (num_recv_tokens < 0 || rows_per_channel <= 0 || scaleup_rank < 0) {
    throw std::invalid_argument(
        "invalid V2 EFA dispatch forward metadata launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const int64_t>(recv_topk_idx_ptr, "recv_topk_idx"),
      checked_ptr<const int32_t>(recv_src_metadata_ptr, "recv_src_metadata"),
      checked_ptr<int32_t>(token_metadata_at_forward_ptr,
                           "token_metadata_at_forward"),
      checked_ptr<int32_t>(channel_linked_list_ptr, "channel_linked_list"),
      num_recv_tokens, rows_per_channel, scaleup_rank, do_expand));
}

void launch_v2_efa_combine_descriptor_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t dispatch_segments_ptr,
    std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    CombineTransferLayout layout, std::uintptr_t cuda_stream_ptr) {
  if (num_dispatch_batches < 0 || payload_bytes <= 0 || max_segments <= 0 ||
      max_batches <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA fused combine descriptor/enqueue launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  V2TransferD2HQueueView queue{
      checked_ptr<V2TransferCmd>(commands_ptr, "commands"),
      checked_ptr<uint64_t>(head_ptr, "head"),
      checked_ptr<uint64_t>(tail_ptr, "tail"),
      static_cast<uint32_t>(checked_queue_capacity(queue_capacity))};
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const DispatchSegmentDescriptor>(dispatch_segments_ptr,
                                                  "dispatch_segments"),
      checked_ptr<const DispatchExpertBatch>(dispatch_batches_ptr,
                                             "dispatch_batches"),
      num_dispatch_batches,
      checked_ptr<CombineSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<CombineExpertBatch>(batches_ptr, "batches"),
	      checked_ptr<uint32_t>(counters_ptr, "counters"), dst_original_rank,
	      payload_bytes, max_segments, max_batches, queue, layout));
}

void launch_v2_efa_combine_forward_metadata_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t forward_metadata_ptr,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_forward_rows, int scaleout_rank,
    int num_max_tokens_per_rank, int payload_bytes, int max_segments,
    int max_batches, std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
    std::uintptr_t cuda_stream_ptr) {
  if (num_forward_rows < 0 || num_max_tokens_per_rank <= 0 ||
      payload_bytes <= 0 || max_segments <= 0 || max_batches <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA combine forward metadata/enqueue launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  V2TransferD2HQueueView queue{
      checked_ptr<V2TransferCmd>(commands_ptr, "commands"),
      checked_ptr<uint64_t>(head_ptr, "head"),
      checked_ptr<uint64_t>(tail_ptr, "tail"),
      static_cast<uint32_t>(checked_queue_capacity(queue_capacity))};
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config,
      checked_ptr<const int32_t>(forward_metadata_ptr, "forward_metadata"),
      checked_ptr<CombineSegmentDescriptor>(segments_ptr, "segments"),
      checked_ptr<CombineExpertBatch>(batches_ptr, "batches"),
      checked_ptr<uint32_t>(counters_ptr, "counters"), num_forward_rows,
      scaleout_rank, num_max_tokens_per_rank, payload_bytes, max_segments,
      max_batches, queue, layout));
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

void V2EfaRuntime::launch_dispatch_enqueue_d2h(
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr, int num_batches,
    std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, DispatchTransferLayout layout,
    const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto plan = build_dispatch_enqueue_d2h_jit_plan(uccl_include_path);
  launch_v2_efa_dispatch_enqueue_d2h_plan(
      plan, segments_ptr, batches_ptr, num_batches, commands_ptr, head_ptr,
      tail_ptr, queue_capacity, layout, cuda_stream_ptr);
}

void V2EfaRuntime::launch_combine_enqueue_d2h(
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr, int num_batches,
    std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
    const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto plan = build_combine_enqueue_d2h_jit_plan(uccl_include_path);
  launch_v2_efa_combine_enqueue_d2h_plan(
      plan, segments_ptr, batches_ptr, num_batches, commands_ptr, head_ptr,
      tail_ptr, queue_capacity, layout, cuda_stream_ptr);
}

void V2EfaRuntime::launch_dispatch_descriptor_enqueue_d2h(
    std::uintptr_t topk_idx_ptr, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, std::uintptr_t counters_ptr, int num_tokens,
    int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
    bool has_topk_weight, bool cached_mode, bool deterministic,
    bool do_cpu_sync, int smem_bytes, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    DispatchTransferLayout layout, const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_dispatch_descriptor_enqueue_d2h_jit_plan(
      num_max_tokens_per_rank, num_channels_per_sm, scale_bytes,
      has_topk_weight, cached_mode, deterministic, do_cpu_sync, smem_bytes,
      uccl_include_path);
  launch_v2_efa_dispatch_descriptor_enqueue_d2h_plan(
      plan, topk_idx_ptr, segments_ptr, batches_ptr, counters_ptr, num_tokens,
      cfg.scaleout_rank, cfg.scaleup_rank, scale_bytes, has_topk_weight,
      static_cast<int>(max_dispatch_segments(num_max_tokens_per_rank,
                                             cfg.num_topk)),
      static_cast<int>(max_expert_batches(cfg.num_experts,
                                          cfg.num_scaleout_ranks,
                                          cfg.num_scaleup_ranks)),
      commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
      cuda_stream_ptr);
}

void V2EfaRuntime::launch_dispatch_direct_enqueue_d2h(
    std::uintptr_t topk_idx_ptr, int num_tokens, int num_max_tokens_per_rank,
    int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
    bool cached_mode, bool deterministic, bool do_cpu_sync, int smem_bytes,
    std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, DispatchTransferLayout layout,
    const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_dispatch_direct_enqueue_d2h_jit_plan(
      num_max_tokens_per_rank, num_channels_per_sm, scale_bytes,
      has_topk_weight, cached_mode, deterministic, do_cpu_sync, smem_bytes,
      uccl_include_path);
  launch_v2_efa_dispatch_direct_enqueue_d2h_plan(
      plan, topk_idx_ptr, num_tokens, cfg.scaleout_rank, commands_ptr,
      head_ptr, tail_ptr, queue_capacity, layout, cuda_stream_ptr);
}

void V2EfaRuntime::launch_dispatch_forward_metadata(
    std::uintptr_t recv_topk_idx_ptr, std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t token_metadata_at_forward_ptr,
    std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
    int num_max_tokens_per_rank, int num_channels_per_sm,
    int rows_per_channel, bool do_expand, const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_dispatch_forward_metadata_jit_plan(
      num_max_tokens_per_rank, num_channels_per_sm, uccl_include_path);
  launch_v2_efa_dispatch_forward_metadata_plan(
      plan, recv_topk_idx_ptr, recv_src_metadata_ptr,
      token_metadata_at_forward_ptr, channel_linked_list_ptr, num_recv_tokens,
      rows_per_channel, cfg.scaleup_rank, do_expand, cuda_stream_ptr);
}

void V2EfaRuntime::launch_combine_descriptor_enqueue_d2h(
    std::uintptr_t dispatch_segments_ptr,
    std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_max_tokens_per_rank, int num_channels,
    int payload_bytes, bool use_expanded_layout,
    bool allow_multiple_reduction, int smem_bytes,
    std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
    const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_combine_descriptor_enqueue_d2h_jit_plan(
      num_max_tokens_per_rank, num_channels, payload_bytes,
      use_expanded_layout, allow_multiple_reduction, smem_bytes,
      uccl_include_path);
  launch_v2_efa_combine_descriptor_enqueue_d2h_plan(
      plan, dispatch_segments_ptr, dispatch_batches_ptr, num_dispatch_batches,
      segments_ptr, batches_ptr, counters_ptr, cfg.rank, payload_bytes,
      static_cast<int>(max_dispatch_segments(num_max_tokens_per_rank,
                                             cfg.num_topk)),
      static_cast<int>(max_expert_batches(cfg.num_experts,
                                          cfg.num_scaleout_ranks,
                                          cfg.num_scaleup_ranks)),
	      commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
	      cuda_stream_ptr);
}

void V2EfaRuntime::launch_combine_forward_metadata_enqueue_d2h(
    std::uintptr_t forward_metadata_ptr, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, std::uintptr_t counters_ptr,
    int num_forward_rows, int num_max_tokens_per_rank, int num_channels,
    int payload_bytes, bool use_expanded_layout,
    bool allow_multiple_reduction, int smem_bytes,
    std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
    const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_combine_forward_metadata_enqueue_d2h_jit_plan(
      num_max_tokens_per_rank, num_channels, payload_bytes,
      use_expanded_layout, allow_multiple_reduction, smem_bytes,
      uccl_include_path);
  launch_v2_efa_combine_forward_metadata_enqueue_d2h_plan(
      plan, forward_metadata_ptr, segments_ptr, batches_ptr, counters_ptr,
      num_forward_rows, cfg.scaleout_rank, num_max_tokens_per_rank,
      payload_bytes,
      static_cast<int>(max_dispatch_segments(num_max_tokens_per_rank,
                                             cfg.num_topk)),
      static_cast<int>(max_dispatch_segments(num_max_tokens_per_rank,
                                             cfg.num_topk)),
      commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
      cuda_stream_ptr);
}

}  // namespace uccl::v2_efa
