#pragma once

#include <cstdint>
#include <string>

#include "v2_efa/combine_plan.hpp"
#include "v2_efa/descriptor.hpp"
#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/jit_plan.hpp"
#include "v2_efa/topology.hpp"
#include "v2_efa/transfer_cmd.hpp"
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

void init_deep_ep_jit_bridge(const std::string& library_root_path,
                             const std::string& cuda_home_path,
                             const std::string& nccl_root_path);
bool is_deep_ep_jit_bridge_initialized();
void compile_v2_efa_jit_plan(const V2EfaJitLaunchPlan& plan);
void launch_v2_efa_dispatch_descriptor_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t topk_idx_ptr,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_tokens, int scaleout_rank,
    int scaleup_rank, int scale_bytes, bool has_topk_weight, int max_segments,
    int max_batches, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_combine_descriptor_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t dispatch_segments_ptr,
    std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, int num_batches, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    DispatchTransferLayout layout, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_combine_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, int num_batches, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    CombineTransferLayout layout, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_descriptor_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t topk_idx_ptr,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int num_tokens, int scaleout_rank,
    int scaleup_rank, int scale_bytes, bool has_topk_weight, int max_segments,
    int max_batches, std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, DispatchTransferLayout layout,
    std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_direct_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t topk_idx_ptr,
    int num_tokens, int scaleout_rank, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    DispatchTransferLayout layout, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_forward_metadata_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t recv_topk_idx_ptr,
    std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t token_metadata_at_forward_ptr,
    std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
    int rows_per_channel, int scaleup_rank, bool do_expand,
    std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_receiver_metadata_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t recv_topk_idx_ptr,
    std::uintptr_t recv_src_global_ptr, std::uintptr_t recv_counts_per_rank_ptr,
    std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t dst_buffer_slot_idx_ptr,
    std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
    std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
    std::uintptr_t expert_counts_aligned_ptr,
    std::uintptr_t expert_counts_scratch_ptr,
    std::uintptr_t next_expanded_scratch_ptr, int num_recv_tokens,
    int num_source_tokens, int num_max_tokens_per_rank, int rank,
    int expert_alignment, bool do_expand,
    std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_materialize_records_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t window_ptr,
    std::uintptr_t batch_counts_ptr, std::uintptr_t batch_offsets_ptr,
    std::uintptr_t recv_x_ptr, std::uintptr_t recv_topk_idx_ptr,
    std::uintptr_t recv_topk_weights_ptr, std::uintptr_t recv_src_global_ptr,
    int num_sources, int max_batches, int num_max_tokens_per_rank,
    int num_experts, int rank, DispatchTransferLayout layout,
    bool has_topk_weight, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_signal_offsets_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t window_ptr,
    std::uintptr_t batch_counts_ptr, std::uintptr_t batch_offsets_ptr,
    std::uintptr_t recv_counts_per_rank_ptr,
    std::uintptr_t total_recv_tokens_ptr, int num_sources, int max_batches,
    DispatchTransferLayout layout, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_dispatch_expand_records_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t recv_x_ptr,
    std::uintptr_t recv_topk_weights_ptr,
    std::uintptr_t recv_src_metadata_ptr, std::uintptr_t expanded_x_ptr,
    std::uintptr_t expanded_topk_weights_ptr, int num_recv_tokens,
    int num_expanded_tokens, bool has_topk_weight,
    std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_combine_descriptor_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t dispatch_segments_ptr,
    std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
    std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
    std::uintptr_t counters_ptr, int dst_original_rank, int payload_bytes,
    int max_segments, int max_batches, std::uintptr_t commands_ptr,
    std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
    CombineTransferLayout layout, std::uintptr_t cuda_stream_ptr = 0);
void launch_v2_efa_combine_forward_metadata_enqueue_d2h_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t forward_metadata_ptr,
    std::uintptr_t channel_linked_list_ptr, std::uintptr_t segments_ptr,
    std::uintptr_t batches_ptr, std::uintptr_t counters_ptr,
    int num_forward_rows, int scaleout_rank, int num_max_tokens_per_rank,
    int payload_bytes, int max_segments, int max_batches,
    std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
    std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
    std::uintptr_t cuda_stream_ptr = 0);

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
  V2EfaJitLaunchPlan build_dispatch_jit_plan(
      int num_max_tokens_per_rank, int num_channels_per_sm,
      int scale_bytes, bool has_topk_weight, bool cached_mode,
      bool deterministic, bool do_cpu_sync, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_combine_jit_plan(
      int num_max_tokens_per_rank, int num_channels, int payload_bytes,
      bool use_expanded_layout, bool allow_multiple_reduction, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_enqueue_d2h_jit_plan(
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_combine_enqueue_d2h_jit_plan(
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_descriptor_enqueue_d2h_jit_plan(
      int num_max_tokens_per_rank, int num_channels_per_sm,
      int scale_bytes, bool has_topk_weight, bool cached_mode,
      bool deterministic, bool do_cpu_sync, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_direct_enqueue_d2h_jit_plan(
      int num_max_tokens_per_rank, int num_channels_per_sm,
      int scale_bytes, bool has_topk_weight, bool cached_mode,
      bool deterministic, bool do_cpu_sync, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_forward_metadata_jit_plan(
      int num_max_tokens_per_rank, int num_channels_per_sm,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_receiver_metadata_jit_plan(
      int num_max_tokens_per_rank,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_materialize_records_jit_plan(
      int num_max_tokens_per_rank,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_signal_offsets_jit_plan(
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_expand_records_jit_plan(
      int num_max_tokens_per_rank,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_combine_descriptor_enqueue_d2h_jit_plan(
      int num_max_tokens_per_rank, int num_channels, int payload_bytes,
      bool use_expanded_layout, bool allow_multiple_reduction, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan
  build_combine_forward_metadata_enqueue_d2h_jit_plan(
      int num_max_tokens_per_rank, int num_channels, int payload_bytes,
      bool use_expanded_layout, bool allow_multiple_reduction, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  void launch_dispatch_descriptors(
      std::uintptr_t topk_idx_ptr, std::uintptr_t segments_ptr,
      std::uintptr_t batches_ptr, std::uintptr_t counters_ptr, int num_tokens,
      int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
      bool has_topk_weight, bool cached_mode, bool deterministic,
      bool do_cpu_sync, int smem_bytes,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_combine_descriptors(
      std::uintptr_t dispatch_segments_ptr,
      std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
      std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
      std::uintptr_t counters_ptr, int num_max_tokens_per_rank,
      int num_channels, int payload_bytes, bool use_expanded_layout,
      bool allow_multiple_reduction, int smem_bytes,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_enqueue_d2h(
      std::uintptr_t segments_ptr, std::uintptr_t batches_ptr, int num_batches,
      std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
      std::uintptr_t tail_ptr, int queue_capacity,
      DispatchTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_combine_enqueue_d2h(
      std::uintptr_t segments_ptr, std::uintptr_t batches_ptr, int num_batches,
      std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
      std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_descriptor_enqueue_d2h(
      std::uintptr_t topk_idx_ptr, std::uintptr_t segments_ptr,
      std::uintptr_t batches_ptr, std::uintptr_t counters_ptr, int num_tokens,
      int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
      bool has_topk_weight, bool cached_mode, bool deterministic,
      bool do_cpu_sync, int smem_bytes, std::uintptr_t commands_ptr,
      std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
      DispatchTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_direct_enqueue_d2h(
      std::uintptr_t topk_idx_ptr, int num_tokens,
      int num_max_tokens_per_rank, int num_channels_per_sm, int scale_bytes,
      bool has_topk_weight, bool cached_mode, bool deterministic,
      bool do_cpu_sync, int smem_bytes, std::uintptr_t commands_ptr,
      std::uintptr_t head_ptr, std::uintptr_t tail_ptr, int queue_capacity,
      DispatchTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_forward_metadata(
      std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_src_metadata_ptr,
      std::uintptr_t token_metadata_at_forward_ptr,
      std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
      int num_max_tokens_per_rank, int num_channels_per_sm,
      int rows_per_channel, bool do_expand,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_receiver_metadata(
      std::uintptr_t recv_topk_idx_ptr, std::uintptr_t recv_src_global_ptr,
      std::uintptr_t recv_counts_per_rank_ptr,
      std::uintptr_t recv_src_metadata_ptr,
      std::uintptr_t dst_buffer_slot_idx_ptr,
      std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
      std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
      std::uintptr_t expert_counts_aligned_ptr,
      std::uintptr_t expert_counts_scratch_ptr,
      std::uintptr_t next_expanded_scratch_ptr, int num_recv_tokens,
      int num_source_tokens, int num_max_tokens_per_rank,
      int expert_alignment, bool do_expand,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_materialize_records(
      std::uintptr_t window_ptr, std::uintptr_t batch_counts_ptr,
      std::uintptr_t batch_offsets_ptr, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_topk_idx_ptr, std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_src_global_ptr, int max_batches,
      int num_max_tokens_per_rank, DispatchTransferLayout layout,
      bool has_topk_weight, const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_signal_offsets(
      std::uintptr_t window_ptr, std::uintptr_t batch_counts_ptr,
      std::uintptr_t batch_offsets_ptr,
      std::uintptr_t recv_counts_per_rank_ptr,
      std::uintptr_t total_recv_tokens_ptr, int max_batches,
      DispatchTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_expand_records(
      std::uintptr_t recv_x_ptr, std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_src_metadata_ptr, std::uintptr_t expanded_x_ptr,
      std::uintptr_t expanded_topk_weights_ptr, int num_recv_tokens,
      int num_expanded_tokens, bool has_topk_weight,
      int num_max_tokens_per_rank,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_combine_descriptor_enqueue_d2h(
      std::uintptr_t dispatch_segments_ptr,
      std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
      std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
      std::uintptr_t counters_ptr, int num_max_tokens_per_rank,
      int num_channels, int payload_bytes, bool use_expanded_layout,
      bool allow_multiple_reduction, int smem_bytes,
      std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
      std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_combine_forward_metadata_enqueue_d2h(
      std::uintptr_t forward_metadata_ptr,
      std::uintptr_t channel_linked_list_ptr, std::uintptr_t segments_ptr,
      std::uintptr_t batches_ptr, std::uintptr_t counters_ptr,
      int num_forward_rows, int num_max_tokens_per_rank, int num_channels,
      int payload_bytes, bool use_expanded_layout,
      bool allow_multiple_reduction, int smem_bytes,
      std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
      std::uintptr_t tail_ptr, int queue_capacity, CombineTransferLayout layout,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;

  [[noreturn]] void launch_dispatch() const;
  [[noreturn]] void launch_combine() const;

 private:
  RuntimeConfig config_;
};

}  // namespace uccl::v2_efa
