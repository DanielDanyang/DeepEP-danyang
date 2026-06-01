#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/efa_adapter.hpp"
#include "v2_efa/proxy.hpp"
#include "v2_efa/runtime.hpp"
#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_cmd_plan.hpp"
#include "v2_efa/transfer_d2h_queue.cuh"
#include "v2_efa/transfer_layout.hpp"
#include "v2_efa/transfer_loopback.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace uccl::v2_efa;

int main() {
  RuntimeConfig runtime_config;
  runtime_config.world_size = 4;
  runtime_config.rank = 0;
  runtime_config.num_scaleout_ranks = 2;
  runtime_config.num_scaleup_ranks = 2;
  runtime_config.scaleout_rank = 0;
  runtime_config.scaleup_rank = 0;
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
  assert(combine_plan.batches[2].dst_scaleout_rank == 0);
  assert(combine_plan.batches[2].dst_scaleup_lane == 0);
  assert(combine_plan.batches[2].expert_id == 4);
  assert(combine_plan.segments[2].dst_scaleout_rank == 0);
  assert(combine_plan.segments[2].dst_scaleup_lane == 0);
  assert(combine_plan.segments[2].expanded_slot_begin == 0);
  assert(combine_plan.segments[2].topk_slot == 1);
  assert(combine_plan.segments[2].reduced_token_slot == 0);
  assert(combine_plan.segments[2].count == 2);

  auto rank3_config = runtime_config;
  rank3_config.rank = 3;
  rank3_config.scaleout_rank = 1;
  rank3_config.scaleup_rank = 1;
  V2EfaRuntime rank3_runtime(rank3_config);
  const auto rank3_combine_plan =
      rank3_runtime.build_reference_combine_plan_from_dispatch(plan, 32);
  assert(rank3_combine_plan.batches[0].dst_original_rank == 3);
  assert(rank3_combine_plan.batches[0].dst_scaleout_rank == 1);
  assert(rank3_combine_plan.batches[0].dst_scaleup_lane == 1);
  assert(rank3_combine_plan.segments[0].dst_scaleout_rank == 1);
  assert(rank3_combine_plan.segments[0].dst_scaleup_lane == 1);

  const auto workspace = runtime.workspace_plan(4);
  assert(workspace.dispatch_counters.bytes ==
         sizeof(uint32_t) * kDescriptorCounterWords);
  assert(workspace.combine_counters.bytes ==
         sizeof(uint32_t) * kDescriptorCounterWords);

  const auto dispatch_jit_plan = runtime.build_dispatch_jit_plan(
      /*num_max_tokens_per_rank=*/4, /*num_channels_per_sm=*/2,
      /*scale_bytes=*/0, /*has_topk_weight=*/true,
      /*cached_mode=*/false, /*deterministic=*/false,
      /*do_cpu_sync=*/false, /*smem_bytes=*/228 * 1024,
      /*uccl_include_path=*/"/tmp/uccl-ep/include");
  assert(dispatch_jit_plan.name == "v2_efa_dispatch");
  assert(dispatch_jit_plan.grid_dim_x == 1);
  assert(dispatch_jit_plan.num_notify_warps == 4);
  assert(dispatch_jit_plan.num_scaleout_warps == 2);
  assert(dispatch_jit_plan.num_forward_warps == 2);
  assert(dispatch_jit_plan.num_threads == (4 + 2 + 2) * 32);
  assert(dispatch_jit_plan.cooperative);
  assert(dispatch_jit_plan.source.find(
             "#include <deep_ep/impls/hybrid_dispatch.cuh>") !=
         std::string::npos);
  assert(dispatch_jit_plan.source.find(
             "\"/tmp/uccl-ep/include/v2_efa/dispatch_jit.cuh\"") !=
         std::string::npos);
  assert(dispatch_jit_plan.source.find(
             "v2_efa_dispatch_descriptor_kernel<2, 2, 8, 2, 32>") !=
         std::string::npos);

  const auto combine_jit_plan = runtime.build_combine_jit_plan(
      /*num_max_tokens_per_rank=*/4, /*num_channels=*/2,
      /*payload_bytes=*/32, /*use_expanded_layout=*/true,
      /*allow_multiple_reduction=*/true, /*smem_bytes=*/228 * 1024,
      /*uccl_include_path=*/"/tmp/uccl-ep/include");
  assert(combine_jit_plan.name == "v2_efa_combine");
  assert(combine_jit_plan.grid_dim_x == 1);
  assert(combine_jit_plan.num_scaleout_warps == 2);
  assert(combine_jit_plan.num_forward_warps == 2);
  assert(combine_jit_plan.num_threads == (2 + 2) * 32);
  assert(combine_jit_plan.cooperative);
  assert(combine_jit_plan.source.find(
             "#include <deep_ep/impls/hybrid_combine.cuh>") !=
         std::string::npos);
  assert(combine_jit_plan.source.find(
             "\"/tmp/uccl-ep/include/v2_efa/combine_jit.cuh\"") !=
         std::string::npos);
  assert(combine_jit_plan.source.find(
             "v2_efa_combine_descriptor_kernel<2, 2, 8, 2, 16>") !=
         std::string::npos);

  const auto dispatch_enqueue_jit_plan =
      runtime.build_dispatch_enqueue_d2h_jit_plan("/tmp/uccl-ep/include");
  assert(dispatch_enqueue_jit_plan.name == "v2_efa_dispatch_enqueue_d2h");
  assert(dispatch_enqueue_jit_plan.grid_dim_x == 1);
  assert(dispatch_enqueue_jit_plan.num_threads == 32);
  assert(!dispatch_enqueue_jit_plan.cooperative);
  assert(dispatch_enqueue_jit_plan.source.find(
             "\"/tmp/uccl-ep/include/v2_efa/dispatch_jit.cuh\"") !=
         std::string::npos);
  assert(dispatch_enqueue_jit_plan.source.find(
             "v2_efa_dispatch_enqueue_d2h_kernel<0>") !=
         std::string::npos);
  const auto dispatch_fused_jit_plan =
      runtime.build_dispatch_descriptor_enqueue_d2h_jit_plan(
          /*num_max_tokens_per_rank=*/4, /*num_channels_per_sm=*/2,
          /*scale_bytes=*/0, /*has_topk_weight=*/true,
          /*cached_mode=*/false, /*deterministic=*/false,
          /*do_cpu_sync=*/false, /*smem_bytes=*/228 * 1024,
          /*uccl_include_path=*/"/tmp/uccl-ep/include");
  assert(dispatch_fused_jit_plan.name ==
         "v2_efa_dispatch_descriptor_enqueue_d2h");
  assert(dispatch_fused_jit_plan.grid_dim_x == 1);
  assert(dispatch_fused_jit_plan.num_threads == 32);
  assert(dispatch_fused_jit_plan.source.find(
             "v2_efa_dispatch_descriptor_enqueue_d2h_kernel<2, 2, 8, 2, 32>") !=
         std::string::npos);

  const auto combine_enqueue_jit_plan =
      runtime.build_combine_enqueue_d2h_jit_plan("/tmp/uccl-ep/include");
  assert(combine_enqueue_jit_plan.name == "v2_efa_combine_enqueue_d2h");
  assert(combine_enqueue_jit_plan.grid_dim_x == 1);
  assert(combine_enqueue_jit_plan.num_threads == 32);
  assert(!combine_enqueue_jit_plan.cooperative);
  assert(combine_enqueue_jit_plan.source.find(
             "\"/tmp/uccl-ep/include/v2_efa/combine_jit.cuh\"") !=
         std::string::npos);
  assert(combine_enqueue_jit_plan.source.find(
             "v2_efa_combine_enqueue_d2h_kernel<0>") !=
         std::string::npos);
  const auto combine_fused_jit_plan =
      runtime.build_combine_descriptor_enqueue_d2h_jit_plan(
          /*num_max_tokens_per_rank=*/4, /*num_channels=*/2,
          /*payload_bytes=*/32, /*use_expanded_layout=*/true,
          /*allow_multiple_reduction=*/true, /*smem_bytes=*/228 * 1024,
          /*uccl_include_path=*/"/tmp/uccl-ep/include");
  assert(combine_fused_jit_plan.name ==
         "v2_efa_combine_descriptor_enqueue_d2h");
  assert(combine_fused_jit_plan.grid_dim_x == 1);
  assert(combine_fused_jit_plan.num_threads == 32);
  assert(combine_fused_jit_plan.source.find(
             "v2_efa_combine_descriptor_enqueue_d2h_kernel<2, 2, 8, 2, 16>") !=
         std::string::npos);

  const auto dispatch_layout = make_contiguous_dispatch_transfer_layout(
      plan, /*src_token_stride=*/32, /*expanded_slot_stride=*/32,
      /*local_payload_base=*/1000, /*remote_payload_base=*/2000);
  assert(dispatch_layout.batch_payload_stride == 64);
  assert(dispatch_layout.remote_signal_base == 2256);
  const auto dispatch_commands =
      build_dispatch_transfer_cmd_plan(plan, dispatch_layout);
  assert(dispatch_commands.commands.size() == plan.segments.size() +
                                               plan.batches.size());
  assert(dispatch_commands.commands[0].kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kDispatchPayload));
  assert(dispatch_commands.commands[0].bytes == 64);
  assert(v2_transfer_local_offset(dispatch_commands.commands[0]) == 1000);
  assert(v2_transfer_remote_offset(dispatch_commands.commands[0]) == 2000);
  assert(dispatch_commands.commands[0].target_rank == 0);
  assert(dispatch_commands.commands[1].kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kDispatchSignal));
  assert(v2_transfer_remote_offset(dispatch_commands.commands[1]) == 2256);
  assert(dispatch_commands.commands[1].signal_value == 2);
  const auto v2_dispatch_cmd =
      make_v2_dispatch_payload_cmd(plan.segments[0], 0, 0, dispatch_layout);
  assert(sizeof(V2TransferCmd) == 16);
  assert(is_v2_transfer_cmd(v2_dispatch_cmd));
  assert(v2_dispatch_cmd.kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kDispatchPayload));
  const auto v2_dispatch_signal_cmd =
      make_v2_dispatch_signal_cmd(plan.batches[0], 0, dispatch_layout);
  assert(v2_dispatch_signal_cmd.signal_value == 2);
  assert(v2_transfer_local_offset(v2_dispatch_cmd) ==
         v2_transfer_local_offset(dispatch_commands.commands[0]));
  assert(v2_transfer_remote_offset(v2_dispatch_cmd) ==
         v2_transfer_remote_offset(dispatch_commands.commands[0]));
  uint64_t packed_first = 0;
  uint64_t packed_second = 0;
  pack_v2_transfer_cmd(v2_dispatch_cmd, &packed_first, &packed_second);
  const auto unpacked_v2_dispatch_cmd =
      unpack_v2_transfer_cmd(packed_first, packed_second);
  assert(unpacked_v2_dispatch_cmd.kind == v2_dispatch_cmd.kind);
  assert(unpacked_v2_dispatch_cmd.target_rank == v2_dispatch_cmd.target_rank);
  assert(unpacked_v2_dispatch_cmd.target_lane == v2_dispatch_cmd.target_lane);
  assert(unpacked_v2_dispatch_cmd.bytes == v2_dispatch_cmd.bytes);
  assert(v2_transfer_local_offset(unpacked_v2_dispatch_cmd) == 1000);
  assert(v2_transfer_remote_offset(unpacked_v2_dispatch_cmd) == 2000);

  const auto combine_layout = make_contiguous_combine_transfer_layout(
      combine_plan, /*expanded_slot_stride=*/32, /*reduced_token_stride=*/32,
      /*local_payload_base=*/4000, /*remote_payload_base=*/5000);
  assert(combine_layout.batch_payload_stride == 128);
  assert(combine_layout.remote_signal_base == 5512);
  const auto combine_commands =
      build_combine_transfer_cmd_plan(combine_plan, combine_layout);
  assert(combine_commands.commands.size() == combine_plan.segments.size() +
                                              combine_plan.batches.size());
  assert(combine_commands.commands[0].kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kCombinePayload));
  assert(combine_commands.commands[0].bytes == 64);
  assert(v2_transfer_local_offset(combine_commands.commands[0]) == 4000);
  assert(v2_transfer_remote_offset(combine_commands.commands[0]) == 5000);
  assert(combine_commands.commands[0].target_rank == 0);
  assert(combine_commands.commands[0].target_lane == 0);
  const auto rank3_combine_layout = make_contiguous_combine_transfer_layout(
      rank3_combine_plan, /*expanded_slot_stride=*/32,
      /*reduced_token_stride=*/32, /*local_payload_base=*/4000,
      /*remote_payload_base=*/5000);
  const auto rank3_combine_commands =
      build_combine_transfer_cmd_plan(rank3_combine_plan, rank3_combine_layout);
  assert(rank3_combine_commands.commands[0].target_rank == 1);
  assert(rank3_combine_commands.commands[0].target_lane == 1);
  const auto direct_combine_payload = make_v2_combine_payload_cmd(
      combine_plan.segments[0], 0, 0, combine_layout);
  const auto direct_combine_signal =
      make_v2_combine_signal_cmd(combine_plan.batches[0], 0, combine_layout);
  assert(direct_combine_payload.kind == combine_commands.commands[0].kind);
  assert(v2_transfer_local_offset(direct_combine_payload) ==
         v2_transfer_local_offset(combine_commands.commands[0]));
  assert(v2_transfer_remote_offset(direct_combine_payload) ==
         v2_transfer_remote_offset(combine_commands.commands[0]));
  assert(v2_transfer_remote_offset(direct_combine_signal) ==
         v2_transfer_remote_offset(combine_commands.commands[1]));

  std::vector<uint8_t> dispatch_local(4096, 0);
  std::vector<uint8_t> dispatch_remote(4096, 0);
  for (int i = 0; i < 64; ++i) {
    dispatch_local[1000 + i] = static_cast<uint8_t>(i + 1);
  }
  const auto dispatch_stats = execute_loopback_transfer_cmds(
      dispatch_commands.commands,
      LoopbackMemoryView{dispatch_local.data(), dispatch_local.size(),
                         dispatch_remote.data(), dispatch_remote.size()});
  assert(dispatch_stats.payload_commands == 4);
  assert(dispatch_stats.signal_commands == 4);
  assert(dispatch_stats.payload_bytes == 256);
  assert(std::memcmp(dispatch_remote.data() + 2000,
                     dispatch_local.data() + 1000, 64) == 0);
  uint32_t dispatch_signal = 0;
  std::memcpy(&dispatch_signal, dispatch_remote.data() + 2256,
              sizeof(uint32_t));
  assert(dispatch_signal == 2);

  const auto& transfer_cmds = dispatch_commands.commands;
  assert(transfer_cmds.size() == dispatch_commands.commands.size());
  assert(is_v2_transfer_cmd(transfer_cmds[0]));
  assert(transfer_cmds[1].kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kDispatchSignal));
  RecordingEfaPostSink v2_sink;
  drain_v2_transfer_cmds_to_efa_posts(transfer_cmds, v2_sink);
  assert(v2_sink.ops.size() == transfer_cmds.size());
  assert(v2_sink.ops[0].kind == EfaPostOpKind::kWrite);
  assert(v2_sink.ops[1].kind == EfaPostOpKind::kSignalWrite);
  std::vector<std::pair<uint64_t, uint64_t>> packed_transfer_cmds;
  for (const auto& command : transfer_cmds) {
    uint64_t first = 0;
    uint64_t second = 0;
    pack_v2_transfer_cmd(command, &first, &second);
    packed_transfer_cmds.emplace_back(first, second);
  }
  RecordingEfaPostSink packed_v2_sink;
  drain_packed_v2_transfer_cmds_to_efa_posts(packed_transfer_cmds,
                                             packed_v2_sink);
  assert(packed_v2_sink.ops.size() == v2_sink.ops.size());
  assert(packed_v2_sink.ops[0].kind == EfaPostOpKind::kWrite);
  assert(packed_v2_sink.ops[0].local_offset == v2_sink.ops[0].local_offset);
  assert(packed_v2_sink.ops[1].kind == EfaPostOpKind::kSignalWrite);
  assert(packed_v2_sink.ops[1].signal_value == 2);

  HostV2TransferD2HQueue<16> d2h_queue;
  const auto d2h_view = d2h_queue.queue().view();
  assert(d2h_view.commands != nullptr);
  assert(d2h_view.head != nullptr);
  assert(d2h_view.tail != nullptr);
  assert(d2h_view.capacity == 16);
  const auto d2h_stats = d2h_queue.submit(dispatch_commands.commands);
  assert(d2h_stats.submitted == dispatch_commands.commands.size());
  assert(d2h_stats.overflow == 0);
  const auto d2h_ready = d2h_queue.poll_ready();
  assert(d2h_ready.size() == dispatch_commands.commands.size());
  assert(d2h_ready[0].kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kDispatchPayload));
  assert(v2_transfer_local_offset(d2h_ready[0]) == 1000);
  RecordingEfaPostSink d2h_sink;
  drain_v2_transfer_cmds_to_efa_posts(d2h_ready, d2h_sink);
  assert(d2h_sink.ops.size() == v2_sink.ops.size());
  assert(d2h_sink.ops[0].remote_offset == v2_sink.ops[0].remote_offset);
  d2h_queue.ack_ready();
  assert(d2h_queue.queue().volatile_tail() == d2h_queue.queue().volatile_head());

  HostV2TransferD2HQueue<16> partial_ready_queue;
  auto first_cmd = dispatch_commands.commands[0];
  auto second_cmd = dispatch_commands.commands[1];
  auto& partial_ring = partial_ready_queue.queue();
  partial_ring.commands[0] = first_cmd;
  const auto first_header = v2_transfer_cmd_header(first_cmd);
  __atomic_store_n(reinterpret_cast<uint32_t*>(&partial_ring.commands[0]),
                   first_header, __ATOMIC_RELEASE);
  auto hidden_second = second_cmd;
  hidden_second.kind = 0;
  partial_ring.commands[1] = hidden_second;
  partial_ring.head = 2;
  partial_ring.tail = 0;
  uint64_t ready_end = 0;
  const auto first_ready = partial_ready_queue.poll_ready(&ready_end);
  assert(first_ready.size() == 1);
  assert(ready_end == 1);
  const auto second_header = v2_transfer_cmd_header(second_cmd);
  __atomic_store_n(reinterpret_cast<uint32_t*>(&partial_ring.commands[1]),
                   second_header, __ATOMIC_RELEASE);
  partial_ready_queue.ack_ready_until(ready_end);
  assert(partial_ring.volatile_tail() == 1);
  const auto second_ready = partial_ready_queue.poll_ready(&ready_end);
  assert(second_ready.size() == 1);
  assert(second_ready[0].kind == second_cmd.kind);
  partial_ready_queue.ack_ready_until(ready_end);
  assert(partial_ring.volatile_tail() == partial_ring.volatile_head());

  HostV2TransferD2HQueue<16> adapter_d2h_queue;
  adapter_d2h_queue.submit(dispatch_commands.commands);
  RecordingEfaPostSink adapter_d2h_sink;
  const auto drained_count =
      drain_v2_d2h_queue_to_efa_posts(adapter_d2h_queue, adapter_d2h_sink);
  assert(drained_count == dispatch_commands.commands.size());
  assert(adapter_d2h_sink.ops.size() == v2_sink.ops.size());
  assert(adapter_d2h_sink.ops[0].local_offset == v2_sink.ops[0].local_offset);
  assert(adapter_d2h_queue.queue().volatile_tail() ==
         adapter_d2h_queue.queue().volatile_head());

  HostV2TransferD2HQueue<16> proxy_d2h_queue;
  proxy_d2h_queue.submit(dispatch_commands.commands);
  proxy_d2h_queue.submit(combine_commands.commands);
  V2ProxyPostSink proxy_sink;
  HostV2TransferProxy<16> proxy(&proxy_sink);
  proxy.add_queue(&proxy_d2h_queue);
  assert(proxy.num_queues() == 1);
  const auto proxy_drained = proxy.drain_once();
  assert(proxy_drained == dispatch_commands.commands.size() +
                              combine_commands.commands.size());
  assert(proxy.stats().drained_commands == proxy_drained);
  assert(proxy_sink.ops.size() == proxy_drained);
  assert(proxy_sink.stats.posted_writes == 8);
  assert(proxy_sink.stats.posted_signals == 8);
  assert(proxy_d2h_queue.queue().volatile_tail() ==
         proxy_d2h_queue.queue().volatile_head());

  assert(v2_sink.ops[0].target_rank == dispatch_commands.commands[0].target_rank);
  assert(v2_sink.ops[0].target_lane == dispatch_commands.commands[0].target_lane);
  assert(v2_sink.ops[0].local_offset ==
         v2_transfer_local_offset(dispatch_commands.commands[0]));
  assert(v2_sink.ops[0].remote_offset ==
         v2_transfer_remote_offset(dispatch_commands.commands[0]));
  assert(v2_sink.ops[1].signal_value == 2);

  EndpointTable endpoints(/*num_ranks=*/2, /*num_lanes=*/2);
  endpoints.set(EfaRemoteEndpoint{/*rank=*/0, /*lane=*/0,
                                  /*remote_base=*/100000, /*rkey=*/123,
                                  /*bytes=*/8192});
  endpoints.set(EfaRemoteEndpoint{/*rank=*/0, /*lane=*/1,
                                  /*remote_base=*/150000, /*rkey=*/234,
                                  /*bytes=*/8192});
  endpoints.set(EfaRemoteEndpoint{/*rank=*/1, /*lane=*/0,
                                  /*remote_base=*/200000, /*rkey=*/456,
                                  /*bytes=*/8192});
  endpoints.set(EfaRemoteEndpoint{/*rank=*/1, /*lane=*/1,
                                  /*remote_base=*/300000, /*rkey=*/789,
                                  /*bytes=*/8192});
  assert(endpoints.get(0, 0).rkey == 123);
  assert(endpoints.get(1, 0).remote_base == 200000);
  const auto resolved_ops = resolve_efa_post_ops(v2_sink.ops, endpoints);
  assert(resolved_ops.size() == v2_sink.ops.size());
  assert(resolved_ops[0].remote_addr == 102000);
  assert(resolved_ops[0].rkey == 123);
  RecordingEfaPostSink rank3_sink;
  drain_v2_transfer_cmds_to_efa_posts(rank3_combine_commands.commands,
                                      rank3_sink);
  const auto resolved_rank3_ops = resolve_efa_post_ops(rank3_sink.ops, endpoints);
  assert(resolved_rank3_ops[0].target_rank == 1);
  assert(resolved_rank3_ops[0].target_lane == 1);
  assert(resolved_rank3_ops[0].remote_addr == 305000);
  assert(resolved_rank3_ops[0].rkey == 789);
  HostV2TransferD2HQueue<16> resolving_proxy_queue;
  resolving_proxy_queue.submit(dispatch_commands.commands);
  resolving_proxy_queue.submit(combine_commands.commands);
  RecordingResolvedEfaPostSink resolved_proxy_sink;
  ResolvingEfaPostSink resolving_sink(&endpoints, &resolved_proxy_sink);
  HostV2TransferProxy<16> resolving_proxy(&resolving_sink);
  resolving_proxy.add_queue(&resolving_proxy_queue);
  assert(resolving_proxy.drain_once() == dispatch_commands.commands.size() +
                                           combine_commands.commands.size());
  assert(resolved_proxy_sink.ops.size() == dispatch_commands.commands.size() +
                                             combine_commands.commands.size());
  assert(resolved_proxy_sink.ops[0].remote_addr == 102000);
  assert(resolving_proxy_queue.queue().volatile_tail() ==
         resolving_proxy_queue.queue().volatile_head());
  RecordingEfaPostSink coalesced_sink;
  CoalescingEfaPostSink coalescing_sink(&coalesced_sink);
  coalescing_sink.post(EfaPostOp{EfaPostOpKind::kWrite,
                                 /*target_rank=*/1,
                                 /*target_lane=*/1,
                                 /*bytes=*/32,
                                 /*signal_value=*/0,
                                 /*local_offset=*/64,
                                 /*remote_offset=*/128});
  coalescing_sink.post(EfaPostOp{EfaPostOpKind::kWrite,
                                 /*target_rank=*/1,
                                 /*target_lane=*/1,
                                 /*bytes=*/32,
                                 /*signal_value=*/0,
                                 /*local_offset=*/96,
                                 /*remote_offset=*/160});
  coalescing_sink.post(EfaPostOp{EfaPostOpKind::kSignalWrite,
                                 /*target_rank=*/1,
                                 /*target_lane=*/1,
                                 /*bytes=*/4,
                                 /*signal_value=*/2,
                                 /*local_offset=*/0,
                                 /*remote_offset=*/256});
  coalescing_sink.flush();
  assert(coalesced_sink.ops.size() == 2);
  assert(coalesced_sink.ops[0].kind == EfaPostOpKind::kWrite);
  assert(coalesced_sink.ops[0].bytes == 64);
  assert(coalesced_sink.ops[0].local_offset == 64);
  assert(coalesced_sink.ops[0].remote_offset == 128);
  assert(coalesced_sink.ops[1].kind == EfaPostOpKind::kSignalWrite);
  assert(coalesced_sink.ops[1].signal_value == 2);
  auto oversized = v2_sink.ops[0];
  oversized.remote_offset = 8188;
  oversized.bytes = 8;
  bool saw_oob = false;
  try {
    (void)resolve_efa_post_op(oversized, endpoints);
  } catch (const std::out_of_range&) {
    saw_oob = true;
  }
  assert(saw_oob);

  std::vector<uint8_t> combine_local(8192, 0);
  std::vector<uint8_t> combine_remote(8192, 0);
  for (int i = 0; i < 64; ++i) {
    combine_local[4000 + i] = static_cast<uint8_t>(255 - i);
  }
  const auto combine_stats = execute_loopback_transfer_cmds(
      combine_commands.commands,
      LoopbackMemoryView{combine_local.data(), combine_local.size(),
                         combine_remote.data(), combine_remote.size()});
  assert(combine_stats.payload_commands == 4);
  assert(combine_stats.signal_commands == 4);
  assert(combine_stats.payload_bytes == 256);
  assert(std::memcmp(combine_remote.data() + 5000,
                     combine_local.data() + 4000, 64) == 0);
  uint32_t combine_signal = 0;
  std::memcpy(&combine_signal, combine_remote.data() + 5512,
              sizeof(uint32_t));
  assert(combine_signal == 2);

  return 0;
}
