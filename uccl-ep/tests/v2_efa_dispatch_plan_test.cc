#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/efa_adapter.hpp"
#include "v2_efa/runtime.hpp"
#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_cmd_plan.hpp"
#include "v2_efa/transfer_d2h_queue.cuh"
#include "v2_efa/transfer_layout.hpp"
#include "v2_efa/transfer_loopback.hpp"
#include "v2_efa/transfer_queue_host.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
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

  std::vector<uint8_t> queued_dispatch_remote(4096, 0);
  HostV2TransferQueue dispatch_queue(
      static_cast<uint32_t>(dispatch_commands.commands.size()));
  const auto queue_stats =
      submit_v2_transfer_cmds(dispatch_queue, dispatch_commands.commands);
  assert(queue_stats.submitted == dispatch_commands.commands.size());
  assert(queue_stats.overflow == 0);
  assert(dispatch_queue.tail() == dispatch_commands.commands.size());
  const auto transfer_cmds = dispatch_queue.snapshot();
  assert(transfer_cmds.size() == dispatch_commands.commands.size());
  assert(is_v2_transfer_cmd(transfer_cmds[0]));
  assert(transfer_cmds[1].kind ==
         static_cast<uint8_t>(V2TransferCmdKind::kDispatchSignal));
  auto v2_view = dispatch_queue.view();
  assert(v2_view.capacity == transfer_cmds.size());
  RecordingEfaPostSink v2_sink;
  drain_v2_transfer_cmds_to_efa_posts(dispatch_queue.snapshot(), v2_sink);
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

  const auto queued_dispatch_stats = dispatch_queue.drain_loopback(
      LoopbackMemoryView{dispatch_local.data(), dispatch_local.size(),
                         queued_dispatch_remote.data(),
                         queued_dispatch_remote.size()});
  assert(queued_dispatch_stats.payload_commands == 4);
  assert(std::memcmp(queued_dispatch_remote.data() + 2000,
                     dispatch_local.data() + 1000, 64) == 0);

  assert(v2_sink.ops[0].target_rank == dispatch_commands.commands[0].target_rank);
  assert(v2_sink.ops[0].target_lane == dispatch_commands.commands[0].target_lane);
  assert(v2_sink.ops[0].local_offset ==
         v2_transfer_local_offset(dispatch_commands.commands[0]));
  assert(v2_sink.ops[0].remote_offset ==
         v2_transfer_remote_offset(dispatch_commands.commands[0]));
  assert(v2_sink.ops[1].signal_value == 2);

  EndpointTable endpoints(/*num_ranks=*/2, /*num_lanes=*/2);
  endpoints.set(EfaRemoteEndpoint{/*rank=*/0, /*lane=*/0,
                                  /*remote_base=*/2000, /*rkey=*/123,
                                  /*bytes=*/4096});
  endpoints.set(EfaRemoteEndpoint{/*rank=*/1, /*lane=*/0,
                                  /*remote_base=*/2256, /*rkey=*/456,
                                  /*bytes=*/4096});
  assert(endpoints.get(0, 0).rkey == 123);
  assert(endpoints.get(1, 0).remote_base == 2256);

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
