#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ep_util.hpp"
#include "v2_efa/efa_adapter.hpp"
#include "v2_efa/runtime.hpp"
#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_cmd_plan.hpp"
#include "v2_efa/transfer_d2h_queue.cuh"
#include "v2_efa/transfer_layout.hpp"
#include "v2_efa/verbs_sink.hpp"

#if UCCL_V2_EFA_HAS_VERBS
#ifdef EFA
#include <infiniband/efadv.h>
#endif
#include <cstdlib>
#endif

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

nb::dict efa_post_op_to_dict(const v2::EfaPostOp& op) {
  nb::dict out;
  out["kind"] = static_cast<uint32_t>(op.kind);
  out["target_rank"] = op.target_rank;
  out["target_lane"] = op.target_lane;
  out["bytes"] = op.bytes;
  out["signal_value"] = op.signal_value;
  out["local_offset"] = op.local_offset;
  out["remote_offset"] = op.remote_offset;
  out["descriptor_index"] = op.descriptor_index;
  out["batch_index"] = op.batch_index;
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
  const auto num_values = static_cast<size_t>(nb::len(values));
  out.reserve(num_values);
  for (size_t i = 0; i < num_values; ++i) {
    out.push_back(nb::cast<int64_t>(values[i]));
  }
  return out;
}

class MappedD2HQueueHandle {
 public:
  explicit MappedD2HQueueHandle(uint32_t capacity)
      : capacity_(capacity), commands_bytes_(capacity * sizeof(v2::V2TransferCmd)) {
    if (capacity_ == 0 || (capacity_ & (capacity_ - 1)) != 0) {
      throw std::invalid_argument(
          "V2 mapped D2H queue capacity must be a positive power of two");
    }
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&commands_host_),
                             commands_bytes_, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&head_host_),
                             sizeof(uint64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&tail_host_),
                             sizeof(uint64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&commands_device_), commands_host_, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&head_device_), head_host_, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&tail_device_), tail_host_, 0));
    reset();
  }

  MappedD2HQueueHandle(const MappedD2HQueueHandle&) = delete;
  MappedD2HQueueHandle& operator=(const MappedD2HQueueHandle&) = delete;

  ~MappedD2HQueueHandle() {
    if (commands_host_ != nullptr) {
      cudaFreeHost(commands_host_);
    }
    if (head_host_ != nullptr) {
      cudaFreeHost(head_host_);
    }
    if (tail_host_ != nullptr) {
      cudaFreeHost(tail_host_);
    }
  }

  uint32_t capacity() const { return capacity_; }

  std::uintptr_t commands_ptr() const {
    return reinterpret_cast<std::uintptr_t>(commands_device_);
  }

  std::uintptr_t head_ptr() const {
    return reinterpret_cast<std::uintptr_t>(head_device_);
  }

  std::uintptr_t tail_ptr() const {
    return reinterpret_cast<std::uintptr_t>(tail_device_);
  }

  uint64_t head() const {
    return __atomic_load_n(head_host_, __ATOMIC_ACQUIRE);
  }

  uint64_t tail() const {
    return __atomic_load_n(tail_host_, __ATOMIC_ACQUIRE);
  }

  void reset() {
    for (uint32_t i = 0; i < capacity_; ++i) {
      commands_host_[i] = v2::V2TransferCmd{};
    }
    __atomic_store_n(head_host_, uint64_t{0}, __ATOMIC_RELEASE);
    __atomic_store_n(tail_host_, uint64_t{0}, __ATOMIC_RELEASE);
  }

  std::vector<v2::V2TransferCmd> poll_ready(
      uint64_t* observed_head = nullptr) const {
    std::vector<v2::V2TransferCmd> out;
    const auto h = head();
    const auto t = tail();
    if (observed_head != nullptr) {
      *observed_head = h;
    }
    out.reserve(static_cast<size_t>(h - t));
    for (uint64_t idx = t; idx < h; ++idx) {
      const auto slot = static_cast<uint32_t>(idx) & (capacity_ - 1);
      const auto kind = __atomic_load_n(&commands_host_[slot].kind,
                                        __ATOMIC_ACQUIRE);
      if (kind == 0) {
        break;
      }
      out.push_back(commands_host_[slot]);
    }
    return out;
  }

  std::vector<v2::EfaPostOp> drain_ready_to_efa_posts(bool coalesce,
                                                      bool ack_after_drain) {
    v2::RecordingEfaPostSink recorder;
    uint64_t observed_head = 0;
    const auto commands = poll_ready(&observed_head);
    if (coalesce) {
      v2::CoalescingEfaPostSink sink(&recorder);
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, sink);
      sink.flush();
    } else {
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, recorder);
    }
    if (ack_after_drain) {
      ack_ready_until(observed_head);
    }
    return recorder.ops;
  }

  void ack_ready_until(uint64_t observed_head) {
    const auto h = observed_head;
    auto t = tail();
    while (t < h) {
      const auto slot = static_cast<uint32_t>(t) & (capacity_ - 1);
      const auto kind = __atomic_load_n(&commands_host_[slot].kind,
                                        __ATOMIC_ACQUIRE);
      if (kind == 0) {
        break;
      }
      __atomic_store_n(&commands_host_[slot].kind, uint8_t{0},
                       __ATOMIC_RELEASE);
      ++t;
    }
    __atomic_store_n(tail_host_, t, __ATOMIC_RELEASE);
  }

  void ack_ready() { ack_ready_until(head()); }

 private:
  uint32_t capacity_ = 0;
  size_t commands_bytes_ = 0;
  v2::V2TransferCmd* commands_host_ = nullptr;
  v2::V2TransferCmd* commands_device_ = nullptr;
  uint64_t* head_host_ = nullptr;
  uint64_t* head_device_ = nullptr;
  uint64_t* tail_host_ = nullptr;
  uint64_t* tail_device_ = nullptr;
};

#if UCCL_V2_EFA_HAS_VERBS

class V2EfaConnectionHandle {
 public:
  V2EfaConnectionHandle(std::uintptr_t local_addr, uint64_t bytes,
                        uint32_t world_size, uint32_t rank,
                        uint32_t num_lanes = 1, int device_index = -1,
                        uint64_t signal_capacity = 65536)
      : local_addr_(local_addr),
        bytes_(bytes),
        world_size_(world_size),
        rank_(rank),
        num_lanes_(std::max<uint32_t>(num_lanes, 1)),
        signal_values_(static_cast<size_t>(std::max<uint64_t>(
            signal_capacity, uint64_t{1}))) {
#ifndef EFA
    throw std::runtime_error(
        "V2EfaConnection requires an EFA build; rebuild uccl-ep with EFA_HOME");
#else
    if (local_addr_ == 0 || bytes_ == 0) {
      throw std::invalid_argument("V2 EFA local window is empty");
    }
    if (world_size_ == 0 || rank_ >= world_size_) {
      throw std::invalid_argument("invalid V2 EFA rank/world_size");
    }

    open_device(device_index);
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr) {
      throw std::runtime_error("ibv_alloc_pd failed for V2 EFA");
    }
    cq_ = ibv_create_cq(context_, kMaxOutstandingSends * num_lanes_, nullptr,
                        nullptr, 0);
    if (cq_ == nullptr) {
      throw std::runtime_error("ibv_create_cq failed for V2 EFA");
    }

    constexpr int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
    mr_ = ibv_reg_mr(pd_, reinterpret_cast<void*>(local_addr_), bytes_, access);
    if (mr_ == nullptr) {
      throw std::runtime_error("ibv_reg_mr failed for V2 EFA local window");
    }
    signal_mr_ =
        ibv_reg_mr(pd_, signal_values_.data(),
                   signal_values_.size() * sizeof(uint32_t),
                   IBV_ACCESS_LOCAL_WRITE);
    if (signal_mr_ == nullptr) {
      throw std::runtime_error("ibv_reg_mr failed for V2 EFA signal scratch");
    }

    qps_.reserve(num_lanes_);
    for (uint32_t i = 0; i < num_lanes_; ++i) {
      qps_.push_back(create_srd_qp());
    }

    if (ibv_query_gid(context_, 1, 0, &gid_) != 0) {
      throw std::runtime_error("ibv_query_gid failed for V2 EFA");
    }
#endif
  }

  V2EfaConnectionHandle(const V2EfaConnectionHandle&) = delete;
  V2EfaConnectionHandle& operator=(const V2EfaConnectionHandle&) = delete;

  ~V2EfaConnectionHandle() { destroy(); }

  nb::dict local_info() const {
    nb::dict out;
    out["rank"] = rank_;
    out["addr"] = local_addr_;
    out["bytes"] = bytes_;
    out["rkey"] = mr_ == nullptr ? uint32_t{0} : mr_->rkey;
    out["lkey"] = mr_ == nullptr ? uint32_t{0} : mr_->lkey;
    out["device_name"] = device_name_;
    std::vector<uint32_t> qpns;
    qpns.reserve(qps_.size());
    for (auto* qp : qps_) {
      qpns.push_back(qp == nullptr ? uint32_t{0} : qp->qp_num);
    }
    out["qpns"] = qpns;
    std::vector<uint8_t> gid(16);
    std::memcpy(gid.data(), gid_.raw, gid.size());
    out["gid"] = gid;
    return out;
  }

  void connect(const nb::sequence& all_infos) {
    if (static_cast<uint32_t>(nb::len(all_infos)) != world_size_) {
      throw std::invalid_argument("V2 EFA endpoint info count != world_size");
    }
    endpoint_table_ =
        std::make_unique<v2::V2VerbsEndpointTable>(world_size_, num_lanes_);
    ahs_.clear();
    ahs_.reserve(world_size_);

    for (uint32_t peer = 0; peer < world_size_; ++peer) {
      nb::dict info = nb::cast<nb::dict>(all_infos[peer]);
      const auto remote_addr =
          static_cast<uint64_t>(nb::cast<std::uintptr_t>(info["addr"]));
      const auto remote_bytes = nb::cast<uint64_t>(info["bytes"]);
      const auto remote_rkey = nb::cast<uint32_t>(info["rkey"]);
      const auto qpns = nb::cast<std::vector<uint32_t>>(info["qpns"]);
      const auto gid = nb::cast<std::vector<uint8_t>>(info["gid"]);
      if (remote_addr == 0 || remote_bytes == 0 || remote_rkey == 0 ||
          qpns.empty() || gid.size() != 16) {
        throw std::invalid_argument("invalid V2 EFA remote endpoint info");
      }
      auto* ah = create_ah(gid);
      ahs_.push_back(ah);
      for (uint32_t lane = 0; lane < num_lanes_; ++lane) {
        v2::V2VerbsEndpoint endpoint;
        endpoint.rank = peer;
        endpoint.lane = lane;
        endpoint.qp = qps_.at(lane % qps_.size());
        endpoint.ah = ah;
        endpoint.dst_qpn = qpns.at(lane % qpns.size());
        endpoint.qkey = v2::kDefaultEfaQKey;
        endpoint.remote_base = remote_addr;
        endpoint.remote_bytes = remote_bytes;
        endpoint.remote_rkey = remote_rkey;
        endpoint_table_->set(endpoint);
      }
    }

    v2::V2VerbsLocalWindow local_window;
    local_window.base = local_addr_;
    local_window.bytes = bytes_;
    local_window.lkey = mr_->lkey;

    v2::V2VerbsSignalScratch signal_scratch;
    signal_scratch.values = signal_values_.data();
    signal_scratch.capacity = signal_values_.size();
    signal_scratch.lkey = signal_mr_->lkey;

    sink_ = std::make_unique<v2::V2EfaVerbsPostSink>(
        local_window, signal_scratch, endpoint_table_.get());
  }

  bool is_connected() const { return sink_ != nullptr; }

  nb::dict drain_queue(MappedD2HQueueHandle& queue, bool coalesce = true,
                       bool ack_after_drain = true) {
    ensure_connected();
    const auto before = sink_->stats();
    uint64_t observed_head = 0;
    const auto commands = queue.poll_ready(&observed_head);
    if (coalesce) {
      v2::CoalescingEfaPostSink coalesced(sink_.get());
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, coalesced);
      coalesced.flush();
    } else {
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, *sink_);
    }
    if (ack_after_drain) {
      queue.ack_ready_until(observed_head);
    }
    const auto after = sink_->stats();
    outstanding_signaled_posts_ +=
        (after.posted_writes - before.posted_writes) +
        (after.posted_signals - before.posted_signals);
    nb::dict out;
    out["drained_commands"] = commands.size();
    out["posted_writes"] = after.posted_writes - before.posted_writes;
    out["posted_signals"] = after.posted_signals - before.posted_signals;
    out["posted_bytes"] = after.posted_bytes - before.posted_bytes;
    out["head"] = queue.head();
    out["tail"] = queue.tail();
    return out;
  }

  nb::dict post_op(const nb::dict& op_dict) {
    ensure_connected();
    v2::EfaPostOp op;
    op.kind = static_cast<v2::EfaPostOpKind>(
        nb::cast<uint32_t>(op_dict["kind"]));
    op.target_rank = nb::cast<uint32_t>(op_dict["target_rank"]);
    op.target_lane = nb::cast<uint32_t>(op_dict["target_lane"]);
    op.bytes = nb::cast<uint32_t>(op_dict["bytes"]);
    if (op_dict.contains("signal_value")) {
      op.signal_value = nb::cast<uint32_t>(op_dict["signal_value"]);
    }
    if (op_dict.contains("local_offset")) {
      op.local_offset = nb::cast<uint64_t>(op_dict["local_offset"]);
    }
    op.remote_offset = nb::cast<uint64_t>(op_dict["remote_offset"]);
    const auto before = sink_->stats();
    sink_->post(op);
    const auto after = sink_->stats();
    outstanding_signaled_posts_ +=
        (after.posted_writes - before.posted_writes) +
        (after.posted_signals - before.posted_signals);
    nb::dict out;
    out["posted_writes"] = after.posted_writes - before.posted_writes;
    out["posted_signals"] = after.posted_signals - before.posted_signals;
    out["posted_bytes"] = after.posted_bytes - before.posted_bytes;
    return out;
  }

  uint32_t poll_completions(uint32_t max_entries = 64) {
    if (cq_ == nullptr || max_entries == 0) {
      return 0;
    }
    std::vector<ibv_wc> wc(max_entries);
    uint32_t total = 0;
    while (total < max_entries) {
      const auto room = static_cast<int>(max_entries - total);
      const int ne = ibv_poll_cq(cq_, room, wc.data() + total);
      if (ne < 0) {
        throw std::runtime_error("ibv_poll_cq failed for V2 EFA");
      }
      if (ne == 0) {
        break;
      }
      for (int i = 0; i < ne; ++i) {
        if (wc[total + i].status != IBV_WC_SUCCESS) {
          throw std::runtime_error("V2 EFA completion status is not success");
        }
      }
      total += static_cast<uint32_t>(ne);
    }
    if (total >= outstanding_signaled_posts_) {
      outstanding_signaled_posts_ = 0;
    } else {
      outstanding_signaled_posts_ -= total;
    }
    if (sink_ != nullptr && total != 0 && outstanding_signaled_posts_ == 0) {
      sink_->reset_signal_scratch();
    }
    return total;
  }

  nb::dict stats() const {
    nb::dict out;
    if (sink_ == nullptr) {
      out["posted_writes"] = uint64_t{0};
      out["posted_signals"] = uint64_t{0};
      out["posted_bytes"] = uint64_t{0};
      return out;
    }
    const auto& stats = sink_->stats();
    out["posted_writes"] = stats.posted_writes;
    out["posted_signals"] = stats.posted_signals;
    out["posted_bytes"] = stats.posted_bytes;
    return out;
  }

 private:
  void ensure_connected() const {
    if (sink_ == nullptr) {
      throw std::runtime_error("V2 EFA connection is not connected");
    }
  }

  void open_device(int requested_index) {
    int num_devices = 0;
    auto** devices = ibv_get_device_list(&num_devices);
    if (devices == nullptr || num_devices == 0) {
      throw std::runtime_error("ibv_get_device_list found no RDMA devices");
    }

    int selected = requested_index;
    if (selected < 0) {
      if (const char* env = std::getenv("UCCL_V2_EFA_DEVICE_INDEX")) {
        selected = std::atoi(env);
      }
    }
    if (selected < 0) {
      selected = 0;
      for (int i = 0; i < num_devices; ++i) {
        const char* name = ibv_get_device_name(devices[i]);
        if (name != nullptr && std::string(name).find("efa") != std::string::npos) {
          selected = i;
          break;
        }
      }
    }
    if (selected < 0 || selected >= num_devices) {
      ibv_free_device_list(devices);
      throw std::out_of_range("V2 EFA device index out of range");
    }
    device_name_ = ibv_get_device_name(devices[selected]);
    context_ = ibv_open_device(devices[selected]);
    ibv_free_device_list(devices);
    if (context_ == nullptr) {
      throw std::runtime_error("ibv_open_device failed for V2 EFA");
    }
  }

  ibv_qp* create_srd_qp() {
#ifndef EFA
    throw std::runtime_error("SRD QP requires EFA");
#else
    ibv_qp_init_attr_ex qp_attr{};
    efadv_qp_init_attr efa_attr{};
    qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
    qp_attr.cap.max_send_wr = kMaxOutstandingSends;
    qp_attr.cap.max_recv_wr = kMaxOutstandingSends;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.cap.max_inline_data = 0;
    qp_attr.pd = pd_;
    qp_attr.qp_context = context_;
    qp_attr.sq_sig_all = 1;
    qp_attr.send_cq = cq_;
    qp_attr.recv_cq = cq_;
    qp_attr.qp_type = IBV_QPT_DRIVER;

    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    efa_attr.flags = EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

    auto* qp = efadv_create_qp_ex(context_, &qp_attr, &efa_attr,
                                  sizeof(efadv_qp_init_attr));
    if (qp == nullptr) {
      throw std::runtime_error("efadv_create_qp_ex failed for V2 EFA");
    }

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qkey = v2::kDefaultEfaQKey;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_QKEY) != 0) {
      throw std::runtime_error("ibv_modify_qp INIT failed for V2 EFA");
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE) != 0) {
      throw std::runtime_error("ibv_modify_qp RTR failed for V2 EFA");
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
      throw std::runtime_error("ibv_modify_qp RTS failed for V2 EFA");
    }
    return qp;
#endif
  }

  ibv_ah* create_ah(const std::vector<uint8_t>& remote_gid) {
    ibv_ah_attr attr{};
    attr.is_global = 1;
    attr.port_num = 1;
    attr.grh.sgid_index = 0;
    std::memcpy(&attr.grh.dgid, remote_gid.data(), 16);
    attr.grh.hop_limit = 255;
    auto* ah = ibv_create_ah(pd_, &attr);
    if (ah == nullptr) {
      throw std::runtime_error("ibv_create_ah failed for V2 EFA");
    }
    return ah;
  }

  void destroy() {
    sink_.reset();
    endpoint_table_.reset();
    for (auto* ah : ahs_) {
      if (ah != nullptr) {
        ibv_destroy_ah(ah);
      }
    }
    ahs_.clear();
    for (auto* qp : qps_) {
      if (qp != nullptr) {
        ibv_destroy_qp(qp);
      }
    }
    qps_.clear();
    if (signal_mr_ != nullptr) {
      ibv_dereg_mr(signal_mr_);
      signal_mr_ = nullptr;
    }
    if (mr_ != nullptr) {
      ibv_dereg_mr(mr_);
      mr_ = nullptr;
    }
    if (cq_ != nullptr) {
      ibv_destroy_cq(cq_);
      cq_ = nullptr;
    }
    if (pd_ != nullptr) {
      ibv_dealloc_pd(pd_);
      pd_ = nullptr;
    }
    if (context_ != nullptr) {
      ibv_close_device(context_);
      context_ = nullptr;
    }
  }

  std::uintptr_t local_addr_ = 0;
  uint64_t bytes_ = 0;
  uint32_t world_size_ = 0;
  uint32_t rank_ = 0;
  uint32_t num_lanes_ = 1;
  std::string device_name_;
  ibv_context* context_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_mr* mr_ = nullptr;
  ibv_mr* signal_mr_ = nullptr;
  ibv_gid gid_{};
  std::vector<ibv_qp*> qps_;
  std::vector<ibv_ah*> ahs_;
  std::vector<uint32_t> signal_values_;
  uint64_t outstanding_signaled_posts_ = 0;
  std::unique_ptr<v2::V2VerbsEndpointTable> endpoint_table_;
  std::unique_ptr<v2::V2EfaVerbsPostSink> sink_;
};

#endif  // UCCL_V2_EFA_HAS_VERBS

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

  nb::class_<MappedD2HQueueHandle>(m, "V2MappedD2HQueue")
      .def(nb::init<uint32_t>(), nb::arg("capacity") = v2::kV2TransferD2HQueueSize)
      .def("capacity", &MappedD2HQueueHandle::capacity)
      .def("commands_ptr", &MappedD2HQueueHandle::commands_ptr)
      .def("head_ptr", &MappedD2HQueueHandle::head_ptr)
      .def("tail_ptr", &MappedD2HQueueHandle::tail_ptr)
      .def("head", &MappedD2HQueueHandle::head)
      .def("tail", &MappedD2HQueueHandle::tail)
      .def("reset", &MappedD2HQueueHandle::reset)
      .def("ack_ready", &MappedD2HQueueHandle::ack_ready)
      .def("poll_ready", [](const MappedD2HQueueHandle& queue) {
        nb::list out;
        for (const auto& cmd : queue.poll_ready()) {
          out.append(transfer_cmd_to_dict(cmd));
        }
        return out;
      })
      .def("drain_ready_to_efa_posts",
           [](MappedD2HQueueHandle& queue, bool coalesce,
              bool ack_after_drain) {
             nb::list out;
             for (const auto& op :
                  queue.drain_ready_to_efa_posts(coalesce, ack_after_drain)) {
               out.append(efa_post_op_to_dict(op));
             }
             return out;
           },
           nb::arg("coalesce") = true,
           nb::arg("ack_after_drain") = true)
      .def("drain_ready",
           [](MappedD2HQueueHandle& queue) {
             nb::list out;
             for (const auto& op :
                  queue.drain_ready_to_efa_posts(true, true)) {
               out.append(efa_post_op_to_dict(op));
             }
             return out;
      });

#if UCCL_V2_EFA_HAS_VERBS
  nb::class_<V2EfaConnectionHandle>(m, "V2EfaConnection")
      .def(nb::init<std::uintptr_t, uint64_t, uint32_t, uint32_t, uint32_t,
                    int, uint64_t>(),
           nb::arg("local_addr"), nb::arg("bytes"), nb::arg("world_size"),
           nb::arg("rank"), nb::arg("num_lanes") = 1,
           nb::arg("device_index") = -1,
           nb::arg("signal_capacity") = 65536)
      .def("local_info", &V2EfaConnectionHandle::local_info)
      .def("connect", &V2EfaConnectionHandle::connect)
      .def("is_connected", &V2EfaConnectionHandle::is_connected)
      .def("drain_queue", &V2EfaConnectionHandle::drain_queue,
           nb::arg("queue"), nb::arg("coalesce") = true,
           nb::arg("ack_after_drain") = true)
      .def("post_op", &V2EfaConnectionHandle::post_op)
      .def("poll_completions", &V2EfaConnectionHandle::poll_completions,
           nb::arg("max_entries") = 64)
      .def("stats", &V2EfaConnectionHandle::stats);
#endif

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
             if (static_cast<size_t>(nb::len(topk_idx_flat)) !=
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
             if (static_cast<size_t>(nb::len(topk_idx_flat)) !=
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
             if (static_cast<size_t>(nb::len(topk_idx_flat)) !=
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
      .def("build_dispatch_enqueue_d2h_jit_plan",
           [](const v2::V2EfaRuntime& self,
              const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(
                 self.build_dispatch_enqueue_d2h_jit_plan(uccl_include_path));
           },
           nb::arg("uccl_include_path") = "")
      .def("compile_dispatch_enqueue_d2h_jit",
           [](const v2::V2EfaRuntime& self,
              const std::string& uccl_include_path) {
             const auto plan =
                 self.build_dispatch_enqueue_d2h_jit_plan(uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("uccl_include_path") = "")
      .def("compile_dispatch_descriptor_enqueue_d2h_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
              bool cached_mode, bool deterministic, bool do_cpu_sync,
              int smem_bytes, const std::string& uccl_include_path) {
             const auto plan =
                 self.build_dispatch_descriptor_enqueue_d2h_jit_plan(
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
      .def("build_dispatch_direct_enqueue_d2h_jit_plan",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
              bool cached_mode, bool deterministic, bool do_cpu_sync,
              int smem_bytes, const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(
                 self.build_dispatch_direct_enqueue_d2h_jit_plan(
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
      .def("compile_dispatch_direct_enqueue_d2h_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
              bool cached_mode, bool deterministic, bool do_cpu_sync,
              int smem_bytes, const std::string& uccl_include_path) {
             const auto plan =
                 self.build_dispatch_direct_enqueue_d2h_jit_plan(
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
      .def("build_combine_enqueue_d2h_jit_plan",
           [](const v2::V2EfaRuntime& self,
              const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(
                 self.build_combine_enqueue_d2h_jit_plan(uccl_include_path));
           },
           nb::arg("uccl_include_path") = "")
      .def("compile_combine_enqueue_d2h_jit",
           [](const v2::V2EfaRuntime& self,
              const std::string& uccl_include_path) {
             const auto plan =
                 self.build_combine_enqueue_d2h_jit_plan(uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("uccl_include_path") = "")
      .def("compile_combine_descriptor_enqueue_d2h_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels, int payload_bytes, bool use_expanded_layout,
              bool allow_multiple_reduction, int smem_bytes,
              const std::string& uccl_include_path) {
             const auto plan =
                 self.build_combine_descriptor_enqueue_d2h_jit_plan(
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
      .def("compile_combine_forward_metadata_enqueue_d2h_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels, int payload_bytes, bool use_expanded_layout,
              bool allow_multiple_reduction, int smem_bytes,
              const std::string& uccl_include_path) {
             const auto plan =
                 self.build_combine_forward_metadata_enqueue_d2h_jit_plan(
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
           nb::arg("smem_bytes") = 0,
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
      .def("launch_dispatch_enqueue_d2h",
           [](const v2::V2EfaRuntime& self, std::uintptr_t segments_ptr,
              std::uintptr_t batches_ptr, int num_batches,
              std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
              std::uintptr_t tail_ptr, int queue_capacity,
              std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base, std::uint32_t src_token_stride,
              std::uint32_t expanded_slot_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t signal_stride,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::DispatchTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.src_token_stride = src_token_stride;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.signal_stride = signal_stride;
             self.launch_dispatch_enqueue_d2h(
                 segments_ptr, batches_ptr, num_batches, commands_ptr,
                 head_ptr, tail_ptr, queue_capacity, layout,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("segments_ptr"),
           nb::arg("batches_ptr"),
           nb::arg("num_batches"),
           nb::arg("commands_ptr") = 0,
           nb::arg("head_ptr") = 0,
           nb::arg("tail_ptr") = 0,
           nb::arg("queue_capacity") = 0,
           nb::arg("local_payload_base") = 0,
           nb::arg("remote_payload_base") = 0,
           nb::arg("remote_signal_base") = 0,
           nb::arg("src_token_stride") = 0,
           nb::arg("expanded_slot_stride") = 0,
           nb::arg("batch_payload_stride") = 0,
           nb::arg("signal_stride") = sizeof(std::uint32_t),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_descriptor_enqueue_d2h",
           [](const v2::V2EfaRuntime& self, std::uintptr_t topk_idx_ptr,
              std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
              std::uintptr_t counters_ptr, int num_tokens,
              int num_max_tokens_per_rank, int num_channels_per_sm,
              int scale_bytes, bool has_topk_weight, bool cached_mode,
              bool deterministic, bool do_cpu_sync, int smem_bytes,
              std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
              std::uintptr_t tail_ptr, int queue_capacity,
              std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base, std::uint32_t src_token_stride,
              std::uint32_t expanded_slot_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t signal_stride,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::DispatchTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.src_token_stride = src_token_stride;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.signal_stride = signal_stride;
             self.launch_dispatch_descriptor_enqueue_d2h(
                 topk_idx_ptr, segments_ptr, batches_ptr, counters_ptr,
                 num_tokens, num_max_tokens_per_rank, num_channels_per_sm,
                 scale_bytes, has_topk_weight, cached_mode, deterministic,
                 do_cpu_sync, smem_bytes, commands_ptr, head_ptr, tail_ptr,
                 queue_capacity, layout, uccl_include_path, cuda_stream_ptr);
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
           nb::arg("commands_ptr"),
           nb::arg("head_ptr"),
           nb::arg("tail_ptr"),
           nb::arg("queue_capacity"),
           nb::arg("local_payload_base"),
           nb::arg("remote_payload_base"),
           nb::arg("remote_signal_base"),
           nb::arg("src_token_stride"),
           nb::arg("expanded_slot_stride"),
           nb::arg("batch_payload_stride"),
           nb::arg("signal_stride") = sizeof(std::uint32_t),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_direct_enqueue_d2h",
           [](const v2::V2EfaRuntime& self, std::uintptr_t topk_idx_ptr,
              int num_tokens, int num_max_tokens_per_rank,
              int num_channels_per_sm, int scale_bytes, bool has_topk_weight,
              bool cached_mode, bool deterministic, bool do_cpu_sync,
              int smem_bytes, std::uintptr_t commands_ptr,
              std::uintptr_t head_ptr, std::uintptr_t tail_ptr,
              int queue_capacity, std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base, std::uint32_t src_token_stride,
              std::uint32_t expanded_slot_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t signal_stride,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::DispatchTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.src_token_stride = src_token_stride;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.signal_stride = signal_stride;
             self.launch_dispatch_direct_enqueue_d2h(
                 topk_idx_ptr, num_tokens, num_max_tokens_per_rank,
                 num_channels_per_sm, scale_bytes, has_topk_weight,
                 cached_mode, deterministic, do_cpu_sync, smem_bytes,
                 commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("topk_idx_ptr"),
           nb::arg("num_tokens"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("scale_bytes") = 0,
           nb::arg("has_topk_weight") = true,
           nb::arg("cached_mode") = false,
           nb::arg("deterministic") = false,
           nb::arg("do_cpu_sync") = false,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("commands_ptr"),
           nb::arg("head_ptr"),
           nb::arg("tail_ptr"),
           nb::arg("queue_capacity"),
           nb::arg("local_payload_base"),
           nb::arg("remote_payload_base"),
           nb::arg("remote_signal_base"),
           nb::arg("src_token_stride"),
           nb::arg("expanded_slot_stride"),
           nb::arg("batch_payload_stride"),
           nb::arg("signal_stride") = sizeof(std::uint32_t),
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
      .def("launch_combine_enqueue_d2h",
           [](const v2::V2EfaRuntime& self, std::uintptr_t segments_ptr,
              std::uintptr_t batches_ptr, int num_batches,
              std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
              std::uintptr_t tail_ptr, int queue_capacity,
              std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base,
              std::uint32_t expanded_slot_stride,
              std::uint32_t reduced_token_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t signal_stride,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::CombineTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.reduced_token_stride = reduced_token_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.signal_stride = signal_stride;
             self.launch_combine_enqueue_d2h(
                 segments_ptr, batches_ptr, num_batches, commands_ptr,
                 head_ptr, tail_ptr, queue_capacity, layout,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("segments_ptr"),
           nb::arg("batches_ptr"),
           nb::arg("num_batches"),
           nb::arg("commands_ptr"),
           nb::arg("head_ptr"),
           nb::arg("tail_ptr"),
           nb::arg("queue_capacity"),
           nb::arg("local_payload_base"),
           nb::arg("remote_payload_base"),
           nb::arg("remote_signal_base"),
           nb::arg("expanded_slot_stride"),
           nb::arg("reduced_token_stride"),
           nb::arg("batch_payload_stride"),
           nb::arg("signal_stride") = sizeof(std::uint32_t),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_combine_descriptor_enqueue_d2h",
           [](const v2::V2EfaRuntime& self,
              std::uintptr_t dispatch_segments_ptr,
              std::uintptr_t dispatch_batches_ptr, int num_dispatch_batches,
              std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
              std::uintptr_t counters_ptr, int num_max_tokens_per_rank,
              int num_channels, int payload_bytes, bool use_expanded_layout,
              bool allow_multiple_reduction, int smem_bytes,
              std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
              std::uintptr_t tail_ptr, int queue_capacity,
              std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base,
              std::uint32_t expanded_slot_stride,
              std::uint32_t reduced_token_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t signal_stride,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::CombineTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.reduced_token_stride = reduced_token_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.signal_stride = signal_stride;
             self.launch_combine_descriptor_enqueue_d2h(
                 dispatch_segments_ptr, dispatch_batches_ptr,
                 num_dispatch_batches, segments_ptr, batches_ptr, counters_ptr,
                 num_max_tokens_per_rank, num_channels, payload_bytes,
                 use_expanded_layout, allow_multiple_reduction, smem_bytes,
                 commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
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
           nb::arg("commands_ptr") = 0,
           nb::arg("head_ptr") = 0,
           nb::arg("tail_ptr") = 0,
           nb::arg("queue_capacity") = 0,
           nb::arg("local_payload_base") = 0,
           nb::arg("remote_payload_base") = 0,
           nb::arg("remote_signal_base") = 0,
           nb::arg("expanded_slot_stride") = 0,
           nb::arg("reduced_token_stride") = 0,
           nb::arg("batch_payload_stride") = 0,
           nb::arg("signal_stride") = sizeof(std::uint32_t),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_combine_forward_metadata_enqueue_d2h",
           [](const v2::V2EfaRuntime& self,
              std::uintptr_t forward_metadata_ptr,
              std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
              std::uintptr_t counters_ptr, int num_forward_rows,
              int num_max_tokens_per_rank, int num_channels, int payload_bytes,
              bool use_expanded_layout, bool allow_multiple_reduction,
              int smem_bytes, std::uintptr_t commands_ptr,
              std::uintptr_t head_ptr, std::uintptr_t tail_ptr,
              int queue_capacity, std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base,
              std::uint32_t expanded_slot_stride,
              std::uint32_t reduced_token_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t signal_stride,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::CombineTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.reduced_token_stride = reduced_token_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.signal_stride = signal_stride;
             self.launch_combine_forward_metadata_enqueue_d2h(
                 forward_metadata_ptr, segments_ptr, batches_ptr, counters_ptr,
                 num_forward_rows, num_max_tokens_per_rank, num_channels,
                 payload_bytes, use_expanded_layout, allow_multiple_reduction,
                 smem_bytes, commands_ptr, head_ptr, tail_ptr, queue_capacity,
                 layout, uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("forward_metadata_ptr"),
           nb::arg("segments_ptr"),
           nb::arg("batches_ptr"),
           nb::arg("counters_ptr"),
           nb::arg("num_forward_rows"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels") = 1,
           nb::arg("payload_bytes") = 0,
           nb::arg("use_expanded_layout") = true,
           nb::arg("allow_multiple_reduction") = true,
           nb::arg("smem_bytes") = 0,
           nb::arg("commands_ptr") = 0,
           nb::arg("head_ptr") = 0,
           nb::arg("tail_ptr") = 0,
           nb::arg("queue_capacity") = 0,
           nb::arg("local_payload_base") = 0,
           nb::arg("remote_payload_base") = 0,
           nb::arg("remote_signal_base") = 0,
           nb::arg("expanded_slot_stride") = 0,
           nb::arg("reduced_token_stride") = 0,
           nb::arg("batch_payload_stride") = 0,
           nb::arg("signal_stride") = sizeof(std::uint32_t),
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
  m.def("v2_has_verbs_sink", []() {
#if UCCL_V2_EFA_HAS_VERBS
    return true;
#else
    return false;
#endif
  });
}
