#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ep_util.hpp"
#include "v2_efa/efa_adapter.hpp"
#include "v2_efa/runtime.hpp"
#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_d2h_queue.cuh"
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

nb::dict transfer_cmd_to_dict(const v2::V2TransferCmd& cmd) {
  nb::dict out;
  out["kind"] = cmd.kind;
  out["target_rank"] = cmd.target_rank;
  out["target_lane"] = cmd.target_lane;
  out["flags"] = cmd.flags;
  out["region"] = v2::v2_transfer_uses_workspace(cmd)
                      ? static_cast<uint32_t>(v2::EfaMemoryRegion::kWorkspace)
                      : static_cast<uint32_t>(v2::EfaMemoryRegion::kBuffer);
  out["remote_offset"] = v2::v2_transfer_remote_offset(cmd);
  if (v2::is_v2_transfer_signal(cmd)) {
    out["bytes"] =
        (cmd.flags & static_cast<uint8_t>(v2::V2TransferCmdFlags::kSignal64))
            ? sizeof(uint64_t)
            : sizeof(uint32_t);
    out["signal_value"] = v2::v2_transfer_signal_value(cmd);
  } else {
    out["bytes"] = cmd.bytes;
    out["local_offset"] = v2::v2_transfer_local_offset(cmd);
  }
  return out;
}

nb::dict efa_post_op_to_dict(const v2::EfaPostOp& op) {
  nb::dict out;
  out["kind"] = static_cast<uint32_t>(op.kind);
  out["region"] = static_cast<uint32_t>(op.region);
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

struct V2ConnectionDrainStats {
  size_t drained_commands = 0;
  uint64_t posted_writes = 0;
  uint64_t posted_signals = 0;
  uint64_t posted_completions = 0;
  uint64_t posted_bytes = 0;
};

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
      uint64_t* observed_ready_end = nullptr) const {
    std::vector<v2::V2TransferCmd> out;
    const auto h = head();
    const auto t = tail();
    out.reserve(static_cast<size_t>(h - t));
    uint64_t ready_end = t;
    for (uint64_t idx = t; idx < h; ++idx) {
      const auto slot = static_cast<uint32_t>(idx) & (capacity_ - 1);
      const auto header = __atomic_load_n(
          reinterpret_cast<const uint32_t*>(&commands_host_[slot]),
          __ATOMIC_ACQUIRE);
      const auto kind = v2::v2_transfer_kind_from_header(header);
      if (kind == 0) {
        break;
      }
      out.push_back(commands_host_[slot]);
      ready_end = idx + 1;
    }
    if (observed_ready_end != nullptr) {
      *observed_ready_end = ready_end;
    }
    return out;
  }

  std::vector<v2::EfaPostOp> drain_ready_to_efa_posts(bool coalesce,
                                                      bool ack_after_drain) {
    v2::RecordingEfaPostSink recorder;
    uint64_t observed_ready_end = 0;
    const auto commands = poll_ready(&observed_ready_end);
    if (coalesce) {
      v2::CoalescingEfaPostSink sink(&recorder);
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, sink);
      sink.flush();
    } else {
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, recorder);
    }
    if (ack_after_drain) {
      ack_ready_until(observed_ready_end);
    }
    return recorder.ops;
  }

  void ack_ready_until(uint64_t observed_ready_end) {
    const auto h = observed_ready_end;
    auto t = tail();
    while (t < h) {
      const auto slot = static_cast<uint32_t>(t) & (capacity_ - 1);
      const auto header = __atomic_load_n(
          reinterpret_cast<const uint32_t*>(&commands_host_[slot]),
          __ATOMIC_ACQUIRE);
      const auto kind = v2::v2_transfer_kind_from_header(header);
      if (kind == 0) {
        break;
      }
      __atomic_store_n(reinterpret_cast<uint32_t*>(&commands_host_[slot]),
                       uint32_t{0},
                       __ATOMIC_RELEASE);
      ++t;
    }
    __atomic_store_n(tail_host_, t, __ATOMIC_RELEASE);
  }

  void ack_ready() {
    uint64_t observed_ready_end = 0;
    (void)poll_ready(&observed_ready_end);
    ack_ready_until(observed_ready_end);
  }

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
                        uint64_t signal_capacity = 65536,
                        std::uintptr_t workspace_addr = 0,
                        uint64_t workspace_bytes = 0)
      : local_addr_(local_addr),
        bytes_(bytes),
        workspace_addr_(workspace_addr),
        workspace_bytes_(workspace_bytes),
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

    lane_contexts_.reserve(num_lanes_);
    lane_pds_.reserve(num_lanes_);
    lane_cqs_.reserve(num_lanes_);
    lane_mrs_.reserve(num_lanes_);
    lane_workspace_mrs_.reserve(num_lanes_);
    lane_signal_mrs_.reserve(num_lanes_);
    lane_gids_.reserve(num_lanes_);
    lane_device_names_.reserve(num_lanes_);
    qps_.reserve(num_lanes_);
    const int base_device = resolve_base_device_index(device_index);
    for (uint32_t lane = 0; lane < num_lanes_; ++lane) {
      open_lane(base_device + static_cast<int>(lane));
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
    out["rkey"] = lane_mrs_.empty() || lane_mrs_[0] == nullptr
                      ? uint32_t{0}
                      : lane_mrs_[0]->rkey;
    out["lkey"] = lane_mrs_.empty() || lane_mrs_[0] == nullptr
                      ? uint32_t{0}
                      : lane_mrs_[0]->lkey;
    out["workspace_addr"] = workspace_addr_;
    out["workspace_bytes"] = workspace_bytes_;
    out["workspace_rkey"] =
        lane_workspace_mrs_.empty() || lane_workspace_mrs_[0] == nullptr
            ? uint32_t{0}
            : lane_workspace_mrs_[0]->rkey;
    out["workspace_lkey"] =
        lane_workspace_mrs_.empty() || lane_workspace_mrs_[0] == nullptr
            ? uint32_t{0}
            : lane_workspace_mrs_[0]->lkey;
    out["device_name"] =
        lane_device_names_.empty() ? std::string{} : lane_device_names_[0];
    std::vector<uint32_t> qpns;
    qpns.reserve(qps_.size());
    for (auto* qp : qps_) {
      qpns.push_back(qp == nullptr ? uint32_t{0} : qp->qp_num);
    }
    out["qpns"] = qpns;
    std::vector<uint8_t> gid(16);
    if (!lane_gids_.empty()) {
      std::memcpy(gid.data(), lane_gids_[0].raw, gid.size());
    }
    out["gid"] = gid;
    nb::list lanes;
    for (uint32_t lane = 0; lane < num_lanes_; ++lane) {
      nb::dict lane_info;
      lane_info["lane"] = lane;
      lane_info["qpn"] = qps_.at(lane) == nullptr ? uint32_t{0}
                                                   : qps_.at(lane)->qp_num;
      lane_info["rkey"] = lane_mrs_.at(lane) == nullptr
                              ? uint32_t{0}
                              : lane_mrs_.at(lane)->rkey;
      lane_info["lkey"] = lane_mrs_.at(lane) == nullptr
                              ? uint32_t{0}
                              : lane_mrs_.at(lane)->lkey;
      lane_info["workspace_rkey"] =
          lane_workspace_mrs_.empty() || lane_workspace_mrs_.at(lane) == nullptr
              ? uint32_t{0}
              : lane_workspace_mrs_.at(lane)->rkey;
      lane_info["workspace_lkey"] =
          lane_workspace_mrs_.empty() || lane_workspace_mrs_.at(lane) == nullptr
              ? uint32_t{0}
              : lane_workspace_mrs_.at(lane)->lkey;
      lane_info["device_name"] = lane_device_names_.at(lane);
      std::vector<uint8_t> lane_gid(16);
      std::memcpy(lane_gid.data(), lane_gids_.at(lane).raw, lane_gid.size());
      lane_info["gid"] = lane_gid;
      lanes.append(lane_info);
    }
    out["lanes"] = lanes;
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
      const auto remote_workspace_addr =
          info.contains("workspace_addr")
              ? static_cast<uint64_t>(
                    nb::cast<std::uintptr_t>(info["workspace_addr"]))
              : uint64_t{0};
      const auto remote_workspace_bytes =
          info.contains("workspace_bytes")
              ? nb::cast<uint64_t>(info["workspace_bytes"])
              : uint64_t{0};
      const auto remote_workspace_rkey =
          info.contains("workspace_rkey")
              ? nb::cast<uint32_t>(info["workspace_rkey"])
              : uint32_t{0};
      const auto qpns = nb::cast<std::vector<uint32_t>>(info["qpns"]);
      const auto gid = nb::cast<std::vector<uint8_t>>(info["gid"]);
      if (remote_addr == 0 || remote_bytes == 0 || remote_rkey == 0 ||
          qpns.empty() || gid.size() != 16) {
        throw std::invalid_argument("invalid V2 EFA remote endpoint info");
      }
      nb::list lane_infos;
      const bool has_lane_infos = info.contains("lanes");
      if (has_lane_infos) {
        lane_infos = nb::cast<nb::list>(info["lanes"]);
      }
      for (uint32_t lane = 0; lane < num_lanes_; ++lane) {
        auto lane_gid = gid;
        auto lane_qpn = qpns.at(lane % qpns.size());
        auto lane_rkey = remote_rkey;
        auto lane_workspace_rkey = remote_workspace_rkey;
        if (has_lane_infos && lane < static_cast<uint32_t>(nb::len(lane_infos))) {
          nb::dict lane_info = nb::cast<nb::dict>(lane_infos[lane]);
          lane_qpn = nb::cast<uint32_t>(lane_info["qpn"]);
          lane_rkey = nb::cast<uint32_t>(lane_info["rkey"]);
          if (lane_info.contains("workspace_rkey")) {
            lane_workspace_rkey = nb::cast<uint32_t>(lane_info["workspace_rkey"]);
          }
          lane_gid = nb::cast<std::vector<uint8_t>>(lane_info["gid"]);
          if (lane_gid.size() != 16) {
            throw std::invalid_argument("invalid V2 EFA lane gid");
          }
        }
        auto* ah = create_ah(lane, lane_gid);
        ahs_.push_back(ah);
        v2::V2VerbsEndpoint endpoint;
        endpoint.rank = peer;
        endpoint.lane = lane;
        endpoint.qp = qps_.at(lane);
        endpoint.ah = ah;
        endpoint.dst_qpn = lane_qpn;
        endpoint.qkey = v2::kDefaultEfaQKey;
        endpoint.remote_base = remote_addr;
        endpoint.remote_bytes = remote_bytes;
        endpoint.remote_rkey = lane_rkey;
        endpoint.remote_workspace_base = remote_workspace_addr;
        endpoint.remote_workspace_bytes = remote_workspace_bytes;
        endpoint.remote_workspace_rkey = lane_workspace_rkey;
        endpoint_table_->set(endpoint);
      }
    }

    v2::V2VerbsLocalWindow local_window;
    local_window.base = local_addr_;
    local_window.bytes = bytes_;
    local_window.lkey = lane_mrs_.at(0)->lkey;
    local_window.lkeys_by_lane.reserve(lane_mrs_.size());
    for (auto* mr : lane_mrs_) {
      local_window.lkeys_by_lane.push_back(mr->lkey);
    }

    v2::V2VerbsLocalWindow workspace_window;
    workspace_window.base = workspace_addr_;
    workspace_window.bytes = workspace_bytes_;
    if (!lane_workspace_mrs_.empty() && lane_workspace_mrs_.at(0) != nullptr) {
      workspace_window.lkey = lane_workspace_mrs_.at(0)->lkey;
      workspace_window.lkeys_by_lane.reserve(lane_workspace_mrs_.size());
      for (auto* mr : lane_workspace_mrs_) {
        workspace_window.lkeys_by_lane.push_back(mr == nullptr ? 0 : mr->lkey);
      }
    }

    v2::V2VerbsSignalScratch signal_scratch;
    signal_scratch.values = signal_values_.data();
    signal_scratch.capacity = signal_values_.size();
    signal_scratch.lkey = lane_signal_mrs_.at(0)->lkey;
    signal_scratch.lkeys_by_lane.reserve(lane_signal_mrs_.size());
    for (auto* mr : lane_signal_mrs_) {
      signal_scratch.lkeys_by_lane.push_back(mr->lkey);
    }

    sink_ = std::make_unique<v2::V2EfaVerbsPostSink>(
        local_window, workspace_window, signal_scratch, endpoint_table_.get());
  }

  bool is_connected() const { return sink_ != nullptr; }

  nb::dict drain_queue(MappedD2HQueueHandle& queue, bool coalesce = true,
                       bool ack_after_drain = true) {
    ensure_connected();
    const auto stats = drain_queue_internal(queue, coalesce, ack_after_drain);
    nb::dict out;
    out["drained_commands"] = stats.drained_commands;
    out["posted_writes"] = stats.posted_writes;
    out["posted_signals"] = stats.posted_signals;
    out["posted_completions"] = stats.posted_completions;
    out["posted_bytes"] = stats.posted_bytes;
    out["head"] = queue.head();
    out["tail"] = queue.tail();
    return out;
  }

  void start_proxy(const nb::sequence& queues, bool coalesce = true,
                   bool ack_after_drain = true, uint32_t num_threads = 1) {
    ensure_connected();
    stop_proxy();
    proxy_queues_.clear();
    for (size_t i = 0; i < nb::len(queues); ++i) {
      proxy_queues_.push_back(&nb::cast<MappedD2HQueueHandle&>(queues[i]));
    }
    if (proxy_queues_.empty()) {
      throw std::invalid_argument("V2 EFA proxy needs at least one D2H queue");
    }
    proxy_drained_commands_.store(0, std::memory_order_release);
    proxy_posted_writes_.store(0, std::memory_order_release);
    proxy_posted_signals_.store(0, std::memory_order_release);
    proxy_posted_completions_.store(0, std::memory_order_release);
    proxy_posted_bytes_.store(0, std::memory_order_release);
    proxy_coalesce_ = coalesce;
    proxy_ack_after_drain_ = ack_after_drain;
    proxy_run_.store(true, std::memory_order_release);
    const auto num_queues = static_cast<uint32_t>(proxy_queues_.size());
    const auto threads =
        std::max<uint32_t>(1, std::min<uint32_t>(num_threads, num_queues));
    proxy_threads_.reserve(threads);
    for (uint32_t thread_idx = 0; thread_idx < threads; ++thread_idx) {
      proxy_threads_.emplace_back([this, thread_idx, threads]() {
        proxy_loop(thread_idx, threads);
      });
    }
  }

  void stop_proxy() {
    proxy_run_.store(false, std::memory_order_release);
    for (auto& thread : proxy_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    proxy_threads_.clear();
  }

  nb::dict proxy_stats() const {
    nb::dict out;
    out["drained_commands"] =
        proxy_drained_commands_.load(std::memory_order_acquire);
    out["posted_writes"] = proxy_posted_writes_.load(std::memory_order_acquire);
    out["posted_signals"] =
        proxy_posted_signals_.load(std::memory_order_acquire);
    out["posted_completions"] =
        proxy_posted_completions_.load(std::memory_order_acquire);
    out["posted_bytes"] = proxy_posted_bytes_.load(std::memory_order_acquire);
    out["running"] = proxy_run_.load(std::memory_order_acquire);
    out["num_threads"] = proxy_threads_.size();
    out["num_queues"] = proxy_queues_.size();
    return out;
  }

  nb::dict post_op(const nb::dict& op_dict) {
    ensure_connected();
    v2::EfaPostOp op;
    op.kind = static_cast<v2::EfaPostOpKind>(
        nb::cast<uint32_t>(op_dict["kind"]));
    if (op_dict.contains("region")) {
      op.region = static_cast<v2::EfaMemoryRegion>(
          nb::cast<uint32_t>(op_dict["region"]));
    }
    op.target_rank = nb::cast<uint32_t>(op_dict["target_rank"]);
    op.target_lane = nb::cast<uint32_t>(op_dict["target_lane"]);
    op.bytes = nb::cast<uint32_t>(op_dict["bytes"]);
    if (op_dict.contains("signal_value")) {
      op.signal_value = nb::cast<uint64_t>(op_dict["signal_value"]);
    }
    if (op_dict.contains("local_offset")) {
      op.local_offset = nb::cast<uint64_t>(op_dict["local_offset"]);
    }
    op.remote_offset = nb::cast<uint64_t>(op_dict["remote_offset"]);
    std::lock_guard<std::mutex> lock(sink_mutex_);
    const auto before = sink_->stats();
    sink_->post(op);
    const auto after = sink_->stats();
    outstanding_signaled_posts_ +=
        after.posted_completions - before.posted_completions;
    nb::dict out;
    out["posted_writes"] = after.posted_writes - before.posted_writes;
    out["posted_signals"] = after.posted_signals - before.posted_signals;
    out["posted_completions"] =
        after.posted_completions - before.posted_completions;
    out["posted_bytes"] = after.posted_bytes - before.posted_bytes;
    return out;
  }

  uint32_t poll_completions(uint32_t max_entries = 64) {
    std::lock_guard<std::mutex> lock(sink_mutex_);
    return poll_completions_unlocked(max_entries);
  }

  uint64_t outstanding_completions() const {
    std::lock_guard<std::mutex> lock(sink_mutex_);
    return outstanding_signaled_posts_;
  }

  uint32_t poll_completions_unlocked(uint32_t max_entries = 64) {
    if (lane_cqs_.empty() || max_entries == 0) {
      return 0;
    }
    std::vector<ibv_wc> wc(max_entries);
    uint32_t total = 0;
    bool made_progress = true;
    while (total < max_entries && made_progress) {
      made_progress = false;
      for (auto* cq : lane_cqs_) {
        if (total >= max_entries) {
          break;
        }
        const auto room = static_cast<int>(max_entries - total);
        const int ne = ibv_poll_cq(cq, room, wc.data() + total);
        if (ne < 0) {
          throw std::runtime_error("ibv_poll_cq failed for V2 EFA");
        }
        if (ne == 0) {
          continue;
        }
        for (int i = 0; i < ne; ++i) {
          if (wc[total + i].status != IBV_WC_SUCCESS) {
            throw std::runtime_error("V2 EFA completion status is not success");
          }
        }
        total += static_cast<uint32_t>(ne);
        made_progress = true;
      }
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
      out["posted_completions"] = uint64_t{0};
      out["posted_bytes"] = uint64_t{0};
      return out;
    }
    const auto& stats = sink_->stats();
    out["posted_writes"] = stats.posted_writes;
    out["posted_signals"] = stats.posted_signals;
    out["posted_completions"] = stats.posted_completions;
    out["posted_bytes"] = stats.posted_bytes;
    return out;
  }

 private:
  void ensure_connected() const {
    if (sink_ == nullptr) {
      throw std::runtime_error("V2 EFA connection is not connected");
    }
  }

  V2ConnectionDrainStats drain_queue_internal(MappedD2HQueueHandle& queue,
                                              bool coalesce,
                                              bool ack_after_drain) {
    std::lock_guard<std::mutex> lock(sink_mutex_);
    const auto before = sink_->stats();
    uint64_t observed_ready_end = 0;
    const auto commands = queue.poll_ready(&observed_ready_end);
    if (coalesce) {
      v2::CoalescingEfaPostSink coalesced(sink_.get());
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, coalesced);
      coalesced.flush();
    } else {
      v2::drain_v2_transfer_cmds_to_efa_posts(commands, *sink_);
    }
    if (ack_after_drain) {
      queue.ack_ready_until(observed_ready_end);
    }
    const auto after = sink_->stats();
    outstanding_signaled_posts_ +=
        after.posted_completions - before.posted_completions;
    V2ConnectionDrainStats stats;
    stats.drained_commands = commands.size();
    stats.posted_writes = after.posted_writes - before.posted_writes;
    stats.posted_signals = after.posted_signals - before.posted_signals;
    stats.posted_completions =
        after.posted_completions - before.posted_completions;
    stats.posted_bytes = after.posted_bytes - before.posted_bytes;
    return stats;
  }

  void proxy_loop(uint32_t thread_idx, uint32_t num_threads) {
    while (proxy_run_.load(std::memory_order_acquire)) {
      bool progressed = false;
      for (size_t queue_idx = thread_idx; queue_idx < proxy_queues_.size();
           queue_idx += num_threads) {
        auto* queue = proxy_queues_[queue_idx];
        if (queue == nullptr) {
          continue;
        }
        const auto stats =
            drain_queue_internal(*queue, proxy_coalesce_,
                                 proxy_ack_after_drain_);
        if (stats.drained_commands != 0) {
          progressed = true;
          proxy_drained_commands_.fetch_add(stats.drained_commands,
                                            std::memory_order_relaxed);
          proxy_posted_writes_.fetch_add(stats.posted_writes,
                                         std::memory_order_relaxed);
          proxy_posted_signals_.fetch_add(stats.posted_signals,
                                          std::memory_order_relaxed);
          proxy_posted_completions_.fetch_add(stats.posted_completions,
                                              std::memory_order_relaxed);
          proxy_posted_bytes_.fetch_add(stats.posted_bytes,
                                        std::memory_order_relaxed);
        }
      }
      {
        std::lock_guard<std::mutex> lock(sink_mutex_);
        (void)poll_completions_unlocked(64);
      }
      if (!progressed) {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
    }
  }

  int resolve_base_device_index(int requested_index) {
    if (requested_index >= 0) {
      return requested_index;
    }
    if (const char* env = std::getenv("UCCL_V2_EFA_DEVICE_INDEX")) {
      return std::atoi(env);
    }

    int num_devices = 0;
    auto** devices = ibv_get_device_list(&num_devices);
    if (devices == nullptr || num_devices == 0) {
      throw std::runtime_error("ibv_get_device_list found no RDMA devices");
    }
    int selected = 0;
    for (int i = 0; i < num_devices; ++i) {
      const char* name = ibv_get_device_name(devices[i]);
      if (name != nullptr && std::string(name).find("efa") != std::string::npos) {
        selected = i;
        break;
      }
    }
    ibv_free_device_list(devices);
    return selected;
  }

  void open_lane(int selected) {
    int num_devices = 0;
    auto** devices = ibv_get_device_list(&num_devices);
    if (devices == nullptr || num_devices == 0) {
      throw std::runtime_error("ibv_get_device_list found no RDMA devices");
    }
    if (selected < 0 || selected >= num_devices) {
      ibv_free_device_list(devices);
      throw std::out_of_range("V2 EFA lane device index out of range");
    }

    const char* name = ibv_get_device_name(devices[selected]);
    const std::string device_name = name == nullptr ? std::string{} : name;
    auto* context = ibv_open_device(devices[selected]);
    ibv_free_device_list(devices);
    if (context == nullptr) {
      throw std::runtime_error("ibv_open_device failed for V2 EFA lane");
    }

    auto* pd = ibv_alloc_pd(context);
    if (pd == nullptr) {
      ibv_close_device(context);
      throw std::runtime_error("ibv_alloc_pd failed for V2 EFA lane");
    }
    auto* cq = ibv_create_cq(context, kMaxOutstandingSends, nullptr, nullptr, 0);
    if (cq == nullptr) {
      ibv_dealloc_pd(pd);
      ibv_close_device(context);
      throw std::runtime_error("ibv_create_cq failed for V2 EFA lane");
    }

    constexpr int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
    auto* mr = ibv_reg_mr(pd, reinterpret_cast<void*>(local_addr_), bytes_,
                          access);
    if (mr == nullptr) {
      ibv_destroy_cq(cq);
      ibv_dealloc_pd(pd);
      ibv_close_device(context);
      throw std::runtime_error("ibv_reg_mr failed for V2 EFA lane window");
    }
    ibv_mr* workspace_mr = nullptr;
    if (workspace_addr_ != 0 && workspace_bytes_ != 0) {
      workspace_mr = ibv_reg_mr(
          pd, reinterpret_cast<void*>(workspace_addr_), workspace_bytes_,
          access);
      if (workspace_mr == nullptr) {
        ibv_dereg_mr(mr);
        ibv_destroy_cq(cq);
        ibv_dealloc_pd(pd);
        ibv_close_device(context);
        throw std::runtime_error(
            "ibv_reg_mr failed for V2 EFA lane workspace");
      }
    }
    auto* signal_mr =
        ibv_reg_mr(pd, signal_values_.data(),
                   signal_values_.size() * sizeof(uint64_t),
                   IBV_ACCESS_LOCAL_WRITE);
    if (signal_mr == nullptr) {
      if (workspace_mr != nullptr) {
        ibv_dereg_mr(workspace_mr);
      }
      ibv_dereg_mr(mr);
      ibv_destroy_cq(cq);
      ibv_dealloc_pd(pd);
      ibv_close_device(context);
      throw std::runtime_error("ibv_reg_mr failed for V2 EFA lane signal");
    }

    ibv_gid gid{};
    if (ibv_query_gid(context, 1, 0, &gid) != 0) {
      ibv_dereg_mr(signal_mr);
      if (workspace_mr != nullptr) {
        ibv_dereg_mr(workspace_mr);
      }
      ibv_dereg_mr(mr);
      ibv_destroy_cq(cq);
      ibv_dealloc_pd(pd);
      ibv_close_device(context);
      throw std::runtime_error("ibv_query_gid failed for V2 EFA lane");
    }

    auto* qp = create_srd_qp(context, pd, cq);
    lane_contexts_.push_back(context);
    lane_pds_.push_back(pd);
    lane_cqs_.push_back(cq);
    lane_mrs_.push_back(mr);
    lane_workspace_mrs_.push_back(workspace_mr);
    lane_signal_mrs_.push_back(signal_mr);
    lane_gids_.push_back(gid);
    lane_device_names_.push_back(device_name);
    qps_.push_back(qp);
  }

  ibv_qp* create_srd_qp(ibv_context* context, ibv_pd* pd, ibv_cq* cq) {
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
    qp_attr.pd = pd;
    qp_attr.qp_context = context;
    qp_attr.sq_sig_all = 1;
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_DRIVER;

    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    efa_attr.flags = EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

    auto* qp = efadv_create_qp_ex(context, &qp_attr, &efa_attr,
                                  sizeof(efadv_qp_init_attr));
    if (qp == nullptr) {
      throw std::runtime_error("efadv_create_qp_ex failed for V2 EFA lane");
    }

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qkey = v2::kDefaultEfaQKey;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_QKEY) != 0) {
      throw std::runtime_error("ibv_modify_qp INIT failed for V2 EFA lane");
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE) != 0) {
      throw std::runtime_error("ibv_modify_qp RTR failed for V2 EFA lane");
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
      throw std::runtime_error("ibv_modify_qp RTS failed for V2 EFA lane");
    }
    return qp;
#endif
  }

  ibv_ah* create_ah(uint32_t lane, const std::vector<uint8_t>& remote_gid) {
    ibv_ah_attr attr{};
    attr.is_global = 1;
    attr.port_num = 1;
    attr.grh.sgid_index = 0;
    std::memcpy(&attr.grh.dgid, remote_gid.data(), 16);
    attr.grh.hop_limit = 255;
    auto* ah = ibv_create_ah(lane_pds_.at(lane), &attr);
    if (ah == nullptr) {
      throw std::runtime_error("ibv_create_ah failed for V2 EFA");
    }
    return ah;
  }

  void destroy() {
    stop_proxy();
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
    for (auto* mr : lane_signal_mrs_) {
      if (mr != nullptr) {
        ibv_dereg_mr(mr);
      }
    }
    lane_signal_mrs_.clear();
    for (auto* mr : lane_workspace_mrs_) {
      if (mr != nullptr) {
        ibv_dereg_mr(mr);
      }
    }
    lane_workspace_mrs_.clear();
    for (auto* mr : lane_mrs_) {
      if (mr != nullptr) {
        ibv_dereg_mr(mr);
      }
    }
    lane_mrs_.clear();
    for (auto* cq : lane_cqs_) {
      if (cq != nullptr) {
        ibv_destroy_cq(cq);
      }
    }
    lane_cqs_.clear();
    for (auto* pd : lane_pds_) {
      if (pd != nullptr) {
        ibv_dealloc_pd(pd);
      }
    }
    lane_pds_.clear();
    for (auto* context : lane_contexts_) {
      if (context != nullptr) {
        ibv_close_device(context);
      }
    }
    lane_contexts_.clear();
    lane_gids_.clear();
    lane_device_names_.clear();
  }

  std::uintptr_t local_addr_ = 0;
  uint64_t bytes_ = 0;
  std::uintptr_t workspace_addr_ = 0;
  uint64_t workspace_bytes_ = 0;
  uint32_t world_size_ = 0;
  uint32_t rank_ = 0;
  uint32_t num_lanes_ = 1;
  std::vector<std::string> lane_device_names_;
  std::vector<ibv_context*> lane_contexts_;
  std::vector<ibv_pd*> lane_pds_;
  std::vector<ibv_cq*> lane_cqs_;
  std::vector<ibv_mr*> lane_mrs_;
  std::vector<ibv_mr*> lane_workspace_mrs_;
  std::vector<ibv_mr*> lane_signal_mrs_;
  std::vector<ibv_gid> lane_gids_;
  std::vector<ibv_qp*> qps_;
  std::vector<ibv_ah*> ahs_;
  std::vector<uint64_t> signal_values_;
  uint64_t outstanding_signaled_posts_ = 0;
  std::unique_ptr<v2::V2VerbsEndpointTable> endpoint_table_;
  std::unique_ptr<v2::V2EfaVerbsPostSink> sink_;
  mutable std::mutex sink_mutex_;
  std::atomic<bool> proxy_run_{false};
  bool proxy_coalesce_ = true;
  bool proxy_ack_after_drain_ = true;
  std::vector<MappedD2HQueueHandle*> proxy_queues_;
  std::vector<std::thread> proxy_threads_;
  std::atomic<uint64_t> proxy_drained_commands_{0};
  std::atomic<uint64_t> proxy_posted_writes_{0};
  std::atomic<uint64_t> proxy_posted_signals_{0};
  std::atomic<uint64_t> proxy_posted_completions_{0};
  std::atomic<uint64_t> proxy_posted_bytes_{0};
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
                    int, uint64_t, std::uintptr_t, uint64_t>(),
           nb::arg("local_addr"), nb::arg("bytes"), nb::arg("world_size"),
           nb::arg("rank"), nb::arg("num_lanes") = 1,
           nb::arg("device_index") = -1,
           nb::arg("signal_capacity") = 65536,
           nb::arg("workspace_addr") = 0,
           nb::arg("workspace_bytes") = 0)
      .def("local_info", &V2EfaConnectionHandle::local_info)
      .def("connect", &V2EfaConnectionHandle::connect)
      .def("is_connected", &V2EfaConnectionHandle::is_connected)
      .def("drain_queue", &V2EfaConnectionHandle::drain_queue,
           nb::arg("queue"), nb::arg("coalesce") = true,
           nb::arg("ack_after_drain") = true)
      .def("start_proxy", &V2EfaConnectionHandle::start_proxy,
           nb::arg("queues"), nb::arg("coalesce") = true,
           nb::arg("ack_after_drain") = true, nb::arg("num_threads") = 1)
      .def("stop_proxy", &V2EfaConnectionHandle::stop_proxy)
      .def("proxy_stats", &V2EfaConnectionHandle::proxy_stats)
      .def("post_op", &V2EfaConnectionHandle::post_op)
      .def("poll_completions", &V2EfaConnectionHandle::poll_completions,
           nb::arg("max_entries") = 64)
      .def("outstanding_completions",
           &V2EfaConnectionHandle::outstanding_completions)
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
      .def("compile_native_hybrid_dispatch_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm, int num_sf_packs,
              int expert_alignment, int num_qps,
              std::int64_t num_timeout_cycles, bool cached_mode,
              bool deterministic, bool do_cpu_sync, int smem_bytes,
              const std::string& uccl_include_path) {
             const auto plan = self.build_native_hybrid_dispatch_jit_plan(
                 num_max_tokens_per_rank, num_channels_per_sm, num_sf_packs,
                 expert_alignment, num_qps, num_timeout_cycles, cached_mode,
                 deterministic, do_cpu_sync, smem_bytes, uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("num_sf_packs") = 0,
           nb::arg("expert_alignment") = 1,
           nb::arg("num_qps") = 1,
           nb::arg("num_timeout_cycles") = 200000000000ll,
           nb::arg("cached_mode") = false,
           nb::arg("deterministic") = false,
           nb::arg("do_cpu_sync") = false,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "")
      .def("build_dispatch_forward_metadata_jit_plan",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm,
              const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(
                 self.build_dispatch_forward_metadata_jit_plan(
                     num_max_tokens_per_rank, num_channels_per_sm,
                     uccl_include_path));
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("uccl_include_path") = "")
      .def("compile_dispatch_forward_metadata_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              int num_channels_per_sm,
              const std::string& uccl_include_path) {
             const auto plan =
                 self.build_dispatch_forward_metadata_jit_plan(
                     num_max_tokens_per_rank, num_channels_per_sm,
                     uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("uccl_include_path") = "")
      .def("build_dispatch_receiver_metadata_jit_plan",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              const std::string& uccl_include_path) {
             return jit_launch_plan_to_dict(
                 self.build_dispatch_receiver_metadata_jit_plan(
                     num_max_tokens_per_rank, uccl_include_path));
           },
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("uccl_include_path") = "")
      .def("compile_dispatch_receiver_metadata_jit",
           [](const v2::V2EfaRuntime& self, int num_max_tokens_per_rank,
              const std::string& uccl_include_path) {
             const auto plan =
                 self.build_dispatch_receiver_metadata_jit_plan(
                     num_max_tokens_per_rank, uccl_include_path);
             v2::compile_v2_efa_jit_plan(plan);
             return jit_launch_plan_to_dict(plan);
           },
           nb::arg("num_max_tokens_per_rank"),
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
      .def("launch_dispatch_descriptor_enqueue_d2h",
           [](const v2::V2EfaRuntime& self, std::uintptr_t x_ptr,
              std::uintptr_t topk_idx_ptr,
              std::uintptr_t topk_weights_ptr,
              std::uintptr_t window_ptr,
              std::uintptr_t segments_ptr, std::uintptr_t batches_ptr,
              std::uintptr_t route_offsets_ptr,
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
              std::uint32_t source_rank_stride,
              std::uint32_t source_signal_stride,
              std::uint32_t token_record_bytes,
              std::uint32_t record_payload_offset,
              std::uint32_t record_src_global_offset,
              std::uint32_t record_topk_idx_offset,
              std::uint32_t record_topk_weight_offset,
              std::uint32_t record_topk_weight_bytes,
              std::uint32_t signal_stride,
              std::uint32_t num_efa_lanes,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::DispatchTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.src_token_stride = src_token_stride;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.source_rank_stride = source_rank_stride;
             layout.source_signal_stride = source_signal_stride;
             layout.token_record_bytes = token_record_bytes;
             layout.record_payload_offset = record_payload_offset;
             layout.record_src_global_offset = record_src_global_offset;
             layout.record_topk_idx_offset = record_topk_idx_offset;
             layout.record_topk_weight_offset = record_topk_weight_offset;
             layout.record_topk_weight_bytes = record_topk_weight_bytes;
             layout.signal_stride = signal_stride;
             layout.num_efa_lanes = std::max<std::uint32_t>(num_efa_lanes, 1);
             self.launch_dispatch_descriptor_enqueue_d2h(
                 x_ptr, topk_idx_ptr, topk_weights_ptr, window_ptr, segments_ptr,
                 batches_ptr, route_offsets_ptr, counters_ptr,
                 num_tokens, num_max_tokens_per_rank,
                 num_channels_per_sm, scale_bytes, has_topk_weight,
                 cached_mode, deterministic, do_cpu_sync, smem_bytes,
                 commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("x_ptr"),
           nb::arg("topk_idx_ptr"),
           nb::arg("topk_weights_ptr"),
           nb::arg("window_ptr"),
           nb::arg("segments_ptr"),
           nb::arg("batches_ptr"),
           nb::arg("route_offsets_ptr"),
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
           nb::arg("source_rank_stride") = 0,
           nb::arg("source_signal_stride") = 0,
           nb::arg("token_record_bytes") = 0,
           nb::arg("record_payload_offset") = 0,
           nb::arg("record_src_global_offset") = 0,
           nb::arg("record_topk_idx_offset") = 0,
           nb::arg("record_topk_weight_offset") = 0,
           nb::arg("record_topk_weight_bytes") = 0,
           nb::arg("signal_stride") = sizeof(std::uint32_t),
           nb::arg("num_efa_lanes") = 1,
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_native_hybrid_dispatch",
           [](const v2::V2EfaRuntime& self, std::uintptr_t x_ptr,
              std::uintptr_t sf_ptr, std::uintptr_t topk_idx_ptr,
              std::uintptr_t topk_weights_ptr,
              std::uintptr_t copied_topk_idx_ptr,
              std::uintptr_t cumulative_local_expert_recv_stats_ptr,
              std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
              std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
              std::uintptr_t dst_buffer_slot_idx_ptr,
              std::uintptr_t token_metadata_at_forward_ptr, int num_tokens,
              int num_max_tokens_per_rank, int num_channels_per_sm,
              int num_sf_packs, int sf_token_stride, int sf_hidden_stride,
              int expert_alignment, int num_qps,
              std::int64_t num_timeout_cycles, bool cached_mode,
              bool deterministic, bool do_cpu_sync, int smem_bytes,
              std::uintptr_t nccl_dev_comm_ptr,
              std::uintptr_t nccl_window_ptr, std::uintptr_t buffer_ptr,
              std::uintptr_t workspace_ptr,
              std::uintptr_t mapped_host_workspace_ptr,
              std::uintptr_t commands_ptr, std::uintptr_t head_ptr,
              std::uintptr_t tail_ptr, int queue_capacity,
              std::uintptr_t buffer_base, std::uintptr_t workspace_base,
              std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base, std::uint32_t src_token_stride,
              std::uint32_t expanded_slot_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t source_rank_stride,
              std::uint32_t source_signal_stride,
              std::uint32_t token_record_bytes,
              std::uint32_t record_payload_offset,
              std::uint32_t record_src_global_offset,
              std::uint32_t record_topk_idx_offset,
              std::uint32_t record_topk_weight_offset,
              std::uint32_t record_topk_weight_bytes,
              std::uint32_t signal_stride, std::uint32_t num_efa_lanes,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::DispatchTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.src_token_stride = src_token_stride;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.source_rank_stride = source_rank_stride;
             layout.source_signal_stride = source_signal_stride;
             layout.token_record_bytes = token_record_bytes;
             layout.record_payload_offset = record_payload_offset;
             layout.record_src_global_offset = record_src_global_offset;
             layout.record_topk_idx_offset = record_topk_idx_offset;
             layout.record_topk_weight_offset = record_topk_weight_offset;
             layout.record_topk_weight_bytes = record_topk_weight_bytes;
             layout.signal_stride = signal_stride;
             layout.num_efa_lanes = std::max<std::uint32_t>(num_efa_lanes, 1);
             self.launch_native_hybrid_dispatch(
                 x_ptr, sf_ptr, topk_idx_ptr, topk_weights_ptr,
                 copied_topk_idx_ptr, cumulative_local_expert_recv_stats_ptr,
                 psum_num_recv_tokens_per_scaleup_rank_ptr,
                 psum_num_recv_tokens_per_expert_ptr, dst_buffer_slot_idx_ptr,
                 token_metadata_at_forward_ptr, num_tokens,
                 num_max_tokens_per_rank, num_channels_per_sm, num_sf_packs,
                 sf_token_stride, sf_hidden_stride, expert_alignment, num_qps,
                 num_timeout_cycles, cached_mode, deterministic, do_cpu_sync,
                 smem_bytes, nccl_dev_comm_ptr, nccl_window_ptr, buffer_ptr,
                 workspace_ptr, mapped_host_workspace_ptr, commands_ptr,
                 head_ptr, tail_ptr, queue_capacity, buffer_base,
                 workspace_base, layout, uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("x_ptr"),
           nb::arg("sf_ptr"),
           nb::arg("topk_idx_ptr"),
           nb::arg("topk_weights_ptr"),
           nb::arg("copied_topk_idx_ptr"),
           nb::arg("cumulative_local_expert_recv_stats_ptr"),
           nb::arg("psum_num_recv_tokens_per_scaleup_rank_ptr"),
           nb::arg("psum_num_recv_tokens_per_expert_ptr"),
           nb::arg("dst_buffer_slot_idx_ptr"),
           nb::arg("token_metadata_at_forward_ptr"),
           nb::arg("num_tokens"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm") = 1,
           nb::arg("num_sf_packs") = 0,
           nb::arg("sf_token_stride") = 0,
           nb::arg("sf_hidden_stride") = 0,
           nb::arg("expert_alignment") = 1,
           nb::arg("num_qps") = 1,
           nb::arg("num_timeout_cycles") = 200000000000ll,
           nb::arg("cached_mode") = false,
           nb::arg("deterministic") = false,
           nb::arg("do_cpu_sync") = true,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("nccl_dev_comm_ptr"),
           nb::arg("nccl_window_ptr"),
           nb::arg("buffer_ptr"),
           nb::arg("workspace_ptr"),
           nb::arg("mapped_host_workspace_ptr"),
           nb::arg("commands_ptr"),
           nb::arg("head_ptr"),
           nb::arg("tail_ptr"),
           nb::arg("queue_capacity"),
           nb::arg("buffer_base"),
           nb::arg("workspace_base"),
           nb::arg("local_payload_base") = 0,
           nb::arg("remote_payload_base") = 0,
           nb::arg("remote_signal_base") = 0,
           nb::arg("src_token_stride") = 0,
           nb::arg("expanded_slot_stride") = 0,
           nb::arg("batch_payload_stride") = 0,
           nb::arg("source_rank_stride") = 0,
           nb::arg("source_signal_stride") = 0,
           nb::arg("token_record_bytes") = 0,
           nb::arg("record_payload_offset") = 0,
           nb::arg("record_src_global_offset") = 0,
           nb::arg("record_topk_idx_offset") = 0,
           nb::arg("record_topk_weight_offset") = 0,
           nb::arg("record_topk_weight_bytes") = 0,
           nb::arg("signal_stride") = sizeof(std::uint32_t),
           nb::arg("num_efa_lanes") = 1,
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_copy_epilogue",
           [](const v2::V2EfaRuntime& self, std::uintptr_t buffer_ptr,
              std::uintptr_t workspace_ptr,
              std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
              std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
              std::uintptr_t recv_x_ptr, std::uintptr_t recv_sf_ptr,
              std::uintptr_t recv_topk_idx_ptr,
              std::uintptr_t recv_topk_weights_ptr,
              std::uintptr_t recv_src_metadata_ptr,
              std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
              int num_max_tokens_per_rank, int num_channels,
              int num_sf_packs, int recv_sf_token_stride,
              int recv_sf_hidden_stride, bool do_expand, bool cached_mode,
              int smem_bytes, const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             self.launch_dispatch_copy_epilogue(
                 buffer_ptr, workspace_ptr,
                 psum_num_recv_tokens_per_scaleup_rank_ptr,
                 psum_num_recv_tokens_per_expert_ptr, recv_x_ptr, recv_sf_ptr,
                 recv_topk_idx_ptr, recv_topk_weights_ptr,
                 recv_src_metadata_ptr, channel_linked_list_ptr,
                 num_recv_tokens, num_max_tokens_per_rank, num_channels,
                 num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                 do_expand, cached_mode, smem_bytes, uccl_include_path,
                 cuda_stream_ptr);
           },
           nb::arg("buffer_ptr"),
           nb::arg("workspace_ptr"),
           nb::arg("psum_num_recv_tokens_per_scaleup_rank_ptr"),
           nb::arg("psum_num_recv_tokens_per_expert_ptr"),
           nb::arg("recv_x_ptr"),
           nb::arg("recv_sf_ptr"),
           nb::arg("recv_topk_idx_ptr"),
           nb::arg("recv_topk_weights_ptr"),
           nb::arg("recv_src_metadata_ptr"),
           nb::arg("channel_linked_list_ptr"),
           nb::arg("num_recv_tokens"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels"),
           nb::arg("num_sf_packs") = 0,
           nb::arg("recv_sf_token_stride") = 0,
           nb::arg("recv_sf_hidden_stride") = 0,
           nb::arg("do_expand") = false,
           nb::arg("cached_mode") = false,
           nb::arg("smem_bytes") = 228 * 1024,
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_forward_metadata",
           [](const v2::V2EfaRuntime& self,
              std::uintptr_t recv_topk_idx_ptr,
              std::uintptr_t recv_src_metadata_ptr,
              std::uintptr_t token_metadata_at_forward_ptr,
              std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
              int num_max_tokens_per_rank, int num_channels_per_sm,
              int rows_per_channel, bool do_expand,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             self.launch_dispatch_forward_metadata(
                 recv_topk_idx_ptr, recv_src_metadata_ptr,
                 token_metadata_at_forward_ptr, channel_linked_list_ptr,
                 num_recv_tokens, num_max_tokens_per_rank,
                 num_channels_per_sm, rows_per_channel, do_expand,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("recv_topk_idx_ptr"),
           nb::arg("recv_src_metadata_ptr"),
           nb::arg("token_metadata_at_forward_ptr"),
           nb::arg("channel_linked_list_ptr"),
           nb::arg("num_recv_tokens"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("num_channels_per_sm"),
           nb::arg("rows_per_channel"),
           nb::arg("do_expand"),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_receiver_metadata",
           [](const v2::V2EfaRuntime& self,
              std::uintptr_t recv_topk_idx_ptr,
              std::uintptr_t recv_src_global_ptr,
              std::uintptr_t recv_counts_per_rank_ptr,
              std::uintptr_t recv_src_metadata_ptr,
              std::uintptr_t dst_buffer_slot_idx_ptr,
              std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
              std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
              std::uintptr_t expert_counts_aligned_ptr,
              std::uintptr_t expert_counts_scratch_ptr,
              std::uintptr_t next_expanded_scratch_ptr,
              int num_recv_tokens, int num_source_tokens,
              int num_max_tokens_per_rank, int expert_alignment,
              bool do_expand, const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             self.launch_dispatch_receiver_metadata(
                 recv_topk_idx_ptr, recv_src_global_ptr,
                 recv_counts_per_rank_ptr, recv_src_metadata_ptr,
                 dst_buffer_slot_idx_ptr,
                 psum_num_recv_tokens_per_scaleup_rank_ptr,
                 psum_num_recv_tokens_per_expert_ptr,
                 expert_counts_aligned_ptr, expert_counts_scratch_ptr,
                 next_expanded_scratch_ptr, num_recv_tokens, num_source_tokens,
                 num_max_tokens_per_rank, expert_alignment, do_expand,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("recv_topk_idx_ptr"),
           nb::arg("recv_src_global_ptr"),
           nb::arg("recv_counts_per_rank_ptr"),
           nb::arg("recv_src_metadata_ptr"),
           nb::arg("dst_buffer_slot_idx_ptr"),
           nb::arg("psum_num_recv_tokens_per_scaleup_rank_ptr"),
           nb::arg("psum_num_recv_tokens_per_expert_ptr"),
           nb::arg("expert_counts_aligned_ptr"),
           nb::arg("expert_counts_scratch_ptr"),
           nb::arg("next_expanded_scratch_ptr"),
           nb::arg("num_recv_tokens"),
           nb::arg("num_source_tokens"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("expert_alignment"),
           nb::arg("do_expand"),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_materialize_records",
           [](const v2::V2EfaRuntime& self, std::uintptr_t window_ptr,
              std::uintptr_t batch_counts_ptr,
              std::uintptr_t batch_offsets_ptr, std::uintptr_t recv_x_ptr,
              std::uintptr_t recv_topk_idx_ptr,
              std::uintptr_t recv_topk_weights_ptr,
              std::uintptr_t recv_src_global_ptr, int max_batches,
              int num_max_tokens_per_rank, std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base, std::uint32_t src_token_stride,
              std::uint32_t expanded_slot_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t source_rank_stride,
              std::uint32_t source_signal_stride,
              std::uint32_t token_record_bytes,
              std::uint32_t record_payload_offset,
              std::uint32_t record_src_global_offset,
              std::uint32_t record_topk_idx_offset,
              std::uint32_t record_topk_weight_offset,
              std::uint32_t record_topk_weight_bytes,
              std::uint32_t signal_stride, bool has_topk_weight,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             v2::DispatchTransferLayout layout;
             layout.local_payload_base = local_payload_base;
             layout.remote_payload_base = remote_payload_base;
             layout.remote_signal_base = remote_signal_base;
             layout.src_token_stride = src_token_stride;
             layout.expanded_slot_stride = expanded_slot_stride;
             layout.batch_payload_stride = batch_payload_stride;
             layout.source_rank_stride = source_rank_stride;
             layout.source_signal_stride = source_signal_stride;
             layout.token_record_bytes = token_record_bytes;
             layout.record_payload_offset = record_payload_offset;
             layout.record_src_global_offset = record_src_global_offset;
             layout.record_topk_idx_offset = record_topk_idx_offset;
             layout.record_topk_weight_offset = record_topk_weight_offset;
             layout.record_topk_weight_bytes = record_topk_weight_bytes;
             layout.signal_stride = signal_stride;
             self.launch_dispatch_materialize_records(
                 window_ptr, batch_counts_ptr, batch_offsets_ptr, recv_x_ptr,
                 recv_topk_idx_ptr, recv_topk_weights_ptr,
                 recv_src_global_ptr, max_batches, num_max_tokens_per_rank,
                 layout, has_topk_weight, uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("window_ptr"),
           nb::arg("batch_counts_ptr"),
           nb::arg("batch_offsets_ptr"),
           nb::arg("recv_x_ptr"),
           nb::arg("recv_topk_idx_ptr"),
           nb::arg("recv_topk_weights_ptr"),
           nb::arg("recv_src_global_ptr"),
           nb::arg("max_batches"),
           nb::arg("num_max_tokens_per_rank"),
           nb::arg("local_payload_base"),
           nb::arg("remote_payload_base"),
           nb::arg("remote_signal_base"),
           nb::arg("src_token_stride"),
           nb::arg("expanded_slot_stride"),
           nb::arg("batch_payload_stride"),
           nb::arg("source_rank_stride"),
           nb::arg("source_signal_stride"),
           nb::arg("token_record_bytes"),
           nb::arg("record_payload_offset"),
           nb::arg("record_src_global_offset"),
           nb::arg("record_topk_idx_offset"),
           nb::arg("record_topk_weight_offset"),
           nb::arg("record_topk_weight_bytes"),
           nb::arg("signal_stride"),
	           nb::arg("has_topk_weight"),
	           nb::arg("uccl_include_path") = "",
	           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_signal_offsets",
           [](const v2::V2EfaRuntime& self, std::uintptr_t window_ptr,
              std::uintptr_t batch_counts_ptr,
              std::uintptr_t batch_offsets_ptr,
              std::uintptr_t recv_counts_per_rank_ptr,
              std::uintptr_t total_recv_tokens_ptr, int max_batches,
              std::uint64_t local_payload_base,
              std::uint64_t remote_payload_base,
              std::uint64_t remote_signal_base, std::uint32_t src_token_stride,
              std::uint32_t expanded_slot_stride,
              std::uint32_t batch_payload_stride,
              std::uint32_t source_rank_stride,
              std::uint32_t source_signal_stride,
              std::uint32_t token_record_bytes,
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
             layout.source_rank_stride = source_rank_stride;
             layout.source_signal_stride = source_signal_stride;
             layout.token_record_bytes = token_record_bytes;
             layout.signal_stride = signal_stride;
             self.launch_dispatch_signal_offsets(
                 window_ptr, batch_counts_ptr, batch_offsets_ptr,
                 recv_counts_per_rank_ptr, total_recv_tokens_ptr, max_batches,
                 layout, uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("window_ptr"),
           nb::arg("batch_counts_ptr"),
           nb::arg("batch_offsets_ptr"),
           nb::arg("recv_counts_per_rank_ptr"),
           nb::arg("total_recv_tokens_ptr"),
           nb::arg("max_batches"),
           nb::arg("local_payload_base"),
           nb::arg("remote_payload_base"),
           nb::arg("remote_signal_base"),
           nb::arg("src_token_stride"),
           nb::arg("expanded_slot_stride"),
           nb::arg("batch_payload_stride"),
           nb::arg("source_rank_stride"),
           nb::arg("source_signal_stride"),
           nb::arg("token_record_bytes"),
           nb::arg("signal_stride"),
           nb::arg("uccl_include_path") = "",
           nb::arg("cuda_stream_ptr") = 0)
      .def("launch_dispatch_expand_records",
           [](const v2::V2EfaRuntime& self, std::uintptr_t recv_x_ptr,
              std::uintptr_t recv_topk_weights_ptr,
              std::uintptr_t recv_src_metadata_ptr,
              std::uintptr_t expanded_x_ptr,
              std::uintptr_t expanded_topk_weights_ptr, int num_recv_tokens,
              int num_expanded_tokens, bool has_topk_weight,
              int num_max_tokens_per_rank,
              const std::string& uccl_include_path,
              std::uintptr_t cuda_stream_ptr) {
             self.launch_dispatch_expand_records(
                 recv_x_ptr, recv_topk_weights_ptr, recv_src_metadata_ptr,
                 expanded_x_ptr, expanded_topk_weights_ptr, num_recv_tokens,
                 num_expanded_tokens, has_topk_weight,
                 num_max_tokens_per_rank, uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("recv_x_ptr"),
           nb::arg("recv_topk_weights_ptr"),
           nb::arg("recv_src_metadata_ptr"),
           nb::arg("expanded_x_ptr"),
           nb::arg("expanded_topk_weights_ptr"),
           nb::arg("num_recv_tokens"),
           nb::arg("num_expanded_tokens"),
           nb::arg("has_topk_weight"),
           nb::arg("num_max_tokens_per_rank"),
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
              std::uintptr_t channel_linked_list_ptr,
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
                 forward_metadata_ptr, channel_linked_list_ptr, segments_ptr,
                 batches_ptr, counters_ptr, num_forward_rows,
                 num_max_tokens_per_rank, num_channels, payload_bytes,
                 use_expanded_layout, allow_multiple_reduction, smem_bytes,
                 commands_ptr, head_ptr, tail_ptr, queue_capacity, layout,
                 uccl_include_path, cuda_stream_ptr);
           },
           nb::arg("forward_metadata_ptr"),
           nb::arg("channel_linked_list_ptr"),
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
           nb::arg("cuda_stream_ptr") = 0);

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
