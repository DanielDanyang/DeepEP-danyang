#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "proxy_ctx.hpp"
#include "v2_efa/efa_adapter.hpp"

#if defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#define UCCL_V2_EFA_HAS_VERBS 1
#endif
#endif

#if UCCL_V2_EFA_HAS_VERBS
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#endif

namespace uccl::v2_efa {

constexpr uint32_t kDefaultEfaQKey = 0x11111111u;

struct V2VerbsPostStats {
  uint64_t posted_writes = 0;
  uint64_t posted_signals = 0;
  // Number of WRs expected to generate CQEs.  Current p5en SRD path keeps
  // QP.sq_sig_all=1, so payload writes and signal writes both enter the CQ.
  // If sq_sig_all is ever disabled, only signaled WRs may increment this.
  uint64_t posted_completions = 0;
  uint64_t posted_bytes = 0;
};

#if UCCL_V2_EFA_HAS_VERBS

struct V2VerbsLocalWindow {
  uint64_t base = 0;
  uint64_t bytes = 0;
  uint32_t lkey = 0;
  std::vector<uint32_t> lkeys_by_lane;
};

struct V2VerbsSignalScratch {
  uint64_t* values = nullptr;
  uint64_t capacity = 0;
  uint32_t lkey = 0;
  std::vector<uint32_t> lkeys_by_lane;
};

struct V2VerbsEndpoint {
  uint32_t rank = 0;
  uint32_t lane = 0;
  ibv_qp* qp = nullptr;
  ibv_ah* ah = nullptr;
  uint32_t dst_qpn = 0;
  uint32_t qkey = kDefaultEfaQKey;
  uint64_t remote_base = 0;
  uint64_t remote_bytes = 0;
  uint32_t remote_rkey = 0;
  uint64_t remote_workspace_base = 0;
  uint64_t remote_workspace_bytes = 0;
  uint32_t remote_workspace_rkey = 0;
};

class V2VerbsEndpointTable {
 public:
  explicit V2VerbsEndpointTable(uint32_t num_ranks = 0,
                                uint32_t num_lanes = 1)
      : num_ranks_(num_ranks),
        num_lanes_(num_lanes),
        endpoints_(static_cast<size_t>(num_ranks) * num_lanes) {}

  uint32_t num_ranks() const { return num_ranks_; }
  uint32_t num_lanes() const { return num_lanes_; }

  void set(const V2VerbsEndpoint& endpoint) {
    endpoints_.at(index(endpoint.rank, endpoint.lane)) = endpoint;
  }

  const V2VerbsEndpoint& get(uint32_t rank, uint32_t lane) const {
    return endpoints_.at(index(rank, lane));
  }

 private:
  size_t index(uint32_t rank, uint32_t lane) const {
    if (rank >= num_ranks_ || lane >= num_lanes_) {
      throw std::out_of_range("V2 EFA verbs endpoint index");
    }
    return static_cast<size_t>(rank) * num_lanes_ + lane;
  }

  uint32_t num_ranks_ = 0;
  uint32_t num_lanes_ = 1;
  std::vector<V2VerbsEndpoint> endpoints_;
};

inline void check_v2_verbs_range(const char* name, uint64_t offset,
                                 uint32_t bytes, uint64_t capacity) {
  if (offset > capacity || static_cast<uint64_t>(bytes) > capacity - offset) {
    throw std::out_of_range(std::string(name) + " exceeds V2 verbs window");
  }
}

class V2EfaVerbsPostSink final : public EfaPostSink {
 public:
  V2EfaVerbsPostSink(V2VerbsLocalWindow local_window,
                     V2VerbsLocalWindow workspace_window,
                     V2VerbsSignalScratch signal_scratch,
                     const V2VerbsEndpointTable* endpoints)
      : local_window_(local_window),
        workspace_window_(workspace_window),
        signal_scratch_(signal_scratch),
        endpoints_(endpoints) {
    if (endpoints_ == nullptr) {
      throw std::invalid_argument("V2 EFA verbs endpoint table is null");
    }
    if (local_window_.base == 0 || local_window_.bytes == 0) {
      throw std::invalid_argument("V2 EFA verbs local window is empty");
    }
  }

  void post(const EfaPostOp& op) override {
    const auto& endpoint = endpoints_->get(op.target_rank, op.target_lane);
    validate_endpoint(endpoint);
    const bool use_workspace = op.region == EfaMemoryRegion::kWorkspace;
    if (use_workspace &&
        (workspace_window_.base == 0 || workspace_window_.bytes == 0 ||
         endpoint.remote_workspace_base == 0 ||
         endpoint.remote_workspace_bytes == 0 ||
         endpoint.remote_workspace_rkey == 0)) {
      throw std::invalid_argument(
          "V2 EFA workspace command requires registered workspace MRs");
    }
    const auto& local_region =
        use_workspace && workspace_window_.base != 0 ? workspace_window_
                                                     : local_window_;
    const uint64_t remote_base =
        use_workspace && endpoint.remote_workspace_base != 0
            ? endpoint.remote_workspace_base
            : endpoint.remote_base;
    const uint64_t remote_bytes =
        use_workspace && endpoint.remote_workspace_bytes != 0
            ? endpoint.remote_workspace_bytes
            : endpoint.remote_bytes;
    const uint32_t remote_rkey =
        use_workspace && endpoint.remote_workspace_rkey != 0
            ? endpoint.remote_workspace_rkey
            : endpoint.remote_rkey;
    check_v2_verbs_range("remote", op.remote_offset, op.bytes, remote_bytes);

    if (op.kind == EfaPostOpKind::kWrite) {
      check_v2_verbs_range("local", op.local_offset, op.bytes,
                           local_region.bytes);
      post_write(endpoint, local_region.base + op.local_offset,
                 local_lkey(local_region, endpoint.lane),
                 remote_base + op.remote_offset,
                 remote_rkey, op.bytes, false);
      stats_.posted_writes += 1;
      // QP.sq_sig_all=1 in V2EfaConnectionHandle::open_lane(), so this
      // unsignaled-by-flag write still produces a completion.
      stats_.posted_completions += 1;
      stats_.posted_bytes += op.bytes;
      return;
    }

    if (op.kind == EfaPostOpKind::kSignalWrite) {
      const auto scratch_idx = reserve_signal_scratch();
      signal_scratch_.values[scratch_idx] = op.signal_value;
      post_write(endpoint,
                 reinterpret_cast<uint64_t>(signal_scratch_.values +
                                            scratch_idx),
                 signal_lkey(endpoint.lane),
                 remote_base + op.remote_offset,
                 remote_rkey, op.bytes, true);
      stats_.posted_signals += 1;
      stats_.posted_completions += 1;
      stats_.posted_bytes += op.bytes;
      return;
    }

    throw std::invalid_argument("unknown V2 EFA post op kind");
  }

  const V2VerbsPostStats& stats() const { return stats_; }

  void reset_signal_scratch() { next_signal_scratch_ = 0; }

 private:
  void validate_endpoint(const V2VerbsEndpoint& endpoint) const {
    if (endpoint.qp == nullptr || endpoint.ah == nullptr ||
        endpoint.remote_base == 0 || endpoint.remote_bytes == 0 ||
        endpoint.remote_rkey == 0) {
      throw std::invalid_argument("incomplete V2 EFA verbs endpoint");
    }
  }

  uint64_t reserve_signal_scratch() {
    if (signal_scratch_.values == nullptr || signal_scratch_.capacity == 0) {
      throw std::invalid_argument("V2 EFA signal scratch is empty");
    }
    if (next_signal_scratch_ >= signal_scratch_.capacity) {
      throw std::overflow_error(
          "V2 EFA signal scratch exhausted before completion polling");
    }
    return next_signal_scratch_++;
  }

  uint32_t local_lkey(const V2VerbsLocalWindow& window, uint32_t lane) const {
    if (!window.lkeys_by_lane.empty()) {
      return window.lkeys_by_lane.at(lane);
    }
    return window.lkey;
  }

  uint32_t signal_lkey(uint32_t lane) const {
    if (!signal_scratch_.lkeys_by_lane.empty()) {
      return signal_scratch_.lkeys_by_lane.at(lane);
    }
    return signal_scratch_.lkey;
  }

  void post_write(const V2VerbsEndpoint& endpoint, uint64_t local_addr,
                  uint32_t local_lkey, uint64_t remote_addr,
                  uint32_t remote_rkey, uint32_t bytes, bool signaled) {
#ifdef EFA
    auto* qpx = reinterpret_cast<ibv_qp_ex*>(endpoint.qp);
    ibv_wr_start(qpx);
    qpx->wr_id = next_wr_id_++;
    qpx->comp_mask = 0;
    qpx->wr_flags = signaled ? IBV_SEND_SIGNALED : 0;
    ibv_wr_rdma_write(qpx, remote_rkey, remote_addr);
    ibv_wr_set_ud_addr(qpx, endpoint.ah, endpoint.dst_qpn, endpoint.qkey);
    ibv_wr_set_sge(qpx, local_lkey, local_addr, bytes);
    const int ret = ibv_wr_complete(qpx);
    if (ret != 0) {
      throw std::runtime_error(
          std::string("ibv_wr_complete failed for V2 EFA write: ret=") +
          std::to_string(ret) + " errno=" + std::to_string(errno) + " (" +
          std::strerror(errno) + ")");
    }
#else
    ibv_sge sge{};
    sge.addr = local_addr;
    sge.length = bytes;
    sge.lkey = local_lkey;

    ibv_send_wr wr{};
    wr.wr_id = next_wr_id_++;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;

    ibv_send_wr* bad = nullptr;
    const int ret = ibv_post_send(endpoint.qp, &wr, &bad);
    if (ret != 0) {
      throw std::runtime_error(
          std::string("ibv_post_send failed for V2 RDMA write: ret=") +
          std::to_string(ret) + " errno=" + std::to_string(errno) + " (" +
          std::strerror(errno) + ")");
    }
#endif
  }

  V2VerbsLocalWindow local_window_;
  V2VerbsLocalWindow workspace_window_;
  V2VerbsSignalScratch signal_scratch_;
  const V2VerbsEndpointTable* endpoints_ = nullptr;
  uint64_t next_wr_id_ = 1;
  uint64_t next_signal_scratch_ = 0;
  V2VerbsPostStats stats_;
};

inline V2VerbsLocalWindow make_v2_verbs_local_window_from_proxy_ctx(
    const ProxyCtx& ctx) {
  if (ctx.mr == nullptr || ctx.mr->addr == nullptr || ctx.mr->length == 0) {
    throw std::invalid_argument("ProxyCtx has no registered local MR");
  }
  V2VerbsLocalWindow window;
  window.base = reinterpret_cast<uint64_t>(ctx.mr->addr);
  window.bytes = ctx.mr->length;
  window.lkey = ctx.mr->lkey;
  return window;
}

inline V2VerbsEndpoint make_v2_verbs_endpoint_from_proxy_ctx(
    const ProxyCtx& ctx, uint32_t rank, uint32_t lane, uint32_t ring_idx = 0) {
  if (ctx.remote_addr == 0 || ctx.remote_len == 0 || ctx.remote_rkey == 0) {
    throw std::invalid_argument("ProxyCtx has no remote MR");
  }
  if (ctx.dst_ah == nullptr) {
    throw std::invalid_argument("ProxyCtx has no EFA address handle");
  }

  V2VerbsEndpoint endpoint;
  endpoint.rank = rank;
  endpoint.lane = lane;
  endpoint.remote_base = ctx.remote_addr;
  endpoint.remote_bytes = ctx.remote_len;
  endpoint.remote_rkey = ctx.remote_rkey;
  endpoint.ah = ctx.dst_ah;
  endpoint.qkey = kDefaultEfaQKey;

  if (!ctx.data_qps_by_channel.empty()) {
    const auto local_idx = ring_idx % ctx.data_qps_by_channel.size();
    endpoint.qp = ctx.data_qps_by_channel[local_idx];
  } else if (ctx.qp != nullptr) {
    endpoint.qp = ctx.qp;
  } else {
    endpoint.qp = ctx.ack_qp;
  }

  if (!ctx.dst_data_qpn_by_ring.empty()) {
    const auto remote_idx = ring_idx % ctx.dst_data_qpn_by_ring.size();
    endpoint.dst_qpn = ctx.dst_data_qpn_by_ring[remote_idx];
  } else {
    endpoint.dst_qpn = ctx.dst_qpn;
  }

  if (endpoint.qp == nullptr) {
    throw std::invalid_argument("ProxyCtx has no usable data QP");
  }
  return endpoint;
}

template <typename ProxyCtxPtrContainer>
inline V2VerbsEndpointTable make_v2_verbs_endpoint_table_from_proxy_ctxs(
    const ProxyCtxPtrContainer& ctxs, uint32_t num_scaleout_ranks,
    uint32_t num_scaleup_lanes, uint32_t ring_idx = 0) {
  V2VerbsEndpointTable table(num_scaleout_ranks, num_scaleup_lanes);
  for (uint32_t scaleout_rank = 0; scaleout_rank < num_scaleout_ranks;
       ++scaleout_rank) {
    for (uint32_t lane = 0; lane < num_scaleup_lanes; ++lane) {
      const auto global_rank = scaleout_rank * num_scaleup_lanes + lane;
      if (global_rank >= ctxs.size() || ctxs[global_rank] == nullptr) {
        throw std::out_of_range("missing ProxyCtx for V2 endpoint");
      }
      table.set(make_v2_verbs_endpoint_from_proxy_ctx(
          *ctxs[global_rank], scaleout_rank, lane, ring_idx));
    }
  }
  return table;
}

#endif  // UCCL_V2_EFA_HAS_VERBS

}  // namespace uccl::v2_efa
