#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_d2h_queue.cuh"

namespace uccl::v2_efa {

enum class EfaPostOpKind : uint32_t {
  kWrite = 1,
  kSignalWrite = 2,
};

struct EfaPostOp {
  EfaPostOpKind kind = EfaPostOpKind::kWrite;
  uint32_t target_rank = 0;
  uint32_t target_lane = 0;
  uint32_t bytes = 0;
  uint32_t signal_value = 0;
  uint64_t local_offset = 0;
  uint64_t remote_offset = 0;
  uint32_t descriptor_index = 0;
  uint32_t batch_index = 0;
};

struct ResolvedEfaPostOp {
  EfaPostOpKind kind = EfaPostOpKind::kWrite;
  uint32_t target_rank = 0;
  uint32_t target_lane = 0;
  uint32_t bytes = 0;
  uint32_t signal_value = 0;
  uint64_t local_offset = 0;
  uint64_t remote_offset = 0;
  uint64_t remote_addr = 0;
  uint32_t rkey = 0;
  uint32_t descriptor_index = 0;
  uint32_t batch_index = 0;
};

struct EfaRemoteEndpoint {
  uint32_t rank = 0;
  uint32_t lane = 0;
  uint64_t remote_base = 0;
  uint32_t rkey = 0;
  uint64_t bytes = 0;
};

class EfaPostSink {
 public:
  virtual ~EfaPostSink() = default;
  virtual void post(const EfaPostOp& op) = 0;
};

class ResolvedEfaPostSink {
 public:
  virtual ~ResolvedEfaPostSink() = default;
  virtual void post_resolved(const ResolvedEfaPostOp& op) = 0;
};

class RecordingEfaPostSink final : public EfaPostSink {
 public:
  void post(const EfaPostOp& op) override { ops.push_back(op); }
  std::vector<EfaPostOp> ops;
};

class RecordingResolvedEfaPostSink final : public ResolvedEfaPostSink {
 public:
  void post_resolved(const ResolvedEfaPostOp& op) override {
    ops.push_back(op);
  }
  std::vector<ResolvedEfaPostOp> ops;
};

class EndpointTable {
 public:
  explicit EndpointTable(uint32_t num_ranks = 0, uint32_t num_lanes = 1)
      : num_ranks_(num_ranks), num_lanes_(num_lanes),
        endpoints_(static_cast<size_t>(num_ranks) * num_lanes) {}

  uint32_t num_ranks() const { return num_ranks_; }
  uint32_t num_lanes() const { return num_lanes_; }

  void set(const EfaRemoteEndpoint& endpoint) {
    endpoints_.at(index(endpoint.rank, endpoint.lane)) = endpoint;
  }

  const EfaRemoteEndpoint& get(uint32_t rank, uint32_t lane) const {
    return endpoints_.at(index(rank, lane));
  }

 private:
  size_t index(uint32_t rank, uint32_t lane) const {
    if (rank >= num_ranks_ || lane >= num_lanes_) {
      throw std::out_of_range("EFA endpoint index");
    }
    return static_cast<size_t>(rank) * num_lanes_ + lane;
  }

  uint32_t num_ranks_ = 0;
  uint32_t num_lanes_ = 1;
  std::vector<EfaRemoteEndpoint> endpoints_;
};

inline EfaPostOp make_efa_post_op(const V2TransferCmd& command) {
  if (!is_v2_transfer_cmd(command)) {
    throw std::invalid_argument("invalid V2 transfer command header");
  }
  EfaPostOp op;
  op.target_rank = command.target_rank;
  op.target_lane = command.target_lane;
  op.bytes = command.bytes;
  op.signal_value = command.signal_value;
  op.local_offset =
      is_v2_transfer_payload(command) ? v2_transfer_local_offset(command) : 0;
  op.remote_offset = v2_transfer_remote_offset(command);

  const auto kind = static_cast<V2TransferCmdKind>(command.kind);
  if (kind == V2TransferCmdKind::kDispatchPayload ||
      kind == V2TransferCmdKind::kCombinePayload) {
    op.kind = EfaPostOpKind::kWrite;
    return op;
  }
  if (kind == V2TransferCmdKind::kDispatchSignal ||
      kind == V2TransferCmdKind::kCombineSignal) {
    op.kind = EfaPostOpKind::kSignalWrite;
    op.bytes = sizeof(uint32_t);
    return op;
  }
  throw std::invalid_argument("unknown V2 transfer command kind");
}

inline ResolvedEfaPostOp resolve_efa_post_op(const EfaPostOp& op,
                                             const EndpointTable& endpoints) {
  const auto& endpoint = endpoints.get(op.target_rank, op.target_lane);
  if (op.remote_offset > endpoint.bytes ||
      static_cast<uint64_t>(op.bytes) > endpoint.bytes - op.remote_offset) {
    throw std::out_of_range("V2 EFA post exceeds remote endpoint window");
  }

  ResolvedEfaPostOp resolved;
  resolved.kind = op.kind;
  resolved.target_rank = op.target_rank;
  resolved.target_lane = op.target_lane;
  resolved.bytes = op.bytes;
  resolved.signal_value = op.signal_value;
  resolved.local_offset = op.local_offset;
  resolved.remote_offset = op.remote_offset;
  resolved.remote_addr = endpoint.remote_base + op.remote_offset;
  resolved.rkey = endpoint.rkey;
  resolved.descriptor_index = op.descriptor_index;
  resolved.batch_index = op.batch_index;
  return resolved;
}

inline std::vector<ResolvedEfaPostOp> resolve_efa_post_ops(
    const std::vector<EfaPostOp>& ops, const EndpointTable& endpoints) {
  std::vector<ResolvedEfaPostOp> resolved;
  resolved.reserve(ops.size());
  for (const auto& op : ops) {
    resolved.push_back(resolve_efa_post_op(op, endpoints));
  }
  return resolved;
}

class ResolvingEfaPostSink final : public EfaPostSink {
 public:
  ResolvingEfaPostSink(const EndpointTable* endpoints,
                       ResolvedEfaPostSink* sink)
      : endpoints_(endpoints), sink_(sink) {
    if (endpoints_ == nullptr || sink_ == nullptr) {
      throw std::invalid_argument("V2 resolving EFA sink input is null");
    }
  }

  void post(const EfaPostOp& op) override {
    sink_->post_resolved(resolve_efa_post_op(op, *endpoints_));
  }

 private:
  const EndpointTable* endpoints_ = nullptr;
  ResolvedEfaPostSink* sink_ = nullptr;
};

inline bool can_coalesce_efa_write(const EfaPostOp& pending,
                                   const EfaPostOp& next) {
  if (pending.kind != EfaPostOpKind::kWrite ||
      next.kind != EfaPostOpKind::kWrite) {
    return false;
  }
  if (pending.target_rank != next.target_rank ||
      pending.target_lane != next.target_lane) {
    return false;
  }
  return pending.local_offset + pending.bytes == next.local_offset &&
         pending.remote_offset + pending.bytes == next.remote_offset;
}

class CoalescingEfaPostSink final : public EfaPostSink {
 public:
  explicit CoalescingEfaPostSink(EfaPostSink* sink) : sink_(sink) {
    if (sink_ == nullptr) {
      throw std::invalid_argument("V2 coalescing EFA sink input is null");
    }
  }

  ~CoalescingEfaPostSink() override { flush(); }

  void post(const EfaPostOp& op) override {
    if (op.kind != EfaPostOpKind::kWrite) {
      flush();
      sink_->post(op);
      return;
    }

    if (has_pending_ && can_coalesce_efa_write(pending_, op)) {
      pending_.bytes += op.bytes;
      return;
    }

    flush();
    pending_ = op;
    has_pending_ = true;
  }

  void flush() {
    if (!has_pending_) {
      return;
    }
    sink_->post(pending_);
    pending_ = EfaPostOp{};
    has_pending_ = false;
  }

 private:
  EfaPostSink* sink_ = nullptr;
  EfaPostOp pending_;
  bool has_pending_ = false;
};

inline EfaPostOp make_efa_post_op_from_packed_v2_transfer(uint64_t first,
                                                          uint64_t second) {
  return make_efa_post_op(unpack_v2_transfer_cmd(first, second));
}

inline void drain_v2_transfer_cmds_to_efa_posts(
    const std::vector<V2TransferCmd>& commands, EfaPostSink& sink) {
  for (const auto& command : commands) {
    sink.post(make_efa_post_op(command));
  }
}

inline void drain_packed_v2_transfer_cmds_to_efa_posts(
    const std::vector<std::pair<uint64_t, uint64_t>>& packed_commands,
    EfaPostSink& sink) {
  for (const auto& command : packed_commands) {
    sink.post(make_efa_post_op_from_packed_v2_transfer(command.first,
                                                       command.second));
  }
}

template <uint32_t Capacity>
inline size_t drain_v2_d2h_queue_to_efa_posts(
    HostV2TransferD2HQueue<Capacity>& queue, EfaPostSink& sink,
    bool ack_after_drain = true) {
  const auto commands = queue.poll_ready();
  drain_v2_transfer_cmds_to_efa_posts(commands, sink);
  if (ack_after_drain) {
    queue.ack_ready();
  }
  return commands.size();
}

}  // namespace uccl::v2_efa
