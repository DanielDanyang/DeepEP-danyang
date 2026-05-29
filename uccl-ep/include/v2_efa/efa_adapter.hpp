#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "v2_efa/transfer_cmd.hpp"

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

class RecordingEfaPostSink final : public EfaPostSink {
 public:
  void post(const EfaPostOp& op) override { ops.push_back(op); }
  std::vector<EfaPostOp> ops;
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
  op.local_offset = command.local_offset;
  op.remote_offset = command.remote_offset;
  op.descriptor_index = command.descriptor_index;
  op.batch_index = command.batch_index;

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

inline void drain_v2_transfer_cmds_to_efa_posts(
    const std::vector<V2TransferCmd>& commands, EfaPostSink& sink) {
  for (const auto& command : commands) {
    sink.post(make_efa_post_op(command));
  }
}

}  // namespace uccl::v2_efa
