#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "v2_efa/proxy_queue_host.hpp"

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

inline EfaPostOp make_efa_post_op(const ProxyCommand& command) {
  EfaPostOp op;
  op.target_rank = command.target_rank;
  op.target_lane = command.target_lane;
  op.bytes = command.bytes;
  op.signal_value = command.signal_value;
  op.local_offset = command.local_offset;
  op.remote_offset = command.remote_offset;
  op.descriptor_index = command.descriptor_index;
  op.batch_index = command.batch_index;

  if (is_payload_command(command.kind)) {
    op.kind = EfaPostOpKind::kWrite;
    return op;
  }
  if (is_signal_command(command.kind)) {
    op.kind = EfaPostOpKind::kSignalWrite;
    op.bytes = sizeof(uint32_t);
    return op;
  }
  throw std::invalid_argument("unknown proxy command kind");
}

inline void drain_host_queue_to_efa_posts(const HostProxyQueue& queue,
                                          EfaPostSink& sink) {
  const auto commands = queue.snapshot();
  for (const auto& command : commands) {
    sink.post(make_efa_post_op(command));
  }
}

}  // namespace uccl::v2_efa
