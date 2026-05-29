#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "v2_efa/efa_adapter.hpp"
#include "v2_efa/transfer_d2h_queue.cuh"

namespace uccl::v2_efa {

struct V2ProxyStats {
  uint64_t drained_commands = 0;
  uint64_t posted_writes = 0;
  uint64_t posted_signals = 0;
};

class V2ProxyPostSink final : public EfaPostSink {
 public:
  void post(const EfaPostOp& op) override {
    ops.push_back(op);
    if (op.kind == EfaPostOpKind::kWrite) {
      stats.posted_writes += 1;
    } else if (op.kind == EfaPostOpKind::kSignalWrite) {
      stats.posted_signals += 1;
    }
  }

  std::vector<EfaPostOp> ops;
  V2ProxyStats stats;
};

template <uint32_t Capacity = kV2TransferD2HQueueSize>
class HostV2TransferProxy {
 public:
  using Queue = HostV2TransferD2HQueue<Capacity>;

  explicit HostV2TransferProxy(EfaPostSink* sink) : sink_(sink) {
    if (sink_ == nullptr) {
      throw std::invalid_argument("V2 proxy sink must not be null");
    }
  }

  void add_queue(Queue* queue) {
    if (queue == nullptr) {
      throw std::invalid_argument("V2 proxy queue must not be null");
    }
    queues_.push_back(queue);
  }

  size_t num_queues() const { return queues_.size(); }
  const V2ProxyStats& stats() const { return stats_; }

  size_t drain_once(bool ack_after_drain = true) {
    size_t total = 0;
    for (auto* queue : queues_) {
      const auto before = total;
      total += drain_v2_d2h_queue_to_efa_posts(*queue, *sink_,
                                               ack_after_drain);
      const auto drained = total - before;
      stats_.drained_commands += drained;
    }
    return total;
  }

 private:
  EfaPostSink* sink_ = nullptr;
  std::vector<Queue*> queues_;
  V2ProxyStats stats_;
};

}  // namespace uccl::v2_efa
