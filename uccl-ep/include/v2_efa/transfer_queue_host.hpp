#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_loopback.hpp"

namespace uccl::v2_efa {

struct HostV2TransferQueueStats {
  uint32_t submitted = 0;
  uint32_t overflow = 0;
};

class HostV2TransferQueue {
 public:
  explicit HostV2TransferQueue(uint32_t capacity)
      : commands_(capacity), capacity_(capacity) {}

  V2TransferQueueView view() {
    return V2TransferQueueView{commands_.data(), &tail_, capacity_};
  }

  uint32_t capacity() const { return capacity_; }
  uint32_t tail() const { return tail_; }
  const HostV2TransferQueueStats& stats() const { return stats_; }

  void clear() {
    tail_ = 0;
    stats_ = HostV2TransferQueueStats{};
  }

  bool submit(const V2TransferCmd& command) {
    const auto slot = tail_++;
    if (slot >= capacity_) {
      stats_.overflow += 1;
      return false;
    }
    commands_[slot] = command;
    stats_.submitted += 1;
    return true;
  }

  std::vector<V2TransferCmd> snapshot() const {
    const auto count = tail_ > capacity_ ? capacity_ : tail_;
    return std::vector<V2TransferCmd>(commands_.begin(),
                                      commands_.begin() + count);
  }

  LoopbackStats drain_loopback(const LoopbackMemoryView& memory) const {
    if (tail_ > capacity_) {
      throw std::overflow_error("host V2 transfer queue overflow");
    }
    LoopbackStats stats;
    for (uint32_t i = 0; i < tail_; ++i) {
      execute_loopback_transfer_cmd(commands_[i], memory, &stats);
    }
    return stats;
  }

 private:
  std::vector<V2TransferCmd> commands_;
  uint32_t tail_ = 0;
  uint32_t capacity_ = 0;
  HostV2TransferQueueStats stats_;
};

inline HostV2TransferQueueStats submit_v2_transfer_cmds(
    HostV2TransferQueue& queue, const std::vector<V2TransferCmd>& commands) {
  for (const auto& command : commands) {
    queue.submit(command);
  }
  return queue.stats();
}

}  // namespace uccl::v2_efa
