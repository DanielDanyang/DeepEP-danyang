#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "v2_efa/proxy_loopback.hpp"
#include "v2_efa/proxy_queue.cuh"
#include "v2_efa/transfer_cmd.hpp"

namespace uccl::v2_efa {

struct HostProxyQueueStats {
  uint32_t submitted = 0;
  uint32_t overflow = 0;
};

class HostProxyQueue {
 public:
  explicit HostProxyQueue(uint32_t capacity)
      : commands_(capacity), capacity_(capacity) {}

  ProxyQueueView view() {
    return ProxyQueueView{commands_.data(), &tail_, capacity_};
  }

  uint32_t capacity() const { return capacity_; }
  uint32_t tail() const { return tail_; }
  const HostProxyQueueStats& stats() const { return stats_; }

  void clear() {
    tail_ = 0;
    stats_ = HostProxyQueueStats{};
  }

  bool submit(const ProxyCommand& command) {
    const auto slot = tail_++;
    if (slot >= capacity_) {
      stats_.overflow += 1;
      return false;
    }
    commands_[slot] = command;
    stats_.submitted += 1;
    return true;
  }

  std::vector<ProxyCommand> snapshot() const {
    const auto count = tail_ > capacity_ ? capacity_ : tail_;
    return std::vector<ProxyCommand>(commands_.begin(),
                                     commands_.begin() + count);
  }

  std::vector<V2TransferCmd> snapshot_v2_transfer_cmds() const {
    const auto count = tail_ > capacity_ ? capacity_ : tail_;
    std::vector<V2TransferCmd> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      out.push_back(make_v2_transfer_cmd(commands_[i]));
    }
    return out;
  }

  LoopbackStats drain_loopback(const LoopbackMemoryView& memory) const {
    if (tail_ > capacity_) {
      throw std::overflow_error("host proxy queue overflow");
    }
    LoopbackStats stats;
    for (uint32_t i = 0; i < tail_; ++i) {
      execute_loopback_proxy_command(commands_[i], memory, &stats);
    }
    return stats;
  }

 private:
  std::vector<ProxyCommand> commands_;
  uint32_t tail_ = 0;
  uint32_t capacity_ = 0;
  HostProxyQueueStats stats_;
};

class HostV2TransferQueue {
 public:
  explicit HostV2TransferQueue(uint32_t capacity)
      : commands_(capacity), capacity_(capacity) {}

  uint32_t capacity() const { return capacity_; }
  uint32_t tail() const { return tail_; }
  const HostProxyQueueStats& stats() const { return stats_; }

  void clear() {
    tail_ = 0;
    stats_ = HostProxyQueueStats{};
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

 private:
  std::vector<V2TransferCmd> commands_;
  uint32_t tail_ = 0;
  uint32_t capacity_ = 0;
  HostProxyQueueStats stats_;
};

inline HostProxyQueueStats submit_proxy_commands(
    HostProxyQueue& queue, const std::vector<ProxyCommand>& commands) {
  for (const auto& command : commands) {
    queue.submit(command);
  }
  return queue.stats();
}

inline HostProxyQueueStats submit_v2_transfer_cmds(
    HostV2TransferQueue& queue, const std::vector<V2TransferCmd>& commands) {
  for (const auto& command : commands) {
    queue.submit(command);
  }
  return queue.stats();
}

}  // namespace uccl::v2_efa
