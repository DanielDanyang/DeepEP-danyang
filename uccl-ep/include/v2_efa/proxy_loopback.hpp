#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "v2_efa/proxy_queue.cuh"

namespace uccl::v2_efa {

struct LoopbackMemoryView {
  uint8_t* local = nullptr;
  size_t local_bytes = 0;
  uint8_t* remote = nullptr;
  size_t remote_bytes = 0;
};

struct LoopbackStats {
  uint32_t payload_commands = 0;
  uint32_t signal_commands = 0;
  uint64_t payload_bytes = 0;
};

inline bool is_payload_command(uint32_t kind) {
  return kind == static_cast<uint32_t>(ProxyCommandKind::kDispatchPayload) ||
         kind == static_cast<uint32_t>(ProxyCommandKind::kCombinePayload);
}

inline bool is_signal_command(uint32_t kind) {
  return kind == static_cast<uint32_t>(ProxyCommandKind::kDispatchSignal) ||
         kind == static_cast<uint32_t>(ProxyCommandKind::kCombineSignal);
}

inline void check_loopback_range(const char* name, uint64_t offset,
                                 uint32_t bytes, size_t capacity) {
  if (offset > capacity || static_cast<uint64_t>(bytes) > capacity - offset) {
    throw std::out_of_range(std::string(name) + " proxy command range");
  }
}

inline void execute_loopback_proxy_command(const ProxyCommand& command,
                                           const LoopbackMemoryView& memory,
                                           LoopbackStats* stats = nullptr) {
  if (is_payload_command(command.kind)) {
    if (memory.local == nullptr || memory.remote == nullptr) {
      throw std::invalid_argument("loopback payload memory must not be null");
    }
    check_loopback_range("local", command.local_offset, command.bytes,
                         memory.local_bytes);
    check_loopback_range("remote", command.remote_offset, command.bytes,
                         memory.remote_bytes);
    std::memcpy(memory.remote + command.remote_offset,
                memory.local + command.local_offset, command.bytes);
    if (stats != nullptr) {
      stats->payload_commands += 1;
      stats->payload_bytes += command.bytes;
    }
    return;
  }

  if (is_signal_command(command.kind)) {
    if (memory.remote == nullptr) {
      throw std::invalid_argument("loopback signal memory must not be null");
    }
    check_loopback_range("remote", command.remote_offset, sizeof(uint32_t),
                         memory.remote_bytes);
    std::memcpy(memory.remote + command.remote_offset, &command.signal_value,
                sizeof(uint32_t));
    if (stats != nullptr) {
      stats->signal_commands += 1;
    }
    return;
  }

  throw std::invalid_argument("unknown proxy command kind");
}

inline LoopbackStats execute_loopback_proxy_commands(
    const std::vector<ProxyCommand>& commands,
    const LoopbackMemoryView& memory) {
  LoopbackStats stats;
  for (const auto& command : commands) {
    execute_loopback_proxy_command(command, memory, &stats);
  }
  return stats;
}

}  // namespace uccl::v2_efa
