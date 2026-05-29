#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "v2_efa/transfer_cmd.hpp"

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

inline bool is_transfer_payload_command(const V2TransferCmd& command) {
  return is_v2_transfer_payload(command);
}

inline bool is_transfer_signal_command(const V2TransferCmd& command) {
  return is_v2_transfer_signal(command);
}

inline void check_loopback_range(const char* name, uint64_t offset,
                                 uint32_t bytes, size_t capacity) {
  if (offset > capacity || static_cast<uint64_t>(bytes) > capacity - offset) {
    throw std::out_of_range(std::string(name) + " transfer command range");
  }
}

inline void execute_loopback_transfer_cmd(const V2TransferCmd& command,
                                          const LoopbackMemoryView& memory,
                                          LoopbackStats* stats = nullptr) {
  if (!is_v2_transfer_cmd(command)) {
    throw std::invalid_argument("invalid V2 transfer command header");
  }

  if (is_transfer_payload_command(command)) {
    if (memory.local == nullptr || memory.remote == nullptr) {
      throw std::invalid_argument("loopback payload memory must not be null");
    }
    const auto local_offset = v2_transfer_local_offset(command);
    const auto remote_offset = v2_transfer_remote_offset(command);
    check_loopback_range("local", local_offset, command.bytes,
                         memory.local_bytes);
    check_loopback_range("remote", remote_offset, command.bytes,
                         memory.remote_bytes);
    std::memcpy(memory.remote + remote_offset,
                memory.local + local_offset, command.bytes);
    if (stats != nullptr) {
      stats->payload_commands += 1;
      stats->payload_bytes += command.bytes;
    }
    return;
  }

  if (is_transfer_signal_command(command)) {
    if (memory.remote == nullptr) {
      throw std::invalid_argument("loopback signal memory must not be null");
    }
    const auto remote_offset = v2_transfer_remote_offset(command);
    check_loopback_range("remote", remote_offset, sizeof(uint32_t),
                         memory.remote_bytes);
    std::memcpy(memory.remote + remote_offset, &command.signal_value,
                sizeof(uint32_t));
    if (stats != nullptr) {
      stats->signal_commands += 1;
    }
    return;
  }

  throw std::invalid_argument("unknown V2 transfer command kind");
}

inline LoopbackStats execute_loopback_transfer_cmds(
    const std::vector<V2TransferCmd>& commands,
    const LoopbackMemoryView& memory) {
  LoopbackStats stats;
  for (const auto& command : commands) {
    execute_loopback_transfer_cmd(command, memory, &stats);
  }
  return stats;
}

}  // namespace uccl::v2_efa
