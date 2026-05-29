#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "v2_efa/transfer_cmd.hpp"
#include "v2_efa/transfer_loopback.hpp"

namespace uccl::v2_efa {

constexpr uint16_t kV2FifoDoorbellMagic = 0xE2FB;
constexpr uint8_t kV2FifoDoorbellVersion = 1;

enum class V2FifoDoorbellKind : uint8_t {
  kTransferCmd = 1,
  kQuiet = 2,
  kBarrier = 3,
};

#pragma pack(push, 1)
struct V2FifoDoorbell {
  uint16_t magic = kV2FifoDoorbellMagic;
  uint8_t version = kV2FifoDoorbellVersion;
  uint8_t kind = static_cast<uint8_t>(V2FifoDoorbellKind::kTransferCmd);
  uint32_t queue_id = 0;
  uint32_t command_index = 0;
  uint16_t command_count = 1;
  uint16_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(V2FifoDoorbell) == 16,
              "V2FifoDoorbell must fit the existing 128-bit FIFO slot");

inline bool is_v2_fifo_doorbell(const V2FifoDoorbell& doorbell) {
  return doorbell.magic == kV2FifoDoorbellMagic &&
         doorbell.version == kV2FifoDoorbellVersion;
}

inline V2FifoDoorbell make_v2_transfer_doorbell(uint32_t queue_id,
                                                uint32_t command_index,
                                                uint16_t command_count = 1) {
  if (command_count == 0) {
    throw std::invalid_argument("V2 FIFO doorbell command_count is zero");
  }
  V2FifoDoorbell doorbell;
  doorbell.queue_id = queue_id;
  doorbell.command_index = command_index;
  doorbell.command_count = command_count;
  return doorbell;
}

inline void pack_v2_fifo_doorbell(const V2FifoDoorbell& doorbell,
                                  uint64_t* first, uint64_t* second) {
  if (first == nullptr || second == nullptr) {
    throw std::invalid_argument("V2 FIFO doorbell pack output is null");
  }
  uint64_t a = 0;
  uint64_t b = 0;
  a |= static_cast<uint64_t>(doorbell.magic);
  a |= static_cast<uint64_t>(doorbell.version) << 16;
  a |= static_cast<uint64_t>(doorbell.kind) << 24;
  a |= static_cast<uint64_t>(doorbell.queue_id) << 32;
  b |= static_cast<uint64_t>(doorbell.command_index);
  b |= static_cast<uint64_t>(doorbell.command_count) << 32;
  b |= static_cast<uint64_t>(doorbell.reserved) << 48;
  *first = a;
  *second = b;
}

inline V2FifoDoorbell unpack_v2_fifo_doorbell(uint64_t first,
                                              uint64_t second) {
  V2FifoDoorbell doorbell;
  doorbell.magic = static_cast<uint16_t>(first & 0xFFFFu);
  doorbell.version = static_cast<uint8_t>((first >> 16) & 0xFFu);
  doorbell.kind = static_cast<uint8_t>((first >> 24) & 0xFFu);
  doorbell.queue_id = static_cast<uint32_t>((first >> 32) & 0xFFFFFFFFu);
  doorbell.command_index = static_cast<uint32_t>(second & 0xFFFFFFFFu);
  doorbell.command_count = static_cast<uint16_t>((second >> 32) & 0xFFFFu);
  doorbell.reserved = static_cast<uint16_t>((second >> 48) & 0xFFFFu);
  return doorbell;
}

struct V2TransferFifoView {
  V2TransferCmd* commands = nullptr;
  V2FifoDoorbell* doorbells = nullptr;
  uint32_t* tail = nullptr;
  uint32_t capacity = 0;
  uint32_t queue_id = 0;
};

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t reserve_v2_transfer_fifo_slot(
    V2TransferFifoView queue) {
  return atomicAdd(queue.tail, 1u);
}

__device__ __forceinline__ void enqueue_v2_transfer_fifo(
    V2TransferFifoView queue, V2TransferCmd command) {
  const auto slot = reserve_v2_transfer_fifo_slot(queue);
  if (slot < queue.capacity) {
    queue.commands[slot] = command;
    __threadfence_system();
    V2FifoDoorbell doorbell;
    doorbell.queue_id = queue.queue_id;
    doorbell.command_index = slot;
    queue.doorbells[slot] = doorbell;
  }
}
#endif

struct HostV2TransferFifoStats {
  uint32_t submitted = 0;
  uint32_t overflow = 0;
};

class HostV2TransferFifo {
 public:
  explicit HostV2TransferFifo(uint32_t capacity, uint32_t queue_id = 0)
      : commands_(capacity), doorbells_(capacity), capacity_(capacity),
        queue_id_(queue_id) {}

  V2TransferFifoView view() {
    return V2TransferFifoView{commands_.data(), doorbells_.data(), &tail_,
                              capacity_, queue_id_};
  }

  uint32_t capacity() const { return capacity_; }
  uint32_t tail() const { return tail_; }
  uint32_t queue_id() const { return queue_id_; }
  const HostV2TransferFifoStats& stats() const { return stats_; }

  void clear() {
    tail_ = 0;
    stats_ = HostV2TransferFifoStats{};
  }

  bool submit(const V2TransferCmd& command) {
    const auto slot = tail_++;
    if (slot >= capacity_) {
      stats_.overflow += 1;
      return false;
    }
    commands_[slot] = command;
    doorbells_[slot] = make_v2_transfer_doorbell(queue_id_, slot);
    stats_.submitted += 1;
    return true;
  }

  const V2TransferCmd& command_for(const V2FifoDoorbell& doorbell) const {
    validate_doorbell(doorbell);
    return commands_[doorbell.command_index];
  }

  std::vector<V2FifoDoorbell> doorbells() const {
    const auto count = visible_count();
    return std::vector<V2FifoDoorbell>(doorbells_.begin(),
                                       doorbells_.begin() + count);
  }

  std::vector<V2TransferCmd> drain_commands() const {
    std::vector<V2TransferCmd> out;
    for (const auto& doorbell : doorbells()) {
      out.push_back(command_for(doorbell));
    }
    return out;
  }

  LoopbackStats drain_loopback(const LoopbackMemoryView& memory) const {
    LoopbackStats stats;
    for (const auto& command : drain_commands()) {
      execute_loopback_transfer_cmd(command, memory, &stats);
    }
    return stats;
  }

 private:
  uint32_t visible_count() const {
    return tail_ > capacity_ ? capacity_ : tail_;
  }

  void validate_doorbell(const V2FifoDoorbell& doorbell) const {
    if (!is_v2_fifo_doorbell(doorbell)) {
      throw std::invalid_argument("invalid V2 FIFO doorbell header");
    }
    if (doorbell.kind != static_cast<uint8_t>(V2FifoDoorbellKind::kTransferCmd)) {
      throw std::invalid_argument("unsupported V2 FIFO doorbell kind");
    }
    if (doorbell.queue_id != queue_id_) {
      throw std::invalid_argument("V2 FIFO doorbell queue mismatch");
    }
    if (doorbell.command_count != 1) {
      throw std::invalid_argument("batched V2 FIFO doorbells not implemented");
    }
    if (doorbell.command_index >= visible_count()) {
      throw std::out_of_range("V2 FIFO doorbell command index");
    }
  }

  std::vector<V2TransferCmd> commands_;
  std::vector<V2FifoDoorbell> doorbells_;
  uint32_t tail_ = 0;
  uint32_t capacity_ = 0;
  uint32_t queue_id_ = 0;
  HostV2TransferFifoStats stats_;
};

inline HostV2TransferFifoStats submit_v2_transfer_fifo_commands(
    HostV2TransferFifo& fifo, const std::vector<V2TransferCmd>& commands) {
  for (const auto& command : commands) {
    fifo.submit(command);
  }
  return fifo.stats();
}

}  // namespace uccl::v2_efa
