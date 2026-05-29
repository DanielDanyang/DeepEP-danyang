#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "v2_efa/transfer_cmd.hpp"

namespace uccl::v2_efa {

constexpr uint32_t kV2TransferD2HQueueSize = 2048;

struct V2TransferD2HQueueView {
  V2TransferCmd* commands = nullptr;
  uint64_t* head = nullptr;
  uint64_t* tail = nullptr;
  uint32_t capacity = 0;
};

template <uint32_t Capacity = kV2TransferD2HQueueSize>
struct alignas(128) V2TransferD2HQueue {
  static_assert((Capacity & (Capacity - 1)) == 0,
                "V2 transfer D2H queue capacity must be a power of two");

  uint64_t head = 0;
  uint64_t tail = 0;
  V2TransferCmd commands[Capacity];
  uint32_t capacity = Capacity;

  static constexpr size_t kAckWords = (Capacity + 63) / 64;
  uint64_t ack_mask[kAckWords] = {};

  V2TransferD2HQueue() {
    for (uint32_t i = 0; i < Capacity; ++i) {
      commands[i] = {};
    }
  }

  static constexpr uint32_t mask() { return Capacity - 1; }

  V2TransferD2HQueueView view() {
    return V2TransferD2HQueueView{commands, &head, &tail, Capacity};
  }

  uint64_t volatile_head() const {
    return __atomic_load_n(&head, __ATOMIC_ACQUIRE);
  }

  uint64_t volatile_tail() const {
    return __atomic_load_n(&tail, __ATOMIC_ACQUIRE);
  }

  void cpu_volatile_store_tail(uint64_t new_tail) {
    __atomic_store_n(&tail, new_tail, __ATOMIC_RELEASE);
  }

  uint8_t volatile_load_kind(uint64_t idx) const {
    return __atomic_load_n(&commands[idx & mask()].kind, __ATOMIC_ACQUIRE);
  }

  V2TransferCmd& load_cmd_entry(uint64_t idx) {
    return commands[idx & mask()];
  }

  const V2TransferCmd& load_cmd_entry(uint64_t idx) const {
    return commands[idx & mask()];
  }

  void volatile_clear_kind(uint64_t idx) {
    __atomic_store_n(&commands[idx & mask()].kind, uint8_t{0},
                     __ATOMIC_RELEASE);
  }

  void mark_acked(uint64_t idx) {
    const auto slot = static_cast<size_t>(idx & mask());
    ack_mask[slot >> 6] |= (uint64_t{1} << (slot & 63));
  }

  void clear_acked(size_t idx) {
    assert(idx < Capacity);
    ack_mask[idx >> 6] &= ~(uint64_t{1} << (idx & 63));
  }

  bool is_acked(size_t idx) const {
    assert(idx < Capacity);
    return (ack_mask[idx >> 6] >> (idx & 63)) & 1u;
  }

  size_t next_unacked(size_t start_idx) const {
    if (start_idx >= Capacity) {
      return Capacity;
    }
    for (size_t idx = start_idx; idx < Capacity; ++idx) {
      if (!is_acked(idx)) {
        return idx;
      }
    }
    return Capacity;
  }

  void clear_acked_range(size_t start, size_t end) {
    for (size_t idx = start; idx < end; ++idx) {
      clear_acked(idx);
    }
  }

  uint64_t advance_tail_from_mask() {
    size_t local = static_cast<size_t>(tail % Capacity);
    size_t next = next_unacked(local);
    if (next == Capacity) {
      if (local != 0) {
        clear_acked_range(local, Capacity);
        tail += Capacity - local;
        const auto wrapped_next = next_unacked(0);
        if (wrapped_next > 0 && wrapped_next <= local) {
          clear_acked_range(0, wrapped_next);
          tail += wrapped_next;
        }
      }
    } else if (next > local) {
      clear_acked_range(local, next);
      tail += next - local;
    }
    cpu_volatile_store_tail(tail);
    return tail;
  }

  bool atomic_set_and_commit(const V2TransferCmd& command,
                             uint64_t* out_slot = nullptr) {
    uint64_t slot = 0;
    while (true) {
      const auto h = __atomic_load_n(&head, __ATOMIC_RELAXED);
      const auto t = __atomic_load_n(&tail, __ATOMIC_ACQUIRE);
      if (h - t == Capacity) {
        return false;
      }
      auto expected = h;
      if (__atomic_compare_exchange_n(&head, &expected, h + 1, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        slot = h;
        break;
      }
    }

    const auto idx = slot & mask();
    auto tmp = command;
    const auto saved_kind = tmp.kind;
    tmp.kind = 0;
    commands[idx] = tmp;
    std::atomic_thread_fence(std::memory_order_release);
    commands[idx].kind = saved_kind;
    if (out_slot != nullptr) {
      *out_slot = slot;
    }
    return true;
  }
};

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ bool enqueue_v2_transfer_d2h(
    V2TransferD2HQueueView queue, V2TransferCmd command,
    uint64_t* out_slot = nullptr) {
  while (true) {
    const auto h = *reinterpret_cast<volatile uint64_t*>(queue.head);
    const auto t = *reinterpret_cast<volatile uint64_t*>(queue.tail);
    if (h - t == queue.capacity) {
      __nanosleep(64);
      continue;
    }
    const auto slot = atomicAdd(
        reinterpret_cast<unsigned long long*>(queue.head),
        static_cast<unsigned long long>(1));
    if (slot - t >= queue.capacity) {
      continue;
    }

    const auto idx = static_cast<uint32_t>(slot) & (queue.capacity - 1);
    const auto saved_kind = command.kind;
    command.kind = 0;
    queue.commands[idx] = command;
    __threadfence_system();
    queue.commands[idx].kind = saved_kind;
    if (out_slot != nullptr) {
      *out_slot = slot;
    }
    return true;
  }
}
#endif

using DefaultV2TransferD2HQueue =
    V2TransferD2HQueue<kV2TransferD2HQueueSize>;

struct HostV2TransferD2HStats {
  uint32_t submitted = 0;
  uint32_t overflow = 0;
};

template <uint32_t Capacity = kV2TransferD2HQueueSize>
class HostV2TransferD2HQueue {
 public:
  V2TransferD2HQueue<Capacity>& queue() { return queue_; }
  const V2TransferD2HQueue<Capacity>& queue() const { return queue_; }

  HostV2TransferD2HStats submit(const std::vector<V2TransferCmd>& commands) {
    HostV2TransferD2HStats stats;
    for (const auto& command : commands) {
      if (queue_.atomic_set_and_commit(command)) {
        stats.submitted += 1;
      } else {
        stats.overflow += 1;
      }
    }
    return stats;
  }

  std::vector<V2TransferCmd> poll_ready() const {
    std::vector<V2TransferCmd> out;
    const auto head = queue_.volatile_head();
    const auto tail = queue_.volatile_tail();
    out.reserve(static_cast<size_t>(head - tail));
    for (uint64_t idx = tail; idx < head; ++idx) {
      if (queue_.volatile_load_kind(idx) == 0) {
        break;
      }
      out.push_back(queue_.load_cmd_entry(idx));
    }
    return out;
  }

  void ack_ready() {
    const auto head = queue_.volatile_head();
    const auto tail = queue_.volatile_tail();
    for (uint64_t idx = tail; idx < head; ++idx) {
      if (queue_.volatile_load_kind(idx) == 0) {
        break;
      }
      queue_.volatile_clear_kind(idx);
      queue_.mark_acked(idx);
    }
    queue_.advance_tail_from_mask();
  }

 private:
  V2TransferD2HQueue<Capacity> queue_;
};

}  // namespace uccl::v2_efa
