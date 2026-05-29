#pragma once

#include <cstdint>

#include "v2_efa/descriptor.hpp"

namespace uccl::v2_efa {

enum class ProxyCommandKind : uint32_t {
  kDispatchPayload = 1,
  kDispatchSignal = 2,
  kCombinePayload = 3,
  kCombineSignal = 4,
};

struct ProxyCommand {
  uint32_t kind = 0;
  uint32_t descriptor_index = 0;
  uint32_t batch_index = 0;
  uint32_t bytes = 0;
  uint64_t local_offset = 0;
  uint64_t remote_offset = 0;
};

struct ProxyQueueView {
  ProxyCommand* commands = nullptr;
  uint32_t* tail = nullptr;
  uint32_t capacity = 0;
};

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t reserve_proxy_command(ProxyQueueView queue) {
  return atomicAdd(queue.tail, 1u);
}

__device__ __forceinline__ void enqueue_proxy_command(ProxyQueueView queue,
                                                      ProxyCommand command) {
  const auto slot = reserve_proxy_command(queue);
  if (slot < queue.capacity) {
    queue.commands[slot] = command;
  }
}
#endif

}  // namespace uccl::v2_efa
