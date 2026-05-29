#pragma once

#include <cstdint>
#include <stdexcept>

#include "v2_efa/proxy_queue.cuh"

namespace uccl::v2_efa {

constexpr uint16_t kV2TransferCmdMagic = 0xE2FA;
constexpr uint8_t kV2TransferCmdVersion = 1;

enum class V2TransferCmdKind : uint8_t {
  kDispatchPayload = 1,
  kDispatchSignal = 2,
  kCombinePayload = 3,
  kCombineSignal = 4,
};

enum class V2TransferCmdFlags : uint32_t {
  kNone = 0,
  kSignal = 1u << 0,
  kPayload = 1u << 1,
  kDispatch = 1u << 2,
  kCombine = 1u << 3,
};

inline constexpr V2TransferCmdFlags operator|(V2TransferCmdFlags a,
                                              V2TransferCmdFlags b) {
  return static_cast<V2TransferCmdFlags>(static_cast<uint32_t>(a) |
                                         static_cast<uint32_t>(b));
}

#pragma pack(push, 1)
struct V2TransferCmd {
  uint16_t magic = kV2TransferCmdMagic;
  uint8_t version = kV2TransferCmdVersion;
  uint8_t kind = 0;
  uint16_t target_rank = 0;
  uint16_t target_lane = 0;

  uint32_t flags = 0;
  uint32_t descriptor_index = 0;
  uint32_t batch_index = 0;
  uint32_t expert_id = 0;
  uint32_t count = 0;
  uint32_t bytes = 0;
  uint32_t signal_value = 0;
  uint32_t reserved32 = 0;

  uint64_t local_offset = 0;
  uint64_t remote_offset = 0;
  uint64_t reserved0 = 0;
};
#pragma pack(pop)

static_assert(sizeof(V2TransferCmd) == 64,
              "V2TransferCmd must stay one 64-byte cache line");

inline bool is_v2_transfer_cmd(const V2TransferCmd& cmd) {
  return cmd.magic == kV2TransferCmdMagic &&
         cmd.version == kV2TransferCmdVersion;
}

inline V2TransferCmdKind proxy_kind_to_v2_transfer_kind(uint32_t kind) {
  if (kind == static_cast<uint32_t>(ProxyCommandKind::kDispatchPayload)) {
    return V2TransferCmdKind::kDispatchPayload;
  }
  if (kind == static_cast<uint32_t>(ProxyCommandKind::kDispatchSignal)) {
    return V2TransferCmdKind::kDispatchSignal;
  }
  if (kind == static_cast<uint32_t>(ProxyCommandKind::kCombinePayload)) {
    return V2TransferCmdKind::kCombinePayload;
  }
  if (kind == static_cast<uint32_t>(ProxyCommandKind::kCombineSignal)) {
    return V2TransferCmdKind::kCombineSignal;
  }
  throw std::invalid_argument("unknown proxy command kind");
}

inline uint32_t v2_transfer_flags(V2TransferCmdKind kind) {
  switch (kind) {
    case V2TransferCmdKind::kDispatchPayload:
      return static_cast<uint32_t>(V2TransferCmdFlags::kDispatch) |
             static_cast<uint32_t>(V2TransferCmdFlags::kPayload);
    case V2TransferCmdKind::kDispatchSignal:
      return static_cast<uint32_t>(V2TransferCmdFlags::kDispatch) |
             static_cast<uint32_t>(V2TransferCmdFlags::kSignal);
    case V2TransferCmdKind::kCombinePayload:
      return static_cast<uint32_t>(V2TransferCmdFlags::kCombine) |
             static_cast<uint32_t>(V2TransferCmdFlags::kPayload);
    case V2TransferCmdKind::kCombineSignal:
      return static_cast<uint32_t>(V2TransferCmdFlags::kCombine) |
             static_cast<uint32_t>(V2TransferCmdFlags::kSignal);
  }
  throw std::invalid_argument("unknown V2 transfer command kind");
}

inline V2TransferCmd make_v2_transfer_cmd(const ProxyCommand& command,
                                          uint32_t expert_id = 0,
                                          uint32_t count = 0) {
  const auto kind = proxy_kind_to_v2_transfer_kind(command.kind);
  V2TransferCmd out;
  out.kind = static_cast<uint8_t>(kind);
  out.target_rank = static_cast<uint16_t>(command.target_rank);
  out.target_lane = static_cast<uint16_t>(command.target_lane);
  out.flags = v2_transfer_flags(kind);
  out.descriptor_index = command.descriptor_index;
  out.batch_index = command.batch_index;
  out.expert_id = expert_id;
  out.count = count;
  out.bytes = command.bytes;
  out.signal_value = command.signal_value;
  out.local_offset = command.local_offset;
  out.remote_offset = command.remote_offset;
  return out;
}

inline ProxyCommand v2_transfer_cmd_to_proxy_command(
    const V2TransferCmd& command) {
  if (!is_v2_transfer_cmd(command)) {
    throw std::invalid_argument("invalid V2 transfer command header");
  }
  ProxyCommand out;
  out.kind = 0;
  switch (static_cast<V2TransferCmdKind>(command.kind)) {
    case V2TransferCmdKind::kDispatchPayload:
      out.kind = static_cast<uint32_t>(ProxyCommandKind::kDispatchPayload);
      break;
    case V2TransferCmdKind::kDispatchSignal:
      out.kind = static_cast<uint32_t>(ProxyCommandKind::kDispatchSignal);
      break;
    case V2TransferCmdKind::kCombinePayload:
      out.kind = static_cast<uint32_t>(ProxyCommandKind::kCombinePayload);
      break;
    case V2TransferCmdKind::kCombineSignal:
      out.kind = static_cast<uint32_t>(ProxyCommandKind::kCombineSignal);
      break;
  }
  out.descriptor_index = command.descriptor_index;
  out.batch_index = command.batch_index;
  out.bytes = command.bytes;
  out.signal_value = command.signal_value;
  out.target_rank = command.target_rank;
  out.target_lane = command.target_lane;
  out.local_offset = command.local_offset;
  out.remote_offset = command.remote_offset;
  return out;
}

}  // namespace uccl::v2_efa
