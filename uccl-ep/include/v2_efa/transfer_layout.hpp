#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "v2_efa/combine_plan.hpp"
#include "v2_efa/dispatch_plan.hpp"
#include "v2_efa/transfer_cmd.hpp"

namespace uccl::v2_efa {

constexpr uint64_t kV2TransferLayoutAlignment = 64;

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline uint32_t checked_u32_layout_value(const char* name, uint64_t value) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(std::string(name) + " exceeds uint32_t");
  }
  return static_cast<uint32_t>(value);
}

inline int max_dispatch_batch_slot_span(const DispatchPlan& plan) {
  int max_span = 0;
  for (const auto& segment : plan.segments) {
    max_span =
        std::max(max_span, segment.expanded_slot_begin + segment.count);
  }
  return max_span;
}

inline int max_combine_batch_slot_span(const CombinePlan& plan) {
  int max_span = 0;
  for (const auto& segment : plan.segments) {
    max_span = std::max(max_span, segment.reduced_token_slot + segment.count);
  }
  return max_span;
}

inline DispatchTransferLayout make_contiguous_dispatch_transfer_layout(
    const DispatchPlan& plan, uint32_t src_token_stride,
    uint32_t expanded_slot_stride, uint64_t local_payload_base = 0,
    uint64_t remote_payload_base = 0,
    uint32_t signal_stride = sizeof(uint32_t)) {
  if (src_token_stride == 0 || expanded_slot_stride == 0 ||
      signal_stride == 0) {
    throw std::invalid_argument("dispatch transfer layout stride is zero");
  }

  const auto max_payload_bytes =
      static_cast<uint64_t>(max_dispatch_batch_slot_span(plan)) *
      expanded_slot_stride;
  const auto batch_payload_stride =
      align_up_u64(max_payload_bytes, kV2TransferLayoutAlignment);
  const auto signal_base =
      remote_payload_base +
      batch_payload_stride * static_cast<uint64_t>(plan.batches.size());

  DispatchTransferLayout layout;
  layout.local_payload_base = local_payload_base;
  layout.remote_payload_base = remote_payload_base;
  layout.remote_signal_base = signal_base;
  layout.src_token_stride = src_token_stride;
  layout.expanded_slot_stride = expanded_slot_stride;
  layout.batch_payload_stride =
      checked_u32_layout_value("dispatch batch payload stride",
                               batch_payload_stride);
  layout.signal_stride = signal_stride;
  return layout;
}

inline CombineTransferLayout make_contiguous_combine_transfer_layout(
    const CombinePlan& plan, uint32_t expanded_slot_stride,
    uint32_t reduced_token_stride, uint64_t local_payload_base = 0,
    uint64_t remote_payload_base = 0,
    uint32_t signal_stride = sizeof(uint32_t)) {
  if (expanded_slot_stride == 0 || reduced_token_stride == 0 ||
      signal_stride == 0) {
    throw std::invalid_argument("combine transfer layout stride is zero");
  }

  const auto max_payload_bytes =
      static_cast<uint64_t>(max_combine_batch_slot_span(plan)) *
      reduced_token_stride;
  const auto batch_payload_stride =
      align_up_u64(max_payload_bytes, kV2TransferLayoutAlignment);
  const auto signal_base =
      remote_payload_base +
      batch_payload_stride * static_cast<uint64_t>(plan.batches.size());

  CombineTransferLayout layout;
  layout.local_payload_base = local_payload_base;
  layout.remote_payload_base = remote_payload_base;
  layout.remote_signal_base = signal_base;
  layout.expanded_slot_stride = expanded_slot_stride;
  layout.reduced_token_stride = reduced_token_stride;
  layout.batch_payload_stride =
      checked_u32_layout_value("combine batch payload stride",
                               batch_payload_stride);
  layout.signal_stride = signal_stride;
  return layout;
}

}  // namespace uccl::v2_efa
