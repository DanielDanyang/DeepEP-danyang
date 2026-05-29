#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace uccl::v2_efa {

constexpr int kDescriptorVersion = 1;
constexpr int kInvalidRank = -1;
constexpr int kInvalidExpert = -1;

enum class DescriptorFlags : uint32_t {
  kNone = 0,
  kHasScale = 1u << 0,
  kHasTopkWeight = 1u << 1,
  kIndexList = 1u << 2,
  kReduce = 1u << 3,
};

inline constexpr DescriptorFlags operator|(DescriptorFlags a,
                                           DescriptorFlags b) {
  return static_cast<DescriptorFlags>(static_cast<uint32_t>(a) |
                                      static_cast<uint32_t>(b));
}

struct DispatchSegmentDescriptor {
  int32_t dst_scaleout_rank = kInvalidRank;
  int32_t dst_scaleup_lane = kInvalidRank;
  int32_t expert_id = kInvalidExpert;
  int32_t count = 0;
  int32_t src_token_begin = 0;
  int32_t src_token_index_offset = -1;
  int32_t topk_slot = 0;
  int32_t expanded_slot_begin = 0;
  int32_t payload_bytes = 0;
  int32_t scale_bytes = 0;
  uint32_t flags = 0;
};

struct DispatchExpertBatch {
  int32_t dst_scaleout_rank = kInvalidRank;
  int32_t dst_scaleup_lane = kInvalidRank;
  int32_t expert_id = kInvalidExpert;
  int32_t first_segment = 0;
  int32_t num_segments = 0;
  int32_t total_tokens = 0;
  int32_t reserved = 0;
};

struct CombineSegmentDescriptor {
  int32_t dst_original_rank = kInvalidRank;
  int32_t src_scaleout_rank = kInvalidRank;
  int32_t expert_id = kInvalidExpert;
  int32_t count = 0;
  int32_t expanded_slot_begin = 0;
  int32_t expanded_slot_index_offset = -1;
  int32_t reduced_token_slot = 0;
  int32_t payload_bytes = 0;
  uint32_t flags = static_cast<uint32_t>(DescriptorFlags::kReduce);
  int32_t reserved = 0;
};

struct CombineExpertBatch {
  int32_t dst_original_rank = kInvalidRank;
  int32_t src_scaleout_rank = kInvalidRank;
  int32_t expert_id = kInvalidExpert;
  int32_t first_segment = 0;
  int32_t num_segments = 0;
  int32_t total_tokens = 0;
  int32_t reserved = 0;
};

struct DescriptorPlanStats {
  int32_t num_dispatch_segments = 0;
  int32_t num_dispatch_batches = 0;
  int32_t num_combine_segments = 0;
  int32_t num_combine_batches = 0;
  int32_t max_tokens_per_segment = 0;
  int32_t max_payload_bytes_per_segment = 0;
};

inline void validate_non_negative(const char* name, int value) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
}

inline int64_t max_dispatch_segments(int num_tokens, int num_topk) {
  validate_non_negative("num_tokens", num_tokens);
  validate_non_negative("num_topk", num_topk);
  return static_cast<int64_t>(num_tokens) * static_cast<int64_t>(num_topk);
}

inline int64_t max_expert_batches(int num_experts, int num_scaleout_ranks,
                                  int num_scaleup_ranks) {
  validate_non_negative("num_experts", num_experts);
  validate_non_negative("num_scaleout_ranks", num_scaleout_ranks);
  validate_non_negative("num_scaleup_ranks", num_scaleup_ranks);
  return static_cast<int64_t>(num_experts) *
         static_cast<int64_t>(num_scaleout_ranks) *
         static_cast<int64_t>(num_scaleup_ranks);
}

}  // namespace uccl::v2_efa
