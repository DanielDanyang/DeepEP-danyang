#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "v2_efa/descriptor.hpp"
#include "v2_efa/topology.hpp"

namespace uccl::v2_efa {

struct DispatchPlan {
  std::vector<DispatchSegmentDescriptor> segments;
  std::vector<DispatchExpertBatch> batches;
};

struct DispatchPlanConfig {
  int world_size = 1;
  int num_scaleup_ranks = 1;
  int local_scaleout_rank = 0;
  int num_experts = 1;
  int num_topk = 1;
  int payload_bytes = 0;
  int scale_bytes = 0;
  bool has_topk_weight = true;
};

namespace detail {

struct DispatchBatchKey {
  int dst_scaleout_rank;
  int dst_scaleup_lane;
  int expert_id;

  bool operator<(const DispatchBatchKey& other) const {
    return std::tie(dst_scaleout_rank, dst_scaleup_lane, expert_id) <
           std::tie(other.dst_scaleout_rank, other.dst_scaleup_lane,
                    other.expert_id);
  }
};

struct MutableDispatchBatch {
  DispatchBatchKey key;
  std::vector<DispatchSegmentDescriptor> segments;
  int next_expanded_slot = 0;
};

inline uint32_t dispatch_flags(const DispatchPlanConfig& config) {
  uint32_t flags = 0;
  if (config.scale_bytes > 0) {
    flags |= static_cast<uint32_t>(DescriptorFlags::kHasScale);
  }
  if (config.has_topk_weight) {
    flags |= static_cast<uint32_t>(DescriptorFlags::kHasTopkWeight);
  }
  return flags;
}

}  // namespace detail

inline DispatchPlan build_reference_dispatch_plan(
    const int64_t* topk_idx, int num_tokens, const DispatchPlanConfig& config) {
  validate_non_negative("num_tokens", num_tokens);
  validate_positive("world_size", config.world_size);
  validate_positive("num_scaleup_ranks", config.num_scaleup_ranks);
  validate_positive("num_experts", config.num_experts);
  validate_positive("num_topk", config.num_topk);
  validate_non_negative("payload_bytes", config.payload_bytes);
  validate_non_negative("scale_bytes", config.scale_bytes);

  if (topk_idx == nullptr && num_tokens > 0) {
    throw std::invalid_argument("topk_idx must not be null");
  }

  std::map<detail::DispatchBatchKey, detail::MutableDispatchBatch> by_key;
  const auto flags = detail::dispatch_flags(config);

  for (int token = 0; token < num_tokens; ++token) {
    for (int topk_slot = 0; topk_slot < config.num_topk; ++topk_slot) {
      const auto expert = static_cast<int>(
          topk_idx[token * config.num_topk + topk_slot]);
      if (expert < 0) {
        continue;
      }
      const auto route =
          route_expert(expert, config.num_experts, config.world_size,
                       config.num_scaleup_ranks, config.local_scaleout_rank);
      detail::DispatchBatchKey key{route.dst_scaleout_rank,
                                   route.dst_scaleup_lane, expert};
      auto it = by_key.find(key);
      if (it == by_key.end()) {
        detail::MutableDispatchBatch batch;
        batch.key = key;
        it = by_key.emplace(key, std::move(batch)).first;
      }

      auto& batch = it->second;
      const int expanded_slot = batch.next_expanded_slot++;
      auto can_extend = false;
      if (!batch.segments.empty()) {
        auto& last = batch.segments.back();
        can_extend =
            last.src_token_index_offset < 0 &&
            last.topk_slot == topk_slot &&
            last.src_token_begin + last.count == token &&
            last.expanded_slot_begin + last.count == expanded_slot &&
            last.payload_bytes == config.payload_bytes &&
            last.scale_bytes == config.scale_bytes && last.flags == flags;
      }

      if (can_extend) {
        batch.segments.back().count += 1;
        continue;
      }

      DispatchSegmentDescriptor segment;
      segment.dst_scaleout_rank = key.dst_scaleout_rank;
      segment.dst_scaleup_lane = key.dst_scaleup_lane;
      segment.expert_id = key.expert_id;
      segment.count = 1;
      segment.src_token_begin = token;
      segment.src_token_index_offset = -1;
      segment.topk_slot = topk_slot;
      segment.expanded_slot_begin = expanded_slot;
      segment.payload_bytes = config.payload_bytes;
      segment.scale_bytes = config.scale_bytes;
      segment.flags = flags;
      batch.segments.push_back(segment);
    }
  }

  DispatchPlan plan;
  for (const auto& item : by_key) {
    const auto& batch = item.second;
    DispatchExpertBatch public_batch;
    public_batch.dst_scaleout_rank = batch.key.dst_scaleout_rank;
    public_batch.dst_scaleup_lane = batch.key.dst_scaleup_lane;
    public_batch.expert_id = batch.key.expert_id;
    public_batch.first_segment = static_cast<int32_t>(plan.segments.size());
    public_batch.num_segments = static_cast<int32_t>(batch.segments.size());
    public_batch.total_tokens = batch.next_expanded_slot;
    plan.segments.insert(plan.segments.end(), batch.segments.begin(),
                         batch.segments.end());
    plan.batches.push_back(public_batch);
  }
  return plan;
}

}  // namespace uccl::v2_efa
