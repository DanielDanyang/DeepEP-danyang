#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "v2_efa/descriptor.hpp"
#include "v2_efa/dispatch_plan.hpp"

namespace uccl::v2_efa {

struct CombinePlan {
  std::vector<CombineSegmentDescriptor> segments;
  std::vector<CombineExpertBatch> batches;
};

struct CombinePlanConfig {
  int dst_original_rank = 0;
  int payload_bytes = 0;
};

namespace detail {

struct CombineBatchKey {
  int dst_original_rank;
  int src_scaleout_rank;
  int expert_id;

  bool operator<(const CombineBatchKey& other) const {
    return std::tie(dst_original_rank, src_scaleout_rank, expert_id) <
           std::tie(other.dst_original_rank, other.src_scaleout_rank,
                    other.expert_id);
  }
};

struct MutableCombineBatch {
  CombineBatchKey key;
  std::vector<CombineSegmentDescriptor> segments;
  int total_tokens = 0;
};

}  // namespace detail

inline CombinePlan build_reference_combine_plan_from_dispatch(
    const DispatchPlan& dispatch_plan, const CombinePlanConfig& config) {
  validate_non_negative("dst_original_rank", config.dst_original_rank);
  validate_non_negative("payload_bytes", config.payload_bytes);

  std::map<detail::CombineBatchKey, detail::MutableCombineBatch> by_key;

  for (const auto& dispatch_segment : dispatch_plan.segments) {
    detail::CombineBatchKey key{config.dst_original_rank,
                                dispatch_segment.dst_scaleout_rank,
                                dispatch_segment.expert_id};
    auto it = by_key.find(key);
    if (it == by_key.end()) {
      detail::MutableCombineBatch batch;
      batch.key = key;
      it = by_key.emplace(key, std::move(batch)).first;
    }

    auto& batch = it->second;
    auto can_extend = false;
    if (!batch.segments.empty()) {
      auto& last = batch.segments.back();
      can_extend =
          last.expanded_slot_index_offset < 0 &&
          last.topk_slot == dispatch_segment.topk_slot &&
          last.expanded_slot_begin + last.count ==
              dispatch_segment.expanded_slot_begin &&
          last.reduced_token_slot + last.count ==
              dispatch_segment.src_token_begin &&
          last.payload_bytes == config.payload_bytes;
    }

    if (can_extend) {
      batch.segments.back().count += dispatch_segment.count;
      batch.total_tokens += dispatch_segment.count;
      continue;
    }

    CombineSegmentDescriptor segment;
    segment.dst_original_rank = key.dst_original_rank;
    segment.src_scaleout_rank = key.src_scaleout_rank;
    segment.expert_id = key.expert_id;
    segment.count = dispatch_segment.count;
    segment.expanded_slot_begin = dispatch_segment.expanded_slot_begin;
    segment.expanded_slot_index_offset = -1;
    segment.topk_slot = dispatch_segment.topk_slot;
    segment.reduced_token_slot = dispatch_segment.src_token_begin;
    segment.payload_bytes = config.payload_bytes;
    segment.flags = static_cast<uint32_t>(DescriptorFlags::kReduce);
    batch.segments.push_back(segment);
    batch.total_tokens += dispatch_segment.count;
  }

  CombinePlan plan;
  for (const auto& item : by_key) {
    const auto& batch = item.second;
    CombineExpertBatch public_batch;
    public_batch.dst_original_rank = batch.key.dst_original_rank;
    public_batch.src_scaleout_rank = batch.key.src_scaleout_rank;
    public_batch.expert_id = batch.key.expert_id;
    public_batch.first_segment = static_cast<int32_t>(plan.segments.size());
    public_batch.num_segments = static_cast<int32_t>(batch.segments.size());
    public_batch.total_tokens = batch.total_tokens;
    plan.segments.insert(plan.segments.end(), batch.segments.begin(),
                         batch.segments.end());
    plan.batches.push_back(public_batch);
  }
  return plan;
}

}  // namespace uccl::v2_efa
