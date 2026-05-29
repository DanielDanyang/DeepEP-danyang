#pragma once

#include <cstddef>
#include <cstdint>

#include "v2_efa/descriptor.hpp"

namespace uccl::v2_efa {

constexpr size_t kWorkspaceAlignment = 128;

inline constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct WorkspaceRegion {
  size_t offset = 0;
  size_t bytes = 0;
};

struct WorkspacePlan {
  WorkspaceRegion dispatch_segments;
  WorkspaceRegion dispatch_batches;
  WorkspaceRegion combine_segments;
  WorkspaceRegion combine_batches;
  WorkspaceRegion dispatch_counters;
  WorkspaceRegion combine_counters;
  size_t total_bytes = 0;
};

inline WorkspacePlan build_workspace_plan(int64_t max_dispatch_segments,
                                          int64_t max_dispatch_batches,
                                          int64_t max_combine_segments,
                                          int64_t max_combine_batches) {
  WorkspacePlan plan;
  size_t cursor = 0;

  auto add_region = [&](size_t bytes) {
    WorkspaceRegion region{cursor, bytes};
    cursor = align_up(cursor + bytes, kWorkspaceAlignment);
    return region;
  };

  plan.dispatch_segments = add_region(
      sizeof(DispatchSegmentDescriptor) *
      static_cast<size_t>(max_dispatch_segments));
  plan.dispatch_batches = add_region(sizeof(DispatchExpertBatch) *
                                     static_cast<size_t>(max_dispatch_batches));
  plan.combine_segments = add_region(sizeof(CombineSegmentDescriptor) *
                                     static_cast<size_t>(max_combine_segments));
  plan.combine_batches = add_region(sizeof(CombineExpertBatch) *
                                    static_cast<size_t>(max_combine_batches));
  plan.dispatch_counters = add_region(sizeof(uint32_t) * 2);
  plan.combine_counters = add_region(sizeof(uint32_t) * 2);
  plan.total_bytes = cursor;
  return plan;
}

}  // namespace uccl::v2_efa
