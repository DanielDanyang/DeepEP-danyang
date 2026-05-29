from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from uccl import ep

from ..utils.event import EventOverlap


_NATIVE_V2_REWRITE_MESSAGE = (
    "uccl-ep is being rewritten as a native DeepEP V2 AWS EFA backend. "
    "The previous V1/UCCL EP transport path has been removed, and dispatch/"
    "combine must be implemented through V2 JIT .cuh kernels before this "
    "ElasticBuffer can run benchmarks."
)


@dataclass
class EPHandle:
    """DeepEP V2 dispatch handle shape.

    This remains as an API placeholder for the native V2 backend. It must be
    populated by V2 descriptor/JIT dispatch, not by the removed staged-token
    transport metadata.
    """

    do_expand: bool
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    num_sms: int
    topk_idx: torch.Tensor
    num_recv_tokens_per_expert_list: list
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor
    dst_buffer_slot_idx: torch.Tensor
    token_metadata_at_forward: Optional[torch.Tensor]
    channel_linked_list: Optional[torch.Tensor]
    transport_handle: Optional[object] = None


class ElasticBuffer:
    """Native V2 AWS EFA buffer surface.

    Construction is intentionally lightweight so tests can inspect topology,
    descriptor sizes, and workspace plans while dispatch/combine are being
    ported to JIT kernels.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_bytes: Optional[int] = None,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        sl_idx: int = 3,
        num_allocated_qps: int = 0,
        num_cpu_timeout_secs: int = 300,
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ) -> None:
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.explicitly_destroy = explicitly_destroy
        self._destroyed = False

        local_world = int(torch.cuda.device_count() or 1)
        self.num_scaleup_ranks = min(max(1, local_world), self.num_ranks)
        self.num_scaleout_ranks = max(1, self.num_ranks // self.num_scaleup_ranks)
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks

        config = ep.V2EfaRuntimeConfig()
        config.rank = self.rank_idx
        config.world_size = self.num_ranks
        config.scaleout_rank = self.scaleout_rank_idx
        config.scaleup_rank = self.scaleup_rank_idx
        config.num_scaleout_ranks = self.num_scaleout_ranks
        config.num_scaleup_ranks = self.num_scaleup_ranks
        config.num_experts = 1
        config.num_topk = max(1, self.num_topk)
        config.hidden = self.hidden
        config.elem_bytes = 1 if use_fp8_dispatch else 2
        config.num_sms = 0
        self.runtime = ep.V2EfaRuntime(config)
        self.num_bytes = num_bytes or self.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_hybrid_mode,
            allow_multiple_reduction,
        )

    def destroy(self) -> None:
        self._destroyed = True

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        if with_cpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier(self.group)
        if with_cpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
    ) -> int:
        elem_bytes = 1 if use_fp8_dispatch else 2
        token_bytes = hidden * elem_bytes
        metadata_bytes = max(num_topk, 1) * 16
        world = group.size()
        raw_bytes = max(1, world * num_max_tokens_per_rank * (token_bytes + metadata_bytes) * 4)
        return _align_2mb(raw_bytes)

    @staticmethod
    def capture() -> EventOverlap:
        return EventOverlap()

    def get_native_v2_status(self) -> str:
        return self.runtime.status()

    def get_native_v2_workspace_plan(self, num_max_tokens_per_rank: Optional[int] = None):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        return self.runtime.workspace_plan(tokens)

    def get_comm_stream(self) -> torch.Stream:
        raise NotImplementedError(_NATIVE_V2_REWRITE_MESSAGE)

    def dispatch(self, *args, **kwargs):
        raise NotImplementedError(_NATIVE_V2_REWRITE_MESSAGE)

    def combine(self, *args, **kwargs):
        raise NotImplementedError(_NATIVE_V2_REWRITE_MESSAGE)


def _align_2mb(x: int) -> int:
    return ((int(x) + (1 << 21) - 1) // (1 << 21)) << 21
