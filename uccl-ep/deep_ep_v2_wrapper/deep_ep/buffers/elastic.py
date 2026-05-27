from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from uccl import ep

from ..utils.event import EventOverlap


@dataclass
class EPHandle:
    """DeepEP V2 dispatch handle shape.

    The fields intentionally mirror `deep_ep.buffers.elastic.EPHandle` so
    existing V2 tests and training code can keep the same access pattern while
    the AWS backend routes internode transport through UCCL-style proxy code.
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
    proxy_handle: Optional[object] = None


class ElasticBuffer:
    """DeepEP V2-compatible AWS EFA buffer.

    This class is the integration surface for the long-term AWS backend.  The
    native transport is expected to replace cross-node V2 Gin operations with:

    1. GPU kernels writing token payloads into registered staging buffers.
    2. GPU kernels submitting 128-bit TransferCmd records.
    3. CPU proxy threads posting EFA verbs RDMA writes/write-with-imm.
    4. Receiver proxy threads publishing ordered tail/count state.

    The Python wrapper is intentionally small; the performance-critical work
    belongs in the native `uccl.ep` extension and DeepEP V2 kernel port.
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
        if deterministic:
            raise NotImplementedError("AWS EFA proxy backend does not support deterministic V2 routing yet")
        if not allow_hybrid_mode:
            raise NotImplementedError("AWS EFA proxy backend currently targets hybrid EP16 first")
        if num_cpu_bytes:
            raise NotImplementedError("CPU elastic buffer segments are not part of the AWS EP backend scope yet")

        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_bytes = num_bytes or self.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_hybrid_mode,
            allow_multiple_reduction,
        )
        self.num_allocated_qps = num_allocated_qps
        self.explicitly_destroy = explicitly_destroy
        self._destroyed = False

        # These are logical V2 topology values for the initial p5en EP16 scope.
        local_world = int(torch.cuda.device_count())
        self.num_scaleup_ranks = local_world
        self.num_scaleout_ranks = max(1, self.num_ranks // max(1, local_world))
        self.scaleout_rank_idx = self.rank_idx // max(1, self.num_scaleup_ranks)
        self.scaleup_rank_idx = self.rank_idx % max(1, self.num_scaleup_ranks)
        self.num_rdma_ranks = self.num_scaleout_ranks
        self.num_nvlink_ranks = self.num_scaleup_ranks

        # Native proxy runtime hook.  This is deliberately explicit so the next
        # porting step can replace it with `ep.ElasticProxyBuffer` or equivalent
        # nanobind bindings without changing the public Python API.
        if not hasattr(ep, "Buffer"):
            raise RuntimeError("uccl.ep native extension is missing Buffer; build uccl-ep first")
        self.runtime = None

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
        # Conservative staging estimate for dispatch+combine plus metadata.
        # Native code will replace this with the exact V2/UCCL layout once the
        # EFA proxy buffer is wired into the extension.
        world = group.size()
        metadata_bytes = max(num_topk, 1) * 16
        bytes_per_rank = num_max_tokens_per_rank * (token_bytes + metadata_bytes)
        return _align_2mb(max(1, world * bytes_per_rank * 4))

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self.runtime is not None and hasattr(self.runtime, "destroy"):
            self.runtime.destroy()
        self._destroyed = True

    def get_physical_domain_size(self) -> Tuple[int, int]:
        return self.num_rdma_ranks, self.num_nvlink_ranks

    def get_logical_domain_size(self) -> Tuple[int, int]:
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event=None,
        previous_event_before_epilogue=None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: bool = False,
        use_tma_aligned_col_major_sf: bool = False,
    ):
        raise NotImplementedError(
            "DeepEP V2 AWS dispatch is not wired yet. Next native step: replace "
            "hybrid_dispatch.cuh remote Gin put/signal with TransferCmd submission."
        )

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias=None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event=None,
        previous_event_before_epilogue=None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        raise NotImplementedError(
            "DeepEP V2 AWS combine is not wired yet. Next native step: port "
            "combine internode writes and completion signaling to the proxy backend."
        )


def _align_2mb(value: int) -> int:
    alignment = 2 * 1024 * 1024
    return ((value + alignment - 1) // alignment) * alignment

