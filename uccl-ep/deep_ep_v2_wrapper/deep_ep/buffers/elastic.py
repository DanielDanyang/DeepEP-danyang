from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from uccl import ep
from uccl.ep import Config

from ..buffer import Buffer as UcclBuffer
from ..utils_uccl import inplace_unique
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
        # AWS p5en 的目标形态是 8 GPU/node；小规模 smoke test 可能只有
        # 1-2 个 rank，所以这里不能直接把 `torch.cuda.device_count()` 当成
        # scaleup 域大小，否则 wrapper 会在 world_size < 8 时暴露假的 EP8。
        local_world = int(torch.cuda.device_count())

        if not hasattr(ep, "ElasticProxyBuffer"):
            raise RuntimeError("uccl.ep native extension is missing ElasticProxyBuffer; rebuild uccl-ep")
        self.runtime = ep.ElasticProxyBuffer(
            self.rank_idx,
            self.num_ranks,
            int(self.num_bytes),
            int(local_world),
            bool(explicitly_destroy),
        )
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.runtime.get_logical_domain_size()
        self.num_rdma_ranks, self.num_nvlink_ranks = self.runtime.get_physical_domain_size()
        self.scaleout_rank_idx = self.runtime.scaleout_rank()
        self.scaleup_rank_idx = self.runtime.scaleup_rank()
        self._legacy_buffer: Optional[UcclBuffer] = None
        self._legacy_hidden = int(hidden)

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
        if self._legacy_buffer is not None:
            self._legacy_buffer.destroy()
            self._legacy_buffer = None
        if self.runtime is not None and hasattr(self.runtime, "destroy"):
            self.runtime.destroy()
        self._destroyed = True

    def _ensure_legacy_buffer(self, hidden: int) -> UcclBuffer:
        if self._legacy_buffer is not None:
            return self._legacy_buffer

        num_sms = 24 if torch.version.cuda else 64
        hidden_bytes = hidden * 2
        config = Config(num_sms, 8, 512, 16, 512)
        align_to = 128

        def align_buffer(size: int, margin: float = 1.2) -> int:
            return ((int(size * margin) + align_to - 1) // align_to) * align_to

        num_nvl_bytes = align_buffer(config.get_nvl_buffer_size_hint(hidden_bytes, self.num_ranks))
        num_rdma_bytes = align_buffer(config.get_rdma_buffer_size_hint(hidden_bytes, self.num_ranks))
        self._legacy_buffer = UcclBuffer(
            self.group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_sms,
            explicitly_destroy=True,
        )
        return self._legacy_buffer

    def _build_legacy_layout(self, topk_idx: torch.Tensor, num_experts: int):
        num_ranks = self.num_ranks
        num_nodes = max(1, self.num_scaleout_ranks)
        experts_per_rank = num_experts // num_ranks
        experts_per_node = num_experts // num_nodes

        rank_idx = topk_idx // experts_per_rank
        rank_idx = rank_idx.to(torch.int64)
        rank_idx.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rank_idx, num_ranks)

        rdma_rank_idx = topk_idx // experts_per_node
        rdma_rank_idx = rdma_rank_idx.to(torch.int64)
        rdma_rank_idx.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rdma_rank_idx, num_nodes)

        num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int32, device=topk_idx.device)
        num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int32, device=topk_idx.device)
        token_idx_in_rank = torch.full(
            (num_ranks, topk_idx.size(0)), -1, dtype=torch.long, device=topk_idx.device
        )
        for rank in range(num_ranks):
            num_tokens_per_rank[rank] = (rank_idx == rank).sum()
            token_sel = (rank_idx == rank).max(dim=-1)[0]
            count = token_sel.sum().item()
            tokens = torch.sort(token_sel.to(torch.int32), descending=True)[1]
            tokens[:count] = torch.sort(tokens[:count])[0]
            token_idx_in_rank[rank][tokens[:count]] = torch.arange(
                count, dtype=torch.long, device=topk_idx.device
            )
        for node in range(num_nodes):
            num_tokens_per_rdma_rank[node] = (rdma_rank_idx == node).sum()
        is_token_in_rank = token_idx_in_rank.T.contiguous().to(torch.int32) >= 0

        num_tokens_per_expert = torch.empty((num_experts,), dtype=torch.int32, device=topk_idx.device)
        for expert in range(num_experts):
            num_tokens_per_expert[expert] = (topk_idx == expert).sum()
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank

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
        if do_expand:
            raise NotImplementedError("UCCL AWS backend does not support V2 expand mode yet")
        if use_tma_aligned_col_major_sf:
            raise NotImplementedError("UCCL AWS backend does not support V2 TMA-aligned scale factors yet")

        x_tensor = x[0] if isinstance(x, tuple) else x
        legacy = self._ensure_legacy_buffer(int(x_tensor.size(1)))
        if handle is not None:
            recv_x, recv_topk_idx, recv_topk_weights, recv_counts, legacy_handle, event = legacy.dispatch(
                x,
                handle=handle.proxy_handle,
                config=legacy.get_dispatch_config(self.num_ranks),
                previous_event=previous_event,
                async_finish=async_with_compute_stream,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
            handle.proxy_handle = legacy_handle
            return recv_x, recv_topk_idx, recv_topk_weights, handle, event

        if topk_idx is None or num_experts is None:
            raise ValueError("topk_idx and num_experts are required for uncached dispatch")
        num_max_tokens_per_rank = num_max_tokens_per_rank or self.num_max_tokens_per_rank or x_tensor.size(0)
        expert_alignment = expert_alignment or 1
        layout = self._build_legacy_layout(topk_idx, int(num_experts))
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, legacy_handle, event = legacy.dispatch(
            x,
            num_tokens_per_rank=layout[0],
            num_tokens_per_rdma_rank=layout[1],
            is_token_in_rank=layout[3],
            num_tokens_per_expert=layout[2],
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
            config=legacy.get_dispatch_config(self.num_ranks),
            previous_event=previous_event,
            async_finish=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        psum_scaleup = torch.cumsum(layout[0].view(self.num_scaleout_ranks, self.num_scaleup_ranks)[self.scaleout_rank_idx], 0)
        local_expert_begin = (self.rank_idx * int(num_experts)) // self.num_ranks
        local_expert_end = ((self.rank_idx + 1) * int(num_experts)) // self.num_ranks
        psum_expert = torch.cumsum(layout[2][local_expert_begin:local_expert_end], 0)
        recv_src_metadata = legacy_handle[-2] if isinstance(legacy_handle, tuple) and len(legacy_handle) >= 2 else torch.empty(0, dtype=torch.int32, device=x_tensor.device)
        dst_buffer_slot_idx = torch.empty(0, dtype=torch.int32, device=x_tensor.device)
        new_handle = EPHandle(
            False,
            int(num_experts),
            int(expert_alignment),
            int(num_max_tokens_per_rank),
            int(num_sms or UcclBuffer.num_sms),
            topk_idx.clone() if do_handle_copy else topk_idx,
            recv_counts,
            psum_scaleup,
            psum_expert,
            recv_src_metadata,
            dst_buffer_slot_idx,
            None,
            None,
            proxy_handle=legacy_handle,
        )
        return recv_x, recv_topk_idx, recv_topk_weights, new_handle, event

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
        if handle.proxy_handle is None:
            raise RuntimeError("UCCL AWS combine requires a dispatch handle from this backend")
        legacy = self._ensure_legacy_buffer(int(x.size(1)))
        return legacy.combine(
            x,
            handle.proxy_handle,
            topk_weights=topk_weights,
            bias=bias,
            config=legacy.get_combine_config(self.num_ranks),
            previous_event=previous_event,
            async_finish=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )


def _align_2mb(value: int) -> int:
    alignment = 2 * 1024 * 1024
    return ((value + alignment - 1) // alignment) * alignment
