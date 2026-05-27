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

    @staticmethod
    def capture() -> EventOverlap:
        return UcclBuffer.capture()

    @staticmethod
    def get_theoretical_num_sms(num_experts: Optional[int] = None, num_topk: Optional[int] = None) -> int:
        return 24 if torch.version.cuda else 64

    @staticmethod
    def get_theoretical_num_qps(num_sms: int) -> int:
        return max(1, int(num_sms))

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        if with_cpu_sync:
            torch.cuda.synchronize()
        dist.barrier(self.group)
        if with_cpu_sync:
            torch.cuda.synchronize()

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

    def _build_v2_metadata(
        self,
        topk_idx: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        num_experts: int,
        num_max_tokens_per_rank: int,
        expert_alignment: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        num_topk = int(topk_idx.size(1))
        experts_per_rank = num_experts // self.num_ranks
        send_counts = torch.empty((self.num_ranks,), dtype=torch.int32, device=topk_idx.device)
        send_src_chunks = []
        for dst_rank in range(self.num_ranks):
            expert_begin = dst_rank * experts_per_rank
            expert_end = expert_begin + experts_per_rank
            selected = ((topk_idx >= expert_begin) & (topk_idx < expert_end)).any(dim=1)
            token_indices = selected.nonzero(as_tuple=True)[0].to(torch.int32)
            send_counts[dst_rank] = token_indices.numel()
            send_src_chunks.append(token_indices + self.rank_idx * int(num_max_tokens_per_rank))

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_src = (
            torch.cat(send_src_chunks, dim=0)
            if send_src_chunks
            else torch.empty((0,), dtype=torch.int32, device=topk_idx.device)
        )
        recv_src = torch.empty((int(recv_counts.sum().item()),), dtype=torch.int32, device=topk_idx.device)
        dist.all_to_all_single(
            recv_src,
            send_src,
            recv_counts.cpu().tolist(),
            send_counts.cpu().tolist(),
            group=self.group,
        )

        recv_metadata = torch.full(
            (max(int(recv_src.numel()), 1), 2 + num_topk),
            -1,
            dtype=torch.int32,
            device=topk_idx.device,
        )
        if recv_src.numel() > 0:
            recv_metadata[: recv_src.numel(), 0] = recv_src
            recv_metadata[: recv_src.numel(), 1] = torch.arange(
                recv_src.numel(), dtype=torch.int32, device=topk_idx.device
            )

        scaleup_counts = torch.empty((self.num_scaleup_ranks,), dtype=torch.int32, device=topk_idx.device)
        for scaleup_rank in range(self.num_scaleup_ranks):
            scaleup_counts[scaleup_rank] = recv_counts[scaleup_rank::self.num_scaleup_ranks].sum()
        psum_scaleup = torch.cumsum(scaleup_counts, 0)

        num_local_experts = num_experts // self.num_ranks
        raw_expert_counts = []
        aligned_expert_counts = []
        for expert in range(num_local_experts):
            count = int((recv_topk_idx == expert).sum().item())
            raw_expert_counts.append(count)
            aligned_expert_counts.append(_align(count, expert_alignment))
        aligned_tensor = torch.tensor(aligned_expert_counts, dtype=torch.int32, device=topk_idx.device)
        psum_expert = torch.cumsum(aligned_tensor, 0)
        dst_buffer_slot_idx = torch.arange(max(int(recv_src.numel()), 1), dtype=torch.int32, device=topk_idx.device)
        return recv_metadata[: recv_src.numel()], psum_scaleup, psum_expert, dst_buffer_slot_idx, aligned_expert_counts, torch.tensor(raw_expert_counts, dtype=torch.int32, device=topk_idx.device)

    def _make_expanded_dispatch(
        self,
        recv_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: Optional[torch.Tensor],
        recv_metadata: torch.Tensor,
        raw_expert_counts: torch.Tensor,
        expert_alignment: int,
    ) -> tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        x_tensor, sf = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        num_recv_tokens, hidden = x_tensor.shape
        num_topk = int(recv_topk_idx.size(1))
        raw_counts = [int(v) for v in raw_expert_counts.cpu().tolist()]
        starts = []
        cursor = 0
        psum_values = []
        for count in raw_counts:
            cursor = _align(cursor, expert_alignment)
            starts.append(cursor)
            cursor += count
            psum_values.append(cursor)
        num_expanded_tokens = max(cursor, 1)

        expanded_x = torch.empty((num_expanded_tokens, hidden), dtype=x_tensor.dtype, device=x_tensor.device)
        expanded_sf = (
            None if sf is None else torch.empty((num_expanded_tokens,) + tuple(sf.shape[1:]), dtype=sf.dtype, device=sf.device)
        )
        expanded_weights = (
            None if recv_topk_weights is None else torch.empty((num_expanded_tokens,), dtype=recv_topk_weights.dtype, device=recv_topk_weights.device)
        )
        expanded_metadata = recv_metadata.clone()
        cursors = starts[:]
        for token_idx in range(num_recv_tokens):
            for topk_idx in range(num_topk):
                expert = int(recv_topk_idx[token_idx, topk_idx].item())
                if expert < 0:
                    continue
                slot = cursors[expert]
                cursors[expert] += 1
                expanded_metadata[token_idx, 2 + topk_idx] = slot
                expanded_x[slot].copy_(x_tensor[token_idx])
                if expanded_sf is not None:
                    expanded_sf[slot].copy_(sf[token_idx])
                if expanded_weights is not None:
                    expanded_weights[slot].copy_(recv_topk_weights[token_idx, topk_idx])

        psum_expert = torch.tensor(psum_values, dtype=torch.int32, device=x_tensor.device)
        packed_x = (expanded_x, expanded_sf) if expanded_sf is not None else expanded_x
        return packed_x, expanded_weights, expanded_metadata, psum_expert

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
        x_tensor = x[0] if isinstance(x, tuple) else x
        legacy = self._ensure_legacy_buffer(int(x_tensor.size(1)))
        if handle is not None:
            recv_x, recv_topk_idx, recv_topk_weights, recv_counts, legacy_handle, event = legacy.dispatch(
                x,
                handle=handle.proxy_handle,
                config=legacy.get_dispatch_config(self.num_ranks),
                previous_event=previous_event,
                async_finish=bool(async_with_compute_stream),
                allocate_on_comm_stream=bool(allocate_on_comm_stream),
            )
            if legacy_handle is not None:
                handle.proxy_handle = legacy_handle
            if recv_topk_idx is None and hasattr(handle, "_cached_recv_topk_idx"):
                recv_topk_idx = handle._cached_recv_topk_idx
            if recv_topk_weights is None and hasattr(handle, "_cached_recv_topk_weights"):
                recv_topk_weights = handle._cached_recv_topk_weights
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
            async_finish=bool(async_with_compute_stream),
            allocate_on_comm_stream=bool(allocate_on_comm_stream),
        )
        recv_src_metadata, psum_scaleup, psum_expert, dst_buffer_slot_idx, expert_counts, raw_expert_counts = \
            self._build_v2_metadata(topk_idx, recv_topk_idx, int(num_experts), int(num_max_tokens_per_rank), int(expert_alignment))
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.copy_(raw_expert_counts)
        if do_expand:
            recv_x, recv_topk_weights, recv_src_metadata, psum_expert = self._make_expanded_dispatch(
                recv_x, recv_topk_idx, recv_topk_weights, recv_src_metadata, raw_expert_counts, int(expert_alignment)
            )
            recv_topk_idx = None
            expert_counts = raw_expert_counts.cpu().tolist()
        local_expert_begin = (self.rank_idx * int(num_experts)) // self.num_ranks
        local_expert_end = ((self.rank_idx + 1) * int(num_experts)) // self.num_ranks
        new_handle = EPHandle(
            bool(do_expand),
            int(num_experts),
            int(expert_alignment),
            int(num_max_tokens_per_rank),
            int(num_sms or UcclBuffer.num_sms),
            topk_idx.clone() if do_handle_copy else topk_idx,
            expert_counts,
            psum_scaleup,
            psum_expert,
            recv_src_metadata,
            dst_buffer_slot_idx,
            None,
            None,
            proxy_handle=legacy_handle,
        )
        if not do_expand:
            new_handle._cached_recv_topk_idx = recv_topk_idx
            new_handle._cached_recv_topk_weights = recv_topk_weights
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
        if handle.do_expand:
            num_topk = int(handle.topk_idx.size(1))
            slots = handle.recv_src_metadata[:, 2 : 2 + num_topk].to(torch.long)
            valid = slots >= 0
            safe_slots = slots.clamp_min(0)
            gathered = x[safe_slots]
            gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
            x = gathered.sum(dim=1)
            topk_weights = None
        return legacy.combine(
            x,
            handle.proxy_handle,
            topk_weights=topk_weights,
            bias=bias,
            config=legacy.get_combine_config(self.num_ranks),
            previous_event=previous_event,
            async_finish=bool(async_with_compute_stream),
            allocate_on_comm_stream=bool(allocate_on_comm_stream),
        )


def _align_2mb(value: int) -> int:
    alignment = 2 * 1024 * 1024
    return ((value + alignment - 1) // alignment) * alignment


def _align(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment
