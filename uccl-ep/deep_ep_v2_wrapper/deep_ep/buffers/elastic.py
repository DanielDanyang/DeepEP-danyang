from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from uccl import ep
from uccl.ep import Config

from ..proxy_transport import ProxyTransport
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
    transport_handle: Optional[object] = None


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
        # Prefer the launcher-provided local world size so reduced smoke tests
        # such as 2 nodes x 2 ranks still expose a real scaleout dimension.
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))

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
        self._transport: Optional[ProxyTransport] = None
        self._transport_hidden = int(hidden)

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
        # Conservative staging estimate for dispatch, combine, and V2 metadata.
        world = group.size()
        metadata_bytes = max(num_topk, 1) * 16
        bytes_per_rank = num_max_tokens_per_rank * (token_bytes + metadata_bytes)
        return _align_2mb(max(1, world * bytes_per_rank * 4))

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self._transport is not None:
            self._transport.destroy()
            self._transport = None
        if self.runtime is not None and hasattr(self.runtime, "destroy"):
            self.runtime.destroy()
        self._destroyed = True

    @staticmethod
    def capture() -> EventOverlap:
        return ProxyTransport.capture()

    def get_theoretical_num_sms(
        self,
        num_experts: Optional[int] = None,
        num_topk: Optional[int] = None,
        num_scaleout_topk: int = 0,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = 200,
        sm_write_gbs: float = 50,
    ) -> int:
        if num_experts is None or num_topk is None:
            return min(torch.cuda.get_device_properties("cuda").multi_processor_count, 64)
        if num_scaleout_topk != 0:
            raise NotImplementedError("group-limited gate SM modeling is not implemented")

        if rdma_gbs == 0 and self.num_rdma_ranks > 1:
            rdma_gbs = float(os.environ.get("EP_RDMA_GBS", "400"))
        if nvlink_gbs == 0:
            nvlink_gbs = float(os.environ.get("EP_NVLINK_GBS", "900"))

        def expected_topk(num_groups: int) -> float:
            if num_groups <= 1:
                return 1.0
            return num_groups * (
                1
                - math.comb(num_experts - num_experts // num_groups, num_topk)
                / math.comb(num_experts, num_topk)
            )

        num_expected_scaleout_topk = (
            expected_topk(self.num_scaleout_ranks) if self.num_scaleout_ranks > 1 else 0
        )
        num_expected_topk = expected_topk(self.num_ranks)

        sm_read = 1 / num_expected_topk
        sm_write = 0.0
        rdma_traffic = 0.0
        nvlink_traffic = 0.0

        if self.num_scaleout_ranks > 1:
            sm_write += 1 / num_expected_topk
            sm_write += (
                (1 / num_expected_topk)
                * (num_expected_scaleout_topk / self.num_scaleout_ranks)
            )
            rdma_traffic += (
                (1 / num_expected_topk)
                * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))
            )
            sm_read += num_expected_scaleout_topk / num_expected_topk
            sm_write += 1
            nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)
        else:
            if self.num_rdma_ranks > 1:
                sm_write += 1 / num_expected_topk
            sm_write += self.num_nvlink_ranks / self.num_ranks
            nvlink_traffic += (
                self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)
            )
            rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks

        if self.num_scaleout_ranks > 1 and rdma_gbs > 0 and (
            rdma_traffic / rdma_gbs
        ) > (nvlink_traffic / nvlink_gbs):
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
        else:
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        num_device_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_sms = num_device_sms
        if bounded_traffic > 0:
            num_sms = max(
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
            )
        num_sms = _align(max(4, math.ceil(num_sms * 1.25)), 2)
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
        num_sms = min(num_sms, num_device_sms)
        auto_sm_cap = int(os.environ.get("EP_UCCL_MAX_AUTO_SMS", "32"))
        if auto_sm_cap > 0:
            num_sms = min(num_sms, auto_sm_cap)

        if os.environ.get("EP_BUFFER_DEBUG", "0") != "0" and self.rank_idx == 0:
            print(
                "EP SM approximation: "
                f"{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, "
                f"{rdma_gbs=}, {nvlink_gbs=}, {num_expected_scaleout_topk=}, "
                f"{num_expected_topk=}, {bounded_traffic=}, {bounded_gbs=}, {num_sms=}",
                flush=True,
            )
        return int(num_sms)

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        num_qps = min(int(num_sms), 9)
        if self.allow_hybrid_mode:
            num_qps = int(num_sms) * 16 + 1
        return min(num_qps, self.num_allocated_qps) if self.num_allocated_qps else num_qps

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        if with_cpu_sync:
            torch.cuda.synchronize()
        dist.barrier(self.group)
        if with_cpu_sync:
            torch.cuda.synchronize()

    def get_comm_stream(self) -> torch.Stream:
        return torch.cuda.ExternalStream(int(self.runtime.get_comm_stream()))

    @staticmethod
    def _compute_stream_ptr() -> int:
        return int(torch.cuda.current_stream().cuda_stream)

    @staticmethod
    def _dtype_code(dtype: torch.dtype) -> int:
        table = {
            torch.uint8: 0,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 3,
            torch.int64: 4,
            torch.float16: 5,
            torch.bfloat16: 6,
            torch.float32: 7,
            torch.float64: 8,
            torch.bool: 9,
            torch.float8_e4m3fn: 10,
            torch.float8_e4m3fnuz: 10,
        }
        if dtype not in table:
            raise ValueError(f"Unsupported V2 native dtype: {dtype}")
        return table[dtype]

    def _ensure_transport(self, hidden: int, num_sms: int = 0) -> ProxyTransport:
        if self._transport is not None:
            return self._transport

        num_sms = int(num_sms or self.get_theoretical_num_sms())
        ProxyTransport.set_num_sms(num_sms)
        hidden_bytes = hidden * 2
        config = Config(num_sms, 8, 512, 16, 512)
        align_to = 128

        def align_buffer(size: int, margin: float = 1.2) -> int:
            return ((int(size * margin) + align_to - 1) // align_to) * align_to

        num_nvl_bytes = align_buffer(config.get_nvl_buffer_size_hint(hidden_bytes, self.num_ranks))
        num_rdma_bytes = align_buffer(config.get_rdma_buffer_size_hint(hidden_bytes, self.num_ranks))
        self._transport = ProxyTransport(
            self.group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            proxy_mode=False,
            num_qps_per_rank=num_sms,
            explicitly_destroy=True,
        )
        return self._transport

    def _build_v2_metadata(
        self,
        transport_handle,
        recv_topk_idx: torch.Tensor,
        num_experts: int,
        num_max_tokens_per_rank: int,
        expert_alignment: int,
        previous_event=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        num_local_experts = num_experts // self.num_ranks
        num_recv_tokens = int(recv_topk_idx.size(0))
        num_topk = int(recv_topk_idx.size(1))
        recv_metadata = torch.empty(
            (max(num_recv_tokens, 1), 2 + num_topk),
            dtype=torch.int32,
            device=recv_topk_idx.device,
        )
        psum_scaleup = torch.empty(
            (self.num_scaleup_ranks,), dtype=torch.int32, device=recv_topk_idx.device
        )
        psum_expert = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=recv_topk_idx.device
        )
        dst_buffer_slot_idx = torch.empty(
            (max(num_recv_tokens, 1),), dtype=torch.int32, device=recv_topk_idx.device
        )
        raw_expert_counts = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=recv_topk_idx.device
        )
        expanded_expert_cursor = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=recv_topk_idx.device
        )
        if len(transport_handle) >= 10:
            # Internode path: UCCL packet metadata carries source RDMA rank,
            # source NVL rank, and source token index.
            recv_src_meta = transport_handle[9]
            self.runtime.build_v2_dispatch_metadata(
                recv_src_meta.data_ptr(),
                recv_topk_idx.data_ptr(),
                num_recv_tokens,
                num_topk,
                self.num_scaleup_ranks,
                num_local_experts,
                int(num_max_tokens_per_rank),
                int(expert_alignment),
                recv_metadata.data_ptr(),
                psum_scaleup.data_ptr(),
                psum_expert.data_ptr(),
                dst_buffer_slot_idx.data_ptr(),
                raw_expert_counts.data_ptr(),
                expanded_expert_cursor.data_ptr(),
                getattr(previous_event, "event", None),
                False,
                False,
                self._compute_stream_ptr(),
            )
        else:
            # Intranode path: transport returns source token indices plus a
            # rank-prefix matrix. The native helper reconstructs V2 global
            # source token ids from those two tensors.
            rank_prefix_matrix = transport_handle[0]
            recv_src_idx = transport_handle[4]
            self.runtime.build_v2_intranode_dispatch_metadata(
                recv_src_idx.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                recv_topk_idx.data_ptr(),
                num_recv_tokens,
                num_topk,
                num_local_experts,
                int(num_max_tokens_per_rank),
                int(expert_alignment),
                recv_metadata.data_ptr(),
                psum_scaleup.data_ptr(),
                psum_expert.data_ptr(),
                dst_buffer_slot_idx.data_ptr(),
                raw_expert_counts.data_ptr(),
                expanded_expert_cursor.data_ptr(),
                getattr(previous_event, "event", None),
                False,
                False,
                self._compute_stream_ptr(),
            )
        recv_metadata = recv_metadata[:num_recv_tokens]
        dst_buffer_slot_idx = dst_buffer_slot_idx[:num_recv_tokens]
        aligned_expert_counts = [
            _align(int(v), expert_alignment) for v in raw_expert_counts.cpu().tolist()
        ]
        return (
            recv_metadata,
            psum_scaleup,
            psum_expert,
            dst_buffer_slot_idx,
            aligned_expert_counts,
            raw_expert_counts,
        )

    def _build_v2_expanded_payload(
        self,
        recv_x,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: Optional[torch.Tensor],
        recv_src_metadata: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
    ):
        x_tensor, x_scales = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        num_recv_tokens, hidden = x_tensor.shape
        num_topk = int(recv_topk_idx.size(1))
        num_expanded_tokens = max(int(psum_num_recv_tokens_per_expert[-1].item()), 1)
        scale_tail_shape = ()
        num_scales = 0
        if x_scales is not None:
            scale_tail_shape = tuple(x_scales.shape[1:])
            num_scales = 1
            for dim in scale_tail_shape:
                num_scales *= int(dim)

        expanded_x = torch.empty(
            (num_expanded_tokens, hidden), dtype=x_tensor.dtype, device=x_tensor.device
        )
        expanded_scales = (
            None
            if x_scales is None
            else torch.empty(
                (num_expanded_tokens,) + scale_tail_shape,
                dtype=x_scales.dtype,
                device=x_scales.device,
            )
        )
        expanded_weights = (
            None
            if recv_topk_weights is None
            else torch.empty(
                (num_expanded_tokens,),
                dtype=recv_topk_weights.dtype,
                device=recv_topk_weights.device,
            )
        )
        self.runtime.build_v2_expanded_payload(
            x_tensor.data_ptr(),
            0 if x_scales is None else x_scales.data_ptr(),
            0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
            recv_src_metadata.data_ptr(),
            num_recv_tokens,
            num_topk,
            int(hidden * x_tensor.element_size()),
            int(num_scales),
            expanded_x.data_ptr(),
            0 if expanded_scales is None else expanded_scales.data_ptr(),
            0 if expanded_weights is None else expanded_weights.data_ptr(),
            None,
            False,
            False,
            self._compute_stream_ptr(),
        )
        packed_x = (expanded_x, expanded_scales) if expanded_scales is not None else expanded_x
        return packed_x, expanded_weights

    def _build_v2_reduced_combine_input(
        self,
        expanded_x: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        num_topk: int,
        previous_event=None,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[torch.Tensor, EventOverlap]:
        num_recv_tokens = int(recv_src_metadata.size(0))
        hidden = int(expanded_x.size(1))
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else torch.cuda.stream(torch.cuda.current_stream())
        )
        with alloc_ctx:
            reduced_x = torch.empty(
                (max(num_recv_tokens, 1), hidden),
                dtype=expanded_x.dtype,
                device=expanded_x.device,
            )
        event = self.runtime.build_v2_reduced_combine_input(
            expanded_x.data_ptr(),
            recv_src_metadata.data_ptr(),
            num_recv_tokens,
            int(num_topk),
            hidden,
            self._dtype_code(expanded_x.dtype),
            reduced_x.data_ptr(),
            getattr(previous_event, "event", None),
            False,
            bool(allocate_on_comm_stream),
            self._compute_stream_ptr(),
        )
        return reduced_x[:num_recv_tokens], EventOverlap(event, None)

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
        if num_sms == 0:
            if handle is not None:
                num_sms = handle.num_sms
            elif num_experts is not None and topk_idx is not None:
                num_sms = self.get_theoretical_num_sms(int(num_experts), int(topk_idx.size(1)))
        transport = self._ensure_transport(int(x_tensor.size(1)), int(num_sms))
        if num_sms:
            ProxyTransport.set_num_sms(int(num_sms))
        if handle is not None:
            recv_x, recv_topk_idx, recv_topk_weights, recv_counts, transport_handle, event = transport.dispatch(
                x,
                handle=handle.transport_handle,
                config=transport.get_dispatch_config(self.num_ranks),
                previous_event=previous_event,
                async_finish=bool(async_with_compute_stream),
                allocate_on_comm_stream=bool(allocate_on_comm_stream),
            )
            if transport_handle is not None:
                handle.transport_handle = transport_handle
            if recv_topk_idx is None and hasattr(handle, "_cached_recv_topk_idx"):
                recv_topk_idx = handle._cached_recv_topk_idx
            if recv_topk_weights is None and hasattr(handle, "_cached_recv_topk_weights"):
                recv_topk_weights = handle._cached_recv_topk_weights
            return recv_x, recv_topk_idx, recv_topk_weights, handle, event

        if topk_idx is None or num_experts is None:
            raise ValueError("topk_idx and num_experts are required for uncached dispatch")
        num_max_tokens_per_rank = num_max_tokens_per_rank or self.num_max_tokens_per_rank or x_tensor.size(0)
        expert_alignment = expert_alignment or 1
        layout = transport.get_dispatch_layout(
            topk_idx,
            int(num_experts),
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, transport_handle, event = transport.dispatch(
            x,
            num_tokens_per_rank=layout[0],
            num_tokens_per_rdma_rank=layout[1],
            is_token_in_rank=layout[3],
            num_tokens_per_expert=layout[2],
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
            config=transport.get_dispatch_config(self.num_ranks),
            previous_event=None,
            async_finish=bool(async_with_compute_stream),
            allocate_on_comm_stream=bool(allocate_on_comm_stream),
        )
        (
            recv_src_metadata,
            psum_scaleup,
            psum_expert,
            dst_buffer_slot_idx,
            expert_counts,
            raw_expert_counts,
        ) = self._build_v2_metadata(
            transport_handle,
            recv_topk_idx,
            int(num_experts),
            int(num_max_tokens_per_rank),
            int(expert_alignment),
            previous_event=event,
        )
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.copy_(raw_expert_counts)
        expanded_psum_expert = psum_expert
        psum_expert = torch.cumsum(
            torch.tensor(expert_counts, dtype=torch.int32, device=x_tensor.device), 0
        )
        if do_expand:
            recv_x, recv_topk_weights = self._build_v2_expanded_payload(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_src_metadata,
                expanded_psum_expert,
            )
            psum_expert = expanded_psum_expert
            recv_topk_idx = None
            expert_counts = raw_expert_counts.cpu().tolist()
        new_handle = EPHandle(
            bool(do_expand),
            int(num_experts),
            int(expert_alignment),
            int(num_max_tokens_per_rank),
            int(num_sms or ProxyTransport.num_sms),
            topk_idx.clone() if do_handle_copy else topk_idx,
            expert_counts,
            psum_scaleup,
            psum_expert,
            recv_src_metadata,
            dst_buffer_slot_idx,
            None,
            None,
            transport_handle=transport_handle,
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
        if handle.transport_handle is None:
            raise RuntimeError("UCCL AWS combine requires a dispatch handle from this backend")
        transport = self._ensure_transport(int(x.size(1)), int(num_sms or handle.num_sms))
        if num_sms or handle.num_sms:
            ProxyTransport.set_num_sms(int(num_sms or handle.num_sms))
        if handle.do_expand:
            num_topk = int(handle.topk_idx.size(1))
            x, reduce_event = self._build_v2_reduced_combine_input(
                x,
                handle.recv_src_metadata,
                num_topk,
                previous_event=previous_event,
                allocate_on_comm_stream=bool(allocate_on_comm_stream),
            )
            previous_event = reduce_event
            topk_weights = None
        return transport.combine(
            x,
            handle.transport_handle,
            topk_weights=topk_weights,
            bias=bias,
            config=transport.get_combine_config(self.num_ranks),
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
