from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

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


@dataclass
class V2TransportHandle:
    dispatch_segments: torch.Tensor
    dispatch_batches: torch.Tensor
    dispatch_route_offsets: torch.Tensor
    dispatch_counters: torch.Tensor
    combine_segments: Optional[torch.Tensor]
    combine_batches: Optional[torch.Tensor]
    combine_counters: Optional[torch.Tensor]
    d2h_queue: object
    combine_d2h_queue: Optional[object]
    dispatch_layout: dict
    combine_layout: Optional[dict]
    num_dispatch_batches: int
    num_dispatch_segments: int
    num_combine_batches: int
    num_combine_segments: int
    payload_bytes: int
    scale_bytes: int
    dispatch_drain_stats: Optional[dict] = None
    combine_drain_stats: Optional[dict] = None
    timings: Optional[dict] = None


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

        local_world = int(
            os.environ.get("LOCAL_WORLD_SIZE")
            or os.environ.get("LOCAL_SIZE")
            or torch.cuda.device_count()
            or 1
        )
        self.num_scaleup_ranks = min(max(1, local_world), self.num_ranks)
        self.num_scaleout_ranks = max(1, self.num_ranks // self.num_scaleup_ranks)
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks

        self.num_experts = self.num_ranks
        self.elem_bytes = 1 if use_fp8_dispatch else 2
        self.num_sms = 0
        self.runtime = self._make_runtime(
            num_experts=self.num_experts,
            num_topk=max(1, self.num_topk),
            hidden=self.hidden,
            elem_bytes=self.elem_bytes,
            num_sms=self.num_sms,
        )
        self.num_bytes = num_bytes or self.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_hybrid_mode,
            allow_multiple_reduction,
        )
        self.num_allocated_qps = int(num_allocated_qps or 129)
        self.allow_hybrid_mode = bool(allow_hybrid_mode)
        self.allow_multiple_reduction = bool(allow_multiple_reduction)
        self.prefer_overlap_with_compute = bool(prefer_overlap_with_compute)
        self._v2_efa_window: Optional[torch.Tensor] = None
        self._v2_efa_connection = None
        self._v2_efa_window_bytes = 0
        self._v2_efa_num_lanes = 1

    def _make_runtime(
        self,
        num_experts: int,
        num_topk: int,
        hidden: int,
        elem_bytes: int,
        num_sms: int,
    ):
        config = ep.V2EfaRuntimeConfig()
        config.rank = self.rank_idx
        config.world_size = self.num_ranks
        config.scaleout_rank = self.scaleout_rank_idx
        config.scaleup_rank = self.scaleup_rank_idx
        config.num_scaleout_ranks = self.num_scaleout_ranks
        config.num_scaleup_ranks = self.num_scaleup_ranks
        config.num_experts = int(num_experts)
        config.num_topk = int(num_topk)
        config.hidden = int(hidden)
        config.elem_bytes = int(elem_bytes)
        config.num_sms = int(num_sms)
        return ep.V2EfaRuntime(config)

    def configure_native_v2(
        self,
        num_experts: int,
        num_topk: Optional[int] = None,
        hidden: Optional[int] = None,
        elem_bytes: Optional[int] = None,
        num_sms: int = 0,
    ) -> None:
        self.num_experts = int(num_experts)
        self.num_topk = int(self.num_topk if num_topk is None else num_topk)
        self.hidden = int(self.hidden if hidden is None else hidden)
        self.elem_bytes = int(self.elem_bytes if elem_bytes is None else elem_bytes)
        self.num_sms = int(num_sms)
        self.runtime = self._make_runtime(
            self.num_experts,
            max(1, self.num_topk),
            self.hidden,
            self.elem_bytes,
            self.num_sms,
        )

    def destroy(self) -> None:
        self._destroyed = True

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        if with_cpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier(self.group)
        if with_cpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_logical_domain_size(self) -> Tuple[int, int]:
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def get_physical_domain_size(self) -> Tuple[int, int]:
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = 200,
        sm_write_gbs: float = 50,
    ) -> int:
        del num_experts, num_topk, num_scaleout_topk, rdma_gbs, nvlink_gbs
        del sm_read_gbs, sm_write_gbs
        if self.num_sms > 0:
            return self.num_sms
        if not torch.cuda.is_available():
            return 4
        return min(20 if self.num_scaleout_ranks > 1 else 64,
                   torch.cuda.get_device_properties("cuda").multi_processor_count)

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        if self.allow_hybrid_mode:
            return min(int(num_sms) * 16 + 1, self.num_allocated_qps)
        return min(int(num_sms), self.num_allocated_qps)

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
        return EventOverlap(None)

    def get_native_v2_status(self) -> str:
        return self.runtime.status()

    def get_native_v2_workspace_plan(self, num_max_tokens_per_rank: Optional[int] = None):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        return self.runtime.workspace_plan(tokens)

    def route_expert(self, expert_id: int):
        return self.runtime.route_expert(int(expert_id))

    def compile_dispatch_descriptor_enqueue_d2h_jit(
        self,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels_per_sm: int = 1,
        scale_bytes: int = 0,
        has_topk_weight: bool = True,
        cached_mode: bool = False,
        deterministic: bool = False,
        do_cpu_sync: bool = False,
        smem_bytes: int = 228 * 1024,
        uccl_include_path: str = "",
    ):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_dispatch_descriptor_enqueue_d2h_jit(
            tokens,
            int(num_channels_per_sm),
            int(scale_bytes),
            bool(has_topk_weight),
            bool(cached_mode),
            bool(deterministic),
            bool(do_cpu_sync),
            int(smem_bytes),
            str(uccl_include_path),
        )

    def compile_dispatch_forward_metadata_jit(
        self,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels_per_sm: int = 1,
        uccl_include_path: str = "",
    ):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_dispatch_forward_metadata_jit(
            tokens,
            int(num_channels_per_sm),
            str(uccl_include_path),
        )

    def compile_dispatch_receiver_metadata_jit(
        self,
        num_max_tokens_per_rank: Optional[int] = None,
        uccl_include_path: str = "",
    ):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_dispatch_receiver_metadata_jit(
            tokens,
            str(uccl_include_path),
        )

    def compile_combine_descriptor_enqueue_d2h_jit(
        self,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels: int = 1,
        payload_bytes: int = 0,
        use_expanded_layout: bool = True,
        allow_multiple_reduction: bool = True,
        smem_bytes: int = 228 * 1024,
        uccl_include_path: str = "",
    ):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_combine_descriptor_enqueue_d2h_jit(
            tokens,
            int(num_channels),
            int(payload_bytes),
            bool(use_expanded_layout),
            bool(allow_multiple_reduction),
            int(smem_bytes),
            str(uccl_include_path),
        )

    def compile_combine_forward_metadata_enqueue_d2h_jit(
        self,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels: int = 1,
        payload_bytes: int = 0,
        use_expanded_layout: bool = True,
        allow_multiple_reduction: bool = True,
        smem_bytes: int = 0,
        uccl_include_path: str = "",
    ):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_combine_forward_metadata_enqueue_d2h_jit(
            tokens,
            int(num_channels),
            int(payload_bytes),
            bool(use_expanded_layout),
            bool(allow_multiple_reduction),
            int(smem_bytes),
            str(uccl_include_path),
        )

    @staticmethod
    def allocate_d2h_queue(capacity: int = 2048):
        return ep.V2MappedD2HQueue(int(capacity))

    def init_native_v2_efa_transport(
        self,
        window: Optional[torch.Tensor] = None,
        num_bytes: Optional[int] = None,
        num_lanes: int = 1,
        device_index: int = -1,
        signal_capacity: int = 65536,
    ):
        """Create the V2-only EFA verbs connection for the native backend.

        The connection registers one caller-owned V2 RDMA window and exchanges
        endpoint metadata through the existing torch distributed group. It does
        not instantiate the removed V1 proxy or old TransferCmd path.
        """

        if not hasattr(ep, "V2EfaConnection"):
            raise RuntimeError("uccl.ep was built without V2 EFA verbs connection support")
        if window is None:
            bytes_to_alloc = int(self.num_bytes if num_bytes is None else num_bytes)
            window = torch.empty((bytes_to_alloc,), dtype=torch.uint8, device="cuda")
        _require_cuda_contiguous(window, "window")
        bytes_in_window = int(window.numel() * window.element_size())
        if num_bytes is not None and int(num_bytes) > bytes_in_window:
            raise ValueError("num_bytes exceeds the provided V2 EFA window")
        if device_index < 0 and "UCCL_V2_EFA_DEVICE_INDEX" not in os.environ:
            local_rank = int(os.environ.get("LOCAL_RANK", self.scaleup_rank_idx))
            efa_stride = int(os.environ.get("UCCL_V2_EFA_DEVICE_STRIDE", "2"))
            efa_offset = int(os.environ.get("UCCL_V2_EFA_DEVICE_OFFSET", "0"))
            device_index = efa_offset + local_rank * max(1, efa_stride)

        connection = ep.V2EfaConnection(
            int(window.data_ptr()),
            int(bytes_in_window if num_bytes is None else num_bytes),
            int(self.num_ranks),
            int(self.rank_idx),
            int(max(1, num_lanes)),
            int(device_index),
            int(signal_capacity),
        )
        local_info = connection.local_info()
        all_infos = [None for _ in range(self.num_ranks)]
        dist.all_gather_object(all_infos, local_info, group=self.group)
        connection.connect(all_infos)
        self._v2_efa_window = window
        self._v2_efa_connection = connection
        self._v2_efa_window_bytes = int(bytes_in_window if num_bytes is None else num_bytes)
        self._v2_efa_num_lanes = int(max(1, num_lanes))
        return local_info

    def _require_v2_efa_window(self, required_bytes: int) -> torch.Tensor:
        if self._v2_efa_window is None or self._v2_efa_connection is None:
            raise RuntimeError("native V2 EFA transport has not been initialized")
        if int(required_bytes) > self._v2_efa_window_bytes:
            raise RuntimeError(
                f"native V2 EFA window is too small: need {int(required_bytes)} bytes, "
                f"have {self._v2_efa_window_bytes}"
            )
        return self._v2_efa_window

    def _make_dispatch_window_layout(
        self,
        num_tokens: int,
        num_max_tokens_per_rank: int,
        payload_bytes: int,
        max_batches: int,
        num_topk: int,
        has_topk_weight: bool,
    ) -> dict:
        record_payload_offset = 0
        record_src_global_offset = _align(int(payload_bytes), 4)
        record_topk_idx_offset = _align(record_src_global_offset + 4, 8)
        record_topk_weight_offset = _align(record_topk_idx_offset + int(num_topk) * 8, 4)
        record_topk_weight_bytes = int(num_topk) * 4 if has_topk_weight else 0
        record_end = (
            record_topk_weight_offset + record_topk_weight_bytes
            if has_topk_weight else
            record_topk_idx_offset + int(num_topk) * 8
        )
        token_record_bytes = _align(record_end, 16)
        src_bytes = _align(int(num_tokens) * int(num_topk) * int(token_record_bytes), 64)
        max_records_per_source = int(num_max_tokens_per_rank) * int(self.num_topk)
        batch_payload_stride = _align(int(num_max_tokens_per_rank) * int(token_record_bytes), 64)
        source_rank_stride = _align(max_records_per_source * int(token_record_bytes), 64)
        remote_payload_bytes = int(self.num_ranks) * source_rank_stride
        remote_payload_base = src_bytes
        remote_signal_base = _align(remote_payload_base + remote_payload_bytes, 64)
        # The last slot is a per-source "done" marker. Receivers spin on it on
        # GPU before reading the batch count table, replacing the old post-RDMA
        # CPU barrier.
        source_signal_stride = _align((int(max_batches) + 1) * 4, 64)
        total_bytes = _align(remote_signal_base + int(self.num_ranks) * source_signal_stride, 64)
        self._require_v2_efa_window(total_bytes)
        return {
            "local_payload_base": 0,
            "remote_payload_base": remote_payload_base,
            "remote_signal_base": remote_signal_base,
            "src_token_stride": int(token_record_bytes),
            "expanded_slot_stride": int(token_record_bytes),
            "batch_payload_stride": batch_payload_stride,
            "source_rank_stride": source_rank_stride,
            "source_signal_stride": source_signal_stride,
            "signal_stride": 4,
            "token_record_bytes": int(token_record_bytes),
            "record_payload_offset": int(record_payload_offset),
            "record_src_global_offset": int(record_src_global_offset),
            "record_topk_idx_offset": int(record_topk_idx_offset),
            "record_topk_weight_offset": int(record_topk_weight_offset),
            "record_topk_weight_bytes": int(record_topk_weight_bytes),
            "descriptor_batched": True,
            "max_batches": int(max_batches),
            "done_signal_index": int(max_batches),
            "num_efa_lanes": int(max(1, self._v2_efa_num_lanes)),
            "src_payload_bytes": src_bytes,
            "remote_payload_bytes": remote_payload_bytes,
            "total_window_bytes": total_bytes,
        }

    def _clear_dispatch_receive_window(self, layout: dict) -> None:
        if self._v2_efa_window is None:
            return
        remote_signal_base = int(layout["remote_signal_base"])
        signal_bytes = int(self.num_ranks) * int(layout["source_signal_stride"])
        window = self._require_v2_efa_window(remote_signal_base + signal_bytes)
        window[remote_signal_base: remote_signal_base + signal_bytes].zero_()

    def has_native_v2_efa_transport(self) -> bool:
        return self._v2_efa_connection is not None

    def drain_native_v2_dispatch_transport(
        self,
        handle: EPHandle,
        coalesce: bool = True,
        ack_after_drain: bool = True,
    ):
        if self._v2_efa_connection is None:
            raise RuntimeError("native V2 EFA transport has not been initialized")
        transport = handle.transport_handle
        if transport is None:
            raise RuntimeError("dispatch handle does not contain V2 transport metadata")
        return self._v2_efa_connection.drain_queue(
            transport.d2h_queue, bool(coalesce), bool(ack_after_drain)
        )

    def drain_native_v2_combine_transport(
        self,
        handle: EPHandle,
        coalesce: bool = True,
        ack_after_drain: bool = True,
    ):
        if self._v2_efa_connection is None:
            raise RuntimeError("native V2 EFA transport has not been initialized")
        transport = handle.transport_handle
        if transport is None or transport.combine_d2h_queue is None:
            raise RuntimeError("combine handle does not contain V2 transport metadata")
        return self._v2_efa_connection.drain_queue(
            transport.combine_d2h_queue, bool(coalesce), bool(ack_after_drain)
        )

    def launch_dispatch_descriptor_enqueue_d2h_queue(
        self,
        x_tensor: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        segments: torch.Tensor,
        batches: torch.Tensor,
        route_offsets: torch.Tensor,
        counters: torch.Tensor,
        queue,
        layout: dict,
        num_tokens: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels_per_sm: int = 1,
        scale_bytes: int = 0,
        has_topk_weight: bool = True,
        cached_mode: bool = False,
        deterministic: bool = False,
        do_cpu_sync: bool = False,
        smem_bytes: int = 228 * 1024,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if self._v2_efa_window is None:
            raise RuntimeError("native V2 EFA window is not initialized")
        _require_cuda_contiguous(x_tensor, "x")
        _require_cuda_contiguous(topk_idx, "topk_idx")
        if topk_weights is not None:
            _require_cuda_contiguous(topk_weights, "topk_weights")
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(route_offsets, "route_offsets")
        _require_cuda_contiguous(counters, "counters")
        if route_offsets.dtype != torch.int32:
            raise TypeError("route_offsets must be torch.int32")
        tokens = int(topk_idx.shape[0] if num_tokens is None else num_tokens)
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_descriptor_enqueue_d2h(
            int(x_tensor.data_ptr()),
            int(topk_idx.data_ptr()),
            0 if topk_weights is None else int(topk_weights.data_ptr()),
            int(self._v2_efa_window.data_ptr()),
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(route_offsets.data_ptr()),
            int(counters.data_ptr()),
            tokens,
            max_tokens,
            int(num_channels_per_sm),
            int(scale_bytes),
            bool(has_topk_weight),
            bool(cached_mode),
            bool(deterministic),
            bool(do_cpu_sync),
            int(smem_bytes),
            int(queue.commands_ptr()),
            int(queue.head_ptr()),
            int(queue.tail_ptr()),
            int(queue.capacity()),
            int(layout.get("local_payload_base", 0)),
            int(layout.get("remote_payload_base", 0)),
            int(layout.get("remote_signal_base", 0)),
            int(layout["src_token_stride"]),
            int(layout["expanded_slot_stride"]),
            int(layout["batch_payload_stride"]),
            int(layout.get("source_rank_stride", 0)),
            int(layout.get("source_signal_stride", 0)),
            int(layout.get("token_record_bytes", 0)),
            int(layout.get("record_payload_offset", 0)),
            int(layout.get("record_src_global_offset", 0)),
            int(layout.get("record_topk_idx_offset", 0)),
            int(layout.get("record_topk_weight_offset", 0)),
            int(layout.get("record_topk_weight_bytes", 0)),
            int(layout.get("signal_stride", 4)),
            int(layout.get("num_efa_lanes", 1)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_forward_metadata(
        self,
        recv_topk_idx: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        token_metadata_at_forward: torch.Tensor,
        channel_linked_list: torch.Tensor,
        num_recv_tokens: int,
        num_max_tokens_per_rank: int,
        num_channels_per_sm: int,
        rows_per_channel: int,
        do_expand: bool,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(recv_topk_idx, "recv_topk_idx")
        _require_cuda_contiguous(recv_src_metadata, "recv_src_metadata")
        _require_cuda_contiguous(token_metadata_at_forward, "token_metadata_at_forward")
        _require_cuda_contiguous(channel_linked_list, "channel_linked_list")
        if recv_topk_idx.dtype != torch.int64:
            raise TypeError("recv_topk_idx must be torch.int64")
        if recv_src_metadata.dtype != torch.int32:
            raise TypeError("recv_src_metadata must be torch.int32")
        if token_metadata_at_forward.dtype != torch.int32:
            raise TypeError("token_metadata_at_forward must be torch.int32")
        if channel_linked_list.dtype != torch.int32:
            raise TypeError("channel_linked_list must be torch.int32")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_forward_metadata(
            int(recv_topk_idx.data_ptr()),
            int(recv_src_metadata.data_ptr()),
            int(token_metadata_at_forward.data_ptr()),
            int(channel_linked_list.data_ptr()),
            int(num_recv_tokens),
            int(num_max_tokens_per_rank),
            int(num_channels_per_sm),
            int(rows_per_channel),
            bool(do_expand),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_receiver_metadata(
        self,
        recv_topk_idx: torch.Tensor,
        recv_src_global: torch.Tensor,
        recv_counts_per_rank: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        dst_buffer_slot_idx: torch.Tensor,
        psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        expert_counts_aligned: torch.Tensor,
        expert_counts_scratch: torch.Tensor,
        next_expanded_scratch: torch.Tensor,
        num_recv_tokens: int,
        num_source_tokens: int,
        num_max_tokens_per_rank: int,
        expert_alignment: int,
        do_expand: bool,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(recv_topk_idx, "recv_topk_idx")
        _require_cuda_contiguous(recv_src_global, "recv_src_global")
        _require_cuda_contiguous(recv_counts_per_rank, "recv_counts_per_rank")
        _require_cuda_contiguous(recv_src_metadata, "recv_src_metadata")
        _require_cuda_contiguous(dst_buffer_slot_idx, "dst_buffer_slot_idx")
        _require_cuda_contiguous(
            psum_num_recv_tokens_per_scaleup_rank,
            "psum_num_recv_tokens_per_scaleup_rank",
        )
        _require_cuda_contiguous(
            psum_num_recv_tokens_per_expert,
            "psum_num_recv_tokens_per_expert",
        )
        _require_cuda_contiguous(expert_counts_aligned, "expert_counts_aligned")
        _require_cuda_contiguous(expert_counts_scratch, "expert_counts_scratch")
        _require_cuda_contiguous(next_expanded_scratch, "next_expanded_scratch")
        if recv_topk_idx.dtype != torch.int64:
            raise TypeError("recv_topk_idx must be torch.int64")
        for name, tensor in (
            ("recv_src_global", recv_src_global),
            ("recv_counts_per_rank", recv_counts_per_rank),
            ("recv_src_metadata", recv_src_metadata),
            ("dst_buffer_slot_idx", dst_buffer_slot_idx),
            ("psum_num_recv_tokens_per_scaleup_rank", psum_num_recv_tokens_per_scaleup_rank),
            ("psum_num_recv_tokens_per_expert", psum_num_recv_tokens_per_expert),
            ("expert_counts_aligned", expert_counts_aligned),
            ("expert_counts_scratch", expert_counts_scratch),
            ("next_expanded_scratch", next_expanded_scratch),
        ):
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be torch.int32")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_receiver_metadata(
            int(recv_topk_idx.data_ptr()),
            int(recv_src_global.data_ptr()),
            int(recv_counts_per_rank.data_ptr()),
            int(recv_src_metadata.data_ptr()),
            int(dst_buffer_slot_idx.data_ptr()),
            int(psum_num_recv_tokens_per_scaleup_rank.data_ptr()),
            int(psum_num_recv_tokens_per_expert.data_ptr()),
            int(expert_counts_aligned.data_ptr()),
            int(expert_counts_scratch.data_ptr()),
            int(next_expanded_scratch.data_ptr()),
            int(num_recv_tokens),
            int(num_source_tokens),
            int(num_max_tokens_per_rank),
            int(expert_alignment),
            bool(do_expand),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_materialize_records(
        self,
        batch_counts: torch.Tensor,
        batch_offsets: torch.Tensor,
        recv_x: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: Optional[torch.Tensor],
        recv_src_global: torch.Tensor,
        layout: dict,
        max_batches: int,
        num_max_tokens_per_rank: int,
        has_topk_weight: bool,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if self._v2_efa_window is None:
            raise RuntimeError("native V2 EFA window is not initialized")
        _require_cuda_contiguous(batch_counts, "batch_counts")
        _require_cuda_contiguous(batch_offsets, "batch_offsets")
        _require_cuda_contiguous(recv_x, "recv_x")
        _require_cuda_contiguous(recv_topk_idx, "recv_topk_idx")
        _require_cuda_contiguous(recv_src_global, "recv_src_global")
        if recv_topk_weights is not None:
            _require_cuda_contiguous(recv_topk_weights, "recv_topk_weights")
        if batch_counts.dtype != torch.int32 or batch_offsets.dtype != torch.int32:
            raise TypeError("batch_counts and batch_offsets must be torch.int32")
        if recv_topk_idx.dtype != torch.int64:
            raise TypeError("recv_topk_idx must be torch.int64")
        if recv_src_global.dtype != torch.int32:
            raise TypeError("recv_src_global must be torch.int32")
        if recv_topk_weights is not None and recv_topk_weights.dtype != torch.float32:
            raise TypeError("recv_topk_weights must be torch.float32")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_materialize_records(
            int(self._v2_efa_window.data_ptr()),
            int(batch_counts.data_ptr()),
            int(batch_offsets.data_ptr()),
            int(recv_x.data_ptr()),
            int(recv_topk_idx.data_ptr()),
            0 if recv_topk_weights is None else int(recv_topk_weights.data_ptr()),
            int(recv_src_global.data_ptr()),
            int(max_batches),
            int(num_max_tokens_per_rank),
            int(layout.get("local_payload_base", 0)),
            int(layout["remote_payload_base"]),
            int(layout["remote_signal_base"]),
            int(layout["src_token_stride"]),
            int(layout["expanded_slot_stride"]),
            int(layout["batch_payload_stride"]),
            int(layout["source_rank_stride"]),
            int(layout["source_signal_stride"]),
            int(layout["token_record_bytes"]),
            int(layout.get("record_payload_offset", 0)),
            int(layout["record_src_global_offset"]),
            int(layout["record_topk_idx_offset"]),
            int(layout.get("record_topk_weight_offset", 0)),
            int(layout.get("record_topk_weight_bytes", 0)),
            int(layout.get("signal_stride", 4)),
            bool(has_topk_weight),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_signal_offsets(
        self,
        batch_counts: torch.Tensor,
        batch_offsets: torch.Tensor,
        recv_counts_per_rank: torch.Tensor,
        total_recv_tokens: torch.Tensor,
        layout: dict,
        max_batches: int,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if self._v2_efa_window is None:
            raise RuntimeError("native V2 EFA window is not initialized")
        for name, tensor in (
            ("batch_counts", batch_counts),
            ("batch_offsets", batch_offsets),
            ("recv_counts_per_rank", recv_counts_per_rank),
            ("total_recv_tokens", total_recv_tokens),
        ):
            _require_cuda_contiguous(tensor, name)
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be torch.int32")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_signal_offsets(
            int(self._v2_efa_window.data_ptr()),
            int(batch_counts.data_ptr()),
            int(batch_offsets.data_ptr()),
            int(recv_counts_per_rank.data_ptr()),
            int(total_recv_tokens.data_ptr()),
            int(max_batches),
            int(layout.get("local_payload_base", 0)),
            int(layout["remote_payload_base"]),
            int(layout["remote_signal_base"]),
            int(layout["src_token_stride"]),
            int(layout["expanded_slot_stride"]),
            int(layout["batch_payload_stride"]),
            int(layout["source_rank_stride"]),
            int(layout["source_signal_stride"]),
            int(layout["token_record_bytes"]),
            int(layout.get("signal_stride", 4)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_expand_records(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: Optional[torch.Tensor],
        recv_src_metadata: torch.Tensor,
        expanded_x: torch.Tensor,
        expanded_topk_weights: Optional[torch.Tensor],
        num_recv_tokens: int,
        num_expanded_tokens: int,
        num_max_tokens_per_rank: int,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(recv_x, "recv_x")
        _require_cuda_contiguous(recv_src_metadata, "recv_src_metadata")
        _require_cuda_contiguous(expanded_x, "expanded_x")
        if recv_topk_weights is not None:
            _require_cuda_contiguous(recv_topk_weights, "recv_topk_weights")
        if expanded_topk_weights is not None:
            _require_cuda_contiguous(expanded_topk_weights, "expanded_topk_weights")
        if recv_src_metadata.dtype != torch.int32:
            raise TypeError("recv_src_metadata must be torch.int32")
        if recv_topk_weights is not None and recv_topk_weights.dtype != torch.float32:
            raise TypeError("recv_topk_weights must be torch.float32")
        if expanded_topk_weights is not None and expanded_topk_weights.dtype != torch.float32:
            raise TypeError("expanded_topk_weights must be torch.float32")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_expand_records(
            int(recv_x.data_ptr()),
            0 if recv_topk_weights is None else int(recv_topk_weights.data_ptr()),
            int(recv_src_metadata.data_ptr()),
            int(expanded_x.data_ptr()),
            0 if expanded_topk_weights is None else int(expanded_topk_weights.data_ptr()),
            int(num_recv_tokens),
            int(num_expanded_tokens),
            bool(recv_topk_weights is not None and expanded_topk_weights is not None),
            int(num_max_tokens_per_rank),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_combine_descriptor_enqueue_d2h_queue(
        self,
        dispatch_segments: torch.Tensor,
        dispatch_batches: torch.Tensor,
        num_dispatch_batches: int,
        segments: torch.Tensor,
        batches: torch.Tensor,
        counters: torch.Tensor,
        queue,
        layout: dict,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels: int = 1,
        payload_bytes: int = 0,
        use_expanded_layout: bool = True,
        allow_multiple_reduction: bool = True,
        smem_bytes: int = 228 * 1024,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(dispatch_segments, "dispatch_segments")
        _require_cuda_contiguous(dispatch_batches, "dispatch_batches")
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(counters, "counters")
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if payload_bytes == 0:
            payload_bytes = self.hidden * 2
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_combine_descriptor_enqueue_d2h(
            int(dispatch_segments.data_ptr()),
            int(dispatch_batches.data_ptr()),
            int(num_dispatch_batches),
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(counters.data_ptr()),
            max_tokens,
            int(num_channels),
            int(payload_bytes),
            bool(use_expanded_layout),
            bool(allow_multiple_reduction),
            int(smem_bytes),
            int(queue.commands_ptr()),
            int(queue.head_ptr()),
            int(queue.tail_ptr()),
            int(queue.capacity()),
            int(layout.get("local_payload_base", 0)),
            int(layout.get("remote_payload_base", 0)),
            int(layout.get("remote_signal_base", 0)),
            int(layout["expanded_slot_stride"]),
            int(layout["reduced_token_stride"]),
            int(layout["batch_payload_stride"]),
            int(layout.get("signal_stride", 4)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_combine_forward_metadata_enqueue_d2h_queue(
        self,
        forward_metadata: torch.Tensor,
        channel_linked_list: torch.Tensor,
        segments: torch.Tensor,
        batches: torch.Tensor,
        counters: torch.Tensor,
        queue,
        layout: dict,
        num_forward_rows: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels: int = 1,
        payload_bytes: int = 0,
        use_expanded_layout: bool = True,
        allow_multiple_reduction: bool = True,
        smem_bytes: int = 0,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(forward_metadata, "forward_metadata")
        _require_cuda_contiguous(channel_linked_list, "channel_linked_list")
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(counters, "counters")
        if forward_metadata.dtype != torch.int32:
            raise TypeError("forward_metadata must be torch.int32")
        if channel_linked_list.dtype != torch.int32:
            raise TypeError("channel_linked_list must be torch.int32")
        rows = int(forward_metadata.shape[0] if num_forward_rows is None else num_forward_rows)
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if payload_bytes == 0:
            payload_bytes = self.hidden * self.elem_bytes
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_combine_forward_metadata_enqueue_d2h(
            int(forward_metadata.data_ptr()),
            int(channel_linked_list.data_ptr()),
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(counters.data_ptr()),
            rows,
            max_tokens,
            int(num_channels),
            int(payload_bytes),
            bool(use_expanded_layout),
            bool(allow_multiple_reduction),
            int(smem_bytes),
            int(queue.commands_ptr()),
            int(queue.head_ptr()),
            int(queue.tail_ptr()),
            int(queue.capacity()),
            int(layout.get("local_payload_base", 0)),
            int(layout.get("remote_payload_base", 0)),
            int(layout.get("remote_signal_base", 0)),
            int(layout["expanded_slot_stride"]),
            int(layout["reduced_token_stride"]),
            int(layout["batch_payload_stride"]),
            int(layout.get("signal_stride", 4)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def get_comm_stream(self) -> torch.Stream:
        return torch.cuda.current_stream()

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
        previous_event: Optional[object] = None,
        previous_event_before_epilogue: Optional[object] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: bool = False,
        use_tma_aligned_col_major_sf: bool = False,
    ):
        del num_qps, previous_event, previous_event_before_epilogue
        del async_with_compute_stream, allocate_on_comm_stream
        del use_tma_aligned_col_major_sf

        if handle is not None:
            if topk_idx is not None or topk_weights is not None:
                raise ValueError("topk_idx/topk_weights must be None when cached handle is used")
            topk_idx = handle.topk_idx
            num_experts = handle.num_experts
            num_max_tokens_per_rank = handle.num_max_tokens_per_rank
            expert_alignment = handle.expert_alignment
            do_expand = handle.do_expand
            do_cpu_sync = False if do_cpu_sync is None else do_cpu_sync

        if topk_idx is None:
            raise ValueError("topk_idx is required for uncached native V2 dispatch")
        if self._v2_efa_connection is None:
            raise RuntimeError("native V2 dispatch requires init_native_v2_efa_transport()")
        _require_cuda_contiguous(topk_idx, "topk_idx")
        if topk_idx.dtype != torch.int64:
            raise TypeError("topk_idx must be torch.int64")

        x_tensor, sf = x if isinstance(x, tuple) else (x, None)
        _require_cuda_contiguous(x_tensor, "x")
        if sf is not None:
            _require_cuda_contiguous(sf, "sf")
        if topk_weights is not None:
            _require_cuda_contiguous(topk_weights, "topk_weights")

        num_tokens, hidden = x_tensor.shape
        num_topk = topk_idx.shape[1]
        num_experts = int(num_experts if num_experts is not None else self.num_experts)
        num_max_tokens_per_rank = int(
            num_max_tokens_per_rank
            if num_max_tokens_per_rank is not None
            else self.num_max_tokens_per_rank
        )
        expert_alignment = int(expert_alignment if expert_alignment is not None else 1)
        do_cpu_sync = True if do_cpu_sync is None else bool(do_cpu_sync)
        num_sms = int(num_sms or self.get_theoretical_num_sms(num_experts, num_topk))
        elem_bytes = int(x_tensor.element_size())
        payload_bytes = int(hidden * elem_bytes)
        scale_bytes = 0 if sf is None else int(sf.shape[1] * sf.element_size())
        self.configure_native_v2(num_experts, num_topk, hidden, elem_bytes, num_sms)

        transport = self._launch_native_dispatch_transport(
            x_tensor=x_tensor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            payload_bytes=payload_bytes,
            scale_bytes=scale_bytes,
            has_topk_weight=topk_weights is not None,
            do_cpu_sync=do_cpu_sync,
        )

        recv_x, recv_sf, recv_topk_idx, recv_topk_weights, recv_src_global, recv_counts = (
            self._materialize_native_dispatch_from_window(
                transport,
                x_tensor,
                topk_weights,
                num_experts,
                num_max_tokens_per_rank,
            )
        )
        num_recv_tokens = int(sum(recv_counts))
        metadata_begin = time.perf_counter()
        recv_src_metadata, dst_buffer_slot_idx, psum_scaleup, psum_expert, expert_counts = (
            self._build_v2_dispatch_metadata(
                recv_topk_idx, recv_src_global, topk_idx, recv_counts,
                num_experts, num_max_tokens_per_rank, expert_alignment,
                do_expand,
            )
        )
        token_metadata_at_forward, channel_linked_list = self._build_forward_metadata_tensors(
            recv_topk_idx,
            recv_src_metadata,
            do_expand,
            num_experts,
            num_max_tokens_per_rank,
        )
        if transport.timings is not None:
            transport.timings["metadata_ms"] = (time.perf_counter() - metadata_begin) * 1000.0

        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.add_(
                torch.tensor(expert_counts, dtype=cumulative_local_expert_recv_stats.dtype,
                             device=cumulative_local_expert_recv_stats.device)
            )

        if do_expand:
            expand_begin = time.perf_counter()
            expanded_tokens = int(psum_expert[-1].item()) if psum_expert.numel() else 0
            expanded_x = torch.empty((max(expanded_tokens, 1), hidden),
                                     dtype=x_tensor.dtype, device=x_tensor.device)
            expanded_x.zero_()
            expanded_weights = None
            if topk_weights is not None:
                expanded_weights = torch.empty((max(expanded_tokens, 1), num_topk),
                                               dtype=topk_weights.dtype,
                                               device=topk_weights.device)
                expanded_weights.zero_()
            self.launch_dispatch_expand_records(
                recv_x=recv_x,
                recv_topk_weights=recv_topk_weights,
                recv_src_metadata=recv_src_metadata,
                expanded_x=expanded_x,
                expanded_topk_weights=expanded_weights,
                num_recv_tokens=num_recv_tokens,
                num_expanded_tokens=expanded_tokens,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
            )
            recv_x_out = expanded_x
            recv_topk_idx_out = None
            recv_topk_weights_out = expanded_weights
            if transport.timings is not None:
                torch.cuda.current_stream().synchronize()
                transport.timings["expand_ms"] = (time.perf_counter() - expand_begin) * 1000.0
        else:
            recv_x_out = recv_x
            recv_topk_idx_out = recv_topk_idx
            recv_topk_weights_out = recv_topk_weights

        if sf is not None:
            recv_x_out = (recv_x_out, recv_sf if not do_expand else None)

        cloned_topk_idx = topk_idx.clone() if do_handle_copy else topk_idx
        new_handle = EPHandle(
            do_expand=do_expand,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_sms=num_sms,
            topk_idx=cloned_topk_idx,
            num_recv_tokens_per_expert_list=expert_counts,
            psum_num_recv_tokens_per_scaleup_rank=psum_scaleup,
            psum_num_recv_tokens_per_expert=psum_expert,
            recv_src_metadata=recv_src_metadata,
            dst_buffer_slot_idx=dst_buffer_slot_idx,
            token_metadata_at_forward=token_metadata_at_forward,
            channel_linked_list=channel_linked_list,
            transport_handle=transport,
        )

        return recv_x_out, recv_topk_idx_out, recv_topk_weights_out, new_handle, EventOverlap(None)

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Optional[object] = None,
        previous_event_before_epilogue: Optional[object] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        del num_sms, num_qps, previous_event, previous_event_before_epilogue
        del async_with_compute_stream, allocate_on_comm_stream
        _require_cuda_contiguous(x, "x")
        if topk_weights is not None:
            _require_cuda_contiguous(topk_weights, "topk_weights")
        del bias
        raise NotImplementedError(
            "native V2 combine is intentionally not wired through a semantic "
            "all-to-all path; finish the V2 combine descriptor/data path first"
        )

    def _launch_native_dispatch_transport(
        self,
        x_tensor: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        num_tokens: int,
        num_max_tokens_per_rank: int,
        payload_bytes: int,
        scale_bytes: int,
        has_topk_weight: bool,
        do_cpu_sync: bool,
    ) -> V2TransportHandle:
        transport_begin = time.perf_counter()
        timings = {}
        sizes = ep.v2_descriptor_sizes()
        max_segments = max(1, int(num_tokens * self.num_topk))
        max_batches = max(1, int(self.num_experts))
        segments = torch.empty((max_segments * int(sizes["dispatch_segment"]),),
                               dtype=torch.uint8, device=topk_idx.device)
        batches = torch.empty((max_batches * int(sizes["dispatch_batch"]),),
                              dtype=torch.uint8, device=topk_idx.device)
        route_offsets = torch.empty((max(1, int(num_tokens * self.num_topk)),),
                                    dtype=torch.int32, device=topk_idx.device)
        counters = torch.zeros((3,), dtype=torch.int32, device=topk_idx.device)
        descriptor_queue_capacity = max_segments + max_batches + int(self.num_ranks) + 1
        queue_capacity = _next_power_of_two(descriptor_queue_capacity)
        queue = self.allocate_d2h_queue(queue_capacity)
        if self._v2_efa_connection is None:
            raise RuntimeError("native V2 EFA dispatch requires an initialized EFA connection")
        stage_begin = time.perf_counter()
        layout = self._make_dispatch_window_layout(
            num_tokens=num_tokens,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            payload_bytes=payload_bytes,
            max_batches=max_batches,
            num_topk=int(topk_idx.shape[1]),
            has_topk_weight=has_topk_weight,
        )
        self._clear_dispatch_receive_window(layout)
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=self.group)
        timings["stage_and_pre_barrier_ms"] = (time.perf_counter() - stage_begin) * 1000.0
        flat_topk_idx = topk_idx.reshape(-1)
        enqueue_begin = time.perf_counter()
        self.launch_dispatch_descriptor_enqueue_d2h_queue(
            x_tensor=x_tensor,
            topk_idx=flat_topk_idx,
            topk_weights=topk_weights if has_topk_weight else None,
            segments=segments,
            batches=batches,
            route_offsets=route_offsets,
            counters=counters,
            queue=queue,
            layout=layout,
            num_tokens=num_tokens,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            scale_bytes=scale_bytes,
            has_topk_weight=has_topk_weight,
            do_cpu_sync=do_cpu_sync,
            smem_bytes=0,
        )
        torch.cuda.current_stream().synchronize()
        timings["descriptor_enqueue_ms"] = (time.perf_counter() - enqueue_begin) * 1000.0
        num_segments = int(counters[0].item())
        num_batches = int(counters[1].item())
        dispatch_drain_stats = None
        drain_begin = time.perf_counter()
        dispatch_drain_stats = self._v2_efa_connection.drain_queue(queue, True, True)
        timings["proxy_drain_ms"] = (time.perf_counter() - drain_begin) * 1000.0
        wait_begin = time.perf_counter()
        self._wait_native_v2_efa_completions(dispatch_drain_stats)
        timings["completion_wait_ms"] = (time.perf_counter() - wait_begin) * 1000.0
        timings["post_barrier_ms"] = 0.0
        timings["transport_total_ms"] = (time.perf_counter() - transport_begin) * 1000.0
        return V2TransportHandle(
            dispatch_segments=segments,
            dispatch_batches=batches,
            dispatch_route_offsets=route_offsets,
            dispatch_counters=counters,
            combine_segments=None,
            combine_batches=None,
            combine_counters=None,
            d2h_queue=queue,
            combine_d2h_queue=None,
            dispatch_layout=layout,
            combine_layout=None,
            num_dispatch_batches=num_batches,
            num_dispatch_segments=num_segments,
            num_combine_batches=0,
            num_combine_segments=0,
            payload_bytes=payload_bytes,
            scale_bytes=scale_bytes,
            dispatch_drain_stats=dispatch_drain_stats,
            timings=timings,
        )

    def _wait_native_v2_efa_completions(self, stats: Optional[dict]) -> None:
        if self._v2_efa_connection is None or stats is None:
            return
        expected = int(stats.get("posted_writes", 0)) + int(stats.get("posted_signals", 0))
        seen = 0
        deadline = time.perf_counter() + 5.0
        spins = 0
        while time.perf_counter() < deadline:
            seen += int(self._v2_efa_connection.poll_completions(max(1, expected - seen)))
            if seen >= expected:
                return
            spins += 1
            if spins % 4096 == 0:
                time.sleep(0)
        raise TimeoutError(f"timed out waiting for {expected} V2 EFA completions, saw {seen}")

    def _native_dispatch_batch_counts_from_window(
        self,
        layout: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, list, int, float]:
        if self._v2_efa_window is None:
            raise RuntimeError("native V2 EFA window is not initialized")
        max_batches = int(layout["max_batches"])
        device = self._v2_efa_window.device
        counts_tensor = torch.empty(
            (int(self.num_ranks), max_batches), dtype=torch.int32, device=device
        )
        offsets_tensor = torch.empty_like(counts_tensor)
        recv_counts_tensor = torch.empty((int(self.num_ranks),), dtype=torch.int32, device=device)
        total_recv_tensor = torch.empty((1,), dtype=torch.int32, device=device)
        begin = time.perf_counter()
        self.launch_dispatch_signal_offsets(
            batch_counts=counts_tensor,
            batch_offsets=offsets_tensor,
            recv_counts_per_rank=recv_counts_tensor,
            total_recv_tokens=total_recv_tensor,
            layout=layout,
            max_batches=max_batches,
        )
        torch.cuda.current_stream().synchronize()
        signal_offsets_ms = (time.perf_counter() - begin) * 1000.0
        recv_counts = [int(v) for v in recv_counts_tensor.cpu().tolist()]
        total_recv = int(total_recv_tensor.item())
        return counts_tensor, offsets_tensor, recv_counts, total_recv, signal_offsets_ms

    def _materialize_native_dispatch_from_window(
        self,
        transport: V2TransportHandle,
        x_tensor: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        num_experts: int,
        num_max_tokens_per_rank: int,
    ):
        if self._v2_efa_connection is None or self._v2_efa_window is None:
            raise RuntimeError("native V2 EFA transport has not been initialized")
        layout = transport.dispatch_layout
        batch_counts, batch_offsets, recv_counts, num_recv_tokens, signal_offsets_ms = (
            self._native_dispatch_batch_counts_from_window(layout)
        )
        if transport.timings is not None:
            transport.timings["signal_offsets_ms"] = signal_offsets_ms
        recv_x = torch.empty(
            (max(1, num_recv_tokens), int(x_tensor.shape[1])),
            dtype=x_tensor.dtype,
            device=x_tensor.device,
        )
        recv_topk_idx = torch.empty(
            (max(1, num_recv_tokens), int(self.num_topk)),
            dtype=torch.int64,
            device=x_tensor.device,
        )
        recv_topk_weights = None
        if topk_weights is not None:
            recv_topk_weights = torch.empty(
                (max(1, num_recv_tokens), int(self.num_topk)),
                dtype=torch.float32,
                device=x_tensor.device,
            )
        recv_src_global = torch.empty(
            (max(1, num_recv_tokens),),
            dtype=torch.int32,
            device=x_tensor.device,
        )
        if num_recv_tokens == 0:
            recv_x.zero_()
            recv_topk_idx.fill_(-1)
            if recv_topk_weights is not None:
                recv_topk_weights.zero_()
            recv_src_global.zero_()
            return recv_x[:0], None, recv_topk_idx[:0], recv_topk_weights, recv_src_global[:0], recv_counts

        materialize_begin = time.perf_counter()
        self.launch_dispatch_materialize_records(
            batch_counts=batch_counts,
            batch_offsets=batch_offsets,
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            recv_src_global=recv_src_global,
            layout=layout,
            max_batches=int(layout["max_batches"]),
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            has_topk_weight=topk_weights is not None,
        )
        torch.cuda.current_stream().synchronize()
        if transport.timings is not None:
            transport.timings["materialize_records_ms"] = (
                time.perf_counter() - materialize_begin
            ) * 1000.0
        return (
            recv_x[:num_recv_tokens],
            None,
            recv_topk_idx[:num_recv_tokens],
            None if recv_topk_weights is None else recv_topk_weights[:num_recv_tokens],
            recv_src_global[:num_recv_tokens],
            recv_counts,
        )

    def _build_forward_metadata_tensors(
        self,
        recv_topk_idx: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        do_expand: bool,
        num_experts: int,
        num_max_tokens_per_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del num_experts
        num_recv = int(recv_src_metadata.shape[0])
        num_topk = int(recv_topk_idx.shape[1]) if recv_topk_idx.numel() else int(self.num_topk)
        dims = 2 + num_topk * 2
        num_channels_per_sm = max(1, self.num_scaleup_ranks)
        num_channels = max(1, int(self.num_sms) * num_channels_per_sm)
        tokens_per_channel = max(1, (num_recv + num_channels - 1) // num_channels)
        rows_per_channel = max(1, self.num_scaleout_ranks) * tokens_per_channel + 1
        metadata = torch.empty(
            (num_channels, rows_per_channel, dims),
            dtype=torch.int32,
            device=recv_src_metadata.device,
        )
        channel_linked_list = torch.empty(
            (num_channels, rows_per_channel, max(1, self.num_scaleup_ranks)),
            dtype=torch.int32,
            device=recv_src_metadata.device,
        )
        if num_recv == 0:
            metadata.fill_(-1)
            channel_linked_list.fill_(-1)
            return metadata, channel_linked_list

        self.launch_dispatch_forward_metadata(
            recv_topk_idx=recv_topk_idx,
            recv_src_metadata=recv_src_metadata,
            token_metadata_at_forward=metadata,
            channel_linked_list=channel_linked_list,
            num_recv_tokens=num_recv,
            num_max_tokens_per_rank=(
                self.num_max_tokens_per_rank
                if num_max_tokens_per_rank is None
                else int(num_max_tokens_per_rank)
            ),
            num_channels_per_sm=num_channels_per_sm,
            rows_per_channel=rows_per_channel,
            do_expand=do_expand,
        )
        return metadata, channel_linked_list

    def _build_v2_dispatch_metadata(
        self,
        recv_topk_idx: torch.Tensor,
        recv_src_global: torch.Tensor,
        topk_idx: torch.Tensor,
        recv_counts: list,
        num_experts: int,
        num_max_tokens_per_rank: int,
        expert_alignment: int,
        do_expand: bool,
    ):
        num_recv, num_topk = recv_topk_idx.shape
        num_local_experts = num_experts // self.num_ranks
        device = recv_topk_idx.device
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.int32, device=device)
        metadata = torch.empty((num_recv, 2 + num_topk), dtype=torch.int32, device=device)
        dst_slot = torch.empty(topk_idx.shape, dtype=torch.int32, device=device)
        psum_scaleup = torch.empty((self.num_scaleup_ranks,), dtype=torch.int32, device=device)
        psum_expert = torch.empty((num_local_experts,), dtype=torch.int32, device=device)
        expert_counts_aligned = torch.empty((num_local_experts,), dtype=torch.int32, device=device)
        expert_counts_scratch = torch.empty((num_local_experts,), dtype=torch.int32, device=device)
        next_expanded_scratch = torch.empty((num_local_experts,), dtype=torch.int32, device=device)

        self.launch_dispatch_receiver_metadata(
            recv_topk_idx=recv_topk_idx,
            recv_src_global=recv_src_global.to(torch.int32),
            recv_counts_per_rank=recv_counts_tensor,
            recv_src_metadata=metadata,
            dst_buffer_slot_idx=dst_slot,
            psum_num_recv_tokens_per_scaleup_rank=psum_scaleup,
            psum_num_recv_tokens_per_expert=psum_expert,
            expert_counts_aligned=expert_counts_aligned,
            expert_counts_scratch=expert_counts_scratch,
            next_expanded_scratch=next_expanded_scratch,
            num_recv_tokens=num_recv,
            num_source_tokens=int(topk_idx.shape[0]),
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=expert_alignment,
            do_expand=do_expand,
        )
        expert_counts = [int(v) for v in expert_counts_aligned.detach().cpu().tolist()]
        return metadata, dst_slot, psum_scaleup, psum_expert, expert_counts


def _align_2mb(x: int) -> int:
    return ((int(x) + (1 << 21) - 1) // (1 << 21)) << 21


def _align(x: int, alignment: int) -> int:
    return ((int(x) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _next_power_of_two(x: int) -> int:
    value = 1
    while value < int(x):
        value <<= 1
    return value


def _require_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _cuda_stream_ptr(stream: Optional[torch.cuda.Stream]) -> int:
    if stream is None:
        stream = torch.cuda.current_stream()
    return int(stream.cuda_stream)
