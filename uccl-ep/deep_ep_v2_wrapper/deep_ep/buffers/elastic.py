from __future__ import annotations

import os
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

    def build_reference_dispatch_plan(
        self,
        topk_idx_flat,
        num_tokens: int,
        payload_bytes: int,
        scale_bytes: int = 0,
        has_topk_weight: bool = True,
    ):
        if isinstance(topk_idx_flat, torch.Tensor):
            topk_idx_flat = topk_idx_flat.detach().cpu().reshape(-1).tolist()
        return self.runtime.build_reference_dispatch_plan(
            topk_idx_flat,
            int(num_tokens),
            int(payload_bytes),
            int(scale_bytes),
            bool(has_topk_weight),
        )

    def build_reference_roundtrip_plan(
        self,
        topk_idx_flat,
        num_tokens: int,
        dispatch_payload_bytes: int,
        combine_payload_bytes: int,
        scale_bytes: int = 0,
        has_topk_weight: bool = True,
    ):
        if isinstance(topk_idx_flat, torch.Tensor):
            topk_idx_flat = topk_idx_flat.detach().cpu().reshape(-1).tolist()
        return self.runtime.build_reference_roundtrip_plan(
            topk_idx_flat,
            int(num_tokens),
            int(dispatch_payload_bytes),
            int(scale_bytes),
            int(combine_payload_bytes),
            bool(has_topk_weight),
        )

    def build_reference_transfer_roundtrip_plan(
        self,
        topk_idx_flat,
        num_tokens: int,
        dispatch_payload_bytes: int,
        combine_payload_bytes: int,
        scale_bytes: int = 0,
        has_topk_weight: bool = True,
    ):
        if isinstance(topk_idx_flat, torch.Tensor):
            topk_idx_flat = topk_idx_flat.detach().cpu().reshape(-1).tolist()
        return self.runtime.build_reference_transfer_roundtrip_plan(
            topk_idx_flat,
            int(num_tokens),
            int(dispatch_payload_bytes),
            int(scale_bytes),
            int(combine_payload_bytes),
            bool(has_topk_weight),
        )

    def build_dispatch_jit_plan(
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
        return self.runtime.build_dispatch_jit_plan(
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

    def compile_dispatch_jit(
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
        return self.runtime.compile_dispatch_jit(
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

    def launch_dispatch_descriptors(
        self,
        topk_idx: torch.Tensor,
        segments: torch.Tensor,
        batches: torch.Tensor,
        counters: torch.Tensor,
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
        """Launch the native V2 dispatch descriptor JIT kernel.

        This is the first real DeepEP-JIT launch bridge. It intentionally
        operates on caller-owned descriptor workspaces; the public dispatch API
        will allocate and thread these through the V2 handle once the EFA proxy
        enqueue path is wired behind the same descriptors.
        """

        _require_cuda_contiguous(topk_idx, "topk_idx")
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(counters, "counters")
        if topk_idx.dtype != torch.int64:
            raise TypeError("topk_idx must be torch.int64")
        if counters.dtype not in (torch.int32, torch.uint32):
            raise TypeError("counters must be torch.int32/torch.uint32")

        tokens = int(topk_idx.shape[0] if num_tokens is None else num_tokens)
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_descriptors(
            int(topk_idx.data_ptr()),
            int(segments.data_ptr()),
            int(batches.data_ptr()),
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
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def build_combine_jit_plan(
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
        return self.runtime.build_combine_jit_plan(
            tokens,
            int(num_channels),
            int(payload_bytes),
            bool(use_expanded_layout),
            bool(allow_multiple_reduction),
            int(smem_bytes),
            str(uccl_include_path),
        )

    def compile_combine_jit(
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
        return self.runtime.compile_combine_jit(
            tokens,
            int(num_channels),
            int(payload_bytes),
            bool(use_expanded_layout),
            bool(allow_multiple_reduction),
            int(smem_bytes),
            str(uccl_include_path),
        )

    def build_dispatch_enqueue_d2h_jit_plan(self, uccl_include_path: str = ""):
        return self.runtime.build_dispatch_enqueue_d2h_jit_plan(str(uccl_include_path))

    def compile_dispatch_enqueue_d2h_jit(self, uccl_include_path: str = ""):
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_dispatch_enqueue_d2h_jit(str(uccl_include_path))

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

    def compile_dispatch_direct_enqueue_d2h_jit(
        self,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels_per_sm: int = 1,
        scale_bytes: int = 0,
        has_topk_weight: bool = True,
        cached_mode: bool = False,
        deterministic: bool = False,
        do_cpu_sync: bool = False,
        smem_bytes: int = 0,
        uccl_include_path: str = "",
    ):
        tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_dispatch_direct_enqueue_d2h_jit(
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

    def build_combine_enqueue_d2h_jit_plan(self, uccl_include_path: str = ""):
        return self.runtime.build_combine_enqueue_d2h_jit_plan(str(uccl_include_path))

    def compile_combine_enqueue_d2h_jit(self, uccl_include_path: str = ""):
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        return self.runtime.compile_combine_enqueue_d2h_jit(str(uccl_include_path))

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

    def _stage_tensor_bytes_to_v2_window(self, tensor: torch.Tensor, base: int) -> int:
        _require_cuda_contiguous(tensor, "tensor")
        num_bytes = int(tensor.numel() * tensor.element_size())
        window = self._require_v2_efa_window(int(base) + num_bytes)
        byte_view = tensor.reshape(-1).view(torch.uint8)
        window[int(base): int(base) + num_bytes].copy_(byte_view.reshape(-1))
        return num_bytes

    def _make_dispatch_window_layout(
        self,
        num_tokens: int,
        num_max_tokens_per_rank: int,
        payload_bytes: int,
        max_batches: int,
    ) -> dict:
        src_bytes = _align(int(num_tokens) * int(payload_bytes), 64)
        batch_payload_stride = _align(int(num_max_tokens_per_rank) * int(payload_bytes), 64)
        remote_payload_bytes = int(max_batches) * batch_payload_stride
        remote_payload_base = src_bytes
        remote_signal_base = _align(remote_payload_base + remote_payload_bytes, 64)
        total_bytes = _align(remote_signal_base + int(max_batches) * 4, 64)
        self._require_v2_efa_window(total_bytes)
        return {
            "local_payload_base": 0,
            "remote_payload_base": remote_payload_base,
            "remote_signal_base": remote_signal_base,
            "src_token_stride": int(payload_bytes),
            "expanded_slot_stride": int(payload_bytes),
            "batch_payload_stride": batch_payload_stride,
            "signal_stride": 4,
            "src_payload_bytes": src_bytes,
            "remote_payload_bytes": remote_payload_bytes,
            "total_window_bytes": total_bytes,
        }

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

    def launch_combine_descriptors(
        self,
        dispatch_segments: torch.Tensor,
        dispatch_batches: torch.Tensor,
        num_dispatch_batches: int,
        segments: torch.Tensor,
        batches: torch.Tensor,
        counters: torch.Tensor,
        num_max_tokens_per_rank: Optional[int] = None,
        num_channels: int = 1,
        payload_bytes: int = 0,
        use_expanded_layout: bool = True,
        allow_multiple_reduction: bool = True,
        smem_bytes: int = 228 * 1024,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Launch the native V2 combine descriptor JIT kernel."""

        _require_cuda_contiguous(dispatch_segments, "dispatch_segments")
        _require_cuda_contiguous(dispatch_batches, "dispatch_batches")
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(counters, "counters")
        if counters.dtype not in (torch.int32, torch.uint32):
            raise TypeError("counters must be torch.int32/torch.uint32")

        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if payload_bytes == 0:
            payload_bytes = self.hidden * 2
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_combine_descriptors(
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
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_enqueue_d2h(
        self,
        segments: torch.Tensor,
        batches: torch.Tensor,
        num_batches: int,
        commands: torch.Tensor,
        head: torch.Tensor,
        tail: torch.Tensor,
        layout: dict,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Launch descriptor-to-D2H-command enqueue for dispatch."""

        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(commands, "commands")
        _require_cuda_contiguous(head, "head")
        _require_cuda_contiguous(tail, "tail")
        queue_capacity = _queue_capacity(commands)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_enqueue_d2h(
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(num_batches),
            int(commands.data_ptr()),
            int(head.data_ptr()),
            int(tail.data_ptr()),
            queue_capacity,
            int(layout.get("local_payload_base", 0)),
            int(layout.get("remote_payload_base", 0)),
            int(layout.get("remote_signal_base", 0)),
            int(layout["src_token_stride"]),
            int(layout["expanded_slot_stride"]),
            int(layout["batch_payload_stride"]),
            int(layout.get("signal_stride", 4)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_enqueue_d2h_queue(
        self,
        segments: torch.Tensor,
        batches: torch.Tensor,
        num_batches: int,
        queue,
        layout: dict,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_enqueue_d2h(
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(num_batches),
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
            int(layout.get("signal_stride", 4)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_descriptor_enqueue_d2h_queue(
        self,
        topk_idx: torch.Tensor,
        segments: torch.Tensor,
        batches: torch.Tensor,
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
        _require_cuda_contiguous(topk_idx, "topk_idx")
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(counters, "counters")
        tokens = int(topk_idx.shape[0] if num_tokens is None else num_tokens)
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_descriptor_enqueue_d2h(
            int(topk_idx.data_ptr()),
            int(segments.data_ptr()),
            int(batches.data_ptr()),
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
            int(layout.get("signal_stride", 4)),
            str(uccl_include_path),
            _cuda_stream_ptr(stream),
        )

    def launch_dispatch_direct_enqueue_d2h_queue(
        self,
        topk_idx: torch.Tensor,
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
        smem_bytes: int = 0,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(topk_idx, "topk_idx")
        if topk_idx.dtype != torch.int64:
            raise TypeError("topk_idx must be torch.int64")
        tokens = int(topk_idx.shape[0] if num_tokens is None else num_tokens)
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_dispatch_direct_enqueue_d2h(
            int(topk_idx.data_ptr()),
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
            int(layout.get("signal_stride", 4)),
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

    def launch_combine_enqueue_d2h(
        self,
        segments: torch.Tensor,
        batches: torch.Tensor,
        num_batches: int,
        commands: torch.Tensor,
        head: torch.Tensor,
        tail: torch.Tensor,
        layout: dict,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Launch descriptor-to-D2H-command enqueue for combine."""

        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(commands, "commands")
        _require_cuda_contiguous(head, "head")
        _require_cuda_contiguous(tail, "tail")
        queue_capacity = _queue_capacity(commands)
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_combine_enqueue_d2h(
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(num_batches),
            int(commands.data_ptr()),
            int(head.data_ptr()),
            int(tail.data_ptr()),
            queue_capacity,
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

    def launch_combine_enqueue_d2h_queue(
        self,
        segments: torch.Tensor,
        batches: torch.Tensor,
        num_batches: int,
        queue,
        layout: dict,
        uccl_include_path: str = "",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_combine_enqueue_d2h(
            int(segments.data_ptr()),
            int(batches.data_ptr()),
            int(num_batches),
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
        _require_cuda_contiguous(segments, "segments")
        _require_cuda_contiguous(batches, "batches")
        _require_cuda_contiguous(counters, "counters")
        if forward_metadata.dtype != torch.int32:
            raise TypeError("forward_metadata must be torch.int32")
        rows = int(forward_metadata.shape[0] if num_forward_rows is None else num_forward_rows)
        max_tokens = self.num_max_tokens_per_rank if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
        if payload_bytes == 0:
            payload_bytes = self.hidden * self.elem_bytes
        if not uccl_include_path:
            uccl_include_path = str(Path(__file__).resolve().parents[3] / "include")
        self.runtime.launch_combine_forward_metadata_enqueue_d2h(
            int(forward_metadata.data_ptr()),
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
            num_tokens=num_tokens,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            payload_bytes=payload_bytes,
            scale_bytes=scale_bytes,
            has_topk_weight=topk_weights is not None,
            do_cpu_sync=do_cpu_sync,
        )

        recv_x, recv_sf, recv_topk_idx, recv_topk_weights, recv_src_global, recv_counts = (
            self._semantic_dispatch_data(x_tensor, sf, topk_idx, topk_weights,
                                         num_max_tokens_per_rank, num_experts)
        )
        self._overlay_native_dispatch_payload_from_window(
            recv_x,
            recv_src_global,
            transport,
            num_max_tokens_per_rank,
            payload_bytes,
        )
        num_recv_tokens = int(sum(recv_counts))
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

        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.add_(
                torch.tensor(expert_counts, dtype=cumulative_local_expert_recv_stats.dtype,
                             device=cumulative_local_expert_recv_stats.device)
            )

        if do_expand:
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
            for slot in range(num_topk):
                idx = recv_src_metadata[:num_recv_tokens, 2 + slot]
                mask = idx >= 0
                if mask.any():
                    expanded_x[idx[mask].long()] = recv_x[mask]
                    if expanded_weights is not None:
                        expanded_weights[idx[mask].long(), slot] = recv_topk_weights[mask, slot]
            recv_x_out = expanded_x
            recv_topk_idx_out = None
            recv_topk_weights_out = expanded_weights
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

        self._launch_native_combine_transport(x, handle)

        combined_x = self._semantic_combine_data(x, handle, bias)
        self._overlay_native_combine_payload_from_window(combined_x, handle)
        combined_topk_weights = handle.topk_idx.new_zeros(handle.topk_idx.shape).to(torch.float32)
        if topk_weights is not None and not handle.do_expand:
            combined_topk_weights = self._semantic_combine_weights(topk_weights, handle)
        return combined_x, combined_topk_weights, EventOverlap(None)

    def _launch_native_dispatch_transport(
        self,
        x_tensor: torch.Tensor,
        topk_idx: torch.Tensor,
        num_tokens: int,
        num_max_tokens_per_rank: int,
        payload_bytes: int,
        scale_bytes: int,
        has_topk_weight: bool,
        do_cpu_sync: bool,
    ) -> V2TransportHandle:
        sizes = ep.v2_descriptor_sizes()
        max_segments = max(1, int(num_tokens * self.num_topk))
        max_batches = max(1, int(self.num_experts * self.num_ranks))
        segments = torch.empty((max_segments * int(sizes["dispatch_segment"]),),
                               dtype=torch.uint8, device=topk_idx.device)
        batches = torch.empty((max_batches * int(sizes["dispatch_batch"]),),
                              dtype=torch.uint8, device=topk_idx.device)
        counters = torch.zeros((3,), dtype=torch.int32, device=topk_idx.device)
        descriptor_queue_capacity = max_segments + max_batches + 1
        direct_queue_capacity = int(num_tokens * self.num_topk * 2 + 1)
        queue_capacity = _next_power_of_two(
            direct_queue_capacity
            if self._v2_efa_connection is not None
            else descriptor_queue_capacity
        )
        queue = self.allocate_d2h_queue(queue_capacity)
        if self._v2_efa_connection is not None:
            layout = self._make_dispatch_window_layout(
                num_tokens=num_tokens,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                payload_bytes=payload_bytes,
                max_batches=max_batches,
            )
            self._stage_tensor_bytes_to_v2_window(x_tensor, int(layout["local_payload_base"]))
        else:
            layout = {
                "local_payload_base": 0,
                "remote_payload_base": 0,
                "remote_signal_base": _align(max_batches * num_max_tokens_per_rank * payload_bytes, 64),
                "src_token_stride": payload_bytes,
                "expanded_slot_stride": payload_bytes,
                "batch_payload_stride": _align(num_max_tokens_per_rank * payload_bytes, 64),
                "signal_stride": 4,
            }
        flat_topk_idx = topk_idx.reshape(-1)
        if self._v2_efa_connection is not None:
            self.launch_dispatch_descriptors(
                topk_idx=flat_topk_idx,
                segments=segments,
                batches=batches,
                counters=counters,
                num_tokens=num_tokens,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                scale_bytes=scale_bytes,
                has_topk_weight=has_topk_weight,
                do_cpu_sync=do_cpu_sync,
            )
            self.launch_dispatch_direct_enqueue_d2h_queue(
                topk_idx=flat_topk_idx,
                queue=queue,
                layout=layout,
                num_tokens=num_tokens,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                scale_bytes=scale_bytes,
                has_topk_weight=has_topk_weight,
                do_cpu_sync=do_cpu_sync,
                smem_bytes=0,
            )
        else:
            self.launch_dispatch_descriptor_enqueue_d2h_queue(
                topk_idx=flat_topk_idx,
                segments=segments,
                batches=batches,
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
        num_segments = int(counters[0].item())
        num_batches = int(counters[1].item())
        dispatch_drain_stats = None
        if self._v2_efa_connection is not None:
            dispatch_drain_stats = self._v2_efa_connection.drain_queue(queue, True, True)
        return V2TransportHandle(
            dispatch_segments=segments,
            dispatch_batches=batches,
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
        )

    def _overlay_native_dispatch_payload_from_window(
        self,
        recv_x: torch.Tensor,
        recv_src_global: torch.Tensor,
        transport: V2TransportHandle,
        num_max_tokens_per_rank: int,
        payload_bytes: int,
    ) -> None:
        """Replace remote dispatch payload rows with bytes delivered by EFA.

        This is an incremental bridge toward native V2 receiver consumption.
        Ordering and metadata still come from the semantic reference path, but
        remote scaleout payload bytes are read from the registered V2 RDMA
        window at the offsets used by the direct enqueue kernel.
        """

        if self._v2_efa_connection is None or self._v2_efa_window is None:
            return
        if transport.dispatch_drain_stats is None:
            return
        if int(transport.dispatch_drain_stats.get("posted_writes", 0)) == 0:
            return
        layout = transport.dispatch_layout or {}
        remote_payload_base = int(layout.get("remote_payload_base", 0))
        expanded_slot_stride = int(layout.get("expanded_slot_stride", payload_bytes))
        window = self._require_v2_efa_window(
            remote_payload_base + int(num_max_tokens_per_rank) * expanded_slot_stride
        )

        for row in range(int(recv_x.shape[0])):
            src_global = int(recv_src_global[row].item())
            src_rank = src_global // int(num_max_tokens_per_rank)
            src_scaleout_rank = src_rank // self.num_scaleup_ranks
            if src_scaleout_rank == self.scaleout_rank_idx:
                continue
            src_token = src_global % int(num_max_tokens_per_rank)
            offset = remote_payload_base + src_token * expanded_slot_stride
            row_bytes = recv_x[row].reshape(-1).view(torch.uint8)
            row_bytes.copy_(window[offset: offset + int(payload_bytes)])

    def _launch_native_combine_transport(self, x: torch.Tensor, handle: EPHandle) -> None:
        transport = handle.transport_handle
        if transport is None or self._v2_efa_connection is None:
            return
        sizes = ep.v2_descriptor_sizes()
        forward_metadata = handle.token_metadata_at_forward
        if forward_metadata is None:
            forward_metadata = self._build_forward_metadata_tensors(
                torch.full(
                    (handle.recv_src_metadata.shape[0], handle.topk_idx.shape[1]),
                    -1,
                    dtype=handle.topk_idx.dtype,
                    device=handle.topk_idx.device,
                ),
                handle.recv_src_metadata,
                handle.do_expand,
                handle.num_experts,
                handle.num_max_tokens_per_rank,
            )[0]
        num_forward_rows = int(forward_metadata.numel() // forward_metadata.shape[-1])
        max_segments = max(1, int(num_forward_rows * handle.topk_idx.shape[1]))
        max_batches = max(1, max_segments)
        segments = torch.empty((max_segments * int(sizes["combine_segment"]),),
                               dtype=torch.uint8, device=x.device)
        batches = torch.empty((max_batches * int(sizes["combine_batch"]),),
                              dtype=torch.uint8, device=x.device)
        counters = torch.zeros((3,), dtype=torch.int32, device=x.device)
        queue = self.allocate_d2h_queue(_next_power_of_two(max_segments + max_batches + 1))
        payload_bytes = int(x.shape[1] * x.element_size())
        layout = self._make_combine_window_layout(
            num_source_tokens=int(x.shape[0]),
            num_max_tokens_per_rank=handle.num_max_tokens_per_rank,
            payload_bytes=payload_bytes,
            max_batches=max_batches,
        )
        self._stage_tensor_bytes_to_v2_window(x, int(layout["local_payload_base"]))
        self.launch_combine_forward_metadata_enqueue_d2h_queue(
            forward_metadata=forward_metadata,
            segments=segments,
            batches=batches,
            counters=counters,
            queue=queue,
            layout=layout,
            num_forward_rows=num_forward_rows,
            num_max_tokens_per_rank=handle.num_max_tokens_per_rank,
            payload_bytes=payload_bytes,
            use_expanded_layout=handle.do_expand,
        )
        torch.cuda.current_stream().synchronize()
        num_segments = int(counters[0].item())
        num_batches = int(counters[1].item())
        combine_drain_stats = self._v2_efa_connection.drain_queue(queue, True, True)
        transport.combine_segments = segments
        transport.combine_batches = batches
        transport.combine_counters = counters
        transport.combine_d2h_queue = queue
        transport.combine_layout = layout
        transport.num_combine_segments = num_segments
        transport.num_combine_batches = num_batches
        transport.combine_drain_stats = combine_drain_stats

    def _build_combine_descriptors_from_forward_metadata(
        self,
        x: torch.Tensor,
        handle: EPHandle,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        payload_bytes = int(x.shape[1] * x.element_size())
        segment_words = 12
        batch_words = 10
        segments = []
        batches = []
        batch_for_dst = {}

        def add_segment(dst_rank: int, src_scaleout_rank: int, source_slot: int,
                        topk_slot: int, reduced_token_slot: int) -> None:
            dst_scaleout_rank = int(dst_rank) // self.num_scaleup_ranks
            dst_scaleup_lane = int(dst_rank) % self.num_scaleup_ranks
            key = (int(dst_rank), dst_scaleout_rank, dst_scaleup_lane)
            if key not in batch_for_dst:
                batch_for_dst[key] = len(batches)
                batches.append([
                    int(dst_rank), dst_scaleout_rank, dst_scaleup_lane,
                    self.scaleout_rank_idx, 0, len(segments), 0, 0, 0, 0,
                ])
            batch_idx = batch_for_dst[key]
            batches[batch_idx][6] += 1
            batches[batch_idx][7] += 1
            segments.append([
                int(dst_rank), dst_scaleout_rank, dst_scaleup_lane,
                int(src_scaleout_rank), 0, 1, int(source_slot), -1,
                int(topk_slot), int(reduced_token_slot), payload_bytes,
                1 << 3,
            ])

        forward_metadata = handle.token_metadata_at_forward
        if forward_metadata is None:
            forward_metadata = self._build_forward_metadata_tensors(
                torch.full(
                    (handle.recv_src_metadata.shape[0], handle.topk_idx.shape[1]),
                    -1,
                    dtype=handle.topk_idx.dtype,
                    device=handle.topk_idx.device,
                ),
                handle.recv_src_metadata,
                handle.do_expand,
                handle.num_experts,
                handle.num_max_tokens_per_rank,
            )[0]
        forward_rows = forward_metadata.reshape(-1, forward_metadata.shape[-1])
        for row in range(int(forward_rows.shape[0])):
            token_global = int(forward_rows[row, 0].item())
            if token_global < 0:
                continue
            dst_rank = token_global // int(handle.num_max_tokens_per_rank)
            src_scaleout_rank = dst_rank // self.num_scaleup_ranks
            if src_scaleout_rank == self.scaleout_rank_idx:
                continue
            reduced_token_slot = token_global % int(handle.num_max_tokens_per_rank)
            if handle.do_expand:
                for topk_slot in range(int(handle.topk_idx.shape[1])):
                    source_slot = int(forward_rows[row, 2 + handle.topk_idx.shape[1] + topk_slot].item())
                    if source_slot >= 0:
                        add_segment(dst_rank, src_scaleout_rank, source_slot,
                                    topk_slot, reduced_token_slot)
            else:
                add_segment(dst_rank, src_scaleout_rank, row, 0, reduced_token_slot)

        if not segments:
            segments = [[0] * segment_words]
        if not batches:
            batches = [[0] * batch_words]

        segments_tensor = torch.tensor(
            [word for segment in segments for word in segment],
            dtype=torch.int32,
            device=x.device,
        ).view(torch.uint8)
        batches_tensor = torch.tensor(
            [word for batch in batches for word in batch],
            dtype=torch.int32,
            device=x.device,
        ).view(torch.uint8)
        counters = torch.tensor(
            [0 if segments == [[0] * segment_words] else len(segments),
             0 if batches == [[0] * batch_words] else len(batches),
             0],
            dtype=torch.int32,
            device=x.device,
        )
        return segments_tensor, batches_tensor, counters

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

    def _make_combine_window_layout(
        self,
        num_source_tokens: int,
        num_max_tokens_per_rank: int,
        payload_bytes: int,
        max_batches: int,
    ) -> dict:
        src_bytes = _align(int(num_source_tokens) * int(payload_bytes), 64)
        reduced_bytes = _align(int(num_max_tokens_per_rank) * int(payload_bytes), 64)
        remote_payload_base = src_bytes
        remote_signal_base = _align(remote_payload_base + int(max_batches) * reduced_bytes, 64)
        total_bytes = _align(remote_signal_base + int(max_batches) * 4, 64)
        self._require_v2_efa_window(total_bytes)
        return {
            "local_payload_base": 0,
            "remote_payload_base": remote_payload_base,
            "remote_signal_base": remote_signal_base,
            "expanded_slot_stride": int(payload_bytes),
            "reduced_token_stride": int(payload_bytes),
            "batch_payload_stride": reduced_bytes,
            "signal_stride": 4,
            "src_payload_bytes": src_bytes,
            "remote_payload_bytes": int(max_batches) * reduced_bytes,
            "total_window_bytes": total_bytes,
        }

    def _overlay_native_combine_payload_from_window(
        self,
        combined_x: torch.Tensor,
        handle: EPHandle,
    ) -> None:
        transport = handle.transport_handle
        if (
            self._v2_efa_connection is None or self._v2_efa_window is None or
            transport is None or transport.combine_layout is None
        ):
            return
        layout = transport.combine_layout
        remote_payload_base = int(layout.get("remote_payload_base", 0))
        reduced_token_stride = int(layout.get("reduced_token_stride", combined_x.shape[1] * combined_x.element_size()))
        batch_payload_stride = int(layout.get("batch_payload_stride", 0))
        payload_bytes = int(combined_x.shape[1] * combined_x.element_size())
        window = self._require_v2_efa_window(
            remote_payload_base + max(1, int(transport.num_combine_batches)) * max(batch_payload_stride, payload_bytes)
        )
        num_experts_per_rank = max(1, handle.num_experts // self.num_ranks)
        for token in range(int(handle.topk_idx.shape[0])):
            remote_scaleout_routes = []
            for topk_slot in range(int(handle.topk_idx.shape[1])):
                expert = int(handle.topk_idx[token, topk_slot].item())
                if expert < 0:
                    continue
                owner_rank = expert // num_experts_per_rank
                owner_scaleout_rank = owner_rank // self.num_scaleup_ranks
                if owner_scaleout_rank != self.scaleout_rank_idx:
                    remote_scaleout_routes.append(owner_scaleout_rank)
            if len(remote_scaleout_routes) != 1:
                continue
            offset = remote_payload_base + token * reduced_token_stride
            combined_x[token].reshape(-1).view(torch.uint8).copy_(
                window[offset: offset + payload_bytes]
            )

    def _semantic_dispatch_data(
        self,
        x: torch.Tensor,
        sf: Optional[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        num_max_tokens_per_rank: int,
        num_experts: int,
    ):
        num_tokens, hidden = x.shape
        num_topk = topk_idx.shape[1]
        num_experts_per_rank = num_experts // self.num_ranks
        weights = topk_weights
        if weights is None:
            weights = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=x.device)

        send_x, send_sf, send_idx, send_w, send_src = [], [], [], [], []
        send_counts = torch.zeros((self.num_ranks,), dtype=torch.int32, device=x.device)
        for dst in range(self.num_ranks):
            begin = dst * num_experts_per_rank
            end = begin + num_experts_per_rank
            mask = ((topk_idx >= begin) & (topk_idx < end)).any(dim=1)
            indices = mask.nonzero(as_tuple=True)[0]
            send_counts[dst] = indices.numel()
            send_x.append(x[indices])
            if sf is not None:
                send_sf.append(sf[indices])
            raw_idx = topk_idx[indices]
            send_idx.append(torch.where((raw_idx >= begin) & (raw_idx < end),
                                        raw_idx, torch.full_like(raw_idx, -1)))
            send_w.append(weights[indices])
            send_src.append(indices.to(torch.int32) + self.rank_idx * num_max_tokens_per_rank)

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_counts_l = [int(v) for v in send_counts.cpu().tolist()]
        recv_counts_l = [int(v) for v in recv_counts.cpu().tolist()]
        num_recv = sum(recv_counts_l)

        send_x_t = torch.cat(send_x, dim=0) if send_x else x[:0]
        recv_x = torch.empty((num_recv, hidden), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(recv_x, send_x_t, recv_counts_l, send_counts_l, group=self.group)

        recv_sf = None
        if sf is not None:
            send_sf_t = torch.cat(send_sf, dim=0) if send_sf else sf[:0]
            recv_sf = torch.empty((num_recv, sf.shape[1]), dtype=sf.dtype, device=sf.device)
            dist.all_to_all_single(recv_sf, send_sf_t, recv_counts_l, send_counts_l, group=self.group)

        send_idx_t = torch.cat(send_idx, dim=0) if send_idx else topk_idx[:0]
        recv_idx = torch.empty((num_recv, num_topk), dtype=topk_idx.dtype, device=topk_idx.device)
        dist.all_to_all_single(recv_idx, send_idx_t, recv_counts_l, send_counts_l, group=self.group)

        send_w_t = torch.cat(send_w, dim=0) if send_w else weights[:0]
        recv_w = torch.empty((num_recv, num_topk), dtype=weights.dtype, device=weights.device)
        dist.all_to_all_single(recv_w, send_w_t, recv_counts_l, send_counts_l, group=self.group)

        send_src_t = torch.cat(send_src, dim=0) if send_src else torch.empty((0,), dtype=torch.int32, device=x.device)
        recv_src = torch.empty((num_recv,), dtype=torch.int32, device=x.device)
        dist.all_to_all_single(recv_src, send_src_t, recv_counts_l, send_counts_l, group=self.group)

        local_begin = self.rank_idx * num_experts_per_rank
        local_end = local_begin + num_experts_per_rank
        mask = (recv_idx >= local_begin) & (recv_idx < local_end)
        recv_idx = recv_idx - local_begin
        recv_idx.masked_fill_(~mask, -1)
        if topk_weights is None:
            recv_w = None
        return recv_x, recv_sf, recv_idx, recv_w, recv_src, recv_counts_l

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

    def _semantic_combine_data(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None],
    ) -> torch.Tensor:
        hidden = x.shape[1]
        out = torch.zeros((handle.topk_idx.shape[0], hidden), dtype=x.dtype, device=x.device)
        src = handle.recv_src_metadata[:, 0].long()
        if handle.do_expand:
            payloads = []
            src_tokens = []
            for slot in range(handle.topk_idx.shape[1]):
                expanded_idx = handle.recv_src_metadata[:, 2 + slot].long()
                mask = expanded_idx >= 0
                if mask.any():
                    payloads.append(x[expanded_idx[mask]])
                    src_tokens.append(src[mask])
            if payloads:
                send_x = torch.cat(payloads, dim=0)
                send_src = torch.cat(src_tokens, dim=0)
                recv_x, recv_token = self._semantic_all_to_all_back(
                    send_x, send_src, handle.num_max_tokens_per_rank)
                out.index_add_(0, recv_token.long(), recv_x)
        else:
            recv_x, recv_token = self._semantic_all_to_all_back(
                x[:src.numel()], src, handle.num_max_tokens_per_rank)
            out.index_add_(0, recv_token.long(), recv_x)
        if isinstance(bias, tuple):
            out += bias[0] + bias[1]
        elif bias is not None:
            out += bias
        return out

    def _semantic_combine_weights(self, topk_weights: torch.Tensor, handle: EPHandle) -> torch.Tensor:
        out = torch.zeros(handle.topk_idx.shape, dtype=topk_weights.dtype, device=topk_weights.device)
        src = handle.recv_src_metadata[:, 0].long()
        recv_w, recv_token = self._semantic_all_to_all_back(
            topk_weights[:src.numel()], src, handle.num_max_tokens_per_rank)
        if recv_w.numel() > 0:
            out[recv_token.long()] = recv_w
        return out

    def _semantic_all_to_all_back(
        self,
        payload: torch.Tensor,
        src_global: torch.Tensor,
        num_max_tokens_per_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if payload.numel() == 0:
            shape = (0,) + tuple(payload.shape[1:])
            return payload.new_empty(shape), torch.empty((0,), dtype=torch.int64, device=payload.device)

        src_rank = (src_global // num_max_tokens_per_rank).long()
        src_token = (src_global % num_max_tokens_per_rank).long()
        send_payload_parts = []
        send_token_parts = []
        send_counts = torch.zeros((self.num_ranks,), dtype=torch.int32, device=payload.device)
        for rank in range(self.num_ranks):
            mask = src_rank == rank
            indices = mask.nonzero(as_tuple=True)[0]
            send_counts[rank] = indices.numel()
            send_payload_parts.append(payload[indices])
            send_token_parts.append(src_token[indices])

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_counts_l = [int(v) for v in send_counts.cpu().tolist()]
        recv_counts_l = [int(v) for v in recv_counts.cpu().tolist()]
        total_recv = sum(recv_counts_l)

        send_payload = torch.cat(send_payload_parts, dim=0)
        recv_shape = (total_recv,) + tuple(payload.shape[1:])
        recv_payload = payload.new_empty(recv_shape)
        dist.all_to_all_single(recv_payload, send_payload, recv_counts_l, send_counts_l, group=self.group)

        send_token = torch.cat(send_token_parts, dim=0)
        recv_token = torch.empty((total_recv,), dtype=send_token.dtype, device=send_token.device)
        dist.all_to_all_single(recv_token, send_token, recv_counts_l, send_counts_l, group=self.group)
        return recv_payload, recv_token


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


def _queue_capacity(commands: torch.Tensor) -> int:
    if commands.element_size() != 1:
        raise TypeError("commands queue storage must use one-byte elements")
    if commands.numel() % 16 != 0:
        raise ValueError("commands queue storage must be a multiple of 16 bytes")
    capacity = commands.numel() // 16
    if capacity <= 0 or capacity & (capacity - 1):
        raise ValueError("commands queue capacity must be a positive power of two")
    return int(capacity)
