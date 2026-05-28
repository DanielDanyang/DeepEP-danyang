import os
from contextlib import nullcontext
import torch
import torch.distributed as dist
from typing import Callable, Tuple, Optional, Union, List

try:
    from uccl import ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise

from uccl.ep import EventHandle, Config

from .utils_uccl import (
    EventOverlap,
    check_nvlink_connections,
    initialize_uccl,
    destroy_uccl,
    _fp8_e4m3_dtype,
)


def _record_stream_safe(tensors, stream):
    """``record_stream`` each CUDA tensor; skip ``None`` and recurse one level
    so a ``handle`` tuple can be passed in alongside flat tensors."""
    if stream is None:
        return
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, torch.Tensor):
            if t.is_cuda:
                t.record_stream(stream)
            continue
        if isinstance(t, (tuple, list)):
            for inner in t:
                if isinstance(inner, torch.Tensor) and inner.is_cuda:
                    inner.record_stream(stream)


class ProxyTransport:
    """
    Internal UCCL proxy transport used by the native DeepEP V2 AWS backend.

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with proxy mode) RDMA communication.
        runtime: the C++ runtime.
    """

    # TODO(MaoZiming): Reduce SMs. UCCL Proxy should reduce the usage of SMs.
    num_sms: int = 20

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        proxy_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_proxy_mode: bool = True,
        allow_mnnvl: bool = False,
        explicitly_destroy: bool = False,
        is_intranode: Optional[bool] = None,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with proxy mode) RDMA communication.
            proxy_mode: whether to enable proxy mode.
            num_qps_per_rank: the number of QPs for RDMA, the proxy mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_proxy_mode: whether allow NVLink traffic for proxy mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            is_intranode: whether to force intranode-only proxy mode. If set to `None`, infer it from the
                process-group topology automatically. Explicit `True` is rejected when the group spans multiple nodes.
        """
        if "LOCAL_RANK" in os.environ:
            device_index = int(os.environ["LOCAL_RANK"])
        else:
            device_index = torch.cuda.current_device()

        if hasattr(ep, "get_rdma_buffer"):
            # Allocate outside PyTorch's CUDA allocator so RDMA/IPC sees a raw
            # cudaMalloc/cudaMallocHost-style allocation instead of a possibly
            # segmented caching-allocator mapping.
            scratch_dlpack, rdma_buffer_is_host_allocated = ep.get_rdma_buffer(
                num_rdma_bytes, device_index
            )
            self.scratch = torch.utils.dlpack.from_dlpack(scratch_dlpack)
        else:
            rdma_buffer_is_host_allocated = False
            if num_rdma_bytes > 0:
                if hasattr(ep, "can_register_rdma_gpu_buffer"):
                    rdma_buffer_is_host_allocated = not bool(
                        ep.can_register_rdma_gpu_buffer(device_index, num_rdma_bytes)
                    )
                elif hasattr(ep, "rdma_buffer_should_use_host_alloc"):
                    rdma_buffer_is_host_allocated = bool(
                        ep.rdma_buffer_should_use_host_alloc(
                            device_index, num_rdma_bytes
                        )
                    )

            if num_rdma_bytes > 0 and rdma_buffer_is_host_allocated:
                # Host-pinned RDMA buffer for NICs that cannot register GPU memory.
                self.scratch = torch.zeros(
                    (num_rdma_bytes,),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
            else:
                # Device buffer for normal RDMA path. Keep a valid pointer even when RDMA is disabled.
                self.scratch = torch.zeros(
                    max(num_rdma_bytes, 1),
                    dtype=torch.uint8,
                    device=f"cuda:{device_index}",
                )

        rdma_buffer_ptr = self.scratch.data_ptr()
        _local_world = int(os.environ.get("LOCAL_WORLD_SIZE", -1))
        self.proxies, self.workers = initialize_uccl(
            rdma_buffer_ptr,
            num_rdma_bytes,
            group.rank(),
            dist.get_world_size(group),
            group,
            use_normal_mode=not proxy_mode,
            is_intranode=is_intranode,
            rdma_buffer_is_host_allocated=rdma_buffer_is_host_allocated,
        )
        check_nvlink_connections(group)

        # Initialize the native proxy runtime
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.proxy_mode = proxy_mode
        self.explicitly_destroy = explicitly_destroy
        self._next_proxy_combine_buffer = None
        runtime_cls = getattr(ep, "NativeElasticProxyBuffer", None)
        if runtime_cls is None:
            raise RuntimeError("uccl.ep is missing NativeElasticProxyBuffer; rebuild uccl-ep")
        self.runtime = runtime_cls(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            proxy_mode,
            explicitly_destroy,
            _local_world,
        )
        if num_rdma_bytes:
            self.runtime.set_rdma_buffer(rdma_buffer_ptr, rdma_buffer_is_host_allocated)

        # Synchronize device IDs
        device_ids = [
            None,
        ] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        # print("Before all_gather_object device_ids", local_device_id, flush=True)
        dist.all_gather_object(device_ids, local_device_id, group)
        # Synchronize IPC handles
        ipc_handles = [
            None,
        ] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        # print("Before all_gather_object ipc_handles", local_ipc_handle, flush=True)
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        rdma_ipc_handles = [None] * self.group_size
        # CUDA IPC only works with device memory; skip when using cudaHostAlloc.
        local_rdma_ipc_handle = (
            self.runtime.get_local_rdma_ipc_handle()
            if self.num_rdma_bytes > 0 and not rdma_buffer_is_host_allocated
            else None
        )
        dist.all_gather_object(rdma_ipc_handles, local_rdma_ipc_handle, group)
        root_unique_id = None
        # Make CPP runtime available
        self.runtime.sync(
            device_ids,
            ipc_handles,
            root_unique_id,
            rdma_ipc_handles,
        )
        assert self.runtime.is_available()
        self.connect_atomic_buffer(self.proxies[0])

        for proxy in self.proxies:
            proxy.set_atomic_buffer_ptr(self.proxies[0].get_atomic_buffer_ptr())

    def _compute_stream_ptr(self, device: torch.device):
        """
        Return the current CUDA stream pointer for proxy runtime calls.
        """
        current = torch.cuda.current_stream(device=device)
        return int(current.cuda_stream)

    def reset_rdma_buffer(self):
        """
        Reset the RDMA buffer, this is useful when you want to reuse the RDMA buffer for another run.

        """
        self.runtime.reset_rdma_buffer()

    def connect_atomic_buffer(self, proxy: "ep.UcclProxy"):
        ep.connect_atomic_buffer(proxy, self.runtime)

    def build_v2_dispatch_metadata(
        self,
        recv_src_meta: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        num_scaleup_ranks: int,
        num_local_experts: int,
        num_max_tokens_per_rank: int,
        expert_alignment: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        num_recv_tokens = int(recv_topk_idx.size(0))
        num_topk = int(recv_topk_idx.size(1))
        with alloc_ctx:
            recv_src_metadata = torch.empty(
                (max(num_recv_tokens, 1), 2 + num_topk),
                dtype=torch.int32,
                device=recv_topk_idx.device,
            )
            psum_scaleup = torch.empty(
                (num_scaleup_ranks,), dtype=torch.int32, device=recv_topk_idx.device
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

        event = self.runtime.build_v2_dispatch_metadata(
            recv_src_meta.data_ptr(),
            recv_topk_idx.data_ptr(),
            num_recv_tokens,
            num_topk,
            int(num_scaleup_ranks),
            int(num_local_experts),
            int(num_max_tokens_per_rank),
            int(expert_alignment),
            recv_src_metadata.data_ptr(),
            psum_scaleup.data_ptr(),
            psum_expert.data_ptr(),
            dst_buffer_slot_idx.data_ptr(),
            raw_expert_counts.data_ptr(),
            expanded_expert_cursor.data_ptr(),
            getattr(previous_event, "event", None),
            bool(async_finish),
            bool(allocate_on_comm_stream),
            self._compute_stream_ptr(recv_topk_idx.device),
        )
        tensors_to_record = (
            recv_src_meta,
            recv_topk_idx,
            recv_src_metadata,
            psum_scaleup,
            psum_expert,
            dst_buffer_slot_idx,
            raw_expert_counts,
            expanded_expert_cursor,
        )
        return (
            recv_src_metadata[:num_recv_tokens],
            psum_scaleup,
            psum_expert,
            dst_buffer_slot_idx[:num_recv_tokens],
            raw_expert_counts,
            EventOverlap(event, tensors_to_record if async_finish else None),
        )

    def build_v2_expanded_payload(
        self,
        recv_x,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: Optional[torch.Tensor],
        recv_src_metadata: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        x_tensor, x_scales = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        num_recv_tokens, hidden = x_tensor.shape
        num_topk = int(recv_topk_idx.size(1))
        num_expanded_tokens = max(int(psum_num_recv_tokens_per_expert[-1].item()), 1)
        num_scales = 0
        scale_tail_shape = ()
        if x_scales is not None:
            scale_tail_shape = tuple(x_scales.shape[1:])
            num_scales = 1
            for dim in scale_tail_shape:
                num_scales *= int(dim)

        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            expanded_x = torch.empty(
                (num_expanded_tokens, hidden),
                dtype=x_tensor.dtype,
                device=x_tensor.device,
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

        event = self.runtime.build_v2_expanded_payload(
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
            getattr(previous_event, "event", None),
            bool(async_finish),
            bool(allocate_on_comm_stream),
            self._compute_stream_ptr(x_tensor.device),
        )
        packed_x = (
            (expanded_x, expanded_scales)
            if expanded_scales is not None
            else expanded_x
        )
        tensors_to_record = (
            x_tensor,
            x_scales,
            recv_topk_idx,
            recv_topk_weights,
            recv_src_metadata,
            psum_num_recv_tokens_per_expert,
            expanded_x,
            expanded_scales,
            expanded_weights,
        )
        return packed_x, expanded_weights, EventOverlap(
            event, tensors_to_record if async_finish else None
        )

    def build_v2_reduced_combine_input(
        self,
        expanded_x: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        num_topk: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        num_recv_tokens = int(recv_src_metadata.size(0))
        hidden = int(expanded_x.size(1))
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
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
            ProxyTransport._dtype_code(expanded_x.dtype),
            reduced_x.data_ptr(),
            getattr(previous_event, "event", None),
            bool(async_finish),
            bool(allocate_on_comm_stream),
            self._compute_stream_ptr(expanded_x.device),
        )
        tensors_to_record = (
            expanded_x,
            recv_src_metadata,
            reduced_x,
        )
        return reduced_x[:num_recv_tokens], EventOverlap(
            event, tensors_to_record if async_finish else None
        )

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, "`explicitly_destroy` flag must be set"

        self.runtime.destroy()
        self.runtime = None
        destroy_uccl(self.proxies, self.workers)

    @staticmethod
    def is_sm90_compiled():
        return ep.is_sm90_compiled()

    @staticmethod
    def _is_efa() -> bool:
        return os.path.exists(os.getenv("EFA_HOME", "/opt/amazon/efa"))

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, "The SM count must be even"
        ProxyTransport.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        stream_ptr = int(torch.cuda.current_stream().cuda_stream)
        return EventOverlap(EventHandle(stream_ptr))

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts = self.runtime.get_comm_stream()
        if isinstance(ts, torch.Stream):
            return torch.cuda.Stream(
                stream_id=ts.stream_id,
                device_index=ts.device_index,
                device_type=ts.device_type,
            )
        return torch.cuda.ExternalStream(int(ts))

    def get_local_buffer_tensor(
        self,
        dtype: torch.dtype,
        size: Optional[torch.Size] = None,
        offset: int = 0,
        use_rdma_buffer: bool = False,
    ) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        assert dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.bool,
        }, f"Unsupported dtype for get_local_buffer_tensor: {dtype}"
        if use_rdma_buffer:
            tensor = self.scratch.view(dtype)
            if offset > 0:
                tensor = tensor[offset:]
        else:
            raise RuntimeError(
                "get_local_buffer_tensor(use_rdma_buffer=False) is not available "
                "without Torch C++ tensor bindings"
            )
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[: size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

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
            raise ValueError(f"Unsupported dtype for uccl combine: {dtype}")
        return table[dtype]

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        def env_int(name: str, default: int) -> int:
            return int(os.environ.get(name, str(default)))

        efa = ProxyTransport._is_efa()
        efa_nvl_send = env_int("EP_UCCL_NVL_SEND_TOKENS", 36)
        efa_nvl_recv = env_int("EP_UCCL_NVL_RECV_TOKENS", 288)
        efa_rdma_send = env_int("EP_UCCL_RDMA_SEND_TOKENS", 20)
        efa_rdma_recv = env_int("EP_UCCL_RDMA_RECV_TOKENS", 512)
        config_map = {
            2: Config(ProxyTransport.num_sms, 24, 256, 6, 128),
            4: Config(ProxyTransport.num_sms, 6, 256, 6, 128),
            8: Config(ProxyTransport.num_sms, 6, 256, 6, 128),
            16: Config(
                ProxyTransport.num_sms,
                efa_nvl_send if efa else 36,
                efa_nvl_recv if efa else 288,
                efa_rdma_send if efa else 20,
                efa_rdma_recv if efa else 128,
            ),
            24: Config(ProxyTransport.num_sms, 8, 288, 32, 128),
            32: Config(ProxyTransport.num_sms, 32, 288, 32, 512 if efa else 128),
            64: Config(ProxyTransport.num_sms, 20, 288, 28, 128),
            128: Config(ProxyTransport.num_sms, 20, 560, 32, 128),
            144: Config(ProxyTransport.num_sms, 32, 720, 12, 128),
            160: Config(ProxyTransport.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        def env_int(name: str, default: int) -> int:
            return int(os.environ.get(name, str(default)))

        efa = ProxyTransport._is_efa()
        efa_nvl_send = env_int("EP_UCCL_COMBINE_NVL_SEND_TOKENS", 36)
        efa_nvl_recv = env_int("EP_UCCL_COMBINE_NVL_RECV_TOKENS", 288)
        efa_rdma_send = env_int("EP_UCCL_COMBINE_RDMA_SEND_TOKENS", 20)
        efa_rdma_recv = env_int("EP_UCCL_COMBINE_RDMA_RECV_TOKENS", 512)
        config_map = {
            2: Config(ProxyTransport.num_sms, 10, 256, 6, 128),
            4: Config(ProxyTransport.num_sms, 9, 256, 6, 128),
            8: Config(ProxyTransport.num_sms, 4, 256, 6, 128),
            16: Config(
                ProxyTransport.num_sms,
                efa_nvl_send if efa else 4,
                efa_nvl_recv if efa else 288,
                efa_rdma_send if efa else 12,
                efa_rdma_recv if efa else 128,
            ),
            24: Config(ProxyTransport.num_sms, 1, 288, 8, 128),
            32: Config(ProxyTransport.num_sms, 1, 288, 8, 512 if efa else 128),
            64: Config(ProxyTransport.num_sms, 1, 288, 20, 128),
            128: Config(ProxyTransport.num_sms, 1, 560, 12, 128),
            144: Config(ProxyTransport.num_sms, 2, 720, 8, 128),
            160: Config(ProxyTransport.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        if allocate_on_comm_stream:
            assert previous_event is not None and async_finish

        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            num_tokens_per_rank = torch.empty(
                (self.group_size,), dtype=torch.int, device=topk_idx.device
            )
            num_tokens_per_rdma_rank = (
                torch.empty(
                    (self.runtime.get_num_rdma_ranks(),),
                    dtype=torch.int,
                    device=topk_idx.device,
                )
                if self.runtime.get_num_rdma_ranks() > 1
                else None
            )
            num_tokens_per_expert = torch.empty(
                (num_experts,), dtype=torch.int, device=topk_idx.device
            )
            is_token_in_rank = torch.empty(
                (topk_idx.size(0), self.group_size),
                dtype=torch.bool,
                device=topk_idx.device,
            )

        event = self.runtime.get_dispatch_layout(
            topk_idx.data_ptr(),
            topk_idx.size(0),
            topk_idx.size(1),
            num_experts,
            num_tokens_per_rank.data_ptr(),
            (
                0
                if num_tokens_per_rdma_rank is None
                else num_tokens_per_rdma_rank.data_ptr()
            ),
            num_tokens_per_expert.data_ptr(),
            is_token_in_rank.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            self._compute_stream_ptr(topk_idx.device),
        )
        previous_tensors_to_record = (
            ()
            if previous_event is None or previous_event.extra_tensors is None
            else previous_event.extra_tensors
        )
        tensors_to_record = previous_tensors_to_record + (
            topk_idx,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(
                event,
                tensors_to_record if async_finish else None,
            ),
        )

    # noinspection PyTypeChecker
    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            for proxy in self.proxies:
                proxy.notify_proxy_thread_adaptive_sleeper()

            return self.internode_dispatch(
                x,
                handle,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                num_worst_tokens,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                num_recv_tokens,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ) = handle
            num_topk = 0
            num_scales = (
                0
                if x_scales is None
                else (1 if x_scales.dim() == 1 else x_scales.size(1))
            )
            scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
            scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
            alloc_recv_tokens = max(num_recv_tokens, 1)
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (alloc_recv_tokens,)
                            if x_scales.dim() == 1
                            else (alloc_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.intranode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0,
                int(num_topk),
                0,
                is_token_in_rank.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                channel_prefix_matrix.data_ptr(),
                0,
                int(num_worst_tokens),
                True,
                config,
                int(num_recv_tokens),
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0,
                0,
                recv_channel_prefix_matrix.data_ptr(),
                recv_src_idx.data_ptr(),
                send_head.data_ptr(),
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
                self._compute_stream_ptr(x.device),
            )
            recv_x = recv_x[:num_recv_tokens]
            if recv_x_scales is not None:
                recv_x_scales = recv_x_scales[:num_recv_tokens]
            previous_tensors_to_record = (
                ()
                if previous_event is None or previous_event.extra_tensors is None
                else previous_event.extra_tensors
            )
            tensors_to_record = previous_tensors_to_record + (
                x,
                x_scales,
                handle,
                recv_x,
                recv_x_scales,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(
                    event,
                    tensors_to_record if async_finish else None,
                ),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            num_channels = int(getattr(config, "num_sms", ProxyTransport.num_sms)) // 2
            rank_prefix_matrix = torch.empty(
                (self.group_size, self.group_size), dtype=torch.int32, device=x.device
            )
            channel_prefix_matrix = torch.empty(
                (self.group_size, num_channels), dtype=torch.int32, device=x.device
            )
            num_recv_tokens, num_recv_tokens_per_expert_list, _ = (
                self.runtime.intranode_prepare(
                    num_tokens_per_rank.data_ptr(),
                    is_token_in_rank.data_ptr(),
                    num_tokens_per_expert.data_ptr(),
                    x.size(0),
                    num_tokens_per_expert.size(0),
                    rank_prefix_matrix.data_ptr(),
                    channel_prefix_matrix.data_ptr(),
                    expert_alignment,
                    num_worst_tokens,
                    config,
                    getattr(previous_event, "event", None),
                    False,
                    False,
                    self._compute_stream_ptr(x.device),
                )
            )
            num_scales = (
                0
                if x_scales is None
                else (1 if x_scales.dim() == 1 else x_scales.size(1))
            )
            scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
            scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
            alloc_recv_tokens = max(num_recv_tokens, 1)
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_src_idx = torch.empty(
                    (alloc_recv_tokens,), dtype=torch.int32, device=x.device
                )
                recv_channel_prefix_matrix = torch.empty(
                    (self.group_size, num_channels), dtype=torch.int32, device=x.device
                )
                send_head = torch.empty(
                    (x.size(0), self.group_size), dtype=torch.int32, device=x.device
                )
                recv_topk_idx = None
                recv_topk_weights = None
                if topk_idx is not None:
                    recv_topk_idx = torch.empty(
                        (alloc_recv_tokens, topk_idx.size(1)),
                        dtype=topk_idx.dtype,
                        device=x.device,
                    )
                    recv_topk_weights = torch.empty(
                        (alloc_recv_tokens, topk_weights.size(1)),
                        dtype=topk_weights.dtype,
                        device=x.device,
                    )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (alloc_recv_tokens,)
                            if x_scales.dim() == 1
                            else (alloc_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.intranode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0 if topk_idx is None else topk_idx.data_ptr(),
                0 if topk_idx is None else int(topk_idx.size(1)),
                0 if topk_weights is None else topk_weights.data_ptr(),
                is_token_in_rank.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                channel_prefix_matrix.data_ptr(),
                int(num_tokens_per_expert.size(0)),
                int(num_worst_tokens),
                False,
                config,
                int(num_recv_tokens),
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0 if recv_topk_idx is None else recv_topk_idx.data_ptr(),
                0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
                recv_channel_prefix_matrix.data_ptr(),
                recv_src_idx.data_ptr(),
                send_head.data_ptr(),
                None,
                async_finish,
                allocate_on_comm_stream,
                self._compute_stream_ptr(x.device),
            )
            recv_x = recv_x[:num_recv_tokens]
            recv_src_idx = recv_src_idx[:num_recv_tokens]
            if recv_topk_idx is not None:
                recv_topk_idx = recv_topk_idx[:num_recv_tokens]
            if recv_topk_weights is not None:
                recv_topk_weights = recv_topk_weights[:num_recv_tokens]
            if recv_x_scales is not None:
                recv_x_scales = recv_x_scales[:num_recv_tokens]
            handle = (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                # Keep the logical count separate from the storage tensor so
                # cached dispatch avoids a GPU sync while cached combine still
                # gets a valid src_idx pointer when the count is 0.
                num_recv_tokens,
                recv_src_idx if num_recv_tokens > 0 else recv_src_idx.new_empty((1,)),
                is_token_in_rank,
                send_head,
            )
            previous_tensors_to_record = (
                ()
                if previous_event is None or previous_event.extra_tensors is None
                else previous_event.extra_tensors
            )
            tensors_to_record = previous_tensors_to_record + (
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_expert,
                handle,
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(
                    event,
                    tensors_to_record if async_finish else None,
                ),
            )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            for proxy in self.proxies:
                proxy.notify_proxy_thread_adaptive_sleeper()

            return self.internode_combine(
                x,
                handle,
                topk_weights,
                bias,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # NOTES: the second `_` is for the sending side, so we should use the third one
        (
            rank_prefix_matrix,
            _,
            channel_prefix_matrix,
            _,
            src_idx,
            is_recv_token_in_rank,
            send_head,
        ) = handle
        bias_0, bias_1 = ProxyTransport._unpack_bias(bias)

        # Launch the kernel
        num_recv_tokens = send_head.size(0)
        num_topk = 0 if topk_weights is None else int(topk_weights.size(1))
        num_x_rows = x.size(0)
        if num_x_rows == 0:
            x = x.new_empty((1, x.size(1)))
        if src_idx.size(0) == 0:
            src_idx = src_idx.new_empty(
                (1,), dtype=src_idx.dtype, device=src_idx.device
            )
        if topk_weights is not None and topk_weights.size(0) == 0:
            topk_weights = topk_weights.new_empty(
                (1, num_topk), dtype=topk_weights.dtype, device=topk_weights.device
            )
        alloc_recv_tokens = max(num_recv_tokens, 1)
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            recv_x = torch.empty(
                (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
            )
            recv_topk_weights = (
                None
                if topk_weights is None
                else torch.empty(
                    (alloc_recv_tokens, num_topk),
                    device=x.device,
                    dtype=topk_weights.dtype,
                )
            )
        event = self.runtime.intranode_combine(
            x.data_ptr(),
            num_x_rows,
            x.size(1),
            ProxyTransport._dtype_code(x.dtype),
            x.element_size(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            num_topk,
            0 if bias_0 is None else bias_0.data_ptr(),
            0 if bias_1 is None else bias_1.data_ptr(),
            src_idx.data_ptr(),
            num_recv_tokens,
            rank_prefix_matrix.data_ptr(),
            channel_prefix_matrix.data_ptr(),
            send_head.data_ptr(),
            config,
            recv_x.data_ptr(),
            0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            self._compute_stream_ptr(x.device),
        )
        recv_x = recv_x[:num_recv_tokens]
        if recv_topk_weights is not None:
            recv_topk_weights = recv_topk_weights[:num_recv_tokens]
        previous_tensors_to_record = (
            ()
            if previous_event is None or previous_event.extra_tensors is None
            else previous_event.extra_tensors
        )
        tensors_to_record = previous_tensors_to_record + (
            x,
            topk_weights,
            bias_0,
            bias_1,
            handle,
            recv_x,
            recv_topk_weights,
        )
        return (
            recv_x,
            recv_topk_weights,
            EventOverlap(
                event,
                tensors_to_record if async_finish else None,
            ),
        )

    # noinspection PyTypeChecker
    def internode_dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        x, x_scales = x if isinstance(x, tuple) else (x, None)
        num_scales = (
            0 if x_scales is None else (1 if x_scales.dim() == 1 else x_scales.size(1))
        )
        scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
        scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
        num_topk = 0 if topk_idx is None else int(topk_idx.size(1))
        num_rdma_ranks = self.runtime.get_num_rdma_ranks()
        num_channels = int(getattr(config, "num_sms", ProxyTransport.num_sms)) // 2
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                num_recv_tokens,
                num_rdma_recv_tokens,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            ) = handle
            # Allocate at least 1 row so data_ptr() is never null (zero-token ranks
            # produce empty tensors whose data_ptr()==0, which trips C++ assertions).
            alloc_recv_tokens = max(num_recv_tokens, 1)
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (alloc_recv_tokens,)
                            if x_scales.dim() == 1
                            else (alloc_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.internode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0,
                int(num_topk),
                0,
                is_token_in_rank.data_ptr(),
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                0,
                int(num_worst_tokens),
                True,
                int(num_rdma_recv_tokens),
                config,
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
                self._compute_stream_ptr(x.device),
            )
            previous_tensors_to_record = (
                ()
                if previous_event is None or previous_event.extra_tensors is None
                else previous_event.extra_tensors
            )
            tensors_to_record = previous_tensors_to_record + (
                x,
                x_scales,
                handle,
                recv_x,
                recv_x_scales,
            )
            # Slice back to the real number of received tokens.
            recv_x = recv_x[:num_recv_tokens]
            if recv_x_scales is not None:
                recv_x_scales = recv_x_scales[:num_recv_tokens]
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(
                    event,
                    tensors_to_record if async_finish else None,
                ),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and num_tokens_per_rdma_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            # Allocate prefix-matrix / prefix-sum tensors on comm_stream so
            # the PyTorch caching allocator ties their lifetime to comm_stream
            # (mirrors DeepEP's at::cuda::setCurrentCUDAStream(comm_stream)
            # around its torch::empty calls).
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                rdma_channel_prefix_matrix = torch.empty(
                    (num_rdma_ranks, num_channels),
                    dtype=torch.int32,
                    device=x.device,
                )
                recv_rdma_rank_prefix_sum = torch.empty(
                    (num_rdma_ranks,),
                    dtype=torch.int32,
                    device=x.device,
                )
                gbl_channel_prefix_matrix = torch.empty(
                    (self.group_size, num_channels),
                    dtype=torch.int32,
                    device=x.device,
                )
                recv_gbl_rank_prefix_sum = torch.empty(
                    (self.group_size,),
                    dtype=torch.int32,
                    device=x.device,
                )
            (
                num_recv_tokens,
                num_rdma_recv_tokens,
                num_recv_tokens_per_expert_list,
                _,
            ) = self.runtime.internode_prepare(
                num_tokens_per_rank.data_ptr(),
                num_tokens_per_rdma_rank.data_ptr(),
                num_tokens_per_expert.data_ptr(),
                is_token_in_rank.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                int(num_scales),
                int(num_topk),
                int(num_tokens_per_expert.size(0)),
                int(expert_alignment),
                int(num_worst_tokens),
                config,
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                getattr(previous_event, "event", None),
                False,
                False,
                self._compute_stream_ptr(x.device),
            )
            # Allocate at least 1 row so data_ptr() is never null (zero-token ranks
            # produce empty tensors whose data_ptr()==0, which trips C++ assertions).
            alloc_recv_tokens = max(num_recv_tokens, 1)
            alloc_rdma_recv_tokens = max(num_rdma_recv_tokens, 1)
            with alloc_ctx:
                recv_x = torch.empty(
                    (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (alloc_recv_tokens,)
                            if x_scales.dim() == 1
                            else (alloc_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
                recv_topk_idx = (
                    None
                    if topk_idx is None
                    else torch.empty(
                        (alloc_recv_tokens, topk_idx.size(1)),
                        dtype=topk_idx.dtype,
                        device=x.device,
                    )
                )
                recv_topk_weights = (
                    None
                    if topk_weights is None
                    else torch.empty(
                        (alloc_recv_tokens, topk_weights.size(1)),
                        dtype=topk_weights.dtype,
                        device=x.device,
                    )
                )
                recv_src_meta = torch.empty(
                    (alloc_recv_tokens, self.runtime.get_source_meta_bytes()),
                    dtype=torch.uint8,
                    device=x.device,
                )
                recv_rdma_channel_prefix_matrix = torch.empty(
                    (num_rdma_ranks, num_channels), dtype=torch.int32, device=x.device
                )
                recv_gbl_channel_prefix_matrix = torch.empty(
                    (self.group_size, num_channels), dtype=torch.int32, device=x.device
                )
                # Keep at least 1 row so data_ptr() is never null on the
                # DP-attention / TBO empty-micro-batch case (x.size(0) == 0).
                send_rdma_head = torch.empty(
                    (max(x.size(0), 1), num_rdma_ranks),
                    dtype=torch.int32,
                    device=x.device,
                )
                send_nvl_head = torch.empty(
                    (alloc_rdma_recv_tokens, self.runtime.get_num_max_nvl_peers()),
                    dtype=torch.int32,
                    device=x.device,
                )
            event = self.runtime.internode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0 if topk_idx is None else topk_idx.data_ptr(),
                int(num_topk),
                0 if topk_weights is None else topk_weights.data_ptr(),
                is_token_in_rank.data_ptr(),
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                int(num_tokens_per_expert.size(0)),
                int(num_worst_tokens),
                False,
                int(num_rdma_recv_tokens),
                config,
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0 if recv_topk_idx is None else recv_topk_idx.data_ptr(),
                0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
                recv_src_meta.data_ptr(),
                recv_rdma_channel_prefix_matrix.data_ptr(),
                recv_gbl_channel_prefix_matrix.data_ptr(),
                send_rdma_head.data_ptr(),
                send_nvl_head.data_ptr(),
                None,
                async_finish,
                allocate_on_comm_stream,
                self._compute_stream_ptr(x.device),
            )
            # Slice recv tensors back to the real number of received tokens.
            recv_x = recv_x[:num_recv_tokens]
            if recv_x_scales is not None:
                recv_x_scales = recv_x_scales[:num_recv_tokens]
            if recv_topk_idx is not None:
                recv_topk_idx = recv_topk_idx[:num_recv_tokens]
            if recv_topk_weights is not None:
                recv_topk_weights = recv_topk_weights[:num_recv_tokens]
            recv_src_meta = recv_src_meta[:num_recv_tokens]
            # Keep at least 1 row so data_ptr() is never null (zero-token combine
            # assertion guard, matching the alloc_rdma_recv_tokens pattern above).
            send_nvl_head = send_nvl_head[: max(num_rdma_recv_tokens, 1)]
            handle = (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                num_recv_tokens,
                num_rdma_recv_tokens,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            )
            previous_tensors_to_record = (
                ()
                if previous_event is None or previous_event.extra_tensors is None
                else previous_event.extra_tensors
            )
            tensors_to_record = previous_tensors_to_record + (
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                handle,
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
            )
            # Mirror DeepEP's record_stream tail: tag every tensor read or
            # written by comm_stream so the caching allocator does not recycle
            # their storage on the compute stream mid-kernel.
            comm_stream = self.get_comm_stream()
            _record_stream_safe(tensors_to_record, comm_stream)
            if allocate_on_comm_stream:
                _record_stream_safe(tensors_to_record, torch.cuda.current_stream())
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(
                    event,
                    tensors_to_record if async_finish else None,
                ),
            )

    # noinspection PyTypeChecker
    def internode_combine(
        self,
        x: torch.Tensor,
        handle: Union[tuple, list],
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Unpack handle and bias
        (
            is_combined_token_in_rank,
            _,
            _,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            gbl_rank_prefix_sum,
            _,
            _,
            src_meta,
            send_rdma_head,
            send_nvl_head,
        ) = handle
        bias_0, bias_1 = ProxyTransport._unpack_bias(bias)

        num_combined_tokens = int(is_combined_token_in_rank.size(0))
        num_topk = 0 if topk_weights is None else int(topk_weights.size(1))

        # Allocate at least 1 row so data_ptr() is never null (zero-token ranks
        # produce empty tensors whose data_ptr()==0, which trips C++ assertions).
        alloc_combined_tokens = max(num_combined_tokens, 1)
        num_x_rows = x.size(0)
        if num_x_rows == 0:
            x = x.new_empty((1, x.size(1)))
        if src_meta.size(0) == 0:
            src_meta = src_meta.new_empty((1,) + src_meta.shape[1:])

        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            combined_x = torch.empty(
                (alloc_combined_tokens, x.size(1)), device=x.device, dtype=x.dtype
            )
            combined_topk_weights = (
                None
                if topk_weights is None
                else torch.empty(
                    (alloc_combined_tokens, num_topk),
                    device=x.device,
                    dtype=topk_weights.dtype,
                )
            )
        event = self.runtime.internode_combine(
            x.data_ptr(),
            num_x_rows,
            x.size(1),
            ProxyTransport._dtype_code(x.dtype),
            x.element_size(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            num_topk,
            0 if bias_0 is None else bias_0.data_ptr(),
            0 if bias_1 is None else bias_1.data_ptr(),
            src_meta.data_ptr(),
            num_combined_tokens,
            is_combined_token_in_rank.data_ptr(),
            rdma_channel_prefix_matrix.data_ptr(),
            rdma_rank_prefix_sum.data_ptr(),
            gbl_channel_prefix_matrix.data_ptr(),
            send_rdma_head.data_ptr(),
            send_nvl_head.data_ptr(),
            config,
            combined_x.data_ptr(),
            0 if combined_topk_weights is None else combined_topk_weights.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            self._compute_stream_ptr(x.device),
        )
        # Slice back to the real number of combined tokens.
        combined_x = combined_x[:num_combined_tokens]
        if combined_topk_weights is not None:
            combined_topk_weights = combined_topk_weights[:num_combined_tokens]
        previous_tensors_to_record = (
            ()
            if previous_event is None or previous_event.extra_tensors is None
            else previous_event.extra_tensors
        )
        tensors_to_record = previous_tensors_to_record + (
            x,
            topk_weights,
            bias_0,
            bias_1,
            handle,
            combined_x,
            combined_topk_weights,
        )
        # See internode_dispatch above for rationale on record_stream tail.
        comm_stream = self.get_comm_stream()
        _record_stream_safe(tensors_to_record, comm_stream)
        if allocate_on_comm_stream:
            _record_stream_safe(tensors_to_record, torch.cuda.current_stream())
        return (
            combined_x,
            combined_topk_weights,
            EventOverlap(
                event,
                tensors_to_record if async_finish else None,
            ),
        )
