from __future__ import annotations

import glob
import os
import socket
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from uccl import ep

from .utils.event import EventOverlap


def detect_group_topology(group: dist.ProcessGroup) -> Tuple[int, int, int, bool]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = torch.cuda.current_device()

    node_token = (
        os.environ.get("NODE_RANK")
        or os.environ.get("GROUP_RANK")
        or os.environ.get("SLURM_NODEID")
        or socket.gethostname()
    )

    world = dist.get_world_size(group)
    node_tokens = [None] * world
    dist.all_gather_object(node_tokens, node_token, group=group)

    token_to_idx = {}
    for token in node_tokens:
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    node_idx = token_to_idx[node_token]
    num_nodes = len(token_to_idx)
    return local_rank, node_idx, num_nodes, num_nodes == 1


def get_cpu_proxies_meta(proxies, rank, scratch_ptr, scratch_bytes, num_ranks, group):
    meta = {
        "rank": rank,
        "ptr": int(scratch_ptr),
        "nbytes": int(scratch_bytes),
        "ip": ep.get_oob_ip(),
        "listen_ports": [proxy.get_listen_port() for proxy in proxies],
    }
    all_meta = [None] * num_ranks
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", torch.cuda.current_device())))
    dist.all_gather_object(all_meta, meta, group=group)
    return {m["rank"]: m for m in all_meta}


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    if "PCIE" not in torch.cuda.get_device_name():
        return
    assert group.size() <= 2, "PCIe GPUs only have pairwise NVLink connections"


def initialize_uccl(
    scratch_ptr,
    scratch_nbytes,
    rank,
    num_ranks,
    group,
    num_experts=0,
    is_intranode: Optional[bool] = None,
    use_normal_mode=False,
    rdma_buffer_is_host_allocated=False,
):
    try:
        mode_token = "_ht_" if use_normal_mode else "_ll_"
        for shm_file in glob.glob(f"/dev/shm/uccl_barrier_*{mode_token}*"):
            os.remove(shm_file)
    except Exception:
        pass

    local_rank, node_idx, num_nodes, detected_is_intranode = detect_group_topology(group)
    if is_intranode is None:
        is_intranode = detected_is_intranode
    elif is_intranode and not detected_is_intranode:
        raise ValueError("Detected multi-node group, but is_intranode=True was requested")

    proxies = []
    for thread_idx in range(ep.get_num_proxy_threads()):
        proxies.append(
            ep.Proxy(
                thread_idx=thread_idx,
                gpu_buffer_addr=scratch_ptr,
                total_size=scratch_nbytes,
                rank=rank,
                node_idx=node_idx,
                local_rank=local_rank,
                num_experts=num_experts,
                num_ranks=num_ranks,
                num_nodes=num_nodes,
                use_normal_mode=use_normal_mode,
                is_intranode=is_intranode,
                gpu_buffer_is_host_allocated=rdma_buffer_is_host_allocated,
            )
        )

    rank2meta = get_cpu_proxies_meta(
        proxies, rank, scratch_ptr, scratch_nbytes, num_ranks, group
    )
    if not is_intranode:
        peers_meta_list = [rank2meta[r] for r in range(num_ranks)]
        for proxy in proxies:
            proxy.set_peers_meta(peers_meta_list)

    ep.register_proxies(local_rank, proxies)

    if not is_intranode and proxies:
        atomic_buffer_ptr = proxies[0].get_atomic_buffer_ptr()
        if atomic_buffer_ptr:
            for proxy in proxies:
                proxy.set_atomic_buffer_ptr(atomic_buffer_ptr)

    dist.barrier(group)
    if not is_intranode:
        for proxy in proxies:
            proxy.start_dual()

    time.sleep(3)
    return proxies, None


def destroy_uccl(proxies, workers):
    if workers is not None:
        try:
            workers.stop()
        except Exception:
            pass
    try:
        for proxy in proxies:
            proxy.stop()
    except Exception:
        pass
    try:
        device_index = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        ep.unregister_proxy(device_index)
    except Exception:
        pass
    try:
        for shm_file in glob.glob("/dev/shm/uccl_barrier_*"):
            os.remove(shm_file)
    except Exception:
        pass


def _fp8_e4m3_dtype() -> torch.dtype:
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(props, "gcnArchName", "")
        if arch.startswith("gfx942"):
            return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn
