import os
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.buffers.elastic import ElasticBuffer
from uccl import ep


def _init_dist():
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:29591",
            rank=rank,
            world_size=world,
        )
    else:
        dist.init_process_group("nccl")


def main():
    _init_dist()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    repo_root = os.environ.get(
        "DEEPEP_REPO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    deep_ep.init_deep_ep_jit(
        os.path.join(repo_root, "deep_ep"),
        os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        deep_ep.find_nccl_root(),
    )

    group = dist.group.WORLD
    buf = ElasticBuffer(
        group,
        num_bytes=4096,
        num_max_tokens_per_rank=4,
        hidden=16,
        num_topk=1,
    )
    info = buf.init_native_v2_efa_transport(num_bytes=4096, num_lanes=1)
    assert buf.has_native_v2_efa_transport()
    print(
        f"rank={rank} device={info['device_name']} qpns={list(info['qpns'])} "
        f"rkey={info['rkey']} connected={buf._v2_efa_connection.is_connected()}",
        flush=True,
    )

    if dist.get_world_size() == 1:
        window = buf._v2_efa_window
        window.zero_()
        window[:16] = torch.arange(16, dtype=torch.uint8, device=window.device)
        torch.cuda.synchronize()
        stats = buf._v2_efa_connection.post_op(
            {
                "kind": 1,
                "target_rank": 0,
                "target_lane": 0,
                "bytes": 16,
                "local_offset": 0,
                "remote_offset": 128,
            }
        )
        for _ in range(1000):
            if buf._v2_efa_connection.poll_completions(16):
                break
            time.sleep(0.001)
        torch.cuda.synchronize()
        got = window[128:144].cpu().tolist()
        assert got == list(range(16)), got
        print(f"rank={rank} self_rdma_stats={stats} ok=True", flush=True)
    else:
        window = buf._v2_efa_window
        window.zero_()
        if rank == 0:
            window[:16] = torch.arange(16, dtype=torch.uint8, device=window.device)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            stats = buf._v2_efa_connection.post_op(
                {
                    "kind": 1,
                    "target_rank": 1,
                    "target_lane": 0,
                    "bytes": 16,
                    "local_offset": 0,
                    "remote_offset": 128,
                }
            )
            for _ in range(1000):
                if buf._v2_efa_connection.poll_completions(16):
                    break
                time.sleep(0.001)
            print(f"rank={rank} remote_rdma_stats={stats}", flush=True)
        dist.barrier()
        if rank == 1:
            torch.cuda.synchronize()
            got = window[128:144].cpu().tolist()
            assert got == list(range(16)), got
            print(f"rank={rank} remote_rdma_recv_ok=True", flush=True)

        window.zero_()
        x = (torch.arange(16, dtype=torch.uint8, device=window.device) + rank * 64).reshape(1, 16)
        topk_idx = torch.tensor([[1 - rank]], dtype=torch.int64, device=window.device)
        _, _, _, handle, _ = buf.dispatch(
            x,
            topk_idx=topk_idx,
            num_experts=2,
            num_max_tokens_per_rank=4,
            do_cpu_sync=True,
        )
        for _ in range(1000):
            if buf._v2_efa_connection.poll_completions(16):
                break
            time.sleep(0.001)
        dist.barrier()
        torch.cuda.synchronize()
        layout = handle.transport_handle.dispatch_layout
        offset = int(layout["remote_payload_base"])
        got = window[offset: offset + 16].cpu().tolist()
        expected = (torch.arange(16, dtype=torch.uint8) + (1 - rank) * 64).tolist()
        assert got == expected, (got, expected, layout, handle.transport_handle.dispatch_drain_stats)
        print(
            f"rank={rank} dispatch_rdma_recv_ok=True "
            f"stats={handle.transport_handle.dispatch_drain_stats}",
            flush=True,
        )

    dist.barrier()
    if rank == 0:
        print("v2_efa_connection_smoke_ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
