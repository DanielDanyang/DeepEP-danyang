#!/usr/bin/env python3
"""UCCL/EFA proxy RDMA FIFO microbenchmark.

这个脚本专门测“不经过 NCCL Gin”的路径：

1. GPU kernel 把命令写入 MSCCLPP FIFO。
2. CPU FifoProxy 从 FIFO 取出命令。
3. CPU proxy 用 EFA verbs 发 7 KiB WRITE。
4. 对端 CPU proxy 只负责 RDMA completion/ordered update。

它不是 DeepEP 语义正确性测试，而是长期 AWS 后端的最小传输内核：
如果这里能接近 UCCL-EP 论文/README 的 p5en 数字，而 Gin 小消息不行，
就说明 DeepEP V2 应该把 scaleout 从 device-side Gin proxy 切到这条
TransferCmd + CPU proxy + EFA verbs 路径。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import torch
import torch.distributed as dist

try:
    from uccl import ep
except ImportError:
    sys.stderr.write("Failed to import uccl.ep\n")
    raise

from utils import get_cpu_proxies_meta, init_dist_under_torchrun


def make_fifo_proxies(
    bench: ep.BenchFifo,
    buf_addr: int,
    total_size: int,
    rank: int,
    node_idx: int,
    local_rank: int,
    mode: str,
    peers_meta_list: list,
) -> List[ep.FifoProxy]:
    proxies: List[ep.FifoProxy] = []
    for i in range(int(bench.env_info().blocks)):
        proxy = ep.FifoProxy(
            thread_idx=i,
            gpu_buffer_addr=buf_addr,
            total_size=total_size,
            rank=rank,
            node_idx=node_idx,
            local_rank=local_rank,
            is_intranode=False,
        )
        proxy.set_fifo(bench.get_fifo(i))
        proxy.set_peers_meta(peers_meta_list)
        proxies.append(proxy)

    for proxy in proxies:
        if mode == "sender":
            proxy.start_sender()
        elif mode == "remote":
            proxy.start_remote()
        else:
            raise ValueError(f"unknown proxy mode: {mode}")
    return proxies


def stop_proxies(proxies: List[ep.FifoProxy]) -> None:
    for proxy in proxies:
        try:
            proxy.stop()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=512)
    parser.add_argument("--remote-sleep-sec", type=float, default=15.0)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    rank, world, group = init_dist_under_torchrun(local_rank, local_world)
    if world != 2:
        raise RuntimeError("proxy_rdma_fifo currently expects exactly 2 ranks")

    node_idx = rank // local_world
    ep.set_device(local_rank)
    scratch = torch.empty(args.size_mb << 20, dtype=torch.uint8, device=f"cuda:{local_rank}")
    torch.cuda.synchronize()

    bench = ep.BenchFifo()
    env = bench.env_info()
    proxies = [
        ep.FifoProxy(
            thread_idx=i,
            gpu_buffer_addr=scratch.data_ptr(),
            total_size=scratch.numel(),
            rank=rank,
            node_idx=node_idx,
            local_rank=local_rank,
            is_intranode=False,
        )
        for i in range(int(env.blocks))
    ]

    # 必须先创建 proxy，再 exchange listen_ports；对端连接需要这些端口。
    rank2meta = get_cpu_proxies_meta(
        proxies, rank, scratch.data_ptr(), scratch.numel(), world, group
    )
    peers_meta_list = [rank2meta[i] for i in range(world)]
    dist.barrier(group)

    active: List[ep.FifoProxy] = []
    try:
        if rank == 0:
            print(
                f"[rank0] GPU={torch.cuda.get_device_name(local_rank)} "
                f"blocks={int(env.blocks)} iterations={int(env.iterations)} "
                f"size_mb={args.size_mb}",
                flush=True,
            )
            # 复用已经 exchange 过端口的 proxy 对象，补上 FIFO 和 peer meta。
            for i, proxy in enumerate(proxies):
                proxy.set_fifo(bench.get_fifo(i))
                proxy.set_peers_meta(peers_meta_list)
                proxy.start_sender()
                active.append(proxy)
            dist.barrier(group)
            bench.launch_gpu_issue_batched_commands()
            bench.sync_stream_interruptible(poll_ms=5, timeout_ms=args.timeout_ms)
            bench.print_block_latencies()
            stats = bench.compute_stats()
            bench.print_summary(stats)
            print(f"[rank0] elapsed_ms={bench.last_elapsed_ms():.3f}", flush=True)
            for proxy in active:
                print(
                    f"[rank0] proxy={proxy.thread_idx} "
                    f"avg_wr_latency_us={proxy.avg_wr_latency_us():.3f} "
                    f"processed={proxy.processed_count()}",
                    flush=True,
                )
        else:
            for i, proxy in enumerate(proxies):
                proxy.set_fifo(bench.get_fifo(i))
                proxy.set_peers_meta(peers_meta_list)
                proxy.start_remote()
                active.append(proxy)
            print(f"[rank1] remote proxies started: {len(active)}", flush=True)
            dist.barrier(group)
            time.sleep(args.remote_sleep_sec)
            for proxy in active:
                print(
                    f"[rank1] proxy={proxy.thread_idx} processed={proxy.processed_count()}",
                    flush=True,
                )
    finally:
        stop_proxies(active)
        try:
            dist.barrier(group)
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
