#!/usr/bin/env python3
"""Smoke benchmark for the AWS DeepEP V2 wrapper delegation path.

Run with `PYTHONPATH=<uccl-ep>/deep_ep_v2_wrapper` so `import deep_ep`
resolves to the AWS wrapper, not the upstream DeepEP package.

The goal is to verify the long-term transport path and print README-style
logical bandwidth numbers:

    deep_ep.ElasticBuffer API -> UCCL legacy HT Buffer -> CPU proxy -> EFA verbs
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.math import count_bytes, per_token_cast_to_fp8, safe_div


def wait_event(event) -> None:
    if event is None:
        return
    if getattr(event, "event", None) is None:
        torch.cuda.synchronize()
        return
    if hasattr(event, "current_stream_wait"):
        event.current_stream_wait()


def init_dist() -> tuple[int, int, dist.ProcessGroup]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    group = dist.new_group(list(range(dist.get_world_size())))
    return dist.get_rank(), dist.get_world_size(), group


def tensor_shape(value) -> tuple:
    if isinstance(value, tuple):
        return tuple(value[0].shape)
    return tuple(value.shape)


def first_tensor(value) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def avg_seconds(values: list[float]) -> float:
    return sum(values) / len(values)


def bandwidth_line(scaleout_bytes: float, scaleup_bytes: float, seconds: float) -> str:
    return (
        f"{scaleout_bytes / seconds / 1e9:.0f} GB/s (SO), "
        f"{scaleup_bytes / seconds / 1e9:.0f} GB/s (SU), "
        f"{seconds * 1e6:.3f} us, {scaleup_bytes:.0f} bytes"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--use-fp8-dispatch", action="store_true")
    parser.add_argument("--ignore-local-traffic", action="store_true")
    args = parser.parse_args()

    rank, world, group = init_dist()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(1234 + rank)

    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        explicitly_destroy=True,
    )

    x_bf16 = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    x = per_token_cast_to_fp8(x_bf16) if args.use_fp8_dispatch else x_bf16
    topk_idx = torch.topk(
        torch.rand((args.num_tokens, args.num_experts), dtype=torch.float32, device="cuda"),
        args.num_topk,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices.to(torch.int64)
    topk_weights = torch.ones((args.num_tokens, args.num_topk), dtype=torch.float32, device="cuda")

    ok = False
    recv_x = recv_topk_idx = recv_topk_weights = handle = combined_x = combined_topk_weights = None
    combine_x = None
    expanded_recv_x = expanded_recv_topk_idx = expanded_recv_topk_weights = expanded_handle = None
    cached_recv_x = cached_recv_topk_idx = cached_recv_topk_weights = cached_handle = None
    try:
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.num_tokens,
            expert_alignment=1,
        )
        wait_event(event)
        expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, expanded_handle, event = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.num_tokens,
            expert_alignment=1,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
        )
        wait_event(event)
        num_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        combine_x = torch.randn(
            (num_recv_tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
        )
        combined_x, combined_topk_weights, event = buffer.combine(
            x=combine_x,
            handle=handle,
            topk_weights=recv_topk_weights,
        )
        wait_event(event)
        torch.cuda.synchronize()
        dist.barrier(group)

        elapsed = []
        expanded_elapsed = []
        cached_elapsed = []
        combine_elapsed = []
        for _ in range(args.iters):
            dist.barrier(group)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
                x=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=args.num_experts,
                num_max_tokens_per_rank=args.num_tokens,
                expert_alignment=1,
            )
            wait_event(event)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - t0)

            dist.barrier(group)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, expanded_handle, event = buffer.dispatch(
                x=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=args.num_experts,
                num_max_tokens_per_rank=args.num_tokens,
                expert_alignment=1,
                do_expand=True,
                use_tma_aligned_col_major_sf=True,
            )
            wait_event(event)
            torch.cuda.synchronize()
            expanded_elapsed.append(time.perf_counter() - t0)

            dist.barrier(group)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle, event = buffer.dispatch(
                x=x,
                handle=handle,
            )
            wait_event(event)
            torch.cuda.synchronize()
            cached_elapsed.append(time.perf_counter() - t0)

            dist.barrier(group)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            combined_x, combined_topk_weights, event = buffer.combine(
                x=combine_x,
                handle=handle,
                topk_weights=recv_topk_weights,
            )
            wait_event(event)
            torch.cuda.synchronize()
            combine_elapsed.append(time.perf_counter() - t0)

        avg_t = avg_seconds(elapsed)
        expanded_avg_t = avg_seconds(expanded_elapsed)
        cached_avg_t = avg_seconds(cached_elapsed)
        combine_avg_t = avg_seconds(combine_elapsed)
        avg_ms = avg_t * 1e3
        expanded_avg_ms = expanded_avg_t * 1e3
        cached_avg_ms = cached_avg_t * 1e3
        combine_avg_ms = combine_avg_t * 1e3

        num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
        dst_scaleout_rank_idx = topk_idx // (args.num_experts // num_scaleout_ranks)
        num_scaleout_send_tokens = 0
        for i in range(num_scaleout_ranks if num_scaleout_ranks > 1 else 0):
            if args.ignore_local_traffic and i == rank // num_scaleup_ranks:
                continue
            num_scaleout_send_tokens += (dst_scaleout_rank_idx == i).any(dim=1).sum().item()

        src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
        num_scaleup_recv_tokens = num_recv_tokens
        if args.ignore_local_traffic:
            same_scaleup = (
                src_token_global_idx // args.num_tokens % num_scaleup_ranks
                == rank % num_scaleup_ranks
            )
            num_scaleup_recv_tokens -= same_scaleup.sum().item()

        dispatch_token_bytes = safe_div(
            count_bytes(recv_x, recv_topk_idx, recv_topk_weights),
            first_tensor(recv_x).size(0),
        )
        dispatch_scaleout_bytes = dispatch_token_bytes * num_scaleout_send_tokens
        dispatch_scaleup_bytes = dispatch_token_bytes * num_scaleup_recv_tokens

        combine_token_bytes = safe_div(
            count_bytes(combine_x, recv_topk_weights), combine_x.size(0)
        )
        combine_scaleout_bytes = combine_token_bytes * num_scaleout_send_tokens
        combine_scaleup_bytes = combine_token_bytes * num_scaleup_recv_tokens

        if rank % int(os.environ["LOCAL_WORLD_SIZE"]) == 0:
            print(
                f"[v2-proxy-smoke] rank={rank}/{world} local_rank={local_rank} "
                f"recv={tensor_shape(recv_x)} combined={None if combined_x is None else tuple(combined_x.shape)} "
                f"dispatch_avg_ms={avg_ms:.3f} expanded_dispatch_avg_ms={expanded_avg_ms:.3f} "
                f"cached_dispatch_avg_ms={cached_avg_ms:.3f} combine_avg_ms={combine_avg_ms:.3f}",
                flush=True,
            )
            print(
                f"[v2-readme] EP: {rank:3}/{world} | dispatch: "
                f"{bandwidth_line(dispatch_scaleout_bytes, dispatch_scaleup_bytes, avg_t)}",
                flush=True,
            )
            print(
                f"[v2-readme] EP: {rank:3}/{world} | expanded dispatch: "
                f"{bandwidth_line(dispatch_scaleout_bytes, dispatch_scaleup_bytes, expanded_avg_t)}",
                flush=True,
            )
            print(
                f"[v2-readme] EP: {rank:3}/{world} | cached dispatch: "
                f"{bandwidth_line(dispatch_scaleout_bytes, dispatch_scaleup_bytes, cached_avg_t)}",
                flush=True,
            )
            print(
                f"[v2-readme] EP: {rank:3}/{world} | combine: "
                f"{bandwidth_line(combine_scaleout_bytes, combine_scaleup_bytes, combine_avg_t)}",
                flush=True,
            )
        ok = True
    finally:
        if ok:
            dist.barrier(group)
        del recv_x, recv_topk_idx, recv_topk_weights, handle, combined_x, combined_topk_weights, combine_x
        del expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, expanded_handle
        del cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle
        del x, x_bf16, topk_idx, topk_weights
        torch.cuda.synchronize()
        buffer.destroy()


if __name__ == "__main__":
    main()
