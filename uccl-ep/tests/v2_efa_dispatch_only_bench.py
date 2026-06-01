import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def _gbps(num_bytes: int, seconds: float) -> float:
    return num_bytes / max(seconds, 1e-12) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--sms", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--window-mb", type=int, default=512)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--remote-pair", action="store_true")
    parser.add_argument("--do-expand", action="store_true")
    args = parser.parse_args()

    repo_root = os.environ.get(
        "DEEPEP_REPO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    sys.path.insert(0, os.path.join(repo_root, "uccl-ep", "deep_ep_v2_wrapper"))
    sys.path.insert(1, repo_root)

    import deep_ep
    from deep_ep.buffers.elastic import ElasticBuffer

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    deep_ep.init_deep_ep_jit(
        os.path.join(repo_root, "deep_ep"),
        os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        deep_ep.find_nccl_root(),
    )

    assert args.experts % world == 0
    assert args.topk <= args.experts // world
    if args.remote_pair:
        assert world % 2 == 0 and local_world * 2 == world
    buf = ElasticBuffer(
        dist.group.WORLD,
        num_bytes=args.window_mb << 20,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
    )
    local_info = buf.init_native_v2_efa_transport(
        num_bytes=args.window_mb << 20,
        num_lanes=args.lanes,
    )

    x = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    dst_rank = (rank + local_world) % world if args.remote_pair else (rank + 1) % world
    expert_begin = dst_rank * (args.experts // world)
    topk_idx = torch.arange(
        expert_begin,
        expert_begin + args.topk,
        dtype=torch.int64,
        device="cuda",
    ).reshape(1, args.topk).repeat(args.tokens, 1)
    topk_weights = torch.ones((args.tokens, args.topk), dtype=torch.float32, device="cuda")

    for _ in range(args.warmup):
        recv_x, recv_idx, recv_w, handle, _ = buf.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=args.experts,
            num_max_tokens_per_rank=args.tokens,
            num_sms=args.sms,
            do_cpu_sync=True,
            do_expand=args.do_expand,
        )
        assert int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()) == args.tokens * args.topk
    torch.cuda.synchronize()
    dist.barrier()

    begin = time.perf_counter()
    last_handle = None
    for _ in range(args.iters):
        recv_x, recv_idx, recv_w, last_handle, _ = buf.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=args.experts,
            num_max_tokens_per_rank=args.tokens,
            num_sms=args.sms,
            do_cpu_sync=True,
            do_expand=args.do_expand,
        )
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - begin
    avg = elapsed / args.iters
    payload_bytes = args.tokens * args.topk * args.hidden * 2
    record_bytes = int(last_handle.transport_handle.dispatch_layout["token_record_bytes"]) * args.tokens * args.topk
    stats = last_handle.transport_handle.dispatch_drain_stats
    timings = last_handle.transport_handle.timings or {}
    if rank == 0:
        print(
            f"dispatch-only EP{world} tokens={args.tokens} hidden={args.hidden} topk={args.topk} "
            f"sms={args.sms} remote_pair={args.remote_pair} do_expand={args.do_expand} "
            f"lanes={args.lanes} device={local_info.get('device_name')} "
            f"avg_us={avg * 1e6:.2f} "
            f"payload_GBps={_gbps(payload_bytes, avg):.2f} "
            f"record_GBps={_gbps(record_bytes, avg):.2f} stats={stats} "
            f"timings={timings}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
