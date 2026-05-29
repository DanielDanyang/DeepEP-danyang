import argparse
import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-topk", type=int, default=1)
    parser.add_argument("--num-sms", type=int, default=4)
    args = parser.parse_args()

    wrapper_path = os.path.join(args.repo_root, "uccl-ep", "deep_ep_v2_wrapper")
    sys.path.insert(0, wrapper_path)
    sys.path.insert(1, args.repo_root)

    import deep_ep

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    _ = torch.empty((1,), device="cuda")
    deep_ep.init_deep_ep_jit(
        os.path.join(args.repo_root, "deep_ep"),
        "/usr/local/cuda",
        deep_ep.find_nccl_root(),
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
    )

    x = torch.arange(args.tokens * args.hidden, device="cuda",
                     dtype=torch.bfloat16).reshape(args.tokens, args.hidden)
    x = x + rank * 1000
    route = (rank + 1) % world_size if world_size > 1 else rank
    topk_idx = torch.full((args.tokens, args.num_topk), route,
                          device="cuda", dtype=torch.int64)
    topk_weights = torch.ones((args.tokens, args.num_topk), device="cuda",
                              dtype=torch.float32)

    recv_x, recv_idx, recv_w, handle, _ = buffer.dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=world_size,
        num_max_tokens_per_rank=args.tokens,
        num_sms=args.num_sms,
        do_cpu_sync=True,
    )
    torch.cuda.synchronize()
    dispatch_ops = handle.transport_handle.d2h_queue.drain_ready()

    combined, combined_w, _ = buffer.combine(
        recv_x.clone(),
        handle,
        recv_w,
        num_sms=args.num_sms,
    )
    torch.cuda.synchronize()
    combine_ops = handle.transport_handle.combine_d2h_queue.drain_ready()
    ok = torch.equal(combined, x)
    weights_ok = torch.equal(combined_w, topk_weights)

    print(
        f"rank={rank} recv={tuple(recv_x.shape)} "
        f"dispatch_desc={handle.transport_handle.num_dispatch_segments}/"
        f"{handle.transport_handle.num_dispatch_batches} "
        f"dispatch_ops={len(dispatch_ops)} "
        f"combine_desc={handle.transport_handle.num_combine_segments}/"
        f"{handle.transport_handle.num_combine_batches} "
        f"combine_ops={len(combine_ops)} ok={ok} weights_ok={weights_ok}",
        flush=True,
    )
    assert ok
    assert weights_ok
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
