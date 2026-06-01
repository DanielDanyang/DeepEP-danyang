import argparse
import os
import sys

import torch
import torch.distributed as dist


def _repo_root() -> str:
    return os.environ.get(
        "DEEPEP_REPO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )


def _assert_equal(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(got.cpu(), expected.cpu()):
        raise AssertionError(
            f"{name} mismatch: got={got.cpu().tolist()} expected={expected.cpu().tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--experts", type=int, default=0)
    parser.add_argument("--sms", type=int, default=8)
    parser.add_argument("--window-mb", type=int, default=512)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument(
        "--remote-pair",
        action="store_true",
        help="route each local rank to the same local rank on the other node",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, os.path.join(repo_root, "uccl-ep", "deep_ep_v2_wrapper"))
    sys.path.insert(1, repo_root)

    import deep_ep
    from deep_ep.buffers.elastic import ElasticBuffer

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    experts = args.experts or world
    assert experts % world == 0
    assert experts // world == 1, "this correctness test expects one local expert per rank"
    if args.remote_pair:
        assert world % 2 == 0 and local_world * 2 == world
        dst_rank = (rank + local_world) % world
        src_rank = (rank + local_world) % world
    else:
        dst_rank = (rank + 1) % world
        src_rank = (rank - 1 + world) % world

    deep_ep.init_deep_ep_jit(
        os.path.join(repo_root, "deep_ep"),
        os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        deep_ep.find_nccl_root(),
    )

    buf = ElasticBuffer(
        dist.group.WORLD,
        num_bytes=args.window_mb << 20,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=1,
    )
    local_info = buf.init_native_v2_efa_transport(
        num_bytes=args.window_mb << 20,
        num_lanes=args.lanes,
    )

    base = torch.arange(args.tokens * args.hidden, dtype=torch.float32, device="cuda")
    x = (base.reshape(args.tokens, args.hidden) + rank * 10000).to(torch.bfloat16)
    topk_idx = torch.full((args.tokens, 1), dst_rank, dtype=torch.int64, device="cuda")
    topk_weights = torch.full((args.tokens, 1), rank + 0.5, dtype=torch.float32, device="cuda")

    recv_x, recv_idx, recv_w, handle, _ = buf.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=experts,
        num_max_tokens_per_rank=args.tokens,
        num_sms=args.sms,
        do_cpu_sync=True,
    )
    torch.cuda.synchronize()

    expected_x = (
        torch.arange(args.tokens * args.hidden, dtype=torch.float32, device="cuda")
        .reshape(args.tokens, args.hidden)
        .add(src_rank * 10000)
        .to(torch.bfloat16)
    )
    expected_src = (
        torch.arange(args.tokens, dtype=torch.int32, device="cuda") +
        src_rank * args.tokens
    )
    _assert_equal("recv_x", recv_x[: args.tokens], expected_x)
    _assert_equal("recv_idx", recv_idx[: args.tokens], torch.zeros_like(recv_idx[: args.tokens]))
    _assert_equal("recv_src", handle.recv_src_metadata[: args.tokens, 0], expected_src)
    expected_w = torch.full((args.tokens, 1), src_rank + 0.5, dtype=torch.float32, device="cuda")
    if not torch.allclose(recv_w[: args.tokens], expected_w):
        raise AssertionError(f"recv_w mismatch on rank {rank}")
    assert int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()) == args.tokens
    assert int(handle.psum_num_recv_tokens_per_expert[-1].item()) == args.tokens

    expanded_x, expanded_idx, expanded_w, expanded_handle, _ = buf.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=experts,
        num_max_tokens_per_rank=args.tokens,
        num_sms=args.sms,
        do_cpu_sync=True,
        do_expand=True,
    )
    torch.cuda.synchronize()
    assert expanded_idx is None
    _assert_equal("expanded_x", expanded_x[: args.tokens], expected_x)
    if not torch.allclose(expanded_w[: args.tokens], expected_w):
        raise AssertionError(f"expanded_w mismatch on rank {rank}")
    assert int(expanded_handle.psum_num_recv_tokens_per_expert[-1].item()) == args.tokens

    if rank == 0:
        print(
            f"dispatch_correctness_ok EP{world} tokens={args.tokens} hidden={args.hidden} "
            f"remote_pair={args.remote_pair} device={local_info.get('device_name')} "
            f"stats={handle.transport_handle.dispatch_drain_stats}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
