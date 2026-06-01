import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    repo_root = os.environ.get(
        "DEEPEP_REPO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    sys.path.insert(0, os.path.join(repo_root, "uccl-ep", "deep_ep_v2_wrapper"))
    sys.path.insert(1, repo_root)

    import deep_ep
    from deep_ep.buffers.elastic import ElasticBuffer

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    deep_ep.init_deep_ep_jit(
        os.path.join(repo_root, "deep_ep"),
        os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        deep_ep.find_nccl_root(),
    )

    world = dist.get_world_size()
    assert world == 2, "dispatch-only smoke currently expects EP1x2"
    num_tokens = 1
    hidden = 16
    num_topk = 1
    buf = ElasticBuffer(
        dist.group.WORLD,
        num_bytes=65536,
        num_max_tokens_per_rank=4,
        hidden=hidden,
        num_topk=num_topk,
    )
    buf.init_native_v2_efa_transport(num_bytes=65536, num_lanes=1)

    x = (torch.arange(hidden, dtype=torch.uint8, device="cuda") + rank * 64).reshape(num_tokens, hidden)
    topk_idx = torch.tensor([[1 - rank]], dtype=torch.int64, device="cuda")
    topk_weights = torch.full((num_tokens, num_topk), 0.25 + rank, dtype=torch.float32, device="cuda")

    recv_x, recv_idx, recv_w, handle, _ = buf.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=world,
        num_max_tokens_per_rank=4,
        do_cpu_sync=True,
    )
    torch.cuda.synchronize()
    expected_payload = (torch.arange(hidden, dtype=torch.uint8) + (1 - rank) * 64).tolist()
    expected_weight = 0.25 + (1 - rank)
    got_payload = recv_x.reshape(-1).cpu().tolist()
    got_idx = recv_idx.reshape(-1).cpu().tolist()
    got_w = recv_w.reshape(-1).cpu().tolist()
    got_src = handle.recv_src_metadata[:1, 0].cpu().tolist()
    expected_src = [(1 - rank) * 4]
    assert got_payload == expected_payload, (rank, got_payload, expected_payload)
    assert got_idx == [0], (rank, got_idx)
    assert abs(got_w[0] - expected_weight) < 1e-6, (rank, got_w, expected_weight)
    assert got_src == expected_src, (rank, got_src, expected_src)
    stats = handle.transport_handle.dispatch_drain_stats

    expanded_x, expanded_idx, expanded_w, expanded_handle, _ = buf.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=world,
        num_max_tokens_per_rank=4,
        do_cpu_sync=True,
        do_expand=True,
    )
    torch.cuda.synchronize()
    expanded_tokens = int(expanded_handle.psum_num_recv_tokens_per_expert[-1].item())
    assert expanded_idx is None
    assert expanded_tokens == 1, (rank, expanded_tokens)
    assert expanded_x[:1].reshape(-1).cpu().tolist() == expected_payload
    assert abs(expanded_w[:1].reshape(-1).cpu().tolist()[0] - expected_weight) < 1e-6

    print(
        f"rank={rank} native_dispatch_only_ok=True "
        f"payload={got_payload[:4]} idx={got_idx} weight={got_w[0]:.2f} "
        f"expanded={expanded_x[:1].reshape(-1).cpu().tolist()[:4]} "
        f"src={got_src} stats={stats}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
