# DeepEP V2 on AWS EFA via UCCL-style proxy

This directory is the AWS-only DeepEP V2 EFA backend workspace.

The implementation starts from UCCL-EP because UCCL-EP already solved the
important EFA problem for DeepEP V1: GPU kernels submit compact transfer
commands, CPU proxy threads issue GPUDirect RDMA through libibverbs/EFA, and
receiver-side ordering is reconstructed with immediate data.

## Goal

Expose a DeepEP-compatible package in the same style as UCCL-EP's
`deep_ep_wrapper`, but targeting DeepEP V2 `ElasticBuffer` instead of the
legacy DeepEP `Buffer` API.

The AWS path should not use NCCL Gin for cross-node dispatch/combine data or
tail signaling. NCCL Gin remains useful as the baseline and as the upstream
DeepEP V2 implementation for IB/CX7, but the long-term AWS path here is:

```text
GPU kernel writes staging buffer
GPU kernel submits 128-bit TransferCmd
CPU proxy posts EFA verbs RDMA write / write-with-imm
receiver proxy applies ordering and publishes tail/count
GPU forwarder consumes published control state
```

## Initial Scope

- AWS EFA only.
- CUDA/Hopper first.
- EP16 first: two p5en nodes, eight H200 GPUs per node.
- DeepEP V2 `ElasticBuffer.dispatch` and `ElasticBuffer.combine` first.
- Engram, PP, AGRS, ROCm, Broadcom, Intel NIC, SGLang, vLLM, and Megatron
  integration are intentionally out of scope for this workspace.

## Current Layout

- `include/`, `src/`: copied UCCL-EP proxy/RDMA/kernel implementation.
- `bench/`: trimmed UCCL-EP EP benchmarks used as references while porting.
- `deep_ep_wrapper/`: original UCCL-EP DeepEP V1-style wrapper kept as a
  reference.
- `deep_ep_v2_wrapper/`: DeepEP V2 wrapper/adapter layer under development.

## Development Phases

1. Build the copied UCCL-EP core in the DeepEP server environment.
2. Add a standalone proxy RDMA microbenchmark that does not depend on NCCL Gin.
3. Implement `deep_ep_v2_wrapper.deep_ep.ElasticBuffer` with V2-compatible
   constructor, dispatch, combine, and handle objects.
4. Replace cross-node V2 Gin operations with `TransferCmd` submission.
5. Add receiver-side ordering and host-mapped tail/count publication.
6. Run `tests/elastic/test_ep.py` EP16 and iterate toward the UCCL-EP p5en
   reference bandwidth.

