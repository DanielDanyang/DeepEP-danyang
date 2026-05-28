# DeepEP / NCCL GIN Worklog

## 2026-05-27 设备空闲检查

- 在 `p5en_0` 和 `p5en_1` 上检查 `nvidia-smi`。
- 两台机器 8 张 GPU 均为 `0 MiB` 显存占用、`0%` GPU util。
- 未发现 `sglang`、`test_ep.py`、`gin_proxy_bench`、`all_reduce_perf`、`alltoall_perf` 等测试/训练进程。

## 2026-05-27 DeepEP 性能基线

单机 EP8：

- 命令：`tests/elastic/test_ep.py --num-processes 8 --test-first-only`
- 环境：`EP_DISABLE_GIN=1`
- 结果：
  - dispatch bottleneck: `331 GB/s (NVLink/SU)`
  - combine bottleneck: `343 GB/s (NVLink/SU)`
- 日志：`/tmp/deepep_perf_single_ep8.log`

双机 EP16：

- 命令：`tests/elastic/test_ep.py --num-processes 8 --test-first-only --num-sms 20`
- 环境：aws-ofi-nccl master `git-c8a3df2`，proxy GIN，`OFI_NCCL_FORCE_NUM_RAILS=4`
- 结果：
  - dispatch bottleneck: `5 GB/s (RDMA/SO)`
  - combine bottleneck: `15 GB/s (RDMA/SO)`
- 日志：
  - `/tmp/deepep_perf_dual_ep16_rank0.log`
  - `/tmp/deepep_perf_dual_ep16_rank1.log`

## 2026-05-27 普通 NCCL/EFA 基线

构建 `nccl-tests`：

- 路径：`/home/ubuntu/efs/yzhou/playground/daniel/nccl-tests`
- NCCL：venv 中 `nvidia-nccl-cu13==2.30.4`
- OFI plugin：`/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib`

EP16 `all_reduce_perf`：

- 命令：`all_reduce_perf -b 8M -e 1G -f 2 -g 1 -n 20 -w 5`
- 1 GiB 结果：
  - `algbw ~= 237 GB/s`
  - `busbw ~= 444 GB/s`
- 日志：`/tmp/nccl_allreduce_ep16_rails4.log`

EP16 `alltoall_perf`：

- 命令：`alltoall_perf -b 8M -e 1G -f 2 -g 1 -n 20 -w 5`
- 1 GiB 结果：
  - `algbw ~= 91 GB/s`
  - `busbw ~= 85 GB/s`
- 日志：`/tmp/nccl_alltoall_ep16_rails4.log`

结论：普通 NCCL over EFA 路径不是 `5 GB/s` 的瓶颈来源。

## 2026-05-27 纯 NCCL GIN Proxy All-to-All Microbenchmark

新增源码：

- `tools/gin_proxy_bench.cu`
- 服务器二进制：`tools/gin_proxy_bench`

实现：

- MPI 启动 rank。
- NCCL 2.30.4 初始化 communicator。
- `ncclMemAlloc` 分配 send/recv buffer。
- `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册 symmetric window。
- `ncclDevCommCreate` 创建 GIN device communicator。
- device kernel 中直接调用 `ncclGin::put + ncclGin_SignalInc + waitSignal + flush`。

EP16 all-to-all，`ctas=16`，`threads=256`，`OFI_NCCL_FORCE_NUM_RAILS=4`：

| Bytes per peer | Time | Per-rank remote BW | Aggregate remote BW |
| --- | --- | --- | --- |
| 1 MiB | 3017 us | 5.21 GB/s | 83.41 GB/s |
| 2 MiB | 3210 us | 9.80 GB/s | 156.81 GB/s |
| 4 MiB | 3596 us | 17.50 GB/s | 279.92 GB/s |
| 8 MiB | 4252 us | 29.59 GB/s | 473.48 GB/s |
| 16 MiB | 6372 us | 39.49 GB/s | 631.90 GB/s |
| 32 MiB | 11552 us | 43.57 GB/s | 697.09 GB/s |
| 64 MiB | 22748 us | 44.25 GB/s | 708.01 GB/s |
| 128 MiB | 45344 us | 44.40 GB/s | 710.39 GB/s |
| 256 MiB | 90714 us | 44.39 GB/s | 710.19 GB/s |

日志：`/tmp/gin_proxy_bench_ep16_cta16_1m_256m.log`

CTA/context sweep，64 MiB/peer：

| CTAs | Per-rank remote BW |
| --- | --- |
| 1 | 43.63 GB/s |
| 2 | 43.74 GB/s |
| 4 | 43.76 GB/s |
| 8 | 43.81 GB/s |
| 16 | 44.04 GB/s |
| 32 | 43.46 GB/s |

Rails sweep，64 MiB/peer：

| Rails | Per-rank remote BW | 备注 |
| --- | --- | --- |
| 1 | 22.22 GB/s | 约为 2/4 rails 的一半 |
| 2 | 44.22 GB/s | plateau |
| 4 | 44.04 GB/s | plateau |
| 8 | N/A | `ncclCommInitRank` 阶段 aws-ofi-nccl master segfault |

结论：纯 proxy GIN 大包上限约 `44 GB/s` per-rank remote，明显高于 DeepEP dispatch 的 `5 GB/s`，但低于普通 NCCL all-to-all 的 `~91 GB/s algbw`。

## 2026-05-27 纯 NCCL GIN Proxy P2P Microbenchmark

对 `tools/gin_proxy_bench.cu` 增加 `--skip-self`，让 `np=2` 时只发跨节点 remote peer，不包含 self-put。

运行方式：

```bash
mpirun --hostfile /tmp/deepep_hosts_1slot -np 2 --map-by ppr:1:node \
  ./tools/gin_proxy_bench --min-bytes 1M --max-bytes 1G \
  --ctas 16 --threads 256 --warmup 5 --iters 20 --skip-self
```

P2P rails=1：

| Bytes | Per-rank remote BW |
| --- | --- |
| 1 MiB | 3.37 GB/s |
| 8 MiB | 16.39 GB/s |
| 64 MiB | 21.48 GB/s |
| 128 MiB | 22.00 GB/s |
| 512 MiB | 22.61 GB/s |
| 1 GiB | 23.08 GB/s |

P2P rails=2：

| Bytes | Per-rank remote BW |
| --- | --- |
| 1 MiB | 3.13 GB/s |
| 8 MiB | 22.05 GB/s |
| 64 MiB | 40.22 GB/s |
| 128 MiB | 41.64 GB/s |
| 512 MiB | 43.78 GB/s |
| 1 GiB | 44.76 GB/s |

P2P rails=4：

| Bytes | Per-rank remote BW |
| --- | --- |
| 1 MiB | 3.21 GB/s |
| 8 MiB | 22.59 GB/s |
| 64 MiB | 39.73 GB/s |
| 128 MiB | 42.12 GB/s |
| 512 MiB | 43.16 GB/s |
| 1 GiB | 43.51 GB/s |

日志：

- `/tmp/gin_proxy_p2p_rails1_1m_1g.log`
- `/tmp/gin_proxy_p2p_rails2_1m_1g.log`
- `/tmp/gin_proxy_p2p_rails4_1m_1g.log`

结论：

- 直接 P2P proxy GIN 的大包上限约 `44 GB/s` per direction。
- rails=1 约 `23 GB/s`，rails=2/4 约 `44 GB/s`，说明单个 P2P 流在 2 条有效 rail 后已经 plateau。
- rails=4 没有比 rails=2 更快。

## 2026-05-27 NIC / Rail 使用确认

使用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` 跑 P2P rails=4 小测试。

日志：`/tmp/gin_proxy_p2p_rails4_debug.log`

关键日志：

- `NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)`
- 每台机器列出 `NIC group 0..7`，每个 group 有 2 个 EFA PCI device。
- `Rank 0: 8 Net devices`，`Rank 1: 8 Net devices`
- `GPU Direct RDMA Enabled for HCA 0..7`
- `Created device with 4 rails (originally found 2 rails)`
- `Channel 00/04` 到 `Channel 03/04`
- `NET/Plugin: Loaded gin plugin Libfabric (v13)`

解释：

- aws-ofi-nccl 确认发现了每台 `16` 个 EFA NIC，并组织成 `8` 个 NCCL Net devices。
- 单个跨节点 P2P 测试不是“用满 16 个 EFA NIC”。它在 NCCL 层看到 `4` 个 channels/rails，并且 rails=2 与 rails=4 性能相同。
- EP16 all-to-all 场景可能覆盖多个 GPU 对应的多个 Net devices，但单个 GPU0-to-GPU0 P2P 流没有用满所有物理 NIC。

## 2026-05-27 整机 16 EFA NIC 的 GIN Proxy Benchmark

对 `tools/gin_proxy_bench.cu` 增加：

- `--remote-only`：每个 rank 只发跨节点 peer，不统计同节点 NVLink/self。
- `--same-local-remote-only`：每个 rank 只发给另一台机器上相同 local rank 的 peer，用来模拟 DeepEP hybrid dispatch 的 scaleout lane。
- `--message-bytes`：把总 payload 切成小 GIN put，模拟 token-sized RDMA。

EP16 remote-only，2 节点各 8 rank，每 rank 发给远端 8 个 peer，`OFI_NCCL_FORCE_NUM_RAILS=2`：

| Bytes/remote peer | Per-rank BW | Aggregate enabled BW |
| --- | ---: | ---: |
| 64 MiB | 45.21 GB/s | 723.34 GB/s |
| 128 MiB | 45.34 GB/s | 725.38 GB/s |
| 256 MiB | 45.76 GB/s | 732.19 GB/s |
| 512 MiB | 45.94 GB/s | 735.01 GB/s |
| 1 GiB | 46.26 GB/s | 740.09 GB/s |

日志：

- `/tmp/gin_proxy_ep16_remote_only_rails1_64m_1g.log`
- `/tmp/gin_proxy_ep16_remote_only_rails2_64m_1g.log`
- `/tmp/gin_proxy_ep16_remote_only_rails4_64m_1g.log`
- `/tmp/gin_proxy_ep16_remote_only_rails2_debug.log`

解释：

- `aggregate_enabled_BW` 是双向总和；单向约为一半。
- `rails=2` 时双向约 `740 GB/s`，单向每 node 约 `370 GB/s`，接近 p5en.48xlarge `16 x 200 Gbps = 400 GB/s` 的理论单向 EFA 上限。
- debug 日志确认 `found 16 nics`、`Rank 0: 8 Net devices`、`Created device with 2 rails`、`Channel 00/16` 等。
- 结论：纯 NCCL GIN proxy 对大包/聚合跨节点流量可以基本跑满 16 张 EFA NIC；DeepEP 慢不是 EFA 大包物理带宽不足。

## 2026-05-27 DeepEP Dispatch Profiling

配置：

```bash
python tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-check \
  --num-sms 20 --num-tokens 8192 --hidden 7168 \
  --num-topk 8 --num-experts 256 --ignore-local-traffic
```

环境：NCCL 2.30.4 + aws-ofi-nccl master `git-c8a3df2`，proxy GIN，`OFI_NCCL_FORCE_NUM_RAILS=2`。

干净日志：

- `/tmp/deepep_profile_ep16_clean_rank0.log`
- `/tmp/deepep_profile_ep16_clean_rank1.log`

主要结果：

| API | Typical kernel time | SO BW | SU BW |
| --- | ---: | ---: | ---: |
| dispatch | 24.0-25.1 ms | 2-3 GB/s | 14-15 GB/s |
| expanded dispatch | 24.0-24.8 ms | 2-3 GB/s | 14-15 GB/s |
| cached dispatch | 23.9-24.2 ms | 3 GB/s | 14-15 GB/s |
| combine | 10.0-13.5 ms | 9-12 GB/s | 50-67 GB/s |

PyTorch profiler trace：

- `/tmp/deepep_profile_ep16_rank0/dispatch_rank0.json`
- `/tmp/deepep_profile_ep16_rank0/cached_dispatch_rank0.json`
- `/tmp/deepep_profile_ep16_rank0/combine_rank0.json`
- `/tmp/deepep_profile_ep16_rank1/dispatch_rank8.json` 等。

rank0 trace 解析：

| Trace | Main kernel | Time over 30 iters | Per iter |
| --- | --- | ---: | ---: |
| dispatch | `hybrid_dispatch_impl` | 718.1 ms | 23.94 ms |
| dispatch | `dispatch_copy_epilogue_impl` | 98.4 ms | 3.28 ms |
| cached dispatch | `hybrid_dispatch_impl` | 718.9 ms | 23.96 ms |
| cached dispatch | `dispatch_copy_epilogue_impl` | 98.6 ms | 3.29 ms |
| combine | `hybrid_combine_impl` | 393.8 ms | 13.13 ms |
| combine | `combine_reduce_epilogue_impl` | 2.91 ms | 0.097 ms |

`spin_kernel` 来自 benchmark barrier profiling 里的 `torch.cuda._sleep`，不是 dispatch 本体。

## 2026-05-27 DeepEP-like Small-message GIN Benchmark

DeepEP hybrid dispatch 代码路径：

- `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
- 每个 scaleout warp 按 token 发送，核心是 `gin.put<ncclTeamTagRail>(..., tma_buffer.get_num_bytes<false>(), stored_dst_scaleout_rank_idx, ncclGinOptFlagsAggregateRequests)`。
- 对 p5en 两节点 EP16，`kNumScaleoutRanks=2`、`kNumScaleupRanks=8`，每个 GPU 的 RDMA scaleout peer 基本是另一台机器同 local rank 的 GPU，再由 forward warps 走本节点 NVLink 分发。
- FP8 hidden=7168 时单 token payload 是约 7-8 KB 级别，而不是多 MB contiguous payload。
- tail/count 维护还会用 `red_add_rel` / `put_value` 这类小 signal/atomic-like 操作；EFA 没有原生 RDMA atomic 和 ordering，这些需要 proxy/reordering 路径处理。

同 lane small-message GIN：EP16，每 rank 只发给远端同 local rank peer，`message_bytes=8K`：

| Total bytes/rank | Per-rank BW | Aggregate BW |
| --- | ---: | ---: |
| 8 MiB | 2.48 GB/s | 39.66 GB/s |
| 16 MiB | 3.53 GB/s | 56.51 GB/s |
| 32 MiB | 3.87 GB/s | 61.92 GB/s |
| 64 MiB | 5.09 GB/s | 81.43 GB/s |
| 128 MiB | 5.51 GB/s | 88.17 GB/s |

日志：`/tmp/gin_proxy_ep16_same_lane_8kmsg_8m_128m.log`

64 MiB/rank，message size sweep：

| Message size | Per-rank BW | Aggregate BW |
| --- | ---: | ---: |
| 4 KiB | 2.81 GB/s | 45.01 GB/s |
| 8 KiB | 5.06 GB/s | 80.93 GB/s |
| 16 KiB | 8.79 GB/s | 140.72 GB/s |
| 32 KiB | 12.53 GB/s | 200.47 GB/s |
| 64 KiB | 16.92 GB/s | 270.77 GB/s |
| 128 KiB | 18.53 GB/s | 296.43 GB/s |

日志：

- `/tmp/gin_proxy_ep16_same_lane_msg4K_64m.log`
- `/tmp/gin_proxy_ep16_same_lane_msg8K_64m.log`
- `/tmp/gin_proxy_ep16_same_lane_msg16K_64m.log`
- `/tmp/gin_proxy_ep16_same_lane_msg32K_64m.log`
- `/tmp/gin_proxy_ep16_same_lane_msg64K_64m.log`
- `/tmp/gin_proxy_ep16_same_lane_msg128K_64m.log`

结论：

- 整机 EFA 大包可以接近跑满；大量 4-16 KiB GIN put 的吞吐会掉到 `~3-9 GB/s/rank`。
- DeepEP dispatch 的 `2-3 GB/s (SO)` 与 4-8 KiB small-message GIN 结果同量级。
- 性能差的主因是 DeepEP hybrid dispatch 在 EFA 上产生大量 token-sized GPU-initiated RDMA writes 和小 signal/atomic-like 操作；EFA 的 SRD/proxy GIN 对这种细粒度、有 ordering/atomic 需求的模式开销很高。

## 2026-05-27 UCCL-EP 论文与代码阅读

本地阅读材料：

- `2512.19849v2.pdf`
- `uccl/ep/README.md`
- `uccl/ep/include/ring_buffer.cuh`
- `uccl/ep/include/uccl_ibgda.cuh`
- `uccl/ep/src/internode.cu`
- `uccl/ep/src/rdma.cpp`

关键发现：

- UCCL-EP 处理的是 DeepEP v1 + NVSHMEM/IBGDA 在 AWS EFA 上的可移植性和性能问题，但根因和当前 V2/Gin profiling 一致：EFA 不适合大量 7 KiB token 级 GPU-initiated RDMA write + remote atomic/order 控制。
- UCCL-EP 的核心不是让 CPU 拷贝 payload，而是让 GPU 通过 128-bit `TransferCmd` 把 `WRITE/ATOMIC/QUIET/BARRIER` 命令发给 CPU proxy；payload 仍在 GPU buffer，CPU proxy 负责发 GPUDirect RDMA。
- EFA 没有 native RDMA atomic，也没有 IB 那样的强 ordering；UCCL 用 `RDMA_WRITE_WITH_IMM` 携带 sequence/control 信息，并在接收端 proxy 里等对应 write 到齐后再发布 atomic/tail update。
- UCCL normal/HT 路径会把 token 组织成 chunk 后发送，常见 chunk 量级是 32 tokens；这和我们 small-message GIN sweep 得出的“4-8 KiB 很差，64-128 KiB 开始明显好转”一致。
- UCCL-EP 在 p5en README 里给出的 EP16 normal kernel 参考量级是 dispatch `~50 GB/s RDMA`、combine `~18 GB/s RDMA`；当前 V2/Gin dispatch `2-3 GB/s SO` 明显还有协议层优化空间。

已更新：

- `efa_dispatch_optimization_proposal.md` 增加 UCCL-EP 阅读结论、V2/Gin 映射表、UCCL-style EFA backend 方案、分阶段实施路线。
- 根据后续讨论修订：长期最佳方案不再以 Gin chunk/coalescing 为主线；chunk 只作为 UCCL-style backend 内部 staging/flow-control 的组织方式或临时对照实验。主线改为移植/重写 UCCL-style `TransferCmd + CPU proxy + RDMA_WRITE_WITH_IMM + receiver-side ordering` transport。

## 2026-05-27 UCCL-style DeepEP V2 AWS Backend 开发启动

本地进度提交：

- `11b9355 Add AWS UCCL EP workspace for DeepEP V2`
  - 新增 `uccl-ep/` 工作目录。
  - 从 `uccl/ep` 复制并精简 UCCL-EP proxy/RDMA/kernel 基线。
  - 删除与当前 AWS DeepEP V2 目标无关的服务集成和基线 benchmark 子目录。
  - 新增 `uccl-ep/README_DEEPEP_V2_AWS.md`。
  - 新增 `uccl-ep/deep_ep_v2_wrapper/`，按 UCCL-EP 原 `deep_ep_wrapper` 的方式准备安装同名 `deep_ep` 包，但目标 API 是 V2 `ElasticBuffer`。
- `5d68916 Add DeepEP V2 elastic proxy binding skeleton`
  - 在 native `uccl.ep` 里新增 `ElasticProxyBuffer` nanobind 绑定骨架。
  - Python `deep_ep_v2_wrapper.deep_ep.ElasticBuffer` 已连接到 `ep.ElasticProxyBuffer`。
  - 当前 dispatch/combine 仍显式 `NotImplementedError`，下一步需要把 V2 internode `gin.put`/`red_add_rel` 路径替换为 `TransferCmd` submission。

远端状态：

- 已把第一版 `uccl-ep/` 同步到 `/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/uccl-ep/`。
- 随后检查到 `p5en_0` 有其他 GPU compute 进程：`/home/ubuntu/efs/zm/mKernel/ziming/bin/python3`，占用约 `14 MiB`。
- 按约束停止所有服务器构建、测试、profiling 和 benchmark；第二个提交尚未同步到服务器。

下一步（等 GPU 空闲后）：

1. 同步 `5d68916` 到服务器。
2. 在 venv 中构建 `uccl-ep`，确认 `import uccl.ep` 和 `hasattr(uccl.ep, "ElasticProxyBuffer")`。
3. 增加 proxy RDMA microbenchmark：GPU 写 `WRITE TransferCmd`，CPU proxy 发 EFA verbs RDMA write，不经过 NCCL Gin。
4. 在 `hybrid_dispatch.cuh` 的 EFA path 中接入 `TransferCmd` 提交流程。

## 2026-05-27 UCCL proxy 路径构建与 benchmark

本地进度提交：

- `99a0d34 Fix elastic proxy skeleton build include`
- `7356d6f Vendor UCCL utility headers for uccl-ep build`
- `d6aeb26 Avoid libnuma dependency in uccl-ep build`

远端构建：

- `p5en_0` 和 `p5en_1` 都已在专用 venv `/home/ubuntu/.venvs/deepep-danyang-cu13` 中安装 `uccl.ep`。
- 构建使用 `/usr/local/cuda-13.0`，因为 PyTorch wheel 是 CUDA 13.0；服务器 `/usr/local/cuda` 默认指向 12.9，会触发 CUDA 版本不匹配。
- smoke:
  - `import uccl.ep` 成功。
  - `ElasticProxyBuffer` binding 可见。
  - `FifoProxy` 构造后可拿到 listen port。
  - `deep_ep_v2_wrapper` 可从 `PYTHONPATH` 导入，版本 `2.0.0+ucclaws`。

新增/修改：

- `uccl-ep/bench/proxy_rdma_fifo.py`
  - 只测 GPU FIFO -> CPU FifoProxy -> EFA verbs RDMA WRITE，不经过 NCCL Gin。
  - 修正 `FifoProxy`：构造时创建内部 `Proxy`，否则 Python 端无法先 exchange listen port 再 set peer meta。
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
  - 小规模 smoke test 的 topology 不再把 `torch.cuda.device_count()` 暴露成假的 scaleup 域。
  - 初步接入 UCCL legacy high-throughput `Buffer`：V2 wrapper 可通过 UCCL layout/delegation 走 CPU proxy/EFA verbs；expanded dispatch 语义仍需补齐。
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py` / `utils_uccl.py`
  - 复用 UCCL-EP 原 `deep_ep_wrapper` 的方式，把 legacy `Buffer` 暴露给 V2 wrapper。

Benchmark 结果：

1. 单 GPU 对单 GPU FIFO microbench
   - 命令：2 节点各 1 rank，`proxy_rdma_fifo.py --size-mb 512`。
   - 日志：
     - `/tmp/uccl_proxy_rdma_fifo_rank0.log`
     - `/tmp/uccl_proxy_rdma_fifo_rank1.log`
   - 配置：4 个 `FifoProxy`，每条命令 `kObjectSize=7168` bytes。
   - 结果：`41.15 Gbps` total，约 `5.14 GB/s`。
   - 解释：这是低延迟 FIFO、单 GPU0->GPU0、只看到 2 张 NIC 的路径，和 DeepEP V2 Gin small-message dispatch 同量级，不能代表 UCCL normal/high-throughput 上限。

2. UCCL-EP high-throughput EP16 internode benchmark
   - 命令：2 节点 x 8 rank，`test_internode.py --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256`。
   - 日志：
     - `/tmp/uccl_ep16_internode_rank0.log`
     - `/tmp/uccl_ep16_internode_rank1.log`
   - NIC 使用：两台机器日志都显示 16 张 EFA NIC 全部被选中，每张出现 2 次。
   - 最佳结果：
     - FP8 dispatch: `48.84 GB/s (RDMA)`, `159.41 GB/s (NVL)`, transmit `1236 us`, config `SMs=24, NVL chunk=12, RDMA chunk=20`。
     - BF16 dispatch: `59.07 GB/s (RDMA)`, `192.79 GB/s (NVL)`, transmit `1982 us`, config `SMs=24, NVL chunk=8, RDMA chunk=12`。
     - combine: `16.60 GB/s (RDMA)`, `54.17 GB/s (NVL)`, transmit `7054 us`, config `SMs=24, NVL chunk=7, RDMA chunk=32`。

当前结论：

- EFA 不是根本带宽上限；UCCL CPU proxy + EFA verbs + receiver-side ordering 在 EP16 上已经把 dispatch 拉到 `~50-60 GB/s RDMA`。
- DeepEP V2 Gin proxy dispatch 的 `2-5 GB/s` 是协议/细粒度小消息路径问题。
- 长期方向继续是把 V2 `ElasticBuffer` 的 scaleout transport 做成 UCCL-style backend；当前 wrapper delegation 已经证明 Python API 层可以挂到 UCCL 路径，剩余关键工作是补齐 V2 expanded/cached handle 语义，并把 layout/metadata 从 Python 原型下沉到 native kernel。

## 2026-05-27 V2 wrapper delegation smoke

本地进度：

- 新增 `uccl-ep/bench/v2_proxy_smoke.py`，用 V2 `deep_ep.ElasticBuffer` API 调 UCCL legacy HT `Buffer`。
- 这个 smoke 刻意避开 V2 expanded dispatch，只验证长期后端形态：`ElasticBuffer.dispatch/combine -> UCCL HT kernels -> CPU proxy -> EFA verbs`。
- 修正脚本退出顺序：先释放 CUDA tensor 引用，再销毁 UCCL buffer；不主动 `destroy_process_group()`，避免 PyTorch 在 CUDA context teardown 后析构 tensor 时触发 `invalid device context`。

远端验证：

- 命令：2 节点 x 8 rank，`v2_proxy_smoke.py --num-tokens 256 --hidden 7168 --num-topk 8 --num-experts 256 --iters 3`。
- 日志：
  - `/tmp/v2_proxy_smoke_rank0.log`
  - `/tmp/v2_proxy_smoke_rank1.log`
- 结果：
  - `rank=0/16`: `recv=(1641, 7168)`, `combined=(256, 7168)`, `dispatch_avg_ms=16.257`
  - `rank=8/16`: `recv=(1690, 7168)`, `combined=(256, 7168)`, `dispatch_avg_ms=16.216`
- 日志未见 `Traceback`、`CUDA error`、`SIGABRT`、`SIGSEGV`。

解释：

- 这不是最终性能 benchmark；当前 layout 仍由 Python 循环构造，且 wrapper 还没有 V2 expanded metadata 语义。
- 它证明了 V2 API 层可以稳定挂到 UCCL/EFA proxy transport，下一步要补齐官方 `tests/elastic/test_ep.py` 依赖的 expanded dispatch、`recv_src_metadata` 和 cached handle 语义。

## 2026-05-27 V2 official first-case compatibility

本地进度：

- `deep_ep_v2_wrapper/deep_ep/__init__.py`
  - 让 wrapper 的 `deep_ep` 包优先覆盖 `ElasticBuffer`，同时把上游 `deep_ep` 路径追加到 `__path__`，所以 `deep_ep.utils.math`、`deep_ep.utils.refs` 等官方 test 依赖仍可导入。
  - 暴露 `Buffer`、`Config`、`EventHandle`、`topk_idx_t`，补齐官方 `tests/elastic/test_ep.py` 的入口符号。
- `deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
  - 新增 `capture()`、`barrier()`、`get_theoretical_num_sms()`、`get_theoretical_num_qps()`。
  - 构造 V2 `recv_src_metadata`、scaleup recv prefix、expert prefix 和 `dst_buffer_slot_idx`。
  - 支持 expanded dispatch：用 UCCL non-expanded receive 结果生成 one-slot-per-expert 的 expanded tensor，并在 metadata 第 2 列之后记录 expanded slot。
  - 支持 expanded combine：根据 metadata slot 把 expanded 输入折回 per-token reduced tensor，再委托 UCCL legacy combine。
  - 修复 cached dispatch：UCCL legacy cached path 不返回新的 topk metadata，也不返回新 handle；V2 wrapper 保留首次 dispatch 的 `proxy_handle` 和 topk metadata。
  - 所有传给 nanobind/native 的 async/allocate 标志显式转成 `bool`，避免 Python `0/1` 与 C++ `bool` 类型不匹配。

远端验证：

- 启动方式必须从 `/tmp` 运行测试，并设置：
  - `PYTHONPATH=/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/uccl-ep/deep_ep_v2_wrapper:/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang:/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/uccl-ep/bench:$PYTHONPATH`
  - 原因：如果从 repo root 运行，Python 的当前目录会让上游 `deep_ep` 抢先被导入，wrapper 不生效。
- 命令：2 节点 x 8 rank，官方 `tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-tokens 256 --hidden 7168 --num-topk 8 --num-experts 256`。
- 日志：
  - `/tmp/v2_official_first_rank0.log`
  - `/tmp/v2_official_first_rank1.log`
- 结果：两端进程返回 `0`；日志显示已进入第一组官方 case：
  - `do_handle_copy=1`
  - `expert_alignment=128`
  - `use_fp8_dispatch=1`
  - `num_bias=0`
  - `with_previous_event=0`
  - `async_with_compute_stream=0`
  - `allocate_on_comm_stream=0`

当前限制：

- 这是 Python 原型兼容层：metadata 构造、expanded scatter/fold 仍在 Python/PyTorch 里做，不是最终性能路径。
- 官方全矩阵 case 尚未跑完；目前只保证 first-case correctness。
- 下一步性能目标仍是把这些 V2 metadata/expanded 语义下沉到 native/CUDA，并复用已验证的 UCCL HT EFA proxy 数据面。

## 2026-05-27 V2 wrapper performance probe

官方 perf 尝试：

- 命令：官方 `tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-check --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256 --num-sms 24 --ignore-local-traffic`。
- 日志：
  - `/tmp/v2_official_perf_rank0.log`
  - `/tmp/v2_official_perf_rank1.log`
- 结果：失败在 perf 打印阶段，`bench_kineto` 按 DeepEP V2 原生 kernel 名 `dispatch_impl` / `dispatch_copy_epilogue_impl` 搜索，但 UCCL HT kernel 名不同，导致测得 `t=0`，最终 `ZeroDivisionError`。
- 结论：这不是 transport 失败，而是官方 profiler 需要为 UCCL backend 增加 kernel-name adapter，或给 UCCL wrapper 单独写 benchmark 打印逻辑。

自定义 V2 wrapper smoke benchmark：

- 脚本：`uccl-ep/bench/v2_proxy_smoke.py`
- 新增输出 cached dispatch wall time：cached path 复用 V2 handle，跳过 Python layout 和 V2 metadata 重建，主要测 UCCL legacy cached dispatch 数据面。
- 4096 tokens:
  - 日志：
    - `/tmp/v2_proxy_smoke_4096_cached_rank0.log`
    - `/tmp/v2_proxy_smoke_4096_cached_rank1.log`
  - `rank=0/16`: `recv=(26718, 7168)`, `dispatch_avg_ms=21.110`, `cached_dispatch_avg_ms=2.344`
  - `rank=8/16`: `recv=(26835, 7168)`, `dispatch_avg_ms=21.213`, `cached_dispatch_avg_ms=2.085`
  - 粗略按远端 BF16 payload 估算，cached dispatch 约 `25-28 GB/s`。
- 8192 tokens:
  - 日志：
    - `/tmp/v2_proxy_smoke_8192_cached_rank0.log`
    - `/tmp/v2_proxy_smoke_8192_cached_rank1.log`
  - `rank=0/16`: `recv=(53780, 7168)`, `dispatch_avg_ms=25.920`, `cached_dispatch_avg_ms=5.584`
  - `rank=8/16`: `recv=(53412, 7168)`, `dispatch_avg_ms=25.722`, `cached_dispatch_avg_ms=5.637`
  - 粗略按远端 BF16 payload 估算，cached dispatch 约 `21 GB/s`。
- 8192 tokens, FP8 dispatch（更接近 README 配置）:
  - 命令增加 `--use-fp8-dispatch`；FP8 smoke 只测 dispatch/cached dispatch，不做 combine。
  - 日志：
    - `/tmp/v2_proxy_smoke_8192_fp8_rank0.log`
    - `/tmp/v2_proxy_smoke_8192_fp8_rank1.log`
  - `rank=0/16`: `recv=(53780, 7168)`, `combined=None`, `dispatch_avg_ms=21.786`, `cached_dispatch_avg_ms=3.034`
  - `rank=8/16`: `recv=(53412, 7168)`, `combined=None`, `dispatch_avg_ms=21.869`, `cached_dispatch_avg_ms=3.008`
  - 粗略按 README “logical bandwidth contains local rank traffic” 口径估算，cached dispatch 约 `40 GB/s`；若只算跨节点 EFA payload，则约 `20 GB/s`。

性能解释：

- uncached wrapper dispatch 被 Python `_build_legacy_layout()`、`_build_v2_metadata()`、额外 source-id all-to-all、expanded metadata/scatter 语义拖慢，不能代表 UCCL 数据面。
- cached wrapper dispatch 已明显好于 DeepEP V2 Gin proxy 的 `~5 GB/s` dispatch；FP8 8K 下约为 README CX7 EP16 `90 GB/s` 的 45% 左右，但离直接 UCCL-EP HT / CX7 上限仍有距离。
- 要继续逼近 README SM90 EP16，下一步不是调环境变量，而是：
  - 把 V2 metadata、source token id、expert prefix、expanded slot 生成下沉到 native/CUDA；
  - 避免 Python `num_recv_tokens * topk` 循环；
  - 给官方 perf 增加 UCCL kernel-name adapter，或让 UCCL backend 直接返回可计时的 event/kernel 名。

## 2026-05-27 V2 expanded payload scatter 下沉

本地代码改动：

- commit: `6584479 Move V2 expanded payload scatter to CUDA`
- 删除 wrapper 中的 Python `_make_expanded_dispatch()` indexing scatter。
- 新增 native CUDA epilogue helper：
  - `uccl-ep/include/internode.cuh`: `build_v2_expanded_payload(...)`
  - `uccl-ep/src/internode.cu`: `v2_expanded_payload_kernel`
  - `uccl-ep/src/uccl_ep.cc`: nanobind runtime method `build_v2_expanded_payload`
  - `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`: Python wrapper 只负责按最终 expert prefix 分配输出 tensor，然后调用 native kernel。
- `ElasticBuffer.dispatch(..., do_expand=True)` 现在复用 native metadata slots，并用 CUDA kernel 把 `recv_x`、FP8 scale 和 `topk_weights` scatter 到 expanded expert-major layout。

本地验证：

- `python -m py_compile uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py uccl-ep/bench/v2_proxy_smoke.py`
- `git diff --check -- uccl-ep`
- 旧 Python scatter/fallback 关键词搜索为空。

服务器状态：

- 已检查 `p5en_0` 和 `p5en_1`，两台机器 GPU 都被 `sglang::scheduler_TP*` 进程占用，每张卡约 `124176 MiB`。
- 按 AGENTS 约束，本轮没有同步服务器、没有构建、没有 benchmark。

## 2026-05-28 expanded dispatch EP16 验证

本地代码改动：

- commit: `4d09c05 Respect local world size in V2 wrapper`
  - `ElasticBuffer` 改为优先读取 `LOCAL_WORLD_SIZE`，避免 2 节点 x 2 rank smoke 被误判为 `Ranks: 1 x 4`。
- commit: `759bc85 Preserve V2 source token metadata through NVL`
  - NVL receiver 读取完整 `SourceMeta` 四个字段，保留 `src_nvl_rank` 和 `src_token_idx`；仍对 routing bits mask 掉 RDMA epoch tag。
- commit: `745d928 Report expanded V2 dispatch timing`
  - `uccl-ep/bench/v2_proxy_smoke.py` 增加 `expanded_dispatch_avg_ms`。

服务器构建：

- 两台机器均在 `/home/ubuntu/.venvs/deepep-danyang-cu13` 中安装 wrapper。
- PyTorch 是 CUDA 13.0 build，所以扩展用 `/usr/local/cuda-13.0` 构建。
- UCCL 扩展用 `USE_DMABUF=1` 重建；EP16 日志确认 GPU RDMA buffer 通过 DMA-BUF 注册，单 rank RDMA buffer 约 `484131712` bytes。

验证结果：

- EP16 correctness：
  - 命令：`tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-tokens 256 --hidden 7168 --num-topk 8 --num-experts 256 --num-sms 20`
  - 结果：`p5en_0` 和 `p5en_1` 均 exit code `0`。
  - 覆盖普通 dispatch、expanded dispatch、cached dispatch、combine、reduced combine 的 first case correctness。
- EP16 FP8 smoke benchmark：
  - 命令：`torchrun --nnodes=2 --nproc_per_node=8 ... uccl-ep/bench/v2_proxy_smoke.py --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 --iters 3 --use-fp8-dispatch`
  - 日志：
    - `/tmp/v2_proxy_smoke_8192_fp8_expanded_rank0.log`
    - `/tmp/v2_proxy_smoke_8192_fp8_expanded_rank1.log`
  - `rank=0/16`: `recv=(53780, 7168)`, `dispatch_avg_ms=4.009`, `expanded_dispatch_avg_ms=5.281`, `cached_dispatch_avg_ms=3.088`
  - `rank=8/16`: `recv=(53412, 7168)`, `dispatch_avg_ms=3.874`, `expanded_dispatch_avg_ms=5.124`, `cached_dispatch_avg_ms=3.098`

备注：

- 2 节点 x 2 rank smoke 仍不适合当前 UCCL legacy HT path，因为 `Config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)` 对 `num_ranks < 8` 返回 0；EP16 是当前有效目标形态。
- 最后检查两台机器 `nvidia-smi --query-compute-apps` 为空，没有残留 benchmark 进程。

## 2026-05-28 README 风格 EP16 benchmark

本地代码改动：

- commit: `1635d8b Print README-style V2 proxy bandwidth`
- `uccl-ep/bench/v2_proxy_smoke.py` 增加 README 类似的 logical bandwidth 输出：
  - dispatch / expanded dispatch / cached dispatch / combine
  - 同时打印 `GB/s (SO)`、`GB/s (SU)`、平均耗时和 logical bytes。
- 说明：这里仍是 UCCL wrapper 的端到端 wall-time 计时，不是官方 `tests/elastic/test_ep.py` 里的 `bench_kineto` kernel-name 精确计时；官方 perf 脚本目前匹配的是 `dispatch_impl` / `combine_impl` 等 DeepEP kernel 名，不能直接识别 UCCL backend 的 kernel。

服务器 benchmark：

- 两台机器运行前 `nvidia-smi --query-compute-apps` 均为空。
- 配置：
  - EP16: `torchrun --nnodes=2 --nproc_per_node=8`
  - `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 --use-fp8-dispatch`
  - `OFI_NCCL_FORCE_NUM_RAILS=4`
  - 未开启 `--ignore-local-traffic`，即按 README 默认口径把 logical local rank traffic 也计入带宽。
- 稳定数据使用 `--iters 10`：
  - 日志：
    - `/tmp/v2_proxy_readme_8192_fp8_iters10_rank0.log`
    - `/tmp/v2_proxy_readme_8192_fp8_iters10_rank1.log`
  - `rank=0/16`:
    - smoke: `recv=(53780, 7168)`, `combined=(8192, 7168)`, `dispatch_avg_ms=4.016`, `expanded_dispatch_avg_ms=4.496`, `cached_dispatch_avg_ms=3.202`, `combine_avg_ms=17.529`
    - README 风格：
      - dispatch: `30 GB/s (SO), 100 GB/s (SU), 4016.400 us, 402704640 bytes`
      - expanded dispatch: `27 GB/s (SO), 90 GB/s (SU), 4495.832 us, 402704640 bytes`
      - cached dispatch: `38 GB/s (SO), 126 GB/s (SU), 3201.701 us, 402704640 bytes`
      - combine: `13 GB/s (SO), 44 GB/s (SU), 17529.499 us, 772711040 bytes`
  - `rank=8/16`:
    - smoke: `recv=(53412, 7168)`, `combined=(8192, 7168)`, `dispatch_avg_ms=4.197`, `expanded_dispatch_avg_ms=4.477`, `cached_dispatch_avg_ms=3.211`, `combine_avg_ms=17.888`
    - README 风格：
      - dispatch: `29 GB/s (SO), 95 GB/s (SU), 4197.211 us, 399949056 bytes`
      - expanded dispatch: `27 GB/s (SO), 89 GB/s (SU), 4476.500 us, 399949056 bytes`
      - cached dispatch: `38 GB/s (SO), 125 GB/s (SU), 3211.029 us, 399949056 bytes`
      - combine: `13 GB/s (SO), 43 GB/s (SU), 17888.255 us, 767423616 bytes`

观察：

- expanded dispatch 已达到 README SM90 EP16 量级的 `~90 GB/s (SU)`；cached dispatch 约 `125-126 GB/s (SU)`。
- combine 仍明显低，约 `43-44 GB/s (SU)`，是下一步主要瓶颈。
- 首次 `--iters 3` 中 rank0 uncached dispatch 有明显 outlier，平均到 `16.988 ms`；`--iters 10` 后回到 `4.0-4.2 ms`，因此后续记录优先使用 `iters=10` 或更高迭代数。

## 2026-05-28 DeepEP V2 风格 benchmark 修正

问题修正：

- 用户指出前一版“README 风格”不够 DeepEP V2：这个判断是对的。
- 前一版主要是 UCCL legacy transport smoke，只把 wall-time 包成 README 类似格式；没有完整按 V2 的五段路径组织，也没有 reduced combine。
- 本轮修改 `uccl-ep/bench/v2_proxy_smoke.py`：
  - 输出改成官方 `tests/elastic/test_ep.py` 的五段符号风格：`* dispatch`、`- expanded dispatch`、`# cached dispatch`、`@ combine`、`+ reduced combine`。
  - combine 输入改成按 V2 source metadata 生成的 pre-combine data，并用 `ordered_accumulate` 构造普通 combine 输入。
  - reduced combine 输入改成 expanded handle slot 语义，不再用随机 tensor 当 smoke。
  - 增加 `--num-sms` / `--num-qps` / `--expert-alignment`。
- 本轮修改 `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`：
  - `_ensure_legacy_buffer()` 不再写死 24 SM，改成使用传入 `num_sms` 或 `get_theoretical_num_sms()`。
  - dispatch / combine 会把 `num_sms` 下发到 UCCL `Buffer.set_num_sms()` 和 config 路径。

验证：

- 本地：
  - `python -m py_compile uccl-ep/bench/v2_proxy_smoke.py uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
  - `git diff --check -- uccl-ep/bench/v2_proxy_smoke.py uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
- 服务器：
  - 两台机器运行前 `nvidia-smi --query-compute-apps` 为空。
  - 已同步 Python 文件到 `/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/`，无需重建 C++ 扩展。

DeepEP V2 风格 EP16 结果：

- 命令核心参数：
  - `torchrun --nnodes=2 --nproc_per_node=8`
  - `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
  - `--iters 10 --num-sms 20 --use-fp8-dispatch`
  - 未开启 `--ignore-local-traffic`
- 日志：
  - `/tmp/v2_proxy_deepep_style_8192_fp8_sm20_iters10_rank0.log`
  - `/tmp/v2_proxy_deepep_style_8192_fp8_sm20_iters10_rank1.log`
- `rank=0/16`:
  - smoke: `dispatch_avg_ms=4.293`, `expanded_dispatch_avg_ms=8.278`, `cached_dispatch_avg_ms=3.588`, `combine_avg_ms=15.894`, `reduced_combine_avg_ms=30.417`
  - `* dispatch`: `28 GB/s (SO), 94 GB/s (SU), 4293.182 us, 402704640 bytes`
  - `- expanded dispatch`: `15 GB/s (SO), 49 GB/s (SU), 8278.449 us, 402704640 bytes`
  - `# cached dispatch`: `34 GB/s (SO), 112 GB/s (SU), 3588.032 us, 402704640 bytes`
  - `@ combine`: `15 GB/s (SO), 49 GB/s (SU), 15893.593 us, 772711040 bytes`
  - `+ reduced combine`: `8 GB/s (SO), 25 GB/s (SU), 30416.621 us, 772711040 bytes`
- `rank=8/16`:
  - smoke: `dispatch_avg_ms=4.221`, `expanded_dispatch_avg_ms=8.421`, `cached_dispatch_avg_ms=3.584`, `combine_avg_ms=15.777`, `reduced_combine_avg_ms=30.370`
  - `* dispatch`: `29 GB/s (SO), 95 GB/s (SU), 4220.511 us, 399949056 bytes`
  - `- expanded dispatch`: `15 GB/s (SO), 47 GB/s (SU), 8420.880 us, 399949056 bytes`
  - `# cached dispatch`: `34 GB/s (SO), 112 GB/s (SU), 3584.325 us, 399949056 bytes`
  - `@ combine`: `15 GB/s (SO), 49 GB/s (SU), 15777.364 us, 767423616 bytes`
  - `+ reduced combine`: `8 GB/s (SO), 25 GB/s (SU), 30370.005 us, 767423616 bytes`

结论：

- 这组结果比上一版更诚实：一旦按 DeepEP V2 的完整语义测，当前 UCCL wrapper 还不是一个真正 native V2 backend。
- 普通 dispatch 和 cached dispatch 还可以，`~94-95 GB/s (SU)` 和 `~112 GB/s (SU)`。
- expanded dispatch、combine、reduced combine 明显暴露 wrapper 方案的问题，尤其 reduced combine 只有 `~25 GB/s (SU)`。
- 下一步应该继续把 V2 expanded/reduced-combine 路径下沉到 native/CUDA，而不是再用 Python wrapper 把 V1/UCCL legacy 语义拼成 V2。

## 2026-05-28 V2 reduced-combine 下沉到 native/CUDA

代码改动：

- 新增 native CUDA helper `build_v2_reduced_combine_input`：
  - `uccl-ep/include/internode.cuh`
  - `uccl-ep/src/internode.cu`
  - `uccl-ep/src/uccl_ep.cc`
  - `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`
- `ElasticBuffer.combine(handle.do_expand=True)` 不再用 Python `x[slots].sum(dim=1)` 构造 reduced combine 输入，改为调用 CUDA kernel：
  - 输入：expanded expert-slot layout `[num_expanded_tokens, hidden]`
  - metadata：`recv_src_metadata[:, 2:2+topk]` 中的 expanded slot
  - 输出：legacy combine 需要的 per-token reduced layout `[num_recv_tokens, hidden]`
- `ElasticBuffer.get_theoretical_num_sms()` 不再固定返回 24：
  - 移植 DeepEP V2 的 bandwidth model，根据 `num_experts/topk/scaleout/scaleup/RDMA/NVLink` 推导 SM 数。
  - AWS 默认 `EP_RDMA_GBS=400`、`EP_NVLINK_GBS=900`；可用环境变量覆盖。
  - 因为当前数据面仍复用 UCCL legacy/V1 staging buffer，自动 SM 最后加 `EP_UCCL_MAX_AUTO_SMS` 上限，默认 32，避免 V1 buffer layout 的 int32 size 限制。显式 `--num-sms` 不走这个自动上限。

构建：

- 两台机器都重新安装 UCCL extension：
  - `CUDA_HOME=/usr/local/cuda-13.0`
  - `USE_DMABUF=1`
  - `MAX_JOBS=16`
- 构建日志：
  - `/tmp/uccl_ep_build_v2_native_rank0.log`
  - `/tmp/uccl_ep_build_v2_native_rank1.log`

验证：

- 错误启动记录：最初误用 `torchrun --nproc_per_node=8 tests/elastic/test_ep.py --num-processes 8`，导致每节点 64 个进程，NCCL 报 `too many XML nodes (max 256)`；这不是代码语义错误。
- 正确 correctness 启动方式：两台各跑一个 Python 进程，由 `test_ep.py --num-processes 8` 内部 spawn 本机 8 个 local ranks。
- EP16 correctness 通过：
  - `tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-tokens 256 --hidden 7168 --num-topk 8 --num-experts 256`
  - 日志：
    - `/tmp/v2_native_reduced_correctness_cap_rank0.log`
    - `/tmp/v2_native_reduced_correctness_cap_rank1.log`

DeepEP V2 风格 EP16 benchmark，自动 SM：

- 命令核心参数：
  - `torchrun --nnodes=2 --nproc_per_node=8`
  - `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
  - `--iters 10 --use-fp8-dispatch`
  - 未传 `--num-sms`，自动选择 `#SM=32`、`#QPs=513/0`
- 日志：
  - `/tmp/v2_native_auto_sm_bench_rank0.log`
  - `/tmp/v2_native_auto_sm_bench_rank1.log`
- `rank=0/16`:
  - `* dispatch`: `20 GB/s (SO), 66 GB/s (SU), 6106.753 us, 402704640 bytes`
  - `- expanded dispatch`: `31 GB/s (SO), 102 GB/s (SU), 3934.980 us, 402704640 bytes`
  - `# cached dispatch`: `46 GB/s (SO), 151 GB/s (SU), 2667.677 us, 402704640 bytes`
  - `@ combine`: `13 GB/s (SO), 42 GB/s (SU), 18186.205 us, 772711040 bytes`
  - `+ reduced combine`: `10 GB/s (SO), 35 GB/s (SU), 22370.132 us, 772711040 bytes`
- `rank=8/16`:
  - `* dispatch`: `20 GB/s (SO), 67 GB/s (SU), 6000.972 us, 399949056 bytes`
  - `- expanded dispatch`: `30 GB/s (SO), 99 GB/s (SU), 4053.083 us, 399949056 bytes`
  - `# cached dispatch`: `47 GB/s (SO), 155 GB/s (SU), 2577.602 us, 399949056 bytes`
  - `@ combine`: `13 GB/s (SO), 42 GB/s (SU), 18293.443 us, 767423616 bytes`
  - `+ reduced combine`: `10 GB/s (SO), 34 GB/s (SU), 22382.804 us, 767423616 bytes`

观察：

- native reduced-combine gather 把 reduced combine 从上一版 `~30.4 ms / ~25 GB/s (SU)` 提升到 `~22.4 ms / ~34-35 GB/s (SU)`。
- expanded dispatch 在自动 `#SM=32` 下回到 `~99-102 GB/s (SU)`，cached dispatch 到 `~151-155 GB/s (SU)`。
- 普通 uncached dispatch 变慢到 `~66-67 GB/s (SU)`，主要因为自动 SM=32 改变了 legacy dispatch config 和 layout/metadata overhead；后续需要把 uncached layout/metadata 也继续 native 化，而不是依赖 V1 layout path。
- combine 本身仍只有 `~42 GB/s (SU)`，说明下一个大瓶颈是 UCCL legacy combine 数据面和 V2 combine/reduce 语义没有真正融合。

## 2026-05-28 清理 V1/NVSHMEM 暴露面，转向 native V2 backend

代码清理：

- 删除旧 DeepEP V1 wrapper：
  - `uccl-ep/deep_ep_wrapper/`
- 删除旧 V1/LL benchmark：
  - `uccl-ep/bench/buffer.py`
  - `uccl-ep/bench/test_intranode.py`
  - `uccl-ep/bench/test_internode.py`
  - `uccl-ep/bench/test_low_latency*.py`
  - `uccl-ep/bench/test_dual_mode.py`
  - `uccl-ep/bench/run_ep.sh`
  - `uccl-ep/bench/utils.py`
- `deep_ep_v2_wrapper/deep_ep/buffer.py` 改为内部
  `deep_ep_v2_wrapper/deep_ep/proxy_transport.py`，不再作为 public `Buffer`
  API 暴露。
- `deep_ep_v2_wrapper/deep_ep/__init__.py` 只导出 `ElasticBuffer` / V2
  handle，不再导出旧 `Buffer`。
- 原来软链接到 `bench/utils.py` 的 `utils_uccl.py` 改成真实精简模块，只保留
  proxy 初始化、销毁、拓扑检测、FP8 dtype 等 V2 backend 必需逻辑。
- CUDA facade 中 `nvshmemi_*` 命名改为 `uccl_proxy_*`：
  - `uccl_proxy_put_nbi_warp`
  - `uccl_proxy_amo_nonfetch_add`
  - `uccl_proxy_quiet`
  - `uccl_proxy_sync_same_gpu_idx`
- 删除低延迟 V1 native path：
  - `uccl-ep/src/internode_ll.cu`
  - `uccl-ep/include/internode_ll.cuh`
  - `uccl-ep/src/uccl_ep.cc` 中的 `low_latency_dispatch` /
    `low_latency_combine` / `clean_low_latency_buffer` methods 和 bindings
  - `uccl-ep/Makefile` 不再编译 `internode_ll.cu`
- `ElasticBuffer` 中 V2 metadata / expanded payload / reduced combine input
  已经直接调用 `ElasticProxyBuffer` native helper，而不是通过旧 wrapper
  helper。

构建：

- 两台机器均在隔离 venv 中重新安装 `uccl.ep`：
  - `cd /home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/uccl-ep`
  - `CUDA_HOME=/usr/local/cuda-13.0`
  - `USE_DMABUF=1`
  - `MAX_JOBS=16`
- 构建结果：
  - `p5en_0`: 通过，14 个 source、107 个 header，日志
    `/tmp/uccl_ep_build_native_v2_no_ll_p5en0.log`
  - `p5en_1`: 通过，14 个 source、107 个 header，日志
    `/tmp/uccl_ep_build_native_v2_no_ll_p5en1.log`

API 验证：

- 两台机器均从 `uccl-ep/deep_ep_v2_wrapper` 独立 import：
  - `deep_ep.__file__` 指向
    `/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/uccl-ep/deep_ep_v2_wrapper/deep_ep/__init__.py`
  - `hasattr(deep_ep, "Buffer") == False`
  - `ElasticProxyBuffer` 暴露
    `build_v2_dispatch_metadata` / `build_v2_expanded_payload` /
    `build_v2_reduced_combine_input` / `get_comm_stream`
  - `ep.Buffer` 不再暴露 `low_latency_dispatch` / `low_latency_combine`

额外 smoke 尝试：

- `torchrun --standalone --nproc_per_node=1 uccl-ep/bench/v2_proxy_smoke.py ...`
  失败原因：当前 config map 不支持 EP1。
- `torchrun --standalone --nproc_per_node=2 uccl-ep/bench/v2_proxy_smoke.py ...`
  失败原因：单机 intranode transport handle 没有 internode `SourceMeta`，
  而当前 V2 metadata helper 仍按 internode handle 读取 `transport_handle[9]`。
- 这说明清理后的主目标已经转向 AWS EP16 internode V2 backend；如果还要保留单机
  EP2/EP8 smoke，需要补一个 intranode V2 source-metadata native helper，不能再靠
  旧 wrapper 语义兜底。

## 2026-05-28 EP 8 x 2 指标口径修正

口径修正：

- 当前双机测试拓扑是 `EP 8 x 2`：每台 p5en 机器 8 个 local ranks，一共 2 个
  scaleout 节点。
- `v2_proxy_smoke.py` 同时打印：
  - `SO`: scale-out，跨节点 EFA/RDMA 方向，是用户要看的 `EP 8 x 2`
    网络指标。
  - `SU`: scale-up，同节点 NVLink 方向，只能反映本机 8 卡内部转发/聚合。
- 之前记录里多次强调 `SU`，这是汇报口径错误；后续 EP 8 x 2 性能目标以
  `SO/RDMA` 为主。

README 形状、RDMA-only benchmark：

- 命令核心参数：
  - `torchrun --nnodes=2 --nproc_per_node=8`
  - `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
  - `--iters 10 --use-fp8-dispatch --ignore-local-traffic`
- 日志：
  - `/tmp/v2_native_ep8x2_rdma_only_rank0.log`
  - `/tmp/v2_native_ep8x2_rdma_only_rank1.log`
- `rank=0/16`:
  - `* dispatch`: `16 GB/s (SO)`, `3752.333 us`
  - `- expanded dispatch`: `16 GB/s (SO)`, `3875.835 us`
  - `# cached dispatch`: `22 GB/s (SO)`, `2779.841 us`
  - `@ combine`: `8 GB/s (SO)`, `15544.473 us`
  - `+ reduced combine`: `6 GB/s (SO)`, `18769.342 us`
- `rank=8/16`:
  - `* dispatch`: `16 GB/s (SO)`, `3719.465 us`
  - `- expanded dispatch`: `15 GB/s (SO)`, `4071.116 us`
  - `# cached dispatch`: `22 GB/s (SO)`, `2758.616 us`
  - `@ combine`: `8 GB/s (SO)`, `15575.101 us`
  - `+ reduced combine`: `6 GB/s (SO)`, `18936.930 us`

结论：

- 真正按 `EP 8 x 2` 的 RDMA/SO 口径看，当前 native V2 UCCL wrapper 仍很低，
  远低于 README SM90 CX7 `EP 8 x 2` 的 dispatch `90 GB/s`、combine `81 GB/s`
  目标。
- 当前主要瓶颈不是同节点 scale-up，而是跨节点 EFA 数据面的 V2 combine/dispatch
  语义仍复用了 legacy token/chunk staging 与 per-token head/tail 协议。

## 2026-05-28 native V2 runtime 统一与 EP 8 x 2 复测

代码清理：

- 删除 C++ 里的轻量 `ElasticProxyBuffer` shim，不再维护一个只负责 V2
  metadata/epilogue 的独立 comm stream。
- `ElasticBuffer` 初始化时只计算逻辑拓扑，不再创建单独 shim runtime；
  第一次 dispatch/combine 时通过 `ProxyTransport` 创建
  `NativeElasticProxyBuffer`，之后 metadata、expanded payload、reduced-combine
  input 和 UCCL proxy 数据面共用同一个 native runtime/comm stream。
- `uccl.ep.Buffer` 不再以 public 名称暴露；legacy base 仅以
  `uccl.ep._LegacyProxyBuffer` 存在，用作 `NativeElasticProxyBuffer` 的内部
  nanobind base。
- `uccl.ep.ElasticProxyBuffer` 已移除；`uccl.ep.NativeElasticProxyBuffer`
  暴露并继承 V2 metadata/expanded/reduced helper。
- `ProxyTransport` 增加 EFA chunk config 环境变量，方便后续稳定扫参：
  - `EP_UCCL_NVL_SEND_TOKENS`
  - `EP_UCCL_NVL_RECV_TOKENS`
  - `EP_UCCL_RDMA_SEND_TOKENS`
  - `EP_UCCL_RDMA_RECV_TOKENS`
  - `EP_UCCL_COMBINE_NVL_SEND_TOKENS`
  - `EP_UCCL_COMBINE_NVL_RECV_TOKENS`
  - `EP_UCCL_COMBINE_RDMA_SEND_TOKENS`
  - `EP_UCCL_COMBINE_RDMA_RECV_TOKENS`

构建与 API 验证：

- 两台机器均重新安装 `uccl.ep`：
  - `p5en_0`: `/tmp/uccl_ep_build_native_runtime_unified_bind_p5en0.log`
  - `p5en_1`: `/tmp/uccl_ep_build_native_runtime_unified_bind_p5en1.log`
- `/tmp` 下 import wrapper 验证：
  - `deep_ep.__file__` 指向
    `uccl-ep/deep_ep_v2_wrapper/deep_ep/__init__.py`
  - `hasattr(deep_ep, "Buffer") == False`
  - `hasattr(ep, "Buffer") == False`
  - `hasattr(ep, "ElasticProxyBuffer") == False`
  - `hasattr(ep, "NativeElasticProxyBuffer") == True`
  - `hasattr(ep.NativeElasticProxyBuffer,
    "build_v2_intranode_dispatch_metadata") == True`

正确性 smoke：

- 单机 EP2：
  - 命令：`torchrun --standalone --nproc_per_node=2
    uccl-ep/bench/v2_proxy_smoke.py --num-tokens 16 --hidden 128
    --num-topk 2 --num-experts 8 --iters 1 --num-sms 4`
  - 日志：`/tmp/v2_native_unified_ep2_smoke.log`
  - 通过。
  - rank0：dispatch `0.641 ms`，expanded dispatch `0.542 ms`，
    cached dispatch `0.142 ms`，combine `0.110 ms`，reduced combine
    `0.188 ms`。

README 形状 EP 8 x 2 RDMA/SO 复测：

- 命令核心参数：
  - `torchrun --nnodes=2 --nproc_per_node=8`
  - `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
  - `--iters 5 --use-fp8-dispatch --ignore-local-traffic`
  - 默认 EFA config：RDMA send `20` tokens，RDMA recv `512` tokens。
- 日志：
  - `/tmp/v2_native_unified_ep8x2_rank0.log`
  - `/tmp/v2_native_unified_ep8x2_rank1.log`
- `rank=0/16`：
  - `* dispatch`: `11 GB/s (SO)`, `62 GB/s (SU)`, `5633.701 us`
  - `- expanded dispatch`: `15 GB/s (SO)`, `85 GB/s (SU)`,
    `4147.691 us`
  - `# cached dispatch`: `23 GB/s (SO)`, `133 GB/s (SU)`,
    `2633.618 us`
  - `@ combine`: `7 GB/s (SO)`, `42 GB/s (SU)`, `15975.414 us`
  - `+ reduced combine`: `6 GB/s (SO)`, `35 GB/s (SU)`, `19417.827 us`
- `rank=8/16`：
  - `* dispatch`: `11 GB/s (SO)`, `62 GB/s (SU)`, `5682.878 us`
  - `- expanded dispatch`: `15 GB/s (SO)`, `88 GB/s (SU)`,
    `3995.730 us`
  - `# cached dispatch`: `24 GB/s (SO)`, `134 GB/s (SU)`,
    `2602.202 us`
  - `@ combine`: `7 GB/s (SO)`, `42 GB/s (SU)`, `15845.487 us`
  - `+ reduced combine`: `6 GB/s (SO)`, `35 GB/s (SU)`, `19282.031 us`

chunk sweep 观察：

- 尝试把 RDMA chunk 调到 send `64` tokens、recv `1024` tokens：
  - 环境变量：
    `EP_UCCL_RDMA_SEND_TOKENS=64`,
    `EP_UCCL_RDMA_RECV_TOKENS=1024`,
    `EP_UCCL_COMBINE_RDMA_SEND_TOKENS=64`,
    `EP_UCCL_COMBINE_RDMA_RECV_TOKENS=1024`
  - 日志：
    `/tmp/v2_native_unified_ep8x2_chunk64_rank0.log`,
    `/tmp/v2_native_unified_ep8x2_chunk64_rank1.log`
  - 结果：失败，多个 rank SIGABRT，根因日志是
    `CUDA error: an illegal memory access was encountered`。
- 结论：当前 legacy UCCL internode kernels 对 chunk 尺寸有隐含布局/流控约束，
  不能简单靠放大 chunk 跑满 EFA；下一步应继续真正重写 V2-native
  dispatch/combine 数据面，而不是在 V1 staging 参数上扫太远。
