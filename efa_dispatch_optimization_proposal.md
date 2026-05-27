# DeepEP 在 AWS EFA 上的 Dispatch 性能提升提案

## 结论摘要

当前 AWS `p5en.48xlarge` 上的慢点不是 EFA 的大包物理带宽。纯 NCCL GIN proxy 的 EP16 remote-only 大包测试可以达到双向 aggregate 约 `740 GB/s`，折合单向每 node 约 `370 GB/s`，已经接近 `16 x 200 Gbps = 400 GB/s` 的理论 EFA 上限。

DeepEP dispatch 慢的主因是：`hybrid_dispatch_impl` 在 scaleout 阶段按 token 发大量 4-8 KiB 级别的 GPU-initiated GIN put，并且伴随 tail/count 的小 signal / atomic-like 更新。AWS EFA 对大包吞吐强，但没有原生 RDMA atomic 和强 ordering；proxy GIN 处理这种小包、细粒度、带同步语义的模式时开销很高。

因此优化方向不是继续调 `num_sms`、增加 rails、或者在 Gin 上做局部合并；长期最优方案应该参照 UCCL-EP：为 DeepEP V2 增加一个 EFA 专用 transport/backend。GPU 仍然负责决定 token routing，但跨节点数据和控制语义交给 CPU proxy 通过 libibverbs/EFA 发 GPUDirect RDMA，并用 immediate data 和 receiver-side ordering 解决 EFA 缺少强 ordering/native atomic 的问题。

结合 `2512.19849v2.pdf` 里的 UCCL-EP 方案后，推荐路线应改成单一主线：不要把 `Gin chunk put` 当作主方案；它最多只是临时对照实验。真正要开发的是 UCCL-style V2 EFA backend，把当前 `gin.put` / `gin.signal` / `red_add_rel` 代表的 device-side network operation 替换成 `TransferCmd -> CPU proxy -> RDMA_WRITE_WITH_IMM -> receiver-side ordering`。

## UCCL-EP 论文和代码给 V2/Gin 的启发

论文 `2512.19849v2.pdf` 的核心判断和我们这次 profiling 对得很准：DeepEP 的跨节点 EP 通信是 7 KiB 左右的 token 级小消息，高频 dispatch/combine 会依赖 GPU-initiated RDMA 的 ordering 和 atomic 语义；AWS EFA 的 SRD 网络大包吞吐强，但没有 IB/CX7 那种适合小 RDMA write + remote atomic 的硬件语义。因此 UCCL-EP 没有继续让 GPU 直接对 NIC 发细粒度 RDMA，而是让 GPU 只发一个很小的控制命令，真正的 GPUDirect RDMA 由 CPU proxy 通过 libibverbs 发出。

UCCL-EP 的几个关键设计值得直接参考：

| UCCL-EP 设计 | 对 V2/Gin 的启发 |
| --- | --- |
| GPU 发 128-bit `TransferCmd` 到 GPU->CPU FIFO | V2 EFA path 可以把高频控制面从 `gin.signal/VASignalAdd` 改成轻量 `TransferCmd` |
| CPU proxy 负责 RDMA write / atomic / drain / barrier | EFA 上避免让 device-side GIN 直接承受大量小 signal/atomic-like 操作 |
| payload 留在 GPU buffer，只把 src/dst offset/bytes 发给 CPU | 不需要 CPU 拷贝 token 数据，仍然走 GPUDirect RDMA |
| RDMA immediate data 携带 sequence/order/control 信息 | EFA 无强 ordering 时，用 receiver-side sequence buffering 确保 write-before-tail |
| EFA atomic 用 host-mapped atomic buffer 模拟 | V2 的 tail/count 可以放到 `cudaHostAllocMapped` 控制区，GPU 轮询 host-mapped counter |
| normal/HT 模式用 ring/chunk 管理 token staging 和流控 | V2 可以借鉴 staging/ring/flow-control 结构，但最终 RDMA 发起方应是 CPU proxy，不是 Gin chunk put |
| 每 GPU 一个 proxy，多线程管理同 NUMA 组 NIC | p5en 每 GPU 附近有多张 EFA，proxy 可以显式做 NIC/QP sharding |

对应代码位置也很清楚：

- `uccl/ep/include/ring_buffer.cuh` 里的 `TransferCmd` 是 16 字节，包含 `WRITE/ATOMIC/QUIET/BARRIER` 四类命令。
- `uccl/ep/include/uccl_ibgda.cuh` 里的 `nvshmemi_ibgda_put_nbi_warp` 不再真的走 NVSHMEM/IBGDA，而是把 write 请求编码成 `TransferCmd` 推给 D2H queue。
- `uccl/ep/include/uccl_ibgda.cuh` 里的 `nvshmemi_ibgda_amo_nonfetch_add` 把 remote atomic add 也编码成 `ATOMIC TransferCmd`。
- `uccl/ep/src/internode.cu` 的 normal path 会先把 token 写入 RDMA staging buffer，再由 coordinator 按 `num_max_rdma_chunked_send_tokens` 批量发出。
- `uccl/ep/src/rdma.cpp` 在 EFA 上用 `IBV_WR_RDMA_WRITE_WITH_IMM` 把 write 和 atomic/tail 信息绑在一起，并通过 `apply_pending_updates` 在接收端等对应 write 到齐后再发布 atomic update。

这说明 UCCL-EP 的价值不只是“换了一个通信库”，而是把 EFA 最薄弱的部分绕开了：EFA 不擅长 GPU 直接发大量小 ordered write/atomic，所以 UCCL 把 ordering/atomic 变成 CPU proxy 协议，把 payload 变成较大的 RDMA write。

### UCCL-EP 到 DeepEP V2 的映射

| DeepEP V2 当前代码 | UCCL-style 替代或借鉴 |
| --- | --- |
| `gin.put<ncclTeamTagRail>(..., single_token_bytes, ...)` | 替换为 `WRITE TransferCmd`：GPU 写 staging buffer 并提交 offset/bytes，CPU proxy 发 GPUDirect RDMA write |
| `gin.red_add_rel<ncclTeamTagRail>` 发布 scaleout tail | 替换为 `ATOMIC TransferCmd` 或 proxy 管理的 host-mapped tail；接收端按 write completion/seq 再发布 |
| `gin.signal(... VASignalAdd ...)` | 从 EFA 主路径移除；Drain/barrier 也应走 proxy command，而不是高频 Gin signal |
| `gin.flush` / barrier | 映射到 UCCL `QUIET/DRAIN/BARRIER`，由 CPU proxy 确认 outstanding WR 完成 |
| NCCL symmetric window offset | 可复用为 proxy RDMA MR 的 base+offset；需要把 V2 `ElasticBuffer` 的可 RDMA 区注册给 proxy |
| `NCCL_GIN_CONNECTION_RAIL` same-lane scaleout | 保留 same-lane hierarchical 拓扑，但每 lane 的 proxy thread 管理该 GPU 近邻 EFA NIC/QP |

因此，V2 和 V1+NVSHMEM 虽然底层 API 不同，但思路可以复用：不要让 EFA 承担 IBGDA/Gin 原始协议里那种高频 device-side network operation；把 RDMA 发起、ordering、atomic、drain/barrier 全部显式协议化，并放到 CPU proxy/EFA verbs 层处理。

## 已有数据支撑

### 纯 GIN 大包能跑满整机 EFA

`tools/gin_proxy_bench.cu --remote-only`，EP16，两节点各 8 rank，每 rank 发远端 8 peers，`OFI_NCCL_FORCE_NUM_RAILS=2`：

| Size/remote peer | Per-rank BW | 双向 aggregate |
| --- | ---: | ---: |
| 64 MiB | 45.21 GB/s | 723.34 GB/s |
| 1 GiB | 46.26 GB/s | 740.09 GB/s |

这说明 NCCL GIN proxy 可以把 p5en 的 16 张 EFA NIC 基本用起来。

### DeepEP dispatch 本体很慢

README 风格 EP16 配置：

```bash
python tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-check \
  --num-sms 20 --num-tokens 8192 --hidden 7168 \
  --num-topk 8 --num-experts 256 --ignore-local-traffic
```

主要结果：

| API | Typical time | SO BW | SU BW |
| --- | ---: | ---: | ---: |
| dispatch | 24.0-25.1 ms | 2-3 GB/s | 14-15 GB/s |
| cached dispatch | 23.9-24.2 ms | 3 GB/s | 14-15 GB/s |
| combine | 10.0-13.5 ms | 9-12 GB/s | 50-67 GB/s |

rank0 trace 解析：

| Kernel | 30 iters total | Per iter |
| --- | ---: | ---: |
| `hybrid_dispatch_impl` | 718.1 ms | 23.94 ms |
| `dispatch_copy_epilogue_impl` | 98.4 ms | 3.28 ms |
| `hybrid_combine_impl` | 393.8 ms | 13.13 ms |
| `combine_reduce_epilogue_impl` | 2.91 ms | 0.097 ms |

`spin_kernel` 是 profiling barrier 里的 `torch.cuda._sleep`，不是 dispatch 本体。

### DeepEP-like 小消息 GIN 与 dispatch 同量级

`tools/gin_proxy_bench.cu --same-local-remote-only --message-bytes`，每 rank 只发给另一台机器同 local rank peer，模拟 hybrid dispatch 的 scaleout lane：

| Message size | 64 MiB/rank per-rank BW |
| ---: | ---: |
| 4 KiB | 2.81 GB/s |
| 8 KiB | 5.06 GB/s |
| 16 KiB | 8.79 GB/s |
| 32 KiB | 12.53 GB/s |
| 64 KiB | 16.92 GB/s |
| 128 KiB | 18.53 GB/s |

DeepEP FP8 `hidden=7168` 时单 token payload 约 7-8 KiB，正好落在最差区间。所以 dispatch 的 2-3 GB/s 与小消息 GIN benchmark 对齐。

## 关键问题代码

### 1. 后端无条件使用 hybrid rail GIN 路径

位置：[csrc/kernels/backend/nccl.cu](/Users/daniel/Documents/code/DeepEP-danyang/csrc/kernels/backend/nccl.cu:95)

```cpp
reqs.ginContextCount = num_allocated_qps;
reqs.ginExclusiveContexts = true;
reqs.ginQueueDepth = 1024;
reqs.ginTrafficClass = sl_idx;
reqs.ginSignalCount = num_ranks + 2 * 2;
reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
```

中文解释：

这里在 `allow_hybrid_mode=true` 时选择 `NCCL_GIN_CONNECTION_RAIL`。在两节点 EP16 上，DeepEP 的逻辑会变成：

- scaleout 维度是节点数，也就是 `2`；
- scaleup 维度是单节点 NVLink GPU 数，也就是 `8`；
- 每个 GPU 通过 rail 只对另一台机器同 local-rank 的 GPU 做跨节点 RDMA；
- 收到跨节点 token 后，再由目标节点的 forward warps 通过 NVLink 分发到真正的 local expert GPU。

这个设计在 IB / CX7 机器上很合理，因为 GPU-initiated RDMA 小写和 remote atomic 语义很强。但在 EFA 上，rail GIN 走 proxy/SRD 语义，细粒度 token put 和 signal/atomic-like 操作成本很高。当前代码没有区分 IB 与 EFA，也没有为 EFA 选择更粗粒度的数据路径。

建议在此处增加网络能力判断，例如：

- 如果是 EFA proxy GIN，保留 GIN correctness，但选择 `EFA_COALESCED_HYBRID` 或 `EFA_DIRECT_CHUNKED` dispatch path；
- 如果是 IB/GDRDMA 原生路径，继续走当前低延迟 token-streaming hybrid path。

### 2. Hybrid dispatch 固定为 token-streaming warp 布局

位置：[csrc/kernels/elastic/dispatch.hpp](/Users/daniel/Documents/code/DeepEP-danyang/csrc/kernels/elastic/dispatch.hpp:267)

```cpp
// Hybrid kernels
EP_HOST_ASSERT(not deterministic);

num_scaleout_warps = num_channels_per_sm;
num_forward_warps = num_channels_per_sm;
num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
```

中文解释：

这里把 hybrid dispatch 固定成三类 warp：

- notify warps：算 rank/expert count，通知 peer；
- scaleout warps：把 token 从本节点发到远端同 lane GPU；
- forward warps：在目标节点上把收到的 token 转发给本节点的最终 GPU。

这套布局天然偏向“token 边产生边发送”的 streaming 协议。它没有 EFA 模式需要的“先聚合再发大块”的阶段，也没有参数控制每次 RDMA 的最小 chunk 大小。结果是每个 scaleout warp 会持续发小 put，而不是把几十或几百个 token 合成一个 256 KiB / 1 MiB 的 put。

建议新增一个 EFA 专用 launch 模式，例如：

- `hybrid_dispatch_impl_efa_coalesced`；
- 或者给当前 `hybrid_dispatch_impl` 增加模板参数 `kScaleoutCoalesceTokens` / `kMinScaleoutChunkBytes`。

但是建议优先新增 kernel，而不是把现有 kernel 塞太多分支。当前 IB 路径已经比较复杂，直接改容易影响官方机器性能。

### 3. notify 阶段包含小 put 和 atomic-like 更新

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:173)

```cpp
gin.put<ncclTeamTagRail>(
    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
    ncclGinOptFlagsAggregateRequests);
gin.put<ncclTeamTagRail>(
    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);
```

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:228)

```cpp
gin.put_value<ncclTeamTagLsa>(
    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
    counter, i);
```

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:236)

```cpp
gin.red_add_rel<ncclTeamTagLsa>(
    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
    counter, dst_scaleup_rank_idx);
```

中文解释：

这些代码负责把每个节点上统计出的 rank/expert count 汇总给目标节点和目标 GPU。对于 correctness 来说它们是必要的，因为后续 copy epilogue 需要知道收到多少 token、每个 expert 的 prefix sum 是多少。

不过 profile 显示 cached dispatch 仍然接近 `24 ms`，说明 notify 阶段不是最大头。它仍然值得优化，但优先级低于 scaleout data path。

建议：

- cached mode 已经移除 notify，dispatch 仍慢，所以先不要把主要精力放在 count path；
- 后续可以把 count path 的多个小 put 合成一个小结构体块，一次发出；
- 对 EFA 避免使用 remote atomic-like `red_add_rel` 做高频更新，能用 single-writer `put_value` 的位置尽量改成 absolute value 写入。

### 4. scaleout tail 每 3 个 token 就做一次 atomic-like 更新

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:26)

```cpp
int kScaleoutUpdateInterval = 3,
int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,
```

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:333)

```cpp
const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
    if (lane_idx < kNumScaleoutRanks and
        (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag)) {
        const auto signaled_tail = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
        const auto ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx);
        const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);

        gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
        stored_old_scaleout_tail = stored_scaleout_tail;
    }
    __syncwarp();
};
```

中文解释：

这是当前 EFA 上最可疑的同步热点之一。每个 channel 的 scaleout sender 每积累 `3` 个 token 就通知一次远端 forwarder：“我这边 tail 增加了多少”。通知方式是 `red_add_rel`，语义接近 remote atomic add + release。

在 IB 上，这种细粒度 tail 更新可以让 forwarder 更早开始工作，降低延迟。但在 EFA 上：

- `3` 个 token 只有几十 KiB；
- 每次 tail 更新都需要 proxy GIN 处理 signal/atomic-like 语义；
- EFA 没有原生 RDMA atomic 和同 QP ordering，release/atomic 语义需要额外协议；
- forwarder 还会 busy-wait 这个 tail，tail 更新一慢，整个 `hybrid_dispatch_impl` 就被拉长。

建议：

1. 把 `kScaleoutUpdateInterval` 变成可配置参数，在 EFA 上默认提高到 `16/32/64` 个 token。
2. 对 `kNumScaleoutRanks=2`、每个 `(channel, source_scaleout_rank)` 单 writer 的位置，把 `red_add_rel(delta)` 改成 `put_value(absolute_tail)` 或普通 put 写 absolute tail。
3. 只有 chunk 数据全部 put 完成后再写 tail；必要时在同一个 GIN context 上做一次 `flush`，然后写 tail，避免 forwarder 提前读到未完成数据。
4. 如果用 absolute tail，receiver 可以接受跳过中间 tail 值，因为它只需要看到单调增长的最大可消费位置。

这是一个低侵入优化点，预计可以先验证 tail-update 开销是否显著。

### 5. scaleout data path 按 token 发 7-8 KiB put

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:382)

```cpp
for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
    ...
    if (scaleout_rank_mask ^ (1 << scaleout_rank_idx)) {
        ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                          tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
    }
    ...
    if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
        gin.put<ncclTeamTagRail>(
                scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                tma_buffer.get_num_bytes<false>(),
                stored_dst_scaleout_rank_idx,
                ncclGinOptFlagsAggregateRequests);
    }
    update_scaleout_tail();
}
```

中文解释：

这是 dispatch 慢的核心代码。每个 channel 遍历 token，发现 token 需要跨节点时，就把这个 token 写到 send buffer，再发一个 GIN put 到远端 scaleout recv buffer。

对于当前 benchmark：

- `hidden=7168`；
- FP8 dispatch 的 hidden payload 大约 `7168 bytes`；
- 加上 scale factor、topk metadata、source metadata，单 token 大约 `7-8 KiB`；
- 每个 token 至少一次 GIN put；
- 每几个 token 又一次 tail 更新。

我们的 microbenchmark 显示，EFA proxy GIN 在 8 KiB message 下只有约 `5 GB/s/rank`，4 KiB 下约 `2.8 GB/s/rank`。这和 DeepEP dispatch 的 `2-3 GB/s SO` 完全同量级。

所以这里不是“代码逻辑错误”，而是协议假设与 EFA 网络特性不匹配。当前代码假设小 GIN put 和小 remote signaling 很便宜；这个假设在 CX7/IB 上成立，在 EFA 上不成立。

### 6. forwarder 依赖远端 tail，容易被小 signal 延迟拖住

位置：[deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:488)

```cpp
while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
    ...
    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
        const uint32_t arrived_or_finished =
            stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
        if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
            return true;
        ...
        if (lane_idx < kNumScaleoutRanks) {
            const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
            math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
        }
        __syncwarp();
        return false;
    });
```

中文解释：

forwarder warp 通过轮询 `scaleout_channel_signaled_tail` 判断远端 scaleout sender 有多少 token 已经到达。这个设计对低延迟网络很好，因为 sender 更新 tail 后，forwarder 立即开始 NVLink 分发。

但在 EFA 上，tail 更新本身是高成本 signal/atomic-like 操作。如果 tail 慢，forwarder 就在这个循环里等；如果把 tail 更新频率调得太低，又会增加远端 token 的等待时间。当前 `3 tokens/update` 处在一个很不适合 EFA 的点：更新太频繁，消息又太小。

建议把 forwarder 模式从 token streaming 调成 chunk streaming：

- sender 侧一次写入一个 chunk，例如 32/64/128 tokens；
- tail 表示 chunk end，而不是每 3 个 token的 tail；
- forwarder 每次处理一个较大的 contiguous token range；
- 保留 finish flag，但减少中间 tail 更新。

### 7. GIN wrapper 对 EFA atomic-like 操作成本不可见

位置：[deep_ep/include/deep_ep/common/handle.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/common/handle.cuh:96)

```cpp
void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                 const int& extra_options = 0) const {
    const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);
    if (dst_ptr != nullptr) {
        ptx::red_add_rel_sys(dst_ptr, value);
    } else {
        gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx,
                   ncclGin_VASignalAdd(nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr, static_cast<uint64_t>(value)),
                   ...);
    }
}
```

中文解释：

这个 wrapper 把“本地/NVLink 可达”和“远端 RDMA”统一成一个接口。对调用者来说 `red_add_rel` 好像只是一个简单的 release add；但在 EFA 上，远端分支会走 `gin.signal + VASignalAdd`，实际是一个需要 proxy 支持的 atomic-like 操作。

当前上层代码大量调用 `red_add_rel` 来做 tail/count 更新，导致 EFA 上的真实成本被隐藏。建议给 wrapper 增加 backend-aware 注释或接口，例如：

```cpp
// 中文注释建议：
// 对 EFA proxy GIN，远端 red_add_rel 不是廉价的硬件 RDMA atomic；
// 它会走 GIN signal / VASignalAdd 路径，适合低频控制信息，不适合每几个 token 更新一次。
// 新代码如果只是单 writer 发布进度，应优先使用 put_value 写 absolute tail，
// 避免把单调进度发布实现成 remote atomic add。
```

### 8. put wrapper 本身没有合并 payload

位置：[deep_ep/include/deep_ep/common/handle.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/common/handle.cuh:172)

```cpp
void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes, const int& dst_rank_idx,
         const int& extra_options = 0,
         const remote_action_t& remote_action = remote_action_t()) const {
    gin.put(TEAM_WORLD_RAIL(),
            dst_rank_idx,
            nccl_window, reinterpret_cast<int64_t>(recv_sym_ptr) - lsa_base_ptr,
            nccl_window, reinterpret_cast<int64_t>(send_sym_ptr) - lsa_base_ptr,
            num_bytes,
            remote_action,
            ...,
            ncclGinOptFlagsDefault | extra_options);
}
```

中文解释：

这里的 `ncclGinOptFlagsAggregateRequests` 只能帮助 GIN 对 request 进行一定程度的聚合或排队，但它不会把上层的多个 token payload 自动变成一个大的 contiguous RDMA payload。真正决定 message size 的是调用方传进来的 `num_bytes`。当前调用方传的是 `tma_buffer.get_num_bytes<false>()`，也就是单 token 大小。

因此优化不能只靠 `AggregateRequests` flag。长期 EFA backend 应该在这里切换 transport：GPU 把 payload 放进 staging buffer 后提交 `WRITE TransferCmd`，由 CPU proxy 负责发 verbs RDMA write；是否按 ring/chunk 管理 staging buffer只是 proxy 后端的内部流控细节，不是继续调用 Gin big put。

## 优化方案

### 主方案：UCCL-style V2 EFA backend

目标：为 DeepEP V2 增加一个真正面向 EFA 的 backend。这个 backend 不把 EFA 当成“慢一点的 IB”，也不继续围绕 NCCL Gin 做补丁，而是显式采用 UCCL-EP 的 GPU->CPU command queue、CPU proxy、RDMA_WRITE_WITH_IMM、host-mapped atomic/control buffer 和 receiver-side ordering。

设计：

1. 新增一个 transport 抽象，例如 `EfaProxyTransport`，只在 AWS EFA path 启用。IB/CX7 继续使用现有 NCCL Gin path，避免破坏官方机器性能。
2. 在 V2 buffer 初始化时，把 `ElasticBuffer` / workspace 里跨节点可写的 GPU memory 注册给 EFA verbs。实现上可以复用 UCCL 的 DMA-BUF MR chunk registration 逻辑，处理大 buffer 超过单个 MR/IOMMU 限制的问题。
3. 为每个 GPU 创建 D2H command queue，直接移植或最小化复用 UCCL 的 128-bit `TransferCmd` / FIFO 结构。GPU kernel 不再直接调用远端 `gin.put` 或 `gin.signal`，而是提交 `WRITE/ATOMIC/DRAIN/BARRIER` 命令。
4. 数据面仍然是 GPUDirect RDMA：token payload 留在 GPU staging buffer；`WRITE TransferCmd` 只携带 `dst_rank/src_offset/dst_offset/bytes/channel/seq` 等控制信息；CPU proxy 根据这些 offset 发 verbs RDMA write。
5. 控制面由 CPU proxy 负责：tail/count/finish/barrier 不走 `gin.red_add_rel` 或 `VASignalAdd`，而是走 `ATOMIC/DRAIN/BARRIER TransferCmd`。
6. 对 EFA 使用 `IBV_WR_RDMA_WRITE_WITH_IMM` 携带 sequence、source rank、channel、token count、tail/atomic 信息。接收端 proxy 根据 immediate data 更新 reorder buffer。
7. 接收端 proxy 只在确认对应 write 到齐后，才发布 host-mapped tail/counter。GPU forwarder 轮询这个 host-mapped control buffer，或者轮询一个由 proxy 低频写入的 GPU-visible control region。
8. 每 GPU 一个 proxy 进程/线程组，多线程管理同 NUMA 组的 EFA NIC/QP。p5en 上应显式把 GPU local rank 映射到相邻 EFA NIC，避免把 16 张 NIC 的利用交给 Gin/NCCL 的隐式 rail 策略。

这条路比只改 Gin 更大，但它最贴合论文已经验证过的 EFA 特性。UCCL-EP README 在 p5en normal kernel 上给出的 EP16 参考量级是 dispatch `~50 GB/s RDMA`、combine `~18 GB/s RDMA`；我们现在 V2 dispatch 只有 `2-3 GB/s SO`，所以这不是小修小补能自然补上的差距。

建议把它作为长期主线，因为它能同时解决四个问题：

- EFA unordered SRD 下的 write-before-tail correctness；
- 没有 native RDMA atomic 时的 tail/count 发布；
- device-side Gin signal/atomic-like 操作在 EFA proxy 路径上的高开销；
- 每 GPU 多 EFA NIC/QP 的显式调度和流控。

### 不推荐作为主线：EFA coalesced Gin dispatch

这个方案是之前的低侵入想法：保留当前两级拓扑思想，即跨节点只发到同 lane GPU，再在目标节点 NVLink forward；但把 scaleout 从 token streaming 改成 chunk streaming，并继续使用 NCCL Gin proxy 做数据面。

现在我不建议把它作为长期方案，原因是它只改变 message size，没有从根上解决 UCCL-EP 论文指出的 EFA 问题：

- RDMA 发起方仍然是 device-side Gin/proxy path，不是 UCCL-style CPU proxy verbs path；
- tail/order/control 仍然容易落回 Gin signal/flush 语义；
- EFA unordered SRD 和无 native atomic 的问题只是被降频，不是被显式建模；
- NIC/QP 调度仍然依赖 NCCL/OFI 内部策略，不如 UCCL proxy 可控。

它可以作为开发过程中的对照实验，帮助量化“只增加 message size”能带来多少收益；但不应该是最终架构。

如果仍然保留这个实验，设计如下：

1. 每个 scaleout channel 不再对每个 token 立即 `gin.put`。
2. channel 先把要发往远端 scaleout rank 的 token 写入一个 contiguous `scaleout_send_chunk_buffer`。
3. chunk 达到阈值后一次 GIN put 到远端 `scaleout_recv_chunk_buffer`。
4. tail 从 token tail 改成 chunk tail，例如发布 `{finish, chunk_id, num_tokens, byte_end}`。
5. forwarder 按 chunk 读取，再逐 token 分发到本节点 scaleup buffer。
6. 如果继续用 Gin，至少做到“data chunk put 完成后再发布 tail”；如果引入 UCCL-style control plane，则 tail 由 CPU proxy 在 write completion/imm seq 到齐后发布。

建议初始参数：

| 参数 | 建议值 |
| --- | --- |
| `kScaleoutCoalesceTokens` | 32 或 64 |
| `kMinScaleoutChunkBytes` | 256 KiB 起步 |
| tail update interval | 每 chunk 一次 |
| message upper bound | 1-4 MiB，避免占用太多 buffer |

预期收益：

- 8 KiB message：`~5 GB/s/rank`；
- 64 KiB message：`~17 GB/s/rank`；
- 128 KiB message：`~18.5 GB/s/rank`；
- 8 MiB/peer remote-only：`~23 GB/s/rank`；
- 32 MiB/peer remote-only：`~44.8 GB/s/rank`。

如果把实际 RDMA message 从 8 KiB 提高到 128 KiB 以上，dispatch SO 有机会从 `2-3 GB/s` 提到 `10-20 GB/s` 量级；如果进一步合并到 MiB 级，可能更高。

风险：

- 会增加本地 pack/unpack 和 metadata 复杂度；
- chunk buffering 会增加峰值 buffer 用量；
- 需要处理 token order、slot index、cached dispatch handle 的兼容；
- 延迟会增加一点，但当前目标是 throughput。

和 UCCL-EP 的关系：

- 这个实验只借了 UCCL 的 staging/chunk 表面形式，没有借到 CPU proxy、WRITE_WITH_IMM 和 receiver-side ordering 的核心；
- 因此它只能回答“Gin 消息变大后能改善多少”，不能代表最终 UCCL-style 后端。

### 方案 B：把 tail 更新从 atomic delta 改成 absolute put_value

目标：先做一个较低侵入的 EFA patch，减少 remote atomic-like 操作。

当前：

```cpp
gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
```

建议 EFA path：

```cpp
// 伪代码：单 writer 发布绝对 tail
gin.put_value<ncclTeamTagRail>(ptr, signaled_tail, lane_idx);
```

为什么可能成立：

- `ptr` 按 `(channel_idx, source_scaleout_rank)` 分开；
- 对每个 location，通常只有对应 source channel writer 更新；
- receiver 只需要看到单调 tail，不需要每个 delta 都被精确累加；
- 如果跳过中间值，只要最终 absolute tail 更大，forwarder 仍可处理 `[old_tail, new_tail)`。

需要验证：

- 当前 `red_add_rel` 是否同时承担 release ordering；
- `put_value` 前是否需要 `flush`、`DRAIN/QUIET` 或 receiver-side sequence 来保证数据 put 先于 tail 可见；
- `finish_flag` 与 tail pack 的单调性。

验证方式：

- 在 EFA 空闲时实现一个编译开关，例如 `EP_EFA_ABSOLUTE_TAIL=1`；
- 跑 EP16 correctness；
- 对比 dispatch SO 和 trace 中 `hybrid_dispatch_impl` 时间；
- 用 small-message benchmark 加一个 “data put + absolute tail put” 模式对比。

UCCL-EP 给这里的额外提醒是：在 EFA 上仅仅把 `red_add_rel(delta)` 换成 `put_value(absolute_tail)` 不一定足够，因为 EFA SRD 不提供我们想象中的强 ordering。正确方向是“absolute tail + 明确完成条件”：要么 Gin data put 后做低频 drain/flush，再写 tail；要么像 UCCL 一样让接收端 proxy 根据 `WRITE_WITH_IMM` sequence 确认数据到齐后再发布 tail。

### 方案 C：增大 `kScaleoutUpdateInterval`

目标：快速验证 tail update 频率对 EFA 的影响。

当前默认是 `3`。可以加模板参数或环境变量生成不同 JIT specialization：

| Interval | 预期 |
| ---: | --- |
| 3 | 当前低延迟设计，EFA 开销高 |
| 8 | 小幅降低 signal 次数 |
| 16 | 可能明显改善 |
| 32/64 | 更接近 chunk streaming |

这个方案实现简单，但上限有限，因为数据 put 仍然是 token-sized。它可以作为定位实验，不能替代方案 A。

### 方案 D：EFA direct chunked dispatch，绕过 remote forwarder

目标：在 EFA 上用 `NCCL_GIN_CONNECTION_FULL` 或普通 NCCL send/recv，把 token 直接发到最终 remote GPU，减少“远端同 lane GPU再 NVLink forward”的复杂同步。

设计：

1. 本地按最终 destination rank 聚合 token；
2. 每个 remote rank 一段 contiguous chunk；
3. 发 `rank -> remote rank` 的 chunk；
4. 接收端直接写入最终 expert buffer，或者写 staging buffer 后 epilogue。

优点：

- 可以更自然地做大 chunk；
- 减少 forwarder 轮询 tail；
- 充分利用多 peer / 多 NIC。

缺点：

- `FULL` 连接和 QP 数更多；
- 需要重新设计 slot metadata；
- 可能影响 IB 上已有高性能路径；
- 对 DeepEP 当前 hierarchical combine/dispatch 设计侵入较大。

这是中长期方案。

### 方案 E：AWS fallback 到普通 NCCL all-to-all 风格

目标：利用普通 NCCL over EFA 已测到的高 all-to-all 吞吐。

已有普通 NCCL 结果：

- `all_reduce_perf` EP16 1 GiB：`~237 GB/s algbw`；
- `alltoall_perf` EP16 1 GiB：`~91 GB/s algbw`。

可以做一个 AWS/EFA fallback：

1. layout/count 阶段算出每 rank 发给每 rank 的 token 数；
2. pack 成固定上限或 padding 后的 dense buffers；
3. 用 NCCL grouped send/recv 或 alltoall-like path 交换；
4. 再执行本地 epilogue。

优点：

- 可能最快拿到比当前 dispatch 更好的 throughput；
- 避免 GPU-side tiny RDMA puts；
- 与 EFA 的强项匹配。

缺点：

- 可能需要 CPU launch / host-side orchestration；
- 对 overlap 和低延迟不如 DeepEP 原路径；
- NCCL 没有通用 alltoallv，需要 padding 或 grouped p2p。

这是工程 fallback，不一定是最终最优。

## 建议实施顺序

### 第零阶段：确定 UCCL-style backend 的边界

1. 把 UCCL-EP 在 p5en 上的 EP16 normal kernel 结果作为第一目标：dispatch RDMA `~50 GB/s`，combine RDMA `~18 GB/s`。
2. 明确不再把 Gin coalescing 当作主线；V2 EFA backend 的边界是替换跨节点 `put/signal/atomic/drain/barrier`，节点内 NVLink path 尽量复用现有代码。
3. 方案评估时记录 proxy command rate、RDMA write size、immediate reorder backlog、host-mapped tail latency、每 NIC/QP 吞吐和 GPU forwarder wait time。

### 第一阶段：移植最小 UCCL proxy 骨架

1. 从 `uccl/ep` 移植 `TransferCmd`、D2H FIFO、host decode、proxy thread 管理的最小子集。
2. 先做独立 microbenchmark：GPU 写 `WRITE TransferCmd`，CPU proxy 发 EFA verbs RDMA write 到远端 GPU buffer。
3. 支持 GPU memory MR registration、DMA-BUF chunk registration、remote MR exchange、QP/CQ 初始化。
4. 在 p5en 上确认单 GPU pair、多 GPU pair、整机 16 EFA NIC 的 proxy RDMA write 吞吐。

目标：先证明不经过 NCCL Gin，UCCL-style proxy 能在 V2 仓库里直接驱动 EFA 跑起来。

### 第二阶段：实现 EFA ordering/atomic/control

1. 移植 UCCL 的 `WRITE_WITH_IMM` sequence 机制，把 `src_rank/channel/buffer_idx/num_tokens/seq` 编进 immediate data。
2. 移植 receiver-side pending update 逻辑：write 到齐前不发布 tail/atomic；乱序到达时先缓冲。
3. 为 V2 dispatch 的 `scaleout_channel_signaled_tail`、rank/expert count、finish flag 建立 host-mapped control buffer。
4. GPU forwarder 从 host-mapped control buffer 或 proxy 发布的 GPU-visible control mirror 读取 tail。

目标：先解决 EFA 上 correctness 和 ordering，而不是先追求 kernel 形态优雅。

### 第三阶段：接入 DeepEP V2 dispatch

1. 在 `hybrid_dispatch.cuh` EFA path 中，把远端 `gin.put<ncclTeamTagRail>` 替换成提交 `WRITE TransferCmd`。
2. 把 `red_add_rel` tail/count 更新替换成 `ATOMIC TransferCmd` 或 proxy-managed absolute tail。
3. 保留现有 TMA pack、slot assignment、NVLink forwarder 逻辑，先最小化改动跨节点 transport。
4. 先只支持 EP16、FP8、`hidden=7168`、`topk=8`、cached/uncached dispatch 的核心路径。

目标：让 V2 dispatch correctness 通过，并把 `hybrid_dispatch_impl` 从 Gin 小消息瓶颈中解放出来。

### 第四阶段：性能工程和泛化

1. 调 proxy threads、QP 数、CQ polling、NIC sharding、command batching、FIFO backpressure。
2. 支持 BF16 / FP8、不同 hidden、不同 topk、unbalanced gate。
3. 支持 cached dispatch handle 和 combine path。
4. 与 NCCL Gin path 做运行时选择：EFA 走 proxy backend，IB/CX7 走原生 Gin backend。

目标：形成 AWS EFA 后端。

## 推荐代码注释位置

如果后续决定直接在源码里加中文注释，建议只加在这些点，避免到处散落：

1. [csrc/kernels/backend/nccl.cu](/Users/daniel/Documents/code/DeepEP-danyang/csrc/kernels/backend/nccl.cu:102)

```cpp
// 中文注释建议：
// hybrid mode 使用 RAIL GIN：跨节点只连同 local-rank 的 GPU，再在节点内 NVLink forward。
// 这对 IB/CX7 的 GPU-initiated 小 RDMA 很友好，但在 AWS EFA proxy GIN 上，
// token-sized put 和 signal/atomic-like 操作会成为瓶颈。长期 EFA 后端不应
// 继续围绕 Gin 做局部优化，而应切到 UCCL-style CPU proxy transport。
```

2. [deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:333)

```cpp
// 中文注释建议：
// 当前 tail 每 kScaleoutUpdateInterval 个 token 发布一次，默认只有 3。
// 在 EFA 上，这个更新会走 proxy GIN 的 signal/atomic-like 路径；
// 如果 message 只有几个 token，大量 tail update 会吞掉 RDMA 吞吐。
// 长期 EFA path 应把 tail 发布改成 ATOMIC TransferCmd 或 proxy-managed tail，
// 由接收端 proxy 在确认对应 write 到齐后再发布。
```

3. [deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:442)

```cpp
// 中文注释建议：
// 这里每个跨节点 token 都会触发一次 GIN put，num_bytes 是单 token payload。
// 对 hidden=7168 的 FP8 dispatch，单次 put 大约 7-8 KiB；
// EFA proxy GIN 在这个粒度只有几 GB/s/rank，远低于大包带宽。
// AWS/EFA 长期后端应把这里替换成 WRITE TransferCmd：
// GPU 只提交 src/dst offset 和 bytes，CPU proxy 负责发 EFA verbs RDMA write。
```

4. [deep_ep/include/deep_ep/common/handle.cuh](/Users/daniel/Documents/code/DeepEP-danyang/deep_ep/include/deep_ep/common/handle.cuh:96)

```cpp
// 中文注释建议：
// red_add_rel 的远端分支会走 GIN signal + VASignalAdd。
// 在没有 native RDMA atomic 的 EFA 上，这不是廉价操作；
// 适合低频控制信息，不适合每几个 token 发布一次进度。
// 长期 EFA backend 应用 ATOMIC TransferCmd / host-mapped control buffer
// 替代这个远端 Gin signal 路径。
```

5. 如果引入 UCCL-style control plane，建议在新 backend 的 `TransferCmd` 提交处加注释：

```cpp
// 中文注释建议：
// EFA 不提供 DeepEP 原 IBGDA/Gin 路径假设的强 ordering 和 native RDMA atomic。
// 这里 GPU 只发布 128-bit TransferCmd，payload 仍在 GPU buffer 中；
// CPU proxy 负责发 GPUDirect RDMA write，并用 immediate sequence 在接收端
// 确认 write 到齐后再发布 tail/atomic。这样可以避免 per-token GIN signal
// 在 EFA proxy 路径上成为瓶颈。
```

## 最终建议

读完 UCCL-EP 论文和 `uccl/ep` 代码后，我的判断变得更明确：只靠调 Gin 参数或 rails 不会解决 V2 dispatch。EFA 上真正的问题是协议层仍在发大量 token-sized GPU-initiated put 和 high-frequency remote signal/atomic-like 控制消息。

最推荐的路线是直接做一条长期主线：

1. 新增 `UCCL-style V2 EFA backend`：跨节点数据、tail/count/order/control 全部迁到 `TransferCmd + CPU proxy + RDMA_WRITE_WITH_IMM + receiver-side ordering`。
2. 保留当前 NCCL Gin path 给 IB/CX7；EFA path 不再把 Gin chunking 作为架构目标。
3. 如果做 Gin coalescing，只把它当作对照实验，用来证明“只改 message size 仍不如完整 proxy backend”。

如果要尽快给训练拿一个可用的 AWS 性能版本，可以同时评估方案 E：普通 NCCL all-to-all 风格 fallback。它不一定优雅，但更贴近 EFA 已验证的高吞吐路径。
