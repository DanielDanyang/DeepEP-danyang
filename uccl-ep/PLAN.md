# DeepEP V2 on AWS EFA — Native UCCL-EP 计划

*合并自 NATIVE_V2_REWRITE_PLAN.md 与 NATIVE_V2_COMPLETION_PLAN.md。*
*最后更新：2026-06-01，基于实际源码精读与 UCCL-EP 论文。*

---

## 一、项目背景与当前状态

### 为什么要重写

原始 `uccl-ep` 以 V1 静态 kernel 为骨架（`internode.cu / intranode.cu / layout.cu`），
依赖 `SourceMeta`、`rank_prefix_matrix` 等 V1 staged layout，
无法表达 DeepEP V2 的 `BufferLayout / TokenLayout / token_metadata_at_forward /
channel_linked_list / expanded dispatch / reduced combine` 语义。

正确方向：复用 EFA proxy/RDMA 基础设施，但 **fork DeepEP V2 JIT kernel**，
把 GIN `put/red_add_rel/signal` 替换为 D2H FIFO + EFA RDMA write。

### 已完成的部分（截至 2026-06-01）

- V1 静态 kernel（internode/intranode/layout）已从 build 中移除，source hygiene 测试固定
- `V2EfaRuntime`、JIT plan、`V2TransferCmd`、D2H queue、EFA verbs sink 已编译通过
- D2H queue publish/ack 协议已修正（atomicExch header，消灭 leaked slot / TOCTOU drop）
- `token_metadata_at_forward / channel_linked_list` 已改成 V2-like 多 channel 形状，
  由 CUDA/JIT metadata kernel 填充
- 三段 JIT pipeline（prepare_descriptors → pack_records → enqueue_d2h）已建立
- Scaffold 路径已有 done-signal / expected_count 机制，用于替代 post-dispatch
  `dist.barrier()`；最终 V2 streaming tail 仍未实现
- Per-NIC lane 多设备支持，`efa-nv-peermem` 确认安装，GPU buffer 已可注册为 RDMA MR
- EP16 remote-pair dispatch correctness 曾通过；EP16 bench 约 2.91 GB/s（scaffold 路径）
- `sq_sig_all=0 + signal-only CQE` 已在 EFA SRD verbs path 上实测失败：
  `ibv_wr_complete ret=22 errno=11`。因此阶段 0 的主路径改为
  `sq_sig_all=1 + all-CQE accounting`：每个 payload/signal WR 都会产 CQE，
  proxy 必须全部 poll 完，才能复用 signal scratch。

### 还没完成的硬阻塞

1. **未 fork `hybrid_dispatch.cuh` 主路径**：sender 仍先 pack 到独立 EFA window，
   未直接从 `scaleout_send_buffer` 发；receiver 仍需 materialize copy，
   未直接写 `scaleout_recv_buffer`。
2. **未实现 streaming tail（`red_add_rel` 替代）**：当前 done-signal 是批量的，
   丢失了 V2 每 3 token 推进的 pipeline overlap。
3. **combine 未 native 实现**：`_semantic_combine_data` fallback 仍存在。
4. **cached dispatch/combine 未复用官方 V2 handle**。

---

## 二、EFA 能达到 90 GB/s 吗？

### 带宽完全对等

| 测试床 | NIC 配置 | 每节点带宽 | 每 GPU 带宽 |
|--------|---------|----------|-----------|
| NV_EFA3（p5en，我们的环境） | EFAv3 200G×16 | 400 GB/s | 2×200G |
| NV_IB（CX7 对照） | ConnectX-7 400G×8 | 400 GB/s | 1×400G |

### 目标数字

| 参考点 | 数值 | 来源 |
|--------|------|------|
| V1 uccl-ep on CX7 EP16 | 61 GB/s | UCCL-EP 论文 |
| V2 DeepEP on CX7 EP16 | **90 GB/s** | DeepSeek README |
| V1 → V2 提升 | +47% | 架构改进（TMA、PDL、forward warp pipeline、expanded slot） |
| EFA vs CX7 | 带宽相同 | 均 400 Gbps/GPU |
| **native V2 on EFA 目标** | **90 GB/s** | 架构提升与网卡无关 |

EFA firmware 小消息速率限制（UCCL-EP 论文脚注 4）AWS 在修复中；
V2 的 TMA + chunk pipeline（32 token/chunk）有效规避了 per-token 小消息问题。

**完成判定：dispatch >= 80 GB/s 为接近完成，= 90 GB/s 为目标完成。
单靠 scaffold 路径优化无法突破 GPU memcpy 瓶颈到 80 GB/s。**

---

## 三、V2 关键架构（代码精读）

### Buffer 结构（`layout.cuh`，GPU 内存）

```
ElasticBuffer.buffer（GPU memory，NCCL symmetric window）= [
  scaleup_buffer:        [kNumScaleupRanks][kNumScaleoutRanks * kNumMaxTokensPerRank][TokenLayout]
  scaleout_send_buffer:  [1][kNumMaxTokensPerRank][TokenLayout]
                         ← scaleout warp TMA store 目标；EFA 从这里读取 payload
  scaleout_recv_buffer:  [kNumScaleoutRanks][kNumChannels * kNumMaxTokensPerChannel][TokenLayout]
                         ← gin.put() 远端目标；EFA RDMA write 应直接写这里
]

TokenLayout = [hidden | sf | topk_idx(int×topk) | topk_weights(float×topk) |
               src_global_idx(int) | linked_list_idx(int) | mbarrier]
```

### WorkspaceLayout（GPU workspace，关键元数据）

```
scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx)  → int64_t
  = math::pack2<int, int64_t>(finish_flag, tail_count)
  内存布局：lo32 = finish_flag, hi32 = tail_count  ← 注意 finish 在低位！
  由 scaleout warp 的 gin.red_add_rel 增量写入（每 kScaleoutUpdateInterval=3 token）
  由 forward warp spin-wait 消费（streaming，不等全量）
  EFA fork: CPU proxy 写绝对值（不是 delta），forward warp 读法不变
```

### V2 dispatch 数据流（`hybrid_dispatch.cuh`）

```
notify warp（SM0 执行，per-SM 汇总 rank/expert count，再 EFA 广播给所有 scaleout ranks）：
  SM local: 统计本 SM 内 topk 落在哪些 expert/rank
  全局聚合: workspace GPU 原子 reduce → SM0 等待 kNumSMs 个 SM 全到
  EFA 广播（sm_idx==0, thread < kNumScaleoutRanks）：
    gin.put<ncclTeamTagRail>(dst.workspace.scaleout_rank_count,  src.workspace.rank_count,   …)
    gin.put<ncclTeamTagRail>(dst.workspace.scaleout_expert_count, src.workspace.expert_count, …)
  ← 这两个是 EFA RDMA，需要在 EFA fork 里替换为 D2H FIFO
  之后通过 NVLink (ncclTeamTagLsa) 汇总到 scaleup rank：
    gin.put_value<ncclTeamTagLsa>(…)
    gin.red_add_rel<ncclTeamTagLsa>(…)
  ← NVLink 路径保留 GIN，不改动

scaleout warp（per channel = sm × kNumChannelsPerSM）：
  for token in channel_tokens:
    TMA store x[token] → scaleout_send_buffer[token_idx]
    if dst != local_rank:
      gin.put<ncclTeamTagRail>(
          remote=receiver.scaleout_recv_buffer[src_scaleout_rank][channel_idx][dst_slot],
          local=scaleout_send_buffer[token_idx], bytes, dst_scaleout_rank_idx)
      // 注意：receiver 侧第一维是 src_scaleout_rank（本 sender 的 rank），不是 dst_rank
    if dst == local_rank:
      TMA store 直接写 scaleout_recv_buffer[local][channel_idx][dst_slot]  ← local bypass，不走 EFA
    tail_count[dst_rank] += 1
    if any_tail_count % 3 == 0:
      gin.red_add_rel<ncclTeamTagRail>(
          receiver.workspace.channel_tail[channel_idx][src_rank] += delta_int64)
      // delta_int64 = pack2(finish_flag_delta=0, tail_count_delta)
  gin.red_add_rel<ncclTeamTagRail>(… pack2(finish_flag=1, tail_count_final))

forward warp（per channel，同一个 kernel）：
  while not all_sources_done:
    spin-wait: channel_tail[channel][src_rank] > old_tail  ← 增量，每3个token推进
    读取方式: ld_acquire_sys(workspace.channel_signaled_tail_ptr)
    解包: unpack2(int64, finish_flag, tail_count)  ← 注意解包顺序！
    for slot in [old_tail, new_tail):
      copy scaleout_recv_buffer[src][channel][slot] → scaleup_buffer[scaleup_rank][slot]
        via gin.get_sym_ptr<ncclTeamTagLsa>  ← NVLink TMA store，保留 GIN
      build token_metadata_at_forward[channel][token_idx]
      build dst_buffer_slot_idx[channel][src_rank][slot][topk]
  // 结束后 forward warp 重置 tail：
  *workspace.channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;  ← Phase 3 才去掉
```

### tail word 格式（`math::pack2<int, int64_t>` 实际内存布局）

```
int64_t tail_word = math::pack2<int, int64_t>(finish_flag, tail_count)
  → 内存：[lo32 = finish_flag, hi32 = tail_count]
  → unpacked_ptr[0] = finish_flag  (低地址 / 低32位)
  → unpacked_ptr[1] = tail_count   (高地址 / 高32位)

读取: math::unpack2<int, int64_t>(tail_word, finish_flag, tail_count)
  → 第一个输出 = lo32 = finish_flag
  → 第二个输出 = hi32 = tail_count

EFA fork 增加 epoch（Phase 3 去 barrier 时）：
  lo32 = (epoch << 1) | finish_flag   ← epoch 在 finish_flag 字段里扩展
  hi32 = tail_count                   ← tail_count 字段不变
  forward warp 额外检查：(lo32 >> 1) == current_epoch
注意：Phase 1 先不加 epoch，格式与 V2 原版相同。
```

### channel → QP/lane 映射（`comm.cuh get_qp_mode`）

```
global_channel_idx = sm_idx × kNumChannelsPerSM + channel_in_sm_idx
qp_idx = (global_channel_idx % kNumAvailableQPs)

EFA 等价：efa_lane(channel_idx) = channel_idx % num_efa_lanes

规则：一个 channel 的所有操作（payload write、tail write）
      必须使用同一个 efa_lane，不能用 expert_id 分 lane。
```

### 当前 EFA window 的真实类型

```python
window = torch.empty((bytes,), dtype=torch.uint8, device="cuda")  # GPU 内存
```
`efa-nv-peermem 1.2.3` 已安装，`ibv_reg_mr(cuda_ptr)` 可用（无需 DMA-BUF）。
现有 window 已是 GPU buffer + GPUDirect RDMA；额外开销是 GPU memcpy，不是 PCIe。

---

## 四、当前 scaffold vs 真实 native V2 差距

| 维度 | 真实 V2 | 当前 scaffold | 实际代价 |
|------|--------|--------------|---------|
| 发送 source | GPU `scaleout_send_buffer`（TMA，buffer 内） | pack_kernel → 独立 `_v2_efa_window` | 额外 GPU memcpy（HBM 带宽） |
| RDMA 目标 | 对端 GPU `scaleout_recv_buffer`（buffer 内） | 对端独立 `_v2_efa_window` | 后续 materialize copy |
| 接收处理 | forward warp 流式 scaleout_recv_buffer→scaleup_buffer | materialize_kernel 批量 window→recv_x | 额外 GPU memcpy |
| Tail 通知 | `red_add_rel` 增量（每 3 token），streaming | done_word 批量（full dispatch 后） | 丢失 pipeline overlap |
| 元数据 | forward warp 原地生成 | Python loop 重建 | +0.43ms/iter |
| Epilogue | `dispatch_copy_epilogue`（官方 V2 JIT） | 自定义 materialize kernel | 不共享 V2 epilogue |

---

## 五、V1/V2 差异原则

默认沿用 V1 的 CPU proxy + FIFO + EFA post 方法；只有 V2 语义或 EFA 约束使 V1 无法正确表达时才允许不同。

| 差异点 | V1 做法 | V2 目标做法 | 必须不同的理由 |
|--------|--------|------------|--------------|
| kernel 形态 | 静态 `.cu` kernel | 跟随 V2 JIT `.cuh` | V2 kernel 依赖运行时参数生成特化代码，静态 kernel 会把 V2 tensor 投影回 V1 packed staging |
| 数据布局 | `SourceMeta`、prefix matrix、packed staging | `BufferLayout / TokenLayout`、expanded dispatch、reduced combine | V1 描述 packed staging；V2 描述 expanded slot 和 reduced combine slot，不是字段改名 |
| command 格式 | 旧 16B `TransferCmd`，V1 bitfield | 新 16B `V2TransferCmd`，V2 layout 解析后的 EFA post 字段 | 旧 bitfield 无法表达 V2 expanded offset、lane、signal value |
| lane 语义 | V1 command 依赖旧 staging/QP 映射 | `efa_lane = channel_idx % num_efa_lanes`（来自 V2 `get_qp_mode` 逻辑） | EFA 没有 GIN symmetric QP；lane 必须由 channel 派生，不是 expert 或 scaleup rank |
| D2H queue | 单 FIFO 或语义分队列 | 多 FIFO 按 channel 分，不按 dispatch/combine 分语义 | 多 FIFO 只表示并行度；按语义分会改变 ordering 和 backpressure |
| Tail 通知 | 无 streaming tail | CPU proxy 在同 lane QP 发 RDMA write 到 receiver GPU workspace tail | EFA 无硬件 remote atomic；用 same-QP ordering 替代 GIN `red_add_rel` |
| MR 注册 | V1 用 host-mapped window 或 CPU buffer | 注册整个 `ElasticBuffer.buffer`（GPU）为 MR | V2 buffer layout 对所有 rank 相同，single buffer_base 统一 local/remote offset 计算 |
| receiver 落点 | V1 staging buffer | 直接写 `scaleout_recv_buffer`（V2 buffer 内） | 避免额外 GPU memcpy；保持官方 V2 epilogue 期待的 layout |
| Python handle | V1 binding 包装成 V2 接口 | `V2EfaRuntime`，返回官方 V2 handle/cache 语义 | 兼容包装导致 cached path 与 V2 语义不一致 |
| proxy 框架 | CPU proxy/FIFO/EFA post 可复用 | 同构复用，只 decode `V2TransferCmd` | transport substrate 与 V1 语义耦合弱，只改 command decode 逻辑 |

---

## 六、完成计划

### 阶段 0：立即修复（服务器空闲时）

**代码状态（2026-06-02）：已按 all-CQE accounting 接入；服务器 EP16 验证待跑。**

**0a. CQE correctness 验证**（当前主路径）
- EFA SRD QP 保持 `sq_sig_all=1`；这是当前 p5en 环境上可通过
  `ibv_wr_complete` 的路径。
- `V2VerbsPostStats.posted_completions` 必须统计所有会进入 CQ 的 WR：
  payload write 和 signal write 都计入。
- `outstanding_signaled_posts_` 名字后续可改，但语义必须是
  outstanding CQE count；只有全部 CQE poll 完后才能复用 signal scratch。
- 跑 EP16 remote-pair correctness，确认不 timeout、不 silent corruption。
- 若后续要重新尝试 signal-only CQE，必须先解决 EFA SRD extended verbs
  对 unsignaled write 的 `ret=22` 问题；不能只改 accounting。

**0b. README-size dispatch bench**
- 用 DeepEP 原始 bench 脚本和对应配置（不手写参数）
- 建立与 GIN path（5 GB/s）可对比的 scaffold baseline

---

### 阶段 1：fork `hybrid_dispatch.cuh`（关键路径，决定能否到 90 GB/s）

**代码状态（2026-06-02）：本地代码已接入 native hybrid dispatch 主路径；服务器 build/JIT/bench 待跑。**
当前已有：
- `V2TransferCmd.kWorkspace`，可区分 buffer MR 与 workspace MR；
- `V2EfaConnection(workspace_addr, workspace_bytes)`，可交换 per-lane workspace rkey；
- verbs sink 按 command region post 到 buffer/workspace；
- `dispatch_jit.cuh` device shims 已被 `uccl-ep/include/v2_efa/hybrid_dispatch_native.cuh`
  调用；
- `hybrid_dispatch_native.cuh` 已从官方 V2 `hybrid_dispatch.cuh` fork，并替换全部
  `ncclTeamTagRail` notify/payload/tail call site；
- `ncclTeamTagLsa` NVLink scaleup 路径保持官方 V2 代码不动；
- `V2EfaRuntime.launch_native_hybrid_dispatch()` 暴露 native main kernel launch；
- `V2EfaRuntime.launch_dispatch_copy_epilogue()` 暴露官方 V2 copy epilogue；
- Python `ElasticBuffer.dispatch()` 已切到 native hybrid 主路径，缺少真实
  `dev_comm/window/buffer/workspace` resource 时直接报错，不再把 scaffold/materialize
  路径当 fallback。
仍缺：
- 服务器上 build 扩展并跑 JIT compile；
- 从官方 DeepEP `ElasticBuffer` 或新的 resource binding 暴露
  `dev_comm/window/buffer/workspace/mapped_host_workspace/host_workspace` 指针；
- CPU-sync 精确长度路径需要 `host_workspace_ptr` reader；当前主路径先支持
  `do_cpu_sync=False` worst-case allocation。

**fork 范围：只替换 `ncclTeamTagRail`（EFA scaleout）操作；保留 `ncclTeamTagLsa`（NVLink scaleup）。**

| GIN 调用 | 位置 | Tag | 处理 |
|---------|-----|-----|------|
| `gin.put<ncclTeamTagRail>` rank_count | notify warp, SM0 | EFA | D2H FIFO（notify cmd） |
| `gin.put<ncclTeamTagRail>` expert_count | notify warp, SM0 | EFA | D2H FIFO（notify cmd） |
| `gin.put<ncclTeamTagRail>` payload | scaleout warp | EFA | D2H FIFO（payload cmd） |
| `gin.red_add_rel<ncclTeamTagRail>` tail | scaleout warp | EFA | D2H FIFO（tail cmd） |
| `gin.put_value<ncclTeamTagLsa>` rank count | notify warp | NVLink | **保留不动** |
| `gin.red_add_rel<ncclTeamTagLsa>` expert count | notify warp | NVLink | **保留不动** |
| `gin.get_sym_ptr<ncclTeamTagLsa>` TMA dst | forward warp | NVLink | **保留不动** |
| `gin.get_sym_ptr<ncclTeamTagLsa>` tail ptr | forward warp | NVLink | **保留不动** |

**GPU barrier 处理：**
- 开始处的 `comm::gpu_barrier<…kHybridDispatchTag0>(gin, …)` 使用 ncclTeamTagRail；Phase 1 仍有 Python `dist.barrier()`，直接移除这个 GPU barrier 即可。
- 结尾的 `comm::gpu_barrier<…kHybridDispatchTag1>(gin, …, do_scaleout=false)` 是 scaleup-only NVLink barrier；保留不动。

**新增 kernel 参数（EFA fork 专属）：**
```cpp
void*     d2h_queue_base,    // D2H TransferCmd ring buffer
uint32_t* d2h_head,          // ring head（GPU 写）
uint32_t  d2h_capacity,
uint64_t  buffer_base,       // ElasticBuffer.buffer GPU 指针（用于 offset 计算）
uint64_t  workspace_base,    // GPU workspace 指针（用于 tail offset 计算）
int       num_efa_lanes,     // EFA lane 数
// epoch 在 Phase 1 不需要；Phase 3 加入
```

#### 1a. GPUDirect RDMA smoke test（hard gate）

```python
buf_send = torch.zeros(4096, dtype=torch.uint8, device="cuda")
buf_recv = torch.zeros(4096, dtype=torch.uint8, device="cuda")
conn = ep.V2EfaConnection(device_index=0, local_addr=buf_send.data_ptr(), bytes=4096)
# 验证 ibv_reg_mr(cuda_ptr) 成功
# 验证 RDMA write 到 peer buf_recv.data_ptr() 后 GPU tensor 内容正确
```

若失败，参考 `uccl-ep/src/rdma.cpp` 的 DMA-BUF 路径修 `open_lane()`。

#### 1b. 注册 V2 buffer 为 EFA RDMA MR

注册**整个** `ElasticBuffer.buffer` 为单一 MR（每 lane 一次）：

```
sender (per lane):
  mr_local = ibv_reg_mr(lane_pd, buffer_base, total_buffer_bytes, IBV_ACCESS_LOCAL_WRITE)

receiver (per lane):
  mr_remote = ibv_reg_mr(lane_pd, buffer_base, total_buffer_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE)
  + 注册 workspace: mr_ws = ibv_reg_mr(..., workspace_base, workspace_bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE)

bootstrap: 通过 all_gather_object 交换 buffer_mr_base、buffer_rkey、
           workspace_mr_base、workspace_rkey

offset 计算（统一）：
  local_offset  = send_slot_ptr  - buffer_base   // 相对 mr_local.addr
  remote_offset = recv_slot_ptr  - buffer_base   // 相对对端 mr_remote.addr
  tail_offset   = tail_ptr       - workspace_base // 相对对端 mr_ws.addr
```

`buffer_base` 统一用于 local 和 remote，因为 V2 buffer layout 对所有 rank 相同（symmetric design）。

#### 1c. 替换 notify warp 的 `gin.put<ncclTeamTagRail>` → D2H FIFO notify command

notify warp (sm_idx == 0, thread_idx < kNumScaleoutRanks) 发送两类 count：
```cpp
// 替换 gin.put<ncclTeamTagRail>(dst.rank_count, src.rank_count, kNumScaleupRanks * sizeof(int), dst_rank)
// → per-rank 一次 RDMA write，固定大小 kNumScaleupRanks * sizeof(int)

d2h_push(V2TransferCmd{
    .dst_rank   = dst_scaleout_rank_idx,         // thread_idx
    .efa_lane   = dst_scaleout_rank_idx % num_efa_lanes,  // notify 用 rank % lanes
    .local_off  = (uint64_t)workspace.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx) - workspace_base,
    .remote_off = (uint64_t)workspace.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx) - workspace_base,
    .bytes      = kNumScaleupRanks * sizeof(int),
    .kind       = kNotifyRankCount,
    .signaled   = true,
});
// 类似地，expert_count 替换为 kNotifyExpertCount cmd
```

注意：notify warp 用的 MR 是 workspace MR（不是 buffer MR），因为 count 在 GPU workspace。

#### 1c'. 替换 `gin.put()` → D2H FIFO payload command（scaleout warp 发送正文数据）

```cpp
// channel_idx 决定 efa_lane，不是 expert_id
const uint64_t efa_lane = channel_idx % num_efa_lanes;

const uint64_t local_off =
    (uint64_t)scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr()
    - (uint64_t)buffer_base;

const uint64_t remote_off =
    (uint64_t)scaleout_recv_buffer
        .get_rank_buffer(scaleout_rank_idx)
        .get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx)
        .get_token_buffer(stored_dst_slot_idx).get_base_ptr()
    - (uint64_t)buffer_base;

d2h_push(V2TransferCmd{
    .dst_rank  = stored_dst_scaleout_rank_idx,
    .efa_lane  = efa_lane,      // channel → lane
    .local_off = local_off,
    .remote_off = remote_off,
    .bytes     = token_bytes,
    .kind      = kDispatchPayload,
    .signaled  = false,          // payload unsignaled
});
```

#### 1d. 替换 `gin.red_add_rel()` → per-channel tail write to GPU workspace

**tail word 格式（Phase 1 与 V2 原版相同，Phase 3 扩展 epoch）：**
```
Phase 1（格式与 V2 完全兼容）：
  int64_t tail_word = math::pack2<int, int64_t>(finish_flag, tail_count)
  内存布局：lo32 = finish_flag, hi32 = tail_count

Phase 3 扩展（去 barrier 时修改 lo32）：
  int64_t tail_word = math::pack2<int, int64_t>((epoch << 1) | finish_flag, tail_count)
  lo32 = (epoch << 1) | finish_flag   // epoch 31 bits，finish 1 bit（bit0）
  hi32 = tail_count                   // 不变

注意：epoch 31 bits，2^31 次 dispatch 后才回绕，实际可忽略。
      回绕前 forward warp 不会看到 epoch 混淆，因为旧 kernel dispatch 完成后才启动新一轮。
```

**EFA native 与 V2 atomic add 的差异：**
- V2 原版：`gin.red_add_rel(ptr, delta)` → 远端 **原子加**
- EFA native：CPU proxy 在同 QP 的 RDMA write 写**绝对值**

差异在于：V2 原版可以有多个 sender 并发 add 同一个 tail entry，EFA native 每个 tail entry 对应一个唯一 sender（`channel_signaled_tail_ptr[channel_idx][scaleout_rank_idx]` 的 `scaleout_rank_idx` 就是 sender 的 rank），所以不存在竞争，RDMA write 绝对值等价。

**tail 的生成主体是 GPU scaleout warp，不是 CPU proxy。**

GPU 在原来 `update_scaleout_tail()` 的位置，已知精确的 `tail_count`（由 GPU 计算），
应在此位置生成 tail `V2TransferCmd` 写入 D2H FIFO，CPU proxy 按 FIFO 顺序 post。
CPU proxy 不应自行推导 tail_count，因为 local bypass、`skip_scaleout_rank`、
coalescing、deduplication 等都会导致 CPU 算错。

```cpp
// 在 scaleout warp 的 update_scaleout_tail() 位置（每 3 token 或 finish 时）：
if (should_update && lane_idx < kNumScaleoutRanks) {
    const uint64_t tail_off =
        (uint64_t)workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx)
        - (uint64_t)workspace_base;

    const int64_t tail_word = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
    // Phase 1: finish_flag = 0 or 1（与 V2 原版格式相同）
    // Phase 3: 改为 (epoch<<1)|finish_flag

    d2h_push(V2TransferCmd{
        .dst_rank   = lane_idx,                        // lane_idx = 目标 scaleout rank
        .efa_lane   = channel_idx % num_efa_lanes,     // 与 payload 同 lane！
        .remote_off = tail_off,                        // 相对 workspace MR base
        .signal_value = tail_word,                     // 绝对值，不是 delta
        .bytes      = sizeof(int64_t),
        .kind       = kDispatchSignal,
        .signaled   = true,   // tail signaled，用于 CQ flow control
        // payload writes（unsignaled）已先写入同一 QP，EFA SRD 同 QP ordering 保证
        // tail 到达时 payload 一定已到达，不需要等 payload CQE 确认再发 tail
    });
}
```

**receiver forward warp（Phase 1）：** 仍读 `workspace.channel_signaled_tail_ptr`（`ld_acquire_sys`），
与 V2 原始代码完全相同，只是值来源从 GIN atomic add 变成 CPU proxy 的 RDMA write 绝对值。
Phase 1 不需要额外 epoch 检查——清零 tail 和 Python `dist.barrier()` 已保证无轮次混淆。

**Phase 3 额外改动：** forward warp spin-wait 增加 epoch 检查：`(lo32 >> 1) == current_epoch && tail_count > old_tail`；同时去掉末尾的 `*channel_signaled_tail_ptr = 0` 清零（epoch 区分轮次后不需要清零）。

#### 1e. forward warp 与 epilogue

- forward warp 结构保留 V2 原始语义（chunk streaming，build metadata）
- forward warp 内 scaleup NVLink TMA store（`gin.get_sym_ptr<ncclTeamTagLsa>`）**不改动**
- `dispatch_copy_epilogue` 保留 V2 原始语义（消费 scaleup_buffer）
- Phase 1 改动集中在：1c(notify warp RDMA) + 1c'(payload RDMA) + 1d(tail RDMA) + 去除开头 GPU barrier

---

### 阶段 2：多线程持久化 CPU 代理

**代码状态（2026-06-02）：C++ proxy 已切到 native dispatch 并发 drain 模式；服务器验证待跑。**

阶段 1 后 proxy post rate 大幅增加，Python 单线程 drain 成为瓶颈：

- C++ proxy 在 native hybrid kernel launch 前启动，kernel/forward warp 等 tail 时
  proxy 已经并发 drain D2H FIFO；
- 4 个持久 C++ proxy 线程（参考 UCCL-EP 论文 Figure 17，当前 API 已支持多 queue；
  Python 主路径先接单 queue，后续按 channel sharding 扩成多 queue）
- 每线程负责 `num_channels / 4` 个 channel + 对应 EFA lane
- `std::atomic<uint64_t>` completion counter 替代 Python `poll_completions()` loop
- GPU kernel 按 `channel_idx % num_fifo_queues` 路由 TransferCmd 到对应队列

---

### 阶段 3：去掉 pre-dispatch `dist.barrier()`

阶段 1 的 tail word 保持 V2 原格式；阶段 3 将 tail word 的 lo32 扩展为
`epoch + finish_flag` 后，去 barrier 只需：
1. receiver 不清零 tail（epoch 区分轮次）
2. sender tail write 带当前 epoch
3. forward warp spin-wait `epoch == current_epoch`
4. Python dispatch 前 `epoch += 1`，去掉 `dist.barrier()` 调用

预期：`stage_and_pre_barrier_ms` 0.9ms → 0.05ms。

---

### 阶段 4：combine native path（fork `hybrid_combine.cuh`）

与 dispatch 对称，替换 GIN 为 D2H FIFO + EFA write：
- 保留 combine forward warp 和 reduce epilogue 结构
- 支持 topk>1 多 contributor reduce
- 去掉 `_semantic_combine_data` fallback

---

### 阶段 5：性能调优

- **QP 数量调优**：每 EFA NIC 可多 QP，但必须保持 channel affinity：
  同一 channel 的所有 payload write 和 tail write 只能用同一个 QP，
  不能 round-robin 到不同 QP（否则 tail ordering 失效）。
  若一个 channel 需要多 QP，则需要 per-QP tail 或显式 completion aggregation，
  这是与当前设计的大改动，阶段 5 才考虑。
- **`kMaxInflight`**：控制 D2H queue 深度，防止 EFA CQ 溢出
- **intra-node token 去重**：`skip_scaleout_rank` 已设置，pack 阶段需完整跳过本地 rank
- **Token coalescing**：V2 channel 遍历是 `token_idx += kNumChannels`，
  local `scaleout_send_buffer` 中 token 是非连续的（stride = kNumChannels）；
  remote `scaleout_recv_buffer[channel][slot]` 可能连续。
  合并条件：local 和 remote **都** 连续时才合并为单次 RDMA write；
  否则用 multi-SGE list 或 channel-contiguous send staging，不能默认合并。

---

## 七、性能路线图

| 阶段 | 关键改动 | 预期 dispatch GB/s | 说明 |
|------|---------|-------------------|------|
| 现状（scaffold） | — | 2.91 | 两次额外 GPU memcpy |
| 0a | signal-only CQE | ~3.2 | completion_wait 减少 |
| scaffold + 多线程 proxy | 阶段 2 alone | 15–25 | GPU memcpy 瓶颈仍在 |
| **阶段 1（fork）** | GPUDirect + scaleout_recv_buffer | **基线大幅提升** | 消灭两次额外 copy |
| 1+2+3 | fork + 多线程 + 无 barrier | **~60–70** | EFA streaming pipeline |
| 1+2+3+4+5 | + combine + 调优 | **~90** | 目标完成 |

---

## 八、完成判定标准

- [ ] GPUDirect smoke test：`ibv_reg_mr(cuda_ptr)` + RDMA write 到对端 GPU tensor 成功
- [ ] EP16 dispatch + combine correctness（topk=8，do_expand=True，multi-contributor reduce）
- [ ] forward warp 直接消费 `scaleout_recv_buffer`（EFA RDMA 落点），并写入 `scaleup_buffer`；
  `dispatch_copy_epilogue` 保持消费 `scaleup_buffer`（V2 原始路径）；无独立 EFA window 中转
- [ ] `_semantic_dispatch_data` / `_semantic_combine_data` 从生产路径消失
- [ ] `dist.barrier()` 在 dispatch/combine 热路径中不再出现
- [ ] forward warp streaming tail（每 3 token 推进）正确工作
- [ ] README-style EP16 dispatch bench **>= 80 GB/s**（否则不算完成）
- [ ] Timing breakdown 能用 RDMA post 数、CQ rate、lane utilization 解释性能

---

## 附录：设计图

### 目标 native V2 数据流

```
GPU notify warp (SM0)               CPU proxy (notify lane)           GPU receiver workspace
-------------------                 -----------------------           ----------------------
count rank/expert per SM
  → GPU workspace (local atomic)
wait kNumSMs arrive
→ D2H FIFO push kNotifyRankCount    post RDMA write (signaled)
→ D2H FIFO push kNotifyExpertCount  remote = receiver.workspace.scaleout_{rank,expert}_count
                                    receiver spin-waits on count > 0
                                    then NVLink (GIN) aggregate to scaleup ranks

GPU scaleout warp                   CPU proxy (per channel)           GPU receiver buffer
-----------------                   -----------------------           -------------------
TMA store x[token]
→ scaleout_send_buffer[token_idx]
        |
        v
D2H FIFO push V2TransferCmd ──────→ poll FIFO
  kind=payload                       decode V2TransferCmd
  efa_lane = chan % lanes             post RDMA write (unsignaled)
  local_off = send_slot - buffer_base  local = sender GPU scaleout_send_buffer
  remote_off = recv_slot - buffer_base remote = receiver GPU scaleout_recv_buffer
  signaled = false                   ↓
        |                            (every 3 tokens or finish:)
        |                            post RDMA write (signaled, same QP)
        |                            value = pack2(finish_flag, tail_count)  [lo=finish, hi=count]
D2H FIFO push V2TransferCmd ──────→ remote = receiver GPU workspace.channel_tail[chan][src_rank]
  kind=tail                          ↓
  efa_lane = chan % lanes (same!)     poll CQ (signaled completions only)
  remote = workspace.tail_ptr                      │
  signal_value = pack2(finish,count)               │ EFA RDMA write (GPUDirect)
  signaled = true                                  ▼
                                    payload lands in scaleout_recv_buffer[src_rank][chan][slot]
                                    tail lands in workspace.channel_tail[chan][src_rank]
                                                   │
                                                   ▼ GPU forward warp (unchanged in Phase 1)
                                    spin-wait: channel_tail > old_tail
                                    copy: scaleout_recv_buffer → scaleup_buffer (via NVLink GIN)
                                    build: token_metadata_at_forward, dst_buffer_slot_idx
                                                   │
                                                   ▼ dispatch_copy_epilogue (V2 JIT)
                                    scaleup_buffer → recv_x / recv_topk_idx / recv_src_metadata
```

### 核心思想

- **transport 方法像 UCCL EP V1**：GPU 写 D2H FIFO，CPU proxy drain，EFA verbs post
- **数据语义像 DeepEP V2**：payload 从 `scaleout_send_buffer` 读，写到 `scaleout_recv_buffer`，tail 写 GPU workspace，forward warp 和 epilogue 保持 V2 原始语义
- **ordering 靠 EFA SRD 同 QP**：payload unsignaled + tail signaled，同 `efa_lane` 的 QP 保证 tail 可见时 payload 已可见
- **scaleup NVLink 不动**：forward warp 里 `gin.get_sym_ptr<ncclTeamTagLsa>` TMA store 仍走 GIN；forked kernel 仍接受 `nccl_dev_comm / nccl_window` 参数用于 NVLink
- **tail word 格式**：`pack2(finish_flag, tail_count)` → `{lo32=finish, hi32=count}`；EFA 写绝对值（非 delta）；Phase 3 在 lo32 里内嵌 epoch
