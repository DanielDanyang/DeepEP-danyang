# DeepEP V2 on AWS EFA: Native UCCL-EP 重写计划

## 结论

当前 `uccl-ep` 的主体不是 DeepEP V2 native backend。它仍然以 UCCL/DeepEP V1
normal path 为骨架：

- `src/internode.cu` / `src/intranode.cu` / `src/layout.cu` 是静态 CUDA kernel。
- 数据面依赖 `SourceMeta`、`rank_prefix_matrix`、`rdma_channel_prefix_matrix`、
  `gbl_channel_prefix_matrix`、`recv_rdma_rank_prefix_sum` 等 V1 staged layout。
- Python 层 `proxy_transport.py` 虽然包装成 V2 handle 形状，但实际仍在驱动 V1 风格的
  prepare/dispatch/combine runtime。
- 这和 DeepEP V2 的真实执行模型不匹配。V2 的核心 kernel 是按运行时配置 JIT 编译的
  `.cuh`，并且围绕 `BufferLayout`、`TokenLayout`、expanded dispatch、
  reduced combine、`token_metadata_at_forward`、`channel_linked_list` 等结构工作。

因此后续不应继续在 V1 静态 kernel 上打补丁。正确方向是把 `uccl-ep` 重写成
DeepEP V2 JIT backend：复用 AWS EFA proxy/RDMA 基础设施，但删除 V1 EP 协议层。

## 当前错误结构

```text
Python wrapper pretending to be V2
        |
        v
proxy_transport.py
        |
        v
V1-style native extension API
        |
        +--> intranode_prepare/dispatch/combine
        +--> internode_prepare/dispatch/combine
        |
        v
static CUDA .cu kernels
        |
        +--> SourceMeta staged token
        +--> rank/channel prefix matrices
        +--> RDMA/NVL packed buffer
        +--> proxy put/atomic per staged packet
```

这条路径的问题不是某一个 kernel 写得不够快，而是抽象层错了：它先把 V2 tensor 和
metadata 投影回 V1 的 packed/staged token 世界，再尝试通过 EFA proxy 传输。这样会天然
丢掉 V2 的 expanded/reduced layout 语义，也无法跟随 V2 JIT 根据 hidden/topk/expert/rank
配置生成不同 kernel 的方式。

## 目标结构

```text
DeepEP V2 ElasticBuffer
        |
        v
V2-native AWS EFA backend
        |
        +--> JIT build dispatch .cuh variant
        +--> JIT build combine .cuh variant
        |
        v
V2 semantic descriptors
        |
        +--> dispatch descriptor:
        |       dst_rank, lane, expert_id,
        |       src token group, expanded slot group, count, payload bytes
        |
        +--> combine descriptor:
                src/dst rank, expert_id,
                reduced token group, original token slot, reduce op, count
        |
        v
AWS EFA proxy transport
        |
        +--> device enqueue batched proxy commands
        +--> host proxy posts EFA writes/sends
        +--> receiver writes directly into V2 layout
```

新的 `uccl-ep` 不再拥有一套独立的 V1 EP layout。它只做一件事：替换 DeepEP V2 里
Gin device communication 的跨机传输实现，同时保持 V2 buffer/handle/kernel 语义。

## 必须删除的 V1 内容

这些文件和接口不应该继续作为 native V2 的组成部分：

- `src/internode.cu`
  - 删除 `SourceMeta`。
  - 删除 `internode_prepare` / `internode_dispatch` / `internode_combine`。
  - 删除 `rdma_channel_prefix_matrix`、`gbl_channel_prefix_matrix`、
    `recv_rdma_rank_prefix_sum` 数据面。
  - 删除按 V1 packed token buffer 发 payload 的 kernel。
- `src/intranode.cu`
  - 删除 V1 intranode `rank_prefix_matrix` 和 NVLink packed path。
  - V2 本身已经有 scaleup/scaleout hybrid JIT kernel，不应在这里维护第二套 intranode
    EP。
- `src/layout.cu`
  - 删除 V1 layout 计算。V2 layout 应来自 DeepEP V2 `BufferLayout` / `TokenLayout`。
- `include/internode.cuh`、`include/intranode.cuh`、`include/layout.hpp`
  - 跟随上述 `.cu` 一起删除。
- `include/ep_config.hpp`
  - 删除依赖 `internode::get_source_meta_bytes()` 的 staging buffer size 计算。
  - 重写成 V2 EFA workspace 描述，只描述 proxy queues、registered windows、
    descriptor buffers 和 receiver scratch。
- `deep_ep_v2_wrapper/deep_ep/proxy_transport.py`
  - 删除 `_LegacyProxyBuffer` 式的兼容调度层。
  - 删除 `rank_prefix_matrix` / `rdma_channel_prefix_matrix` handle 字段。
  - 改成直接调用 V2-native runtime，返回官方 V2 handle 语义。
- `src/uccl_ep.cc`
  - 删除暴露给 Python 的 V1 prepare/dispatch/combine binding。
  - 保留或重写为 `V2EfaRuntime` binding。
- `Makefile` 和 `setup.py`
  - 停止 glob 编译 `src/*.cu`。
  - 显式列出 native V2 所需源文件，避免旧 `.cu` 被无意重新带回 build。

删除完成后的硬性检查：

```bash
rg "SourceMeta|rank_prefix_matrix|rdma_channel_prefix_matrix|gbl_channel_prefix_matrix|recv_rdma_rank_prefix_sum" uccl-ep
rg "internode_prepare|internode_dispatch|internode_combine|intranode_prepare|intranode_dispatch|intranode_combine" uccl-ep
```

以上命令在 native V2 数据面中应无命中。少量历史文档命中可以保留，但源码和 public
API 不能再命中。

## 可以复用的内容

这些内容和 V1 EP 语义耦合较弱，可以作为 EFA transport substrate 复用，必要时改名：

- `src/rdma.cpp`、`include/rdma.hpp`
  - EFA device discovery、MR registration、QP/CQ 管理、remote key exchange 可以复用。
  - 但其中 command payload 里带 `low_latency_buffer_idx`、expert counter、V1 buffer offset
    的部分要拆出来重写。
- `src/proxy.cpp`、`src/uccl_proxy.cpp`、`include/proxy*.hpp`
  - host proxy thread、doorbell、CQ polling、批量 post 的框架可以复用。
  - 旧的 V1 command format 不能直接复用为 V2 protocol。
- `include/d2h_queue_*`、`include/ring_buffer*.cuh`
  - device-to-host queue 思路可以复用。
  - 字段命名和 encoding 要从 `low_latency/is_combine/expert` 改成 V2 descriptor/packet。
- `include/common.hpp`、`include/exception.cuh`、`include/ep_launch.cuh`
  - 通用 assert、launch helper 可以复用。
- Python 里的 topology/bootstrap 工具
  - `utils_uccl.py` 中的 rank/node/local rank 发现可以保留。
  - 但返回对象不应再构造 V1 transport handle。

## 新目录建议

```text
uccl-ep/
  include/v2_efa/
    descriptor.hpp
    runtime.hpp
    proxy_queue.cuh
    dispatch_jit.cuh
    combine_jit.cuh
    workspace.hpp

  src/v2_efa/
    runtime.cc
    bootstrap.cc
    proxy_transport.cc
    descriptor_kernels.cu

  deep_ep_v2_wrapper/deep_ep/
    buffers/elastic.py
    efa_backend.py
```

`dispatch_jit.cuh` 和 `combine_jit.cuh` 不应该是旧 static kernel 的新名字。它们应该以
DeepEP V2 官方 `.cuh` 为母体，保持 JIT 参数化方式，只把其中跨机 `ncclGin` put/signal/flush
段替换成 AWS EFA proxy descriptor enqueue。

## Dispatch 重写方案

V2 dispatch 的核心不是生成 V1 packed send buffer，而是直接服务 expanded layout。

```text
topk_idx / token layout
        |
        v
count + slot assignment
        |
        v
native dispatch descriptors
        |
        +--> local/scaleup path: follow original V2 JIT
        |
        +--> remote/scaleout path:
                pack small semantic batches by (dst_rank, expert, lane)
                enqueue EFA proxy packets
                receiver writes expanded output slots directly
```

descriptor 最小字段：

```text
DispatchDescriptor
  dst_scaleout_rank
  dst_scaleup_lane
  expert_id
  src_token_base or src_token_index_ptr
  expanded_slot_base
  count
  hidden_bytes
  scale_bytes
  flags
```

注意：同一个 expert 的 token 在原始 token order 中不一定连续，所以 descriptor 不能只是假设
`src_token_begin + count` 连续。需要支持两层表达：

```text
ExpertBatch
  (dst_rank, lane, expert_id)
  first_segment
  num_segments

Segment
  src_token_begin or src_index_ptr
  expanded_slot_begin
  count
```

小消息性能优化的关键是 semantic batching，而不是 EFA 只做大块 memcpy。batch 的边界应该是
V2 语义里的 expert/lane/slot，而不是 V1 packed buffer chunk。

## Combine 重写方案

V2 combine 应该从 expanded/reduced-combine metadata 反向生成 descriptor，不再读取 V1
`SourceMeta`。

```text
expanded output / expert results
        |
        v
V2 combine metadata
        |
        +--> token_metadata_at_forward
        +--> channel_linked_list
        +--> psum_num_recv_tokens_per_expert
        |
        v
native combine descriptors
        |
        v
EFA proxy gather/reduce/send
        |
        v
owner rank reduced output layout
```

descriptor 最小字段：

```text
CombineDescriptor
  dst_original_rank
  src_scaleout_rank
  expert_id
  expanded_slot_base or expanded_slot_index_ptr
  reduced_token_slot
  count
  hidden_bytes
  reduce_op
  flags
```

receiver 端要直接落到 V2 reduced-combine layout，让官方 V2 reduce epilogue 或其 AWS 变体消费，
而不是先回到 V1 combined token staging。

## JIT 接入方式

新的 runtime 要仿照 DeepEP V2 官方代码，而不是仿照 UCCL V1：

- `csrc/kernels/elastic/dispatch.hpp`
  - 复制其 `DispatchRuntime` 的 JIT 参数组织方式。
  - 保持 `BufferLayout` / `TokenLayout` / `num_sms` / `hidden` / `num_topk` 等编译期参数。
  - 将 Gin device communication include 替换为 `v2_efa/proxy_queue.cuh`。
- `csrc/kernels/elastic/combine.hpp`
  - 同样保持 `CombineRuntime` / `CombineReduceEpilogueRuntime` 的 JIT 编译模型。
  - 替换跨机传输段，不替换 V2 reduce 语义。
- `csrc/elastic/buffer.hpp`
  - Python wrapper 应尽量贴近这里的 handle 和 tensor allocation，而不是维护旧 UCCL buffer。

目标调用链：

```text
Python ElasticBuffer.dispatch()
        |
        v
V2EfaRuntime.launch_dispatch()
        |
        v
jit::compiler->build("v2_efa_dispatch", params)
        |
        v
generated CUDA from .cuh
        |
        v
device enqueue EFA proxy descriptors
```

## 九步开发计划

### 1. 删除旧数据面

- 从 build 中移除 `internode.cu`、`intranode.cu`、`layout.cu`。
- 删除对应 header 和 binding。
- 删除 `proxy_transport.py` 中所有 V1 handle 字段。
- 暂时让 `ElasticBuffer` 在 AWS native backend 未完成路径上 fail-fast，避免误跑旧实现。

交付标准：源码中不再有 V1 prefix matrix / `SourceMeta` / static internode binding。

### 2. 建立 native V2 runtime 壳

- 新增 `V2EfaRuntime` nanobind 类。
- 暴露 `init`, `alloc_workspace`, `launch_dispatch`, `launch_combine`。
- 先只支持 AWS EFA，多机 EP8x2，不做 IB/NVSHMEM 兼容。

当前进度：

- 已新增 `V2EfaRuntime` skeleton。
- 已暴露 runtime config、descriptor stats、workspace plan。
- dispatch/combine 仍明确返回未实现错误。

交付标准：Python 可以 import，runtime 可以初始化 EFA proxy 资源，但 dispatch/combine
明确返回未实现错误。

### 3. 接入 V2 JIT 编译器

- 从官方 V2 `dispatch.hpp` / `combine.hpp` 抽出 JIT 参数构造。
- 新增 `include/v2_efa/dispatch_jit.cuh` 和 `include/v2_efa/combine_jit.cuh`。
- 第一版 kernel 可以只做 descriptor 生成和校验，不发网络。

交付标准：不同 hidden/topk/expert/num_sms 配置能生成不同 JIT kernel。

### 4. 设计 V2 EFA command queue

- 重新定义 device-to-host command，字段围绕 V2 descriptor，而不是 V1 buffer offset。
- proxy 支持按 `(dst_rank, expert, lane)` 批量 post。
- 保留 EFA MR/QP/CQ 管理，删除旧 `low_latency_buffer_idx` 语义。

当前进度：

- 已定义 dispatch/combine descriptor 和 proxy command scaffold。
- 已实现 CPU reference dispatch planner，用来固定 CUDA/JIT descriptor 语义。
- dispatch planner 当前按 `(dst_scaleout_rank, dst_scaleup_lane, expert_id)` 做
  semantic batching，并保留 `topk_slot`。
- 已实现 CPU reference combine planner，从 dispatch plan 反推 reduced-combine
  descriptor，保留 expanded/reduced slot 和 `topk_slot`。
- `dispatch_jit.cuh` 已实现 device-side reference descriptor generator，后续需要把
  serial expert scan 并行化并接入 DeepEP V2 `hybrid_dispatch` JIT。
- `combine_jit.cuh` 已实现从 dispatch descriptor 反推 combine descriptor 的
  device-side reference generator，后续需要改为直接读取 V2 forward metadata。
- 已新增 reference proxy command planner，将 descriptor 转成 payload/signal command，
  后续 device enqueue 和 host proxy command format 都应对齐这套语义。
- 已新增 device-side reference enqueue kernel，把 dispatch/combine descriptor 写入
  `ProxyQueueView`。当前是 serial reference 版本，后续需要并行化并接到 retained
  EFA host proxy。
- 已新增 host loopback executor，用本地 byte buffers 验证 payload copy 和 signal write。
- proxy command 已包含 signal value 和目标 rank/lane；layout 里有
  `batch_payload_stride`，用于隔离 per-expert semantic batch 的远端 payload 区域。

交付标准：单机 loopback 或 fake remote 可以验证 descriptor enqueue/dequeue 正确。

### 5. Dispatch direct expanded layout

- 在 V2 dispatch JIT kernel 中生成 dispatch descriptor。
- sender 按 expert/lane semantic batch pack payload。
- receiver 直接写入 V2 expanded layout 和必要 metadata。

交付标准：EP8x2 dispatch correctness 通过，且不经过 V1 staging buffer。

### 6. Cached dispatch

- 复用 V2 handle 里的 descriptor/cache 信息。
- cached path 不能退回 `rank_prefix_matrix`。

交付标准：DeepEP V2 cached dispatch correctness 通过。

### 7. Combine direct reduced layout

- 从 V2 forward metadata 生成 combine descriptor。
- 支持 reduced-combine，receiver 直接写回 owner rank 的 reduced layout。
- reduce epilogue 保持 V2 语义。

交付标准：combine / reduced combine correctness 通过。

### 8. 性能 profiling 与调优

- 比较三条基线：
  - 官方 DeepEP V2 + NCCL GIN proxy。
  - 原始 UCCL EP V1 on AWS。
  - native V2 EFA backend。
- 观察：
  - 每 rank descriptor 数。
  - 每 descriptor 平均 payload。
  - proxy command 合并率。
  - EFA CQ/post 饱和度。
  - 每 NIC/rail 实际流量。
- 目标不是单纯做大块传输，而是复刻 UCCL EP 的 per-expert / low-latency semantic batching。

交付标准：README style EP8x2 有稳定 bw，瓶颈可以用 descriptor/proxy counters 解释。

### 9. 收敛到 README SM90 EP16 目标

- 逐步调 batch 粒度、lane mapping、proxy queue 深度、MR layout、receiver slot 写入策略。
- 每次改动记录到 `worklog.md`，保留命令、环境、bw、profiling 结论。
- 只有在两台机器空闲时运行 benchmark；发现其他用户 GPU 进程立即停止服务器操作。

交付标准：EP16 dispatch/combine 接近官方 README 中 SM90 多机量级，或明确证明 AWS EFA
proxy/native 约束下的上界并给出下一步需要改动的 transport 能力。

## 第一批代码改动顺序

```text
commit A: remove V1 public API and static kernels from build
commit B: add V2EfaRuntime skeleton and fail-fast Python wrapper
commit C: add V2 descriptor structs and unit tests
commit D: add JIT dispatch descriptor-generation kernel
commit E: connect descriptor queue to EFA proxy
commit F: implement dispatch direct expanded layout
commit G: implement combine direct reduced layout
commit H: benchmark/profiling counters and tuning
```

这之后 `uccl-ep` 才能被称为 native V2。此前所有从 V1 `internode.cu` 改出来的性能结果都只能
作为反例和诊断材料，不能作为最终实现基础。
