# UCCL-EP 原始代码与当前 DeepEP V2 AWS 适配详解

本文档解释三件事：

1. 原始 UCCL-EP/V1 代码是怎么工作的。
2. 当前 `uccl-ep` 里为了支持 DeepEP V2 做了哪些改动。
3. 为什么现在性能仍然差，以及下一步应该继续往哪里改。

结论先说在前面：当前代码还不是一个真正 native DeepEP V2 backend。它更像一个“V2 API + V2 metadata/CUDA epilogue + V1/UCCL legacy transport 数据面”的混合体。我们已经把一些 Python 热路径下沉到 CUDA，但 dispatch/combine 的核心数据面仍然是 UCCL-EP 为 DeepEP V1 设计的 high-throughput path，所以性能不会天然接近官方 DeepEP V2 on CX7 的水平。

## 1. 背景和目标

DeepEP 官方 V2 在 IB/CX7 机器上依赖 NCCL GIN device-side communication。GPU kernel 可以在 device 端发 `gin.put`、`gin.signal`、`gin.red_add_rel`，并依赖 IB/NIC 对小 RDMA write、ordering、remote atomic-like 语义的良好支持。

AWS p5en 的网络是 EFA。EFA 的大包吞吐很好，但它不是 IB verbs 语义的简单等价物：

- 大包 RDMA write 吞吐高。
- 小包 GPU-initiated 操作开销高。
- native remote atomic / ordering 能力弱，需要 proxy 和软件协议补足。
- NCCL GIN proxy 能跑通，但 DeepEP V2 dispatch 这种 4-8 KiB token 级小消息模式性能很差。

所以我们参考 UCCL-EP 的思路：GPU 不直接对 EFA 发大量细粒度 RDMA/atomic，而是把网络操作编码成小命令交给 CPU proxy。CPU proxy 使用 libibverbs/EFA 发 GPUDirect RDMA write/write-with-imm，并在 receiver 侧重建 ordering 和 tail/counter 语义。

整体目标图：

```text
官方 DeepEP V2 on IB/CX7
-----------------------

GPU kernel
  |
  | device-side gin.put / gin.signal / gin.red_add_rel
  v
NCCL GIN + IB NIC
  |
  v
remote GPU symmetric window


AWS 目标路径
------------

GPU kernel
  |
  | writes staging buffer
  | emits compact TransferCmd
  v
GPU -> CPU D2H queue
  |
  v
CPU proxy thread
  |
  | ibv_post_send(RDMA_WRITE_WITH_IMM / atomic emulation / quiet)
  v
EFA NIC
  |
  v
remote GPU staging buffer + receiver ordering state
```

## 2. 原始 UCCL-EP/V1 代码结构

当前 `uccl-ep/` 是从 UCCL-EP 拷过来的工作区。关键目录如下：

```text
uccl-ep/
  include/
    ring_buffer.cuh          # TransferCmd 定义与 D2H queue ring buffer
    d2h_queue_device.cuh     # device 端把 TransferCmd 写入 D2H queue
    uccl_ibgda.cuh           # 用 TransferCmd 模拟 NVSHMEM/IBGDA put/atomic/quiet
    internode.cuh            # high-throughput internode dispatch/combine 声明
    intranode.cuh            # NVLink/intranode path
    rdma.hpp                 # RDMA immediate data、QP/MR 辅助结构
    proxy*.hpp               # CPU proxy 上下文

  src/
    internode.cu             # V1 high-throughput internode kernels
    intranode.cu             # V1 intranode kernels
    rdma.cpp                 # EFA/verbs RDMA post_send、completion、ordering
    proxy.cpp                # CPU proxy 主循环
    uccl_ep.cc               # nanobind Python extension

  deep_ep_wrapper/           # 原始 UCCL-EP 对 DeepEP V1 Buffer API 的包装
  deep_ep_v2_wrapper/        # 当前我们新增的 DeepEP V2 ElasticBuffer 适配层
```

原始 UCCL-EP 的核心不是“一个 Python wrapper”，而是一套 GPU/CPU 协议：

```text
                 original UCCL-EP high-level architecture

         GPU SMs                                  CPU proxy threads
  +--------------------+                    +-------------------------+
  | dispatch/combine   |                    | poll D2H queue          |
  | CUDA kernels       |                    | batch TransferCmd       |
  +---------+----------+                    +------------+------------+
            |                                            |
            | write payload to staging buffer             |
            |                                            |
            | push 128-bit TransferCmd                   |
            v                                            |
  +--------------------+                                 |
  | D2H ring buffer    |---------------------------------+
  +--------------------+                                 |
                                                         v
                                            +-------------------------+
                                            | ibv_post_send           |
                                            | RDMA_WRITE_WITH_IMM     |
                                            | ATOMIC/QUIET/BARRIER    |
                                            +------------+------------+
                                                         |
                                                         v
                                            +-------------------------+
                                            | EFA NIC                 |
                                            +-------------------------+
```

### 2.1 TransferCmd

位置：

- `uccl-ep/include/ring_buffer.cuh`
- `uccl-ep/include/d2h_queue_device.cuh`
- `uccl-ep/include/uccl_ibgda.cuh`

`TransferCmd` 是 128 bit，即 16 字节。GPU kernel 不直接调用 host verbs，而是把一次 RDMA/atomic/quiet/barrier 操作编码成一个 `TransferCmd`，写入 GPU->CPU queue。

概念结构如下：

```text
128-bit TransferCmd

+----------------+----------------+----------------+----------------+
| cmd_type       | peer / rank     | src/dst offset | bytes / value  |
+----------------+----------------+----------------+----------------+
        |                |                |                |
        |                |                |                +-- payload bytes or atomic value
        |                |                +------------------- remote/local offset
        |                +------------------------------------ target rank/NIC context
        +----------------------------------------------------- WRITE / ATOMIC / QUIET / BARRIER
```

原始 UCCL-EP 里，很多函数名仍然像 NVSHMEM/IBGDA，例如 `nvshmemi_ibgda_put_nbi_warp`。但在这份代码里，它已经不是真的走 NVSHMEM，而是把 put/atomic 编码成 `TransferCmd`。

简化后的语义：

```text
GPU wants to do:

  put(remote_ptr, local_ptr, bytes, peer)

UCCL-EP actually does:

  1. write local payload into registered staging buffer
  2. create TransferCmd {
       type = WRITE,
       src_offset = ...,
       dst_offset = ...,
       bytes = ...,
       peer = ...
     }
  3. push TransferCmd to D2H queue
  4. CPU proxy posts ibv RDMA write
```

### 2.2 CPU proxy

位置：

- `uccl-ep/src/proxy.cpp`
- `uccl-ep/src/uccl_proxy.cpp`
- `uccl-ep/src/rdma.cpp`

CPU proxy 负责：

- 轮询 GPU 写入的 D2H queue。
- 解析 `TransferCmd`。
- 按 command 类型分类：WRITE / ATOMIC / QUIET / BARRIER。
- 批量调用 `ibv_post_send`。
- 处理 completion。
- 对 EFA 缺失的 ordering/atomic 语义做软件层补偿。

原始路径图：

```text
GPU D2H queue entries

   [WRITE] [WRITE] [WRITE] [ATOMIC] [QUIET] [BARRIER]
      |       |       |        |       |        |
      +-------+-------+--------+-------+--------+
                              |
                              v
                      CPU proxy poll loop
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
         RDMA WRITE      atomic emulation     drain/barrier
              |               |                |
              v               v                v
      RDMA_WRITE_WITH_IMM   pending update   completion fence
```

### 2.3 RDMA_WRITE_WITH_IMM 与 receiver-side ordering

位置：

- `uccl-ep/src/rdma.cpp`
- `uccl-ep/include/rdma.hpp`

EFA 上不能简单假设“data write 到了，tail/counter 就可以立刻发布”。UCCL-EP 用 immediate data 和 pending update 机制来维护顺序。

粗略流程：

```text
sender proxy                         receiver proxy / CQ
------------                         -------------------

post RDMA_WRITE_WITH_IMM
  imm = {seq, type, rank, ...}
       |
       v
EFA network
       |
       v
remote CQE with imm
       |
       v
receiver sees:
  - data write completion signal
  - seq / ordering metadata
       |
       v
apply_pending_updates()
       |
       v
publish atomic/tail/counter visible to GPU
```

这就是 UCCL-EP 适合 EFA 的关键：payload 仍然是 GPU memory 到 GPU memory 的 RDMA write，但 ordering 和 atomic-like 发布不依赖 EFA 原生 remote atomic。

## 3. 原始 UCCL-EP/V1 dispatch/combine 数据面

UCCL-EP 的原始 high-throughput EP 数据面是给 DeepEP V1 `Buffer` API 设计的。

### 3.1 原始 dispatch

位置：

- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`
- `uccl-ep/src/internode.cu`
- `uccl-ep/src/uccl_ep.cc`

虽然文件在 `deep_ep_v2_wrapper` 里，但 `Buffer` 本身是从 UCCL-EP V1 wrapper 迁过来的。核心 API 是：

```text
Buffer.dispatch(...)
  |
  +-- get_dispatch_layout(...)
  |
  +-- internode_dispatch(...)
        |
        +-- runtime.internode_dispatch(...)
              |
              +-- uccl::internode::notify_dispatch(...)
              +-- uccl::internode::dispatch(...)
```

原始 dispatch 数据流：

```text
local token x/topk
       |
       v
Python layout:
  num_tokens_per_rank
  num_tokens_per_rdma_rank
  is_token_in_rank
  num_tokens_per_expert
       |
       v
CUDA notify_dispatch
  exchange counts / clean buffers / setup prefix
       |
       v
CUDA dispatch
  copy token payload into RDMA/NVL staging
  emit TransferCmd for cross-node payload
       |
       v
CPU proxy posts EFA RDMA
       |
       v
remote staging buffer
       |
       v
NVL/local forwarding
       |
       v
recv_x, recv_topk_idx, recv_topk_weights, SourceMeta
```

### 3.2 原始 combine

Combine 是 dispatch 的反向。V1 的语义更接近：

```text
per received token reduced input
       |
       v
legacy combine data path
       |
       +-- maybe NVL local combine
       +-- maybe RDMA cross-node combine
       |
       v
combined_x at original source rank
```

关键问题是：DeepEP V2 的 reduced combine 输入可以是 expanded expert-slot layout，而 V1 legacy combine 期望的是 per-token layout。这是我们当前性能差和代码绕的一个核心来源。

## 4. DeepEP V2 原生语义与 V1 的差异

DeepEP V2 `ElasticBuffer` 相比 V1 `Buffer` 有明显不同：

```text
DeepEP V2 dispatch returns:

  recv_x
  recv_topk_idx
  recv_topk_weights
  handle
    |
    +-- do_expand
    +-- num_experts
    +-- expert_alignment
    +-- num_max_tokens_per_rank
    +-- num_sms
    +-- topk_idx
    +-- psum_num_recv_tokens_per_scaleup_rank
    +-- psum_num_recv_tokens_per_expert
    +-- recv_src_metadata
    +-- dst_buffer_slot_idx
    +-- token_metadata_at_forward
    +-- channel_linked_list
```

V2 的几个重要语义：

1. `dispatch` 有普通模式、expanded 模式、cached 模式。
2. `expanded dispatch` 不是简单返回 token-major layout，而是 expert-slot layout。
3. `combine` 有普通 combine 和 reduced combine。
4. `num_sms` 不是常量，官方 V2 通过 bandwidth model 根据拓扑和 `topk/expert` 估计。
5. V2 handle 的 metadata 会被后续 cached dispatch/combine 复用。

V1 与 V2 的 layout 差异：

```text
V1-style received token layout

recv_x:
  row 0 -> token A, already reduced/selected for rank
  row 1 -> token B
  row 2 -> token C

recv_topk_idx:
  row 0 -> local expert ids for token A
  row 1 -> local expert ids for token B


V2 expanded expert-slot layout

expanded_x:
  expert 0 slots:
    slot 0 -> token A topk[?]
    slot 1 -> token C topk[?]
  expert 1 slots:
    slot 2 -> token A topk[?]
  expert 2 slots:
    slot 3 -> token B topk[?]

recv_src_metadata:
  row token A -> [global_src_token, dst_slot, slot_for_topk0, slot_for_topk1, ...]
  row token B -> [global_src_token, dst_slot, slot_for_topk0, slot_for_topk1, ...]
```

所以如果只是把 V1 combine 包起来，就会需要一个额外转换：

```text
expanded expert-slot input
        |
        | gather slots for each token
        | sum valid topk slots
        v
per-token reduced input
        |
        v
legacy combine
```

这个转换以前在 Python/Torch 里做，非常慢且临时张量巨大；现在已经下沉到 CUDA，但它仍然说明核心 combine 数据面还不是 V2-native。

## 5. 当前我对 UCCL-EP 做过的改动

下面按时间/功能解释，而不是按 commit 罗列。

### 5.1 新增 DeepEP V2 wrapper 入口

新增路径：

- `uccl-ep/deep_ep_v2_wrapper/deep_ep/__init__.py`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`

目标是让测试脚本中的：

```python
import deep_ep
buffer = deep_ep.ElasticBuffer(...)
```

解析到 AWS/UCCL backend，而不是 upstream DeepEP package。

当前包装关系：

```text
tests/elastic/test_ep.py
        |
        v
deep_ep.ElasticBuffer          <-- V2-compatible wrapper
        |
        v
uccl-ep deep_ep_v2_wrapper/deep_ep/buffers/elastic.py
        |
        | delegates transport to
        v
uccl-ep deep_ep_v2_wrapper/deep_ep/buffer.py
        |
        | legacy UCCL-EP Buffer API
        v
uccl.ep nanobind extension
        |
        v
src/internode.cu + src/proxy.cpp + src/rdma.cpp
```

这张图也暴露了当前问题：V2 wrapper 下面还是 legacy `Buffer`。

### 5.2 V2 metadata 下沉

相关文件：

- `uccl-ep/src/internode.cu`
- `uccl-ep/include/internode.cuh`
- `uccl-ep/src/uccl_ep.cc`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`

新增/修改的 native helper：

```text
build_v2_dispatch_metadata(...)
```

它把 UCCL V1 transport 收到的 `SourceMeta` 和 `recv_topk_idx` 转成 V2 handle 需要的字段：

```text
legacy SourceMeta + recv_topk_idx
          |
          v
CUDA metadata kernels
          |
          +-- recv_src_metadata
          +-- psum_num_recv_tokens_per_scaleup_rank
          +-- psum_num_recv_tokens_per_expert
          +-- dst_buffer_slot_idx
          +-- raw_num_recv_tokens_per_expert
          +-- expanded expert slots
```

metadata 关系图：

```text
SourceMeta from UCCL transport

+---------------+-------------------------+--------------+---------------+
| src_rdma_rank | is_token_in_nvl_bits    | src_nvl_rank | src_token_idx |
+---------------+-------------------------+--------------+---------------+
        |                         |              |              |
        +-------------------------+--------------+--------------+
                                  |
                                  v
V2 recv_src_metadata row

+------------------+-----------------+------------+------------+-----+
| global_src_token | dst_buffer_slot  | topk0_slot | topk1_slot | ... |
+------------------+-----------------+------------+------------+-----+
```

注意：之前有一个 bug 是 NVL receiver 只保留了 `SourceMeta` 的前两个字段，导致 `src_token_idx` 错乱。后来修了，完整保留四个 int。

### 5.3 expanded payload scatter 下沉

相关 helper：

```text
build_v2_expanded_payload(...)
```

它把普通 dispatch 收到的 token-major payload scatter 到 V2 expanded expert-slot layout。

之前 Python/Torch 方式大概是：

```text
expanded_x[slots] = recv_x[token_indices]
expanded_weights[slots] = recv_topk_weights[token, topk]
```

现在变成 CUDA kernel：

```text
for each (token, topk):
    slot = recv_src_metadata[token, 2 + topk]
    if slot >= 0:
        expanded_x[slot] = recv_x[token]
        expanded_scale[slot] = recv_scale[token]
        expanded_weight[slot] = recv_topk_weight[token, topk]
```

ASCII 图：

```text
recv_x token-major

  token0  [xxxxxxxx]
  token1  [yyyyyyyy]
  token2  [zzzzzzzz]

recv_src_metadata slots

  token0 -> expert slot 4, slot 9
  token1 -> expert slot 2
  token2 -> expert slot 5, slot 6

CUDA scatter

expanded_x expert-slot

  slot2  [yyyyyyyy]
  slot4  [xxxxxxxx]
  slot5  [zzzzzzzz]
  slot6  [zzzzzzzz]
  slot9  [xxxxxxxx]
```

这一步对 expanded dispatch 性能帮助很大。

### 5.4 reduced combine slot gather 下沉

最新改动：

- `uccl-ep/include/internode.cuh`
- `uccl-ep/src/internode.cu`
- `uccl-ep/src/uccl_ep.cc`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`

新增 helper：

```text
build_v2_reduced_combine_input(...)
```

修改前 Python 路径：

```python
slots = handle.recv_src_metadata[:, 2 : 2 + num_topk].to(torch.long)
valid = slots >= 0
safe_slots = slots.clamp_min(0)
gathered = x[safe_slots]
gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
x = gathered.sum(dim=1)
```

这个问题很严重，因为它会产生：

```text
gathered shape = [num_recv_tokens, topk, hidden]
```

对于 EP16 / 8192 tokens / hidden 7168，这个临时张量非常大，而且完全走 PyTorch eager。

修改后：

```text
expanded_x [num_expanded_tokens, hidden]
recv_src_metadata slots
        |
        v
CUDA build_v2_reduced_combine_input
        |
        v
reduced_x [num_recv_tokens, hidden]
        |
        v
legacy combine
```

kernel 语义：

```text
for token in num_recv_tokens:
  for hidden_idx in hidden:
    acc = 0
    for k in topk:
      slot = recv_src_metadata[token, 2 + k]
      if slot >= 0:
        acc += expanded_x[slot, hidden_idx]
    reduced_x[token, hidden_idx] = acc
```

ASCII 对照：

```text
Before
------

expanded_x
   |
   | PyTorch advanced indexing: x[safe_slots]
   v
[num_recv_tokens, topk, hidden] huge temporary
   |
   | masked_fill + sum(dim=1)
   v
reduced_x


After
-----

expanded_x + recv_src_metadata
   |
   | one CUDA kernel
   v
reduced_x
```

性能变化：

```text
reduced combine before:
  ~30.4 ms, ~25 GB/s (SU)

reduced combine after:
  ~22.4 ms, ~34-35 GB/s (SU)
```

这证明 Python gather 确实是一个显著瓶颈，但不是唯一瓶颈。

### 5.5 SM 选择从固定 24 改为 V2 bandwidth model

你指出的 “固定 24 个 SM 怎么可能是 V2” 是对的。

之前 wrapper 中有：

```python
def get_theoretical_num_sms(...):
    return 24 if torch.version.cuda else 64
```

这明显不是 V2。现在改成接近 upstream DeepEP V2 的 bandwidth model：

```text
input:
  num_experts
  num_topk
  num_scaleout_ranks
  num_scaleup_ranks
  RDMA GB/s
  NVLink GB/s
  per-SM read/write GB/s

estimate:
  expected scaleout topk
  expected all-rank topk
  sm_read
  sm_write
  rdma_traffic
  nvlink_traffic

output:
  recommended num_sms
```

SM model 图：

```text
num_experts/topk/topology
          |
          v
expected_topk(num_scaleout_ranks)
expected_topk(num_ranks)
          |
          v
traffic model
  +-------------------+
  | HBM read/write    |
  | RDMA traffic      |
  | NVLink traffic    |
  +-------------------+
          |
          v
bandwidth bound
  max(RDMA time, NVLink time)
          |
          v
estimated SM count
          |
          v
legacy safety cap for current UCCL V1 data path
```

当前还有一个现实限制：底层数据面仍然是 UCCL legacy/V1 staging buffer。SM 变大后，legacy buffer size 会随 channel 数膨胀，并触发 C++ 里 `num_nvl_bytes <= int32` 的断言。所以现在自动 SM 会先按 V2 模型算，再受：

```text
EP_UCCL_MAX_AUTO_SMS
```

限制，默认 32。

这不是长期正确方案。长期正确方案应该是 V2 native buffer layout 支持更高 SM/channel 数，而不是靠 legacy cap。

## 6. 当前完整数据流

当前代码运行 `deep_ep.ElasticBuffer.dispatch(... do_expand=True)` 时，实际路径如下：

```text
DeepEP V2 API
  ElasticBuffer.dispatch(do_expand=True)
        |
        v
V2 wrapper
  _ensure_legacy_buffer()
  get_dispatch_layout()               <-- still V1/legacy layout
        |
        v
legacy UCCL Buffer.dispatch()
        |
        v
internode_dispatch()
        |
        v
UCCL V1 internode CUDA kernel
  notify_dispatch
  dispatch
        |
        v
TransferCmd -> CPU proxy -> EFA RDMA
        |
        v
recv_x, recv_topk_idx, recv_topk_weights, SourceMeta
        |
        v
new V2 native metadata kernel
  build_v2_dispatch_metadata
        |
        v
new V2 expanded payload kernel
  build_v2_expanded_payload
        |
        v
V2-style expanded output + handle
```

Combine/reduced-combine 当前路径：

```text
DeepEP V2 API
  ElasticBuffer.combine(handle=expanded_handle)
        |
        v
handle.do_expand == True
        |
        v
new CUDA kernel
  build_v2_reduced_combine_input
        |
        v
per-token reduced_x
        |
        v
legacy UCCL Buffer.combine()
        |
        v
internode_combine()
        |
        v
UCCL V1 combine data path
        |
        v
combined_x
```

这张图很重要：

```text
what is V2-native today?

  V2 API handle                  yes
  V2 recv_src_metadata           yes, native CUDA
  V2 expanded dispatch scatter   yes, native CUDA
  V2 reduced-combine gather      yes, native CUDA

what is still V1/legacy?

  dispatch layout                mostly V1
  RDMA/NVL staging layout        V1
  notify protocol                V1
  combine data movement          V1
  reduction protocol             V1
  buffer/channel sizing          V1
```

## 7. 当前 benchmark 结果和解释

最新 EP16，自动 SM：

```text
config:
  EP16 = 2 nodes x 8 GPUs
  tokens = 8192
  hidden = 7168
  topk = 8
  experts = 256
  FP8 dispatch
  #SM = 32

rank 0:
  * dispatch:          20 GB/s (SO),  66 GB/s (SU),  6106.753 us
  - expanded dispatch: 31 GB/s (SO), 102 GB/s (SU),  3934.980 us
  # cached dispatch:   46 GB/s (SO), 151 GB/s (SU),  2667.677 us
  @ combine:           13 GB/s (SO),  42 GB/s (SU), 18186.205 us
  + reduced combine:   10 GB/s (SO),  35 GB/s (SU), 22370.132 us

rank 8:
  * dispatch:          20 GB/s (SO),  67 GB/s (SU),  6000.972 us
  - expanded dispatch: 30 GB/s (SO),  99 GB/s (SU),  4053.083 us
  # cached dispatch:   47 GB/s (SO), 155 GB/s (SU),  2577.602 us
  @ combine:           13 GB/s (SO),  42 GB/s (SU), 18293.443 us
  + reduced combine:   10 GB/s (SO),  34 GB/s (SU), 22382.804 us
```

### 7.1 为什么 expanded dispatch 看起来还行

Expanded dispatch 已经有两个关键优化：

1. transport 用 UCCL proxy，而不是 NCCL GIN 小 put。
2. payload scatter 已经是 CUDA kernel，而不是 Python indexing。

所以它可以到 `~99-102 GB/s (SU)`。

### 7.2 为什么 cached dispatch 更快

Cached dispatch 跳过 layout/metadata 计算，复用 handle。当前能到 `~151-155 GB/s (SU)`。

这说明大包数据面不是完全跑不动；问题更多在 uncached layout/metadata 和 combine/reduce 语义。

### 7.3 为什么普通 dispatch 反而差

普通 dispatch 需要：

- Python/legacy layout。
- UCCL V1 notify。
- legacy dispatch。
- V2 metadata build。
- 可能还有额外 handle/copy 工作。

它现在只有 `~66-67 GB/s (SU)`，说明 uncached dispatch 的控制和 metadata 路径还没有真正 native 化。

### 7.4 为什么 combine/reduced combine 仍然差

Reduced combine 的 Python gather 已经被移掉，性能从 `~25 GB/s` 提到 `~35 GB/s`，但仍然远低于目标。

剩下的问题是：

```text
V2 reduced-combine semantic
        |
        v
converted to V1 per-token input
        |
        v
legacy combine data path
```

也就是说，V2 的 combine/reduce 没有和 UCCL transport 原生融合。它仍然在“先转成 V1 能理解的形状，再跑 V1 combine”。

## 8. 当前代码中最值得继续改的地方

### 8.1 去掉 V1 layout 依赖

现在：

```text
V2 topk_idx
   |
   v
legacy get_dispatch_layout()
   |
   v
legacy num_tokens_per_rank / is_token_in_rank / num_tokens_per_expert
```

建议改成：

```text
V2 topk_idx
   |
   v
native V2 layout kernel
   |
   +-- per scaleout lane send spans
   +-- per local expert slot spans
   +-- token -> slot mapping
   +-- source metadata
   +-- transfer descriptors
```

也就是不要先构造 V1 layout，再翻译成 V2 metadata。应该直接生成 V2 dispatch 所需的 metadata 和 transfer plan。

### 8.2 V2 native dispatch staging

现在 dispatch 数据面是：

```text
token-major input -> V1 staging -> V1 transport -> token-major recv -> V2 expanded scatter
```

更合理的 V2 path：

```text
token-major input
   |
   v
V2 dispatch kernel
  directly writes:
    - RDMA staging chunks
    - NVL staging chunks
    - source metadata
    - expert slot metadata
    - TransferCmd list
   |
   v
CPU proxy RDMA
   |
   v
remote V2 staging / expanded slots
```

目标图：

```text
current
-------

input token
  -> V1 send buffer
  -> RDMA
  -> V1 recv buffer
  -> V2 metadata
  -> V2 expanded scatter


desired
-------

input token
  -> V2 transfer plan
  -> RDMA directly into V2 recv/expanded-compatible staging
  -> minimal epilogue
```

### 8.3 V2 native combine

当前 combine：

```text
expanded slots
  -> CUDA gather/sum to per-token reduced_x
  -> V1 combine
  -> combined output
```

更好的 combine：

```text
expanded slots
  -> V2 combine kernel emits per-destination transfer chunks
  -> CPU proxy RDMA
  -> remote reduction / accumulation protocol
  -> combined output
```

两种可能方案：

```text
Option A: reduce before network

expanded slots
  -> local per-token reduce
  -> network combine

优点：发送 bytes 少
缺点：本地 reduction 仍有额外 HBM IO


Option B: network-aware expanded combine

expanded slots
  -> group by destination source rank/token
  -> send chunks directly
  -> receiver accumulates

优点：更接近 V2 semantic
缺点：receiver reduction/order 协议更复杂
```

短期建议先做 Option A 的 native 优化，因为 correctness 风险低。长期目标应走 Option B 或类似官方 V2 的 combine/reduce protocol。

### 8.4 解除 legacy buffer/channel 限制

现在自动 SM 模型算出来可能是 64，但 legacy buffer layout 会爆：

```text
num_nvl_bytes <= int32 max
```

所以我加了：

```text
EP_UCCL_MAX_AUTO_SMS=32
```

这只是保护。长期应该：

- 重新设计 V2 staging layout。
- 不让 buffer size 随 `num_sms * num_channels * worst_tokens` 粗暴放大。
- 将 channel 数、transfer chunk 数、SM 数解耦。
- 支持 64+ SM 的 V2 native path。

## 9. 建议的下一步开发路线

### Phase 1: 消灭 Python 和 V1 wrapper 热点

已经完成：

- V2 metadata native CUDA。
- expanded dispatch payload scatter native CUDA。
- reduced combine input gather native CUDA。
- SM model 不再固定 24。

继续做：

- uncached dispatch layout 下沉到 CUDA，减少 Python/Torch layout 和 CPU sync。
- `num_recv_tokens_per_expert_list` 尽量避免 `.cpu().tolist()` 在 fast path 上发生，或者仅 correctness/debug 需要时同步。
- bench 中区分：
  - layout time
  - transport time
  - V2 metadata time
  - epilogue/scatter/reduce time

### Phase 2: V2 transfer plan

新增真正 V2 的 transfer descriptor：

```text
struct V2TransferDesc {
  int src_token;
  int dst_scaleout_rank;
  int dst_scaleup_rank;
  int dst_expert;
  int dst_slot;
  int bytes;
  int flags;
}
```

CUDA kernel 直接从 `topk_idx` 生成 per-channel/per-peer descriptor，再由 GPU 写 staging 和提交 `TransferCmd`。

目标：

```text
topk_idx
  -> V2 descriptor
  -> staging chunk
  -> TransferCmd
  -> EFA RDMA
```

不要再经过 V1 `is_token_in_rank` 和 V1 rank/expert count layout。

### Phase 3: V2 native combine/reduced-combine

将 combine 从：

```text
V2 expanded -> reduced_x -> V1 combine
```

改成：

```text
V2 expanded -> V2 combine transfer plan -> EFA RDMA -> V2 reduce epilogue
```

这应该是提高 `@ combine` 和 `+ reduced combine` 的关键。

### Phase 4: EFA-specific scheduling

利用 p5en 的 16 张 EFA NIC：

```text
GPU0 proxy
  |
  +-- NIC rail 0
  +-- NIC rail 1

GPU1 proxy
  |
  +-- NIC rail 0
  +-- NIC rail 1

...
```

需要显式考虑：

- local rank 到 EFA device 的 affinity。
- QP sharding。
- transfer chunk size。
- completion moderation。
- receiver-side ordering queue 深度。

不要指望 NCCL/Gin 自动替我们把 V2 小消息跑满。

## 10. 当前文档对应的代码索引

原始 UCCL-EP/V1 数据面：

- `uccl-ep/include/ring_buffer.cuh`
- `uccl-ep/include/d2h_queue_device.cuh`
- `uccl-ep/include/uccl_ibgda.cuh`
- `uccl-ep/src/proxy.cpp`
- `uccl-ep/src/rdma.cpp`
- `uccl-ep/src/internode.cu`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffer.py`

V2 wrapper：

- `uccl-ep/deep_ep_v2_wrapper/deep_ep/__init__.py`
- `uccl-ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`

我新增/修改的 V2 native helpers：

- `build_v2_dispatch_metadata`
- `build_v2_expanded_payload`
- `build_v2_reduced_combine_input`

benchmark：

- `uccl-ep/bench/v2_proxy_smoke.py`
- `tests/elastic/test_ep.py`

记录：

- `worklog.md`

## 11. 一句话总结

UCCL-EP 原始代码解决的是“DeepEP V1 在 EFA 上如何用 CPU proxy 发 GPUDirect RDMA”。我目前做的是把 DeepEP V2 API 和部分 V2 metadata/epilogue 接到这套 V1 数据面上，并逐步把 Python 热路径下沉到 CUDA。性能差的根本原因是：核心 dispatch/combine 传输协议仍然不是 V2-native，尤其 combine/reduced-combine 还在复用 V1 legacy combine。下一步应该围绕 V2 transfer plan 和 V2 native combine 数据面继续重写，而不是继续在 wrapper 层修修补补。
