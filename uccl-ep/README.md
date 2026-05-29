# UCCL EP for DeepEP V2 on AWS EFA

这个目录的当前实现已经被标记为过渡实现：它名义上服务 DeepEP V2 on AWS EFA，但源码
主体仍然继承了 DeepEP V1/UCCL EP 的 static CUDA kernel、staged buffer 和 prefix-matrix
协议。后续开发不再在这条路径上继续补丁式优化，而是按
`NATIVE_V2_REWRITE_PLAN.md` 删除 V1 数据面并重写为 DeepEP V2 JIT backend。

## 当前边界

- Python 入口在 `deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`。
- 内部 transport 在 `deep_ep_v2_wrapper/deep_ep/proxy_transport.py`。
- Native 扩展在 `src/uccl_ep.cc`，但其中仍暴露 V1 风格的
  `internode_prepare/dispatch/combine` 和 `intranode_prepare/dispatch/combine`。
- CUDA kernel 仍在 `src/internode.cu`、`src/intranode.cu`、`src/layout.cu`，这些不是
  DeepEP V2 的 JIT `.cuh` kernel。
- `ProxyTransport` 仍然包含 V1 handle 字段，例如 `rank_prefix_matrix`、
  `rdma_channel_prefix_matrix`、`gbl_channel_prefix_matrix`。
- 下一步不是继续调这些 kernel，而是删除旧数据面，新增 `V2EfaRuntime` 和 V2 JIT
  descriptor/proxy backend。

## 构建

在 AWS p5en 机器上使用隔离 venv：

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/uccl-ep
source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export USE_DMABUF=1
export MAX_JOBS=16
python setup.py install
```

## DeepEP V2 Smoke

先确认没有其他人占用 GPU，再运行最小 V2 smoke：

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang
source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate
export PYTHONPATH=$PWD/uccl-ep/deep_ep_v2_wrapper:$PYTHONPATH
python -c "import deep_ep; print(deep_ep.ElasticBuffer)"
```

双机 benchmark 继续使用仓库里的 DeepEP V2 test/bench 脚本，并显式设置 AWS OFI
master、EFA provider 和 rails：

```bash
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
unset EP_DISABLE_GIN
unset OFI_NCCL_GIN_GDAKI
export NCCL_NET_PLUGIN=ofi
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=4
export NCCL_SOCKET_IFNAME=enp71s0
```
