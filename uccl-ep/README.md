# UCCL EP for DeepEP V2 on AWS EFA

这个目录现在只维护 DeepEP V2 在 AWS EFA 上的 native proxy backend。旧的
DeepEP V1 wrapper、低延迟 API 示例和 V1 benchmark 入口已经移除，避免后续开发继续
围绕旧接口语义打补丁。

## 当前边界

- Python 入口在 `deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`。
- 内部 transport 在 `deep_ep_v2_wrapper/deep_ep/proxy_transport.py`。
- Native 扩展在 `src/uccl_ep.cc`，CUDA kernel 在 `src/internode.cu` 和
  `src/intranode.cu`。
- V2 metadata、expanded dispatch payload、reduced combine input 已经由
  `NativeElasticProxyBuffer` 直接调用 CUDA helper，并和 UCCL proxy 数据面共用同一个
  native runtime/comm stream。
- `uccl.ep.Buffer` 和 `uccl.ep.ElasticProxyBuffer` 不再作为 public API 暴露；
  旧 base 只保留为内部 `_LegacyProxyBuffer`，用于逐步替换剩余 V1 internode
  kernel 数据面。
- 剩余大块工作是把 dispatch/combine 的实际跨机 transfer plan 继续下沉到 native
  V2 kernels，最终删除 `ProxyTransport` 里的兼容 transport 调度层。

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
