from pathlib import Path
import os
import subprocess

import torch
from uccl import ep as _ep
from uccl.ep import Config, EventHandle

_UPSTREAM_DEEP_EP = Path(__file__).resolve().parents[3] / "deep_ep"
if _UPSTREAM_DEEP_EP.exists():
    __path__.append(str(_UPSTREAM_DEEP_EP))

from .buffers.elastic import EPHandle, ElasticBuffer
from .utils.event import EventOverlap

topk_idx_t = torch.int64


def find_cuda_home() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    try:
        nvcc = subprocess.check_output(["which", "nvcc"], stderr=subprocess.DEVNULL).decode().strip()
        return str(Path(nvcc).resolve().parents[1])
    except Exception:
        return "/usr/local/cuda"


def find_nccl_root() -> str:
    from .utils.find_pkgs import find_nccl_root as _find_nccl_root

    return _find_nccl_root()


def init_deep_ep_jit(
    library_root_path: str | None = None,
    cuda_home_path: str | None = None,
    nccl_root_path: str | None = None,
) -> None:
    root = Path(library_root_path) if library_root_path is not None else _UPSTREAM_DEEP_EP
    _ep.init_deep_ep_jit(
        str(root),
        cuda_home_path or find_cuda_home(),
        nccl_root_path or find_nccl_root(),
    )

__all__ = [
    "Config",
    "EventHandle",
    "EventOverlap",
    "EPHandle",
    "ElasticBuffer",
    "find_cuda_home",
    "find_nccl_root",
    "init_deep_ep_jit",
    "topk_idx_t",
]

__version__ = "2.0.0+ucclaws"
