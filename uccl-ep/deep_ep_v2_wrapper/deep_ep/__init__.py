from pathlib import Path

import torch
from uccl.ep import Config, EventHandle

_UPSTREAM_DEEP_EP = Path(__file__).resolve().parents[3] / "deep_ep"
if _UPSTREAM_DEEP_EP.exists():
    __path__.append(str(_UPSTREAM_DEEP_EP))

from .buffers.elastic import EPHandle, ElasticBuffer
from .utils.event import EventOverlap

topk_idx_t = torch.int64

__all__ = [
    "Config",
    "EventHandle",
    "EventOverlap",
    "EPHandle",
    "ElasticBuffer",
    "topk_idx_t",
]

__version__ = "2.0.0+ucclaws"
