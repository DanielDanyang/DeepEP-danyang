from pathlib import Path

_UPSTREAM_UTILS = Path(__file__).resolve().parents[4] / "deep_ep" / "utils"
if _UPSTREAM_UTILS.exists():
    __path__.append(str(_UPSTREAM_UTILS))

from .event import EventOverlap

__all__ = ["EventOverlap"]
