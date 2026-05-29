from __future__ import annotations

from typing import NamedTuple

from .utils.event import EventOverlap


class IntranodeDispatchHandle(NamedTuple):
    """Removed V1 handle placeholder.

    Native V2 must not rebuild the old rank-prefix-matrix transport handle.
    """


class InternodeDispatchHandle(NamedTuple):
    """Removed V1 handle placeholder.

    Native V2 must generate descriptor batches from DeepEP V2 metadata instead.
    """


class ProxyTransport:
    """Fail-fast placeholder while the native V2 EFA runtime is rebuilt.

    The previous implementation was a V1/UCCL EP transport shim. Keeping it
    callable would make tests silently exercise the wrong code path, so the
    constructor now fails until `V2EfaRuntime` owns dispatch/combine.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "uccl-ep is being rewritten as a native DeepEP V2 AWS EFA backend; "
            "the old V1 ProxyTransport has been removed."
        )

    @staticmethod
    def capture() -> EventOverlap:
        return EventOverlap()
