from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from uccl.ep import EventHandle


@dataclass
class EventOverlap:
    event: Optional[EventHandle]
    extra_tensors: Optional[Tuple[object, ...]] = None

    def current_stream_wait(self) -> None:
        if self.event is None:
            return
        if hasattr(self.event, "current_stream_wait"):
            self.event.current_stream_wait()
            return
        torch.cuda.current_stream().wait_event(self.event)

