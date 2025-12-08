from typing import Optional
from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT
