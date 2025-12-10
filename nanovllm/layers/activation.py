import torch
import torch.nn as nn
from torch.nn import functional as F


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        kernel fusion for the silu and multiplication since they are both element-wise operations
        """
        y1, y2 = torch.chunk(x, 2, dim=-1)
        return F.silu(y1) * y2
