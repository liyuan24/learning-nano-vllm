from typing import List, Optional
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class LinearBase(nn.Module):
    def __init__(self, input_size: int, output_size: int, tp_dim: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.weights = nn.Parameter(torch.empty(output_size, input_size))
        self.weights.weights_loader = self.weights_loader

    def weights_loader(self, params: nn.Parameter, weights: torch.Tensor) -> None:
        shard_size = params.data.size(self.tp_dim)
        shard_start = shard_size * self.tp_rank
        params.data.copy_(weights.narrow(self.tp_dim, shard_start, shard_size))


class ColumnParallelLinear(LinearBase):
    """
    Used for matrix multiplication x * W^T. The weights W^T are split across multiple devices along its column dimension.
    So each GPU will only own one shard of the weights after the split.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        input_size: the size of input of x
        output_size: the total size of output
        """
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "output_size must be divisible by tp_size"
        super().__init__(input_size, output_size // tp_size, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [total_tokens, input_size]
        output: [total_tokens, output_size_per_rank]
        """
        return F.linear(x, self.weights)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV projections are concatenated together to save the number of kernels. It is a special case of ColumnParallelLinear.
    For weights column, the first part is for Q, the second part is for K, the third part is for V.
    It enables the attention head parallelism.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
    ):
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.tp_size = dist.get_world_size()
        assert (
            total_num_heads % self.tp_size == 0
        ), "total_num_heads must be divisible by tp_size"
        assert (
            total_num_kv_heads % self.tp_size == 0
        ), "total_num_kv_heads must be divisible by tp_size"
        self.head_dim = head_dim
        self.num_heads = total_num_heads // self.tp_size
        self.num_kv_heads = total_num_kv_heads // self.tp_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        super().__init__(hidden_size, output_size)

    def weights_loader(
        self, params: nn.Parameter, weights: torch.Tensor, loaded_shard_id: str
    ) -> None:
        assert loaded_shard_id in [
            "q",
            "k",
            "v",
        ], "loaded_shard_id must be in [q, k, v]"
        if loaded_shard_id == "q":
            shard_start = 0
            shard_size = self.num_heads * self.head_dim
        elif loaded_shard_id == "k":
            shard_start = self.num_heads * self.head_dim
            shard_size = self.num_kv_heads * self.head_dim
        else:
            shard_start = (
                self.num_heads * self.head_dim + self.num_kv_heads * self.head_dim
            )
            shard_size = self.num_kv_heads * self.head_dim
        loaded_data = weights.chunk(self.tp_size, dim=0)[self.tp_rank]
        param_data = params.data.narrow(0, shard_start, shard_size)
        param_data.copy_(loaded_data)


class CombinedColumnParallelLinear(ColumnParallelLinear):
    """
    Multiple linear projects are combined together to save the number of kernels.
    """

    def __init__(self, input_size: int, output_sizes: List[int]):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes))

    def weights_loader(
        self, params: nn.Parameter, weights: torch.Tensor, loaded_shard_id: str
    ) -> None:
        output_size = self.output_sizes[loaded_shard_id]
        shard_start = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = output_size // self.tp_size
        loaded_data = weights.chunk(self.tp_size, dim=0)[self.tp_rank]
        param_data = params.data.narrow(0, shard_start, shard_size)
        param_data.copy_(loaded_data)


class RowParallelLinear(LinearBase):
    """
    Used for matrix multiplication x * W^T. The weights W^T are split across multiple devices along its row dimension.
    So each GPU will only own one shard of the weights after the split.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        input_size: the size of input of x
        output_size: the total size of output
        """
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "input_size must be divisible by tp_size"
        super().__init__(input_size // tp_size, output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [total_tokens, input_size_per_rank]
        output: [total_tokens, output_size]
        """
        y = F.linear(x, self.weights)
        # when multiple devices, we need to reduce across all devices by summing up the outputs
        if dist.get_world_size() > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
