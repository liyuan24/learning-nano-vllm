"""
This is the embedding layer and lm head that can be splitted across multiple devices
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from nanovllm.utils.context import get_context


class VocabSplitEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        assert vocab_size % self.tp_size == 0, "vocab_size must be divisible by tp_size"
        self.tp_rank = dist.get_rank()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_size_per_rank = vocab_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.vocab_size_per_rank, hidden_size))
        self.weight.weights_loader = self.weights_loader

    def weights_loader(self, params: nn.Parameter, weights: torch.Tensor) -> None:
        shard_size = params.data.size(0)
        shard_start = shard_size * self.tp_rank
        # Using narrow() instead of slicing for better memory efficiency
        # narrow(dim, start, length) creates a view (no copy) vs slicing which may copy
        params.data.copy_(weights.narrow(0, shard_start, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [total_tokens,]
        output: [total_tokens, hidden_size]
        """
        # need to know whether the input_ids can be processed by this device
        if self.tp_size > 1:
            shard_start = self.tp_rank * self.vocab_size_per_rank
            shard_end = shard_start + self.vocab_size_per_rank
            shard_mask = (x >= shard_start) & (x < shard_end)
            x = shard_mask * (x - shard_start)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # zero out the tokens that are not processed by this device
            y = shard_mask.unsqueeze(1) * y
            # sum across GPUs
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y


class ParallelLMHead(VocabSplitEmbedding):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__(vocab_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [total_tokens, hidden_size]
        output: [total_tokens, vocab_size]
        """
        context = get_context()
        if context.is_prefill:
            # the last token ids for all sequences
            lask_token_inds = context.cu_seqlens_q[1:] - 1
            x = x[lask_token_inds].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            # gather logits from all devices to all_logits list on rank 0
            dist.all_gather(logits, all_logits, dst=0)
            logits = torch.cat(all_logits, dim=1) if self.tp_rank == 0 else None
        return logits
