from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import (
    CombinedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.norm import RMSNorm
from nanovllm.layers.rope import RoPE
from nanovllm.utils.context import set_context


class Qwen3Attention(nn.Module):
    """
    Per device attention layer for Qwen3.
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert (
            hidden_size % total_num_heads == 0
        ), "hidden_size must be divisible by total_num_heads"
        self.head_dim = hidden_size // total_num_heads
        tp_size = dist.get_world_size()
        self.hidden_size = hidden_size
        assert (
            total_num_heads % tp_size == 0
        ), "total_num_heads must be divisible by tp_size"
        self.num_heads = total_num_heads // tp_size
        assert (
            total_num_kv_heads % tp_size == 0
        ), "total_num_kv_heads must be divisible by tp_size"
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.rope = RoPE(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
        )
        self.o_proj = RowParallelLinear(input_size=hidden_size, output_size=hidden_size)
        self.attn = Attention(scale=self.head_dim**-0.5)
        self.q_norm = RMSNorm(norm_size=self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(norm_size=self.head_dim, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens, hidden_size]
            position_ids: [total_tokens]
        Returns:
            output: [total_tokens, hidden_size]
        """
        x = self.qkv_proj(x)
        # q shape: [total_tokens, num_heads * head_dim]
        # k shape: [total_tokens, num_kv_heads * head_dim]
        # v shape: [total_tokens, num_kv_heads * head_dim]
        q, k, v = torch.split(
            x,
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # apply QK norm
        q, k = self.q_norm(q), self.k_norm(k)
        # apply rotary positional embeding
        q, k = self.rope(q, k, position_ids)
        # shape: [total_tokens, num_heads, head_dim]
        o = self.attn(q, k, v)
        # shape: [total_tokens, num_heads * head_dim]
        o = o.flatten(1, -1)
        # apply output projection, shape: [total_tokens, hidden_size]
        return self.o_proj(o)


class Qwen3MLP(nn.Module):
    """
    Per device MLP layer for Qwen3.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_gate_proj = CombinedColumnParallelLinear(
            input_size=hidden_size, output_sizes=[intermediate_size, intermediate_size]
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size, output_size=hidden_size
        )
        self.activation_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens, hidden_size]
        Returns:
            output: [total_tokens, hidden_size]
        """
        # shape: [total_tokens, 2 * intermediate_size / tp_size]
        x = self.up_gate_proj(x)
        # shape: [total_tokens, intermediate_size / tp_size]
        x = self.activation_fn(x)
        # shape: [total_tokens, hidden_size]
        x = self.down_proj(x)
        return x


class Qwen3Block(nn.Module):
    """
    Per device block for Qwen3, which consists of an attention layer and an MLP layer
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        max_position_embeddings: int,
        intermediate_size: int,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention = Qwen3Attention(
            hidden_size,
            total_num_heads,
            total_num_kv_heads,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)
        self.attn_norm = RMSNorm(norm_size=hidden_size, eps=rms_norm_eps)
        self.mlp_norm = RMSNorm(norm_size=hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        we can simply do
        x = x + self.self_attn(positions, norm(hiddent_states))
        x = x + self.mlp(norm(x))

        But we can see that add is one kernel and norm is another kernel. Here we fuse the add and norm into one kernel.
        
        Arguments:
            x: [total_tokens, hidden_size]
            position_ids: [total_tokens]
            residual: [total_tokens, hidden_size]
        Returns:
            output: [total_tokens, hidden_size]
        """
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        # shape: [total_tokens, hidden_size]
        x = self.attention(x, position_ids)
        x, residual = self.mlp_norm(x, residual)
        x = self.mlp(x)
        return x, residual
