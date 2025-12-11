import torch
import torch.distributed as dist
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP
from nanovllm.utils.context import set_context

"""
pytest tests/test_qwen3.py -v
"""


def test_qwen3_attention():
    """
    Test that Qwen3Attention correctly processes input tensors and produces
    the expected output shape.
    """
    # Test parameters
    hidden_size = 1024
    total_num_heads = 32
    total_num_kv_heads = 32
    max_position_embeddings = 10
    num_tokens = 4
    rank = 0
    world_size = 1

    # Initialize distributed process group
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:2333",
            rank=rank,
            world_size=world_size,
        )
    except RuntimeError:
        # Process group might already be initialized
        pass

    # Set default dtype and device
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    # Setup context for attention
    block_size = 256
    num_blocks = 10
    num_hidden_layers = 28
    kv_cache = torch.zeros(
        2,
        num_hidden_layers,
        num_blocks,
        block_size,
        total_num_kv_heads,
        hidden_size // total_num_kv_heads,
    )
    is_prefill = True
    cu_seqlens_q = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    max_seqlen_q = 1
    max_seqlen_k = 1
    block_tables = torch.tensor(
        [
            [0],
            [1],
            [2],
            [3],
        ],
        dtype=torch.int32,
    )
    context_lens = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    slot_mapping = torch.tensor(
        [0, 1 * block_size + 1, 2 * block_size + 2, 3 * block_size + 3],
        dtype=torch.int32,
    )
    set_context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        block_tables,
        context_lens,
        slot_mapping,
    )

    # Create Qwen3Attention instance
    qwen3_attention = Qwen3Attention(
        hidden_size, total_num_heads, total_num_kv_heads, max_position_embeddings
    )
    qwen3_attention.attn.k_cache = kv_cache[0][0]  # each layer has its own kv cache
    qwen3_attention.attn.v_cache = kv_cache[1][0]

    # Create input tensors
    x = torch.randn(num_tokens, hidden_size).to(torch.bfloat16)
    position_ids = torch.arange(num_tokens).to(torch.long)

    # Run forward pass
    with torch.no_grad():
        o = qwen3_attention(x, position_ids)

    # Verify output shape
    assert o.shape == (
        num_tokens,
        hidden_size,
    ), f"Expected output shape ({num_tokens}, {hidden_size}), got {o.shape}"

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


def test_qwen3_mlp():
    """
    Test that Qwen3MLP correctly processes input tensors and produces
    the expected output shape.
    """
    # Test parameters
    num_tokens = 4
    hidden_size = 1024
    intermediate_size = hidden_size * 4
    rank = 0
    world_size = 1

    # Initialize distributed process group
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:2333",
            rank=rank,
            world_size=world_size,
        )
    except RuntimeError:
        # Process group might already be initialized
        pass

    # Set default dtype and device
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    # Create Qwen3Attention instance
    qwen3_mlp = Qwen3MLP(hidden_size, intermediate_size)

    # Create input tensors
    x = torch.randn(num_tokens, hidden_size).to(torch.bfloat16)

    # Run forward pass
    with torch.no_grad():
        o = qwen3_mlp(x)

    # Verify output shape
    assert o.shape == (
        num_tokens,
        hidden_size,
    ), f"Expected output shape ({num_tokens}, {hidden_size}), got {o.shape}"

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
