import torch
from nanovllm.layers.attention import store_kv_cache

"""
pytest tests/test_attention.py -v
"""

def test_store_kv_cache():
    """
    Test that store_kv_cache correctly stores k and v tensors into the cache
    at positions specified by slot_mapping.
    """
    # Setup test parameters
    total_tokens = 4
    num_kv_heads = 2
    head_dim = 8
    num_blocks = 2
    block_size = 4
    
    # Create input k and v tensors [total_tokens, num_kv_heads, head_dim]
    k = torch.randn(total_tokens, num_kv_heads, head_dim).to('cuda')
    v = torch.randn(total_tokens, num_kv_heads, head_dim).to('cuda')
    
    # Create empty cache [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim).to('cuda')
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim).to('cuda')
    
    # Create slot_mapping: map each token to a slot in the cache
    # For this test, we'll map tokens 0,1,2,3 to slots 0,1,2,3
    slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.long).to('cuda')
    
    # Store the original k and v for comparison
    k_original = k.clone()
    v_original = v.clone()
    
    # Call the function
    store_kv_cache(k, v, k_cache, v_cache, slot_mapping)
    
    # Verify that k_cache and v_cache have been updated correctly
    # Token 0 should be stored at slot 0 (block 0, position 0)
    assert torch.allclose(k_cache[0, 0], k_original[0])
    assert torch.allclose(v_cache[0, 0], v_original[0])
    
    # Token 1 should be stored at slot 1 (block 0, position 1)
    assert torch.allclose(k_cache[0, 1], k_original[1])
    assert torch.allclose(v_cache[0, 1], v_original[1])
    
    # Token 2 should be stored at slot 2 (block 0, position 2)
    assert torch.allclose(k_cache[0, 2], k_original[2])
    assert torch.allclose(v_cache[0, 2], v_original[2])
    
    # Token 3 should be stored at slot 3 (block 0, position 3)
    assert torch.allclose(k_cache[0, 3], k_original[3])
    assert torch.allclose(v_cache[0, 3], v_original[3])
    
    # Verify that other slots remain zero
    assert torch.allclose(k_cache[1, 0], torch.zeros(num_kv_heads, head_dim).to('cuda'))
    assert torch.allclose(v_cache[1, 0], torch.zeros(num_kv_heads, head_dim).to('cuda'))


def test_store_kv_cache_with_skip_slot():
    """
    Test that store_kv_cache correctly handles slot_mapping with -1 (skip).
    """
    # Setup test parameters
    total_tokens = 3
    num_kv_heads = 2
    head_dim = 8
    num_blocks = 1
    block_size = 4
    
    # Create input k and v tensors
    k = torch.randn(total_tokens, num_kv_heads, head_dim).to('cuda')
    v = torch.randn(total_tokens, num_kv_heads, head_dim).to('cuda')
    
    # Create empty cache
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim).to('cuda')
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim).to('cuda')
    
    # Create slot_mapping with -1 for token 1 (should be skipped)
    slot_mapping = torch.tensor([0, -1, 2], dtype=torch.long).to('cuda')
    
    # Store the original k and v for comparison
    k_original = k.clone()
    v_original = v.clone()
    
    # Call the function
    store_kv_cache(k, v, k_cache, v_cache, slot_mapping)
    
    # Verify that token 0 is stored at slot 0
    assert torch.allclose(k_cache[0, 0], k_original[0])
    assert torch.allclose(v_cache[0, 0], v_original[0])
    
    # Verify that token 1 is NOT stored (slot -1 means skip)
    # Slot 1 should remain zero
    assert torch.allclose(k_cache[0, 1], torch.zeros(num_kv_heads, head_dim).to('cuda'))
    assert torch.allclose(v_cache[0, 1], torch.zeros(num_kv_heads, head_dim).to('cuda'))
    
    # Verify that token 2 is stored at slot 2
    assert torch.allclose(k_cache[0, 2], k_original[2])
    assert torch.allclose(v_cache[0, 2], v_original[2])

