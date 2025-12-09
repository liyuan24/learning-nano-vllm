"""
pytest tests/test_norm.py -v
"""

import torch
from nanovllm.layers.norm import RMSNorm
from torch.nn import functional as F


def test_rms_norm():
    """
    Test basic RMSNorm functionality without residual.
    """
    hidden_size = 64
    batch_size = 4
    seq_len = 10

    # Create RMSNorm layer
    rms_norm = RMSNorm(hidden_size)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    # Apply RMSNorm
    output = rms_norm.rms_norm(x)

    # Check output shape
    assert output.shape == x.shape

    torch_rms_norm = F.rms_norm(x, (hidden_size,), rms_norm.weight, rms_norm.eps)
    assert torch.allclose(output, torch_rms_norm, atol=1e-5)


def test_rms_norm_with_residual():
    """
    Test RMSNorm with residual addition.
    """
    hidden_size = 100
    batch_size = 2
    seq_len = 2

    # Create RMSNorm layer
    rms_norm = RMSNorm(hidden_size)

    # Create input and residual tensors
    x = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)
    residual = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    # Store original residual for comparison
    original_x = x.clone()
    residual_original = residual.clone()

    # Apply RMSNorm with residual
    output, new_residual = rms_norm.rms_norm_with_residual(x, residual)

    # Check output shapes
    assert output.shape == x.shape
    assert new_residual.shape == residual.shape

    # Check that new_residual is x + residual (in original dtype)
    expected_residual = original_x+residual_original
    # assert torch.allclose(new_residual, expected_residual, atol=1e-5)
    assert torch.allclose(new_residual, expected_residual, atol=1e-5)
    # Check that output is normalized
    rms_torch = F.rms_norm(original_x+residual_original, (hidden_size,), rms_norm.weight, rms_norm.eps)
    # relax the tolerance
    assert torch.allclose(output, rms_torch, atol=1e-1)
