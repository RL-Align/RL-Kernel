# SPDX-License-Identifier: Apache-2.0
"""INK-17 attention O-projection contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.attention_residual import attention_o_residual


def test_o_projection_uses_fixed_hidden_reduction_and_residual_order():
    attention = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 10
    residual = torch.full_like(attention, 0.25)
    weight = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) / 20
    bias = torch.arange(4, dtype=torch.float32) / 10
    output = attention_o_residual(attention, residual, weight, bias)
    expected = attention.reshape(-1, 4) @ weight.transpose(0, 1) + bias + residual.reshape(-1, 4)
    torch.testing.assert_close(output, expected.reshape_as(output))


def test_o_projection_preserves_bfloat16_and_backward():
    attention = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(attention)
    weight = torch.randn(8, 8, dtype=torch.bfloat16, requires_grad=True)
    output = attention_o_residual(attention, residual, weight)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert attention.grad is not None and torch.isfinite(attention.grad.float()).all()
    assert weight.grad is not None and torch.isfinite(weight.grad.float()).all()


def test_o_projection_rejects_mismatched_residual():
    with pytest.raises(ValueError, match="same"):
        attention_o_residual(torch.randn(2, 4), torch.randn(2, 5), torch.randn(4, 4))
