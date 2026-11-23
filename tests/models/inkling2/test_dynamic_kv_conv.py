# SPDX-License-Identifier: Apache-2.0
"""INK-16 dynamic K/V convolution contract tests."""

import torch

from rl_engine.models.inkling2.dynamic_kv_conv import dynamic_kv_conv


def test_dynamic_kv_conv_matches_explicit_tap_reference():
    x = torch.arange(1 * 5 * 2 * 3, dtype=torch.float32).reshape(1, 5, 2, 3) / 10
    coefficients = torch.ones(1, 5, 2, 3, 4)
    output, state = dynamic_kv_conv(x, coefficients)
    history = torch.cat((torch.zeros(1, 2, 3, 3), x.permute(0, 2, 3, 1)), dim=-1)
    expected = torch.zeros_like(output)
    for tap in range(4):
        expected = expected + history[..., tap : tap + 5].permute(0, 3, 1, 2)
    assert torch.equal(output, expected)
    assert torch.equal(state, history[..., -3:])


def test_chunked_dynamic_weights_preserve_state_transition():
    torch.manual_seed(5)
    x = torch.randn(2, 8, 2, 4)
    coefficients = torch.randn(2, 8, 2, 4, 4)
    full, full_state = dynamic_kv_conv(x, coefficients)
    first, state = dynamic_kv_conv(x[:, :3], coefficients[:, :3])
    second, state = dynamic_kv_conv(x[:, 3:], coefficients[:, 3:], state)
    assert torch.equal(torch.cat((first, second), dim=1), full)
    assert torch.equal(state, full_state)


def test_dynamic_kv_conv_backward_is_defined():
    x = torch.randn(2, 4, 2, 3, requires_grad=True)
    coefficients = torch.randn(2, 4, 2, 3, 4, requires_grad=True)
    output, _ = dynamic_kv_conv(x, coefficients)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert coefficients.grad is not None and torch.isfinite(coefficients.grad).all()
