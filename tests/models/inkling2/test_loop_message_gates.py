# SPDX-License-Identifier: Apache-2.0
"""INK-20 CPU gate arithmetic, stream isolation and gradient tests."""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.models.inkling2.message_gates import (
    GateWeights,
    blend_cross_pass_message,
    blend_loop_streams,
)


def _weights(width=8, *, active=False):
    down = torch.ones(64, 2 * width) * (0.1 if active else 0.0)
    gain = torch.ones(width, 64) * (0.05 if active else 0.0)
    forget = torch.zeros(width, 64)
    return GateWeights(down, gain, forget)


def test_zero_initialized_gates_are_identity_in_fp32():
    destination = torch.randn(4, 8, dtype=torch.bfloat16)
    message = torch.randn_like(destination)
    result = blend_cross_pass_message(destination, message, _weights())
    assert result.dtype == torch.float32
    assert torch.equal(result, destination.float())


def test_controller_matches_bf16_projection_fp32_blend_reference():
    destination = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
    message = destination.flip(0) + 0.25
    weights = _weights(active=True)
    out = blend_cross_pass_message(destination, message, weights)
    pair = torch.stack((destination, message), dim=1)
    normalized = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
    projected = F.linear(normalized.flatten(1).bfloat16(), weights.down.bfloat16())
    expected_gain = F.linear(projected, weights.gain.bfloat16()).float()
    expected = destination + torch.tanh(expected_gain) * message
    assert torch.equal(out, expected)


def test_four_streams_do_not_share_gate_weights():
    names = ("attn", "k", "v", "mlp")
    destinations = {name: torch.full((2, 8), 0.2) for name in names}
    messages = {name: torch.ones(2, 8) for name in names}
    weights = {name: _weights(active=name == "attn") for name in names}
    out = blend_loop_streams(destinations, messages, weights)
    assert tuple(out) == names
    assert not torch.equal(out["attn"], destinations["attn"])
    for name in names[1:]:
        assert torch.equal(out[name], destinations[name])


def test_gradients_flow_through_message_and_low_rank_weights():
    d = torch.randn(3, 8, requires_grad=True)
    m = torch.randn(3, 8, requires_grad=True)
    w = GateWeights(
        (torch.randn(64, 16) / 10).requires_grad_(),
        (torch.randn(8, 64) / 10).requires_grad_(),
        (torch.randn(8, 64) / 10).requires_grad_(),
    )
    blend_cross_pass_message(d, m, w).square().mean().backward()
    for tensor in (d, m, w.down, w.gain, w.forget):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_gate_fails_closed_on_unqualified_geometry():
    d = torch.ones(2, 8)
    with pytest.raises(ValueError, match="rank-64"):
        blend_cross_pass_message(
            d, d, GateWeights(torch.zeros(2, 16), torch.zeros(8, 2), torch.zeros(8, 2))
        )
    with pytest.raises(ValueError, match="matching"):
        blend_cross_pass_message(d, d[:1], _weights())
    with pytest.raises(ValueError, match="exactly"):
        blend_loop_streams({"attn": d}, {"attn": d}, {"attn": _weights()})
