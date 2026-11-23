# SPDX-License-Identifier: Apache-2.0
"""INK-26 expert gate/up contract tests."""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.models.inkling2.expert_mlp import expert_gate_up, softcap


def test_softcap_matches_declared_formula():
    value = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0])
    expected = 4.0 * torch.tanh(value / 4.0)
    assert torch.equal(softcap(value, 4.0), expected)


def test_gate_up_matches_fp32_softcapped_swiglu_reference():
    hidden = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10
    gate_weight = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20
    up_weight = torch.flip(gate_weight, dims=[1])
    output = expert_gate_up(hidden, gate_weight, up_weight)

    gate = F.linear(hidden.float(), gate_weight)
    up = F.linear(hidden.float(), up_weight)
    expected = F.silu(4.0 * torch.tanh(gate / 4.0)) * (25.0 * torch.tanh(up / 25.0))
    assert torch.equal(output, expected)


def test_gate_up_preserves_bfloat16_output_and_backward():
    hidden = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    output = expert_gate_up(hidden, gate, up)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    for tensor in (hidden, gate, up):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad.float()).all()


def test_gate_up_rejects_mismatched_intermediate_size():
    with pytest.raises(ValueError, match="share shape"):
        expert_gate_up(torch.randn(2, 4), torch.randn(3, 4), torch.randn(5, 4))
