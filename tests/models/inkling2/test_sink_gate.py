# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.sink_gate import apply_attention_sink_gate

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_virtual_sink_math_and_backward(device):
    torch.manual_seed(13)
    scores = torch.randn(3, 4, 5, device=device, requires_grad=True)
    value = torch.randn(3, 4, 5, 7, device=device, requires_grad=True)
    sink = torch.randn(3, 4, device=device, requires_grad=True)
    gate = torch.randn(3, 4, 7, device=device, requires_grad=True)
    context = (scores.softmax(-1).unsqueeze(-1) * value).sum(-2)
    result = apply_attention_sink_gate(context, scores.logsumexp(-1), sink, gate)
    probs = torch.cat((scores, sink.unsqueeze(-1)), dim=-1).softmax(-1)[..., :-1]
    reference = (probs.unsqueeze(-1) * value).sum(-2) * gate.sigmoid()
    torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-5)
    args = scores, value, sink, gate
    grad = torch.randn_like(result)
    for a, b in zip(torch.autograd.grad(result, args, grad),
                    torch.autograd.grad(reference, args, grad), strict=True):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_context_and_token_specific_gates(device):
    context = torch.ones(2, 2, 3, device=device).bfloat16()
    lse = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    sink = torch.tensor([[0.0, 1.0], [9.0, -9.0]], device=device).bfloat16()
    gate = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]], device=device).bfloat16()
    y = apply_attention_sink_gate(context, lse, sink, gate)
    expected = (context.float() * gate.float().sigmoid().reshape_as(context))
    expected = expected * (lse - sink.float()).sigmoid().unsqueeze(-1)
    assert y.dtype == torch.float32 and torch.equal(y, expected)
    assert torch.equal(y[1:], apply_attention_sink_gate(context[1:], lse[1:], sink[1:], gate[1:]))


def test_invalid_sink_shape_and_lse_dtype():
    x = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="sink logits"):
        apply_attention_sink_gate(x, torch.ones(2, 3), torch.ones(3), x)
    with pytest.raises(TypeError, match="LSE"):
        apply_attention_sink_gate(x, torch.ones(2, 3).bfloat16(), torch.ones(2, 3), x)
