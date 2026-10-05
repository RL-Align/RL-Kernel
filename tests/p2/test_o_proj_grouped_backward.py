# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent VJP checks; only projection widths shrink in the CPU fixture."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.p2.o_proj import oracle
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.p2.o_proj.rope_consumer import fixture_cos_sin


@pytest.fixture
def small_projection(monkeypatch):
    # Retain the real 512-wide head and [448:512] GPT-J rotation. Full-size
    # projection dimensions are exercised separately by the CUDA tests.
    for name, value in {
        "N_O_PROJ_GROUPS": 2,
        "HEADS_PER_GROUP": 1,
        "N_Q_HEADS": 2,
        "GROUP_FLAT_DIM": 512,
        "O_LORA_RANK": 3,
        "CONCAT_Z_DIM": 6,
        "HIDDEN_SIZE": 6,
    }.items():
        monkeypatch.setattr(oracle, name, value)
    generator = torch.Generator().manual_seed(61)
    cos, sin = fixture_cos_sin(torch.tensor([1, 7, 19]))
    return SimpleNamespace(
        o=torch.randn(3, 2, 512, generator=generator) * 0.2,
        w_a=torch.randn(2, 3, 512, generator=generator) * 0.2,
        w_b=torch.randn(6, 6, generator=generator) * 0.2,
        d_y=torch.randn(3, 6, generator=generator),
        cos=cos,
        sin=sin,
    )


def test_explicit_backward_matches_independent_double_autograd(small_projection):
    case = small_projection
    op = OProjGroupedOp(backend="oracle")
    result = op.forward_fp32(case.o, case.w_a, case.w_b, case.cos, case.sin)
    actual = op.backward(case.d_y, result.saved, case.cos, case.sin)

    # This reference uses neither the shipped projection, split/concat helpers,
    # RoPE consumer, nor sequential_linear to compute its forward or VJP.
    o, w_a, w_b = [
        value.double().requires_grad_() for value in (case.o, case.w_a, case.w_b)
    ]
    pairs = o[..., 448:512].reshape(3, 2, 32, 2)
    c, s = case.cos.double()[:, None], case.sin.double()[:, None]
    rotation = torch.stack(
        (pairs[..., 0] * c + pairs[..., 1] * s,
         pairs[..., 1] * c - pairs[..., 0] * s),
        dim=-1,
    ).flatten(-2)
    rotated = torch.cat((o[..., :448], rotation), dim=-1)
    z = torch.einsum("tgk,grk->tgr", rotated, w_a).flatten(1)
    y = z @ w_b.T
    expected = torch.autograd.grad(y, (o, w_a, w_b), case.d_y.double())

    torch.testing.assert_close(result.y.double(), y, atol=2e-6, rtol=2e-5)
    for gradient, reference in zip(actual, expected, strict=True):
        assert gradient.dtype == torch.float32
        torch.testing.assert_close(gradient.double(), reference, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("token_order, expected", [([0, 1, 2], 0.0), ([0, 2, 1], 1.0)])
def test_weight_vjp_reduces_tokens_in_fp32_order(small_projection, token_order, expected):
    case = small_projection
    case.o.fill_(1)
    case.w_a.zero_()
    case.w_a[:, :, 0] = 1
    case.w_b.copy_(torch.eye(6))
    case.cos.fill_(1)
    case.sin.zero_()
    op = OProjGroupedOp(backend="oracle")
    result = op.forward_fp32(case.o, case.w_a, case.w_b, case.cos, case.sin)
    # In FP32, ((2**24 + 1) - 2**24) == 0, but reordering tokens gives 1.
    # A double, pairwise, or reversed reduction cannot satisfy both cases.
    d_y = torch.tensor([2**24, 1, -(2**24)], dtype=torch.float32)[token_order]
    d_y = d_y[:, None].expand(3, 6).contiguous()
    _, d_w_a, d_w_b = op.backward(d_y, result.saved, case.cos, case.sin)
    assert d_w_a.dtype == d_w_b.dtype == torch.float32
    assert torch.equal(d_w_a, torch.full_like(d_w_a, expected))
    assert torch.equal(d_w_b, torch.full_like(d_w_b, expected))
