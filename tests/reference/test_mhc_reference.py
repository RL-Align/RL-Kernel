# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The DeepSeek-V4 mHC pre, post and head references, on CPU.

Checks the formulas that the device backends are compared against: the
Sinkhorn rounds, the [source, destination] direction of comb_mix, that mHC
post adds the old residual exactly once, the output dtypes, shape handling,
and that the registry hands out the references off ROCm.
"""

import pytest
import torch

from rl_engine.reference.mhc import NativeMHCHeadOp, NativeMHCPostOp, NativeMHCPreOp, sinkhorn_dsv4
from rl_engine.runtime.registry import kernel_registry
from rl_engine.validation.common.batch_invariance import bits_equal

_N, _C = 4, 64  # DeepSeek-V4 uses n = 4; a small hidden size keeps CPU tests fast
_MIX = 2 * _N + _N * _N


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _residual(m, seed):
    return torch.randn(m, _N, _C, generator=_gen(seed)).to(torch.bfloat16)


def _pre_params(seed=0):
    gen = _gen(seed)
    fn = torch.randn(_MIX, _N * _C, generator=gen) / (_N * _C) ** 0.5
    return fn, 0.5 + torch.rand(3, generator=gen), 0.1 * torch.randn(_MIX, generator=gen)


def _head_params(seed=0):
    gen = _gen(seed)
    fn = torch.randn(_N, _N * _C, generator=gen) / (_N * _C) ** 0.5
    return fn, 0.5 + torch.rand(1, generator=gen), 0.1 * torch.randn(_N, generator=gen)


def _post_inputs(m, seed):
    gen = _gen(seed)
    y = torch.randn(m, _C, generator=gen).to(torch.bfloat16)
    residual = torch.randn(m, _N, _C, generator=gen).to(torch.bfloat16)
    post_mix = 2.0 * torch.sigmoid(torch.randn(m, _N, generator=gen))
    comb_mix = torch.softmax(torch.randn(m, _N, _N, generator=gen), dim=-1)
    return y, residual, post_mix, comb_mix


# ---------------------------------------------------------------------------
# Sinkhorn
# ---------------------------------------------------------------------------


def test_sinkhorn_without_rounds_returns_the_logits():
    logits = torch.randn(3, _N, _N, generator=_gen(1))
    assert sinkhorn_dsv4(logits, 0, 1e-6) is logits


def test_sinkhorn_converges_to_a_doubly_stochastic_matrix():
    comb = sinkhorn_dsv4(torch.randn(3, _N, _N, generator=_gen(2)), 20, 1e-6)
    ones = torch.ones(3, _N)
    torch.testing.assert_close(comb.sum(-1), ones, atol=1e-4, rtol=0)
    torch.testing.assert_close(comb.sum(-2), ones, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# mHC pre
# ---------------------------------------------------------------------------


def test_pre_outputs_and_dtypes():
    fn, scale, base = _pre_params()
    residual = _residual(5, 3)
    post, comb, layer_input = NativeMHCPreOp()(residual, fn, scale, base)
    assert (post.shape, comb.shape, layer_input.shape) == ((5, _N), (5, _N, _N), (5, _C))
    assert (post.dtype, comb.dtype, layer_input.dtype) == (
        torch.float32,
        torch.float32,
        torch.bfloat16,
    )
    assert NativeMHCPreOp().forward_fp32(residual, fn, scale, base)[2].dtype == torch.float32


def test_pre_layer_input_is_the_pre_weighted_stream_sum():
    fn, scale, base = _pre_params()
    fn[:_N] = 0.0  # the pre logits are then exactly base[:n]
    residual = _residual(5, 4)
    _, _, layer_input = NativeMHCPreOp().forward_fp32(residual, fn, scale, base)
    pre = torch.sigmoid(base[:_N]) + 1e-6
    expected = (pre[:, None] * residual.float()).sum(-2)
    torch.testing.assert_close(layer_input, expected, atol=1e-6, rtol=1e-6)


def test_pre_rejects_bad_shapes():
    fn, scale, base = _pre_params()
    with pytest.raises(ValueError, match="fn"):
        NativeMHCPreOp()(_residual(2, 5), fn[:, :10], scale, base)
    with pytest.raises(ValueError, match="scale and base"):
        NativeMHCPreOp()(_residual(2, 5), fn, scale[:2], base)


# ---------------------------------------------------------------------------
# mHC post
# ---------------------------------------------------------------------------


# With comb_mix = identity and post_mix = 0 the output is exactly the old
# residual; adding the residual a second time would double it.
def test_post_adds_the_residual_once():
    y, residual, post_mix, _ = _post_inputs(8, 6)
    identity = torch.eye(_N).expand(8, _N, _N)
    out = NativeMHCPostOp()(y, residual, torch.zeros_like(post_mix), identity)
    assert bits_equal(out, residual)


# comb_mix[h, j] moves source stream h into destination stream j (the transpose
# of H_res in the mHC paper).
def test_post_comb_direction_is_source_to_destination():
    y, residual, post_mix, _ = _post_inputs(2, 7)
    comb = torch.zeros(2, _N, _N)
    comb[:, 0, 1] = 1.0
    out = NativeMHCPostOp().forward_fp32(y, residual, torch.zeros_like(post_mix), comb)
    assert torch.equal(out[:, 1], residual[:, 0].float())
    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))


def test_post_accepts_leading_dims_and_post_mix_with_trailing_one():
    y, residual, post_mix, comb_mix = _post_inputs(6, 8)
    op = NativeMHCPostOp()
    flat = op(y, residual, post_mix, comb_mix)
    shaped = op(
        y.view(2, 3, _C),
        residual.view(2, 3, _N, _C),
        post_mix.view(2, 3, _N, 1),
        comb_mix.view(2, 3, _N, _N),
    )
    assert flat.dtype == torch.bfloat16
    assert bits_equal(shaped.view(6, _N, _C), flat)


def test_post_rejects_bad_shapes():
    y, residual, post_mix, comb_mix = _post_inputs(2, 9)
    with pytest.raises(ValueError, match="comb_mix"):
        NativeMHCPostOp()(y, residual, post_mix, comb_mix[:, :2])


# ---------------------------------------------------------------------------
# mHC head
# ---------------------------------------------------------------------------


def test_head_accepts_leading_dims():
    fn, scale, base = _head_params()
    residual = _residual(6, 10)
    op = NativeMHCHeadOp()
    flat = op(residual, fn, scale, base)
    shaped = op(residual.view(2, 3, _N, _C), fn, scale, base)
    assert flat.dtype == torch.bfloat16
    assert bits_equal(shaped.view(6, _C), flat)


def test_head_rejects_bad_shapes():
    fn, scale, base = _head_params()
    with pytest.raises(ValueError, match="fn"):
        NativeMHCHeadOp()(_residual(2, 11), fn[:, :100], scale, base)


@pytest.mark.parametrize(
    ("op_type", "name"),
    [
        ("mhc_pre", "NativeMHCPreOp"),
        ("mhc_post", "NativeMHCPostOp"),
        ("mhc_head", "NativeMHCHeadOp"),
    ],
)
def test_registry_selects_the_reference_on_cpu(op_type, name):
    assert type(kernel_registry.get_op(op_type, device="cpu")).__name__ == name
