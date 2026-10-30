# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Self-consistency tests for the P1 FP32 oracle (P1-D1..P1-D6; issue #2).

Every hand-written backward is cross-checked against a plain-torch autograd
graph built from the same frozen formulas. The autograd graph uses library
reductions on purpose: it validates the *math*, while the byte-level golden
manifest validates the *arithmetic order*.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from rl_engine.mhc import fixtures, oracle
from rl_engine.mhc.contract import LayerContract, ResidualBatch
from rl_engine.mhc.reduction import HC_MULT

EPS = 1e-6


def _close(got: torch.Tensor, want: torch.Tensor, rel: float = 2e-3) -> bool:
    got, want = got.float(), want.float()
    return bool((got - want).abs().max() <= rel * want.abs().max() + 1e-5)


# A BF16 output carries ~1 ulp = 2^-8 of relative error against an FP32
# reference, so boundaries that round to BF16 get the looser bound.
BF16_ULP = 1e-2


# --- P1-D1: hc_split_sinkhorn ---------------------------------------------


def _sinkhorn_autograd(h: torch.Tensor, iters: int = 20):
    """The same frozen schedule, written with library ops, for autograd."""
    pre = torch.sigmoid(h[:, 0:4]) + EPS
    post = 2.0 * torch.sigmoid(h[:, 4:8])
    logits = h[:, 8:24].reshape(-1, 4, 4)
    m = torch.softmax(logits, dim=2) + EPS
    m = m / (m.sum(dim=1, keepdim=True) + EPS)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=2, keepdim=True) + EPS)
        m = m / (m.sum(dim=1, keepdim=True) + EPS)
    return pre, post, m


def test_sinkhorn_forward_matches_the_literal_schedule() -> None:
    g = torch.Generator().manual_seed(11)
    h = torch.randn(5, 24, generator=g)
    pre, post, c, _ = oracle.hc_split_sinkhorn_fwd(h, LayerContract(hidden=128))
    pre_ref, post_ref, c_ref = _sinkhorn_autograd(h)
    assert _close(pre, pre_ref) and _close(post, post_ref) and _close(c, c_ref)


def test_sinkhorn_backward_matches_autograd() -> None:
    g = torch.Generator().manual_seed(12)
    h = torch.randn(5, 24, generator=g).requires_grad_(True)
    dpre = torch.randn(5, 4, generator=g)
    dpost = torch.randn(5, 4, generator=g)
    dc = torch.randn(5, 4, 4, generator=g)

    pre_r, post_r, c_r = _sinkhorn_autograd(h)
    (pre_r * dpre).sum().add((post_r * dpost).sum()).add((c_r * dc).sum()).backward()

    _, _, _, saved = oracle.hc_split_sinkhorn_fwd(h.detach(), LayerContract(hidden=128))
    dh = oracle.hc_split_sinkhorn_bwd(dpre, dpost, dc, saved)
    assert _close(dh, h.grad)


def test_sinkhorn_runs_exactly_20_column_normalizations() -> None:
    _, _, _, saved = oracle.hc_split_sinkhorn_fwd(torch.zeros(1, 24), LayerContract(hidden=128))
    kinds = [k for k, _, _ in saved["steps"]]
    assert kinds.count("col") == 20 and kinds.count("row") == 19
    assert kinds[0] == "col", "the schedule starts with a column normalize"


def test_sinkhorn_row_and_column_order_changes_bytes() -> None:
    """Row-first and column-first both converge, so they agree to ~1e-7 -- but
    they are not byte-equal. Under a strict-bytes contract that is a failure,
    which is exactly why issue #2 forbids swapping the order."""
    g = torch.Generator().manual_seed(13)
    h = torch.randn(3, 24, generator=g)
    _, _, c, _ = oracle.hc_split_sinkhorn_fwd(h, LayerContract(hidden=128))
    m = torch.softmax(h[:, 8:24].reshape(-1, 4, 4), dim=2) + EPS
    m = m / (m.sum(dim=2, keepdim=True) + EPS)  # row first: the wrong order
    for _ in range(19):
        m = m / (m.sum(dim=1, keepdim=True) + EPS)
        m = m / (m.sum(dim=2, keepdim=True) + EPS)
    assert _close(c, m, rel=1e-3), "both orders converge, so a tolerance test would pass"
    assert not torch.equal(c, m), "but the bytes differ -- the order is load-bearing"


def test_sinkhorn_eps_guard_is_not_a_clamp() -> None:
    """``sum + eps`` and ``clamp(sum, min=eps)`` differ; issue #2 pins ``sum + eps``."""
    h = torch.zeros(1, 24)
    _, _, c, _ = oracle.hc_split_sinkhorn_fwd(h, LayerContract(hidden=128))
    m = torch.softmax(h[:, 8:24].reshape(-1, 4, 4), dim=2) + EPS
    m = m / torch.clamp(m.sum(dim=1, keepdim=True), min=EPS)
    for _ in range(19):
        m = m / torch.clamp(m.sum(dim=2, keepdim=True), min=EPS)
        m = m / torch.clamp(m.sum(dim=1, keepdim=True), min=EPS)
    assert not torch.equal(c, m)


def test_pre_and_post_use_their_own_activations() -> None:
    h = torch.zeros(1, 24)
    pre, post, _, _ = oracle.hc_split_sinkhorn_fwd(h, LayerContract(hidden=128))
    assert _close(pre, torch.full((1, 4), 0.5 + EPS))
    assert _close(post, torch.full((1, 4), 1.0))


# --- P1-D2: fp32_gemm_rms -------------------------------------------------


def test_gemm_rms_forward_and_backward_match_autograd() -> None:
    g = torch.Generator().manual_seed(21)
    k, n = 64, 24
    x = torch.randn(6, k, generator=g).requires_grad_(True)
    w = torch.randn(n, k, generator=g).requires_grad_(True)
    dp = torch.randn(6, n, generator=g)
    dr = torch.randn(6, generator=g)

    p_ref = x @ w.t()
    r_ref = 1.0 / (torch.sqrt((x * x).sum(dim=1)) / math.sqrt(k) + EPS)
    (p_ref * dp).sum().add((r_ref * dr).sum()).backward()

    p, r, saved = oracle.fp32_gemm_rms_fwd(x.detach(), w.detach(), EPS)
    dx, dw = oracle.fp32_gemm_rms_bwd(dp, dr, x.detach(), w.detach(), saved)
    assert _close(p, p_ref) and _close(r, r_ref)
    assert _close(dx, x.grad) and _close(dw, w.grad)


def test_controller_rms_is_not_the_rmsnorm_formula() -> None:
    """``1/(sqrt(mean)+eps)`` vs ``rsqrt(mean+eps)`` -- issue #2 forbids mixing them."""
    x = torch.full((1, 64), 1e-3)
    _, r, _ = oracle.fp32_gemm_rms_fwd(x, torch.zeros(24, 64), EPS)
    wrong = torch.rsqrt((x * x).mean(dim=1) + EPS)
    assert not torch.equal(r, wrong)


# --- P1-D3: mhc_post ------------------------------------------------------


def test_mhc_post_forward_and_backward_match_autograd() -> None:
    g = torch.Generator().manual_seed(31)
    t, d = 5, 32
    r_old = torch.randn(t, 4, d, generator=g).to(torch.bfloat16)
    y = torch.randn(t, d, generator=g).to(torch.bfloat16)
    c = torch.randn(t, 4, 4, generator=g)
    post = torch.rand(t, 4, generator=g) * 2.0
    dr_new = torch.randn(t, 4, d, generator=g)

    r32 = r_old.float().requires_grad_(True)
    y32 = y.float().requires_grad_(True)
    c_a = c.clone().requires_grad_(True)
    post_a = post.clone().requires_grad_(True)
    ref = torch.einsum("tij,tid->tjd", c_a, r32) + post_a.unsqueeze(2) * y32.unsqueeze(1)
    (ref * dr_new).sum().backward()

    out = oracle.mhc_post_fwd(r_old, y, c, post)
    assert _close(out, ref, rel=BF16_ULP)
    d_r_old, dy, dc, dpost = oracle.mhc_post_bwd(dr_new, r_old, y, c, post)
    assert _close(d_r_old, r32.grad)
    assert _close(dy, y32.grad)
    assert _close(dc, c_a.grad)
    assert _close(dpost, post_a.grad)


def test_mhc_post_downcasts_exactly_once() -> None:
    out = oracle.mhc_post_fwd(
        torch.ones(1, 4, 8, dtype=torch.bfloat16),
        torch.ones(1, 8, dtype=torch.bfloat16),
        torch.full((1, 4, 4), 0.1),
        torch.full((1, 4), 0.3),
    )
    assert out.dtype == torch.bfloat16
    # 4*0.1 + 0.3 computed in FP32 then rounded once != rounding each partial.
    assert out[0, 0, 0].float() == torch.tensor(0.7).to(torch.bfloat16).float()


# --- P1-D4: h_aggregate ---------------------------------------------------


def test_h_aggregate_forward_and_backward_match_autograd() -> None:
    g = torch.Generator().manual_seed(41)
    pre = torch.rand(4, 4, generator=g) + 0.1
    r_old = torch.randn(4, 4, 16, generator=g).to(torch.bfloat16)
    dh = torch.randn(4, 16, generator=g)

    pre_a = pre.clone().requires_grad_(True)
    r_a = r_old.float().requires_grad_(True)
    ref = (pre_a.unsqueeze(2) * r_a).sum(dim=1)
    (ref * dh).sum().backward()

    assert _close(oracle.h_aggregate_fwd(pre, r_old), ref, rel=BF16_ULP)
    dr, dpre = oracle.h_aggregate_bwd(dh, pre, r_old)
    assert _close(dr, r_a.grad) and _close(dpre, pre_a.grad)


# --- P1-D5: rmsnorm_residual ----------------------------------------------


def test_rmsnorm_residual_forward_and_backward_match_autograd() -> None:
    g = torch.Generator().manual_seed(51)
    t, d = 6, 64
    x = torch.randn(t, d, generator=g).to(torch.bfloat16)
    gamma = (1.0 + torch.randn(d, generator=g) * 0.1).to(torch.bfloat16)
    dy = torch.randn(t, d, generator=g).to(torch.bfloat16)
    d_res = torch.randn(t, d, generator=g).to(torch.bfloat16)

    x_a = x.float().requires_grad_(True)
    gamma_a = gamma.float().requires_grad_(True)
    r_ref = torch.rsqrt((x_a * x_a).mean(dim=1) + EPS)
    y_ref = x_a * r_ref.unsqueeze(1) * gamma_a
    ((y_ref * dy.float()).sum() + (x_a * d_res.float()).sum()).backward()

    y, residual, saved = oracle.rmsnorm_residual_fwd(x, gamma, EPS)
    assert _close(y, y_ref, rel=BF16_ULP)
    assert torch.equal(residual, x), "the residual fork keeps the original BF16 bytes"
    dx, dgamma = oracle.rmsnorm_residual_bwd(dy, d_res, x, gamma, saved)
    assert _close(dx, x_a.grad)
    assert _close(dgamma, gamma_a.grad)


def test_rmsnorm_is_not_an_add_then_norm() -> None:
    """The fork is of the *unnormalized* input; it is not ``x += residual``."""
    x = torch.full((1, 32), 2.0, dtype=torch.bfloat16)
    gamma = torch.ones(32, dtype=torch.bfloat16)
    _, residual, _ = oracle.rmsnorm_residual_fwd(x, gamma, EPS)
    assert torch.equal(residual, x)


def test_rmsnorm_zero_row_is_finite_via_eps() -> None:
    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    gamma = torch.ones(32, dtype=torch.bfloat16)
    y, _, saved = oracle.rmsnorm_residual_fwd(x, gamma, EPS)
    assert torch.isfinite(y).all() and torch.isfinite(saved["r"]).all()


def test_rmsnorm_uses_rsqrt_not_one_over_sqrt() -> None:
    """``rsqrt(m+eps)`` and ``1/sqrt(m+eps)`` differ in the last bit on real rows.

    The oracle must take the ``rsqrt`` branch exactly, and the two forms must
    genuinely disagree -- otherwise this test would be vacuous.
    """
    from rl_engine.mhc.reduction import fixed_sumsq

    g = torch.Generator().manual_seed(52)
    x = (torch.randn(256, 128, generator=g) * 3.0).to(torch.bfloat16)
    _, _, saved = oracle.rmsnorm_residual_fwd(x, torch.ones(128, dtype=torch.bfloat16), EPS)
    m = fixed_sumsq(x.float(), dim=1) / 128.0
    assert torch.equal(saved["r"], torch.rsqrt(m + EPS))
    assert not torch.equal(
        torch.rsqrt(m + EPS), 1.0 / torch.sqrt(m + EPS)
    ), "the two forms agreed on every row; pick a sharper fixture"


# --- P1-D6: fixed-K GEMM reference ----------------------------------------


def test_fixed_k_gemm_backward_matches_autograd() -> None:
    g = torch.Generator().manual_seed(61)
    x = torch.randn(5, 40, generator=g).requires_grad_(True)
    w = torch.randn(7, 40, generator=g).requires_grad_(True)
    dy = torch.randn(5, 7, generator=g)
    (x @ w.t() * dy).sum().backward()
    dx, dw = oracle.fixed_k_gemm_bwd(dy, x.detach(), w.detach())
    assert _close(dx, x.grad) and _close(dw, w.grad)


# --- block composition ----------------------------------------------------


def _block_autograd(batch: ResidualBatch, grads):
    """Autograd model of the whole P1 block, from the same frozen formulas."""
    d = batch.hidden
    r = batch.r_old.float().requires_grad_(True)
    y = batch.y_sublayer.float().requires_grad_(True)
    w = batch.controller.weight.clone().requires_grad_(True)
    a_pre = batch.controller.alpha_pre.clone().requires_grad_(True)
    a_post = batch.controller.alpha_post.clone().requires_grad_(True)
    a_res = batch.controller.alpha_res.clone().requires_grad_(True)
    bias = batch.controller.bias.clone().requires_grad_(True)
    gamma = batch.norm.gamma.float().requires_grad_(True)

    x_flat = r.reshape(batch.tokens, batch.contract.flat_k)
    p = x_flat @ w.t()
    k = batch.contract.flat_k
    scale = 1.0 / (torch.sqrt((x_flat * x_flat).sum(dim=1)) / math.sqrt(k) + EPS)
    alpha = torch.cat([a_pre.expand(4), a_post.expand(4), a_res.expand(16)], dim=-1)
    h = (scale.unsqueeze(1) * p) * alpha + bias
    pre, post, c = _sinkhorn_autograd(h)

    hidden = (pre.unsqueeze(2) * r).sum(dim=1).to(torch.bfloat16).float()
    rstd = torch.rsqrt((hidden * hidden).mean(dim=1) + EPS)
    normalized = hidden * rstd.unsqueeze(1) * gamma
    r_new = torch.einsum("tij,tid->tjd", c, r) + post.unsqueeze(2) * y.unsqueeze(1)

    loss = (
        (r_new * grads.d_r_new.float()).sum()
        + (normalized * grads.d_normalized.float()).sum()
        + (hidden * grads.d_residual.float()).sum()
    )
    loss.backward()
    del d
    return {
        "d_r_old": r.grad,
        "dy_sublayer": y.grad,
        "d_controller_weight": w.grad,
        "d_alpha_pre": a_pre.grad,
        "d_alpha_post": a_post.grad,
        "d_alpha_res": a_res.grad,
        "d_bias": bias.grad,
        "d_gamma": gamma.grad,
    }


@pytest.mark.parametrize("case", ["one_row", "packed_t16", "packed_t7_odd"])
def test_block_backward_matches_autograd(case: str) -> None:
    batch = fixtures.make_batch(case)
    grads = fixtures.make_grads(case, batch)
    _, saved = oracle.mhc_block_forward(batch)
    got = oracle.mhc_block_backward(batch, saved, grads)
    want = _block_autograd(batch, grads)
    for key, ref in want.items():
        assert _close(got[key], ref, rel=3e-2), f"{key} diverges from autograd"


def test_fused_equals_unfused_bytes() -> None:
    """The fused boundary is *defined* as the unfused composition, so this test
    proves the oracle is self-consistent -- it does not prove that any real
    fused kernel matches. That obligation lands on the kernel: a fused
    implementation is byte-equal only if it keeps the same reduction layout and
    the same downcast points. P1-D5 owns proving it for a TE-backed provider.
    """
    unfused = fixtures.make_batch("packed_t16")
    fused = dataclasses.replace(
        unfused, contract=dataclasses.replace(unfused.contract, fusion_mode="fused-pre-norm")
    ).sealed()
    r_a, saved_a = oracle.mhc_block_forward(unfused)
    r_b, saved_b = oracle.mhc_block_forward(fused)
    assert torch.equal(r_a, r_b)
    assert torch.equal(saved_a["normalized"], saved_b["normalized"])
    assert torch.equal(saved_a["residual"], saved_b["residual"])


def test_mixer_frozen_leaks_no_controller_gradient() -> None:
    batch = fixtures.make_batch("mixer_frozen")
    grads = fixtures.make_grads("mixer_frozen", batch)
    _, saved = oracle.mhc_block_forward(batch)
    out = oracle.mhc_block_backward(batch, saved, grads)
    for key in ("d_controller_weight", "d_alpha_pre", "d_alpha_post", "d_alpha_res", "d_bias"):
        assert out[key] is None, f"stop-grad mixer leaked {key}"
    assert out["d_r_old"] is not None and torch.isfinite(out["d_r_old"]).all()


def test_unsupported_modes_fail_closed() -> None:
    batch = fixtures.make_batch("packed_t16")
    bad = dataclasses.replace(
        batch, contract=dataclasses.replace(batch.contract, fusion_mode="fused-everything")
    )
    with pytest.raises(ValueError, match="fusion_mode"):
        bad.validate()
    with pytest.raises(NotImplementedError, match="fail-closed"):
        oracle._check_modes(bad.contract)


# --- invariance (issue #2 acceptance: same row, different batch/pad/stride) -


def test_same_row_same_bytes_across_batch_and_padding() -> None:
    batch = fixtures.make_batch("packed_t16")
    full, _ = oracle.mhc_block_forward(batch)
    for start, stop in ((5, 6), (0, 3), (9, 16), (4, 12)):
        part, _ = oracle.mhc_block_forward(fixtures.slice_batch(batch, start, stop))
        assert torch.equal(part, full[start:stop]), f"rows {start}:{stop} moved with the batch"


def test_non_contiguous_stride_does_not_change_bytes() -> None:
    batch = fixtures.make_batch("packed_t16")
    want, _ = oracle.mhc_block_forward(batch)
    padded_r = torch.zeros(batch.tokens, HC_MULT, batch.hidden * 2, dtype=torch.bfloat16)
    padded_r[:, :, ::2] = batch.r_old
    padded_y = torch.zeros(batch.tokens, batch.hidden * 2, dtype=torch.bfloat16)
    padded_y[:, ::2] = batch.y_sublayer
    strided = dataclasses.replace(batch, r_old=padded_r[:, :, ::2], y_sublayer=padded_y[:, ::2])
    assert not strided.r_old.is_contiguous()
    got, _ = oracle.mhc_block_forward(strided)
    assert torch.equal(got, want)


def test_layer_index_and_token_id_do_not_change_arithmetic() -> None:
    batch = fixtures.make_batch("packed_t16")
    want, _ = oracle.mhc_block_forward(batch)
    moved = dataclasses.replace(
        batch,
        token_id=batch.token_id + 99999,
        contract=dataclasses.replace(batch.contract, layer_index=41),
    ).sealed()
    got, _ = oracle.mhc_block_forward(moved)
    assert torch.equal(got, want)
