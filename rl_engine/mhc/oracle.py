# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 oracle for the six P1 operators (P1-D1..P1-D6; issue #2).

Numeric profile ``oracle-fp32-mhc-v1``:

- Every multiply and accumulate is FP32. Long reductions use the serial
  ascending left fold, 4-way reductions the pinned ``(a0+a1)+(a2+a3)`` tree
  (see :mod:`rl_engine.mhc.reduction`); nothing here calls ``torch.sum``,
  ``matmul``, ``mean`` or ``einsum``.
- Every multiply and add rounds separately (mul-then-add, no FMA fusion). A
  strict CUDA kernel reproduces this with ``__fmul_rn`` / ``__fadd_rn`` or
  registers its own numeric profile.
- ``PRE``/``POST``/``C``, the controller projection ``P`` and the RMS scale
  ``r`` stay FP32 across operator boundaries -- never cast to BF16 in transit.
- Each operator performs exactly one FP32->BF16 downcast, at its output.
  ``mhc_pre`` downcasts the aggregated hidden; ``rmsnorm_residual`` downcasts
  the normalized row; ``mhc_post`` downcasts ``R_new``. Nothing else.
- Gradients are returned FP32 (the accumulator dtype) and round to BF16 only
  when they cross an outer block edge.

The oracle favors auditability over speed; use the start-kit fixture sizes.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch

from rl_engine.mhc.contract import (
    COMB_SLICE,
    POST_SLICE,
    PRE_SLICE,
    GradBoundary,
    LayerContract,
    ResidualBatch,
)
from rl_engine.mhc.reduction import (
    HC_MULT,
    fixed_dot,
    fixed_sum,
    fixed_sumsq,
    stream4_max,
    stream4_sum,
    stream4_sum_dim,
)
from rl_engine.mhc.trace import MHCTrace


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32)


# ---------------------------------------------------------------------------
# 6. fixed_k_gemm -- P1-D6 reference (also the core of fp32_gemm_rms)
# ---------------------------------------------------------------------------


def fixed_k_gemm_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``X @ W.T`` with one FP32 accumulator and K walked left to right.

    ``x``: [M, K], ``w``: [N, K] -> FP32 [M, N]. The batch-invariant GEMM
    reference P1 owns for P2/P3/P5/P7 (issue #2, ``P1-D6``): a single unsplit
    K pass, no Split-K / Stream-K / atomic partial merge, and a single cast at
    the output performed by the caller.
    """
    return fixed_dot(x, w)


def fixed_k_gemm_bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(dX, dW)`` FP32. ``dX = dY @ W`` (over N), ``dW = dY.T @ X`` (over M)."""
    dx = fixed_dot(dy, w.t())
    dw = fixed_dot(dy.t(), x.t())
    return dx, dw


# ---------------------------------------------------------------------------
# 1. hc_split_sinkhorn -- P1-D1
# ---------------------------------------------------------------------------


def _softmax_row(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise softmax over the last (j) axis of [T, 4, 4], max-shifted.

    Frozen: subtract the row max before ``exp`` (both the max and the
    denominator use the pinned 4-way tree), then divide. Returns ``(S, S)``
    where the second value is what backward needs.
    """
    mx = stream4_max(logits, dim=2)  # [T, 4]
    shifted = logits - mx.unsqueeze(2)
    e = torch.exp(shifted)
    den = stream4_sum_dim(e, dim=2)  # [T, 4]
    s = e / den.unsqueeze(2)
    return s, s


def _row_normalize(m: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``M / (sum_row(M) + eps)``; ``sum_row(M)[i] = sum_j M[i, j]``."""
    rs = stream4_sum_dim(m, dim=2) + eps  # [T, 4]
    return m / rs.unsqueeze(2), rs


def _col_normalize(m: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``M / (sum_col(M) + eps)``; ``sum_col(M)[j] = sum_i M[i, j]``."""
    cs = stream4_sum_dim(m, dim=1) + eps  # [T, 4]
    return m / cs.unsqueeze(1), cs


def _row_normalize_bwd(dn: torch.Tensor, m: torch.Tensor, rs_e: torch.Tensor) -> torch.Tensor:
    g = stream4_sum_dim(dn * m, dim=2)  # [T, 4]
    return dn / rs_e.unsqueeze(2) - (g / (rs_e * rs_e)).unsqueeze(2)


def _col_normalize_bwd(dn: torch.Tensor, m: torch.Tensor, cs_e: torch.Tensor) -> torch.Tensor:
    g = stream4_sum_dim(dn * m, dim=1)  # [T, 4]
    return dn / cs_e.unsqueeze(1) - (g / (cs_e * cs_e)).unsqueeze(1)


def hc_split_sinkhorn_fwd(
    h: torch.Tensor, contract: LayerContract | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Split the controller row into ``(PRE, POST, C)``.

    ``h``: FP32 [T, 24]. Layout is frozen as ``PRE[0:4], POST[4:8],
    COMB[8:24]``. Returns ``(pre, post, c, saved)`` with

    - ``PRE[i] = sigmoid(h[i]) + 1e-6``          -> FP32 [T, 4]
    - ``POST[i] = 2 * sigmoid(h[4 + i])``        -> FP32 [T, 4]
    - ``C`` = Sinkhorn-Knopp of ``L = h[8:24].reshape(4, 4)`` -> FP32 [T, 4, 4]

    The Sinkhorn schedule is literal: ``M = softmax_row(L) + eps``, one column
    normalize, then 19 rounds of (row normalize, column normalize). Every
    intermediate is saved so backward can walk the *same* 39 normalizations in
    reverse rather than a mathematically equivalent shortcut.
    """
    contract = contract or LayerContract()
    eps = contract.mhc_eps
    h32 = _f32(h)
    if h32.shape[1] != contract.controller_n:
        raise ValueError(f"h has {h32.shape[1]} controller values, want {contract.controller_n}")

    sig_pre = torch.sigmoid(h32[:, PRE_SLICE])
    pre = sig_pre + eps
    sig_post = torch.sigmoid(h32[:, POST_SLICE])
    post = 2.0 * sig_post

    logits = h32[:, COMB_SLICE].reshape(-1, HC_MULT, HC_MULT)
    s, _ = _softmax_row(logits)
    m = s + eps

    steps: list[tuple[str, torch.Tensor, torch.Tensor]] = []  # (kind, M_in, denom_eps)
    m_next, cs = _col_normalize(m, eps)
    steps.append(("col", m, cs))
    m = m_next
    for _ in range(contract.sinkhorn_iters - 1):
        m_next, rs = _row_normalize(m, eps)
        steps.append(("row", m, rs))
        m = m_next
        m_next, cs = _col_normalize(m, eps)
        steps.append(("col", m, cs))
        m = m_next

    saved = {"sig_pre": sig_pre, "sig_post": sig_post, "softmax": s, "steps": steps, "eps": eps}
    return pre, post, m, saved


def hc_split_sinkhorn_bwd(
    dpre: torch.Tensor,
    dpost: torch.Tensor,
    dc: torch.Tensor,
    saved: dict[str, Any],
) -> torch.Tensor:
    """Backward of ``hc_split_sinkhorn``. Returns ``dh`` FP32 [T, 24].

    Walks the recorded normalization steps in reverse, one VJP per step, so
    the association order matches the forward graph exactly (issue #2 forbids
    a simplified but differently-associated form). The sigmoid legs and the
    softmax leg then fold back into the same [T, 24] row.
    """
    dm = _f32(dc)
    for kind, m_in, denom_e in reversed(saved["steps"]):
        if kind == "row":
            dm = _row_normalize_bwd(dm, m_in, denom_e)
        else:
            dm = _col_normalize_bwd(dm, m_in, denom_e)

    # M0 = softmax(L) + eps  ->  dS = dM0
    s = saved["softmax"]
    inner = stream4_sum_dim(dm * s, dim=2)  # [T, 4]
    dlogits = s * (dm - inner.unsqueeze(2))

    sig_pre, sig_post = saved["sig_pre"], saved["sig_post"]
    dh_pre = _f32(dpre) * (sig_pre * (1.0 - sig_pre))
    dh_post = _f32(dpost) * (2.0 * (sig_post * (1.0 - sig_post)))
    return torch.cat([dh_pre, dh_post, dlogits.reshape(dlogits.shape[0], HC_MULT * HC_MULT)], dim=1)


# ---------------------------------------------------------------------------
# 2. fp32_gemm_rms -- P1-D2 (controller projection + controller RMS scale)
# ---------------------------------------------------------------------------


def fp32_gemm_rms_fwd(
    x_flat: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Controller projection and RMS scale from the same flattened residual.

    ``x_flat``: FP32 [T, K], ``weight``: FP32 [N, K]. Returns ``(P, r, saved)``:

    - ``P[t, n] = sum_k X[t, k] * W[n, k]``  (fixed-K, ascending, unsplit)
    - ``s = sum_k X[t, k]^2``; ``norm = sqrt(s)``; ``q = norm / sqrt(K)``;
      ``r = 1 / (q + eps)``

    Note this is the *controller* RMS: ``1 / (sqrt(mean(X^2)) + eps)``, and
    deliberately **not** the ``rsqrt(mean + eps)`` of ``rmsnorm_residual``.
    The two must never be interchanged (issue #2 acceptance).
    """
    x32 = _f32(x_flat)
    p = fixed_k_gemm_fwd(x32, _f32(weight))
    s = fixed_sumsq(x32, dim=1)  # [T]
    norm = torch.sqrt(s)
    q = norm / math.sqrt(float(x32.shape[1]))
    r = 1.0 / (q + eps)
    return p, r, {"s": s, "norm": norm, "q": q, "r": r, "k": int(x32.shape[1])}


def fp32_gemm_rms_bwd(
    dp: torch.Tensor,
    dr: torch.Tensor,
    x_flat: torch.Tensor,
    weight: torch.Tensor,
    saved: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(dX, dW)`` FP32; ``dX`` sums the GEMM leg and the RMS leg.

    ``dX_gemm = dP @ W``, ``dW = dP.T @ X``. For the RMS leg, with ``K``,
    ``q = sqrt(s)/sqrt(K)`` and ``r = 1/(q + eps)``:
    ``dX_rms[k] = g_r * ((-r^2 * X[k]) / (K * q))``.
    """
    x32 = _f32(x_flat)
    dx_gemm, dw = fixed_k_gemm_bwd(_f32(dp), x32, _f32(weight))
    r, q, k = saved["r"], saved["q"], float(saved["k"])
    neg_r2 = -(r * r)
    denom = k * q
    dx_rms = _f32(dr).unsqueeze(1) * ((neg_r2.unsqueeze(1) * x32) / denom.unsqueeze(1))
    return dx_gemm + dx_rms, dw


# ---------------------------------------------------------------------------
# 4a. h_aggregate -- the PRE-weighted four-stream merge inside mhc_pre
# ---------------------------------------------------------------------------


def h_aggregate_fwd(pre: torch.Tensor, r_old: torch.Tensor) -> torch.Tensor:
    """``H = (PRE0*R0 + PRE1*R1) + (PRE2*R2 + PRE3*R3)`` -> BF16 [T, D].

    ``pre``: FP32 [T, 4], ``r_old``: BF16 [T, 4, D]. The four streams are
    promoted to FP32 before any arithmetic; the single FP32->BF16 downcast at
    the end is the only rounding point in this operator.
    """
    r32 = _f32(r_old)
    parts = [pre[:, i].unsqueeze(1) * r32[:, i, :] for i in range(HC_MULT)]
    return stream4_sum(parts).to(torch.bfloat16)


def h_aggregate_bwd(
    dh: torch.Tensor, pre: torch.Tensor, r_old: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(dR_from_aggregate [T, 4, D], dPRE [T, 4])``, both FP32.

    ``dR[i, d] = PRE[i] * dH[d]``; ``dPRE[i] = sum_d dH[d] * R[i, d]`` over the
    full hidden width with the serial ascending fold.
    """
    dh32 = _f32(dh)
    r32 = _f32(r_old)
    dr = torch.stack([pre[:, i].unsqueeze(1) * dh32 for i in range(HC_MULT)], dim=1)
    dpre = torch.stack([fixed_sum(dh32 * r32[:, i, :], dim=1) for i in range(HC_MULT)], dim=1)
    return dr, dpre


# ---------------------------------------------------------------------------
# 4b. mhc_pre -- P1-D4 (composite: gemm_rms -> affine -> sinkhorn -> aggregate)
# ---------------------------------------------------------------------------


def _controller_affine(
    p: torch.Tensor, r: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = ((r * P) * alpha) + bias``. Returns ``(h, m)`` with ``m = r * P``.

    ``alpha`` is the [24] broadcast of the three learnable scalars; the
    association ``((r * P) * alpha) + bias`` is Megatron's
    ``h = r * proj * alpha_ + self.bias``.
    """
    m = r.unsqueeze(1) * p
    return (m * alpha.unsqueeze(0)) + bias.unsqueeze(0), m


def mhc_pre_fwd(
    batch: ResidualBatch, ops: Any = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """mHC entry: ``R_old -> (hidden BF16, PRE, POST, C)``.

    Steps, in the frozen order:
    ``X_flat = reshape(R, [T, 4D])`` -> ``P, r = fp32_gemm_rms(X_flat)`` ->
    ``h = ((r * P) * alpha) + bias`` -> ``PRE, POST, C = hc_split_sinkhorn(h)``
    -> ``H = h_aggregate(PRE, R)``.
    """
    ops = ops if ops is not None else sys.modules[__name__]
    contract = batch.contract
    x_flat = _f32(batch.r_old).reshape(batch.tokens, contract.flat_k)
    p, r, gemm_saved = ops.fp32_gemm_rms_fwd(x_flat, batch.controller.weight, contract.mhc_eps)
    alpha = batch.controller.expanded_alpha(contract)
    h, m = _controller_affine(p, r, alpha, batch.controller.bias)
    pre, post, c, sink_saved = ops.hc_split_sinkhorn_fwd(h, contract)
    hidden = ops.h_aggregate_fwd(pre, batch.r_old)
    saved = {
        "x_flat": x_flat,
        "p": p,
        "r": r,
        "m": m,
        "h": h,
        "pre": pre,
        "post": post,
        "c": c,
        "gemm": gemm_saved,
        "sinkhorn": sink_saved,
    }
    return hidden, pre, post, c, saved


def mhc_pre_bwd(
    dhidden: torch.Tensor,
    dpost: torch.Tensor,
    dc: torch.Tensor,
    batch: ResidualBatch,
    saved: dict[str, Any],
    ops: Any = None,
) -> dict[str, torch.Tensor | None]:
    """Composite backward: aggregate -> sinkhorn -> affine -> gemm_rms.

    ``dhidden`` is the gradient of the aggregated hidden; ``dpost``/``dc``
    arrive from :func:`mhc_post_bwd`. Returns ``d_r_old`` (aggregate leg plus
    controller leg, summed in FP32) and the controller parameter gradients.
    Under ``trainability='mixer-frozen'`` the controller parameter gradients
    are ``None`` -- a stop-grad mixer must not leak ``dMixWeight``.
    """
    ops = ops if ops is not None else sys.modules[__name__]
    contract = batch.contract
    dr_aggregate, dpre = ops.h_aggregate_bwd(dhidden, saved["pre"], batch.r_old)
    dh = ops.hc_split_sinkhorn_bwd(dpre, dpost, dc, saved["sinkhorn"])

    alpha = batch.controller.expanded_alpha(contract)
    m, p, r = saved["m"], saved["p"], saved["r"]
    dbias = fixed_sum(dh, dim=0)
    # dAlpha is three scalars, not 24: each learnable alpha is broadcast over
    # its segment, so its gradient is a reduction over that segment on top of
    # the token reduction. Both folds are the pinned ascending order.
    dalpha_per_n = fixed_sum(dh * m, dim=0)  # [24]
    dalpha_pre = fixed_sum(dalpha_per_n[PRE_SLICE], dim=0).reshape(1)
    dalpha_post = fixed_sum(dalpha_per_n[POST_SLICE], dim=0).reshape(1)
    dalpha_res = fixed_sum(dalpha_per_n[COMB_SLICE], dim=0).reshape(1)
    dm = dh * alpha.unsqueeze(0)
    dp = dm * r.unsqueeze(1)
    dr_scale = fixed_sum(dm * p, dim=1)  # [T], over the 24 controller values

    dx_flat, dweight = ops.fp32_gemm_rms_bwd(
        dp, dr_scale, saved["x_flat"], batch.controller.weight, saved["gemm"]
    )
    dr_controller = dx_flat.reshape(batch.tokens, contract.hc_mult, contract.hidden)
    d_r_old = dr_aggregate + dr_controller

    frozen = contract.trainability == "mixer-frozen"
    return {
        "d_r_old": d_r_old,
        "d_controller_weight": None if frozen else dweight,
        "d_alpha_pre": None if frozen else dalpha_pre,
        "d_alpha_post": None if frozen else dalpha_post,
        "d_alpha_res": None if frozen else dalpha_res,
        "d_bias": None if frozen else dbias,
    }


# ---------------------------------------------------------------------------
# 3. mhc_post -- P1-D3
# ---------------------------------------------------------------------------


def mhc_post_fwd(
    r_old: torch.Tensor, y: torch.Tensor, c: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """``R_new[j, d] = (C[0,j]R0 + C[1,j]R1) + (C[2,j]R2 + C[3,j]R3) + POST[j]*y[d]``.

    ``r_old``: BF16 [T, 4, D], ``y``: BF16 [T, D], ``c``: FP32 [T, 4, 4],
    ``post``: FP32 [T, 4] -> BF16 [T, 4, D]. One FP32->BF16 downcast, at the
    output.
    """
    r32 = _f32(r_old)
    y32 = _f32(y)
    columns = []
    for j in range(HC_MULT):
        parts = [c[:, i, j].unsqueeze(1) * r32[:, i, :] for i in range(HC_MULT)]
        old_mix = stream4_sum(parts)
        columns.append(old_mix + post[:, j].unsqueeze(1) * y32)
    return torch.stack(columns, dim=1).to(torch.bfloat16)


def mhc_post_bwd(
    dr_new: torch.Tensor,
    r_old: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    post: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(dR_old, dy, dC, dPOST)`` FP32.

    ``dR_old[i,d] = sum_j C[i,j] G[j,d]`` and ``dy[d] = sum_j POST[j] G[j,d]``
    use the pinned 4-way tree; ``dC[i,j] = sum_d R_old[i,d] G[j,d]`` and
    ``dPOST[j] = sum_d y[d] G[j,d]`` use the serial ascending fold over the
    hidden width.
    """
    g = _f32(dr_new)
    r32 = _f32(r_old)
    y32 = _f32(y)
    dr_old = torch.stack(
        [
            stream4_sum([c[:, i, j].unsqueeze(1) * g[:, j, :] for j in range(HC_MULT)])
            for i in range(HC_MULT)
        ],
        dim=1,
    )
    dy = stream4_sum([post[:, j].unsqueeze(1) * g[:, j, :] for j in range(HC_MULT)])
    dc = torch.stack(
        [
            torch.stack(
                [fixed_sum(r32[:, i, :] * g[:, j, :], dim=1) for j in range(HC_MULT)], dim=1
            )
            for i in range(HC_MULT)
        ],
        dim=1,
    )
    dpost = torch.stack([fixed_sum(y32 * g[:, j, :], dim=1) for j in range(HC_MULT)], dim=1)
    return dr_old, dy, dc, dpost


# ---------------------------------------------------------------------------
# 5. rmsnorm_residual -- P1-D5
# ---------------------------------------------------------------------------


def rmsnorm_residual_fwd(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """RMSNorm with a fork of the *unnormalized* input as the residual branch.

    This is **not** ``x += residual`` followed by a norm. ``x``: BF16 [T, D],
    ``gamma``: BF16 [D]. Returns ``(y BF16, residual BF16, saved)``:

    ``s = sum_d FP32(x[d])^2``; ``m = s / D``; ``r = rsqrt(m + eps)``;
    ``y[d] = (FP32(x[d]) * r) * FP32(gamma[d])``.

    ``rsqrt`` is mandatory here -- mixing in ``1 / sqrt(...)`` changes bytes,
    and the controller RMS in :func:`fp32_gemm_rms_fwd` uses that other form
    on purpose. The residual fork keeps the original BF16 bytes untouched.

    P1-5 (#18) says to prefer TE's ``TEFusedResidualRMSNorm`` first and to
    self-write only when TE's reduction/dtype fails the deterministic contract.
    It fails: that module fuses the fork and the norm and refuses to expose the
    intermediate (it raises on any forward hook), so the two boundaries cannot
    be hashed separately and a divergence cannot be localized. v1 is therefore
    unfused; a TE fast path can be swapped back through the provider hook once
    it is proven byte-equal.
    """
    x32 = _f32(x)
    d = x32.shape[1]
    s = fixed_sumsq(x32, dim=1)
    m = s / float(d)
    r = torch.rsqrt(m + eps)
    y = (x32 * r.unsqueeze(1)) * _f32(gamma).unsqueeze(0)
    residual = x.clone()
    return y.to(torch.bfloat16), residual, {"r": r, "x32": x32, "d": d}


def rmsnorm_residual_bwd(
    dy: torch.Tensor,
    d_residual: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    saved: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(dX, dGamma)`` FP32.

    ``u[d] = dy[d]*gamma[d]``; ``q = sum_j u[j]*x[j]``;
    ``dx_norm[d] = (r*u[d]) - (((x[d]*r^3)*q)/D)``;
    ``dgamma[d] = sum_t (dy[t,d]*x[t,d])*r[t]``.
    The input also feeds the residual fork, so ``dX = dx_norm + d_residual``.
    """
    x32, r, d = saved["x32"], saved["r"], saved["d"]
    dy32 = _f32(dy)
    u = dy32 * _f32(gamma).unsqueeze(0)
    q = fixed_sum(u * x32, dim=1)  # [T]
    r3 = (r * r) * r
    dx_norm = (r.unsqueeze(1) * u) - (((x32 * r3.unsqueeze(1)) * q.unsqueeze(1)) / float(d))
    dgamma = fixed_sum((dy32 * x32) * r.unsqueeze(1), dim=0)
    return dx_norm + _f32(d_residual), dgamma


# ---------------------------------------------------------------------------
# Fused pre+norm boundary (issue #2: fused/unfused equivalence case)
# ---------------------------------------------------------------------------


def mhc_pre_rmsnorm_fused_fwd(
    batch: ResidualBatch, ops: Any = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The Miles/XoRL fused boundary: one call for pre-mix + normalize.

    Miles fuses the four-stream merge and the normalization into a single
    launch, so M0/M1 have no separate boundary there. The oracle defines the
    fused joint boundary as *exactly* the unfused composition, which makes
    "fused equals unfused" a testable claim rather than an assumption. A
    kernel whose fused residual store changes the reduction layout must say
    so explicitly (register a different numeric profile) -- it may not present
    itself as the same kernel.
    """
    ops = ops if ops is not None else sys.modules[__name__]
    hidden, pre, post, c, saved = ops.mhc_pre_fwd(batch, ops=ops)
    normalized, residual, norm_saved = ops.rmsnorm_residual_fwd(
        hidden, batch.norm.gamma, batch.contract.rmsnorm_eps
    )
    saved = {**saved, "hidden": hidden, "norm": norm_saved}
    return hidden, normalized, residual, post, c, {**saved, "pre": pre}


# ---------------------------------------------------------------------------
# Block composition (the full P1 forward/backward chain)
# ---------------------------------------------------------------------------

SUPPORTED_FUSION = ("unfused", "fused-pre-norm")
SUPPORTED_TRAINABILITY = ("full", "mixer-frozen")


def _check_modes(contract: LayerContract) -> None:
    """Fail-closed on any fusion / trainability mode the kit does not define."""
    if contract.fusion_mode not in SUPPORTED_FUSION:
        raise NotImplementedError(
            f"fusion_mode {contract.fusion_mode!r} is not defined by the P1 contract; "
            f"supported: {SUPPORTED_FUSION} (fail-closed, issue #2 acceptance 4)"
        )
    if contract.trainability not in SUPPORTED_TRAINABILITY:
        raise NotImplementedError(
            f"trainability {contract.trainability!r} is not defined by the P1 contract; "
            f"supported: {SUPPORTED_TRAINABILITY} (fail-closed, issue #2 acceptance 4)"
        )


def mhc_block_forward(
    batch: ResidualBatch, trace: MHCTrace | None = None, ops: Any = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full P1 forward: ``R_old -> mhc_pre -> rmsnorm_residual -> [sublayer] -> mhc_post``.

    The transformer sublayer is external: its output arrives as
    ``batch.y_sublayer``. Returns ``(R_new BF16, saved)``.
    """
    ops = ops if ops is not None else sys.modules[__name__]
    batch.validate()
    _check_modes(batch.contract)

    if batch.contract.fusion_mode == "fused-pre-norm":
        hidden, normalized, residual, post, c, saved = ops.mhc_pre_rmsnorm_fused_fwd(batch, ops=ops)
        pre = saved["pre"]
        norm_saved = saved["norm"]
    else:
        hidden, pre, post, c, saved = ops.mhc_pre_fwd(batch, ops=ops)
        normalized, residual, norm_saved = ops.rmsnorm_residual_fwd(
            hidden, batch.norm.gamma, batch.contract.rmsnorm_eps
        )

    r_new = ops.mhc_post_fwd(batch.r_old, batch.y_sublayer, c, post)

    if trace is not None:
        trace.note("reduction_tree", "long=serial-ascending-left-fold; stream4=(a0+a1)+(a2+a3)")
        trace.note("fma", "mul-then-add, no fusion")
        trace.note("rsqrt", "rmsnorm=rsqrt(mean+eps); controller=1/(sqrt(mean)+eps)")
        trace.note("downcast_points", "mhc_pre.hidden, rmsnorm.normalized, mhc_post.r_new")
        trace.note("fusion_mode", batch.contract.fusion_mode)
        trace.note("trainability", batch.contract.trainability)
        trace.note("weight_fingerprint", batch.compute_weight_fingerprint())
        trace.record("controller.p", saved["p"])
        trace.record("controller.r", saved["r"])
        trace.record("controller.h", saved["h"])
        trace.record("split.pre", pre)
        trace.record("split.post", post)
        trace.record("split.c", c)
        trace.record("pre.hidden", hidden)
        trace.record("norm.normalized", normalized)
        trace.record("norm.residual", residual)
        trace.record("post.r_new", r_new)

    saved = {
        **saved,
        "hidden": hidden,
        "normalized": normalized,
        "residual": residual,
        "norm": norm_saved,
        "post": post,
        "c": c,
        "pre": pre,
    }
    return r_new, saved


def mhc_block_backward(
    batch: ResidualBatch,
    saved: dict[str, Any],
    grads: GradBoundary,
    trace: MHCTrace | None = None,
    ops: Any = None,
) -> dict[str, torch.Tensor | None]:
    """Full P1 backward. Returns ``dStream[0..3]`` (as ``d_r_old``), ``dy``,
    ``dX``/``dResidual`` at the norm edge, ``dGamma`` and the controller
    parameter gradients.

    ``grads.d_normalized`` and ``grads.d_residual`` come from the sublayer
    owner: P1 never differentiates attention/FFN/MoE. ``dy_sublayer`` is an
    output boundary handed back to that owner.
    """
    ops = ops if ops is not None else sys.modules[__name__]
    grads.validate(batch)
    _check_modes(batch.contract)

    dr_old_post, dy, dc, dpost = ops.mhc_post_bwd(
        grads.d_r_new, batch.r_old, batch.y_sublayer, saved["c"], saved["post"]
    )
    dhidden, dgamma = ops.rmsnorm_residual_bwd(
        grads.d_normalized, grads.d_residual, saved["hidden"], batch.norm.gamma, saved["norm"]
    )
    pre_grads = ops.mhc_pre_bwd(dhidden, dpost, dc, batch, saved, ops=ops)
    d_r_old = pre_grads["d_r_old"] + dr_old_post

    out: dict[str, torch.Tensor | None] = {
        "d_r_old": d_r_old,
        "dy_sublayer": dy,
        "d_hidden": dhidden,
        "d_gamma": dgamma,
        "d_c": dc,
        "d_post": dpost,
        "d_controller_weight": pre_grads["d_controller_weight"],
        "d_alpha_pre": pre_grads["d_alpha_pre"],
        "d_alpha_post": pre_grads["d_alpha_post"],
        "d_alpha_res": pre_grads["d_alpha_res"],
        "d_bias": pre_grads["d_bias"],
    }
    if trace is not None:
        trace.record("bwd.dy_sublayer", dy)
        trace.record("bwd.d_c", dc)
        trace.record("bwd.d_post", dpost)
        trace.record("bwd.d_hidden", dhidden)
        trace.record("bwd.d_gamma", dgamma)
        trace.record("bwd.d_r_old", d_r_old)
        for i in range(HC_MULT):
            trace.record(f"bwd.d_stream{i}", d_r_old[:, i, :])
    return out


__all__ = [
    "SUPPORTED_FUSION",
    "SUPPORTED_TRAINABILITY",
    "fixed_k_gemm_bwd",
    "fixed_k_gemm_fwd",
    "fp32_gemm_rms_bwd",
    "fp32_gemm_rms_fwd",
    "h_aggregate_bwd",
    "h_aggregate_fwd",
    "hc_split_sinkhorn_bwd",
    "hc_split_sinkhorn_fwd",
    "mhc_block_backward",
    "mhc_block_forward",
    "mhc_post_bwd",
    "mhc_post_fwd",
    "mhc_pre_bwd",
    "mhc_pre_fwd",
    "mhc_pre_rmsnorm_fused_fwd",
    "rmsnorm_residual_bwd",
    "rmsnorm_residual_fwd",
]
