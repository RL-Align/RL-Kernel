# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden DSv4 manifold hyper-connections (mHC), numeric profile ``dsv4-mhc-v1``.

Semantics follow DeepSeek-V4's reference inference code (``Block.hc_pre`` / ``hc_post`` and
``ParallelHead.hc_head``), which vLLM also implements:

    x      = residual.flatten(-2).float()                       # [T, K], K = hc * H
    r      = rsqrt(mean(x^2) + eps)
    mixes  = (x @ fn.T) * r
    pre    = sigmoid(mixes[:4]  * scale[0] + base[:4]) + eps
    post   = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb   = sinkhorn(softmax(mixes[8:].view(4, 4) * scale[2] + base[8:]) + eps)
    y      = sum_i pre_i * residual_i                           # -> BF16
    new_j  = post_j * x + sum_i comb[i, j] * residual_i         # -> BF16 (hc_post)
    head   = sum_i (sigmoid(mixes_h * scale + base) + eps)_i * residual_i   # hc_head

P1's ``oracle-fp32-mhc-v1`` uses ``1 / (sqrt(mean) + eps)``; this profile does not.

The arithmetic is fixed so that the CUDA kernels can match it byte for byte:

- **The controller GEMM and sum of squares** (over K = 4H = 16384) come from DeepGEMM's
  ``tf32_hc_prenorm_gemm`` (the kernel vLLM's rollout runs) with a fixed
  ``SPLITS = 16`` split-K. It returns per-split partials ``[16, T, N]`` and ``[16, T]``.
  - That kernel is batch-invariant at a fixed split count (probed). Its TF32 tensor-core
    accumulation is not reproduced here, so this golden starts from those partials.
  - The partials reduce by halving: ``p[0:8] + p[8:16]``, then 4, 2, 1.
  - ``hc_head``'s ``fn`` (4 rows) is zero-padded to 8 rows for the kernel.
- **Dot products over H** (the fused RMSNorm and the backward) use ``SLICES = 128``
  slices.
  - Slice ``i`` owns the runs ``[8 i + 1024 j, + 8)``, so each CUDA thread reads 16 bytes
    at a time.
  - Each slice accumulates with ``fmaf`` in ascending ``j``, then ascending position within
    the run.
  - The 128 partials reduce by halving within each 32, then ``(g0 + g1) + (g2 + g3)``.
- **RMSNorm** (``attn_norm`` / ``ffn_norm`` after ``mhc_pre``, ``norm`` after ``hc_head``) is
  fused when its weight is given, as in DeepSeek's ``RMSNorm``:
  ``out = BF16((y * r) * w)``, with ``r = 1 / sqrt(sum(y^2) / H + eps)`` over the BF16
  ``y``. Its weight is frozen too (dX only).
- **Everything else** is FP32 with separately rounded multiplies and adds.
  - Sums of four terms are ``(a0 + a1) + (a2 + a3)``.
  - Division and square root are correctly rounded; ``exp`` is CUDA's ``expf``.
  - ``r = 1 / sqrt(ss / K + eps)``: the correctly rounded form of ``rsqrt``.
- **BF16** rounds to nearest even, once, at each operator's output.

Batch invariance: every value of a token depends only on that token, the weights and the
constants. Backward returns dX only (the mHC parameters are frozen).

The backward's FMAs are emulated exactly in float64. The product of two FP32 values is
exact in double, and ``TwoSum`` recovers the rounding error of the add, which decides the
rare FP32 ties.
"""

from __future__ import annotations

import torch
from torch import Tensor

HC = 4
SLICES = 128
RUN = 8
SPLITS = 16
NORM_EPS = 1e-6
SINKHORN_ITERS = 20
HC_EPS = 1e-6
RMS_EPS = 1e-6
POST_MULT = 2.0


# --------------------------------------------------------------------------- FP32 helpers
def _c(value: float, like: Tensor) -> Tensor:
    return torch.tensor(value, dtype=torch.float32, device=like.device)


def fma_f32(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """``fmaf(a, b, c)`` for FP32 tensors (broadcasting), exactly rounded."""
    a64, b64, c64 = a.double(), b.double(), c.double()
    p = a64 * b64  # exact: 24 + 24 significant bits
    s = p + c64
    # TwoSum: s + err == p + c exactly.
    bb = s - p
    err = (p - (s - bb)) + (c64 - bb)
    r = s.float()
    rd = r.double()
    # s can sit exactly on a midpoint between two FP32 values only if the double add was
    # exact or rounded onto it; the true sum p + c is then off the midpoint by err.
    other = torch.nextafter(
        r, torch.where(s > rd, torch.full_like(r, float("inf")), torch.full_like(r, -float("inf")))
    )
    mid = (rd + other.double()) * 0.5
    tie = (s != rd) & (s == mid) & (err != 0)
    toward_other = tie & ((err > 0) == (other.double() > rd))
    return torch.where(toward_other, other, r)


def _tree4(a0: Tensor, a1: Tensor, a2: Tensor, a3: Tensor) -> Tensor:
    return (a0 + a1) + (a2 + a3)


def slice_tree(partials: Tensor) -> Tensor:
    """Reduce the last axis (``SLICES`` partials) by the fixed tree."""
    assert partials.shape[-1] == SLICES
    g = partials.reshape(*partials.shape[:-1], 4, 32)
    for half in (16, 8, 4, 2, 1):
        g = g[..., :half] + g[..., half : 2 * half]
    g = g[..., 0]  # [..., 4]
    return _tree4(g[..., 0], g[..., 1], g[..., 2], g[..., 3])


def slice_dot(a: Tensor, b: Tensor) -> Tensor:
    """``sum_h a[..., h] * b[..., h]`` by the slice scheme (runs of 8). Shapes broadcast."""
    n = a.shape[-1]
    assert n % RUN == 0 and b.shape[-1] == n
    span = SLICES * RUN
    n_pad = (n + span - 1) // span * span
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    af = torch.nn.functional.pad(a.float(), (0, n_pad - n)).reshape(*a.shape[:-1], -1, SLICES, RUN)
    bf = torch.nn.functional.pad(b.float(), (0, n_pad - n)).reshape(*b.shape[:-1], -1, SLICES, RUN)
    acc = torch.zeros(*shape, SLICES, dtype=torch.float32, device=a.device)
    for j in range(n_pad // span):
        if j * span >= n:
            break
        for e in range(RUN):
            acc = fma_f32(af[..., j, :, e], bf[..., j, :, e], acc)
    return slice_tree(acc)


def split_sum(partials: Tensor) -> Tensor:
    """Reduce the leading ``SPLITS`` axis of GEMM partials by the halving tree."""
    assert partials.shape[0] == SPLITS
    p = partials.float()
    while p.shape[0] > 1:
        half = p.shape[0] // 2
        p = p[:half] + p[half:]
    return p[0]


def gemm_partials(residual: Tensor, fn: Tensor) -> tuple[Tensor, Tensor]:
    """DeepGEMM ``tf32_hc_prenorm_gemm`` partials ``([16, T, Np], [16, T])`` (CUDA only).

    ``fn`` rows are zero-padded to a multiple of 8 (``hc_head`` has 4).
    """
    from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm

    dg, _ = require_deepgemm(residual.device)
    t = residual.shape[0]
    x = residual.reshape(t, residual.shape[1] * residual.shape[2]).contiguous()
    n = fn.shape[0]
    n_pad = (n + 7) // 8 * 8
    f = fn.float()
    if n_pad != n:
        f = torch.cat([f, f.new_zeros(n_pad - n, f.shape[1])])
    pm = torch.empty(SPLITS, t, n_pad, dtype=torch.float32, device=residual.device)
    ps = torch.empty(SPLITS, t, dtype=torch.float32, device=residual.device)
    if t:
        dg.tf32_hc_prenorm_gemm(x, f.contiguous(), pm, ps, SPLITS)
    return pm, ps


def rms_scale(ss: Tensor, k: int, eps: float = RMS_EPS) -> Tensor:
    mean = ss / _c(float(k), ss)
    return _c(1.0, ss) / torch.sqrt(mean + _c(eps, ss))


def _sigmoid(x: Tensor) -> Tensor:
    one = _c(1.0, x)
    return one / (one + torch.exp(-x))


def _logits(raw: Tensor, r: Tensor, scale: Tensor, base: Tensor, seg: list[int]) -> Tensor:
    m = raw * r[:, None]
    s = scale.float()[torch.tensor(seg, device=raw.device)]
    return m * s[None, :] + base.float()[None, :]


def _segments(n: int) -> list[int]:
    return [0] * HC + [1] * HC + [2] * (HC * HC) if n == 2 * HC + HC * HC else [0] * n


def _rows(m: Tensor) -> Tensor:  # [T, 4, 4] -> sum over j, [T, 4]
    return _tree4(m[:, :, 0], m[:, :, 1], m[:, :, 2], m[:, :, 3])


def _cols(m: Tensor) -> Tensor:  # [T, 4, 4] -> sum over i, [T, 4]
    return _tree4(m[:, 0, :], m[:, 1, :], m[:, 2, :], m[:, 3, :])


def controller(
    raw: Tensor,
    ss: Tensor,
    scale: Tensor,
    base: Tensor,
    k: int,
    *,
    eps: float = HC_EPS,
    rms_eps: float = RMS_EPS,
    iters: int = SINKHORN_ITERS,
) -> dict[str, Tensor]:
    """Everything between the GEMM and the stream mixes, with the intermediates backward needs."""
    r = rms_scale(ss, k, rms_eps)
    logits = _logits(raw, r, scale, base, _segments(raw.shape[1]))
    e = _c(eps, raw)
    out: dict[str, Tensor] = {"r": r, "logits": logits}
    sig_pre = _sigmoid(logits[:, :HC])
    out["sig_pre"] = sig_pre
    out["pre"] = sig_pre + e
    if raw.shape[1] == HC:
        return out
    sig_post = _sigmoid(logits[:, HC : 2 * HC])
    out["sig_post"] = sig_post
    out["post"] = _c(POST_MULT, raw) * sig_post
    lg = logits[:, 2 * HC :].reshape(-1, HC, HC)
    mx = lg.max(dim=2, keepdim=True).values
    ex = torch.exp(lg - mx)
    soft = ex / _rows(ex)[:, :, None]
    m = soft + e
    steps: list[tuple[str, Tensor, Tensor]] = []
    d = _cols(m) + e
    steps.append(("col", m, d))
    m = m / d[:, None, :]
    for _ in range(iters - 1):
        d = _rows(m) + e
        steps.append(("row", m, d))
        m = m / d[:, :, None]
        d = _cols(m) + e
        steps.append(("col", m, d))
        m = m / d[:, None, :]
    out.update(soft=soft, steps=steps, comb=m)  # type: ignore[arg-type]
    return out


def aggregate(pre: Tensor, residual: Tensor) -> Tensor:
    """``y = (p0 x0 + p1 x1) + (p2 x2 + p3 x3)`` -> BF16 [T, H]."""
    x = residual.float()
    parts = [pre[:, i, None] * x[:, i, :] for i in range(HC)]
    return _tree4(*parts).to(torch.bfloat16)


def rms_norm(y: Tensor, weight: Tensor, eps: float = NORM_EPS) -> tuple[Tensor, Tensor]:
    """DeepSeek RMSNorm on BF16 ``y`` [T, H]: ``(BF16((y * r) * w), r [T])``."""
    yf = y.float()
    ss = slice_dot(yf, yf)
    r = rms_scale(ss, y.shape[-1], eps)
    return ((yf * r[:, None]) * weight.float()[None, :]).to(torch.bfloat16), r


def rms_norm_backward(dz: Tensor, y: Tensor, weight: Tensor, r: Tensor) -> Tensor:
    """dX of ``rms_norm`` (frozen weight): ``u r - (sum(u y) r^3 / H) y`` with ``u = dz w``."""
    yf = y.float()
    u = dz.float() * weight.float()[None, :]
    dot = slice_dot(u, yf)
    c = ((dot * r) * (r * r)) / _c(float(y.shape[-1]), r)
    return u * r[:, None] - c[:, None] * yf


# --------------------------------------------------------------------------- public forward
def _maybe_norm(y: Tensor, norm_weight: Tensor | None) -> Tensor:
    return y if norm_weight is None else rms_norm(y, norm_weight)[0]


def mhc_pre_from_partials(
    residual: Tensor,
    pmix: Tensor,
    pss: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``residual`` BF16 [T, 4, H] and the GEMM partials -> ``(out BF16 [T, H], post [T, 4],
    comb [T, 4, 4], raw [T, 24], ss [T])``.

    ``out`` is the stream mix ``y``, or ``rms_norm(y, norm_weight)`` when a weight is given.
    ``raw`` and ``ss`` are what the backward recomputes the controller from.
    """
    raw = split_sum(pmix)[:, :24]
    ss = split_sum(pss)
    ctl = controller(raw, ss, scale, base, residual.shape[1] * residual.shape[2])
    y = aggregate(ctl["pre"], residual)
    return _maybe_norm(y, norm_weight), ctl["post"], ctl["comb"], raw, ss


def mhc_pre(
    residual: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm_weight: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``mhc_pre_from_partials`` on DeepGEMM's partials (CUDA only)."""
    pm, ps = gemm_partials(residual, fn)
    return mhc_pre_from_partials(residual, pm, ps, scale, base, norm_weight)


def mhc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """``new_j = post_j * x + ((c0j r0 + c1j r1) + (c2j r2 + c3j r3))`` -> BF16 [T, 4, H]."""
    xf = x.float()
    rf = residual.float()
    cols = []
    for j in range(HC):
        mix = _tree4(*[comb[:, i, j, None] * rf[:, i, :] for i in range(HC)])
        cols.append(post[:, j, None] * xf + mix)
    return torch.stack(cols, dim=1).to(torch.bfloat16)


def hc_head_from_partials(
    residual: Tensor,
    pmix: Tensor,
    pss: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``residual`` BF16 [T, 4, H], partials of the 8-row padded ``fn``, ``scale`` [1],
    ``base`` [4] -> ``(out BF16 [T, H], raw [T, 4], ss [T])`` (``out`` normed if a weight
    is given)."""
    raw = split_sum(pmix)[:, :HC]
    ss = split_sum(pss)
    ctl = controller(raw, ss, scale, base, residual.shape[1] * residual.shape[2])
    return _maybe_norm(aggregate(ctl["pre"], residual), norm_weight), raw, ss


def hc_head(
    residual: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm_weight: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """``hc_head_from_partials`` on DeepGEMM's partials (CUDA only)."""
    pm, ps = gemm_partials(residual, fn)
    return hc_head_from_partials(residual, pm, ps, scale, base, norm_weight)


# --------------------------------------------------------------------------- backward (dX only)
def mhc_post_backward(
    d_new: Tensor, x: Tensor, residual: Tensor, post: Tensor, comb: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns ``(dx [T, H], d_residual [T, 4, H], d_post [T, 4], d_comb [T, 4, 4])``, FP32."""
    d = d_new.float()
    xf = x.float()
    rf = residual.float()
    dx = _tree4(*[post[:, j, None] * d[:, j, :] for j in range(HC)])
    dres = torch.stack(
        [_tree4(*[comb[:, i, j, None] * d[:, j, :] for j in range(HC)]) for i in range(HC)], dim=1
    )
    dpost = slice_dot(d, xf[:, None, :])
    dcomb = slice_dot(rf[:, :, None, :], d[:, None, :, :])  # [T, i, j]
    return dx, dres, dpost, dcomb


def _norm_backward(dout: Tensor, m_in: Tensor, d: Tensor, kind: str) -> Tensor:
    """Backward of ``M / (sum + eps)`` over rows (``kind='row'``) or columns."""
    prod = dout * m_in
    if kind == "row":
        g = _rows(prod)[:, :, None]
        dd = d[:, :, None]
    else:
        g = _cols(prod)[:, None, :]
        dd = d[:, None, :]
    return dout / dd - g / (dd * dd)


def _controller_backward(
    raw: Tensor,
    ctl: dict[str, Tensor],
    scale: Tensor,
    k: int,
    ss: Tensor,
    dpre: Tensor,
    dpost: Tensor | None,
    dcomb: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """``(draw [T, N], dss [T])`` from the gradients of pre / post / comb."""
    one = _c(1.0, raw)
    sp = ctl["sig_pre"]
    dlog = [dpre * (sp * (one - sp))]
    if dpost is not None:
        so = ctl["sig_post"]
        dlog.append((dpost * _c(POST_MULT, raw)) * (so * (one - so)))
        g = dcomb.float()
        for kind, m_in, d in reversed(ctl["steps"]):  # type: ignore[union-attr]
            g = _norm_backward(g, m_in, d, kind)
        soft = ctl["soft"]
        gs = _rows(g * soft)[:, :, None]
        dlog.append((soft * (g - gs)).reshape(-1, HC * HC))
    dlogits = torch.cat(dlog, dim=1)
    seg = torch.tensor(_segments(raw.shape[1]), device=raw.device)
    dm = dlogits * scale.float()[seg][None, :]
    r = ctl["r"]
    draw = dm * r[:, None]
    prod = dm * raw
    dr = prod[:, 0]
    for n in range(1, prod.shape[1]):
        dr = dr + prod[:, n]
    # r = (ss / K + eps)^(-1/2), so dr / dss = -0.5 * r^3 / K.
    r3 = (r * r) * r
    dss = ((dr * _c(-0.5, ss)) * r3) / _c(float(k), ss)
    return draw, dss


def _gemm_backward(residual: Tensor, fn: Tensor, draw: Tensor, dss: Tensor) -> Tensor:
    """``dx[k] = (fma chain over n of draw_n * fn[n, k]) + (2 x[k]) * dss`` -> FP32 [T, 4, H]."""
    t = residual.shape[0]
    x = residual.reshape(t, -1).float()
    f = fn.float()
    g = torch.zeros_like(x)
    for n in range(f.shape[0]):
        g = fma_f32(draw[:, n, None], f[None, n, :], g)
    g = g + (x * _c(2.0, x)) * dss[:, None]
    return g.reshape(residual.shape)


def _unnorm(
    dy: Tensor, ctl: dict[str, Tensor], residual: Tensor, norm_weight: Tensor | None
) -> Tensor:
    """Gradient of the stream mix y: through the fused RMSNorm (recomputing y) if any."""
    if norm_weight is None:
        return dy.float()
    y = aggregate(ctl["pre"], residual)
    _, r = rms_norm(y, norm_weight)
    return rms_norm_backward(dy, y, norm_weight, r)


def mhc_pre_backward(
    dy: Tensor,
    d_post: Tensor,
    d_comb: Tensor,
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    raw: Tensor,
    ss: Tensor,
    norm_weight: Tensor | None = None,
) -> Tensor:
    """d_residual [T, 4, H] (FP32) of ``mhc_pre`` given the gradient of its output and of
    post/comb. With ``norm_weight``, ``dy`` is the gradient of the normed output."""
    k = residual.shape[1] * residual.shape[2]
    ctl = controller(raw, ss, scale, base, k)
    dyf = _unnorm(dy, ctl, residual, norm_weight)
    rf = residual.float()
    dpre = slice_dot(rf, dyf[:, None, :])  # [T, 4]
    dres_agg = torch.stack([ctl["pre"][:, i, None] * dyf for i in range(HC)], dim=1)
    draw, dss = _controller_backward(raw, ctl, scale, k, ss, dpre, d_post.float(), d_comb.float())
    return dres_agg + _gemm_backward(residual, fn, draw, dss)


def hc_head_backward(
    dy: Tensor,
    residual: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    raw: Tensor,
    ss: Tensor,
    norm_weight: Tensor | None = None,
) -> Tensor:
    """d_residual [T, 4, H] (FP32) of ``hc_head`` (``dy`` of the normed output if a weight
    is given)."""
    k = residual.shape[1] * residual.shape[2]
    ctl = controller(raw, ss, scale, base, k)
    dyf = _unnorm(dy, ctl, residual, norm_weight)
    rf = residual.float()
    dpre = slice_dot(rf, dyf[:, None, :])
    dres_agg = torch.stack([ctl["pre"][:, i, None] * dyf for i in range(HC)], dim=1)
    draw, dss = _controller_backward(raw, ctl, scale, k, ss, dpre, None, None)
    return dres_agg + _gemm_backward(residual, fn, draw, dss)
