# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden DSv4 mHC (dsv4-mhc-v1): FMA emulation, closeness to DeepSeek's reference
semantics in float64 (forward and autograd dX), and batch invariance."""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import pytest
import torch

from rl_engine.kernels.ops.pytorch import mhc_dsv4 as g

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _neighbors(x: float) -> tuple[float, float]:
    t = torch.tensor(x, dtype=torch.float32)
    return (
        torch.nextafter(t, torch.tensor(-math.inf)).item(),
        torch.nextafter(t, torch.tensor(math.inf)).item(),
    )


def _is_correctly_rounded(r: float, exact: Fraction) -> bool:
    lo, hi = _neighbors(r)
    d = abs(Fraction(r) - exact)
    for n in (lo, hi):
        dn = abs(Fraction(n) - exact)
        if dn < d:
            return False
        if dn == d:  # tie: r must be the even one
            bits = struct.unpack("I", struct.pack("f", r))[0]
            if bits & 1:
                return False
    return True


def test_fma_f32_exact_rounding():
    """Random FMAs, plus true double-rounding ties: a*b + c lands on an FP32 midpoint in
    double while the exact sum is just off it (or exactly on it, for round-to-even)."""
    gen = torch.Generator().manual_seed(2)
    n = 2_000_000
    a = torch.randn(n, generator=gen)
    b = torch.randn(n, generator=gen)
    p = a.double() * b.double()
    q = p.float()
    up = torch.nextafter(q, torch.full_like(q, math.inf))
    dn = torch.nextafter(q, torch.full_like(q, -math.inf))
    mid = torch.where(
        p > q.double(), (q.double() + up.double()) / 2, (q.double() + dn.double()) / 2
    )
    gap = mid - p
    exact_gap = (gap.float().double() == gap) & (gap != 0)
    cases = [(a[:2000], b[:2000], torch.randn(2000, generator=gen) * 1e-2)]
    hits = 0
    for f in (1 + 2.0**-20, 1 - 2.0**-20, 1.0):
        c = (gap * f).float()
        idx = torch.nonzero(exact_gap & (c.double() == gap * f)).flatten()[:300]
        hits += len(idx)
        cases.append((a[idx], b[idx], c[idx]))
    assert hits > 300
    for aa, bb, cc in cases:
        out = g.fma_f32(aa, bb, cc)
        for i in range(0, len(aa), 3):
            exact = Fraction(aa[i].item()) * Fraction(bb[i].item()) + Fraction(cc[i].item())
            assert _is_correctly_rounded(out[i].item(), exact), (
                aa[i].item(),
                bb[i].item(),
                cc[i].item(),
            )


def _ref(residual, fn, scale, base, hc_head=False):
    """DeepSeek's hc_pre / hc_head in float64 (rsqrt form), for closeness only."""
    t = residual.shape[0]
    x = residual.double().reshape(t, -1)
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + g.RMS_EPS)
    mixes = (x @ fn.double().t()) * rs
    if hc_head:
        pre = torch.sigmoid(mixes * scale.double() + base.double()) + g.HC_EPS
        return (pre[:, :, None] * residual.double()).sum(1)
    pre = torch.sigmoid(mixes[:, :4] * scale[0].double() + base[:4].double()) + g.HC_EPS
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1].double() + base[4:8].double())
    comb = mixes[:, 8:].reshape(t, 4, 4) * scale[2].double() + base[8:].double().reshape(1, 4, 4)
    comb = comb.softmax(-1) + g.HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + g.HC_EPS)
    for _ in range(g.SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + g.HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + g.HC_EPS)
    y = (pre[:, :, None] * residual.double()).sum(1)
    return y, post, comb


def _partials(res, fn):
    """Synthetic GEMM partials (16 K chunks, FP32 matmul), padded to a multiple of 8 rows."""
    t = res.shape[0]
    x = res.reshape(t, -1).float()
    n_pad = (fn.shape[0] + 7) // 8 * 8
    f = torch.cat([fn.float(), fn.new_zeros(n_pad - fn.shape[0], fn.shape[1])])
    xc = x.reshape(t, g.SPLITS, -1).permute(1, 0, 2)
    fc = f.reshape(n_pad, g.SPLITS, -1).permute(1, 0, 2)
    return torch.bmm(xc, fc.transpose(1, 2)), (xc * xc).sum(-1)


def _pre(res, fn, scale, base):
    return g.mhc_pre_from_partials(res, *_partials(res, fn), scale, base)


def _head(res, fn, scale, base):
    return g.hc_head_from_partials(res, *_partials(res, fn), scale, base)


def _inputs(t, h, device, seed=0, n=24):
    gen = torch.Generator(device=device).manual_seed(seed)
    res = torch.randn(t, 4, h, generator=gen, device=device).bfloat16()
    fn = torch.randn(n, 4 * h, generator=gen, device=device) * (1.0 / (4 * h) ** 0.5)
    scale = torch.rand(3 if n == 24 else 1, generator=gen, device=device) + 0.5
    base = torch.randn(n, generator=gen, device=device) * 0.5
    return res, fn, scale, base


@pytest.mark.parametrize("device", DEVICES)
def test_forward_close_to_reference(device):
    res, fn, scale, base = _inputs(5, 256, device)
    y, post, comb, raw, ss = _pre(res, fn, scale, base)
    ry, rpost, rcomb = _ref(res, fn, scale, base)
    assert y.dtype == torch.bfloat16 and y.shape == (5, 256)
    assert (post.double() - rpost).abs().max() < 1e-5
    assert (comb.double() - rcomb).abs().max() < 1e-5
    assert ((y.double() - ry).abs() / (ry.abs() + 1e-2)).max() < 1e-2
    x = torch.randn(5, 256, device=device).bfloat16()
    new = g.mhc_post(x, res, post, comb)
    rnew = post.double()[:, :, None] * x.double()[:, None, :] + torch.einsum(
        "tij,tih->tjh", comb.double(), res.double()
    )
    assert ((new.double() - rnew).abs() / (rnew.abs() + 1e-2)).max() < 1e-2
    hres, hfn, hscale, hbase = _inputs(5, 256, device, seed=1, n=4)
    hy, _, _ = _head(hres, hfn, hscale, hbase)
    rhy = _ref(hres, hfn, hscale, hbase, hc_head=True)
    assert ((hy.double() - rhy).abs() / (rhy.abs() + 1e-2)).max() < 1e-2


@pytest.mark.parametrize("device", DEVICES)
def test_backward_close_to_autograd(device):
    res, fn, scale, base = _inputs(3, 128, device, seed=2)
    y, post, comb, raw, ss = _pre(res, fn, scale, base)
    x = torch.randn(3, 128, device=device).bfloat16()
    gen = torch.Generator(device=device).manual_seed(3)
    d_new = torch.randn(3, 4, 128, generator=gen, device=device)
    dy_layer = torch.randn(3, 128, generator=gen, device=device)
    # golden: post backward, then pre backward with dy from the layer
    dx, dres_post, dpost, dcomb = g.mhc_post_backward(d_new, x, res, post, comb)
    dres_pre = g.mhc_pre_backward(dy_layer, dpost, dcomb, res, fn, scale, base, raw, ss)

    rr = res.double().requires_grad_(True)
    xx = x.double().requires_grad_(True)
    ry, rpost, rcomb = _ref(rr, fn, scale, base)
    rnew = rpost[:, :, None] * xx[:, None, :] + torch.einsum("tij,tih->tjh", rcomb, rr)
    loss = (rnew * d_new.double()).sum() + (ry * dy_layer.double()).sum()
    loss.backward()
    want_res = rr.grad
    got_res = dres_post.double() + dres_pre.double()
    rel = (got_res - want_res).abs().max() / want_res.abs().max()
    assert rel < 1e-4, rel
    relx = (dx.double() - xx.grad).abs().max() / xx.grad.abs().max()
    assert relx < 1e-5, relx

    hres, hfn, hscale, hbase = _inputs(3, 128, device, seed=4, n=4)
    _, hraw, hss = _head(hres, hfn, hscale, hbase)
    hdy = torch.randn(3, 128, generator=gen, device=device)
    got = g.hc_head_backward(hdy, hres, hfn, hscale, hbase, hraw, hss)
    hr = hres.double().requires_grad_(True)
    (_ref(hr, hfn, hscale, hbase, hc_head=True) * hdy.double()).sum().backward()
    relh = (got.double() - hr.grad).abs().max() / hr.grad.abs().max()
    assert relh < 1e-4, relh


@pytest.mark.parametrize("device", DEVICES)
def test_batch_invariant(device):
    res, fn, scale, base = _inputs(6, 128, device, seed=5)
    full = _pre(res, fn, scale, base)
    part = _pre(res[2:4].contiguous(), fn, scale, base)
    for a, b in zip(full, part):
        assert torch.equal(a[2:4], b)


def _ref_norm(y, w):
    yd = y.double()
    return yd * torch.rsqrt(yd.square().mean(-1, keepdim=True) + g.NORM_EPS) * w.double()


@pytest.mark.parametrize("device", DEVICES)
def test_fused_norm_forward_and_backward(device):
    res, fn, scale, base = _inputs(4, 256, device, seed=21)
    gen = torch.Generator(device=device).manual_seed(22)
    w = (torch.rand(256, generator=gen, device=device) + 0.5).bfloat16()
    out, post, comb, raw, ss = g.mhc_pre_from_partials(res, *_partials(res, fn), scale, base, w)
    y = _pre(res, fn, scale, base)[0]
    ref = _ref_norm(y, w)
    assert out.dtype == torch.bfloat16
    assert ((out.double() - ref).abs() / (ref.abs() + 1e-2)).max() < 1e-2
    # backward through the norm: d_residual against float64 autograd of the whole chain
    dz = torch.randn(4, 256, generator=gen, device=device)
    zero_p = torch.zeros(4, 4, device=device)
    zero_c = torch.zeros(4, 4, 4, device=device)
    got = g.mhc_pre_backward(dz, zero_p, zero_c, res, fn, scale, base, raw, ss, norm_weight=w)
    rr = res.double().requires_grad_(True)
    ry, _, _ = _ref(rr, fn, scale, base)
    (_ref_norm(ry, w) * dz.double()).sum().backward()
    rel = (got.double() - rr.grad).abs().max() / rr.grad.abs().max()
    assert rel < 1e-2, rel
