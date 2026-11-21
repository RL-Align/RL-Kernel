# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 mHC (dsv4-mhc-v1): byte-equal to the golden (forward and dX backward),
batch-invariant, and the autograd functions compose like the golden."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch import mhc_dsv4 as g

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def ops():
    from rl_engine.kernels.ops.cuda import mhc_dsv4 as c

    try:
        c.dsv4_mhc._extension()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return c


def _inputs(t, h, seed=0, n=24, mag=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    res = (torch.randn(t, 4, h, generator=gen, device="cuda") * mag).bfloat16()
    fn = torch.randn(n, 4 * h, generator=gen, device="cuda") * (1.0 / (4 * h) ** 0.5)
    scale = torch.rand(3 if n == 24 else 1, generator=gen, device="cuda") + 0.5
    base = torch.randn(n, generator=gen, device="cuda") * 0.5
    return res, fn, scale, base


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    eq = a.view(torch.int16 if a.element_size() == 2 else torch.int32) == b.view(
        torch.int16 if b.element_size() == 2 else torch.int32
    )
    assert bool(eq.all()), f"{(~eq).sum().item()} of {eq.numel()} differ"


CASES = [
    (1, 256, 1.0),
    (5, 256, 30.0),
    (37, 256, 1e-3),
    (4, 1152, 1.0),
    (3, 4096, 1.0),
    (17, 4096, 4.0),
]


def _nw(h, norm, seed=3):
    if not norm:
        return None
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(h, generator=gen, device="cuda") * 2.0 + 0.1).bfloat16()


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("t,h,mag", CASES)
def test_forward_byte_equal(ops, t, h, mag, norm):
    res, fn, scale, base = _inputs(t, h, seed=t + h, mag=mag)
    w = _nw(h, norm)
    got = ops.mhc_pre_cuda(res, fn, scale, base, w)
    want = g.mhc_pre(res, fn, scale, base, w)
    for a, b in zip(got, want):
        _same(a, b)
    x = (torch.randn(t, h, device="cuda") * mag).bfloat16()
    _same(ops.mhc_post_cuda(x, res, got[1], got[2]), g.mhc_post(x, res, want[1], want[2]))
    hres, hfn, hscale, hbase = _inputs(t, h, seed=t + h + 1, n=4, mag=mag)
    for a, b in zip(
        ops.hc_head_cuda(hres, hfn, hscale, hbase, w), g.hc_head(hres, hfn, hscale, hbase, w)
    ):
        _same(a, b)


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("t,h,mag", CASES)
def test_backward_byte_equal(ops, t, h, mag, norm):
    res, fn, scale, base = _inputs(t, h, seed=2 * t + h, mag=mag)
    w = _nw(h, norm)
    y, post, comb, raw, ss = g.mhc_pre(res, fn, scale, base, w)
    gen = torch.Generator(device="cuda").manual_seed(t)
    x = torch.randn(t, h, generator=gen, device="cuda").bfloat16()
    d_new = torch.randn(t, 4, h, generator=gen, device="cuda")
    dy = torch.randn(t, h, generator=gen, device="cuda")
    got = ops.mhc_post_backward_cuda(d_new, x, res, post, comb)
    want = g.mhc_post_backward(d_new, x, res, post, comb)
    for a, b in zip(got, want):
        _same(a, b)
    _same(
        ops.mhc_pre_backward_cuda(dy, want[2], want[3], res, fn, scale, base, raw, ss, w),
        g.mhc_pre_backward(dy, want[2], want[3], res, fn, scale, base, raw, ss, w),
    )
    hres, hfn, hscale, hbase = _inputs(t, h, seed=t + 7, n=4, mag=mag)
    _, hraw, hss = g.hc_head(hres, hfn, hscale, hbase, w)
    _same(
        ops.hc_head_backward_cuda(dy, hres, hfn, hscale, hbase, hraw, hss, w),
        g.hc_head_backward(dy, hres, hfn, hscale, hbase, hraw, hss, w),
    )


@pytest.mark.parametrize("t,h", [(5, 256), (9, 4096)])
def test_backward_bf16_io(ops, t, h):
    """BF16 gradients in and BF16 dX out equal the FP32 golden rounded once."""
    res, fn, scale, base = _inputs(t, h, seed=5 * t + h)
    w = _nw(h, True)
    y, post, comb, raw, ss = g.mhc_pre(res, fn, scale, base, w)
    gen = torch.Generator(device="cuda").manual_seed(t + 1)
    x = torch.randn(t, h, generator=gen, device="cuda").bfloat16()
    d_new = torch.randn(t, 4, h, generator=gen, device="cuda").bfloat16()
    dy = torch.randn(t, h, generator=gen, device="cuda").bfloat16()
    bf = torch.bfloat16
    got = ops.mhc_post_backward_cuda(d_new, x, res, post, comb, out_dtype=bf)
    want = g.mhc_post_backward(d_new.float(), x, res, post, comb)
    _same(got[0], want[0].to(bf))
    _same(got[1], want[1].to(bf))
    _same(got[2], want[2])
    _same(got[3], want[3])
    _same(
        ops.mhc_pre_backward_cuda(dy, want[2], want[3], res, fn, scale, base, raw, ss, w, bf),
        g.mhc_pre_backward(dy.float(), want[2], want[3], res, fn, scale, base, raw, ss, w).to(bf),
    )


def test_batch_invariant(ops):
    res, fn, scale, base = _inputs(70, 4096, seed=9)
    w = _nw(4096, True)
    full = ops.mhc_pre_cuda(res, fn, scale, base, w)
    for lo, hi in ((0, 1), (13, 14), (20, 53)):
        part = ops.mhc_pre_cuda(res[lo:hi].contiguous(), fn, scale, base, w)
        for a, b in zip(full, part):
            _same(a[lo:hi].contiguous(), b)
    again = ops.mhc_pre_cuda(res, fn, scale, base, w)
    for a, b in zip(full, again):
        _same(a, b)


def test_autograd_block_matches_golden(ops):
    t, h = 6, 256
    res, fn, scale, base = _inputs(t, h, seed=11)
    w = torch.randn(h, h, device="cuda").bfloat16() * 0.05
    nw = _nw(h, True)

    r = res.clone().requires_grad_(True)
    y, post, comb = ops.mhc_pre(r, fn, scale, base, nw)
    x = (y.float() @ w.float()).bfloat16()
    new = ops.mhc_post(x, r, post, comb)
    d_new = torch.randn(t, 4, h, device="cuda").bfloat16()
    new.backward(d_new)

    gy, gpost, gcomb, graw, gss = g.mhc_pre(res, fn, scale, base, nw)
    gx = (gy.float() @ w.float()).bfloat16()
    dx, dres_post, dpost, dcomb = g.mhc_post_backward(d_new.float(), gx, res, gpost, gcomb)
    dy = dx.bfloat16().float() @ w.float().t()
    dres_pre = g.mhc_pre_backward(dy, dpost, dcomb, res, fn, scale, base, graw, gss, nw)
    want = (dres_post.bfloat16().float() + dres_pre.bfloat16().float()).bfloat16()
    _same(new.detach(), g.mhc_post(gx, res, gpost, gcomb))
    # autograd sums the two BF16 contributions; compare closely.
    assert (r.grad.float() - want.float()).abs().max() <= 1e-2 * want.float().abs().max()


def test_rejects_bad_shapes(ops):
    ext = ops.dsv4_mhc._extension()

    def call(t, h, w):
        res = torch.zeros(t, 4, h, device="cuda", dtype=torch.bfloat16)
        pm = torch.zeros(16, t, 24, device="cuda")
        ps = torch.zeros(16, t, device="cuda")
        sc, bs = torch.ones(3, device="cuda"), torch.zeros(24, device="cuda")
        return ext.dsv4_mhc_pre(res, pm, ps, sc, bs, 1e-6, 1e-6, 20, 2.0, w, 1e-6)

    call(2, 256, torch.ones(256, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="norm_weight"):
        call(2, 256, torch.ones(256, device="cuda"))
    with pytest.raises(RuntimeError, match="at most 4096"):
        call(1, 4224, None)


def test_empty(ops):
    res, fn, scale, base = _inputs(0, 256)
    y, post, comb, raw, ss = ops.mhc_pre_cuda(res, fn, scale, base)
    assert y.shape == (0, 256) and comb.shape == (0, 4, 4)
