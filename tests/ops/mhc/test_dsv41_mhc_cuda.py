# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4.1 mHC (dsv41-mhc-v1): the delayed pre and the final collapse are byte-equal to
the golden (forward and dX backward) up to H = 5120, batch-invariant, agree with vLLM's
reference semantics, and compose under autograd; DSv4's kernels at H = 5120 too."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch import mhc as g
from rl_engine.kernels.ops.pytorch.mhc import dsv41_mhc as g41

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def ops():
    from rl_engine.kernels.ops.cuda.mhc import dsv4_mhc as c4
    from rl_engine.kernels.ops.cuda.mhc import dsv41_mhc as c41

    try:
        c41._extension()
        c4._extension()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return c41, c4


def _inputs(t, h, seed=0, mag=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    res = (torch.randn(t, 4, h, generator=gen, device="cuda") * mag).bfloat16()
    fn = torch.randn(24, 4 * h, generator=gen, device="cuda") * (1.0 / (4 * h) ** 0.5)
    scale = torch.rand(3, generator=gen, device="cuda") + 0.5
    base = torch.randn(24, generator=gen, device="cuda") * 0.5
    pre_in = torch.rand(t, 4, generator=gen, device="cuda") + 1e-6
    return res, fn, scale, base, pre_in


def _nw(h, norm, seed=3):
    if not norm:
        return None
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(h, generator=gen, device="cuda") * 2.0 + 0.1).bfloat16()


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    eq = a.view(torch.int16 if a.element_size() == 2 else torch.int32) == b.view(
        torch.int16 if b.element_size() == 2 else torch.int32
    )
    assert bool(eq.all()), f"{(~eq).sum().item()} of {eq.numel()} differ"


CASES = [
    (1, 256, 1.0),
    (5, 1152, 30.0),
    (3, 4096, 1.0),
    (7, 5120, 1.0),
    (17, 5120, 4.0),
    (33, 5120, 1e-3),
]


@pytest.mark.parametrize("eps", [1e-20, 1e-6])  # DSv4.1's norm_eps, DSv4's
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("t,h,mag", CASES)
def test_forward_byte_equal(ops, t, h, mag, norm, carried, eps):
    c41, _ = ops
    res, fn, scale, base, pre_in = _inputs(t, h, seed=t + h, mag=mag)
    p = pre_in if carried else None
    w = _nw(h, norm)
    got = c41.mhc_pre_delayed_cuda(res, fn, scale, base, p, w, eps)
    want = g41.mhc_pre_delayed(res, fn, scale, base, p, w, eps)
    assert len(got) == len(want) == 6
    for a, b in zip(got, want):
        _same(a, b)
    _same(c41.hc_collapse_cuda(res, pre_in, w, eps), g41.hc_collapse(res, pre_in, w, eps))


def test_stream0_is_a_copy(ops):
    c41, _ = ops
    res, fn, scale, base, _ = _inputs(9, 5120, seed=1)
    _same(c41.mhc_pre_delayed_cuda(res, fn, scale, base)[0], res[:, 0].contiguous())


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("h", [256, 4096])
def test_own_pre_carried_in_equals_dsv4(ops, h, norm):
    """Feeding the controller's own pre back as pre_in gives dsv4-mhc-v1's bytes (at DSv4's
    eps)."""
    c41, c4 = ops
    res, fn, scale, base, _ = _inputs(11, h, seed=h)
    w = _nw(h, norm)
    y4, post4, comb4, raw4, ss4 = c4.mhc_pre_cuda(res, fn, scale, base, w)
    pre = g41.mhc_pre_delayed(res, fn, scale, base, None, w, 1e-6)[3]
    y, post, comb, pre_out, raw, ss = c41.mhc_pre_delayed_cuda(res, fn, scale, base, pre, w, 1e-6)
    for a, b in ((y, y4), (post, post4), (comb, comb4), (raw, raw4), (ss, ss4), (pre_out, pre)):
        _same(a, b)


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("t,h,mag", CASES)
def test_backward_byte_equal(ops, t, h, mag, norm, carried):
    c41, _ = ops
    res, fn, scale, base, pre_in = _inputs(t, h, seed=2 * t + h, mag=mag)
    p = pre_in if carried else None
    w = _nw(h, norm)
    _, _, _, _, raw, ss = g41.mhc_pre_delayed(res, fn, scale, base, p, w)
    gen = torch.Generator(device="cuda").manual_seed(t)
    dy = torch.randn(t, h, generator=gen, device="cuda")
    dpost = torch.randn(t, 4, generator=gen, device="cuda")
    dcomb = torch.randn(t, 4, 4, generator=gen, device="cuda")
    dpre = torch.randn(t, 4, generator=gen, device="cuda")
    got = c41.mhc_pre_delayed_backward_cuda(
        dy, dpost, dcomb, dpre, res, fn, scale, base, raw, ss, p, w
    )
    want = g41.mhc_pre_delayed_backward(dy, dpost, dcomb, dpre, res, fn, scale, base, raw, ss, p, w)
    _same(got[0], want[0])
    if carried:
        _same(got[1], want[1])
    else:
        assert got[1] is None and want[1] is None
    for a, b in zip(
        c41.hc_collapse_backward_cuda(dy, res, pre_in, w),
        g41.hc_collapse_backward(dy, res, pre_in, w),
    ):
        _same(a, b)


@pytest.mark.parametrize("t,h", [(5, 5120), (130, 5120)])
def test_dsv4_kernels_at_5120(ops, t, h):
    """mhc_post (shared by DSv4.1) and its backward, and DSv4's pre / head at H = 5120."""
    _, c4 = ops
    res, fn, scale, base, _ = _inputs(t, h, seed=t)
    w = _nw(h, True)
    got = c4.mhc_pre_cuda(res, fn, scale, base, w)
    want = g.mhc_pre(res, fn, scale, base, w)
    for a, b in zip(got, want):
        _same(a, b)
    x = torch.randn(t, h, device="cuda").bfloat16()
    _same(c4.mhc_post_cuda(x, res, got[1], got[2]), g.mhc_post(x, res, want[1], want[2]))
    d_new = torch.randn(t, 4, h, device="cuda")
    for a, b in zip(
        c4.mhc_post_backward_cuda(d_new, x, res, want[1], want[2]),
        g.mhc_post_backward(d_new, x, res, want[1], want[2]),
    ):
        _same(a, b)
    hfn = torch.randn(4, 4 * h, device="cuda") * (1.0 / (4 * h) ** 0.5)
    hscale, hbase = torch.rand(1, device="cuda") + 0.5, torch.randn(4, device="cuda") * 0.5
    for a, b in zip(
        c4.hc_head_cuda(res, hfn, hscale, hbase, w), g.hc_head(res, hfn, hscale, hbase, w)
    ):
        _same(a, b)


@pytest.mark.parametrize("carried", [False, True])
def test_backward_bf16_io(ops, carried):
    """BF16 gradients in and BF16 dX out equal the FP32 golden rounded once."""
    c41, _ = ops
    t, h = 6, 5120
    res, fn, scale, base, pre_in = _inputs(t, h, seed=21)
    p = pre_in if carried else None
    w = _nw(h, True)
    _, _, _, _, raw, ss = g41.mhc_pre_delayed(res, fn, scale, base, p, w)
    gen = torch.Generator(device="cuda").manual_seed(4)
    dy = torch.randn(t, h, generator=gen, device="cuda").bfloat16()
    dpost = torch.randn(t, 4, generator=gen, device="cuda")
    dcomb = torch.randn(t, 4, 4, generator=gen, device="cuda")
    dpre = torch.randn(t, 4, generator=gen, device="cuda")
    bf = torch.bfloat16
    got = c41.mhc_pre_delayed_backward_cuda(
        dy, dpost, dcomb, dpre, res, fn, scale, base, raw, ss, p, w, out_dtype=bf
    )
    want = g41.mhc_pre_delayed_backward(
        dy.float(), dpost, dcomb, dpre, res, fn, scale, base, raw, ss, p, w
    )
    _same(got[0], want[0].to(bf))
    got_c = c41.hc_collapse_backward_cuda(dy, res, pre_in, w, out_dtype=bf)
    want_c = g41.hc_collapse_backward(dy.float(), res, pre_in, w)
    _same(got_c[0], want_c[0].to(bf))
    _same(got_c[1], want_c[1])


def test_batch_invariant(ops):
    c41, _ = ops
    res, fn, scale, base, pre_in = _inputs(70, 5120, seed=9)
    w = _nw(5120, True)
    for p in (None, pre_in):
        full = c41.mhc_pre_delayed_cuda(res, fn, scale, base, p, w)
        for lo, hi in ((0, 1), (13, 14), (20, 53)):
            sub = None if p is None else p[lo:hi].contiguous()
            part = c41.mhc_pre_delayed_cuda(res[lo:hi].contiguous(), fn, scale, base, sub, w)
            for a, b in zip(full, part):
                _same(a[lo:hi].contiguous(), b)
    full = c41.hc_collapse_cuda(res, pre_in, w)
    _same(full[20:53], c41.hc_collapse_cuda(res[20:53].contiguous(), pre_in[20:53], w))


def _vllm_reference(residual, fn, scale, base, pre_mix):
    """vLLM's ``mhc_pre_delayed_torch`` (``vllm/model_executor/kernels/mhc/torch.py``) in
    float64, plus the BF16 layer input."""
    hc = 4
    x = residual.flatten(1).double()
    mixes = (x @ fn.double().t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    s, b = scale.double(), base.double()
    pre = torch.sigmoid(mixes[:, :hc] * s[0] + b[:hc]) + 1e-6
    post = torch.sigmoid(mixes[:, hc : 2 * hc] * s[1] + b[hc : 2 * hc]) * 2.0
    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * s[2] + b[2 * hc :].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    if pre_mix is None:
        y = residual[:, 0]
    else:
        y = (pre_mix.double().unsqueeze(-1) * residual.double()).sum(dim=1).bfloat16()
    return y, post, comb, pre


@pytest.mark.parametrize("carried", [False, True])
def test_matches_vllm_reference(ops, carried):
    c41, _ = ops
    t, h = 16, 5120
    res, fn, scale, base, pre_in = _inputs(t, h, seed=5)
    p = pre_in if carried else None
    y, post, comb, pre_out, _, _ = c41.mhc_pre_delayed_cuda(res, fn, scale, base, p)
    ry, rpost, rcomb, rpre = _vllm_reference(res, fn, scale, base, p)
    # TF32 controller GEMM: a few 1e-4 relative on the coefficients; y within one BF16 ulp.
    torch.testing.assert_close(post.double(), rpost, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(comb.double(), rcomb, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(pre_out.double(), rpre, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(y.float(), ry.float(), rtol=1e-2, atol=1e-6)


def test_autograd_chain_matches_golden(ops):
    """pre (carried pre in) -> linear -> post -> collapse with the pre it returned."""
    c41, c4 = ops
    t, h = 6, 256
    res, fn, scale, base, pre_in = _inputs(t, h, seed=11)
    lin = torch.randn(h, h, device="cuda").bfloat16() * 0.05
    nw, nw2 = _nw(h, True), _nw(h, True, seed=4)
    dout = torch.randn(t, h, device="cuda").bfloat16()

    r = res.clone().requires_grad_(True)
    p0 = pre_in.clone().requires_grad_(True)
    x, post, comb, pre1 = c41.mhc_pre_delayed(r, fn, scale, base, p0, nw)
    z = (x.float() @ lin.float()).bfloat16()
    new = c4.mhc_post(z, r, post, comb)
    out = c41.hc_collapse(new, pre1, nw2)
    out.backward(dout)

    gx, gpost, gcomb, gpre1, graw, gss = g41.mhc_pre_delayed(res, fn, scale, base, pre_in, nw)
    gz = (gx.float() @ lin.float()).bfloat16()
    gnew = g.mhc_post(gz, res, gpost, gcomb)
    _same(out.detach(), g41.hc_collapse(gnew, gpre1, nw2))
    d_new, d_pre1 = g41.hc_collapse_backward(dout.float(), gnew, gpre1, nw2)
    dz, dres_post, dpost, dcomb = g.mhc_post_backward(
        d_new.bfloat16().float(), gz, res, gpost, gcomb
    )
    dy = (dz.bfloat16().float() @ lin.float().t()).bfloat16()
    dres_pre, dp0 = g41.mhc_pre_delayed_backward(
        dy.float(), dpost, dcomb, d_pre1, res, fn, scale, base, graw, gss, pre_in, nw
    )
    _same(p0.grad, dp0)
    want = (dres_post.bfloat16().float() + dres_pre.bfloat16().float()).bfloat16()
    # autograd sums the two BF16 contributions; compare closely.
    assert (r.grad.float() - want.float()).abs().max() <= 1e-2 * want.float().abs().max()


def test_rejects_bad_shapes(ops):
    c41, _ = ops
    ext = c41._extension()

    def call(t, h, pre_in):
        res = torch.zeros(t, 4, h, device="cuda", dtype=torch.bfloat16)
        pm = torch.zeros(16, t, 24, device="cuda")
        ps = torch.zeros(16, t, device="cuda")
        sc, bs = torch.ones(3, device="cuda"), torch.zeros(24, device="cuda")
        return ext.dsv41_mhc_pre(res, pm, ps, sc, bs, 1e-6, 1e-6, 20, 2.0, pre_in, None, 1e-6)

    call(2, 5120, torch.ones(2, 4, device="cuda"))
    with pytest.raises(RuntimeError, match="pre_in"):
        call(2, 256, torch.ones(3, 4, device="cuda"))
    with pytest.raises(RuntimeError, match="at most 5120"):
        call(1, 5248, None)


def test_empty(ops):
    c41, _ = ops
    res, fn, scale, base, pre_in = _inputs(0, 5120)
    y, post, comb, pre_out, raw, ss = c41.mhc_pre_delayed_cuda(res, fn, scale, base, pre_in)
    assert y.shape == (0, 5120) and pre_out.shape == (0, 4)
    assert c41.hc_collapse_cuda(res, pre_in).shape == (0, 5120)
