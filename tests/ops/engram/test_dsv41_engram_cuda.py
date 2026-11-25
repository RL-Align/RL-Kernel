# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4.1 Engram gate (dsv41-engram-v1): forward byte-equal to the golden and
batch-invariant up to H = 5120; the dX backward matches FP64 autograd; masked tokens pass
through."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch import engram as g

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def ops():
    from rl_engine.kernels.ops.cuda import engram as c

    try:
        c.dsv41_engram._extension()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return c


def _inputs(t, h, seed=0, mag=1.0, kmag=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(t, 4, h, generator=gen, device="cuda") * mag).bfloat16()
    kv = torch.randn(t, 5 * h, generator=gen, device="cuda") * kmag
    q = (torch.randn(4, h, generator=gen, device="cuda") * 0.5 + 1).bfloat16()
    k = (torch.randn(4, h, generator=gen, device="cuda") * 0.5 + 1).bfloat16()
    mask = torch.rand(t, generator=gen, device="cuda") > 0.2
    return x, kv, q, k, mask


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    view = torch.int16 if a.element_size() == 2 else torch.int32
    eq = a.view(view) == b.view(view)
    assert bool(eq.all()), f"{(~eq).sum().item()} of {eq.numel()} differ"


CASES = [(1, 256, 1.0, 1.0), (7, 1152, 30.0, 0.1), (5, 4096, 1.0, 1.0), (19, 5120, 1e-3, 5.0)]


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("t,h,mag,kmag", CASES)
def test_forward_byte_equal(ops, t, h, mag, kmag, masked):
    x, kv, q, k, mask = _inputs(t, h, seed=t + h, mag=mag, kmag=kmag)
    m = mask if masked else None
    got = ops.engram_gate_cuda(x, kv, q, k, m)
    want = g.engram_gate(x, kv, q, k, m)
    _same(got[0], want[0])
    _same(got[1], want[1])
    if masked:
        _same(got[0][~mask], x[~mask])
        assert bool((got[1][~mask] == 0).all())


def test_batch_invariant(ops):
    x, kv, q, k, mask = _inputs(64, 5120, seed=3)
    full = ops.engram_gate_cuda(x, kv, q, k, mask)
    for lo, hi in ((0, 1), (17, 18), (20, 53)):
        part = ops.engram_gate_cuda(
            x[lo:hi].contiguous(), kv[lo:hi].contiguous(), q, k, mask[lo:hi].contiguous()
        )
        _same(full[0][lo:hi], part[0])
        _same(full[1][lo:hi], part[1])


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("t,h", [(6, 256), (9, 5120)])
def test_backward_matches_fp64(ops, t, h, masked):
    x, kv, q, k, mask = _inputs(t, h, seed=2 * t + h)
    m = mask if masked else None
    dout = torch.randn(t, 4, h, device="cuda")
    got = ops.engram_gate_backward_cuda(dout, x, kv, q, k, m)
    ref = g.engram_gate_backward_reference(dout, x, kv, q, k, m)
    err = (got.double() - ref).abs().max() / ref.abs().max()
    assert float(err) < 1e-5, float(err)
    if masked:
        _same(got[~mask], dout[~mask])


def test_autograd(ops):
    x, kv, q, k, mask = _inputs(4, 512, seed=8)
    xr = x.clone().requires_grad_(True)
    out = ops.engram_gate(xr, kv, q, k, mask)
    dout = torch.randn_like(out)
    out.backward(dout)
    _same(out.detach(), g.engram_gate(x, kv, q, k, mask)[0])
    want = ops.engram_gate_backward_cuda(dout, x, kv, q, k, mask, out_dtype=torch.bfloat16)
    _same(xr.grad, want)


def test_rejects_bad_shapes(ops):
    x, kv, q, k, _ = _inputs(2, 256)
    with pytest.raises(RuntimeError, match="5 H"):
        ops.engram_gate_cuda(x, kv[:, :-128].contiguous(), q, k)
    xb = torch.zeros(1, 4, 5248, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="at most 5120"):
        ops.engram_gate_cuda(xb, torch.zeros(1, 5 * 5248, device="cuda"), q, k)
