# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 derivatives, then exact repeatability on a single device."""

import os

import pytest
import torch

from rl_engine.ops.mhc.training import mhc_head, mhc_post, mhc_pre, mhc_pre_norm


def inputs(t, d=5, device="cpu"):
    rng = torch.Generator().manual_seed(91)

    def make(shape, dtype):
        return torch.randn(shape, generator=rng).to(device=device, dtype=dtype).requires_grad_()

    return [
        make((t, 4, d), torch.bfloat16),
        make((24, 4 * d), torch.float),
        make((3,), torch.float),
        make((24,), torch.float),
        make((d,), torch.float),
    ]


def oracle(row, fn, scale, base, rounds, head=False):
    flat = row.flatten(1)
    # Quantize the master for the value and differentiate w.r.t. its effective FP32 value.
    mix = (flat @ fn.T) * (flat.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    if head:
        w = (mix * scale + base).sigmoid() + 1e-6
        return (w.unsqueeze(-1) * row).sum(1)
    a, b, c = mix.split((4, 4, 16), -1)
    w = (a * scale[0] + base[:4]).sigmoid() + 1e-6
    post = 2 * (b * scale[1] + base[4:8]).sigmoid()
    z = (c * scale[2] + base[8:]).reshape(-1, 4, 4)
    if rounds:
        z = z.softmax(-1) + 1e-6
        z = z / (z.sum(-2, keepdim=True) + 1e-6)
        for _ in range(rounds - 1):
            z = z / (z.sum(-1, keepdim=True) + 1e-6)
            z = z / (z.sum(-2, keepdim=True) + 1e-6)
    return post, z, (w.unsqueeze(-1) * row).sum(1)


@pytest.mark.parametrize("rounds", [0, 1, 20])
@pytest.mark.parametrize("op", ["pre", "head", "post"])
def test_independent_math(op, rounds):
    x, w, s, b, g = inputs(2)
    if op == "post":
        args = (
            x[:, 0, :].detach().requires_grad_(),
            x,
            b[:8].reshape(2, 4),
            w[:2, :16].reshape(2, 4, 4),
        )
    elif op == "head":
        args = (x, w[:4], s[:1], b[:4])
    else:
        args = (x, w, s, b)
    ref = [v.detach().double().requires_grad_() for v in args]
    if op == "post":
        y, r, p, c = ref
        want = ((y.unsqueeze(1) * p.unsqueeze(-1)) + torch.einsum("thc,thj->tjc", r, c),)
        out = (mhc_post(*args, backend="reference"),)
    else:
        if op == "head":
            want = (oracle(*ref, rounds, True),)
            out = (mhc_head(*args, backend="reference"),)
        else:
            want = oracle(*ref, rounds)
            out = mhc_pre(*args, backend="reference", sinkhorn_repeat=rounds)
    rng = torch.Generator().manual_seed(46)
    cot = [torch.randn(o.shape, generator=rng).to(o.dtype) for o in out]
    actual = torch.autograd.grad(out, args, cot)
    expected = torch.autograd.grad(want, ref, [v.double() for v in cot])
    for vals in (zip(out, want, strict=True), zip(actual, expected, strict=True)):
        for got, ideal in vals:
            tol = 0.005 if got.dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(got.double(), ideal, rtol=tol, atol=tol)


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def exercise(backend, device, d):
    x, w, s, b, g = inputs(17, d, device)

    def run(rows, recompute):
        p, c, y, h = mhc_pre_norm(
            rows, w, s, b, g, backend=backend, recompute=recompute, sinkhorn_repeat=2
        )
        z = mhc_post(y, rows, p, c, backend=backend)
        out = mhc_head(z, w[:4], s[:1], b[:4], backend=backend, recompute=recompute)
        ds = torch.autograd.grad(
            (out, h), (rows, w, s, b, g), (torch.ones_like(out), torch.ones_like(h))
        )
        return (p, c, y, h, z, out, *ds)

    baseline = run(x, False)
    for mode in (False, True):
        for a, bv in zip(baseline, run(x, mode), strict=True):
            bits(a, bv)
    for count, pos in ((1, 0), (16, 5), (17, 16)):
        rows = x[:count].detach().clone()
        rows[pos] = x[0].detach()
        other = run(rows.requires_grad_(), True)
        for a, bv in zip(baseline[:7], other[:7], strict=True):
            bits(a[0], bv[pos])
    empty = run(x[:0], True)
    assert all(not t.count_nonzero() for t in empty)
    with torch.no_grad():
        b.add_(0.25)
    assert not torch.equal(baseline[0], run(x, False)[0])


def test_composed_save_recompute_and_updates():
    exercise("reference", "cpu", 5)


@pytest.mark.parametrize("mode", [False, True])
def test_mutation_guard(mode):
    x, w, s, b, _ = inputs(2)
    out = mhc_head(x, w[:4], s[:1], b[:4], backend="reference", recompute=mode)
    with torch.no_grad():
        w.add_(1)
    with pytest.raises(RuntimeError, match="modified"):
        out.sum().backward()


def require_rocm():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("ROCm device required")
        pytest.skip("requires ROCm")


def test_rocm_composed():
    require_rocm()
    exercise("rocm", "cuda", 65)


def test_parameter_gradients_fold_rows_in_ascending_order():
    row, fn, scale, base, _ = inputs(3)
    args = (fn[:4], scale[:1], base[:4])
    row = row[:1].expand_as(row)
    cot = torch.tensor([[1e8] * 5, [-1e8] * 5, [1.0] * 5], dtype=torch.bfloat16)

    def params(x, dy):
        h = mhc_head(x, *args, backend="reference")
        return torch.autograd.grad(h, args, dy)

    expected = [torch.zeros_like(p) for p in args]
    partials = [params(row[i : i + 1], cot[i : i + 1]) for i in range(3)]
    for partial in partials:
        for acc, v in zip(expected, partial, strict=True):
            acc.add_(v)
    reverse = [(c + b) + a for a, b, c in zip(*partials, strict=True)]
    assert any(not torch.equal(a, b) for a, b in zip(expected, reverse, strict=True))
    for a, b in zip(expected, params(row, cot), strict=True):
        bits(a, b)


def test_unsupported_device_fails_before_dispatch():
    row, fn, scale, base, _ = inputs(0)
    with pytest.raises(ValueError, match="unsupported"):
        mhc_pre(row, fn, scale, base, backend="rocm")
