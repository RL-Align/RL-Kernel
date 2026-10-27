# SPDX-License-Identifier: Apache-2.0
"""Run with RLK_REQUIRE_ROCM=1 on each target GPU to disallow skipped acceptance."""

import os

import pytest
import torch

from rl_engine.backends.rocm.moe.clamp_swiglu import RocmClampSwiGLUWeighted as Backend
from rl_engine.ops.moe.clamp_swiglu import clamp_swiglu_weighted, clamp_swiglu_weighted_packed
from rl_engine.reference.moe.clamp_swiglu import ClampSwiGLUReference as Reference


@pytest.fixture(scope="module", autouse=True)
def require_device():
    available = bool(torch.version.hip and torch.cuda.is_available())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if available else "none"
    if arch not in ("gfx942", "gfx950"):
        reason = f"task 46 acceptance needs gfx942 or gfx950, found {arch}"
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def _bytes_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _evaluate(g, u, p, dh):
    h, state = Backend.forward(g, u, p)
    dg, du = Backend.backward(dh, g, u, p, state)
    return h, state.sigmoid, state.silu, state.unweighted, dg, du


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 31, 33, 257, 2048, 4096])
@pytest.mark.parametrize("weighted", [False, True])
def test_same_row_bytes_across_batches_positions_and_repeats(dtype, width, weighted):
    rng = torch.Generator().manual_seed(460)
    row = (torch.randn(1, 2 * width, generator=rng) * 8).to(device="cuda", dtype=dtype)
    dh = torch.randn(1, width, generator=rng).to(device="cuda", dtype=dtype)
    p = torch.tensor([0.37], device="cuda") if weighted else None
    expected = _evaluate(*row.chunk(2, dim=1), p, dh)
    for m in (1, 16, 17, 64, 65):
        for pos in sorted({0, m // 2, m - 1}):
            x = (torch.randn(m, 2 * width, generator=rng) * 4).to(device="cuda", dtype=dtype)
            x[pos] = row[0]
            grad = torch.randn(m, width, generator=rng).to(device="cuda", dtype=dtype)
            grad[pos] = dh[0]
            weights = torch.rand(m, generator=rng).cuda() if weighted else None
            if weighted:
                weights[pos] = p[0]
            actual = _evaluate(*x.chunk(2, dim=1), weights, grad)
            repeated = _evaluate(*x.chunk(2, dim=1), weights, grad)
            for want, got, again in zip(expected, actual, repeated):
                _bytes_equal(want, got[pos : pos + 1])
                _bytes_equal(got, again)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weighted", [False, True])
def test_numerical_accuracy_and_clamp_boundaries(dtype, weighted):
    # FP64 staged math checks accuracy; batch sweeps above check exact bytes separately.
    rng = torch.Generator().manual_seed(461)
    g = (torch.randn(4, 257, generator=rng) * 8).to(dtype)
    u = (torch.randn(4, 257, generator=rng) * 8).to(dtype)
    g[:, :5] = torch.tensor([9.5, 10.0, 10.5, -10.0, 0.0]).to(dtype)
    u[:, :5] = torch.tensor([-10.5, -10.0, 9.5, 10.0, 10.5]).to(dtype)
    p = torch.tensor([0.0, 1e-8, 0.37, 1.0]) if weighted else None
    dy = torch.randn(g.shape, generator=rng)
    h, sig, silu, raw, dg, du = _evaluate(
        g.cuda(), u.cuda(), None if p is None else p.cuda(), dy.cuda()
    )
    gc = g.double().clamp(max=10.0) if weighted else g.double()
    uc = u.double().clamp(-10.0, 10.0) if weighted else u.double()
    sig64 = torch.sigmoid(gc)
    silu64 = gc * sig64
    raw64 = silu64 * uc
    weighted64 = raw64 if p is None else raw64 * p.double()[:, None]
    torch.testing.assert_close(h.cpu().double(), weighted64, rtol=0.004, atol=1e-7)
    for actual, want in ((sig, sig64), (silu, silu64), (raw, raw64)):
        torch.testing.assert_close(actual.cpu().double(), want, rtol=3e-6, atol=3e-6)
    _, state = Reference.forward(g, u, p)
    for actual, want in zip((dg, du), Reference.backward(dy, g, u, p, state)):
        torch.testing.assert_close(actual.cpu(), want, rtol=3e-5, atol=5e-6)
    if weighted:
        # Multiplication by a zero mask can retain a negative zero sign.
        assert torch.count_nonzero(dg[:, 1:3]) == 0
        assert torch.count_nonzero(du[:, [0, 1, 3, 4]]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weighted", [False, True])
def test_strides_autograd_and_recompute(dtype, weighted):
    rng = torch.Generator().manual_seed(462)
    x = torch.randn(10, 1028, generator=rng).to(device="cuda", dtype=dtype)[::2, ::2]
    dy = torch.randn(10, 514, generator=rng).cuda()[::2, ::2].bfloat16()
    p = torch.full((10,), 0.37, device="cuda")[::2] if weighted else None
    expected = _evaluate(*x.chunk(2, dim=1), p, dy)
    for packed in (False, True):
        for recompute in (False, True):
            leaf = x.detach().requires_grad_()
            if packed:
                y = clamp_swiglu_weighted_packed(leaf, p, recompute=recompute)
            else:
                y = clamp_swiglu_weighted(*leaf.chunk(2, dim=1), p, recompute=recompute)
            dx = torch.autograd.grad(y, leaf, dy)[0]
            _bytes_equal(y, expected[0])
            _bytes_equal(dx, torch.cat(expected[-2:], dim=1).to(dtype))
    contiguous = _evaluate(*x.contiguous().chunk(2, dim=1), p, dy.contiguous())
    for a, b in zip(expected, contiguous):
        _bytes_equal(a, b)


def test_zero_rows_and_zero_stride_weight():
    for m in (0, 3):
        g = torch.full((m, 33), 2.0, device="cuda")
        u = torch.full_like(g, 3.0)
        p = torch.tensor([0.0], device="cuda").expand(m)
        h, _, _, raw, dg, du = _evaluate(g, u, p, torch.ones_like(g))
        assert h.shape == raw.shape == dg.shape == du.shape == (m, 33)
        assert torch.count_nonzero(h) == torch.count_nonzero(dg) == torch.count_nonzero(du) == 0
        if m:
            assert torch.all(raw > 0)


def test_two_input_updates_are_visible():
    g, u = torch.ones(3, 33, device="cuda"), torch.ones(3, 33, device="cuda")
    p = torch.full((3,), 0.37, device="cuda")
    outputs = []
    for _ in range(3):
        outputs.append(clamp_swiglu_weighted(g, u, p))
        g.add_(0.5)
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])


def test_weight_once_and_rounding_boundary():
    g = torch.tensor([[1.234, -0.743, 3.123]], device="cuda")
    u = torch.tensor([[2.345, 1.237, 0.583]], device="cuda")
    p = torch.tensor([0.371], device="cuda")
    y, state = Backend.forward(g, u, p)
    raw = state.unweighted.cpu()
    # Check the actual saved FP32 values, independent of host/device exp differences.
    _bytes_equal(y.cpu(), (raw * p.cpu()[:, None]).bfloat16())
    assert not torch.equal(y.cpu(), (raw.bfloat16().float() * p.cpu()[:, None]).bfloat16())
    assert not torch.equal(y.cpu(), (raw * p.cpu()[:, None].square()).bfloat16())
