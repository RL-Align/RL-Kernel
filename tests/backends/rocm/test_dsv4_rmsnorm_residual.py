# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Task 5 byte acceptance on MI300X/MI355X. Run with RLK_REQUIRE_ROCM=1 to forbid skips.

The reference is executed on the same device so ``rsqrt`` never crosses
devices inside a byte comparison.
"""

import os

import pytest
import torch

from rl_engine.backends.rocm.norm.rmsnorm_residual import RocmRMSNormResidual as Backend
from rl_engine.contracts.operators.rmsnorm_residual import EPS, RMSNormResidualContractError
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual
from rl_engine.reference.norm.rmsnorm_residual import RMSNormResidualReference as Reference


@pytest.fixture(scope="module", autouse=True)
def require_device():
    available = bool(torch.version.hip and torch.cuda.is_available())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if available else "none"
    if arch not in ("gfx942", "gfx950"):
        reason = f"task 5 acceptance needs gfx942 or gfx950, found {arch}"
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def _bytes_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _inputs(t, d, seed, grad_dtype=torch.bfloat16):
    rng = torch.Generator().manual_seed(seed)
    x = (torch.randn(t, d, generator=rng) * 3).to("cuda", torch.bfloat16)
    gamma = (torch.randn(d, generator=rng) + 1.0).cuda()
    dy = torch.randn(t, d, generator=rng).to("cuda", grad_dtype)
    dres = torch.randn(t, d, generator=rng).to("cuda", grad_dtype)
    return x, gamma, dy, dres


def _run(impl, x, gamma, dy, dres):
    y, residual, saved = impl.forward(x, gamma, EPS)
    dx, dgamma = impl.backward(dy, dres, x, gamma, saved)
    assert residual.data_ptr() != x.data_ptr()
    for value in (y, saved.r, dx, dgamma):
        assert torch.isfinite(value).all()
    return y, residual, saved.r, dx, dgamma


def _compare(x, gamma, dy, dres):
    got = _run(Backend, x, gamma, dy, dres)
    want = _run(Reference, x, gamma, dy, dres)
    for name, g, w in zip(("y", "residual", "r", "dx", "dgamma"), got, want, strict=True):
        assert g.dtype == w.dtype, name
        _bytes_equal(g, w)
    return got


@pytest.mark.parametrize("d", [1, 7, 128, 4096])
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_bytes_match_reference_across_shapes(d, grad_dtype):
    for t in (1, 7, 16, 17, 64):
        _compare(*_inputs(t, d, seed=t, grad_dtype=grad_dtype))


def test_edge_values_match_reference():
    d = 128
    x = torch.zeros(6, d, dtype=torch.bfloat16)
    x[1] = 2.0**-60  # subnormal-ish squares
    x[2] = 2.0**6
    x[3] = torch.linspace(-4, 4, d).to(torch.bfloat16)
    x[4] = 1.0
    x[5, ::2] = 65504.0 / 64  # large but finite after squaring and averaging
    _, gamma, dy, dres = _inputs(6, d, seed=99)
    _compare(x.cuda(), gamma, dy, dres)
    _compare(x.cuda(), gamma, dy, torch.zeros_like(dres))


def test_same_row_bytes_across_batch_position_padding_stride_and_repeat():
    for d in (128, 4096):
        x, gamma, dy, dres = _inputs(7, d, seed=7)
        full = _compare(x, gamma, dy, dres)
        again = _compare(x, gamma, dy, dres)
        for want, got in zip(full, again, strict=True):
            _bytes_equal(want, got)

        for row in (0, 3, 6):
            one = _compare(x[row : row + 1], gamma, dy[row : row + 1], dres[row : row + 1])
            for index in range(4):
                _bytes_equal(one[index], full[index][row : row + 1])

        for m, pos in ((16, 0), (17, 9), (65, 64)):
            big_x, _, big_dy, big_dres = _inputs(m, d, seed=m)
            big_x[pos] = x[0]
            big_dy[pos] = dy[0]
            big_dres[pos] = dres[0]
            batched = _compare(big_x, gamma, big_dy, big_dres)
            for index in range(4):
                _bytes_equal(batched[index][pos : pos + 1], full[index][:1])

        def pad(value):
            zero = torch.zeros_like(value[:1])
            return torch.cat((zero, value, zero), dim=0)

        padded = _compare(pad(x), gamma, pad(dy), pad(dres))
        for index in range(4):
            _bytes_equal(padded[index][1:-1], full[index])

        def strided(value):
            storage = torch.empty(
                (*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype, device=value.device
            )
            view = storage[..., ::2]
            view.copy_(value)
            return view

        laid_out = _compare(strided(x), gamma, strided(dy), strided(dres))
        for want, got in zip(full, laid_out, strict=True):
            _bytes_equal(want, got)


def test_fp32_gamma_is_not_rounded_to_bf16():
    x, _, _, dres = _inputs(7, 4096, seed=53)
    gamma = torch.full((4096,), 1.0 + 2.0**-10, device="cuda", dtype=torch.float32)
    dy = torch.zeros_like(x)
    dy[:, 0] = 1
    want_y, _, want_saved = Reference.forward(x, gamma, EPS)
    rounded_y, _, _ = Reference.forward(x, gamma.bfloat16().float(), EPS)
    assert not torch.equal(want_y, rounded_y), "fixture must expose gain rounding"
    y, _, saved = Backend.forward(x, gamma, EPS)
    _bytes_equal(y, want_y)
    _bytes_equal(saved.r, want_saved.r)


def test_empty_batch_runs():
    x, gamma, dy, dres = _inputs(0, 128, seed=1)
    y, residual, r, dx, dgamma = _compare(x, gamma, dy, dres)
    assert y.shape == (0, 128) and r.shape == (0,) and dx.shape == (0, 128)
    assert torch.equal(dgamma, torch.zeros_like(dgamma))


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("branch", ["y", "residual", "both"])
def test_autograd_through_the_rocm_backend(branch, recompute):
    x, gamma, dy, dres = _inputs(7, 128, seed=3)
    x.requires_grad_(True)
    gamma.requires_grad_(True)
    y, residual = rmsnorm_residual(x, gamma, backend="rocm", recompute=recompute)
    if branch == "y":
        y.backward(dy)
        dres = torch.zeros_like(dres)
    elif branch == "residual":
        residual.backward(dres)
        dy = torch.zeros_like(dy)
    else:
        torch.autograd.backward((y, residual), (dy, dres))
    x0, gamma0 = x.detach(), gamma.detach()
    _, _, saved = Reference.forward(x0, gamma0, EPS)
    dx, dgamma = Reference.backward(dy, dres, x0, gamma0, saved)
    _bytes_equal(x.grad, dx.to(torch.bfloat16))
    _bytes_equal(gamma.grad, dgamma)


def test_invalid_inputs_are_rejected_before_launch():
    x, gamma, dy, dres = _inputs(2, 128, seed=4)
    for bad in (x.float(), x.flatten(), x.cpu()):
        with pytest.raises((RMSNormResidualContractError, RuntimeError)):
            Backend.forward(bad, gamma, EPS)
    for bad_gamma in (gamma.bfloat16(), gamma[:-1], gamma.cpu()):
        with pytest.raises(RMSNormResidualContractError):
            Backend.forward(x, bad_gamma, EPS)
    with pytest.raises(RMSNormResidualContractError):
        Backend.forward(x, gamma, 1e-5)
    _, _, saved = Backend.forward(x, gamma, EPS)
    with pytest.raises(RMSNormResidualContractError):
        Backend.backward(dy.double(), dres, x, gamma, saved)
    with pytest.raises(RMSNormResidualContractError):
        Backend.backward(dy, dres, x, gamma, {"r": saved.r})
