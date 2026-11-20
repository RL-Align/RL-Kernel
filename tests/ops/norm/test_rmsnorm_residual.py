# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU contract tests for task 5. GPU byte acceptance lives in tests/backends/rocm."""

import pytest
import torch

from rl_engine.contracts.operators.rmsnorm_residual import (
    EPS,
    RMSNormResidualContractError,
    RMSNormResidualSaved,
    describe,
)
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual, select_backend
from rl_engine.reference.norm.rmsnorm_residual import RMSNormResidualReference as Reference


def _bytes_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _inputs(t, d, seed=5):
    rng = torch.Generator().manual_seed(seed)
    x = (torch.randn(t, d, generator=rng) * 3).to(torch.bfloat16)
    gamma = torch.randn(d, generator=rng) + 1.0
    dy = torch.randn(t, d, generator=rng).to(torch.bfloat16)
    dres = torch.randn(t, d, generator=rng).to(torch.bfloat16)
    return x, gamma, dy, dres


def _run(x, gamma, dy, dres):
    y, residual, saved = Reference.forward(x, gamma, EPS)
    dx, dgamma = Reference.backward(dy, dres, x, gamma, saved)
    return y, residual, saved.r, dx, dgamma


@pytest.mark.parametrize("d", [1, 7, 128, 4096])
def test_reference_matches_fp64_math_and_keeps_the_residual_bytes(d):
    x, gamma, dy, dres = _inputs(5, d)
    y, residual, r, dx, dgamma = _run(x, gamma, dy, dres)

    x64, g64 = x.double(), gamma.double()
    r64 = torch.rsqrt(x64.square().mean(dim=1) + EPS)
    y64 = x64 * r64[:, None] * g64
    torch.testing.assert_close(r.double(), r64, rtol=2e-5, atol=0)  # 4096-term FP32 fold
    torch.testing.assert_close(y.double(), y64, rtol=2e-2, atol=1e-2)  # one BF16 rounding

    leaf = x64.clone().requires_grad_(True)
    gleaf = g64.clone().requires_grad_(True)
    out = leaf * torch.rsqrt(leaf.square().mean(dim=1, keepdim=True) + EPS) * gleaf
    torch.autograd.backward((out, leaf), (dy.double(), dres.double()))
    torch.testing.assert_close(dx.double(), leaf.grad, rtol=1e-4, atol=5e-5)  # FP32 cancellation
    torch.testing.assert_close(dgamma.double(), gleaf.grad, rtol=1e-4, atol=5e-5)

    _bytes_equal(residual, x)
    assert residual.data_ptr() != x.data_ptr()
    assert y.dtype == residual.dtype == torch.bfloat16
    assert dx.dtype == dgamma.dtype == r.dtype == torch.float32


def test_reference_rows_do_not_depend_on_batch_padding_or_strides():
    x, gamma, dy, dres = _inputs(7, 128)
    full = _run(x, gamma, dy, dres)
    again = _run(x, gamma, dy, dres)
    for want, got in zip(full, again, strict=True):
        _bytes_equal(want, got)

    for row in (0, 3, 6):
        one = _run(x[row : row + 1], gamma, dy[row : row + 1], dres[row : row + 1])
        for index in range(4):  # y, residual, r, dx are row-local; dgamma is not
            _bytes_equal(one[index], full[index][row : row + 1])

    def pad(value):
        zero = torch.zeros_like(value[:1])
        return torch.cat((zero, value, zero), dim=0)

    padded = _run(pad(x), gamma, pad(dy), pad(dres))
    for index in range(4):
        _bytes_equal(padded[index][1:-1], full[index])

    def strided(value):
        storage = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
        view = storage[..., ::2]
        view.copy_(value)
        assert not view.is_contiguous()
        return view

    laid_out = _run(strided(x), strided(gamma), strided(dy), strided(dres))
    for want, got in zip(full, laid_out, strict=True):
        _bytes_equal(want, got)

    # The token fold of dgamma is pinned too: the same rows in the same order give the same bytes.
    _bytes_equal(padded[4], full[4])


def test_empty_batch_runs_and_returns_zero_dgamma():
    x, gamma, dy, dres = _inputs(0, 128)
    y, residual, r, dx, dgamma = _run(x, gamma, dy, dres)
    assert y.shape == residual.shape == dx.shape == (0, 128)
    assert r.shape == (0,)
    assert torch.equal(dgamma, torch.zeros(128))


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("branch", ["y", "residual", "both"])
def test_autograd_merges_both_branches_and_recompute_matches_saved(branch, recompute):
    x, gamma, dy, dres = _inputs(7, 128)
    x.requires_grad_(True)
    gamma.requires_grad_(True)
    y, residual = rmsnorm_residual(x, gamma, backend="reference", recompute=recompute)
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
    assert gamma.grad.dtype == torch.float32


def test_contract_fails_closed():
    x, gamma, dy, dres = _inputs(2, 128)
    for bad in (x.float(), x[:, :64].reshape(1, -1)[:, :100], x.flatten(), x.unsqueeze(0)):
        with pytest.raises(RMSNormResidualContractError):
            Reference.forward(bad, gamma, EPS)
    for bad_gamma in (gamma.bfloat16(), gamma.double(), gamma[:-1]):
        with pytest.raises(RMSNormResidualContractError):
            Reference.forward(x, bad_gamma, EPS)
    with pytest.raises(RMSNormResidualContractError):
        Reference.forward(x, gamma, 1e-5)
    with pytest.raises(RMSNormResidualContractError):
        Reference.forward(torch.zeros(2, 16385, dtype=torch.bfloat16), torch.ones(16385), EPS)

    _, _, saved = Reference.forward(x, gamma, EPS)
    with pytest.raises(RMSNormResidualContractError):
        Reference.backward(dy.double(), dres, x, gamma, saved)
    with pytest.raises(RMSNormResidualContractError):
        Reference.backward(dy, dres[:1], x, gamma, saved)
    with pytest.raises(RMSNormResidualContractError):
        Reference.backward(dy, dres, x, gamma, {"r": saved.r})
    with pytest.raises(RMSNormResidualContractError):
        Reference.backward(dy, dres, x, gamma, RMSNormResidualSaved(saved.r[:1], 128))
    with pytest.raises(RMSNormResidualContractError):
        Reference.backward(dy, dres, x, gamma, RMSNormResidualSaved(saved.r, 64))

    with pytest.raises(ValueError):
        select_backend("cuda")
    if not torch.version.hip:
        with pytest.raises(RuntimeError):
            select_backend("rocm").forward(x, gamma, EPS)

    fields = describe()
    assert fields["eps"] == EPS and fields["rsqrt_form"] == "rsqrt(mean + eps)"
    assert fields["downcast_points"] == ["y"]
