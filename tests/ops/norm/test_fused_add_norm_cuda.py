# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA [F1] kernels against the reference and against Triton.

Same shape as the Triton test file. Correctness is tolerance-based against the
native gold; batch invariance is asserted bitwise because it is a property of one
kernel. CUDA against Triton is held to the same tolerance row, and the bitwise
question between them is measured in the PR rather than asserted here, for the
reasons in the Triton module docstring.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.norm.fused_add_norm import (  # noqa: E402
    CudaFusedAddNormOp,
    cuda_fused_add_norm_available,
    fused_add_norm_cuda_forward,
)
from rl_engine.contracts.numerical import resolve_tolerance_support
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.reference.norm.fused_add_norm import (
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFusedAddNormOp,
)

requires_cuda_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and cuda_fused_add_norm_available()),
    reason="CUDA fused add-norm extension is not built",
)

_CONTRACT = load_contract()
SHAPES = [(1, HIDDEN_SIZE), (8, HIDDEN_SIZE), (64, HIDDEN_SIZE), (2, 3, HIDDEN_SIZE)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
_KW = {"post_eps": POST_NORM_EPS, "pre_eps": RMS_NORM_EPS}


def _rand(shape, *, seed, dtype, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)


def _inputs(shape, dtype, *, seed=0):
    return (
        _rand(shape, seed=200 + seed, dtype=dtype),
        _rand(shape, seed=201 + seed, dtype=dtype),
        _rand((shape[-1],), seed=202 + seed, dtype=dtype),
        _rand((shape[-1],), seed=203 + seed, dtype=dtype),
    )


def _one_ulp(reference: torch.Tensor) -> float:
    peak = float(reference.abs().max())
    if peak == 0.0:
        return 0.0
    hi = torch.nextafter(
        torch.tensor(peak, dtype=reference.dtype), torch.tensor(float("inf"), dtype=reference.dtype)
    )
    return float(hi - peak)


def _assert_against_gold(judgment, dtype, got, want, label=""):
    """Contract reduction row; FP16 (optional in the contract) gets one ULP.

    Same rule as the Triton tests and for the same measured reason: the kernel
    and the gold differ by one FP16 ULP on a handful of elements where the
    reduction order lands on the far side of a tie in the residual add.
    """
    t = resolve_tolerance(_CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype)
    atol = t.atol
    support = resolve_tolerance_support(
        _CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype
    )
    if support.status != "applicable":
        atol = max(atol, _one_ulp(want))
    torch.testing.assert_close(
        got.cpu().float(),
        want.float(),
        atol=atol,
        rtol=t.rtol,
        msg=lambda built: f"{label}{built}" if label else built,
    )


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cuda_forward_matches_native(dtype, shape):
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    want_res, want_pre = NativeFusedAddNormOp().forward(
        hidden.cpu(), residual.cpu(), post_w.cpu(), pre_w.cpu()
    )
    got_res, got_pre = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)[:2]
    _assert_against_gold("forward_accuracy", dtype, got_res, want_res, "residual_out: ")
    _assert_against_gold("forward_accuracy", dtype, got_pre, want_pre, "pre_normed: ")


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_backward_matches_autograd_on_the_reference(dtype):
    shape = (8, HIDDEN_SIZE)
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    g_res = _rand(shape, seed=210, dtype=dtype)
    g_pre = _rand(shape, seed=211, dtype=dtype)

    ref_leaves = [t.cpu().detach().clone().requires_grad_(True) for t in (hidden, residual)]
    ref_weights = [t.cpu().detach().clone().requires_grad_(True) for t in (post_w, pre_w)]
    out_res, out_pre = NativeFusedAddNormOp().forward(*ref_leaves, *ref_weights)
    torch.autograd.backward((out_res, out_pre), grad_tensors=(g_res.cpu(), g_pre.cpu()))

    leaves = [t.detach().clone().requires_grad_(True) for t in (hidden, residual)]
    weights = [t.detach().clone().requires_grad_(True) for t in (post_w, pre_w)]
    got_res, got_pre = CudaFusedAddNormOp().forward(*leaves, *weights, **_KW)
    torch.autograd.backward((got_res, got_pre), grad_tensors=(g_res, g_pre))

    for got, want, name in zip(
        leaves + weights, ref_leaves + ref_weights, ("hidden", "residual", "post_w", "pre_w")
    ):
        _assert_against_gold("gradient_accuracy", dtype, got.grad, want.grad, f"{name}: ")


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_forward_is_batch_invariant_bitwise(dtype):
    """One block per row with a fixed reduction tree: a row's bytes do not move."""
    big = 16
    hidden, residual, post_w, pre_w = _inputs((big, HIDDEN_SIZE), dtype)
    full_res, full_pre = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)[:2]
    for row in (0, 1, big // 2, big - 1):
        sl = slice(row, row + 1)
        part_res, part_pre = fused_add_norm_cuda_forward(
            hidden[sl], residual[sl], post_w, pre_w, **_KW
        )[:2]
        assert torch.equal(full_res[sl], part_res), f"residual drifted at row {row}"
        assert torch.equal(full_pre[sl], part_pre), f"pre-norm drifted at row {row}"


@requires_cuda_kernel
def test_cuda_backward_is_batch_invariant_bitwise():
    big, dtype = 16, torch.float32
    hidden, residual, post_w, pre_w = _inputs((big, HIDDEN_SIZE), dtype)
    g_res = _rand((big, HIDDEN_SIZE), seed=220, dtype=dtype)
    g_pre = _rand((big, HIDDEN_SIZE), seed=221, dtype=dtype)

    def run(sl):
        leaves = [t[sl].detach().clone().requires_grad_(True) for t in (hidden, residual)]
        res, pre = CudaFusedAddNormOp().forward(*leaves, post_w, pre_w, **_KW)
        torch.autograd.backward((res, pre), grad_tensors=(g_res[sl], g_pre[sl]))
        return [t.grad for t in leaves]

    full = run(slice(None))
    for row in (0, 1, big - 1):
        sl = slice(row, row + 1)
        for whole, piece, name in zip(full, run(sl), ("d_hidden", "d_residual")):
            assert torch.equal(whole[sl], piece), f"{name} drifted at row {row}"


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cuda_and_triton_agree_within_contract(dtype, shape):
    """The two GPU backends inside the reduction tolerance row, forward and backward.

    Not bitwise. The element-wise arithmetic is written in the same order in both
    kernels, so what differs between them is the row reduction: a shared-memory
    tree in CUDA against whatever ``tl.sum`` lowers to in Triton. The size of that
    difference is measured and reported in the PR; here the two are held to the
    same contract row each is already held to against the gold.
    """
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    g_res = _rand(shape, seed=230, dtype=dtype)
    g_pre = _rand(shape, seed=231, dtype=dtype)

    def run(op_cls):
        leaves = [t.detach().clone().requires_grad_(True) for t in (hidden, residual)]
        weights = [t.detach().clone().requires_grad_(True) for t in (post_w, pre_w)]
        res, pre = op_cls().forward(*leaves, *weights, **_KW)
        torch.autograd.backward((res, pre), grad_tensors=(g_res, g_pre))
        return [res.detach(), pre.detach()] + [t.grad for t in leaves + weights]

    cuda_out = run(CudaFusedAddNormOp)
    triton_out = run(triton_mod.TritonFusedAddNormOp)
    names = ("residual_out", "pre_normed", "d_hidden", "d_residual", "d_post_w", "d_pre_w")
    judgments = ("forward_accuracy",) * 2 + ("gradient_accuracy",) * 4
    for c, t, name, judgment in zip(cuda_out, triton_out, names, judgments):
        _assert_against_gold(judgment, dtype, c, t.cpu(), f"{name}: ")


@requires_cuda_kernel
@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda h, r, pw, vw: (h[..., :-1], r, pw, vw), ValueError, "share shape"),
        (lambda h, r, pw, vw: (h, r.to(torch.float16), pw, vw), ValueError, "share dtype"),
        (lambda h, r, pw, vw: (h, r, pw[:-1], vw), ValueError, "post_weight must be 1-D"),
        (lambda h, r, pw, vw: (h, r, pw, vw[:-1]), ValueError, "pre_weight must be 1-D"),
        (lambda h, r, pw, vw: (h.cpu(), r.cpu(), pw, vw), RuntimeError, "GPU tensor"),
    ],
)
def test_cuda_fails_closed(mutate, exc, match):
    args = _inputs((4, HIDDEN_SIZE), torch.float32)
    with pytest.raises(exc, match=match):
        fused_add_norm_cuda_forward(*mutate(*args), **_KW)
