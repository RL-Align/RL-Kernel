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

import dataclasses

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
# 512 rows is the benchmark's largest default shape; weight gradients sum over
# rows, so the row count is where accumulated error would show first.
SHAPES = [
    (1, HIDDEN_SIZE),
    (8, HIDDEN_SIZE),
    (64, HIDDEN_SIZE),
    (512, HIDDEN_SIZE),
    (2, 3, HIDDEN_SIZE),
]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
_KW = {"post_eps": POST_NORM_EPS, "pre_eps": RMS_NORM_EPS}


# The judgment each output is held to, keyed by name rather than by position in a
# parallel tuple. A field added to _Outputs without a row here raises KeyError at
# the comparison instead of silently inheriting its neighbour's tolerance row;
# checking a gradient against the forward row is the mistake this prevents.
_JUDGMENTS = {
    "residual_out": "forward_accuracy",
    "pre_normed": "forward_accuracy",
    "d_hidden": "gradient_accuracy",
    "d_residual": "gradient_accuracy",
    "d_post_w": "gradient_accuracy",
    "d_pre_w": "gradient_accuracy",
}


@dataclasses.dataclass(frozen=True)
class _Outputs:
    """Both forward outputs and all four gradients of one backend, named.

    Frozen and non-iterable on purpose: the two backends are compared field by
    field, so ``zip`` over two six-element lists -- and the silent truncation and
    reordering it allows -- is not expressible here.
    """

    residual_out: torch.Tensor
    pre_normed: torch.Tensor
    d_hidden: torch.Tensor
    d_residual: torch.Tensor
    d_post_w: torch.Tensor
    d_pre_w: torch.Tensor


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


# FP16 reduction is ``optional`` in the contract and this operator does not meet
# the declared row: the row sum over 6656 elements shows through a 10-bit mantissa.
# The allowance is a fraction of one ULP at the tensor's peak. Measured over six
# shapes, six seeds and both backends, the worst output needs 0.135 of one peak
# ULP beyond the row, so 0.25 leaves about 1.9x headroom and is four times tighter
# than the "one peak ULP" this started as.
#
# Why the peak and not each element's own magnitude, which would read tighter: the
# residual add cancels. Measured, ``residual_out`` has an element at |value|
# 1.56e-02 carrying an error of 1.95e-03, which is 128 ULPs of its own magnitude
# but a quarter of one ULP at the row's scale, because the rounding comes from the
# operands rather than the result. An elementwise bound calls that a failure and is
# wrong to.
#
# The number is checked by injecting bugs, not by whether the tests pass; the
# margins are in the PR.
_FP16_PEAK_ULP_BUDGET = 0.25


def _peak_ulp(reference: torch.Tensor, dtype: torch.dtype) -> float:
    """One ULP of ``dtype`` at the reference tensor's peak magnitude."""
    peak = torch.tensor(float(reference.abs().max()), dtype=dtype)
    if float(peak) == 0.0:
        return 0.0
    step = torch.nextafter(peak, torch.tensor(float("inf"), dtype=dtype)) - peak
    return float(step)


def _assert_within(judgment, dtype, got, want, label, contract, relax):
    """Compare at the contract row, plus a measured budget where it is optional.

    FP32 and BF16 are ``applicable`` for reduction judgments and are held to the
    contract exactly: measured over the same sweep, they need zero extra allowance
    on all six outputs, so ``relax`` is false for them and nothing is added.

    ``torch.testing.assert_close`` only takes scalar tolerances and the rtol term
    is elementwise, so the comparison is done by hand; the message reports the
    worst offender as a ratio of its own allowance, which is the number to read
    when it fails.
    """
    tolerance = resolve_tolerance(contract, judgment=judgment, op_class="reduction", dtype=dtype)
    got_f = got.cpu().float()
    want_f = want.cpu().float()
    allowed = tolerance.atol + tolerance.rtol * want_f.abs()
    if relax:
        allowed = allowed + _FP16_PEAK_ULP_BUDGET * _peak_ulp(want_f, dtype)
    error = (got_f - want_f).abs()
    ratio = error / allowed
    worst = int(ratio.argmax())
    assert bool((error <= allowed).all()), (
        f"{label}{judgment} outside the reduction row for {dtype}: "
        f"{int((ratio > 1).sum())} of {error.numel()} elements, worst "
        f"|error|={error.flatten()[worst]:.3e} against allowance "
        f"{allowed.flatten()[worst]:.3e} at |reference|={want_f.abs().flatten()[worst]:.3e} "
        f"(ratio {ratio.flatten()[worst]:.2f})"
    )


def _assert_against_gold(judgment, dtype, got, want, label=""):
    """Contract reduction row; FP16 (optional in the contract) gets one ULP.

    Same rule as the Triton tests and for the same measured reason: the kernel
    and the gold differ by one FP16 ULP on a handful of elements where the
    reduction order lands on the far side of a tie in the residual add.
    """
    support = resolve_tolerance_support(
        _CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype
    )
    _assert_within(judgment, dtype, got, want, label, _CONTRACT, support.status != "applicable")


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
@pytest.mark.parametrize("rows", [8, 512], ids=lambda r: f"{r}rows")
def test_cuda_backward_matches_autograd_on_the_reference(dtype, rows):
    # 512 rows: weight gradients sum over rows, so the gradient-accumulation
    # dtype only shows here; at 8 rows FP32 and eager accumulation both pass.
    shape = (rows, HIDDEN_SIZE)
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
        leaves + weights,
        ref_leaves + ref_weights,
        ("hidden", "residual", "post_w", "pre_w"),
        strict=True,
    ):
        _assert_against_gold("gradient_accuracy", dtype, got.grad, want.grad, f"{name}: ")


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_forward_is_batch_invariant_bitwise(dtype):
    """One block per row with a fixed reduction tree: a row's bytes do not move.

    Sub-batches wider than one row are the point of the sweep. A single-row slice
    launches one block and cannot show a reduction that reaches across rows or a
    grid-dependent tree; sizes of 2 and above change the grid while the rows are
    the same rows, which is the shape of the Axis-A requirement.
    """
    big = 64
    hidden, residual, post_w, pre_w = _inputs((big, HIDDEN_SIZE), dtype)
    full_res, full_pre = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)[:2]
    for count in (1, 2, 7, 32, 63):
        for start in (0, 1, big - count):
            sl = slice(start, start + count)
            part_res, part_pre = fused_add_norm_cuda_forward(
                hidden[sl], residual[sl], post_w, pre_w, **_KW
            )[:2]
            where = f"rows [{start}, {start + count})"
            assert torch.equal(full_res[sl], part_res), f"residual drifted at {where}"
            assert torch.equal(full_pre[sl], part_pre), f"pre-norm drifted at {where}"


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
    # Only d_hidden and d_residual are row-local. The weight gradients sum over
    # rows, so a sub-batch computes a different quantity rather than a slice of
    # this one, and the reference test covers them where they are still per-row.
    for count in (1, 2, 7, 15):
        for start in (0, 1, big - count):
            sl = slice(start, start + count)
            for whole, piece, name in zip(full, run(sl), ("d_hidden", "d_residual"), strict=True):
                where = f"rows [{start}, {start + count})"
                assert torch.equal(whole[sl], piece), f"{name} drifted at {where}"


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
        return _Outputs(
            residual_out=res.detach(),
            pre_normed=pre.detach(),
            d_hidden=leaves[0].grad,
            d_residual=leaves[1].grad,
            d_post_w=weights[0].grad,
            d_pre_w=weights[1].grad,
        )

    cuda_out = run(CudaFusedAddNormOp)
    triton_out = run(triton_mod.TritonFusedAddNormOp)
    for field in dataclasses.fields(_Outputs):
        name = field.name
        _assert_against_gold(
            _JUDGMENTS[name],
            dtype,
            getattr(cuda_out, name),
            getattr(triton_out, name),
            f"{name}: ",
        )


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


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_zero_rows_does_not_poison_the_context(dtype):
    """An empty batch is a no-op, not a launch error, and must not be sticky.

    ``<<<0, threads>>>`` returns ``cudaErrorInvalidConfiguration``. With no launch
    check the wrapper returned normally with plausible empty outputs and the error
    surfaced later against an unrelated call, after which every CUDA call in the
    process failed -- unrecoverable. Empty batches happen in RL serving whenever a
    scheduler step has no tokens. The assertion that matters is the last one: an
    unrelated kernel still runs afterwards.
    """
    hidden, residual, post_w, pre_w = _inputs((0, HIDDEN_SIZE), dtype)
    res_out, pre_out = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)[:2]
    torch.cuda.synchronize()
    assert res_out.shape == (0, HIDDEN_SIZE)
    assert pre_out.shape == (0, HIDDEN_SIZE)

    leaves = [t.detach().clone().requires_grad_(True) for t in (hidden, residual)]
    weights = [t.detach().clone().requires_grad_(True) for t in (post_w, pre_w)]
    res, pre = CudaFusedAddNormOp().forward(*leaves, *weights, **_KW)
    torch.autograd.backward((res, pre), grad_tensors=(torch.zeros_like(res), torch.zeros_like(pre)))
    torch.cuda.synchronize()
    for weight in weights:
        assert torch.equal(weight.grad, torch.zeros_like(weight.grad))

    probe = torch.ones(16, 16, device="cuda")
    assert float(probe.sum()) == 256.0, "the CUDA context did not survive the empty launch"


@requires_cuda_kernel
def test_cuda_binding_rejects_fp64_instead_of_silently_narrowing():
    """The extension boundary fails closed on FP64 rather than open.

    ``AT_DISPATCH_FLOATING_TYPES_AND2`` instantiates ``double`` and the kernel's
    generic load/store cast through ``float``, so a FP64 call used to return FP64
    tensors carrying FP32 precision -- measured at about 7e-07 of error -- with
    nothing raised. The Python wrapper rejects FP64 before this, so the only way in
    is the binding itself, which is exactly the path a C++ or benchmark caller
    takes. Failing open on precision is the wrong default for an operator whose
    contract is about where the casts are.
    """
    from rl_engine import _C

    kw = {"device": "cuda", "dtype": torch.float64}
    hidden = torch.randn(4, 32, **kw)
    residual = torch.randn_like(hidden)
    post_w = torch.randn(32, **kw)
    pre_w = torch.randn_like(post_w)

    with pytest.raises(RuntimeError, match="fp16, bf16, or fp32"):
        _C.fused_add_norm_forward(hidden, residual, post_w, pre_w, POST_NORM_EPS, RMS_NORM_EPS)

    # The supported dtypes still go through.
    out = _C.fused_add_norm_forward(
        hidden.bfloat16(),
        residual.bfloat16(),
        post_w.bfloat16(),
        pre_w.bfloat16(),
        POST_NORM_EPS,
        RMS_NORM_EPS,
    )
    assert out[0].dtype == torch.bfloat16
