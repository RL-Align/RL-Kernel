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
    fused_add_norm_cuda_backward,
    fused_add_norm_cuda_forward,
)
from rl_engine.reference.norm.fused_add_norm import (
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFusedAddNormOp,
)
from tests.helpers.fused_add_norm_tolerance import assert_within

requires_cuda_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and cuda_fused_add_norm_available()),
    reason="CUDA fused add-norm extension is not built",
)

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


def test_the_forward_and_gradient_rows_are_not_interchangeable():
    """Why the judgment per output matters: the two rows are not the same number.

    From the contract at ``op_class="reduction"``, BF16 allows 0.05 absolute on a
    forward judgment and 0.1 on a gradient one, with the same 0.02 relative term. So
    checking a forward output against the gradient row is twice as loose as it should
    be, which is the direction that hides a fault; the reverse is merely strict. An
    error of 0.07 lands between them and is the discriminator here.
    """
    want = torch.zeros(4, 8)
    got = want + 0.07
    assert_within("gradient_accuracy", torch.bfloat16, got, want, "gradient row: ")
    with pytest.raises(AssertionError, match="forward_accuracy outside"):
        assert_within("forward_accuracy", torch.bfloat16, got, want, "forward row: ")


def test_every_output_has_a_judgment_row():
    """The guard ``_JUDGMENTS``' comment promises, asserted rather than implied.

    The map sat unread for a while, with every call site passing its own literal, so
    the promise that a new output cannot inherit its neighbour's tolerance row was
    not kept by anything. ``_assert_against_gold`` looks the judgment up now, which
    makes an unknown name a KeyError, and this is the other half: a field added to
    ``_Outputs`` without a row here fails at collection rather than at whichever
    comparison happens to touch it first.
    """
    fields = {f.name for f in dataclasses.fields(_Outputs)}
    assert fields == set(_JUDGMENTS), (
        f"_Outputs and _JUDGMENTS disagree: only in _Outputs {sorted(fields - set(_JUDGMENTS))}, "
        f"only in _JUDGMENTS {sorted(set(_JUDGMENTS) - fields)}"
    )


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


def _assert_against_gold(output, dtype, got, want):
    """Contract reduction row, plus the shared FP16 budget where it applies.

    The policy and the measurements behind it live in
    ``tests/helpers/fused_add_norm_tolerance.py``, in one place because the three
    [F1] files have to agree on what FP16 is held to.

    The judgment comes from ``_JUDGMENTS`` keyed by the output's name rather than
    from the caller, which is what the map was written for and was not doing: it
    sat unread while every call site passed its own literal, so the guard its
    comment advertised did not exist and a new output could have been checked
    against the forward row by whoever added it. An unknown name is a KeyError here.
    """
    assert_within(_JUDGMENTS[output], dtype, got, want, f"{output}: ")


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cuda_forward_matches_native(dtype, shape):
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    want_res, want_pre = NativeFusedAddNormOp().forward(
        hidden.cpu(), residual.cpu(), post_w.cpu(), pre_w.cpu()
    )
    got_res, got_pre = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)[:2]
    _assert_against_gold("residual_out", dtype, got_res, want_res)
    _assert_against_gold("pre_normed", dtype, got_pre, want_pre)


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
        _assert_against_gold(f"d_{name}", dtype, got.grad, want.grad)


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
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_backward_is_batch_invariant_bitwise(dtype):
    big = 16
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
def test_cuda_and_triton_are_bitwise_equal(dtype, shape):
    """RFC #501's CUDA == Triton requirement, asserted on all six outputs.

    Bitwise, not tolerance. Three things had to line up, each measured on the way:

    * the row reduction order. ``tl.sum`` picks its own tree, so Triton now folds
      in the order the CUDA kernel already used -- lane ``t`` over ``x[t + k*LANES]``
      with ``k`` ascending, then a stride-halving tree over the lanes. See
      ``_reduction_shape`` in the Triton module, which owns the lane counts jointly
      with ``choose_threads`` in the .cu.
    * the divide by the width. Triton's ``/`` lowers to ``div.full.f32``, an
      approximate division; the CUDA kernel's is ``div.rn.f32``. That one
      difference alone left the statistics a ULP apart on 22% of rows after the
      orders already matched. Triton now uses ``tl.div_rn``.
    * the reciprocal square root. nvcc's ``rsqrtf`` is libdevice's refined non-FTZ
      sequence and ``tl.rsqrt`` is the bare ``rsqrt.approx.ftz.f32``; they disagree
      on about 12% of inputs. The CUDA kernel now emits the same instruction.

    This is the test that keeps all three. If it starts failing, one of the three
    moved, and the message above says where to look first.
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
        a = getattr(cuda_out, name).contiguous()
        b = getattr(triton_out, name).contiguous()
        # Byte views, not ``==``: that treats +0 and -0 as equal and NaN as unequal
        # to itself, so it is the wrong comparison for an exactness claim.
        differing = int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
        assert differing == 0, (
            f"{name}: CUDA and Triton differ on {differing} of {a.numel() * a.element_size()} "
            f"bytes at {dtype}, shape {tuple(a.shape)}; max |difference| "
            f"{(a.float() - b.float()).abs().max().item():.3e}"
        )


# The two wrappers validate in Python before reaching their kernels, and every
# divergence between them has been found by the review rather than by me: the rstd
# checks, the zero-width check, the weights' device, the residual's device, the rank
# guard, and then the weights' dtype. Nine findings of the same shape. One table
# driven against both of them, asserting the same exception type and the same text,
# is the thing that would have caught all nine, so it is here rather than a tenth
# fix. Anything genuinely one-sided stays out of it and is named below.
_BOTH_WRAPPERS_REJECT = [
    (lambda h, r, pw, vw: (h[..., :-1], r, pw, vw), ValueError, "share shape"),
    (lambda h, r, pw, vw: (h, r.to(torch.float16), pw, vw), ValueError, "share dtype"),
    (lambda h, r, pw, vw: (h, r, pw[:-1], vw), ValueError, "post_weight must be 1-D"),
    (lambda h, r, pw, vw: (h, r, pw, vw[:-1]), ValueError, "pre_weight must be 1-D"),
    (lambda h, r, pw, vw: (h.cpu(), r.cpu(), pw, vw), RuntimeError, "GPU tensor"),
    (
        lambda h, r, pw, vw: (h[..., :0], r[..., :0], pw[:0], vw[:0]),
        ValueError,
        "non-zero last dimension",
    ),
    (lambda h, r, pw, vw: (h[0, 0], r[0, 0], pw, vw), ValueError, "at least one dimension"),
    (lambda h, r, pw, vw: (h, r, pw.to(torch.int64), vw), TypeError, "fp16, bf16, or fp32"),
    (lambda h, r, pw, vw: (h, r, pw.cpu(), vw), RuntimeError, "GPU tensor"),
    (lambda h, r, pw, vw: (h, r, pw.to(torch.bfloat16), vw), ValueError, "share dtype"),
]


def test_the_cuda_op_refuses_to_construct_without_the_extension(monkeypatch):
    """The availability gate every other CUDA op in the table has.

    ``KernelRegistry._get_or_create_backend`` calls ``op_class()``, treats any
    exception as "this backend is not available", and moves to the next candidate.
    This operator is listed ahead of Triton and PyTorch, and the class had no
    ``__init__``, so on a CUDA host with an unbuilt extension it constructed fine,
    was returned as the selected backend, and failed later inside ``forward``.
    """
    from rl_engine.backends.cuda.norm import fused_add_norm as cuda_mod

    monkeypatch.setattr(cuda_mod, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(cuda_mod, "_C", None)
    with pytest.raises(RuntimeError, match="compiled rl_engine._C extension"):
        cuda_mod.CudaFusedAddNormOp()


@requires_cuda_kernel
def test_the_registry_falls_through_when_the_extension_is_missing(monkeypatch):
    """With the gate in place, dispatch reaches the next candidate instead of failing."""
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    from rl_engine.backends.cuda.norm import fused_add_norm as cuda_mod
    from rl_engine.runtime.registry import KernelRegistry

    monkeypatch.setattr(cuda_mod, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(cuda_mod, "_C", None)
    op = KernelRegistry().get_op("fused_post_norm_residual_pre_norm", device="cuda")
    assert isinstance(
        op, triton_mod.TritonFusedAddNormOp
    ), f"dispatch returned {type(op).__name__} with the extension unavailable"
    # And the control: with the extension there, the CUDA candidate still wins.
    monkeypatch.undo()
    assert isinstance(
        KernelRegistry().get_op("fused_post_norm_residual_pre_norm", device="cuda"),
        CudaFusedAddNormOp,
    )


# The backward half of the same idea. hidden is BF16 in the test below, so the
# dtype cases are keyed to that.
_BOTH_BACKWARDS_REJECT = [
    (
        lambda kw: kw | {"grad_residual_out": kw["grad_residual_out"][..., :-1]},
        ValueError,
        "share hidden's shape",
    ),
    (
        lambda kw: kw | {"grad_pre_normed": kw["grad_pre_normed"][:1]},
        ValueError,
        "share hidden's shape",
    ),
    (
        lambda kw: kw | {"grad_residual_out": kw["grad_residual_out"].float()},
        TypeError,
        "must have hidden's dtype",
    ),
    (
        lambda kw: kw | {"grad_pre_normed": kw["grad_pre_normed"].double()},
        TypeError,
        "fp16, bf16, or fp32",
    ),
    (lambda kw: kw | {"rstd_post": kw["rstd_post"][:-1]}, ValueError, "one entry per row"),
    (lambda kw: kw | {"rstd_pre": kw["rstd_pre"].unsqueeze(0)}, ValueError, "one entry per row"),
    (lambda kw: kw | {"rstd_post": kw["rstd_post"].to(torch.bfloat16)}, TypeError, "float32"),
    (
        lambda kw: kw | {"post_weight": kw["post_weight"].to(torch.float32)},
        ValueError,
        "share dtype",
    ),
    (lambda kw: kw | {"hidden": kw["hidden"].cpu()}, RuntimeError, "GPU tensor"),
]


@requires_cuda_kernel
@pytest.mark.parametrize("shape", [(4, HIDDEN_SIZE), (2, 3, HIDDEN_SIZE)], ids=["2d", "3d"])
def test_both_backwards_return_the_same_shapes_and_dtypes(shape):
    """The two public backwards are now a pair, so their returns have to match too.

    Values are compared bitwise elsewhere, in the CUDA-against-Triton test. This is
    the part a caller trips over before it gets to values: both flatten a 3-D input
    to rows by width and both return FP32, so swapping one for the other cannot
    change the shape or the dtype of anything downstream.
    """
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    hidden, residual, post_w, pre_w = _inputs(shape, torch.bfloat16)
    cuda_fwd = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)
    triton_fwd = triton_mod.fused_add_norm_triton_forward(hidden, residual, post_w, pre_w, **_KW)
    g = torch.randn_like(hidden)
    cuda_bwd = fused_add_norm_cuda_backward(
        g, g, hidden, cuda_fwd[0], post_w, pre_w, cuda_fwd[2], cuda_fwd[3]
    )
    triton_bwd = triton_mod.fused_add_norm_triton_backward(
        g, g, hidden, triton_fwd[0], post_w, pre_w, triton_fwd[2], triton_fwd[3]
    )
    for got, want, name in zip(
        cuda_bwd,
        triton_bwd,
        ("d_hidden", "d_residual", "partial_post_w", "partial_pre_w"),
        strict=True,
    ):
        assert (
            got.shape == want.shape
        ), f"{name}: CUDA {tuple(got.shape)}, Triton {tuple(want.shape)}"
        assert got.dtype is want.dtype is torch.float32, f"{name}: {got.dtype} and {want.dtype}"
    for got, want, name in zip(cuda_fwd, triton_fwd, ("residual_out", "pre_normed"), strict=False):
        assert (
            got.shape == want.shape
        ), f"{name}: CUDA {tuple(got.shape)}, Triton {tuple(want.shape)}"


@requires_cuda_kernel
@pytest.mark.parametrize("mutate, exc, match", _BOTH_BACKWARDS_REJECT)
def test_both_backwards_fail_closed_identically(mutate, exc, match):
    """The backward pair, held to the same bar as the forward pair.

    This path had no Python entry on the CUDA side at all: the only way in was
    ``_C.fused_add_norm_backward``, which checks the same things in C++ and raises a
    TORCH_CHECK ``RuntimeError`` where the Triton wrapper raises a ``TypeError`` or a
    ``ValueError``. So a caller catching an exception type got backend-dependent
    behaviour on the backward while the forward was already pinned.
    ``fused_add_norm_cuda_backward`` is that entry point, the autograd Function goes
    through it rather than past it, and this table is what keeps the two sets of
    checks worded identically rather than drifting again.

    A strided ``rstd`` is not in the table: it is covered on each side separately,
    because building one needs a different allocation than a ``mutate`` lambda over
    this dict can express.
    """
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    hidden, residual, post_w, pre_w = _inputs((4, HIDDEN_SIZE), torch.bfloat16)
    res_out, _, rstd_post, rstd_pre = fused_add_norm_cuda_forward(
        hidden, residual, post_w, pre_w, **_KW
    )
    kwargs = {
        "grad_residual_out": torch.randn_like(res_out),
        "grad_pre_normed": torch.randn_like(res_out),
        "hidden": hidden,
        "residual_out": res_out,
        "post_weight": post_w,
        "pre_weight": pre_w,
        "rstd_post": rstd_post,
        "rstd_pre": rstd_pre,
    }
    bad = mutate(kwargs)
    with pytest.raises(exc, match=match) as cuda_error:
        fused_add_norm_cuda_backward(**bad)
    with pytest.raises(exc, match=match) as triton_error:
        triton_mod.fused_add_norm_triton_backward(**bad)
    assert type(cuda_error.value) is type(triton_error.value), (
        f"CUDA raised {type(cuda_error.value).__name__}, "
        f"Triton raised {type(triton_error.value).__name__}"
    )
    assert str(cuda_error.value) == str(
        triton_error.value
    ), f"CUDA said {str(cuda_error.value)!r}, Triton said {str(triton_error.value)!r}"


@requires_cuda_kernel
@pytest.mark.parametrize("mutate, exc, match", _BOTH_WRAPPERS_REJECT)
def test_both_wrappers_fail_closed_identically(mutate, exc, match):
    """The same bad call has to fail the same way on both backends.

    Not merely "both raise". The C++ binding catches everything in this table, so
    the CUDA path always raised something; what differed was that it raised a
    TORCH_CHECK ``RuntimeError`` from the extension where the Triton wrapper raised
    a ``TypeError`` or ``ValueError`` from Python, which is a backend-dependent
    failure mode and makes any test that asserts an exception type backend-specific.
    So this compares the type and the message text, not just the fact of a raise.

    What is deliberately not here, because it is a real asymmetry rather than an
    oversight: the Triton side rejects a width needing more than ``_MAX_CHUNKS``
    statically unrolled chunks and the CUDA kernel loops instead and has no such
    ceiling. That one is in the doc's Known Limitations.
    """
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    bad = mutate(*_inputs((4, HIDDEN_SIZE), torch.float32))
    with pytest.raises(exc, match=match) as cuda_error:
        fused_add_norm_cuda_forward(*bad, **_KW)
    with pytest.raises(exc, match=match) as triton_error:
        triton_mod.fused_add_norm_triton_forward(*bad, **_KW)
    assert type(cuda_error.value) is type(triton_error.value), (
        f"CUDA raised {type(cuda_error.value).__name__}, "
        f"Triton raised {type(triton_error.value).__name__}"
    )
    assert str(cuda_error.value) == str(
        triton_error.value
    ), f"CUDA said {str(cuda_error.value)!r}, Triton said {str(triton_error.value)!r}"


@requires_cuda_kernel
@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda h, r, pw, vw: (h[..., :-1], r, pw, vw), ValueError, "share shape"),
        (lambda h, r, pw, vw: (h, r.to(torch.float16), pw, vw), ValueError, "share dtype"),
        (lambda h, r, pw, vw: (h, r, pw[:-1], vw), ValueError, "post_weight must be 1-D"),
        (lambda h, r, pw, vw: (h, r, pw, vw[:-1]), ValueError, "pre_weight must be 1-D"),
        (lambda h, r, pw, vw: (h.cpu(), r.cpu(), pw, vw), RuntimeError, "GPU tensor"),
        # Zero width used to surface as .view(-1, 0) complaining about an
        # ambiguous unspecified dimension, which says nothing about the cause.
        (
            lambda h, r, pw, vw: (h[..., :0], r[..., :0], pw[:0], vw[:0]),
            ValueError,
            "non-zero last dimension",
        ),
        # A 0-D hidden cleared every check above, since a 0-D residual equals it,
        # and then died on hidden.shape[-1] as an IndexError about a tuple index.
        # The oracle names the argument; a wrapper that does not is a failure mode
        # that differs from the thing it is measured against.
        (
            lambda h, r, pw, vw: (h[0, 0], r[0, 0], pw, vw),
            ValueError,
            "at least one dimension",
        ),
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


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [1, 3, 33, 63, 65, 100, 129, 257, 513, 1000, 8191])
def test_cuda_and_triton_are_bitwise_equal_at_awkward_widths(dtype, width):
    """The bitwise claim has to hold where the lane masking actually does work.

    The test above runs at the Muse width only, and 6656 is 13 x 512, so every
    chunk is full and the mask never fires. These widths are chosen around the
    thresholds in ``choose_threads`` and ``_reduction_shape`` (64, 128, 256, then
    512) and deliberately not multiples of the lane count, so the last chunk is
    partial on one side and the CUDA loop simply stops earlier on the other. That
    only agrees because a masked lane contributes an exact ``+ 0.0``.

    Forward only: the backward needs the saved statistics and is covered at the
    Muse width, while what is new here is the masking, which is shared by both
    kernels' reductions.
    """
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    hidden, residual, post_w, pre_w = _inputs((5, width), dtype)
    cuda_out = fused_add_norm_cuda_forward(hidden, residual, post_w, pre_w, **_KW)
    triton_out = triton_mod.fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    names = ("residual_out", "pre_normed", "rstd_post", "rstd_pre")
    for name, a, b in zip(names, cuda_out, triton_out, strict=True):
        a = a.contiguous()
        b = b.contiguous()
        differing = int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
        assert differing == 0, (
            f"{name}: width {width} at {dtype} differs on {differing} bytes; "
            f"lanes/chunks {triton_mod._reduction_shape(width)[:2]}"
        )


@requires_cuda_kernel
def test_cuda_binding_backward_validates_weights_and_statistics():
    """The binding fails closed on the four things the automated review found.

    All four were reachable only by calling `_C` directly, which is the path a C++
    or benchmark caller takes, and two of them were measured before the fix:

    * a ``post_weight`` shorter than the width. The forward checked the rank and
      the length and the backward did not, so the kernel indexed ``post_w + col``
      for every ``col < H`` and read past the allocation. At 4x64 with a
      half-width weight the call returned without raising.
    * a non-contiguous ``rstd``. Only the rank and length were checked, and the
      kernel reads through ``data_ptr<float>()`` with an implicit stride of one, so
      ``rstd[::2]`` was read as if packed. Measured: the gradients came out
      different from the contiguous call, with nothing raised.
    * a wrong ``rstd`` dtype, which ``data_ptr<float>()`` would throw on, but with
      a message about the pointer cast rather than about the input.
    * tensors on another device, which the launchers would dereference under
      ``hidden``'s CUDAGuard. Not reachable with one GPU, so it is checked here by
      construction rather than by running it.
    """
    from rl_engine import _C

    H, T = 64, 4
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    hidden = torch.randn(T, H, **kw)
    residual = torch.randn_like(hidden)
    post_w = torch.randn(H, **kw)
    pre_w = torch.randn_like(post_w)
    res_out, _, rstd_post, rstd_pre = _C.fused_add_norm_forward(
        hidden, residual, post_w, pre_w, POST_NORM_EPS, RMS_NORM_EPS
    )
    g = torch.randn_like(hidden)
    good = (g, g, hidden, res_out, post_w, pre_w, rstd_post, rstd_pre)

    # The good call still works, so the guards are not rejecting everything.
    _C.fused_add_norm_backward(*good)

    short = torch.randn(H // 2, **kw)
    # A zero feature dimension passed every other check and reached the kernel,
    # where the row mean is a division by zero: measured, rstd_post came back
    # [nan, nan, nan, nan] through this entry point.
    zero = torch.zeros(T, 0, **kw)
    with pytest.raises(RuntimeError, match="must be non-zero"):
        _C.fused_add_norm_forward(
            zero, zero, torch.zeros(0, **kw), torch.zeros(0, **kw), POST_NORM_EPS, RMS_NORM_EPS
        )

    with pytest.raises(RuntimeError, match="must equal weight size"):
        _C.fused_add_norm_backward(g, g, hidden, res_out, short, short, rstd_post, rstd_pre)

    # Both weights 2-D, so this reaches the rank check rather than the earlier
    # "must share shape" one in fused_add_norm_check_pair.
    two_d = torch.randn(1, H, **kw)
    with pytest.raises(RuntimeError, match="post_weight must be 1D"):
        _C.fused_add_norm_backward(g, g, hidden, res_out, two_d, two_d.clone(), rstd_post, rstd_pre)

    padded = torch.zeros(T * 2, device="cuda", dtype=torch.float32)
    padded[::2] = rstd_post
    strided = padded[::2]
    assert not strided.is_contiguous()
    with pytest.raises(RuntimeError, match="rstd_post must be contiguous"):
        _C.fused_add_norm_backward(g, g, hidden, res_out, post_w, pre_w, strided, rstd_pre)

    with pytest.raises(RuntimeError, match="rstd_pre must be float32"):
        _C.fused_add_norm_backward(
            g, g, hidden, res_out, post_w, pre_w, rstd_post, rstd_pre.to(torch.bfloat16)
        )

    with pytest.raises(RuntimeError, match=r"rstd_post must be \[T\]"):
        _C.fused_add_norm_backward(g, g, hidden, res_out, post_w, pre_w, rstd_post[:-1], rstd_pre)

    # The launcher dispatches on hidden's dtype and then casts the cotangent
    # pointer, so a mismatch used to surface from data_ptr with a message about the
    # cast rather than about the input.
    # Both cotangents FP32, so this reaches the comparison against hidden rather
    # than the earlier "must share dtype" one between the two of them.
    with pytest.raises(RuntimeError, match="grad_residual_out must have hidden's dtype"):
        _C.fused_add_norm_backward(
            g.float(), g.float(), hidden, res_out, post_w, pre_w, rstd_post, rstd_pre
        )


@requires_cuda_kernel
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_cuda_binding_rejects_tensors_on_another_device():
    """Cross-device inputs raise instead of being read under hidden's guard."""
    from rl_engine import _C

    H, T = 64, 4
    hidden = torch.randn(T, H, device="cuda:0", dtype=torch.bfloat16)
    residual = torch.randn(T, H, device="cuda:1", dtype=torch.bfloat16)
    post_w = torch.randn(H, device="cuda:0", dtype=torch.bfloat16)
    pre_w = torch.randn_like(post_w)
    with pytest.raises(RuntimeError, match="same device"):
        _C.fused_add_norm_forward(hidden, residual, post_w, pre_w, POST_NORM_EPS, RMS_NORM_EPS)


@requires_cuda_kernel
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("which", ["residual", "post_weight", "pre_weight"])
def test_both_wrappers_reject_another_device_identically(which):
    """The cross-device half of the shared table, which needs a second GPU.

    It cannot live in ``_BOTH_WRAPPERS_REJECT`` because that table runs on one
    device, and these three were exactly where the wrappers diverged: the Triton
    side gained them a round at a time while this side relied on the binding, which
    raises a different exception type with a different message.
    """
    triton_mod = pytest.importorskip("rl_engine.backends.shared.triton.norm.fused_add_norm")
    hidden, residual, post_w, pre_w = _inputs((4, HIDDEN_SIZE), torch.bfloat16)
    args = {"residual": residual, "post_weight": post_w, "pre_weight": pre_w}
    args[which] = args[which].to("cuda:1")
    bad = (hidden, args["residual"], args["post_weight"], args["pre_weight"])
    with pytest.raises(RuntimeError, match="same device as hidden") as cuda_error:
        fused_add_norm_cuda_forward(*bad, **_KW)
    with pytest.raises(RuntimeError, match="same device as hidden") as triton_error:
        triton_mod.fused_add_norm_triton_forward(*bad, **_KW)
    assert str(cuda_error.value) == str(
        triton_error.value
    ), f"CUDA said {str(cuda_error.value)!r}, Triton said {str(triton_error.value)!r}"
