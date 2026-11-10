# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton [F1] kernels against the reference, plus the invariants the RFC asks for.

Correctness against the native gold is tolerance-based, which is what the repo
does for norms: ``test_rms_norm.py`` says so in as many words, "NOT torch.equal,
precisely because the reduction order is allowed to differ". The gold uses a
different, shape-invariant chunked statistic, so that bar is the right one there.

Two things here are asserted bitwise instead. Batch invariance, because it is a
property of a single implementation. And CUDA against Triton, in
``test_fused_add_norm_cuda.py``, because this row does pin the reduction order
across the two backends; see this module's kernel docstring for the three
differences that had to be closed.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from rl_engine.reference.norm.fused_add_norm import (
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFusedAddNormOp,
)
from tests.helpers.fused_add_norm_tolerance import assert_within

triton = pytest.importorskip("triton")

from rl_engine.backends.shared.triton.norm import fused_add_norm as triton_norm  # noqa: E402
from rl_engine.backends.shared.triton.norm.fused_add_norm import (  # noqa: E402
    TritonFusedAddNormOp,
    fused_add_norm_triton_backward,
    fused_add_norm_triton_forward,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_triton_ops_package_does_not_resolve_to_the_cuda_backend():
    """The new ``kernels/ops/triton/norm`` shim must not hand back CUDA modules.

    It did: its ``__getattr__`` imported ``rl_engine.backends.cuda.norm``, copied
    from the CUDA shim beside it, where all seven sibling packages under
    ``kernels/ops/triton/`` target ``rl_engine.backends.shared.triton.*``.

    Measured reachability, which is why this runs in a subprocess with a pinned
    import order: with the wrong target, ``pkg.fused_add_norm`` raises
    ``AttributeError`` in a fresh interpreter, but in a process that has already
    imported ``rl_engine.backends.cuda.norm.fused_add_norm`` -- which any CUDA test
    run does -- the submodule is an attribute of that package and the Triton import
    path silently returns the CUDA module. A comparison written that way would have
    compared CUDA against itself and reported perfect agreement. In-process this is
    unpinnable: whichever of the two packages is imported first wins, because the
    parent ``kernels/ops/triton/__init__.py`` forwards unknown names to
    ``rl_engine.backends.shared.triton``, which re-exports no subpackage names.
    """
    program = textwrap.dedent(
        """
        import rl_engine.backends.cuda.norm.fused_add_norm  # the order that hid it
        import rl_engine.kernels.ops.triton.norm as pkg

        module = pkg.fused_add_norm
        op = pkg.TritonFusedAddNormOp
        assert module.__name__ == "rl_engine.backends.shared.triton.norm.fused_add_norm", (
            f"Triton shim resolved to {module.__name__}"
        )
        assert op.__module__ == "rl_engine.backends.shared.triton.norm.fused_add_norm", (
            f"Triton op resolved to {op.__module__}"
        )
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def _assert_against_gold(judgment, dtype, got, want, label=""):
    """Contract reduction row, plus the shared FP16 budget where it applies.

    The policy and the measurements behind it live in
    ``tests/helpers/fused_add_norm_tolerance.py``. What a gold comparison cannot do
    at any dtype is police the eps values: measured at 64x6656 the worst violation
    ratio is 0.24 at fp32 and 0.36 at bf16 against a bar of 1.0, so no tolerance
    here would notice a swap. That belongs where the signal is visible, in
    ``test_production_eps_are_not_interchangeable`` in ``test_fused_add_norm.py``,
    and ``test_eps_values_reach_the_kernel`` below covers the cruder failure of eps
    being ignored, which is detectable bitwise.
    """
    assert_within(judgment, dtype, got, want, label)


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


def _rand(shape, *, seed, dtype, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)


def _inputs(shape, dtype, *, seed=0):
    return (
        _rand(shape, seed=100 + seed, dtype=dtype),
        _rand(shape, seed=101 + seed, dtype=dtype),
        _rand((shape[-1],), seed=102 + seed, dtype=dtype),
        _rand((shape[-1],), seed=103 + seed, dtype=dtype),
    )


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_triton_forward_matches_native(dtype, shape):
    """Both outputs inside the reduction tolerance row for this dtype."""
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)

    want_res, want_pre = NativeFusedAddNormOp().forward(
        hidden.cpu(), residual.cpu(), post_w.cpu(), pre_w.cpu()
    )
    got_res, got_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )[:2]

    _assert_against_gold("forward_accuracy", dtype, got_res, want_res, "residual_out: ")
    _assert_against_gold("forward_accuracy", dtype, got_pre, want_pre, "pre_normed: ")


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [8, 512], ids=lambda r: f"{r}rows")
def test_triton_backward_matches_autograd_on_the_reference(dtype, rows):
    """Gradients inside tolerance against autograd on the native forward.

    Both cotangents are random and non-zero, so neither the rstd term nor the
    second path into the residual output can hide.
    """
    # 512 rows: weight gradients sum over rows, so the gradient-accumulation
    # dtype only shows here; at 8 rows FP32 and eager accumulation both pass.
    shape = (rows, HIDDEN_SIZE)
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    g_res = _rand(shape, seed=110, dtype=dtype)
    g_pre = _rand(shape, seed=111, dtype=dtype)

    ref_leaves = [t.cpu().detach().clone().requires_grad_(True) for t in (hidden, residual)]
    ref_weights = [t.cpu().detach().clone().requires_grad_(True) for t in (post_w, pre_w)]
    out_res, out_pre = NativeFusedAddNormOp().forward(*ref_leaves, *ref_weights)
    torch.autograd.backward((out_res, out_pre), grad_tensors=(g_res.cpu(), g_pre.cpu()))

    leaves = [t.detach().clone().requires_grad_(True) for t in (hidden, residual)]
    weights = [t.detach().clone().requires_grad_(True) for t in (post_w, pre_w)]
    got_res, got_pre = TritonFusedAddNormOp().forward(
        *leaves, *weights, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    torch.autograd.backward((got_res, got_pre), grad_tensors=(g_res, g_pre))

    for got, want, name in zip(
        leaves + weights,
        ref_leaves + ref_weights,
        ("hidden", "residual", "post_w", "pre_w"),
        strict=True,
    ):
        _assert_against_gold("gradient_accuracy", dtype, got.grad, want.grad, f"{name}: ")


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_forward_is_batch_invariant_bitwise(dtype):
    """A row's bytes do not change with batch size or batch position.

    Bitwise, unlike the comparisons against the gold above: this is a property of
    this kernel alone, so no other implementation's reduction order is involved.
    """
    width = HIDDEN_SIZE
    big = 64
    hidden, residual, post_w, pre_w = _inputs((big, width), dtype)
    full_res, full_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )[:2]

    # Sub-batches wider than one row are the point of the sweep: a single-row slice
    # launches one program and cannot show a reduction that reaches across rows or a
    # grid-dependent tree.
    for count in (1, 2, 7, 32, 63):
        for start in (0, 1, big - count):
            sl = slice(start, start + count)
            part_res, part_pre = fused_add_norm_triton_forward(
                hidden[sl],
                residual[sl],
                post_w,
                pre_w,
                post_eps=POST_NORM_EPS,
                pre_eps=RMS_NORM_EPS,
            )[:2]
            where = f"rows [{start}, {start + count})"
            assert torch.equal(full_res[sl], part_res), f"residual drifted at {where}"
            assert torch.equal(full_pre[sl], part_pre), f"pre-norm drifted at {where}"


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_backward_is_batch_invariant_bitwise(dtype):
    """Axis A applies to gradients too, not just the forward."""
    width, big = HIDDEN_SIZE, 16
    hidden, residual, post_w, pre_w = _inputs((big, width), dtype)
    g_res = _rand((big, width), seed=120, dtype=dtype)
    g_pre = _rand((big, width), seed=121, dtype=dtype)

    def run(sl):
        leaves = [t[sl].detach().clone().requires_grad_(True) for t in (hidden, residual)]
        res, pre = TritonFusedAddNormOp().forward(
            *leaves, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
        )
        torch.autograd.backward((res, pre), grad_tensors=(g_res[sl], g_pre[sl]))
        return [t.grad for t in leaves]

    full = run(slice(None))
    # Only d_hidden and d_residual are row-local; the weight gradients sum over rows.
    for count in (1, 2, 7, 15):
        for start in (0, 1, big - count):
            sl = slice(start, start + count)
            for whole, piece, name in zip(full, run(sl), ("d_hidden", "d_residual"), strict=True):
                where = f"rows [{start}, {start + count})"
                assert torch.equal(whole[sl], piece), f"{name} drifted at {where}"


@requires_cuda
def test_the_triton_op_refuses_to_construct_without_triton(monkeypatch):
    """Same gate as the CUDA class, so dispatch falls through rather than failing late.

    The registry treats an exception from ``op_class()`` as "not available" and
    tries the next candidate, so without this a host with no Triton selected this
    backend and raised inside ``forward`` instead of reaching the PyTorch reference.
    """
    monkeypatch.setattr(triton_norm, "_TRITON_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Triton is not available"):
        triton_norm.TritonFusedAddNormOp()


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
def test_triton_fails_closed(mutate, exc, match):
    """Mismatched shape, dtype, weight width or device raises rather than guessing."""
    args = _inputs((4, HIDDEN_SIZE), torch.float32)
    with pytest.raises(exc, match=match):
        fused_add_norm_triton_forward(*mutate(*args), post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS)


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_eps_values_reach_the_kernel(dtype):
    """Both eps parameters change the output, so neither is ignored or hardcoded.

    Bitwise rather than by tolerance. The gold comparisons above cannot see an eps
    swap, as ``_assert_against_gold`` records with measurements, so nothing there
    would catch a kernel that dropped the parameters and baked in one value. This
    does: each eps is perturbed on its own and the corresponding output has to
    move. Deliberately not asserting *how much*, only that the wiring is live.
    """
    hidden, residual, post_w, pre_w = _inputs((8, HIDDEN_SIZE), dtype)
    kw = {"post_eps": POST_NORM_EPS, "pre_eps": RMS_NORM_EPS}
    base_res, base_pre = fused_add_norm_triton_forward(hidden, residual, post_w, pre_w, **kw)[:2]

    # A large post_eps shrinks the post-norm, which moves the residual sum.
    moved_res, _ = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, **{**kw, "post_eps": 1.0}
    )[:2]
    assert not torch.equal(base_res, moved_res), "post_eps does not reach the kernel"

    # A large pre_eps shrinks the pre-norm only; the residual output must not move.
    same_res, moved_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, **{**kw, "pre_eps": 1.0}
    )[:2]
    assert not torch.equal(base_pre, moved_pre), "pre_eps does not reach the kernel"
    assert torch.equal(base_res, same_res), "pre_eps leaked into the residual output"


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_zero_rows_is_a_no_op(dtype):
    """Triton skips empty grids, so this is a parity test against the CUDA path.

    The CUDA kernel had to be taught to return before the launch; this pins that
    the two backends agree on an empty batch rather than one raising.
    """
    hidden, residual, post_w, pre_w = _inputs((0, HIDDEN_SIZE), dtype)
    res_out, pre_out = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )[:2]
    torch.cuda.synchronize()
    assert res_out.shape == (0, HIDDEN_SIZE)
    assert pre_out.shape == (0, HIDDEN_SIZE)

    probe = torch.ones(16, 16, device="cuda")
    assert float(probe.sum()) == 256.0


_OVERFLOW_ROWS = 2**31 // HIDDEN_SIZE + 1  # 322,639 at H=6656: the first wrapping row
# Four tensors of the row count live at once inside the forward: hidden and
# residual going in, residual_out and pre_normed coming out, plus two FP32 rstd
# vectors of one entry per row, which round to nothing beside them. Measured with
# max_memory_allocated at 20k and 60k rows: 8.00 bytes per element exactly, which
# is what this computes. Spelled out because "2 * 4" read as one number and was
# taken for a dtype width.
_OVERFLOW_LIVE_TENSORS = 4
_OVERFLOW_GIB = (
    _OVERFLOW_ROWS * HIDDEN_SIZE * torch.bfloat16.itemsize * _OVERFLOW_LIVE_TENSORS / 1024**3
)


@requires_cuda
def test_triton_row_offset_does_not_wrap_at_two_billion_elements():
    """``row * H`` has to be computed in int64, not widened after the fact.

    ``tl.program_id`` is int32 and ``H`` is a constexpr int, so the product wrapped
    at 2**31 and the kernel read out of bounds: measured as a CUDA illegal memory
    access at exactly this row count, and correct results one thousand rows below
    it. The CUDA kernel casts to ``size_t`` before multiplying, so this was a
    backend divergence as well as a fault.

    The skip is decided here rather than in a ``skipif``, after dropping the
    caching allocator's blocks, because a collection-time check reads the free
    memory of an idle card and this test needs about 16 GiB by the time it
    actually runs. A ``skipif`` made it pass alone and fail inside a full session.
    """
    gc.collect()
    torch.cuda.empty_cache()
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gib < _OVERFLOW_GIB + 4.0:
        pytest.skip(f"needs about {_OVERFLOW_GIB + 4.0:.0f} GiB free, {free_gib:.1f} available")

    dtype = torch.bfloat16
    post_w = torch.randn(HIDDEN_SIZE, device="cuda", dtype=dtype)
    pre_w = torch.randn_like(post_w)
    hidden = torch.randn(_OVERFLOW_ROWS, HIDDEN_SIZE, device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden)
    try:
        res_out = fused_add_norm_triton_forward(
            hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
        )[0]
        torch.cuda.synchronize()

        # The last row is the one the wrapped offset would have missed.
        hid = hidden[-1].float()
        rstd = torch.rsqrt((hid * hid).mean() + POST_NORM_EPS)
        post = (hid * rstd * (1.0 + post_w.float())).to(dtype).float()
        want = (residual[-1].float() + post).to(dtype)
        assert torch.equal(res_out[-1], want)
    finally:
        del hidden, residual
        gc.collect()
        torch.cuda.empty_cache()


@requires_cuda
@pytest.mark.parametrize(
    "mutate, exc, match",
    [
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
        (lambda kw: kw | {"rstd_post": kw["rstd_post"][:-1]}, ValueError, "one entry per row"),
        (
            lambda kw: kw | {"rstd_pre": kw["rstd_pre"].unsqueeze(0)},
            ValueError,
            "one entry per row",
        ),
        (
            lambda kw: kw | {"post_weight": kw["post_weight"].to(torch.float32)},
            ValueError,
            "share dtype",
        ),
        (lambda kw: kw | {"hidden": kw["hidden"].cpu()}, RuntimeError, "GPU tensor"),
        # hidden is BF16 below. The binding rejects both of these, the first from
        # the comparison against hidden and the second from the supported set, and
        # this side accepted both and cast them to FP32 on load.
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
    ],
)
def test_triton_backward_validates_its_inputs(mutate, exc, match):
    """The public backward fails closed, like the CUDA binding does in C++.

    It recomputed ``block`` inline and never called ``_validate``, so it had no
    dtype, device, shape or ``_MAX_BLOCK`` check. A cotangent of the wrong shape
    read and wrote the wrong rows rather than raising.
    """
    shape = (4, HIDDEN_SIZE)
    dtype = torch.bfloat16
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    res_out, _, rstd_post, rstd_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
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
    with pytest.raises(exc, match=match):
        fused_add_norm_triton_backward(**mutate(kwargs))


@requires_cuda
def test_triton_backward_narrowing_follows_hidden_not_the_cotangent(monkeypatch):
    """The accumulation dtype comes from the input, as the reference defines it.

    The kernel read ``G_RES.dtype.element_ty``, so an FP32 cotangent against BF16
    ``hidden`` turned both narrowings into the identity and returned the gradient
    of ``forward_fp32`` instead -- a different computation, whose ``d_post_w`` the
    reference measures as 2.6-2.9x outside the contract row.

    The public entry point now rejects that cotangent outright, because the CUDA
    binding does, so the mismatch is fed in one level down with the guard patched
    out. Asserted both ways: the guard is there, and behind it the kernel still
    reads its narrowing from ``hidden``. Without the second half the first would
    just be hiding the old bug rather than fixing it.
    """
    shape = (8, HIDDEN_SIZE)
    dtype = torch.bfloat16
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    res_out, _, rstd_post, rstd_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    g_res = _rand(shape, seed=240, dtype=dtype)
    g_pre = _rand(shape, seed=241, dtype=dtype)

    narrow = fused_add_norm_triton_backward(
        g_res, g_pre, hidden, res_out, post_w, pre_w, rstd_post, rstd_pre
    )
    with pytest.raises(TypeError, match="must have hidden's dtype"):
        fused_add_norm_triton_backward(
            g_res.float(), g_pre.float(), hidden, res_out, post_w, pre_w, rstd_post, rstd_pre
        )
    monkeypatch.setattr(triton_norm, "_validate_cotangents", lambda *a: None)
    wide = fused_add_norm_triton_backward(
        g_res.float(), g_pre.float(), hidden, res_out, post_w, pre_w, rstd_post, rstd_pre
    )
    # Same mathematical inputs, only the cotangent's storage dtype differs, so the
    # narrowing must land in the same place and the results must be identical.
    for got, want, name in zip(
        wide, narrow, ("d_hidden", "d_residual", "partial_post_w", "partial_pre_w"), strict=True
    ):
        assert torch.equal(got, want), f"{name} changed with the cotangent's dtype"


@requires_cuda
@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda kw: kw | {"rstd_post": kw["rstd_post"].cpu()}, RuntimeError, "GPU tensor"),
        (lambda kw: kw | {"rstd_pre": kw["rstd_pre"].to(torch.bfloat16)}, TypeError, "float32"),
        (lambda kw: kw | {"rstd_post": kw["rstd_post"][:-1]}, ValueError, "one entry per row"),
    ],
)
def test_triton_backward_rstd_checks_match_the_cuda_binding(mutate, exc, match):
    """The two backends must reject the same statistics, not one of the two.

    The CUDA binding checks device, contiguity, float32 and shape on
    ``rstd_post``/``rstd_pre``; this side checked rank and length only, and the
    kernels read them with ``tl.load(RSTD_POST + row)``, an implicit stride of one.
    So a strided view ran here and raised there, which is the backend divergence
    the weight checks beside it exist to avoid, one argument over.
    """
    shape = (4, HIDDEN_SIZE)
    dtype = torch.bfloat16
    hidden, residual, post_w, pre_w = _inputs(shape, dtype)
    res_out, _, rstd_post, rstd_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
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
    with pytest.raises(exc, match=match):
        fused_add_norm_triton_backward(**mutate(kwargs))


@requires_cuda
def test_triton_backward_rejects_a_strided_rstd():
    """The case the CUDA binding's comment measured, on this side too."""
    shape = (4, HIDDEN_SIZE)
    hidden, residual, post_w, pre_w = _inputs(shape, torch.bfloat16)
    res_out, _, rstd_post, rstd_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    padded = torch.zeros(shape[0] * 2, device="cuda", dtype=torch.float32)
    padded[::2] = rstd_post
    strided = padded[::2]
    assert strided.shape == rstd_post.shape and not strided.is_contiguous()
    g = torch.randn_like(res_out)
    with pytest.raises(ValueError, match="rstd_post must be contiguous"):
        fused_add_norm_triton_backward(g, g, hidden, res_out, post_w, pre_w, strided, rstd_pre)


@requires_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("which", ["residual", "post_weight", "pre_weight", "grad_residual_out"])
def test_triton_rejects_tensors_on_another_device(which):
    """Same device as hidden, not merely some GPU, matching the CUDA binding.

    The launch takes its device and stream from ``hidden``, so a weight, a
    cotangent or the residual on another device is dereferenced as if it lived
    there. This side checked ``is_cuda`` only while the binding checked the device,
    and each round of this fix reached one argument and not its neighbour.

    What happens without the check, measured on two H100s: with peer access off
    Triton refuses the launch itself, reporting ValueError("Pointer argument (at 1)
    cannot be accessed from Triton (cpu tensor?)") against a tensor that is on a
    GPU, and with it on the launch runs, reads the other device through the peer
    mapping, and returns an answer bitwise equal to the same-device one. One
    cross-device copy earlier in the process is enough to turn peer access on, which
    is usually how a tensor ends up on the wrong device, so the silent outcome is
    the likely one and is the reason this is a check and not a comment.
    """
    shape = (4, HIDDEN_SIZE)
    hidden, residual, post_w, pre_w = _inputs(shape, torch.bfloat16)
    if which == "residual":
        with pytest.raises(RuntimeError, match="same device as hidden"):
            fused_add_norm_triton_forward(
                hidden,
                residual.to("cuda:1"),
                post_w,
                pre_w,
                post_eps=POST_NORM_EPS,
                pre_eps=RMS_NORM_EPS,
            )
        return
    if which == "post_weight":
        with pytest.raises(RuntimeError, match="same device as hidden"):
            fused_add_norm_triton_forward(
                hidden,
                residual,
                post_w.to("cuda:1"),
                pre_w,
                post_eps=POST_NORM_EPS,
                pre_eps=RMS_NORM_EPS,
            )
        return
    if which == "pre_weight":
        with pytest.raises(RuntimeError, match="same device as hidden"):
            fused_add_norm_triton_forward(
                hidden,
                residual,
                post_w,
                pre_w.to("cuda:1"),
                post_eps=POST_NORM_EPS,
                pre_eps=RMS_NORM_EPS,
            )
        return
    res_out, _, rstd_post, rstd_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    g = torch.randn_like(res_out)
    with pytest.raises(RuntimeError, match="same device as hidden"):
        fused_add_norm_triton_backward(
            g.to("cuda:1"), g, hidden, res_out, post_w, pre_w, rstd_post, rstd_pre
        )


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_zero_rows_takes_the_explicit_guard(dtype):
    """An empty batch returns without launching, on both backends.

    Triton 3.6.0 makes ``kernel[(0,)]`` a no-op, which I measured, so this passed
    before the guard existed. It is explicit now because that is an undocumented
    launcher detail, the H100 used for the benchmark runs Triton 3.3.1, and the CUDA
    launcher has to guard the same case because a zero-block grid there is a sticky
    cudaErrorInvalidConfiguration. The probe at the end is the assertion that
    matters either way.
    """
    hidden, residual, post_w, pre_w = _inputs((0, HIDDEN_SIZE), dtype)
    res_out, pre_out, rstd_post, rstd_pre = fused_add_norm_triton_forward(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    assert res_out.shape == (0, HIDDEN_SIZE)
    assert rstd_post.shape == (0,)
    grads = fused_add_norm_triton_backward(
        hidden, hidden, hidden, res_out, post_w, pre_w, rstd_post, rstd_pre
    )
    assert all(g.shape == (0, HIDDEN_SIZE) for g in grads)
    torch.cuda.synchronize()
    probe = torch.ones(16, 16, device="cuda")
    assert float(probe.sum()) == 256.0, "the CUDA context did not survive the empty call"
