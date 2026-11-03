# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton [F1] kernels against the reference, plus the invariants the RFC asks for.

Correctness here is tolerance-based against the native gold, which is what the
repo does for norms: ``test_rms_norm.py`` says so in as many words, "NOT
torch.equal, precisely because the reduction order is allowed to differ". What is
asserted bitwise is batch invariance, because that one is a property of a single
implementation and does not depend on anyone else's float order.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from rl_engine.contracts.numerical import resolve_tolerance_support
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.reference.norm.fused_add_norm import (
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFusedAddNormOp,
)

triton = pytest.importorskip("triton")

from rl_engine.backends.shared.triton.norm.fused_add_norm import (  # noqa: E402
    TritonFusedAddNormOp,
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


_CONTRACT = load_contract()


def _tol(judgment: str, dtype: torch.dtype) -> tuple[float, float]:
    """Tolerance from the contract, ``reduction`` class, for the required dtypes.

    A norm is a reduction, not an elementwise op: ``OP_SPECS["rms_norm"]`` sets
    ``op_class="reduction"`` and those rows are looser, because a row sum over
    6656 elements plus the cancellation in the backward is not the error budget of
    a pointwise multiply. Read from the contract rather than copied, so a
    recalibration cannot leave this test asserting a stale number.
    """
    t = resolve_tolerance(_CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype)
    return t.atol, t.rtol


def _is_required(judgment: str, dtype: torch.dtype) -> bool:
    """Whether the contract makes this dtype a required bar for this judgment."""
    support = resolve_tolerance_support(
        _CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype
    )
    return support.status == "applicable"


def _one_ulp(reference: torch.Tensor) -> float:
    """One ULP at the reference's largest magnitude, in its own dtype."""
    peak = float(reference.abs().max())
    if peak == 0.0:
        return 0.0
    return float(
        torch.nextafter(
            torch.tensor(peak, dtype=reference.dtype),
            torch.tensor(float("inf"), dtype=reference.dtype),
        )
        - peak
    )


def _assert_against_gold(judgment, dtype, got, want, label=""):
    """Compare to the gold at the contract bar, with one documented exception.

    FP32 and BF16 are ``applicable`` for reduction judgments and are held to the
    contract exactly. FP16 is ``optional``, and it cannot be held to the declared
    1e-3: measured at 64x6656 the kernel and the gold differ on 7 elements in
    425984 by 3.91e-3, which is one FP16 ULP at the 5.5 magnitude where it lands.
    The cause is the reduction order, per the kernel module's docstring, surfacing
    through an FP16 tie in the residual add.

    What a gold comparison cannot do, at any dtype, is police the eps values. I
    measured the worst violation ratio at 64x6656, where 1.0 is the bar:

        dtype   correct kernel   post_eps and pre_eps exchanged
        fp32    0.01             0.24
        bf16    0.17             0.36
        fp16    1.92             1.92   (against the unrelaxed row)

    Swapping 1e-5 for 1e-8 moves a unit-variance row far less than any declared
    row allows, so no tolerance comparison here would notice. That check belongs
    where the signal is visible: ``test_fp64_check_rejects_swapped_eps`` in
    ``test_fused_add_norm.py`` scales rows down until the eps term dominates.
    ``test_eps_values_reach_the_kernel`` below covers the cruder failure of eps
    being ignored or hardcoded, which is detectable bitwise.

    The FP16 relaxation costs nothing measurable. Injecting three plausible kernel
    bugs and asking whether the comparison notices:

        injected bug                  fp16 relaxed   bf16 at contract
        dropped the (1 + w) offset    caught         caught
        post and pre weights swapped  caught         caught
        lost the mid narrowing        missed         missed

    So the relaxed FP16 bar sees exactly what the required BF16 bar sees. The
    narrowing is the one both miss, and it has its own bitwise test against an
    eager composition in ``test_fused_add_norm.py``.
    """
    atol, rtol = _tol(judgment, dtype)
    if not _is_required(judgment, dtype):
        atol = max(atol, _one_ulp(want))
    torch.testing.assert_close(
        got.cpu().float(),
        want.cpu().float(),
        atol=atol,
        rtol=rtol,
        msg=lambda built: f"{label}{built}" if label else built,
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
def test_triton_backward_is_batch_invariant_bitwise():
    """Axis A applies to gradients too, not just the forward."""
    width, big = HIDDEN_SIZE, 16
    dtype = torch.float32
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
_OVERFLOW_GIB = _OVERFLOW_ROWS * HIDDEN_SIZE * 2 * 4 / 1024**3


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
