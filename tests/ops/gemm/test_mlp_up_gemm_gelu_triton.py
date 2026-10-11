# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Invariance + accuracy tests for the Triton mlp_up_gemm_gelu backend (WS1).

Covers the row contract ``mlp-up-gemm-gelu-mma`` as implemented by
``rl_engine.backends.shared.triton.gemm.mlp_up_gemm_gelu``:

* API and fail-closed behaviour (fp32 input, non-CUDA input, K mismatch, bias
  length mismatch, fp32 bias),
* determinism across repeated runs,
* row invariance -- a logical row's bytes must not depend on ``M``, on the batch
  position or on the launch geometry -- and tiling invariance, both bitwise,
* the fp32 ``pre``: byte-equal to the Hopper CUDA backend's on the same operands
  (both implement the same mma order) and inside a tight fp32-ulp bound of the
  fp32 CPU reference's tree (the two associate the K reduction differently),
* the GELU value against the reference under the declared tolerances, with the
  deviation shown to be confined to the tanh,
* backward: the gate against ``mlp_up_gemm_gelu_reference_gate``, then ``dx`` /
  ``dW`` / ``db`` against the reference tree computed with the *device's own*
  gate (so the transcendental is out of the contraction comparison), and ``db``
  against the ascending-row fp32 fold,
* and byte equality with the Hopper CUDA kernel, because both implement the
  same pinned schedule (ascending k-chunks chained into one FP32 accumulator per
  output element, no split-K, bias once in FP32, one bf16 cast at the store, the
  same ``tanhf``-based GELU and gate) -- which is the
  ``mlp-up-gemm-gelu-mma`` contract, *not* the row's other (portable fp32 tree)
  contract, so that class pins ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=hopper`` and
  skips without the SM90 build.

The fp32 CPU tree walks K python-level steps, each one a small vectorized fp64
``[M, N]`` op -- O(M * K * N) element-ops, measured at ~1 s per row at
K = 3072/N = 12288 (a 256-row call is minutes). Every reference call here is
therefore bounded to ``TREE_REFERENCE_ROWS`` rows: the ``pre`` anchor, the
accuracy gate and the backward comparison run against one cached bounded slice,
and the 6889/6032-token tiers use the portable tree path as their oracle
(``test_backward_at_the_reference_shapes``), which is byte-equal to that
reference at the anchor and depends only on K. The reference runs under
``_reference_threads`` (its K small elementwise fp64 ops would otherwise pay a
pool barrier each with torch's default 128-thread intra-op pool); the thread
count cannot change a byte of the result.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

from rl_engine.reference.gemm.mlp_up_gemm_gelu import (
    gelu_tanh_argument,
    gelu_tanh_fp32,
    left_fold_bias_gradient,
    mlp_up_gemm_gelu_reference_backward,
    mlp_up_gemm_gelu_reference_forward,
    mlp_up_gemm_gelu_reference_gate,
    mlp_up_gemm_gelu_reference_pre,
)

try:
    import triton  # noqa: F401

    from rl_engine.backends.shared.triton.gemm.mlp_up_gemm_gelu import (
        MLP_UP_GEMM_GELU_CONTRACT,
        TRITON_BACKEND_IMPL,
        TritonMlpUpGemmGeluOp,
        _launch_forward,
        _launch_gate,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - environment without Triton
    _HAS_TRITON = False


def _hopper_backend():
    """The Hopper (``mlp-up-gemm-gelu-mma``) CUDA backend, or ``None`` if unusable.

    The byte-equality claim is about the *hardware-order* contract: Triton and
    the Hopper TMA + wgmma kernel are two implementations of it. The portable
    fp32 tree kernel in the same extension is the row's other contract and is
    deliberately *not* byte-equal to either (it is byte-equal to the fp32 CPU
    reference instead), so it is not what this file pins.
    """

    try:
        from rl_engine.backends.cuda.gemm.mlp_up_gemm_gelu import (
            CudaMlpUpGemmGeluOp,
            mma_backend_available,
            sm90_backend_compiled,
        )
    except ImportError:  # pragma: no cover - no CUDA backend in this build
        return None
    if not (mma_backend_available() and sm90_backend_compiled()):
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return None
    return CudaMlpUpGemmGeluOp()


_HOPPER_BACKEND = _hopper_backend()
_HOPPER_REASON = (
    "the Hopper mlp-up-gemm-gelu-mma path needs KERNEL_ALIGN_FORCE_SM90=1 and a cc 9.0 device"
)

pytestmark = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="the Triton mlp_up_gemm_gelu backend needs Triton and a CUDA GPU",
)
BYTE_EXACT = pytest.mark.skipif(_HOPPER_BACKEND is None, reason=_HOPPER_REASON)


@contextmanager
def _pinned(name):
    """Run the block with ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND`` pinned."""

    switch = "RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND"
    previous = os.environ.get(switch)
    os.environ[switch] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(switch, None)
        else:
            os.environ[switch] = previous


# --- declared contract bounds (see the row's docs page) ----------------------
BF16_ULP = 2.0**-8
BIT_EXACT_MAX_K = 128
DECLARED_MIN_IDENTICAL_FRACTION = 0.99
DECLARED_MAX_ULPS = 8.0
GATE_MIN_IDENTICAL_FRACTION = 0.99
GATE_MAX_ULPS = 2.0
GELU_MAX_ULPS = 2.0

# The mma order (ascending k16 chunks chained into one fp32 accumulator) and the
# reference's 32-wide-leaf mid-split tree associate the same sum differently, so
# the Triton ``pre`` is *not* byte-equal to the reference tree's: measured on the
# H100 at K = 3072, only ~0.5% of elements are bit-identical and the worst element
# is ~40 fp32 ulps from the reference (ulps of the reference's largest magnitude,
# the scale `_deviation` uses) -- the expected size of an accumulation-order
# difference, which grows like sqrt(K) rounding steps and is still two orders of
# magnitude tighter than the bf16 store tolerance. ``sm90 pre == triton pre`` *is*
# byte-equal, which :class:`TestCudaByteEquality` asserts.
FP32_MAX_ULPS = 64.0

# --- the model's own geometry (issue #386) -----------------------------------
# MMDiT MLP 3072 -> 12288 -> 3072 with bias, so both stream up projections are
# [*, 3072] -> [*, 12288] (the down row is the mirrored [*, 12288] -> [*, 3072]);
# the reference shapes 1024^2 / 1328^2 / 1664x928 pack 16x16 pixels per token,
# i.e. 4096 / 6889 / 6032 image tokens, and the sigma schedule anchors 256
# tokens. Every shape keeps the model's K and N; the token count is a reference
# tier, the anchor, or one of the two synthetic cases the RFC allows (a reduction
# below the mma chain, and odd tails).
MODEL_K = 3072
MODEL_N = 12288
REF_TOKENS = (4096, 6889, 6032)  # 1024^2, 1328^2, 1664x928 image tokens
ANCHOR_TOKENS = 256  # the sigma schedule's low-token anchor

IMG_MLP_UP_SHAPE = (REF_TOKENS[0], MODEL_K, MODEL_N)  # img_mlp.net.0 / net.1
SMALL = (64, MODEL_K, MODEL_N)  # short token count, the model's K and N
SHORT_K = (64, 96, MODEL_N)  # below the mma chain length: bit-exactness must hold
GATE_SHAPE = (ANCHOR_TOKENS, MODEL_K, MODEL_N)  # what the fp32 CPU gate affords
BACKWARD_GATE_SHAPE = (32, MODEL_K, MODEL_N)

# The fp32 CPU tree walks K python-level steps of [M, N] fp64 -- O(M * K * N)
# element-ops, measured at ~1 s per row at K = 3072/N = 12288 -- so it never runs
# on a whole tier: every call is bounded to TREE_REFERENCE_ROWS rows, and the
# 6889/6032-token tiers use the portable tree path as their oracle instead
# (``test_backward_at_the_reference_shapes``). One cache entry per (kind, K, N,
# rows) keeps the anchor, the accuracy gate and the backward check from paying for
# the same tree twice, and `_inputs` draws its bounded slice from seeded
# generators so every shape at a given K/N shares it.
TREE_REFERENCE_ROWS = 16
_REFERENCE_CACHE: dict = {}


def _inputs(shape, dtype=torch.bfloat16, seed=0):
    """Model-like operands whose bounded *slice* does not depend on the tokens.

    ``x``, ``weight`` and ``bias`` are drawn from three generators seeded the same
    way, so ``x[:rows]``, ``weight`` and ``bias`` are a function of
    ``(seed, k_dim, n_dim)`` alone: shapes that differ only in their token count
    have identical operands in any bounded slice, which is what lets one cached
    fp32-reference computation serve every shape at a given K/N.

    Model-like scales: standard-normal activations and fan-in-normalized weights
    keep the reference pre-activation O(1), so the tolerance below is expressed in
    ulps of the reference magnitude.
    """

    rows, k_dim, n_dim = shape
    x = torch.randn(rows, k_dim, generator=torch.Generator().manual_seed(seed))
    weight = torch.randn(n_dim, k_dim, generator=torch.Generator().manual_seed(seed + 1))
    bias = torch.randn(n_dim, generator=torch.Generator().manual_seed(seed + 2))
    return (
        x.to(dtype).cuda(),
        (weight / (k_dim**0.5)).to(dtype).cuda(),
        (bias / (k_dim**0.5)).to(dtype).cuda(),
    )


def _saturating_inputs(shape, seed=0):
    """Operands whose pre-activation is large enough to saturate the GELU tanh.

    ``tanh(S t)`` is exactly ``+-1.0f`` in fp32 only for ``|S t| >= ~9``, which
    the model-like scales above never reach (the fan-in-normalized weights make
    the pre-activation O(1)). Scaling the weight rows by 16 makes ``pre`` O(16),
    so roughly half of the elements sit in the saturated region -- which is where
    the GELU value and the gate are polynomial-exact and must match the reference
    bit for bit.
    """

    rows, k_dim, n_dim = shape
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k_dim, generator=gen)
    weight = torch.randn(n_dim, k_dim, generator=gen) / (k_dim**0.5) * 16.0
    bias = torch.randn(n_dim, generator=gen) / (k_dim**0.5) * 16.0
    return (
        x.to(torch.bfloat16).cuda(),
        weight.to(torch.bfloat16).cuda(),
        bias.to(torch.bfloat16).cuda(),
    )


# The fp32 CPU reference walks K python-level steps, each one a *small* vectorized
# fp64 [M, N] op, and torch's default intra-op pool (128 threads on this host)
# turns every one of those steps into a full barrier: measured, one bounded call
# costs ~20-30 s there against ~6 s pinned. Every op in the reference is
# elementwise (fp64 mul/add/tanh), so the thread count cannot move a rounding
# decision -- this only removes pool overhead from the oracle.
@contextmanager
def _reference_threads():
    """Run the block with the intra-op pool pinned for the fp32 CPU tree."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _run_reference(fn, *args, **kwargs):
    """One fp32 CPU reference call under the reference thread pin."""

    with _reference_threads():
        return fn(*args, **kwargs)


def _reference_forward(x, weight, bias):
    """The fp32 reference GELU value (no output cast)."""

    return _run_reference(
        mlp_up_gemm_gelu_reference_forward,
        x.float().cpu(),
        weight.float().cpu(),
        bias.float().cpu(),
    )


def _reference_pre(x, weight, bias):
    """The fp32 reference pre-activation (the tree plus bias once)."""

    return _run_reference(
        mlp_up_gemm_gelu_reference_pre,
        x.float().cpu(),
        weight.float().cpu(),
        bias.float().cpu(),
    )


def _reference_rows(shape, kind, rows):
    """Cached fp32 CPU reference for ``shape`` (``pre`` or ``forward``).

    The cache key is ``(kind, K, N, rows)`` and *not* the token count: ``_inputs``
    draws its operands from seeded generators, so ``x[:rows]``/``weight``/``bias``
    are identical for every shape that shares K and N, and one bounded computation
    then serves every shape here. ``rows`` is clamped to the shape, and ``forward``
    is derived from the cached ``pre`` (``mlp_up_gemm_gelu_reference_forward`` is
    exactly ``gelu_tanh_fp32`` of ``mlp_up_gemm_gelu_reference_pre``), so no shape
    pays for the K-step tree twice.
    """

    k_dim, n_dim = shape[1], shape[2]
    rows = min(rows, shape[0])
    key = (kind, k_dim, n_dim, rows)
    pre_key = ("pre", k_dim, n_dim, rows)
    if key not in _REFERENCE_CACHE:
        if pre_key not in _REFERENCE_CACHE:
            x, weight, bias = _inputs(shape)
            _REFERENCE_CACHE[pre_key] = (
                _run_reference(
                    mlp_up_gemm_gelu_reference_pre,
                    x[:rows].float().cpu(),
                    weight.float().cpu(),
                    bias.float().cpu(),
                ),
            )
        (pre,) = _REFERENCE_CACHE[pre_key]
        if kind == "forward":
            with _reference_threads():
                pre = gelu_tanh_fp32(pre)
        _REFERENCE_CACHE[key] = (pre,)
    return _REFERENCE_CACHE[key]


def _grad_of(shape, rows=None, seed=11):
    rows = shape[0] if rows is None else rows
    gen = torch.Generator().manual_seed(seed)
    return (
        (torch.randn(rows, shape[2], generator=gen) / (shape[2] ** 0.5)).to(torch.bfloat16).cuda()
    )


def _pre_of(x, weight, bias=None):
    """The Triton kernel's fp32 ``pre`` (same launch the forward uses)."""

    return _launch_forward(x.reshape(-1, x.size(-1)).contiguous(), weight, bias, True)[1]


def _gate_of(grad, pre):
    """The Triton gate: ``bf16(f32(grad) * gelu_tanh_grad_fp32(pre))``."""

    return _launch_gate(grad.contiguous(), pre.contiguous())


def _deviation(got, ref):
    """Declared tolerance metric: the output is compared against the
    correctly-rounded *bf16* reference (comparing a bf16 store against the raw
    fp32 value would count zero matches for any kernel), with the ulp scale
    taken from the fp32 reference magnitude."""

    ref_bf16 = ref.to(torch.bfloat16)
    got_f, ref_f = got.detach().float().cpu(), ref_bf16.float().cpu()
    identical = float((got_f == ref_f).float().mean())
    ulp = BF16_ULP * ref.float().abs().max().clamp_min(1e-12)
    worst = float((got_f - ref_f).abs().max() / ulp)
    return identical, worst


def _gate_deviation(got, ref):
    """Identity/ulp of the gate against the correctly-rounded reference gate."""

    got_f, ref_f = got.detach().float().cpu(), ref.to(torch.bfloat16).float().cpu()
    identical = float((got_f == ref_f).float().mean())
    ulp = BF16_ULP * ref_f.abs().max().clamp_min(1e-12)
    worst = float((got_f - ref_f).abs().max() / ulp)
    return identical, worst


def _fp32_ulp_of_max_magnitude(t: torch.Tensor) -> float:
    """One fp32 ulp of the tensor's largest magnitude."""

    import math

    return 2.0 ** (math.floor(math.log2(float(t.float().abs().max().clamp_min(1e-30)))) - 23)


def _pre_deviation(got: torch.Tensor, ref: torch.Tensor):
    """Identity and worst fp32-ulp deviation of two fp32 pre-activations.

    The mma order is a different association of the same sum than the reference's
    tree, so the identity fraction is *reported* rather than asserted to be 1; the
    worst element must stay inside the fp32 bound. Both operands are compared on
    the host (the reference is a CPU tensor, the device side is not).
    """

    got_c, ref_c = got.detach().float().cpu(), ref.detach().float().cpu()
    identical = float((got_c == ref_c).float().mean())
    worst = float((got_c.double() - ref_c.double()).abs().max() / _fp32_ulp_of_max_magnitude(ref_c))
    return identical, worst


def _assert_identical_tolerance(got, ref, label):
    identical, worst = _deviation(got, ref)
    assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, (
        f"{label}: only {identical * 100:.4f}% of elements are bit-identical to the "
        f"reference (declared >= {DECLARED_MIN_IDENTICAL_FRACTION * 100:.0f}%)"
    )
    assert worst <= DECLARED_MAX_ULPS, (
        f"{label}: worst element is {worst:.4f} bf16-ulps from the reference "
        f"(declared <= {DECLARED_MAX_ULPS})"
    )
    return identical, worst


def _assert_same_bytes(actual, expected, label):
    assert actual.shape == expected.shape, f"{label}: {actual.shape} vs {expected.shape}"
    assert actual.dtype == expected.dtype, f"{label}: {actual.dtype} vs {expected.dtype}"
    assert actual.is_contiguous() and expected.is_contiguous()
    assert torch.equal(
        actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    ), f"{label}: raw bytes differ"


def _byte_mismatches(got, want):
    """Number of differing raw bytes (dtype agnostic: fp32 pre or bf16 stores)."""

    a = got.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    b = want.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    assert a.shape == b.shape, f"shape mismatch: {got.shape} vs {want.shape}"
    return int((a != b).sum())


def _grads(op, x, weight, bias, grad):
    """Run backward through the op and return ``(dx, dW, db)``."""

    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = bias.detach().clone().requires_grad_(True)
    op(x, weight, bias=bias).backward(grad)
    return x.grad, weight.grad, bias.grad


# ---------------------------------------------------------------------------
# API and fail-closed behaviour
# ---------------------------------------------------------------------------
class TestApi:
    def test_class_flags(self):
        op = TritonMlpUpGemmGeluOp()
        assert op.op_class == "reduction"
        assert op.is_batch_invariant is True
        assert MLP_UP_GEMM_GELU_CONTRACT == "mlp-up-gemm-gelu-mma"
        assert TRITON_BACKEND_IMPL == "triton_mlp_up_gemm_gelu_pinned_config"

    def test_forward_shape_dtype_and_lead_dims(self):
        x, weight, bias = _inputs((7, 96, MODEL_N))
        op = TritonMlpUpGemmGeluOp()
        out = op(x, weight, bias=bias)
        assert out.shape == (7, MODEL_N)
        assert out.dtype is torch.bfloat16
        flat = op(x.reshape(-1, 96), weight, bias=bias)
        assert torch.equal(out, flat.reshape(7, MODEL_N))
        lead = op(x.reshape(7, 1, 96), weight, bias=bias)
        assert lead.shape == (7, 1, MODEL_N)
        assert torch.equal(lead, out.unsqueeze(1))

    def test_fail_closed_on_fp32_input(self):
        x, weight, bias = _inputs(SMALL, dtype=torch.float32)
        with pytest.raises(ValueError, match="bf16"):
            TritonMlpUpGemmGeluOp()(x, weight, bias=bias)

    def test_fail_closed_on_non_cuda_device(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(RuntimeError, match="accelerator"):
            TritonMlpUpGemmGeluOp()(x.cpu(), weight.cpu(), bias=bias.cpu())

    def test_fail_closed_on_k_mismatch(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(ValueError, match="K must match"):
            TritonMlpUpGemmGeluOp()(x, weight[:, :-16].contiguous(), bias=bias)

    def test_fail_closed_on_bias_length_mismatch(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(ValueError, match="bias length"):
            TritonMlpUpGemmGeluOp()(x, weight, bias=bias[:-1].contiguous())

    def test_fail_closed_on_fp32_bias(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(ValueError, match="bias must be bf16"):
            TritonMlpUpGemmGeluOp()(x, weight, bias=bias.float())

    def test_strided_bias_is_read_at_the_right_elements(self):
        """A view bias must produce the same bytes as its contiguous copy.

        The kernel indexes the bias at unit stride, so the launcher has to make it
        contiguous (the CUDA backend converts too). Before that, a stride-2 bf16
        bias was read at the wrong elements and the forward silently returned
        wrong values -- and disagreed with the CUDA backend on the same inputs.
        """

        x, weight, _ = _inputs(SMALL, seed=15)
        bias = torch.randn(SMALL[2] * 2, device="cuda", dtype=torch.bfloat16)[::2]
        assert bias.shape == (SMALL[2],) and bias.stride(0) == 2
        op = TritonMlpUpGemmGeluOp()
        got = op(x, weight, bias=bias)
        want = op(x, weight, bias=bias.contiguous())
        _assert_same_bytes(got, want, "strided bias forward")

    def test_no_bias_matches_reference(self):
        # Bounded: the fp32 CPU tree costs ~1 s per row at the model's K/N.
        checked = TREE_REFERENCE_ROWS
        x, weight, _ = _inputs(SMALL)
        got = TritonMlpUpGemmGeluOp()(x, weight)
        ref = _run_reference(
            mlp_up_gemm_gelu_reference_forward,
            x[:checked].float().cpu(),
            weight.float().cpu(),
            torch.zeros(weight.size(0)),
        )
        identical, worst = _deviation(got[:checked], ref)
        assert worst <= 1.0, f"no-bias path: worst={worst:.3f} ulp"
        assert identical >= 0.99, f"no-bias path: identical={identical}"

    def test_pre_matches_the_reference_within_fp32_ulps(self):
        """``pre`` is the anchor, compared under the mma contract's fp32 bound.

        The mma schedule (ascending k16 chunks chained into one fp32 accumulator)
        and the reference's 32-wide-leaf mid-split tree associate the same sum
        differently, so their fp32 pre-activations are not byte-equal (measured at
        K = 3072: ~0.5% of elements identical, worst element ~40 fp32 ulps from the
        reference -- a sqrt(K)-step accumulation-order difference). Byte equality
        of ``pre`` is the *tree* contract's claim, and between the two mma
        implementations (Triton vs Hopper, :class:`TestCudaByteEquality`). The
        identity fraction is reported next to the bound.

        The fp32 CPU tree costs ~1 s per row at the model's K/N, so the comparison
        is the cached ``TREE_REFERENCE_ROWS`` slice -- one cache entry, shared with
        the accuracy gate and the backward check below.
        """

        shape = (ANCHOR_TOKENS, MODEL_K, MODEL_N)
        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _inputs(shape)
        pre = _pre_of(x, weight, bias)
        assert pre.shape == (shape[0], MODEL_N) and pre.dtype is torch.float32
        (ref,) = _reference_rows(shape, "pre", checked)
        identical, worst = _pre_deviation(pre[:checked], ref)
        assert worst <= FP32_MAX_ULPS, (
            f"pre: {worst:.2f} fp32 ulps from the reference tree "
            f"({identical * 100:.2f}% of elements bit-identical)"
        )

    def test_pre_matches_the_reference_without_bias(self):
        shape = (SMALL[0], MODEL_K, MODEL_N)
        # Bounded like every reference call here (the fp32 CPU tree costs ~1 s per
        # row at the model's K/N) and under the mma contract's fp32-ulp bound above.
        checked = TREE_REFERENCE_ROWS
        x, weight, _ = _inputs(shape)
        pre = _pre_of(x, weight, None)
        ref = _run_reference(
            mlp_up_gemm_gelu_reference_pre,
            x[:checked].float().cpu(),
            weight.float().cpu(),
            None,
        )
        identical, worst = _pre_deviation(pre[:checked], ref)
        assert worst <= FP32_MAX_ULPS, (
            f"pre (no bias): {worst:.2f} fp32 ulps from the reference tree "
            f"({identical * 100:.2f}% of elements bit-identical)"
        )

    def test_emit_pre_false_does_not_materialize_pre(self):
        x, weight, bias = _inputs(SMALL)
        x2d = x.reshape(-1, MODEL_K)
        out_with, pre_with = _launch_forward(x2d, weight, bias, True)
        assert pre_with.numel() == x.size(0) * MODEL_N and pre_with.dtype is torch.float32
        out_without, pre_without = _launch_forward(x2d, weight, bias, False)
        assert pre_without.numel() == 0, "emit_pre=False must not write pre"
        # the store is a pure function of the pre-activation: same bytes, no pre
        _assert_same_bytes(out_without, out_with, "emit_pre=False forward")

    def test_emit_pre_false_matches_the_op_under_no_grad(self):
        x, weight, bias = _inputs(SMALL)
        out_grad = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        out_plain, pre = _launch_forward(x.reshape(-1, MODEL_K), weight, bias, False)
        assert pre.numel() == 0
        _assert_same_bytes(out_plain, out_grad, "inference forward")


# ---------------------------------------------------------------------------
# determinism + invariance (bitwise)
# ---------------------------------------------------------------------------
class TestInvariance:
    def test_forward_deterministic_over_three_reruns(self):
        x, weight, bias = _inputs(SMALL)
        op = TritonMlpUpGemmGeluOp()
        first = op(x, weight, bias=bias)
        for _ in range(3):
            assert torch.equal(op(x, weight, bias=bias), first)

    def test_forward_row_invariant(self):
        x, weight, bias = _inputs(SMALL)
        op = TritonMlpUpGemmGeluOp()
        full = op(x, weight, bias=bias)
        for rows in (1, 7, x.size(0) - 1):
            part = op(x[:rows].contiguous(), weight, bias=bias)
            assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_forward_tiling_invariant(self):
        """A padded M tile must not leak into a logical row's bytes."""

        x, weight, bias = _inputs((130, 96, MODEL_N))
        op = TritonMlpUpGemmGeluOp()
        base = op(x[:64].contiguous(), weight, bias=bias)
        padded = op(x, weight, bias=bias)
        assert torch.equal(padded[:64], base)

    def test_forward_model_shape_row_invariant(self):
        """The model's K/N with a partial tile and a single token."""

        x, weight, bias = _inputs((133, MODEL_K, MODEL_N))
        op = TritonMlpUpGemmGeluOp()
        full = op(x, weight, bias=bias)
        for rows in (1, 7, 132):
            assert torch.equal(op(x[:rows].contiguous(), weight, bias=bias), full[:rows])

    def test_pre_row_and_tiling_invariant(self):
        """The fp32 anchor is row- and tile-invariant just like the store."""

        x, weight, bias = _inputs((130, 96, MODEL_N))
        full = _pre_of(x, weight, bias)
        for rows in (1, 7, 64):
            assert torch.equal(_pre_of(x[:rows].contiguous(), weight, bias), full[:rows])
            assert torch.equal(op_pre(x[:rows].contiguous(), weight, bias), full[:rows])

    def test_backward_deterministic(self):
        x, weight, bias = _inputs(SMALL)
        grad = _grad_of(SMALL, seed=13)
        op = TritonMlpUpGemmGeluOp()
        first = _grads(op, x, weight, bias, grad)
        for _ in range(3):
            again = _grads(op, x, weight, bias, grad)
            assert all(torch.equal(a, b) for a, b in zip(again, first))

    def test_dx_row_invariant(self):
        x, weight, _ = _inputs(SMALL)
        grad = _grad_of(SMALL, seed=14)
        op = TritonMlpUpGemmGeluOp()
        bias = torch.zeros(SMALL[2]).to(torch.bfloat16).cuda()
        full, _, _ = _grads(op, x, weight, bias, grad)
        for rows in (1, 7, SMALL[0] - 1):
            part, _, _ = _grads(op, x[:rows].contiguous(), weight, bias, grad[:rows].contiguous())
            assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_dw_padding_invariant(self):
        """Zero-padded batch rows contribute exact zeroes, never rounding drift."""

        x, weight, _ = _inputs((64, 96, MODEL_N))
        grad = _grad_of((64, 96, MODEL_N), seed=15)
        op = TritonMlpUpGemmGeluOp()
        bias = torch.zeros(MODEL_N).to(torch.bfloat16).cuda()
        _, base_dw, _ = _grads(op, x[:7].contiguous(), weight, bias, grad[:7].contiguous())
        padded_x = torch.zeros_like(x)
        padded_x[:7] = x[:7]
        padded_grad = torch.zeros_like(grad)
        padded_grad[:7] = grad[:7]
        _, padded_dw, _ = _grads(op, padded_x, weight, bias, padded_grad)
        assert torch.equal(padded_dw, base_dw)

    def test_db_padding_invariant(self):
        x, weight, bias = _inputs((24, MODEL_K, MODEL_N))
        grad = _grad_of((24, MODEL_K, MODEL_N), seed=16)
        op = TritonMlpUpGemmGeluOp()
        _, _, base_db = _grads(op, x[:9].contiguous(), weight, bias, grad[:9].contiguous())
        padded = torch.zeros_like(grad)
        padded[:9] = grad[:9]
        _, _, padded_db = _grads(op, x, weight, bias, padded)
        assert torch.equal(padded_db, base_db)


def op_pre(x, weight, bias=None):
    """The op's fp32 ``pre`` through its parity helper."""

    op = TritonMlpUpGemmGeluOp()
    return op._forward_with_pre(x, weight, bias=bias)[1]


# ---------------------------------------------------------------------------
# accuracy against the independent fp32 reference
# ---------------------------------------------------------------------------
class TestAccuracy:
    # synthetic short reductions: the model's own length is covered at K = 3072 below
    @pytest.mark.parametrize("k_dim", [16, 32, 48, 96, 128])
    def test_short_k_agrees_within_one_ulp(self, k_dim):
        """Below the mma chain length both orders agree to under one bf16 ulp.

        Not to byte equality: the tensor core's k16 grouping and the reference's
        correctly-rounded FMA chain may differ in the last fp32 bit of ``pre``,
        which only moves an element's bf16 rounding when it sits within ~1e-6 of
        a boundary.
        """

        assert k_dim <= BIT_EXACT_MAX_K
        x, weight, bias = _inputs((32, k_dim, MODEL_N))
        got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        identical, worst = _deviation(got, _reference_forward(x, weight, bias))
        assert worst <= 1.0, f"K={k_dim}: worst={worst:.3f} ulp"
        assert identical >= 0.99, f"K={k_dim}: identical={identical}"

    def test_model_shape_matches_reference_within_declared_tolerance(self):
        """The model's K/N at the 256-token anchor; the fp32 reference covers a slice.

        The reference cannot run a whole tier (it costs ~1 s per row), so it runs on
        the cached ``TREE_REFERENCE_ROWS`` slice and the device output is compared
        over those rows. The slice's ``forward`` is derived from its cached ``pre``,
        so this shares the anchor's single K-step tree.
        """

        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _inputs(GATE_SHAPE)
        got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        assert got.dtype is torch.bfloat16
        (ref,) = _reference_rows(GATE_SHAPE, "forward", checked)
        _assert_identical_tolerance(got[:checked], ref, "forward")

    def test_deviation_is_confined_to_the_tanh(self):
        """Fed the device's own ``pre``, only the tanh can move the GELU store.

        Where ``|S t| >= 9`` the fp32 tanh is exactly ``+-1.0f`` for both the
        correctly-rounded reference and Triton's ``libdevice.tanh``; the GELU then
        collapses to the pinned polynomial and must match bit for bit. This is the
        *mma* contract, so the pre-activation itself is a different association of
        the sum than the reference tree's: it is compared under ``FP32_MAX_ULPS``
        here (with the identity fraction reported), and the GELU reference is fed
        the device's own ``pre``, which isolates the transcendental as the single
        free operand -- the claim under test. Mismatches may only live outside the
        saturated region. Both parts run on the bounded ``TREE_REFERENCE_ROWS``
        slice (the fp32 CPU tree costs ~1 s per row at the model's K/N).
        """

        shape = GATE_SHAPE
        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _saturating_inputs(shape)
        got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        pre = _pre_of(x, weight, bias)
        ref_pre = _reference_pre(x[:checked], weight, bias)
        pre_identical, worst_pre = _pre_deviation(pre[:checked], ref_pre)
        assert worst_pre <= FP32_MAX_ULPS, (
            f"pre: {worst_pre:.2f} fp32 ulps from the reference tree "
            f"({pre_identical * 100:.2f}% of elements bit-identical)"
        )
        device_pre = pre[:checked].float().cpu()
        want = gelu_tanh_fp32(device_pre).bfloat16()
        # 12 is safely past tanh's saturation point: tanh(12) = 1 - 1.1e-10, which
        # rounds to exactly 1.0f for both libdevice's tanh and the correctly-rounded
        # reference (whereas near 9.0 the two could still land on different bits).
        saturated = gelu_tanh_argument(device_pre).abs() >= 12.0
        assert bool(saturated.any()), "the saturation mask must be non-empty"
        differ = got[:checked].cpu().bfloat16() != want
        assert int((differ & saturated).sum()) == 0, (
            f"{int((differ & saturated).sum())} saturated elements differ; the "
            "deviation is not confined to the tanh"
        )
        identical, worst = _deviation(got[:checked], gelu_tanh_fp32(device_pre))
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"identical={identical}"
        assert worst <= GELU_MAX_ULPS, f"worst={worst:.3f} ulp outside the tanh"
        free = int((differ & ~saturated).sum())
        assert free <= max(
            1, int(0.05 * int((~saturated).sum()))
        ), f"too many non-saturated mismatches: {free}"

    def test_seed_permutations_hold_the_same_bounds(self):
        # Bounded slices (one fp32 CPU tree per seed, ~1 s per row at the model's K/N).
        checked = TREE_REFERENCE_ROWS
        for seed in (1, 2, 3):
            x, weight, bias = _inputs(SMALL, seed=seed)
            got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
            ref = _run_reference(
                mlp_up_gemm_gelu_reference_forward,
                x[:checked].float().cpu(),
                weight.float().cpu(),
                bias.float().cpu(),
            )
            _assert_identical_tolerance(got[:checked], ref, f"forward seed={seed}")


# ---------------------------------------------------------------------------
# correctness against the exact result, not just the fp32 model
# ---------------------------------------------------------------------------
def _ulp_of_max_magnitude(t: torch.Tensor) -> float:
    """One bf16 ulp of the tensor's largest magnitude (the contract's absolute scale)."""

    import math

    return 2.0 ** (math.floor(math.log2(float(t.float().abs().max()))) - 7)


class TestExactTruth:
    """The fp32 model cannot be the only oracle: it shares any structural mistake.

    Byte-equality with the reference is self-consistency -- it cannot see a
    transposed weight, a dropped leaf or a wrong cast if the model has it too,
    and it says nothing about the accuracy of the fp32 accumulation order. These
    tests pin the structural exactness of the anchor directly, then the GELU
    value and the contractions against oracles computed independently of both
    implementations.
    """

    def test_integer_inputs_make_the_anchor_bit_exact(self):
        """Small integers make fp32 accumulation exact, so ``pre`` must be exact.

        ``|x|, |W| <= 3`` over ``K = 3072`` keeps every partial sum below
        ``2**24``, i.e. representable, so the sum is order independent: the output
        must equal the exact dot product bit for bit, which no transposed weight,
        mis-indexed column or dropped term can survive.
        """

        rows, k_dim, n_dim = 32, MODEL_K, MODEL_N
        gen = torch.Generator().manual_seed(7)
        xi = torch.randint(-3, 4, (rows, k_dim), generator=gen).float()
        wi = torch.randint(-3, 4, (n_dim, k_dim), generator=gen).float()
        bi = torch.randint(-8, 9, (n_dim,), generator=gen).float()
        x, weight, bias = xi.bfloat16().cuda(), wi.bfloat16().cuda(), bi.bfloat16().cuda()
        pre = _pre_of(x, weight, bias)
        pre_nobias = _pre_of(x, weight, None)
        exact = (xi.double() @ wi.double().T).float()
        assert _byte_mismatches(pre, (exact.double() + bi.double()).float().cuda()) == 0
        assert _byte_mismatches(pre_nobias, exact.cuda()) == 0

    def test_full_k_identity_and_bias(self):
        k_dim, n_dim = MODEL_K, 8
        x = torch.ones(3, k_dim, dtype=torch.bfloat16).cuda()
        weight = torch.ones(n_dim, k_dim, dtype=torch.bfloat16).cuda()
        # every term is 1 and every partial sum is an exact small integer, so the
        # total is exactly K in any association order: a dropped or doubled leaf
        # moves it, and the fp32 store of K is itself exact
        pre = _pre_of(x, weight)
        assert torch.equal(pre, torch.full((3, n_dim), float(k_dim), dtype=torch.float32).cuda())
        bias = torch.arange(n_dim, dtype=torch.float32).bfloat16().cuda()
        want = (torch.tensor(float(k_dim)) + bias.float()).expand(3, n_dim).contiguous()
        assert _byte_mismatches(_pre_of(x, weight, bias), want.cuda()) == 0
        assert (
            _byte_mismatches(
                _pre_of(torch.zeros_like(x), weight, bias),
                bias.float().expand(3, n_dim).contiguous().cuda(),
            )
            == 0
        )

    @pytest.mark.parametrize("k_dim", [3072, 3071, 3073])
    def test_one_hot_covers_every_leaf_boundary(self, k_dim):
        """Every leaf's first and last reduction index must contribute exactly once."""

        weight = torch.ones(1, k_dim, dtype=torch.bfloat16).cuda()
        positions = sorted(
            {0, k_dim - 1, k_dim // 2}
            | {
                i * 32 + off
                for i in range((k_dim + 31) // 32)
                for off in (0, 31)
                if i * 32 + off < k_dim
            }
        )
        for k in positions:
            x = torch.zeros(1, k_dim, dtype=torch.bfloat16).cuda()
            x[0, k] = 1.0
            assert _pre_of(x, weight).float()[0, 0].item() == 1.0, f"k={k} not counted exactly once"

    @pytest.mark.parametrize(
        "shape",
        [SHORT_K, GATE_SHAPE] + [(tokens, MODEL_K, MODEL_N) for tokens in REF_TOKENS],
    )
    def test_gelu_matches_the_fp64_truth(self, shape):
        """The GELU value against an oracle outside both implementations.

        The exact fp64 pre-activation is rounded once to fp32 and put through the
        reference's pinned sequence; the device differs only by its ``tanhf``, so
        the bf16 store must agree except near a tie. Bounds keep headroom over
        the measurement.
        """

        x, weight, bias = _inputs(shape)
        got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        exact_pre = ((x.double() @ weight.double().T).float() + bias.float()).float()
        truth = gelu_tanh_fp32(exact_pre).bfloat16()
        deviation = (got.float().double() - truth.float().double()).abs() / _ulp_of_max_magnitude(
            truth
        )
        assert (
            deviation.max().item() <= GELU_MAX_ULPS
        ), f"{shape}: worst {deviation.max().item():.3f} ulp"
        identical = float((got == truth).float().mean())
        assert identical >= 0.99, f"{shape}: only {identical:.4f} bit-identical"

    def test_backward_matches_exact_fp64(self):
        """``dx``/``dW``/``db`` against the contractions of the *device* gate.

        ``dx = gate @ W`` and ``dW = gateᵀ @ x`` are evaluated in fp64 from the
        bf16 gate the kernel produced, so the oracle is independent of both
        implementations' association orders and of the GELU derivative.
        """

        shape = GATE_SHAPE
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, seed=11)
        dx, dW, db = _grads(TritonMlpUpGemmGeluOp(), x, weight, bias, grad)
        gate = _gate_of(grad, _pre_of(x, weight, bias))
        truth_dx = (gate.double() @ weight.double()).bfloat16()
        truth_dw = (gate.double().t() @ x.double()).bfloat16()
        with _reference_threads():
            truth_db = left_fold_bias_gradient(gate.float().cpu()).to(torch.bfloat16).cuda()
        for name, got, want in (("dx", dx, truth_dx), ("dW", dW, truth_dw), ("db", db, truth_db)):
            deviation = (
                got.float().double() - want.float().double()
            ).abs() / _ulp_of_max_magnitude(want)
            assert deviation.max().item() <= 2.0, f"{name}: worst {deviation.max().item():.3f} ulp"
            assert (
                float((got == want).float().mean()) >= 0.99
            ), f"{name} diverges from the exact gradient"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
class TestGate:
    def test_gate_matches_the_reference_gate(self):
        """Same ``pre``, same ``grad_out``; the transcendental is the only free operand."""

        shape = GATE_SHAPE
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape)
        pre = _pre_of(x, weight, bias)
        got = _gate_of(grad, pre)
        ref = _run_reference(mlp_up_gemm_gelu_reference_gate, pre.float().cpu(), grad.float().cpu())
        assert got.ndim == 2 and got.shape == grad.shape and got.dtype is torch.bfloat16
        identical, worst = _gate_deviation(got, ref)
        assert identical >= GATE_MIN_IDENTICAL_FRACTION, f"gate identical={identical}"
        assert worst <= GATE_MAX_ULPS, f"gate worst={worst:.3f} ulp"

    def test_gate_is_tanh_free_where_the_tanh_saturates(self):
        """On the saturated region the gate is the pinned polynomial, byte for byte."""

        shape = GATE_SHAPE
        x, weight, bias = _saturating_inputs(shape)
        grad = _grad_of(shape)
        pre = _pre_of(x, weight, bias)
        got = _gate_of(grad, pre)
        ref = _run_reference(mlp_up_gemm_gelu_reference_gate, pre.float().cpu(), grad.float().cpu())
        saturated = gelu_tanh_argument(pre.float().cpu()).abs() >= 12.0
        assert bool(saturated.any()), "the saturation mask must be non-empty"
        differ = (got.cpu().bfloat16() != ref) & saturated
        assert int(differ.sum()) == 0, (
            f"{int(differ.sum())} saturated gate elements differ; the deviation is not "
            "confined to the tanh"
        )


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
class TestBackward:
    @pytest.mark.parametrize("shape", [SMALL, BACKWARD_GATE_SHAPE])
    def test_gradients_match_reference_within_declared_tolerance(self, shape):
        """Device gradients vs the fp32 tree fed the *device's own* gate, bounded.

        The reference walks K python-level steps plus the two folds (~1 s per row
        at the model's K/N), so it runs on the bounded ``TREE_REFERENCE_ROWS`` slice
        and the device gradients are taken on that same slice: ``dW``/``db`` fold
        over the batch rows, so slicing keeps both sides comparable.
        """

        checked = min(shape[0], TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, checked)
        dx, dw, db = _grads(TritonMlpUpGemmGeluOp(), x[:checked], weight, bias, grad)
        pre = _pre_of(x[:checked], weight, bias)
        gate = _gate_of(grad, pre)
        ref_dx, ref_dw, ref_db = _run_reference(
            mlp_up_gemm_gelu_reference_backward,
            x[:checked].float().cpu(),
            weight.float().cpu(),
            pre.float().cpu(),
            grad.float().cpu(),
            gate=gate.float().cpu(),
        )
        assert dx.dtype is torch.bfloat16 and dw.dtype is torch.bfloat16
        _assert_identical_tolerance(dx, ref_dx, f"dx {shape}")
        _assert_identical_tolerance(dw, ref_dw, f"dW {shape}")
        # the shared fp32 fold: bit-exact after the single bf16 cast
        assert db.dtype is torch.bfloat16
        assert torch.equal(db.cpu(), ref_db.to(torch.bfloat16))

    @pytest.mark.parametrize("rows", [REF_TOKENS[1], REF_TOKENS[2]])
    def test_backward_at_the_reference_shapes(self, rows):
        """Backward at 6889 and 6032 rows (the 1328^2 and 1664x928 tiers).

        The fp32 CPU reference walks K python-level steps over ``[M, N]``, so it
        cannot be run at these token counts. The tier oracle is the *other*
        implementation of the same mma order: the Hopper CUDA kernel, pinned with
        ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=hopper``. Both chain the reduction
        through the same k-chunks into one fp32 accumulator and evaluate the same
        epilogue, so ``dx``/``dW``/``db`` are byte-equal to it (the portable tree
        path is a *different* order and a different gate, so it is not this
        contract's oracle -- it is byte-equal to the fp32 CPU reference instead,
        pinned on bounded slices in ``tests/ops/gemm/test_mlp_up_gemm_gelu.py``). ``db`` is
        additionally checked bit-for-bit against the independent fp32 fold of the
        gate (``left_fold_bias_gradient``), which is O(M*N) and so affordable at
        any M.
        """

        if _HOPPER_BACKEND is None:
            pytest.skip(_HOPPER_REASON)

        x, weight, bias = _inputs((rows, MODEL_K, MODEL_N), seed=31)
        grad = _grad_of((rows, MODEL_K, MODEL_N), seed=32)
        got_dx, got_dw, got_db = _grads(TritonMlpUpGemmGeluOp(), x, weight, bias, grad)
        with _pinned("hopper"):
            want_dx, want_dw, want_db = _grads(_HOPPER_BACKEND, x, weight, bias, grad)
        _assert_same_bytes(got_dx, want_dx, f"dx rows={rows}")
        _assert_same_bytes(got_dw, want_dw, f"dW rows={rows}")
        _assert_same_bytes(got_db, want_db, f"db rows={rows}")

        gate = _gate_of(grad, _pre_of(x, weight, bias))
        with _reference_threads():
            folded = left_fold_bias_gradient(gate.float().cpu()).to(torch.bfloat16).cuda()
        assert torch.equal(got_db, folded), f"triton db rows={rows}"

    def test_db_is_the_ascending_fp32_fold_of_the_gate(self):
        """``db`` is one correctly-rounded FP32 add per gate row, in index order."""

        shape = (24, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, seed=12)
        _, _, db = _grads(TritonMlpUpGemmGeluOp(), x, weight, bias, grad)
        gate = _gate_of(grad, _pre_of(x, weight, bias))
        with _reference_threads():
            folded = left_fold_bias_gradient(gate.float().cpu())
        assert torch.equal(db, folded.to(torch.bfloat16).cuda())

    def test_backward_uses_the_autograd_contract(self):
        """``torch.autograd.grad`` (no ``.grad`` buffer) agrees with ``backward``."""

        x, weight, bias = _inputs(SMALL)
        grad = _grad_of(SMALL, seed=17)
        op = TritonMlpUpGemmGeluOp()
        xr = x.detach().clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        br = bias.detach().clone().requires_grad_(True)
        out = op(xr, wr, bias=br)
        assert out.requires_grad
        grads = torch.autograd.grad(out, (xr, wr, br), grad_outputs=grad)
        expected = _grads(op, x, weight, bias, grad)
        assert all(torch.equal(a, b) for a, b in zip(grads, expected))

    def test_no_bias_skips_the_bias_gradient(self):
        x, weight, _ = _inputs(SMALL)
        grad = _grad_of(SMALL, seed=18)
        xr = x.detach().clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        TritonMlpUpGemmGeluOp()(xr, wr).backward(grad)
        assert xr.grad is not None and wr.grad is not None


# ---------------------------------------------------------------------------
# cross-backend: byte equality with the Hopper hardware-order schedule
# ---------------------------------------------------------------------------
def _cuda_with_pre(x2d, weight, bias=None):
    """``(out, pre)`` from the Hopper CUDA path (call under ``_pinned("hopper")``)."""

    return _HOPPER_BACKEND._forward_with_pre(x2d, weight, bias=bias)


@BYTE_EXACT
class TestCudaByteEquality:
    """Triton and the Hopper kernel implement the same pinned arithmetic.

    Both are ``mlp-up-gemm-gelu-mma``; the CUDA side is pinned with
    ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=hopper`` so this cannot accidentally
    compare Triton against the portable tree contract (which is a different
    order, byte-equal to the fp32 CPU reference rather than to Triton).
    """

    @pytest.mark.parametrize("rows", [1, 7, 129, *REF_TOKENS])
    def test_forward_bytes_equal(self, rows):
        x, weight, bias = _inputs((rows, MODEL_K, MODEL_N), seed=5)
        got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        with _pinned("hopper"):
            want, pre = _cuda_with_pre(x, weight, bias=bias)
        _assert_same_bytes(got, want, f"forward rows={rows}")
        assert (
            _byte_mismatches(_pre_of(x, weight, bias), pre) == 0
        ), f"pre rows={rows} differs across backends"

    def test_forward_bytes_equal_without_bias_and_at_small_k(self):
        x, weight, _ = _inputs((33, MODEL_K, MODEL_N), seed=6)
        got = TritonMlpUpGemmGeluOp()(x, weight)
        with _pinned("hopper"):
            want = _HOPPER_BACKEND(x, weight)
        _assert_same_bytes(got, want, "forward no bias")

    def test_gate_bytes_equal(self):
        shape = (129, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape, seed=10)
        grad = _grad_of(shape, seed=10)
        pre = _pre_of(x, weight, bias)
        got_gate = _gate_of(grad, pre)
        with _pinned("hopper"):
            _, cuda_pre = _cuda_with_pre(x, weight, bias=bias)
            from rl_engine.backends.extension import _C

            want_gate = _C.mlp_up_gemm_gelu_cuda_gate(grad, cuda_pre)
        assert _byte_mismatches(pre, cuda_pre) == 0
        _assert_same_bytes(got_gate, want_gate, "gate")

    @pytest.mark.parametrize("rows", [7, 129, *REF_TOKENS])
    def test_backward_bytes_equal(self, rows):
        x, weight, bias = _inputs((rows, MODEL_K, MODEL_N), seed=7)
        grad = _grad_of((rows, MODEL_K, MODEL_N), seed=21)
        got_dx, got_dw, got_db = _grads(TritonMlpUpGemmGeluOp(), x, weight, bias, grad)
        with _pinned("hopper"):
            want_dx, want_dw, want_db = _grads(_HOPPER_BACKEND, x, weight, bias, grad)
        _assert_same_bytes(got_dx, want_dx, f"dx rows={rows}")
        _assert_same_bytes(got_dw, want_dw, f"dW rows={rows}")
        _assert_same_bytes(got_db, want_db, f"db rows={rows}")

    def test_forward_bytes_equal_at_the_model_shape(self):
        x, weight, bias = _inputs(IMG_MLP_UP_SHAPE, seed=8)
        got = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
        with _pinned("hopper"):
            want = _HOPPER_BACKEND(x, weight, bias=bias)
        _assert_same_bytes(got, want, "forward 4096x3072x12288")

    def test_the_hopper_path_is_the_mma_contract(self):
        """The path Triton is compared against publishes ``mlp-up-gemm-gelu-mma``."""

        from rl_engine.backends.cuda.gemm.mlp_up_gemm_gelu import (
            MMA_CONTRACT,
            mlp_up_gemm_gelu_backend_used,
            mlp_up_gemm_gelu_contract_used,
        )

        x, weight, _ = _inputs((7, MODEL_K, MODEL_N), seed=9)
        with _pinned("hopper"):
            assert mlp_up_gemm_gelu_backend_used(x, weight) == "hopper"
            assert mlp_up_gemm_gelu_contract_used(x, weight) == MMA_CONTRACT
