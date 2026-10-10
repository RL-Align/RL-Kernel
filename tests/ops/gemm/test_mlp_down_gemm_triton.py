# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Invariance + accuracy tests for the Triton mlp_down_gemm backend (WS1).

Covers the row contract ``mlp-down-gemm-mma`` as implemented by
``rl_engine.backends.shared.triton.gemm.mlp_down_gemm``:

* API and fail-closed behaviour (fp32 input, non-CUDA input, K mismatch, bias
  length mismatch),
* determinism across repeated runs,
* row invariance -- a logical row's bytes must not depend on ``M``, on the batch
  position or on the launch geometry -- and tiling invariance, both bitwise,
* accuracy against the independent fp32 CPU reference under the declared
  tolerances (>= 99% bit-identical elements, every element inside 8 bf16 ulps of
  max|reference|),
* backward: ``dx``/``dW`` against the reference tree, ``db`` against the
  ascending-row fp32 fold,
* and byte equality with the Hopper CUDA kernel, because both implement the
  same pinned schedule (ascending k-chunks of 16 chained into one accumulator
  per output element, FP32 accumulator, no split-K, bias once in FP32, one bf16
  cast at the store) -- which is the ``mlp-down-gemm-mma`` contract, *not*
  the row's other (portable fp32 tree) contract, so that class pins
  ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND=hopper`` and skips without the SM90 build.
  Measured on H100 PCIe / Triton 3.6 with seed-fixed inputs at
  ``S = 512, K = 12288, N = 3072``: forward, ``dx``, ``dW`` and ``db`` are 100%
  bit-identical to the Hopper CUDA kernel (``torch.equal`` on the raw bytes), and
  that held for every tile/warp/stage configuration swept (128 forward, 32
  ``dx``, 16 ``dW``, 15 ``db``), including the wgmma.m64n128k16 lowering Triton
  selects for ``BLOCK_M >= 64``.

The model-shape accuracy gate here runs at ``M = 256`` because the fp32 CPU
reference costs ~15 s per call at ``K = 12288``; the full 4096-row gate for the
row lives in ``tests/ops/gemm/test_mlp_down_gemm.py``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

from rl_engine.reference.gemm.mlp_down_gemm import (
    left_fold_bias_gradient,
    mlp_down_gemm_reference_backward,
    mlp_down_gemm_reference_forward,
)

try:
    import triton  # noqa: F401

    from rl_engine.backends.shared.triton.gemm.mlp_down_gemm import (
        TritonMlpDownGemmOp,
        _mlp_down_gemm_db_kernel,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - environment without Triton
    _HAS_TRITON = False


def _hopper_backend():
    """The Hopper (``mlp-down-gemm-mma``) CUDA backend, or ``None`` if unusable.

    The byte-equality claim is about the *hardware-order* contract: Triton and
    the Hopper TMA + wgmma kernel are two implementations of it. The portable
    fp32 tree kernel in the same extension is the row's other contract and is
    deliberately *not* byte-equal to either (it is byte-equal to the fp32 CPU
    reference instead), so it is not what this file pins.
    """

    try:
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mma_backend_available,
            sm90_backend_compiled,
        )
    except ImportError:  # pragma: no cover - no CUDA backend in this build
        return None
    if not (mma_backend_available() and sm90_backend_compiled()):
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return None
    return CudaMlpDownGemmOp()


_HOPPER_BACKEND = _hopper_backend()
_HOPPER_REASON = (
    "the Hopper mlp-down-gemm-mma path needs KERNEL_ALIGN_FORCE_SM90=1 and a cc 9.0 device"
)

pytestmark = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="the Triton mlp_down_gemm backend needs Triton and a CUDA GPU",
)
BYTE_EXACT = pytest.mark.skipif(_HOPPER_BACKEND is None, reason=_HOPPER_REASON)


@contextmanager
def _pinned(name):
    """Run the block with ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND`` pinned."""

    switch = "RL_KERNEL_MLP_DOWN_GEMM_BACKEND"
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

# --- the model's own geometry (issue #386) -----------------------------------
# MMDiT MLP 3072 -> 12288 -> 3072 with bias, so both stream down projections are
# [*, 12288] -> [*, 3072]; the reference shapes 1024^2 / 1328^2 / 1664x928 pack
# 16x16 pixels per token, i.e. 4096 / 6889 / 6032 image tokens, and the sigma
# schedule anchors 256 tokens. Every shape keeps the model's K and N; the token
# count is a reference tier, the anchor, or one of the two synthetic cases the RFC
# allows (a reduction below the mma chain, and odd tails).
MODEL_K = 12288
MODEL_N = 3072
REF_TOKENS = (4096, 6889, 6032)  # 1024^2, 1328^2, 1664x928 image tokens
ANCHOR_TOKENS = 256  # the sigma schedule's low-token anchor

IMG_MLP_DOWN_SHAPE = (REF_TOKENS[0], MODEL_K, MODEL_N)  # img_mlp.net.2
SMALL = (64, MODEL_K, MODEL_N)  # short token count, the model's K and N
SHORT_K = (64, 96, MODEL_N)  # below the mma chain length: bit-exactness must hold
GATE_SHAPE = (ANCHOR_TOKENS, MODEL_K, MODEL_N)  # what the fp32 CPU gate affords
BACKWARD_GATE_SHAPE = (32, MODEL_K, MODEL_N)


def _inputs(shape, dtype=torch.bfloat16, seed=0):
    rows, k_dim, n_dim = shape
    gen = torch.Generator().manual_seed(seed)
    # Model-like scales: standard-normal activations and fan-in-normalized
    # weights keep the reference output O(1), so the tolerance below is
    # expressed in ulps of the reference magnitude.
    x = torch.randn(rows, k_dim, generator=gen)
    weight = torch.randn(n_dim, k_dim, generator=gen) / (k_dim**0.5)
    bias = torch.randn(n_dim, generator=gen) / (k_dim**0.5)
    return (
        x.to(dtype).cuda(),
        weight.to(dtype).cuda(),
        bias.to(dtype).cuda(),
    )


def _reference_forward(x, weight, bias):
    return mlp_down_gemm_reference_forward(
        x.float().cpu(), weight.float().cpu(), bias.float().cpu()
    )


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


def _assert_tolerance_against_device_reference(got, ref, label):
    """The same declared bounds against a bf16 reference already on the device.

    Used where the fp32 CPU reference is out of budget (it walks K python-level
    steps, each a vectorized fp64 ``[M, N]`` op): ``ref`` is the portable tree
    path's store, which is the correctly-rounded bf16 value the CPU comparison
    rounds to anyway, and the ulp scale is that reference's magnitude -- i.e. the
    metric of :func:`_deviation` without the host round trip.
    """

    identical = float((got == ref).float().mean())
    ulp = BF16_ULP * ref.float().abs().max().clamp_min(1e-12)
    worst = float((got.float() - ref.float()).abs().max() / ulp)
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
        op = TritonMlpDownGemmOp()
        assert op.op_class == "reduction"
        assert op.is_batch_invariant is True

    def test_forward_shape_dtype_and_lead_dims(self):
        x, weight, bias = _inputs((7, 96, MODEL_N))
        op = TritonMlpDownGemmOp()
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
            TritonMlpDownGemmOp()(x, weight, bias=bias)

    def test_fail_closed_on_non_cuda_device(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(RuntimeError, match="accelerator"):
            TritonMlpDownGemmOp()(x.cpu(), weight.cpu(), bias=bias.cpu())

    def test_fail_closed_on_k_mismatch(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(ValueError, match="K must match"):
            TritonMlpDownGemmOp()(x, weight[:, :-16].contiguous(), bias=bias)

    def test_fail_closed_on_bias_length_mismatch(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(ValueError, match="bias length"):
            TritonMlpDownGemmOp()(x, weight, bias=bias[:-1].contiguous())

    def test_fail_closed_on_fp32_bias(self):
        x, weight, bias = _inputs(SMALL)
        with pytest.raises(ValueError, match="bias must be bf16"):
            TritonMlpDownGemmOp()(x, weight, bias=bias.float())

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
        op = TritonMlpDownGemmOp()
        got = op(x, weight, bias=bias)
        want = op(x, weight, bias=bias.contiguous())
        _assert_same_bytes(got, want, "strided bias forward")

    def test_no_bias_matches_reference(self):
        x, weight, _ = _inputs(SMALL)
        got = TritonMlpDownGemmOp()(x, weight)
        identical, worst = _deviation(got, _reference_forward(x, weight, torch.zeros(weight.size(0))))
        assert worst <= 1.0, f"no-bias path: worst={worst:.3f} ulp"
        assert identical >= 0.99, f"no-bias path: identical={identical}"

    def test_bias_is_added_once_in_fp32(self):
        """Zero weights isolate the epilogue: the output must be exactly ``b``."""

        x, weight, bias = _inputs(SMALL)
        weight = torch.zeros_like(weight)
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        assert torch.equal(got, bias.unsqueeze(0).expand_as(got).contiguous())


# ---------------------------------------------------------------------------
# determinism + invariance (bitwise)
# ---------------------------------------------------------------------------
class TestInvariance:
    def test_forward_deterministic_over_three_reruns(self):
        x, weight, bias = _inputs(SMALL)
        op = TritonMlpDownGemmOp()
        first = op(x, weight, bias=bias)
        for _ in range(3):
            assert torch.equal(op(x, weight, bias=bias), first)

    def test_forward_row_invariant(self):
        x, weight, bias = _inputs(SMALL)
        op = TritonMlpDownGemmOp()
        full = op(x, weight, bias=bias)
        for rows in (1, 7, x.size(0) - 1):
            part = op(x[:rows].contiguous(), weight, bias=bias)
            assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_forward_tiling_invariant(self):
        """A padded M tile must not leak into a logical row's bytes."""

        x, weight, bias = _inputs((130, 96, MODEL_N))
        op = TritonMlpDownGemmOp()
        base = op(x[:64].contiguous(), weight, bias=bias)
        padded = op(x, weight, bias=bias)
        assert torch.equal(padded[:64], base)

    def test_forward_model_shape_row_invariant(self):
        """The model's K/N with a partial tile and a single token."""

        x, weight, bias = _inputs((133, MODEL_K, MODEL_N))
        op = TritonMlpDownGemmOp()
        full = op(x, weight, bias=bias)
        for rows in (1, 7, 132):
            assert torch.equal(op(x[:rows].contiguous(), weight, bias=bias), full[:rows])

    def test_backward_deterministic(self):
        x, weight, bias = _inputs(SMALL)
        grad = torch.randn(SMALL[0], SMALL[2], generator=torch.Generator().manual_seed(13))
        grad = grad.to(torch.bfloat16).cuda()
        op = TritonMlpDownGemmOp()
        first = _grads(op, x, weight, bias, grad)
        for _ in range(3):
            again = _grads(op, x, weight, bias, grad)
            assert all(torch.equal(a, b) for a, b in zip(again, first))

    def test_dx_row_invariant(self):
        x, weight, _ = _inputs(SMALL)
        grad = torch.randn(SMALL[0], SMALL[2], generator=torch.Generator().manual_seed(14))
        grad = grad.to(torch.bfloat16).cuda()
        op = TritonMlpDownGemmOp()
        full, _, _ = _grads(op, x, weight, torch.zeros(SMALL[2]).to(torch.bfloat16).cuda(), grad)
        for rows in (1, 7, SMALL[0] - 1):
            part, _, _ = _grads(
                op,
                x[:rows].contiguous(),
                weight,
                torch.zeros(SMALL[2]).to(torch.bfloat16).cuda(),
                grad[:rows].contiguous(),
            )
            assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_dw_padding_invariant(self):
        """Zero-padded batch rows contribute exact zeroes, never rounding drift."""

        x, weight, _ = _inputs((64, 96, MODEL_N))
        grad = torch.randn(64, MODEL_N, generator=torch.Generator().manual_seed(15))
        grad = grad.to(torch.bfloat16).cuda()
        op = TritonMlpDownGemmOp()
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
        grad = torch.randn(24, MODEL_N, generator=torch.Generator().manual_seed(16))
        grad = grad.to(torch.bfloat16).cuda()
        op = TritonMlpDownGemmOp()
        _, _, base_db = _grads(op, x[:9].contiguous(), weight, bias, grad[:9].contiguous())
        padded = torch.zeros_like(grad)
        padded[:9] = grad[:9]
        _, _, padded_db = _grads(op, x, weight, bias, padded)
        assert torch.equal(padded_db, base_db)


# ---------------------------------------------------------------------------
# accuracy against the independent fp32 reference
# ---------------------------------------------------------------------------
class TestAccuracy:
    # synthetic short reductions: the model's own length is covered at K = 12288 below
    @pytest.mark.parametrize("k_dim", [16, 32, 48, 96, 128])
    def test_short_k_agrees_within_one_ulp(self, k_dim):
        """Below the mma chain length both orders agree to under one bf16 ulp.

        Not to byte equality: the tensor core's k16 grouping and the reference's
        correctly-rounded FMA chain may differ in the last fp32 bit, which only
        moves an element's bf16 rounding when it sits within ~1e-6 of a boundary.
        """

        assert k_dim <= BIT_EXACT_MAX_K
        x, weight, bias = _inputs((32, k_dim, MODEL_N))
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        identical, worst = _deviation(got, _reference_forward(x, weight, bias))
        assert worst <= 1.0, f"K={k_dim}: worst={worst:.3f} ulp"
        assert identical >= 0.99, f"K={k_dim}: identical={identical}"

    def test_model_shape_matches_reference_within_declared_tolerance(self):
        x, weight, bias = _inputs(GATE_SHAPE)
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        assert got.dtype is torch.bfloat16
        _assert_identical_tolerance(got, _reference_forward(x, weight, bias), "forward")

    def test_seed_permutations_hold_the_same_bounds(self):
        for seed in (1, 2, 3):
            x, weight, bias = _inputs(SMALL, seed=seed)
            got = TritonMlpDownGemmOp()(x, weight, bias=bias)
            _assert_identical_tolerance(
                got, _reference_forward(x, weight, bias), f"forward seed={seed}"
            )


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
    tests pin the structural exactness directly, then the arithmetic against an
    fp64 reference computed independently of both implementations. Measured on
    this part (H100 PCIe, 24 seeds/shapes, ~100M elements): every element within
    1 bf16 ulp of the largest magnitude, >= 99.3% bit-identical to the correctly
    rounded exact value, worst 1.0000 ulp; cuBLAS fp32 from the same inputs
    measures 99.93% exact / 0.25 ulp. The bounds below keep headroom.
    """

    def test_integer_inputs_are_bit_exact(self):
        """Small integers make fp32 accumulation exact, so the result must be.

        ``|x|, |W| <= 3`` over ``K = 12288`` keeps every partial sum below
        ``2**24``, i.e. representable, so the sum is order independent: the output
        must equal the exact dot product bit for bit, which no transposed weight,
        mis-indexed column or dropped term can survive.
        """

        rows, k_dim, n_dim = 32, 12288, 3072
        gen = torch.Generator().manual_seed(7)
        xi = torch.randint(-3, 4, (rows, k_dim), generator=gen).float()
        wi = torch.randint(-3, 4, (n_dim, k_dim), generator=gen).float()
        bi = torch.randint(-8, 9, (n_dim,), generator=gen).float()
        x, weight, bias = xi.bfloat16().cuda(), wi.bfloat16().cuda(), bi.bfloat16().cuda()
        op = TritonMlpDownGemmOp()
        assert torch.equal(
            op(x, weight, bias=bias), (xi.double() @ wi.double().T + bi.double()).bfloat16().cuda()
        )
        assert torch.equal(op(x, weight), (xi.double() @ wi.double().T).bfloat16().cuda())

    def test_full_k_identity_and_bias(self):
        op = TritonMlpDownGemmOp()
        k_dim, n_dim = 12288, 8
        x = torch.ones(3, k_dim, dtype=torch.bfloat16).cuda()
        weight = torch.ones(n_dim, k_dim, dtype=torch.bfloat16).cuda()
        # every term is 1 and every partial sum is an exact small integer, so the
        # total is exactly K in any association order: a dropped or doubled leaf
        # moves it, and the bf16 store of K is itself exact
        assert torch.equal(
            op(x, weight), torch.full((3, n_dim), float(k_dim), dtype=torch.bfloat16).cuda()
        )
        bias = torch.arange(n_dim, dtype=torch.float32).bfloat16().cuda()
        want = (
            (torch.tensor(float(k_dim)) + bias.float()).bfloat16().expand(3, n_dim).contiguous()
        )
        assert torch.equal(op(x, weight, bias=bias), want.cuda())
        assert torch.equal(op(torch.zeros_like(x), weight, bias=bias), bias.expand(3, n_dim).contiguous())

    @pytest.mark.parametrize("k_dim", [12288, 12287, 12289])
    def test_one_hot_covers_every_leaf_boundary(self, k_dim):
        """Every leaf's first and last reduction index must contribute exactly once."""

        op = TritonMlpDownGemmOp()
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
            assert op(x, weight).float()[0, 0].item() == 1.0, f"k={k} not counted exactly once"

    @pytest.mark.parametrize(
        "shape",
        [SHORT_K, GATE_SHAPE] + [(tokens, MODEL_K, MODEL_N) for tokens in REF_TOKENS],
    )
    def test_matches_exact_fp64_truth(self, shape):
        x, weight, bias = _inputs(shape)
        truth64 = x.double() @ weight.double().T
        truth = (truth64.float() + bias.float()).bfloat16()
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        deviation = (
            (got.float().double() - truth.float().double()).abs() / _ulp_of_max_magnitude(truth)
        )
        assert deviation.max().item() <= 2.0, f"{shape}: worst {deviation.max().item():.3f} ulp"
        identical = float((got == truth).float().mean())
        assert identical >= 0.99, f"{shape}: only {identical:.4f} bit-identical"

    def test_backward_matches_exact_fp64(self):
        x, weight, bias = _inputs(GATE_SHAPE)
        grad = torch.randn(GATE_SHAPE[0], GATE_SHAPE[2], generator=torch.Generator().manual_seed(11))
        grad = grad.bfloat16().cuda()
        dx, dW, db = _grads(TritonMlpDownGemmOp(), x, weight, bias, grad)
        x64, w64, b64 = (t.double().requires_grad_(True) for t in (x, weight, bias))
        (x64 @ w64.T + b64).backward(grad.double())
        for name, got, want in (("dx", dx, x64.grad), ("dW", dW, w64.grad), ("db", db, b64.grad)):
            truth = want.bfloat16()
            deviation = (
                (got.float().double() - truth.float().double()).abs() / _ulp_of_max_magnitude(truth)
            )
            assert deviation.max().item() <= 2.0, f"{name}: worst {deviation.max().item():.3f} ulp"
            assert float((got == truth).float().mean()) >= 0.99, f"{name} diverges from the exact gradient"


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
class TestBackward:
    @pytest.mark.parametrize("shape", [SMALL, BACKWARD_GATE_SHAPE])
    def test_gradients_match_reference_within_declared_tolerance(self, shape):
        x, weight, bias = _inputs(shape)
        grad = torch.randn(shape[0], shape[2], generator=torch.Generator().manual_seed(11))
        grad = (grad / (shape[2] ** 0.5)).to(torch.bfloat16).cuda()
        dx, dw, db = _grads(TritonMlpDownGemmOp(), x, weight, bias, grad)
        ref_dx, ref_dw, ref_db = mlp_down_gemm_reference_backward(
            x.float().cpu(), weight.float().cpu(), grad.float().cpu()
        )
        assert dx.dtype is torch.bfloat16 and dw.dtype is torch.bfloat16
        _assert_identical_tolerance(dx, ref_dx, f"dx {shape}")
        _assert_identical_tolerance(dw, ref_dw, f"dW {shape}")
        assert db.dtype is torch.bfloat16
        assert torch.allclose(db.float().cpu(), ref_db, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("rows", [REF_TOKENS[1], REF_TOKENS[2]])
    def test_backward_at_the_reference_shapes(self, rows):
        """Backward at 6889 and 6032 rows (the 1328^2 and 1664x928 tiers).

        The fp32 CPU reference walks K python-level steps over ``[M, N]``, so it
        cannot be run at these token counts. The tier oracle is the portable tree
        path: it is byte-equal to that reference at 4096 rows, at the anchor and at
        the K tails (``tests/ops/gemm/test_mlp_down_gemm.py``), and its order depends only on
        K, so it is the reference result here too -- and unlike the reference it
        runs at these M. ``db`` is the same ascending fold in both contracts and is
        additionally checked bit-for-bit against the independent fp32 reference
        fold, which is O(M*N) and therefore affordable at any M.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mma_backend_available,
        )

        if not mma_backend_available():
            pytest.skip("the portable CUDA path needs the CUDA extension built")

        x, weight, bias = _inputs((rows, MODEL_K, MODEL_N), seed=31)
        grad = torch.randn(rows, MODEL_N, generator=torch.Generator().manual_seed(32))
        grad = (grad / (MODEL_N**0.5)).to(torch.bfloat16).cuda()
        got_dx, got_dw, got_db = _grads(TritonMlpDownGemmOp(), x, weight, bias, grad)
        with _pinned("general"):
            tree_dx, tree_dw, tree_db = _grads(CudaMlpDownGemmOp(), x, weight, bias, grad)
        _assert_tolerance_against_device_reference(got_dx, tree_dx, f"dx rows={rows}")
        _assert_tolerance_against_device_reference(got_dw, tree_dw, f"dW rows={rows}")
        _assert_same_bytes(got_db, tree_db, f"db rows={rows}")

        folded = left_fold_bias_gradient(grad.float().cpu()).to(torch.bfloat16).cuda()
        assert torch.equal(tree_db, folded), f"tree db rows={rows}"
        assert torch.equal(got_db, folded), f"triton db rows={rows}"

    def test_db_is_the_ascending_fp32_fold(self):
        """``db`` is one correctly-rounded FP32 add per row, in index order."""

        shape = (24, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape)
        grad = torch.randn(shape[0], shape[2], generator=torch.Generator().manual_seed(12))
        grad = grad.to(torch.bfloat16).cuda()
        _, _, db = _grads(TritonMlpDownGemmOp(), x, weight, bias, grad)
        folded = left_fold_bias_gradient(grad.float().cpu())
        # The op's bf16 store is the fold rounded once; the fold itself is
        # reproduced bit-for-bit when the kernel writes fp32.
        assert torch.equal(db, folded.to(torch.bfloat16).cuda())
        exact = torch.empty(shape[2], dtype=torch.float32, device="cuda")
        _mlp_down_gemm_db_kernel[(-(-shape[2] // 128),)](
            grad,
            exact,
            shape[0],
            N=shape[2],
            stride_gm=grad.stride(0),
            stride_gn=grad.stride(1),
            BLOCK_N=128,
            BLOCK_M=128,
            num_warps=4,
            num_stages=1,
        )
        assert torch.equal(exact.cpu(), folded)

    def test_backward_uses_the_autograd_contract(self):
        """``torch.autograd.grad`` (no ``.grad`` buffer) agrees with ``backward``."""

        x, weight, bias = _inputs(SMALL)
        grad = torch.randn(SMALL[0], SMALL[2], generator=torch.Generator().manual_seed(17))
        grad = grad.to(torch.bfloat16).cuda()
        op = TritonMlpDownGemmOp()
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
        grad = torch.randn(SMALL[0], SMALL[2], generator=torch.Generator().manual_seed(18))
        grad = grad.to(torch.bfloat16).cuda()
        xr = x.detach().clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        TritonMlpDownGemmOp()(xr, wr).backward(grad)
        assert xr.grad is not None and wr.grad is not None


# ---------------------------------------------------------------------------
# cross-backend: byte equality with the Hopper hardware-order schedule
# ---------------------------------------------------------------------------
@BYTE_EXACT
class TestCudaByteEquality:
    """Triton and the Hopper kernel implement the same pinned arithmetic.

    Both are ``mlp-down-gemm-mma``; the CUDA side is pinned with
    ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND=hopper`` so this cannot accidentally
    compare Triton against the portable tree contract (which is a different
    order, byte-equal to the fp32 CPU reference rather than to Triton).
    """

    @pytest.mark.parametrize("rows", [1, 7, 129, *REF_TOKENS])
    def test_forward_bytes_equal(self, rows):
        x, weight, bias = _inputs((rows, MODEL_K, MODEL_N), seed=5)
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        with _pinned("hopper"):
            want = _HOPPER_BACKEND(x, weight, bias=bias)
        _assert_same_bytes(got, want, f"forward rows={rows}")

    def test_forward_bytes_equal_without_bias_and_at_small_k(self):
        x, weight, bias = _inputs((33, MODEL_K, MODEL_N), seed=6)
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        with _pinned("hopper"):
            want = _HOPPER_BACKEND(x, weight, bias=bias)
        _assert_same_bytes(got, want, "forward K=128")
        with _pinned("hopper"):
            want_nobias = _HOPPER_BACKEND(x, weight)
        _assert_same_bytes(TritonMlpDownGemmOp()(x, weight), want_nobias, "forward no bias")

    @pytest.mark.parametrize("rows", [7, 129, *REF_TOKENS])
    def test_backward_bytes_equal(self, rows):
        x, weight, bias = _inputs((rows, MODEL_K, MODEL_N), seed=7)
        grad = torch.randn(rows, MODEL_N, generator=torch.Generator().manual_seed(21))
        grad = (grad / (MODEL_N**0.5)).to(torch.bfloat16).cuda()
        got_dx, got_dw, got_db = _grads(TritonMlpDownGemmOp(), x, weight, bias, grad)
        with _pinned("hopper"):
            want_dx, want_dw, want_db = _grads(_HOPPER_BACKEND, x, weight, bias, grad)
        _assert_same_bytes(got_dx, want_dx, f"dx rows={rows}")
        _assert_same_bytes(got_dw, want_dw, f"dW rows={rows}")
        _assert_same_bytes(got_db, want_db, f"db rows={rows}")

    def test_forward_bytes_equal_at_the_model_shape(self):
        x, weight, bias = _inputs(IMG_MLP_DOWN_SHAPE, seed=8)
        got = TritonMlpDownGemmOp()(x, weight, bias=bias)
        with _pinned("hopper"):
            want = _HOPPER_BACKEND(x, weight, bias=bias)
        _assert_same_bytes(got, want, "forward 4096x12288x3072")

    def test_the_hopper_path_is_the_mma_contract(self):
        """The path Triton is compared against publishes ``mlp-down-gemm-mma``."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            MMA_CONTRACT,
            mlp_down_gemm_backend_used,
            mlp_down_gemm_contract_used,
        )

        x, weight, _ = _inputs((7, MODEL_K, MODEL_N), seed=9)
        with _pinned("hopper"):
            assert mlp_down_gemm_backend_used(x, weight) == "hopper"
            assert mlp_down_gemm_contract_used(x, weight) == MMA_CONTRACT
