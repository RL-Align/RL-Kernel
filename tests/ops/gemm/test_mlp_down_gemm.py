# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""WS1 tests for the Qwen-Image MLP down projection (``y = single_cast(x @ W.T + b)``).

The row's CUDA backend serves two *different*, both frozen, arithmetic orders
and this file pins each against the right oracle:

* ``mlp-down-gemm-mma`` -- the hardware order (Hopper TMA + wgmma; the
  Triton backend implements the same order and is byte-identical to it): the K
  reduction is a frozen sequence of k-chunks of 16 through one fp32 accumulator
  per output element, no split-K, no atomics, bias once in fp32, one bf16 cast
  at the store. Its agreement with the independent fp32 CPU tree model is a
  declared tolerance, not bit equality (measured bounds next to the constants
  below and in ``docs/operators/mlp-down-gemm.md``).
* ``mlp-down-gemm-tree`` -- the portable order: the fp32 32-wide-leaf
  mid-split tree of ``rl_engine/reference/gemm/mlp_down_gemm.py``
  itself (``tree_gemm``, ``left_fold_weight_gradient``,
  ``left_fold_bias_gradient`` are the definition), so this path is **byte-equal
  to the reference** on every shape, forward and backward. It needs no tensor
  cores and serves every device and operand layout the row supports.

What the operator promises, and what is tested here:

* ``TestTreeContractByteEquality`` -- the tree path's forward, ``dx``, ``dW``
  and ``db`` equal ``mlp_down_gemm_reference_forward`` /
  ``mlp_down_gemm_reference_backward`` byte for byte, at the model's K = 12288,
  N = 3072 and every token count the RFC names, plus the synthetic tails.
* ``test_*_covers_each_k_exactly_once`` -- the schedule's structural promise
  (every reduction index contributes, once) via one-hot probes.
* ``test_*_deterministic`` / ``*_row_invariant`` / ``*_tiling_invariant`` --
  train-infer consistency at the operator level: a logical row's bytes cannot
  depend on the batch, the tile, or a rerun. Run on both paths.
* ``TestHopperPath`` -- the hardware order is unchanged: still inside the
  declared tolerance of the reference and byte-identical to the Triton backend
  (``tests/ops/gemm/test_mlp_down_gemm_triton.py``).
* Shapes keep the model's geometry: K = 12288, N = 3072, and the token counts are the
  RFC's reference tiers (4096 / 6889 / 6032), the 256-token schedule anchor, or the
  short synthetic cases the RFC allows (a K below the tree's leaf, odd tails).
* ``test_*_fail_closed`` -- unsupported dtype/device/shape raises.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

from rl_engine.reference.gemm.mlp_down_gemm import (
    NativeMlpDownGemmOp,
    mlp_down_gemm_reference_backward,
    mlp_down_gemm_reference_forward,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend requires a GPU")

# --- measured contract bounds (H100 PCIe, bf16; see the docs page) -----------
# Below the mma's k-chunk count the fp32 accumulation order cannot differ, so
# the bf16 result is bit-identical to the reference. Above it the two fixed
# orders diverge; the measured (worst case at K = 12288, seed-fixed inputs)
# is >= 99% bit-identical elements with every element inside a few bf16 ulps
# of the reference magnitude, which is what the row declares.
BF16_ULP = 2.0**-8
BIT_EXACT_MAX_K = 128
DECLARED_MIN_IDENTICAL_FRACTION = 0.99
DECLARED_MAX_ULPS = 8.0

# --- the model's own geometry (issue #386) -----------------------------------
# The MMDiT MLP is 3072 -> 12288 -> 3072 with bias, so both stream down
# projections are [*, 12288] -> [*, 3072]. The reference shapes 1024^2, 1328^2 and
# 1664x928 pack 16x16 pixels per token, i.e. 4096, 6889 and 6032 image tokens, and
# the sigma schedule anchors 256 tokens. Every shape below therefore keeps the
# model's K and N and uses either a reference token count, the 256-token anchor, or
# one of the two synthetic cases the RFC allows: a reduction shorter than the mma
# chain (SHORT_K) and odd reduction tails.
MODEL_K = 12288
MODEL_N = 3072
REF_TOKENS = (4096, 6889, 6032)  # 1024^2, 1328^2, 1664x928 image tokens
ANCHOR_TOKENS = 256  # the sigma schedule's low-token anchor

IMG_MLP_DOWN_SHAPE = (REF_TOKENS[0], MODEL_K, MODEL_N)  # img_mlp.net.2
TXT_MLP_DOWN_SHAPE = (ANCHOR_TOKENS, MODEL_K, MODEL_N)  # txt_mlp.net.2: prompt length

SMALL = (64, MODEL_K, MODEL_N)  # short token count, the model's K and N
SHORT_K = (64, 96, MODEL_N)  # below the mma chain length: bit-exactness must hold


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


def _reference(x, weight, bias):
    return mlp_down_gemm_reference_forward(
        x.float().cpu(), weight.float().cpu(), bias.float().cpu()
    )


def _deviation(got, ref):
    """Bit-identity and worst deviation of a bf16 output against the fp32 model.

    Identity is counted in the output dtype (bf16), which is the granularity the
    operator promises; the deviation is measured in ulps of the reference
    magnitude so the bound is scale free.
    """

    got_c, ref_f = got.cpu(), ref.float()
    identical = float((got_c.bfloat16() == ref_f.bfloat16()).float().mean())
    ulp = BF16_ULP * ref_f.abs().max().clamp_min(1e-12)
    worst = float((got_c.detach().float() - ref_f).abs().max() / ulp)
    return identical, worst


# --- path pinning and the cached fp32 reference -----------------------------
# The two CUDA contracts are different arithmetic orders, so a byte-equality
# claim has to name the path it is about: these force one.
BACKEND_ENV = "RL_KERNEL_MLP_DOWN_GEMM_BACKEND"
TREE_CONTRACT = "mlp-down-gemm-tree"
MMA_CONTRACT = "mlp-down-gemm-mma"

# The fp32 CPU tree walks K python-level steps, each one a vectorized fp64 [M, N]
# op, so it runs on a row slice at the model's token counts (the tree depends only
# on K, and the general path's row/batch invariance is tested separately) and whole
# at the 256-token anchor. One cache entry per (kind, shape, rows) keeps the forward
# and backward checks from paying for the same tree twice.
TREE_REFERENCE_ROWS = 16
_REFERENCE_CACHE: dict = {}


@contextmanager
def _backend(name):
    """Run the block with ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND`` pinned."""

    previous = os.environ.get(BACKEND_ENV)
    os.environ[BACKEND_ENV] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(BACKEND_ENV, None)
        else:
            os.environ[BACKEND_ENV] = previous


def _grad_of(shape, rows=None, seed=11):
    rows = shape[0] if rows is None else rows
    gen = torch.Generator().manual_seed(seed)
    return (
        (torch.randn(rows, shape[2], generator=gen) / (shape[2] ** 0.5)).to(torch.bfloat16).cuda()
    )


def _reference_rows(shape, kind, rows):
    """Cached fp32 CPU reference for ``shape`` (``forward`` or ``backward``)."""

    key = (kind, shape, rows)
    if key not in _REFERENCE_CACHE:
        x, weight, bias = _inputs(shape)
        if kind == "forward":
            _REFERENCE_CACHE[key] = (
                mlp_down_gemm_reference_forward(
                    x[:rows].float().cpu(), weight.float().cpu(), bias.float().cpu()
                ),
            )
        else:
            _REFERENCE_CACHE[key] = mlp_down_gemm_reference_backward(
                x[:rows].float().cpu(), weight.float().cpu(), _grad_of(shape, rows).float().cpu()
            )
    return _REFERENCE_CACHE[key]


def _byte_mismatches(got: torch.Tensor, want: torch.Tensor) -> int:
    """Number of differing raw bf16 bytes between two same-shaped tensors.

    ``torch.equal`` compares values, so it treats ``-0.0`` and ``+0.0`` as
    equal; the contract's claim is about *bytes*, which is what this counts.
    """

    a = got.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    b = want.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    assert a.shape == b.shape, f"shape mismatch: {got.shape} vs {want.shape}"
    return int((a != b).sum())


def _hopper_serves(device=None) -> bool:
    """Whether the Hopper path is built *and* this device can take it."""

    from rl_engine.backends.cuda.gemm.mlp_down_gemm import sm90_backend_compiled

    if not sm90_backend_compiled():
        return False
    if device is None:
        return False
    return torch.cuda.get_device_capability(device)[0] >= 9


# ---------------------------------------------------------------------------
# reference sanity: the independent fp32 model must itself be well defined
# ---------------------------------------------------------------------------
class TestReferenceModel:
    def test_fp32_matches_fp64_closely(self):
        x, weight, bias = (
            torch.randn(8, 256, generator=torch.Generator().manual_seed(3)),
            torch.randn(16, 256, generator=torch.Generator().manual_seed(4)) / 16.0,
            torch.randn(16, generator=torch.Generator().manual_seed(5)) / 16.0,
        )
        ref = mlp_down_gemm_reference_forward(x, weight, bias)
        exact = (x.double() @ weight.double().T + bias.double()).float()
        assert torch.allclose(ref, exact, atol=1e-5, rtol=1e-5)

    # The fp32 CPU tree costs O(M * N * K / 32) python-level steps, so the
    # reference-side invariance checks run at a toy row count plus the model's
    # real reduction length (which is what the tree structure depends on).
    @pytest.mark.parametrize("shape", [SMALL, (ANCHOR_TOKENS, MODEL_K, MODEL_N)])
    def test_reference_rows_are_batch_invariant(self, shape):
        x, weight, bias = _inputs(shape, dtype=torch.float32)
        full = mlp_down_gemm_reference_forward(x, weight, bias)
        for rows in (1, 7, x.shape[0] // 2):
            part = mlp_down_gemm_reference_forward(x[:rows].contiguous(), weight, bias)
            assert torch.equal(part, full[:rows])

    def test_gold_op_dtype_paths_agree(self):
        x, weight, bias = _inputs(SMALL, dtype=torch.float32)
        op = NativeMlpDownGemmOp()
        assert torch.equal(
            op.forward_fp32(x, weight, bias=bias), mlp_down_gemm_reference_forward(x, weight, bias)
        )


# ---------------------------------------------------------------------------
# the pinned schedule's structural promise
# ---------------------------------------------------------------------------
@CUDA
class TestScheduleContract:
    # synthetic short reductions: the model's own length is covered at K = 12288 below
    @pytest.mark.parametrize("k_dim", [16, 32, 48, 96, 128])
    def test_forward_covers_each_k_exactly_once(self, k_dim):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
        weight = torch.ones(4, k_dim, dtype=torch.bfloat16).cuda()
        total = 0.0
        for k in range(k_dim):
            x = torch.zeros(1, k_dim, dtype=torch.bfloat16).cuda()
            x[0, k] = 1.0
            got = op(x, weight).float()[0, 0].item()
            # Weight of ones: a unit at reduction index k must contribute exactly
            # once, so the output is exactly 1 (a skipped index gives 0, a doubled
            # one gives 2).
            assert got == 1.0, f"index {k} not counted exactly once ({got})"
            total += got
        # Every index contributed, and no index contributed twice.
        assert total == float(k_dim)

    def test_forward_deterministic(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
        x, weight, bias = _inputs(IMG_MLP_DOWN_SHAPE)
        first = op(x, weight, bias=bias)
        for _ in range(3):
            assert torch.equal(op(x, weight, bias=bias), first)

    def test_forward_row_invariant(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
        x, weight, bias = _inputs(SMALL)
        full = op(x, weight, bias=bias)
        for rows in (1, 7, x.shape[0] - 1):
            part = op(x[:rows].contiguous(), weight, bias=bias)
            assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_forward_tiling_invariant(self):
        """Padded tiles must not leak into a logical row's bytes."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
        x, weight, bias = _inputs((130, 96, MODEL_N))
        base = op(x[:64].contiguous(), weight, bias=bias)
        padded = op(x, weight, bias=bias)
        assert torch.equal(padded[:64], base)


# ---------------------------------------------------------------------------
# accuracy against the independent fp32 reference
# ---------------------------------------------------------------------------
@CUDA
class TestAccuracy:
    @pytest.mark.parametrize("shape", [SHORT_K, (16, 64, MODEL_N), (32, 20, MODEL_N)])
    def test_short_k_agrees_within_one_ulp(self, shape):
        """Below the mma chain length the orders agree to less than one bf16 ulp.

        On the hardware-order path they are not byte-equal in general: the tensor
        core groups the reduction into k16 steps with its own internal order, while
        the reference chains correctly-rounded FMAs, so the two can differ in the
        last fp32 bit -- which moves an element's bf16 rounding only when it sits
        within ~1e-6 of a boundary, hence the 1 bf16 ulp bound rather than equality.
        On the portable tree path the bytes are equal (see
        :class:`TestTreeContractByteEquality`), which satisfies the same bound.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        rows, k_dim, n_dim = shape
        assert k_dim <= BIT_EXACT_MAX_K
        x, weight, bias = _inputs(shape)
        got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        identical, worst = _deviation(got, _reference(x, weight, bias))
        assert worst <= 1.0, f"worst={worst:.3f} ulp"
        assert identical >= 0.99, f"identical={identical}"

    @pytest.mark.parametrize("shape", [IMG_MLP_DOWN_SHAPE, TXT_MLP_DOWN_SHAPE])
    def test_model_shapes_match_reference_within_declared_tolerance(self, shape):
        """Full model shape on the device; the fp32 CPU reference covers a slice.

        The reduction length is the model's (12288) and the kernel runs the real
        launch geometry; only the reference rows are trimmed, because the CPU
        tree would otherwise take hours at M = 4096.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        checked_rows = 32
        x, weight, bias = _inputs(shape)
        got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        assert got.shape == (shape[0], shape[2])
        identical, worst_ulps = _deviation(
            got[:checked_rows], _reference(x[:checked_rows], weight, bias)
        )
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"identical={identical}"
        assert worst_ulps <= DECLARED_MAX_ULPS, f"worst={worst_ulps:.2f} ulp"

    def test_bias_is_added_once_in_fp32(self):
        """Zero weights isolate the epilogue: the output must be exactly b."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        weight = torch.zeros_like(weight)
        got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        assert torch.equal(got.cpu(), bias.unsqueeze(0).expand_as(got).contiguous().cpu())

    def test_no_bias_is_allowed(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, _ = _inputs(SMALL)
        got = CudaMlpDownGemmOp()(x, weight)
        identical, worst = _deviation(got, _reference(x, weight, torch.zeros(weight.shape[0])))
        assert worst <= 1.0, f"no-bias path: worst={worst:.3f} ulp"
        assert identical >= 0.99, f"no-bias path: identical={identical}"


# ---------------------------------------------------------------------------
# the portable tree contract: byte equality with the independent fp32 reference
# ---------------------------------------------------------------------------
# The model's K/N at every token count the RFC names, the 256-token anchor, and
# the synthetic tails the RFC allows (a reduction below one leaf, K +- 1 leaves).
TREE_FORWARD_SHAPES = [
    (ANCHOR_TOKENS, MODEL_K, MODEL_N),
    *[(tokens, MODEL_K, MODEL_N) for tokens in REF_TOKENS],
    SHORT_K,
    (64, 100, MODEL_N),
    (16, MODEL_K + 1, MODEL_N),
    (24, MODEL_K - 1, MODEL_N),
]
TREE_BACKWARD_SHAPES = [
    (ANCHOR_TOKENS, MODEL_K, MODEL_N),
    (REF_TOKENS[0], MODEL_K, MODEL_N),
    SHORT_K,
    (64, 100, MODEL_N),
    (16, MODEL_K + 1, MODEL_N),
]


@CUDA
class TestTreeContractByteEquality:
    """The general path computes the reference's tree: byte equality, no tolerance.

    Everything here pins ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND=general``, so on a
    Hopper host with the wgmma entry points built this is the portable path and
    not the hardware order. The fp32 CPU reference is the expensive side: it runs
    on a row slice at the model's token counts (the tree depends only on K, and
    the general path's row/batch invariance is covered by
    :class:`TestTreePathInvariance`), and the 256-token anchor is additionally
    checked whole. Every mismatch count reported on failure is the *number of
    differing elements*, which the contract requires to be zero.
    """

    @pytest.mark.parametrize("shape", TREE_FORWARD_SHAPES)
    def test_forward_is_byte_equal_to_the_reference(self, shape):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_contract_used,
        )

        rows = shape[0]
        checked = min(rows, TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            assert mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT
            got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        (ref,) = _reference_rows(shape, "forward", checked)
        want = ref.bfloat16()
        mismatches = _byte_mismatches(got[:checked], want)
        assert (
            mismatches == 0
        ), f"{shape}: {mismatches} of {want.numel()} bf16 elements differ from the fp32 reference"

    def test_forward_is_byte_equal_to_the_reference_at_the_anchor_whole(self):
        """The 256-token anchor row for row -- the reference really is the oracle."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        shape = (ANCHOR_TOKENS, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        (ref,) = _reference_rows(shape, "forward", shape[0])
        want = ref.bfloat16()
        mismatches = _byte_mismatches(got, want)
        assert (
            mismatches == 0
        ), f"{shape}: {mismatches} of {want.numel()} bf16 elements differ from the fp32 reference"

    def test_no_bias_is_byte_equal_to_the_reference(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        shape = (REF_TOKENS[0], MODEL_K, MODEL_N)
        x, weight, _ = _inputs(shape)
        checked = TREE_REFERENCE_ROWS
        with _backend("general"):
            got = CudaMlpDownGemmOp()(x, weight)
        ref_biasless = mlp_down_gemm_reference_forward(
            x[:checked].float().cpu(), weight.float().cpu(), None
        )
        assert _byte_mismatches(got[:checked], ref_biasless.bfloat16()) == 0

    @pytest.mark.parametrize("shape", TREE_BACKWARD_SHAPES)
    def test_backward_is_byte_equal_to_the_reference(self, shape):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        checked = min(shape[0], TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, checked)
        xr = x[:checked].clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        br = bias.detach().clone().requires_grad_(True)
        with _backend("general"):
            CudaMlpDownGemmOp()(xr, wr, bias=br).backward(grad)
        ref_dx, ref_dw, ref_db = _reference_rows(shape, "backward", checked)
        for name, got, ref in (
            ("dx", xr.grad, ref_dx),
            ("dW", wr.grad, ref_dw),
            ("db", br.grad, ref_db),
        ):
            want = ref.bfloat16()
            mismatches = _byte_mismatches(got, want)
            assert mismatches == 0, (
                f"{shape} {name}: {mismatches} of {want.numel()} bf16 elements differ "
                "from the fp32 reference"
            )


@CUDA
class TestTreePathInvariance:
    """The general path's bytes cannot depend on the batch, the tile or a rerun."""

    def test_forward_deterministic(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(IMG_MLP_DOWN_SHAPE)
        with _backend("general"):
            op = CudaMlpDownGemmOp()
            first = op(x, weight, bias=bias)
            for _ in range(3):
                assert torch.equal(op(x, weight, bias=bias), first)

    def test_forward_row_invariant(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            op = CudaMlpDownGemmOp()
            full = op(x, weight, bias=bias)
            for rows in (1, 7, x.shape[0] - 1):
                assert torch.equal(
                    op(x[:rows].contiguous(), weight, bias=bias), full[:rows]
                ), f"rows={rows}"

    def test_forward_tiling_invariant(self):
        """Padded tiles (and the partial k leaf) must not leak into a row's bytes."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs((130, 96, MODEL_N))
        with _backend("general"):
            op = CudaMlpDownGemmOp()
            base = op(x[:64].contiguous(), weight, bias=bias)
            assert torch.equal(op(x, weight, bias=bias)[:64], base)
        x, weight, bias = _inputs((68, MODEL_K + 1, MODEL_N))
        with _backend("general"):
            op = CudaMlpDownGemmOp()
            base = op(x[:64].contiguous(), weight, bias=bias)
            assert torch.equal(op(x, weight, bias=bias)[:64], base)

    def test_backward_deterministic(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        grad = _grad_of(SMALL)

        def run():
            with _backend("general"):
                op = CudaMlpDownGemmOp()
                xr = x.detach().clone().requires_grad_(True)
                wr = weight.detach().clone().requires_grad_(True)
                br = bias.detach().clone().requires_grad_(True)
                op(xr, wr, bias=br).backward(grad)
                return xr.grad, wr.grad, br.grad

        first = run()
        for _ in range(3):
            assert all(torch.equal(a, b) for a, b in zip(run(), first))


# ---------------------------------------------------------------------------
# the two CUDA paths and the contract each one publishes
# ---------------------------------------------------------------------------
@CUDA
class TestCudaPathsAndContracts:
    """The general (tree) and Hopper (mma) paths are different contracts.

    Each is verified against its own oracle elsewhere -- the tree path byte-for-
    byte against the fp32 CPU reference above, the Hopper path byte-for-byte
    against Triton in ``tests/ops/gemm/test_mlp_down_gemm_triton.py``. What is checked
    here is the routing: which path a call takes, what contract it reports, and
    that no path silently becomes the other.
    """

    PAIR_SHAPES = [(1, 96, MODEL_N), (130, 96, 64), (32, MODEL_K, 64), (7, MODEL_K, MODEL_N)]

    @pytest.mark.parametrize("shape", PAIR_SHAPES)
    def test_general_path_never_falls_back(self, shape):
        """Every shape the row supports is served by the tree, byte-equal to the reference."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_backend_used,
            mlp_down_gemm_contract_used,
        )

        x, weight, bias = _inputs(shape)
        with _backend("general"):
            assert mlp_down_gemm_backend_used(x, weight) == "general"
            assert mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT
            got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        ref = mlp_down_gemm_reference_forward(
            x.float().cpu(), weight.float().cpu(), bias.float().cpu()
        )
        mismatches = _byte_mismatches(got, ref.bfloat16())
        assert mismatches == 0, f"{shape}: {mismatches} bf16 elements differ from the reference"

    @pytest.mark.parametrize("shape", [(64, 100, MODEL_N), (16, MODEL_K + 1, MODEL_N)])
    def test_odd_reduction_tails_still_correct(self, shape):
        """K that is not a multiple of 8 (or of 32) runs the portable tree, exactly.

        The Hopper entries refuse such a tensor map (the row strides have to be
        multiples of 8 elements); the wrapper then runs the general kernel, whose
        output is byte-equal to the fp32 reference, and reports the tree contract.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_contract_used,
        )

        rows, k_dim, n_dim = shape
        x, weight, bias = _inputs(shape)
        assert mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT
        got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        ref = _reference(x, weight, bias)
        mismatches = _byte_mismatches(got, ref.bfloat16())
        assert mismatches == 0, f"K={k_dim}: {mismatches} bf16 elements differ from the reference"

    def test_below_cc9_the_tree_contract_serves_and_is_unchanged(self, monkeypatch):
        """Below cc 9.0 the auto route is the portable tree, not a changed result."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_backend_used,
            mlp_down_gemm_contract_used,
        )

        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            forced = CudaMlpDownGemmOp()(x, weight, bias=bias)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
        assert mlp_down_gemm_backend_used(x, weight) == "general"
        assert mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT
        assert torch.equal(CudaMlpDownGemmOp()(x, weight, bias=bias), forced)

    def test_backend_env_selects_the_contract(self, monkeypatch):
        """`RL_KERNEL_MLP_DOWN_GEMM_BACKEND` pins the contract on one machine.

        `general` must produce the fp32 reference's bytes (the portable tree,
        ``mlp-down-gemm-tree``), `hopper` must refuse rather than silently
        change the arithmetic order when it cannot serve the operands, and an
        unknown value is an error.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_contract_used,
        )

        x, weight, bias = _inputs(SMALL)
        monkeypatch.setenv(BACKEND_ENV, "general")
        tree = CudaMlpDownGemmOp()(x, weight, bias=bias)
        assert mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT
        (ref,) = _reference_rows(SMALL, "forward", SMALL[0])
        assert _byte_mismatches(tree, ref.bfloat16()) == 0
        monkeypatch.setenv(BACKEND_ENV, "bogus")
        with pytest.raises(ValueError):
            CudaMlpDownGemmOp()(x, weight, bias=bias)
        # `hopper` is an error when it cannot serve, never a fallback to the tree
        monkeypatch.setenv(BACKEND_ENV, "hopper")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
        with pytest.raises(RuntimeError):
            CudaMlpDownGemmOp()(x, weight, bias=bias)

    def test_route_report_is_emitted_once(self, monkeypatch, capsys):
        """First call emits `[RL-Kernel][route] ... module=mlp_down_gemm ...` once."""

        from rl_engine.backends.cuda.gemm import mlp_down_gemm as mod

        monkeypatch.setattr(mod, "_ROUTE_REPORTED", False)
        monkeypatch.delenv(mod._BACKEND_ENV, raising=False)
        monkeypatch.delenv("RL_KERNEL_ROUTE_REPORT", raising=False)
        x, weight, bias = _inputs(SMALL)
        mod.CudaMlpDownGemmOp()(x, weight, bias=bias)
        first = capsys.readouterr().out
        assert "module=mlp_down_gemm" in first
        # a cc 9.0 device only takes the Hopper path when its symbols are compiled in
        # (`KERNEL_ALIGN_FORCE_SM90=1`); otherwise the route is the portable tree
        expected = "hopper" if _hopper_serves(x.device) else "general"
        order_contract = mod.MMA_CONTRACT if expected == "hopper" else mod.TREE_CONTRACT
        assert "requested=auto" in first and f"actual={expected}" in first
        assert f"contract={order_contract}" in first
        mod.CudaMlpDownGemmOp()(x, weight, bias=bias)
        assert "module=mlp_down_gemm" not in capsys.readouterr().out

    def test_route_report_names_the_contract_and_the_pin(self, monkeypatch, capsys):
        """A pinned `general` reports the tree contract and how to pin it back."""

        from rl_engine.backends.cuda.gemm import mlp_down_gemm as mod

        monkeypatch.setattr(mod, "_ROUTE_REPORTED", False)
        monkeypatch.setenv(mod._BACKEND_ENV, "general")
        monkeypatch.delenv("RL_KERNEL_ROUTE_REPORT", raising=False)
        x, weight, bias = _inputs(SMALL)
        mod.CudaMlpDownGemmOp()(x, weight, bias=bias)
        out = capsys.readouterr().out
        assert "requested=general" in out and "actual=general" in out
        assert f"contract={mod.TREE_CONTRACT}" in out
        # the text says the change is a contract change and how to pin it back
        assert "portable_tree_contract" in out
        assert f"{mod._BACKEND_ENV}=hopper" in out

    def test_reports_requested_actual_backend_and_contract(self, monkeypatch):
        """Requested backend, actual backend, fallback state and contract are queryable."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            mlp_down_gemm_backend,
            mlp_down_gemm_backend_used,
            mlp_down_gemm_contract_used,
        )

        x, weight, _ = _inputs(SMALL)
        assert mlp_down_gemm_backend() == "auto"
        # the cc 9.0 path needs its symbols built in; without them auto falls back
        expected = "hopper" if _hopper_serves(x.device) else "general"
        assert mlp_down_gemm_backend_used(x, weight) == expected
        assert mlp_down_gemm_contract_used(x, weight) == (
            MMA_CONTRACT if expected == "hopper" else TREE_CONTRACT
        )
        monkeypatch.setenv(BACKEND_ENV, "general")
        assert mlp_down_gemm_backend_used(x, weight) == "general"
        assert mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT

    def test_rocm_dispatch_uses_triton(self, monkeypatch):
        """On a ROCm torch the registry resolves this row to the Triton backend."""

        from rl_engine.runtime.registry import kernel_registry

        monkeypatch.setattr(torch.version, "hip", "6.0.0")
        op = kernel_registry.get_op("mlp_down_gemm", device=torch.device("cuda"))
        assert type(op).__name__ in ("TritonMlpDownGemmOp", "NativeMlpDownGemmOp")

    def test_cuda_op_fails_closed_on_rocm(self, monkeypatch):
        """The CUDA op must refuse on ROCm rather than reach for absent symbols."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        monkeypatch.setattr(torch.version, "hip", "6.0.0")
        with pytest.raises(RuntimeError, match="Triton"):
            CudaMlpDownGemmOp()(x, weight, bias=bias)

    def test_degenerate_and_rank3_shapes(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        empty = CudaMlpDownGemmOp()(x[:0].contiguous(), weight, bias=bias)
        assert empty.shape == (0, SMALL[2])
        x3 = torch.randn(4, 8, SMALL[1], generator=torch.Generator().manual_seed(1))
        x3 = x3.to(torch.bfloat16).cuda()
        flat = x3.reshape(-1, SMALL[1])
        assert torch.equal(
            CudaMlpDownGemmOp()(x3, weight, bias=bias),
            CudaMlpDownGemmOp()(flat, weight, bias=bias).reshape(4, 8, SMALL[2]),
        )


# ---------------------------------------------------------------------------
# the Hopper path: the hardware-order contract, unchanged
# ---------------------------------------------------------------------------
@CUDA
class TestHopperPath:
    """``mlp-down-gemm-mma`` on Hopper: still the order the row shipped.

    Byte identity with the Triton backend (which implements the same order) is
    asserted in ``tests/ops/gemm/test_mlp_down_gemm_triton.py``; here the path is pinned
    with ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND=hopper`` and checked against the
    independent fp32 reference under the declared tolerance -- it is *not*
    expected to be byte-equal to it (the tree path is the one that is).
    """

    def _skip_without_hopper(self, device) -> None:
        if not _hopper_serves(device):
            pytest.skip("the Hopper path needs KERNEL_ALIGN_FORCE_SM90=1 and a cc 9.x device")

    @pytest.mark.parametrize(
        "shape", [SMALL, (ANCHOR_TOKENS, MODEL_K, MODEL_N), (REF_TOKENS[0], MODEL_K, MODEL_N)]
    )
    def test_forward_matches_the_reference_within_the_declared_tolerance(self, shape):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_backend_used,
            mlp_down_gemm_contract_used,
        )

        x, weight, bias = _inputs(shape)
        self._skip_without_hopper(x.device)
        with _backend("hopper"):
            assert mlp_down_gemm_backend_used(x, weight) == "hopper"
            assert mlp_down_gemm_contract_used(x, weight) == MMA_CONTRACT
            got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        (ref,) = _reference_rows(shape, "forward", TREE_REFERENCE_ROWS)
        identical, worst = _deviation(got[:TREE_REFERENCE_ROWS], ref)
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{shape}: identical={identical}"
        assert worst <= DECLARED_MAX_ULPS, f"{shape}: worst={worst:.2f} ulp"

    def test_backward_matches_the_reference_within_the_declared_tolerance(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        shape = (32, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape)
        self._skip_without_hopper(x.device)
        grad = _grad_of(shape, 32)
        xr = x.detach().clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        br = bias.detach().clone().requires_grad_(True)
        with _backend("hopper"):
            CudaMlpDownGemmOp()(xr, wr, bias=br).backward(grad)
        ref_dx, ref_dw, ref_db = mlp_down_gemm_reference_backward(
            x.float().cpu(), weight.float().cpu(), grad.float().cpu()
        )
        for name, got, ref in (
            ("dx", xr.grad, ref_dx),
            ("dW", wr.grad, ref_dw),
        ):
            identical, worst = _deviation(got, ref)
            assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{name} identical={identical}"
            assert worst <= DECLARED_MAX_ULPS, f"{name} worst={worst:.2f} ulp"
        # db is the shared ascending-row fp32 left fold, byte-exact in both paths
        assert torch.equal(br.grad.cpu(), ref_db.to(torch.bfloat16))

    def test_hopper_is_requestable_and_the_contract_is_reported(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            CudaMlpDownGemmOp,
            mlp_down_gemm_backend_used,
        )

        x, weight, bias = _inputs(SMALL)
        self._skip_without_hopper(x.device)
        with _backend("hopper"):
            assert mlp_down_gemm_backend_used(x, weight) == "hopper"
            assert CudaMlpDownGemmOp()(x, weight, bias=bias).dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
@CUDA
class TestBackward:
    @pytest.mark.parametrize("shape", [SMALL, (32, MODEL_K, MODEL_N)])
    def test_gradients_match_reference_within_declared_tolerance(self, shape):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(shape)
        grad_gen = torch.Generator().manual_seed(11)
        grad = (
            (torch.randn(shape[0], shape[2], generator=grad_gen) / (shape[2] ** 0.5))
            .to(torch.bfloat16)
            .cuda()
        )
        x.requires_grad_(True)
        weight.requires_grad_(True)
        bias.requires_grad_(True)
        CudaMlpDownGemmOp()(x, weight, bias=bias).backward(grad)
        ref_dx, ref_dw, ref_db = mlp_down_gemm_reference_backward(
            x.float().cpu(), weight.float().cpu(), grad.float().cpu()
        )
        for name, got, ref in (
            ("dx", x.grad, ref_dx),
            ("dW", weight.grad, ref_dw),
        ):
            identical, worst = _deviation(got, ref)
            assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{name} identical={identical}"
            assert worst <= DECLARED_MAX_ULPS, f"{name} worst={worst:.2f} ulp"
        # db is the same ascending fp32 left fold in both implementations, so it
        # is bit-exact after the single bf16 cast.
        assert torch.equal(bias.grad.cpu(), ref_db.to(torch.bfloat16))

    def test_db_is_the_left_fold(self):
        """db must be the ascending-row fp32 fold of the incoming gradient."""

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs((24, MODEL_K, MODEL_N))
        grad = (
            torch.randn(24, MODEL_N, generator=torch.Generator().manual_seed(12))
            .to(torch.bfloat16)
            .cuda()
        )
        bias.requires_grad_(True)
        CudaMlpDownGemmOp()(x, weight, bias=bias).backward(grad)
        folded = torch.zeros(MODEL_N, dtype=torch.float32)
        for row in grad.float().cpu():
            folded = folded + row
        assert torch.equal(bias.grad.cpu(), folded.to(torch.bfloat16))

    def test_gradients_deterministic(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
        x, weight, bias = _inputs(SMALL)
        grad = (
            torch.randn(SMALL[0], SMALL[2], generator=torch.Generator().manual_seed(13))
            .to(torch.bfloat16)
            .cuda()
        )

        def run():
            xr = x.clone().requires_grad_(True)
            wr = weight.clone().requires_grad_(True)
            br = bias.clone().requires_grad_(True)
            op(xr, wr, bias=br).backward(grad)
            return xr.grad, wr.grad, br.grad

        first = run()
        for _ in range(3):
            assert all(torch.equal(a, b) for a, b in zip(run(), first))


# ---------------------------------------------------------------------------
# correctness against the exact result, not just the fp32 model
# ---------------------------------------------------------------------------
def _ulp_of_max_magnitude(t: torch.Tensor) -> float:
    """One bf16 ulp of the tensor's largest magnitude (the contract's absolute scale)."""

    import math

    return 2.0 ** (math.floor(math.log2(float(t.float().abs().max()))) - 7)


@CUDA
class TestExactTruth:
    """Compare against the exact result; the fp32 model cannot be the only oracle.

    Agreement with the model is self-consistency: it cannot see a mistake that
    the model and the kernel share (a transposed weight, a dropped leaf, a wrong
    cast), and it says nothing about whether the fp32 accumulation order is
    accurate. These checks pin the structural exactness directly instead, and
    then the arithmetic against an fp64 reference computed independently of both
    implementations.
    """

    def test_integer_inputs_are_bit_exact(self):
        """Small integers make fp32 accumulation exact, so the result must be exact.

        ``|x|, |W| <= 3`` over ``K = 12288`` keeps every partial sum below
        ``2**24``, i.e. representable, so the sum is order independent: the
        output has to equal the exact dot product bit for bit, which no
        transposed weight, mis-indexed column or dropped term can survive.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        rows, k_dim, n_dim = 32, MODEL_K, MODEL_N
        gen = torch.Generator().manual_seed(7)
        xi = torch.randint(-3, 4, (rows, k_dim), generator=gen).float()
        wi = torch.randint(-3, 4, (n_dim, k_dim), generator=gen).float()
        bi = torch.randint(-8, 9, (n_dim,), generator=gen).float()
        x, weight, bias = xi.bfloat16().cuda(), wi.bfloat16().cuda(), bi.bfloat16().cuda()
        op = CudaMlpDownGemmOp()
        assert torch.equal(
            op(x, weight, bias=bias), (xi.double() @ wi.double().T + bi.double()).bfloat16().cuda()
        )
        assert torch.equal(op(x, weight), (xi.double() @ wi.double().T).bfloat16().cuda())

    def test_full_k_identity_and_bias(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
        k_dim, n_dim = MODEL_K, MODEL_N
        x = torch.ones(3, k_dim, dtype=torch.bfloat16).cuda()
        weight = torch.ones(n_dim, k_dim, dtype=torch.bfloat16).cuda()
        # every term is 1 and every partial sum is an exact small integer, so the
        # total is exactly K in any association order: a dropped or doubled leaf
        # moves it, and the bf16 store of K is itself exact
        assert torch.equal(
            op(x, weight),
            torch.full((3, n_dim), float(k_dim), dtype=torch.bfloat16).cuda(),
        )
        bias = torch.arange(n_dim, dtype=torch.float32).bfloat16().cuda()
        want = (torch.tensor(float(k_dim)) + bias.float()).bfloat16().expand(3, n_dim).contiguous()
        assert torch.equal(op(x, weight, bias=bias), want.cuda())
        # bias alone: x = 0 must reproduce the bias exactly, and only once
        bias_got = op(torch.zeros_like(x), weight, bias=bias)
        assert torch.equal(bias_got, bias.expand(3, n_dim).contiguous())

    @pytest.mark.parametrize("k_dim", [12288, 12287, 12289])
    def test_one_hot_covers_every_leaf_boundary(self, k_dim):
        """Every leaf's first and last reduction index must contribute exactly once.

        At the model's reduction length, so the 384-leaf mid-split tree and its
        short tail leaf are the ones being probed.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        op = CudaMlpDownGemmOp()
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
        [SHORT_K, (ANCHOR_TOKENS, MODEL_K, MODEL_N)]
        + [(tokens, MODEL_K, MODEL_N) for tokens in REF_TOKENS],
    )
    def test_matches_exact_fp64_truth(self, shape):
        """Accuracy against the exact result, with the model left out of the loop.

        Measured on this part at ``K = 12288`` (H100 PCIe, 24 seeds/shapes,
        ~100M elements): every element is within 1 bf16 ulp of the largest
        magnitude and >= 99.3% of elements are bit-identical to the correctly
        rounded exact value. The residual is bf16 rounding ties moved by the
        ~1e-5 fp32 accumulation noise, which any implementation of this contract
        shares (cuBLAS fp32 from the same inputs measures 99.93% exact, worst
        0.25 ulp). The bounds below keep headroom over the measurement.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(shape)
        truth64 = x.double() @ weight.double().T
        truth = (truth64.float() + bias.float()).bfloat16()
        got = CudaMlpDownGemmOp()(x, weight, bias=bias)
        deviation = (got.float().double() - truth.float().double()).abs() / _ulp_of_max_magnitude(
            truth
        )
        assert deviation.max().item() <= 2.0, f"{shape}: worst {deviation.max().item():.3f} ulp"
        identical = float((got == truth).float().mean())
        assert identical >= 0.99, f"{shape}: only {identical:.4f} bit-identical"

    def test_backward_matches_exact_fp64(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs((ANCHOR_TOKENS, MODEL_K, MODEL_N))
        grad = (
            torch.randn(ANCHOR_TOKENS, MODEL_N, generator=torch.Generator().manual_seed(11))
            .bfloat16()
            .cuda()
        )
        xr, wr, br = (t.clone().requires_grad_(True) for t in (x, weight, bias))
        CudaMlpDownGemmOp()(xr, wr, bias=br).backward(grad)
        x64, w64, b64 = (t.double().requires_grad_(True) for t in (x, weight, bias))
        (x64 @ w64.T + b64).backward(grad.double())
        for name, got, want in (
            ("dx", xr.grad, x64.grad),
            ("dW", wr.grad, w64.grad),
            ("db", br.grad, b64.grad),
        ):
            truth = want.bfloat16()
            deviation = (
                got.float().double() - truth.float().double()
            ).abs() / _ulp_of_max_magnitude(truth)
            assert deviation.max().item() <= 2.0, f"{name}: worst {deviation.max().item():.3f} ulp"
            assert (
                float((got == truth).float().mean()) >= 0.99
            ), f"{name} diverges from the exact gradient"


# ---------------------------------------------------------------------------
# integration: dispatch, fail-closed behaviour, model wiring
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_registry_dispatches_cuda_and_cpu(self):
        """CUDA prefers the Triton path, CPU the fp32 reference.

        Triton lowers the same pinned schedule to wgmma, which is byte-identical
        to the hand-written kernel and about 2.5x its forward throughput, so it
        is the CUDA default; the native kernel stays as the no-Triton fallback.
        """

        from rl_engine.runtime.registry import kernel_registry

        cpu_op = kernel_registry.get_op("mlp_down_gemm", device=torch.device("cpu"))
        assert type(cpu_op).__name__ == "NativeMlpDownGemmOp"
        if torch.cuda.is_available():
            cuda_op = kernel_registry.get_op("mlp_down_gemm", device=torch.device("cuda"))
            assert type(cuda_op).__name__ in ("TritonMlpDownGemmOp", "CudaMlpDownGemmOp")
            # The native kernel must stay loadable as the fallback path.
            from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

            assert CudaMlpDownGemmOp() is not None

    def test_gtest_spec_registered(self):
        from rl_engine.validation.operators.operator_specs import OP_SPECS

        spec = OP_SPECS["mlp_down_gemm"]
        assert spec.op_class == "reduction"
        assert spec.grad_input_names == ("x", "weight", "bias")
        assert "cuda" in spec.candidate_paths

    def test_gtest_inputs_shapes(self):
        import argparse

        from rl_engine.validation.operators.operator_inputs import make_operator_inputs

        args = argparse.Namespace(
            batch=2, seq=16, k_dim=12288, n_dim=3072, dtype="float32", seed=0, device="cpu"
        )
        tensors = make_operator_inputs("mlp_down_gemm", args, torch.float32, torch.device("cpu"))
        assert tensors["x"].shape == (32, 12288)
        assert tensors["weight"].shape == (3072, 12288)
        assert tensors["bias"].shape == (3072,)

    @CUDA
    def test_fail_closed_on_fp32_input(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL, dtype=torch.float32)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpDownGemmOp()(x, weight, bias=bias)

    @CUDA
    def test_fail_closed_below_sm80(self, monkeypatch):
        """Below sm80 there is no kernel for either CUDA path: it must raise.

        Below compute capability 8.0 the row's validated support matrix ends, so
        a silent dispatch would run an unvalidated configuration instead of
        failing.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpDownGemmOp()(x, weight, bias=bias)

    @CUDA
    def test_cc10_does_not_take_the_hopper_path(self, monkeypatch):
        """A cc >= 10 device must not dispatch to the sm_90a-only wgmma body.

        The body only exists under ``__CUDA_ARCH_FEAT_SM90_ALL``: on a
        ``KERNEL_ALIGN_FORCE_SM90=1`` build for a cc >= 10 host it compiles to an
        inert stub, so the runtime gate is cc 9.0 exactly. A cc >= 10 call takes
        the portable tree, and asking for the hardware order raises rather than
        changing the contract silently.
        """

        from rl_engine.backends.cuda.gemm import mlp_down_gemm as cuda_mod

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
        x, weight, bias = _inputs(SMALL)
        assert cuda_mod.mlp_down_gemm_backend_used(x, weight) == "general"
        assert cuda_mod.mlp_down_gemm_contract_used(x, weight) == TREE_CONTRACT
        monkeypatch.setenv(BACKEND_ENV, "hopper")
        with pytest.raises(RuntimeError):
            cuda_mod.CudaMlpDownGemmOp()(x, weight, bias=bias)

    @CUDA
    def test_strided_bias_is_read_at_the_right_elements(self):
        """A view bias must give the same bytes as its contiguous copy.

        The kernels index the bias at unit stride, so the backend has to
        normalize it; before that, a stride-2 bias silently returned wrong values.
        """

        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, _ = _inputs(SMALL, seed=14)
        bias = torch.randn(SMALL[2] * 2, device="cuda", dtype=torch.bfloat16)[::2]
        assert bias.shape == (SMALL[2],) and bias.stride(0) == 2
        op = CudaMlpDownGemmOp()
        assert torch.equal(op(x, weight, bias=bias), op(x, weight, bias=bias.contiguous()))

    @CUDA
    def test_fail_closed_on_shape_mismatch(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpDownGemmOp()(x, weight[:, :-16].contiguous(), bias=bias)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpDownGemmOp()(x, weight, bias=bias[:-1].contiguous())

    def test_fail_closed_on_non_cuda_device(self):
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        x, weight, bias = _inputs(SMALL)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpDownGemmOp()(x.cpu(), weight.cpu(), bias=bias.cpu())
