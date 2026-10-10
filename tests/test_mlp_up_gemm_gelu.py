# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""WS1 tests for the Qwen-Image MLP up projection + tanh-GELU (WS1).

The operator is the MMDiT feed-forward up projection frozen with GELU inside::

    pre = tree_gemm(x, Wᵀ) + b                  fp32, the byte-equality anchor
    y   = bf16_rne(gelu_tanh_fp32(pre))         the single cast

    gate = bf16_rne(f32(grad_y) * gelu_tanh_grad_fp32(pre))
    dx   = tree_gemm(gate, W)                   reduction over N = 12288
    dW   = left_fold(gate, x)                   [12288, 3072], fp32 left fold
    db   = left_fold(gate)                      [12288],        fp32 left fold

The row's CUDA backend serves two *different*, both frozen, arithmetic orders
and this file pins each against the right oracle:

* ``mlp-up-gemm-gelu-mma`` -- the hardware order (Hopper TMA + wgmma; the
  Triton backend implements the same order and is byte-identical to it for the
  pre-activation, the gate and the three contractions, and therefore for ``y``).
  It associates the reduction differently than the reference's tree (ascending
  k16 chunks chained into one fp32 accumulator versus a 32-wide-leaf mid-split
  tree), so its agreement with the independent fp32 CPU tree model is a declared
  tolerance for the GELU *value* and for ``dx``/``dW``, and a tight fp32-ulp
  bound for ``pre``; ``db`` is the same fold of the same gate and is therefore
  bit-identical (measured bounds next to the constants below and in
  ``docs/operators/mlp-up-gemm-gelu.md``).
* ``mlp-up-gemm-gelu-tree`` -- the portable order: the fp32 32-wide-leaf
  mid-split tree of ``rl_engine/kernels/ops/pytorch/linear/mlp_up_gemm_gelu.py``
  itself (``tree_gemm``, ``left_fold_weight_gradient``,
  ``left_fold_bias_gradient`` are the definition), so this path is **byte-equal
  to the reference's pre-activation, gate and gradients** on every shape. It
  needs no tensor cores and serves every device and operand layout the row
  supports.

What the operator promises, and what is tested here:

* ``TestPreContract`` -- the device forward's fp32 ``pre`` (``emit_pre``) equals
  ``NativeMlpUpGemmGeluOp.pre_fp32`` byte for byte at the model's K = 3072,
  N = 12288 and every token count the RFC names, plus the synthetic K tails; and
  a logical row's ``pre`` bytes cannot depend on the batch, the tile, a reshaped
  batch, or a rerun.
* ``TestGeluValue`` -- ``y`` against the fp32 reference GELU under the declared
  tolerance (miss ratio + worst bf16 ulp), with the deviation shown to be
  confined to the tanh: ``pre`` is byte-identical on both sides and the GELU is
  byte-exact wherever the tanh saturates.
* ``TestTreeContractByteEquality`` -- the tree path's ``pre``, ``dx``, ``dW``
  and ``db`` equal the fp32 reference byte for byte (``dx``/``dW``/``db`` with
  the *device's own gate* fed to the reference, so the transcendental is out of
  the comparison).
* ``TestGateAndEmitPre`` -- the device gate against
  ``mlp_up_gemm_gelu_reference_gate`` on the same ``pre``/``grad_out`` (declared
  tolerance, byte-equal on the saturated region), and the ``emit_pre=False``
  inference path: identical ``y`` bytes, no ``pre`` materialized.
* ``test_*_covers_each_k_exactly_once`` -- the schedule's structural promise
  (every reduction index contributes, once) via one-hot probes on ``pre``.
* ``test_*_deterministic`` / ``*_row_invariant`` / ``*_tiling_invariant`` --
  train-infer consistency at the operator level: a logical row's bytes cannot
  depend on the batch, the tile, or a rerun. Run on both paths.
* ``TestHopperPath`` -- the hardware order: ``pre`` inside the fp32 bound (its
  order is not the reference's tree), ``db`` bit-identical, the GELU values and
  the two GEMM contractions inside the declared tolerance.
* Shapes keep the model's geometry: K = 3072, N = 12288, and the token counts
  are the RFC's reference tiers (4096 / 6889 / 6032), the 256-token schedule
  anchor, or the short synthetic cases the RFC allows (a K below the tree's
  leaf, odd tails).
* ``test_*_fail_closed`` -- unsupported dtype/device/shape raises.

The fp32 CPU tree is this suite's expensive oracle: it walks K python-level
steps, each one a small vectorized fp64 ``[M, N]`` op, so one call costs
O(M * K * N) element-ops -- measured at ~1 s per row at the model's K/N (a
256-row call is minutes, a 4096-row call an hour). Every reference call is
therefore bounded to ``TREE_REFERENCE_ROWS`` rows, and the full-size tiers keep
their shape with a device-vs-device oracle instead:

* the byte-equality anchor (``pre``), the gate-fed ``dx``/``dW``/``db`` and the
  GELU-value tolerance compare the device against the fp32 CPU reference on the
  bounded slice -- one cached computation per (kind, K, N, rows), reused by every
  test that needs it (``_inputs`` draws its slice from seeded generators, so the
  256-token anchor and all three reference tiers share one entry);
* the full-size tiers (4096 / 6889 / 6032 tokens and the 256-token anchor) are
  pinned device-vs-device: the tree path's rows are batch/tile invariant
  (``TestTreePathInvariance``, ``TestPreContract``) and byte-equal to the bounded
  fp32 reference, which extends the bounded anchor to every row of the tier;
* the fp32 reference's GELU is derived from its own cached pre-activation
  (``reference_forward`` is exactly ``gelu_tanh_fp32`` of ``reference_pre``), so
  no shape pays for the K-step tree twice;
* the reference runs under ``_reference_threads``: its K small elementwise fp64
  ops would otherwise pay a pool barrier each with torch's default intra-op pool
  (128 threads on this host). The thread count cannot change a byte.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear.mlp_up_gemm_gelu import (
    NativeMlpUpGemmGeluOp,
    gelu_tanh_argument,
    gelu_tanh_fp32,
    left_fold_bias_gradient,
    mlp_up_gemm_gelu_reference_backward,
    mlp_up_gemm_gelu_reference_forward,
    mlp_up_gemm_gelu_reference_gate,
    mlp_up_gemm_gelu_reference_pre,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend requires a GPU")

# --- measured contract bounds (H100 PCIe, bf16; see the docs page) -----------
# The pre-activation is the tree's byte-equality anchor and needs no bound. The
# GELU value and the gate are declared-tolerance comparisons: `tanhf` (max
# 2 ulp) is the operator's only free operand, and a 2 ulp fp32 tanh error only
# moves the single bf16 store when the value sits within ~1e-6 of a tie, so the
# measured miss ratio is a fraction of a percent with every element inside a few
# bf16 ulps of the reference magnitude. `dx`/`dW` on the hardware order walk a
# different association over the same gate, so they get the same bound (the
# portable tree path is byte-equal instead); `db` is the shared fp32 fold and is
# bit-identical under both contracts.
BF16_ULP = 2.0**-8
BIT_EXACT_MAX_K = 128
DECLARED_MIN_IDENTICAL_FRACTION = 0.99
DECLARED_MAX_ULPS = 8.0
GATE_MIN_IDENTICAL_FRACTION = 0.99
GATE_MAX_ULPS = 2.0
GELU_MAX_ULPS = 2.0

# The mma order (wgmma k16 chunks chained into one fp32 accumulator) and the
# reference's 32-wide-leaf mid-split tree associate the same sum differently, so
# their fp32 pre-activations are *not* byte-equal: measured on the H100 at
# K = 3072, only ~0.5% of elements are bit-identical and the worst element is
# 34-40 fp32 ulps from the reference (ulps of the reference's largest magnitude,
# the same scale `_deviation` uses). That is the expected size of an
# accumulation-order difference: it grows like sqrt(K) fp32 rounding steps
# (sqrt(3072) * 2**-23 ~ 6.6e-6, i.e. ~55 ulps of the max magnitude), and it is
# still two orders of magnitude tighter than the bf16 store tolerance. Byte
# equality of ``pre`` belongs to the portable tree contract (whose ``pre`` *is*
# the reference's own tree) and to the two mma implementations against each other
# (Hopper vs Triton, ``tests/test_mlp_up_gemm_gelu_triton.py``).
FP32_MAX_ULPS = 64.0

# --- the model's own geometry (issue #386) -----------------------------------
# The MMDiT MLP is 3072 -> 12288 -> 3072 with bias, so both stream up
# projections are [*, 3072] -> [*, 12288] (the down row is the mirrored
# [*, 12288] -> [*, 3072]). The reference shapes 1024^2, 1328^2 and 1664x928 pack
# 16x16 pixels per token, i.e. 4096, 6889 and 6032 image tokens, and the sigma
# schedule anchors 256 tokens. Every shape below therefore keeps the model's K
# and N and uses either a reference token count, the 256-token anchor, or one of
# the two synthetic cases the RFC allows: a reduction shorter than the mma chain
# (SHORT_K) and odd reduction tails.
MODEL_K = 3072
MODEL_N = 12288
REF_TOKENS = (4096, 6889, 6032)  # 1024^2, 1328^2, 1664x928 image tokens
ANCHOR_TOKENS = 256  # the sigma schedule's low-token anchor

IMG_MLP_UP_SHAPE = (REF_TOKENS[0], MODEL_K, MODEL_N)  # img_mlp.net.0 / net.1
TXT_MLP_UP_SHAPE = (ANCHOR_TOKENS, MODEL_K, MODEL_N)  # prompt length
SMALL = (64, MODEL_K, MODEL_N)  # short token count, the model's K and N
SHORT_K = (64, 96, MODEL_N)  # below the mma chain length: bit-exactness must hold


def _inputs(shape, dtype=torch.bfloat16, seed=0):
    """Model-like operands whose bounded *slice* does not depend on the tokens.

    ``x``, ``weight`` and ``bias`` are drawn from three generators seeded the same
    way, so ``x[:rows]``, ``weight`` and ``bias`` are a function of
    ``(seed, k_dim, n_dim)`` alone: two shapes that differ only in their token
    count have identical operands in any bounded slice. That is what lets one
    cached fp32-reference computation serve the 256-token anchor and all three
    reference tiers (the fp32 CPU tree is the suite's expensive side).

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


def _cpu_inputs(shape, dtype=torch.float32, seed=0):
    """CPU operands for the reference-only checks (no GPU needed).

    Built like :func:`_inputs`: the bounded slice is a function of
    ``(seed, k_dim, n_dim)``, so every reference-only check can be bounded without
    changing which operands it sees.
    """

    rows, k_dim, n_dim = shape
    x = torch.randn(rows, k_dim, generator=torch.Generator().manual_seed(seed))
    weight = torch.randn(n_dim, k_dim, generator=torch.Generator().manual_seed(seed + 1))
    bias = torch.randn(n_dim, generator=torch.Generator().manual_seed(seed + 2))
    return (
        x.to(dtype),
        (weight / (k_dim**0.5)).to(dtype),
        (bias / (k_dim**0.5)).to(dtype),
    )


# The fp32 CPU reference walks K python-level steps, each one a *small* vectorized
# fp64 [M, N] op (`tree_gemm`), and torch's default intra-op pool (128 threads on
# this host) turns every one of those steps into a full barrier: measured, one
# bounded call costs ~20-30 s there against ~6 s pinned. Every op in the reference
# is elementwise (fp64 mul/add/tanh), so the thread count cannot move a rounding
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


def _reference(x, weight, bias):
    """The fp32 reference GELU value (no output cast) for device operands."""

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


def _fp32_ulp_of_max_magnitude(t: torch.Tensor) -> float:
    """One fp32 ulp of the tensor's largest magnitude."""

    return 2.0 ** (math.floor(math.log2(float(t.float().abs().max().clamp_min(1e-30)))) - 23)


def _pre_deviation(got: torch.Tensor, ref: torch.Tensor):
    """Identity and worst fp32-ulp deviation of two fp32 pre-activations.

    The mma contract's ``pre`` is a different association of the same sum than the
    reference's tree, so the identity fraction is *reported* rather than asserted
    to be 1; the worst element must stay inside the fp32 bound. Both operands are
    compared on the host (the reference is a CPU tensor, the device side is not).
    """

    got_c, ref_c = got.detach().float().cpu(), ref.detach().float().cpu()
    identical = float((got_c == ref_c).float().mean())
    worst = float((got_c.double() - ref_c.double()).abs().max() / _fp32_ulp_of_max_magnitude(ref_c))
    return identical, worst


def _gate_deviation(got, ref):
    """Identity/ulp of a bf16 gate against the correctly-rounded reference gate."""

    got_f, ref_f = got.detach().float().cpu(), ref.to(torch.bfloat16).float().cpu()
    identical = float((got_f == ref_f).float().mean())
    ulp = BF16_ULP * ref_f.abs().max().clamp_min(1e-12)
    worst = float((got_f - ref_f).abs().max() / ulp)
    return identical, worst


# --- path pinning and the cached fp32 reference -----------------------------
# The two CUDA contracts are different arithmetic orders, so a byte-equality
# claim has to name the path it is about: these force one.
BACKEND_ENV = "RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND"
TREE_CONTRACT = "mlp-up-gemm-gelu-tree"
MMA_CONTRACT = "mlp-up-gemm-gelu-mma"

# The fp32 CPU tree walks K python-level steps, each one a vectorized fp64 [M, N]
# op -- O(M * K * N) element-ops, measured at ~1 s per row at K = 3072/N = 12288.
# It therefore never runs on a whole tier: every call is bounded to
# TREE_REFERENCE_ROWS rows, and the full-size tiers keep their shape but are pinned
# device-vs-device (`TestTreePathInvariance` row/batch invariance) instead of
# against the fp32 CPU tree. One cache entry per (kind, K, N, rows) keeps the
# forward, pre and backward checks from paying for the same tree twice, and
# `_inputs` draws its operands from seeded generators so the anchor and all three
# reference tiers share the same bounded computation.
TREE_REFERENCE_ROWS = 16
_REFERENCE_CACHE: dict = {}


@contextmanager
def _backend(name):
    """Run the block with ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND`` pinned."""

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


def op_forward(x, weight, bias=None):
    """``CudaMlpUpGemmGeluOp`` forward on 2-D or lead-dim operands."""

    from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

    return CudaMlpUpGemmGeluOp()(x, weight, bias=bias)


def _forward_device(x2d, weight, bias=None, *, emit_pre=True, sm90=False):
    """Call the pinned ``_C`` forward entry, returning ``(out, pre)``.

    The signature is the operator contract's (``x, weight, bias, emit_pre`` ->
    ``(out, pre)``), so this is the lowest-level probe of the ``pre`` and
    ``emit_pre`` promises; higher-level checks use the op.
    """

    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    symbol = "mlp_up_gemm_gelu_cuda_forward_sm90" if sm90 else "mlp_up_gemm_gelu_cuda_forward"
    if not _EXT_AVAILABLE or _C is None or not hasattr(_C, symbol):
        pytest.skip("the CUDA extension does not provide the mlp_up_gemm_gelu forward entry points")
    return getattr(_C, symbol)(x2d, weight, bias, emit_pre)


def _gate_device(grad, pre):
    """The device gate: ``bf16(f32(grad) * gelu_tanh_grad_fp32(pre))``."""

    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or _C is None or not hasattr(_C, "mlp_up_gemm_gelu_cuda_gate"):
        pytest.skip("the CUDA extension does not provide the mlp_up_gemm_gelu gate entry point")
    return _C.mlp_up_gemm_gelu_cuda_gate(grad, pre)


def _route_pre(x2d, weight, bias=None):
    """The fp32 ``pre`` of the path this call routes to.

    The two CUDA contracts build the backward's gate from their *own* forward's
    pre-activation, and their orders differ (the mma order is not the portable
    tree's), so a test that feeds the reference a gate must use the pre of the path
    that actually ran: ``mlp_up_gemm_gelu_backend_used`` is exactly the route
    decision, and the ``_C`` entry it names is the one the op would call.
    """

    from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
        mlp_up_gemm_gelu_backend_used,
    )

    sm90 = mlp_up_gemm_gelu_backend_used(x2d, weight) == "hopper"
    return _forward_device(x2d, weight, bias, emit_pre=True, sm90=sm90)[1]


def _reference_rows(shape, kind, rows):
    """Cached fp32 CPU reference for ``shape`` (``pre`` or ``forward``).

    The cache key is ``(kind, K, N, rows)`` and *not* the token count: ``_inputs``
    draws its operands from seeded generators, so ``x[:rows]``/``weight``/``bias``
    are identical for every shape that shares K and N, and one bounded computation
    then serves the 256-token anchor and all three reference tiers. ``rows`` is
    clamped to the shape, and ``forward`` is derived from the cached ``pre``
    (``mlp_up_gemm_gelu_reference_forward`` is exactly ``gelu_tanh_fp32`` of
    ``mlp_up_gemm_gelu_reference_pre``), so no shape pays for the K-step tree twice.
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


def _byte_mismatches(got: torch.Tensor, want: torch.Tensor) -> int:
    """Number of differing raw bytes between two same-shaped tensors.

    ``torch.equal`` compares values, so it treats ``-0.0`` and ``+0.0`` as
    equal; the contract's claim is about *bytes*, which is what this counts. It
    is dtype agnostic, so the fp32 ``pre`` anchor uses it exactly like the bf16
    outputs.
    """

    a = got.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    b = want.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    assert a.shape == b.shape, f"shape mismatch: {got.shape} vs {want.shape}"
    return int((a != b).sum())


def _gate_fold(gate: torch.Tensor) -> torch.Tensor:
    """The ascending-row fp32 fold of the gate, rounded once to bf16."""

    with _reference_threads():
        return left_fold_bias_gradient(gate.float().cpu()).to(torch.bfloat16)


def _gate_fingerprint(gate: torch.Tensor) -> bytes:
    """A cheap digest of the gate's first row, used to key the backward cache."""

    return bytes(
        gate[:1].detach().float().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()
    )


def _reference_backward_with_gate(x, weight, pre, grad, gate, key):
    """``mlp_up_gemm_gelu_reference_backward`` with the device's own gate, cached.

    The gate is formed by the device kernel from the same ``pre`` and
    ``grad_output``, then handed to the reference as ``gate=``: that replaces the
    reference's transcendental derivative, so what the byte comparison isolates is
    exactly the three contractions. ``key`` names the bounded slice
    (``("backward", K, N, rows)``): the slice's operands are a function of
    (K, N, rows) alone, so the 256-token anchor and the 4096-token tier share one
    K-step tree plus the two folds. The gate's first row is folded into the cache
    key so a different device gate (a different ``pre``) can never reuse another
    gate's reference.
    """

    key = (*key, _gate_fingerprint(gate))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = _run_reference(
            mlp_up_gemm_gelu_reference_backward,
            x.float().cpu(),
            weight.float().cpu(),
            pre.float().cpu(),
            grad.float().cpu(),
            gate=gate.float().cpu(),
        )
    return _REFERENCE_CACHE[key]


def _hopper_serves(device=None) -> bool:
    """Whether the Hopper path is built *and* this device can take it."""

    from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import sm90_backend_compiled

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
        x, weight, bias = _cpu_inputs((8, 256, 16), seed=3)
        pre = _run_reference(mlp_up_gemm_gelu_reference_pre, x, weight, bias)
        exact_pre = x.double() @ weight.double().T + bias.double()
        assert torch.allclose(pre, exact_pre.float(), atol=1e-5, rtol=1e-5)
        exact_gelu = (
            0.5
            * exact_pre
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (exact_pre + 0.044715 * exact_pre**3)))
        )
        assert torch.allclose(gelu_tanh_fp32(pre), exact_gelu.float(), atol=1e-5, rtol=1e-5)

    # The fp32 CPU tree walks K python-level steps of [M, N] fp64, so it costs
    # O(M * K * N) element-ops (~1 s per row at the model's K/N): these
    # reference-side invariance checks run on a toy geometry plus the model's real
    # reduction length (which is what the tree structure depends on) at
    # TREE_REFERENCE_ROWS rows. The device-side invariance of the same tree is
    # covered at the full shapes by TestTreePathInvariance/TestPreContract.
    @pytest.mark.parametrize("shape", [(8, 256, 16), (TREE_REFERENCE_ROWS, MODEL_K, MODEL_N)])
    def test_reference_rows_are_batch_invariant(self, shape):
        x, weight, bias = _cpu_inputs(shape)
        full_pre = _run_reference(mlp_up_gemm_gelu_reference_pre, x, weight, bias)
        full = _run_reference(mlp_up_gemm_gelu_reference_forward, x, weight, bias)
        for rows in (1, 7, x.shape[0] // 2):
            part_pre = _run_reference(
                mlp_up_gemm_gelu_reference_pre, x[:rows].contiguous(), weight, bias
            )
            part = _run_reference(
                mlp_up_gemm_gelu_reference_forward, x[:rows].contiguous(), weight, bias
            )
            assert torch.equal(part_pre, full_pre[:rows])
            assert torch.equal(part, full[:rows])

    def test_gold_op_dtype_paths_agree(self):
        # The native op *is* the fp32 tree, so it is bounded exactly like every
        # other reference call here (and pinned: it shares the reference's K-step
        # elementwise profile).
        x, weight, bias = _cpu_inputs((TREE_REFERENCE_ROWS, MODEL_K, MODEL_N))
        op = NativeMlpUpGemmGeluOp()
        with _reference_threads():
            got_forward = op.forward_fp32(x, weight, bias=bias)
            got_pre = op.pre_fp32(x, weight, bias=bias)
        assert torch.equal(
            got_forward,
            _run_reference(mlp_up_gemm_gelu_reference_forward, x, weight, bias),
        )
        assert torch.equal(
            got_pre,
            _run_reference(mlp_up_gemm_gelu_reference_pre, x, weight, bias),
        )


# ---------------------------------------------------------------------------
# the fp32 pre-activation: the byte-equality anchor
# ---------------------------------------------------------------------------
@CUDA
class TestPreContract:
    """``pre`` is the whole operator's anchor: the GELU is a pure function of it.

    The pinned ``_C`` forward entry returns ``(out, pre)``; these checks read the
    fp32 ``pre`` directly so the byte-equality claim does not depend on the GELU
    value comparison at all. The general (portable tree) path is pinned, so the
    bytes are the reference's tree; the same anchor is asserted for the Hopper
    path in :class:`TestHopperPath`.
    """

    @pytest.mark.parametrize("shape", [SMALL, (REF_TOKENS[0], MODEL_K, MODEL_N)])
    def test_pre_is_byte_equal_to_the_reference(self, shape):
        rows = shape[0]
        checked = min(rows, TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        (ref,) = _reference_rows(shape, "pre", checked)
        assert pre.shape == (rows, MODEL_N)
        assert pre.dtype is torch.float32
        mismatches = _byte_mismatches(pre[:checked], ref)
        assert mismatches == 0, (
            f"{shape}: {mismatches} of {ref.numel()} fp32 pre-activation elements "
            "differ from the fp32 reference"
        )

    def test_pre_is_byte_equal_at_the_anchor(self):
        """The 256-token anchor: the bounded fp32 reference + a device-vs-device tier check.

        The fp32 CPU tree costs O(M * K * N) element-ops (~1 s per row at the
        model's K/N), so the whole 256-row anchor cannot be computed. The anchor
        claim is the byte equality of ``pre`` against the cached bounded reference
        (one cache entry, shared with every other tier at this K/N), and the whole
        tier's bytes are pinned device-vs-device: the same rows produced in a
        smaller batch are byte-identical, so the bounded anchor extends to every
        row of the tier.
        """

        shape = (ANCHOR_TOKENS, MODEL_K, MODEL_N)
        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
            _, part = _forward_device(
                x[:checked].contiguous().reshape(-1, shape[1]), weight, bias, emit_pre=True
            )
        (ref,) = _reference_rows(shape, "pre", checked)
        assert pre.shape == (ANCHOR_TOKENS, MODEL_N)
        assert _byte_mismatches(pre[:checked], ref) == 0
        assert torch.equal(pre[:checked], part), "the anchor's rows are not batch invariant"

    def test_no_bias_pre_is_byte_equal(self):
        shape = (REF_TOKENS[0], MODEL_K, MODEL_N)
        x, weight, _ = _inputs(shape)
        # Bounded like every reference call here (the fp32 CPU tree costs ~1 s per
        # row at the model's K/N); the biasless operand set is its own oracle, so it
        # is not the cached (pre, K, N, rows) entry.
        checked = TREE_REFERENCE_ROWS
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, None, emit_pre=True)
        ref = _run_reference(
            mlp_up_gemm_gelu_reference_pre,
            x[:checked].float().cpu(),
            weight.float().cpu(),
            None,
        )
        assert _byte_mismatches(pre[:checked], ref) == 0

    @pytest.mark.parametrize("k_dim", [16, 32, 48, 96, 128])
    def test_pre_agrees_below_one_leaf(self, k_dim):
        """A reduction below the 32-wide leaf still counts every index once."""

        x, weight, _ = _inputs((4, k_dim, 64), seed=k_dim)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, k_dim), weight, None, emit_pre=True)
        ref = _run_reference(
            mlp_up_gemm_gelu_reference_pre, x.float().cpu(), weight.float().cpu(), None
        )
        assert _byte_mismatches(pre, ref) == 0

    @pytest.mark.parametrize("k_dim", [MODEL_K + 1, MODEL_K - 1])
    def test_pre_agrees_on_odd_reduction_tails(self, k_dim):
        """K +- 1 leaves a short tail leaf (and a non-multiple-of-8 row stride)."""

        x, weight, bias = _inputs((16, k_dim, 64), seed=k_dim % 100)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, k_dim), weight, bias, emit_pre=True)
        ref = _run_reference(
            mlp_up_gemm_gelu_reference_pre,
            x.float().cpu(),
            weight.float().cpu(),
            bias.float().cpu(),
        )
        assert _byte_mismatches(pre, ref) == 0

    def test_pre_deterministic(self):
        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            _, first = _forward_device(x.reshape(-1, MODEL_K), weight, bias, emit_pre=True)
            for _ in range(3):
                _, again = _forward_device(x.reshape(-1, MODEL_K), weight, bias, emit_pre=True)
                assert torch.equal(again, first)

    def test_pre_row_invariant(self):
        """A logical row's pre-activation bytes cannot depend on the batch."""

        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            _, full = _forward_device(x.reshape(-1, MODEL_K), weight, bias, emit_pre=True)
            for rows in (1, 7, x.shape[0] - 1):
                _, part = _forward_device(
                    x[:rows].contiguous().reshape(-1, MODEL_K), weight, bias, emit_pre=True
                )
                assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_pre_tiling_invariant(self):
        """Padded tiles (and the partial k leaf) must not leak into a row's bytes."""

        x, weight, bias = _inputs((130, 96, MODEL_N))
        with _backend("general"):
            _, base = _forward_device(
                x[:64].contiguous().reshape(-1, 96), weight, bias, emit_pre=True
            )
            _, padded = _forward_device(x.reshape(-1, 96), weight, bias, emit_pre=True)
            assert torch.equal(padded[:64], base)
        x, weight, bias = _inputs((68, MODEL_K + 1, MODEL_N))
        with _backend("general"):
            _, base = _forward_device(
                x[:64].contiguous().reshape(-1, MODEL_K + 1), weight, bias, emit_pre=True
            )
            _, padded = _forward_device(x.reshape(-1, MODEL_K + 1), weight, bias, emit_pre=True)
            assert torch.equal(padded[:64], base)

    def test_pre_reshaped_batch_invariant(self):
        """The op's lead dims are a reshape only: the fp32 rows are unchanged."""

        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            _, flat = _forward_device(x.reshape(-1, MODEL_K), weight, bias, emit_pre=True)
            _, lead = _forward_device(
                x.reshape(4, 16, MODEL_K).reshape(-1, MODEL_K), weight, bias, emit_pre=True
            )
            assert torch.equal(lead, flat)


# ---------------------------------------------------------------------------
# the GELU value: declared tolerance, deviation confined to the tanh
# ---------------------------------------------------------------------------
@CUDA
class TestGeluValue:
    @pytest.mark.parametrize("shape", [SHORT_K, (16, 64, MODEL_N), (32, 20, MODEL_N)])
    def test_short_k_agrees_within_one_ulp(self, shape):
        """Below the mma chain length the orders agree to less than one bf16 ulp.

        On the hardware-order path the tensor core groups the reduction into k16
        steps with its own internal order, while the reference chains
        correctly-rounded FMAs, so the two can differ in the last fp32 bit of
        ``pre`` -- which moves a GELU value's bf16 rounding only when it sits
        within ~1e-6 of a boundary, hence the 1 bf16 ulp bound rather than
        equality. On the portable tree path ``pre`` is byte-equal (see
        :class:`TestPreContract`), so the only residual there is the tanh.
        """

        rows, k_dim, n_dim = shape
        assert k_dim <= BIT_EXACT_MAX_K
        x, weight, bias = _inputs(shape)
        got = op_forward(x, weight, bias)
        identical, worst = _deviation(got, _reference(x, weight, bias))
        assert worst <= 1.0, f"worst={worst:.3f} ulp"
        assert identical >= 0.99, f"identical={identical}"

    @pytest.mark.parametrize("shape", [IMG_MLP_UP_SHAPE, TXT_MLP_UP_SHAPE])
    def test_model_shapes_match_reference_within_declared_tolerance(self, shape):
        """Full model shape on the device; the fp32 CPU reference covers a slice.

        The reference runs on the cached ``TREE_REFERENCE_ROWS`` slice of the
        model's own K/N (a whole tier costs ~1 s per row), and its GELU value is
        derived from its own cached ``pre`` (``forward`` is exactly
        ``gelu_tanh_fp32(pre)``), so the value comparison and the ``pre``
        byte-equality anchor below share one K-step tree.
        """

        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _inputs(shape)
        got = op_forward(x, weight, bias)
        assert got.shape == (shape[0], shape[2])
        (ref_pre,) = _reference_rows(shape, "pre", checked)
        identical, worst_ulps = _deviation(got[:checked], gelu_tanh_fp32(ref_pre))
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"identical={identical}"
        assert worst_ulps <= DECLARED_MAX_ULPS, f"worst={worst_ulps:.2f} ulp"
        # The reference that produced those values used the *same* fp32
        # pre-activation the device did, so the residual is the tanh alone.
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        assert _byte_mismatches(pre[:checked], ref_pre) == 0

    def test_deviation_is_confined_to_the_tanh(self):
        """Where the tanh saturates, the GELU is polynomial-exact: byte equality.

        ``tanh(S t)`` is exactly ``+-1.0f`` in fp32 once ``|S t|`` exceeds ~9, so
        both the correctly-rounded reference and the device's ``tanhf`` return the
        same value and the GELU collapses to the pinned polynomial. Any element
        there must match the reference's bf16 store bit for bit; mismatches may
        only live in the non-saturated region, which is what "the deviation is
        confined to the tanh" means operationally.
        """

        shape = (ANCHOR_TOKENS, MODEL_K, MODEL_N)
        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _saturating_inputs(shape)
        with _backend("general"):
            got, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        # The fp32 CPU tree costs ~1 s per row at this K/N: the anchor and the
        # saturation mask are computed on the bounded slice.
        ref_pre = _reference_pre(x[:checked], weight, bias)
        assert _byte_mismatches(pre[:checked], ref_pre) == 0, "pre is not the anchor here"
        want = gelu_tanh_fp32(ref_pre).bfloat16()
        # 12 is safely past tanhf's saturation point: tanh(12) = 1 - 1.1e-10, which
        # rounds to exactly 1.0f for both the device and the correctly-rounded
        # reference (whereas near 9.0 the two could still land on different bits).
        saturated = gelu_tanh_argument(ref_pre).abs() >= 12.0
        assert bool(saturated.any()), "the saturation mask must be non-empty"
        differ = got[:checked].cpu().bfloat16() != want
        assert int((differ & saturated).sum()) == 0, (
            f"{int((differ & saturated).sum())} saturated elements differ; the "
            "deviation is not confined to the tanh"
        )
        identical, worst = _deviation(got[:checked], gelu_tanh_fp32(ref_pre))
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"identical={identical}"
        assert worst <= GELU_MAX_ULPS, f"worst={worst:.3f} ulp outside the tanh"
        free = int((differ & ~saturated).sum())
        assert free <= max(
            1, int(0.05 * int((~saturated).sum()))
        ), f"too many non-saturated mismatches: {free}"


# ---------------------------------------------------------------------------
# the pinned schedule's structural promise
# ---------------------------------------------------------------------------
@CUDA
class TestScheduleContract:
    # synthetic short reductions: the model's own length is covered at K = 3072 below
    @pytest.mark.parametrize("k_dim", [16, 32, 48, 96, 128])
    def test_forward_covers_each_k_exactly_once(self, k_dim):
        weight = torch.ones(4, k_dim, dtype=torch.bfloat16).cuda()
        total = 0.0
        for k in range(k_dim):
            x = torch.zeros(1, k_dim, dtype=torch.bfloat16).cuda()
            x[0, k] = 1.0
            # The fp32 anchor (not the GELU) is what "counted once" is about.
            _, pre = _forward_device(x, weight, None, emit_pre=True)
            got = pre.float()[0, 0].item()
            # Weight of ones: a unit at reduction index k must contribute exactly
            # once, so the pre-activation is exactly 1 (a skipped index gives 0, a
            # doubled one gives 2).
            assert got == 1.0, f"index {k} not counted exactly once ({got})"
            total += got
        # Every index contributed, and no index contributed twice.
        assert total == float(k_dim)

    def test_forward_deterministic(self):
        x, weight, bias = _inputs(IMG_MLP_UP_SHAPE)
        first = op_forward(x, weight, bias)
        for _ in range(3):
            assert torch.equal(op_forward(x, weight, bias), first)

    def test_forward_row_invariant(self):
        x, weight, bias = _inputs(SMALL)
        full = op_forward(x, weight, bias)
        for rows in (1, 7, x.shape[0] - 1):
            part = op_forward(x[:rows].contiguous(), weight, bias)
            assert torch.equal(part, full[:rows]), f"rows={rows}"

    def test_forward_tiling_invariant(self):
        """Padded tiles must not leak into a logical row's bytes."""

        x, weight, bias = _inputs((130, 96, MODEL_N))
        base = op_forward(x[:64].contiguous(), weight, bias)
        padded = op_forward(x, weight, bias)
        assert torch.equal(padded[:64], base)


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
    (16, MODEL_K - 1, MODEL_N),
]
TREE_BACKWARD_SHAPES = [
    (ANCHOR_TOKENS, MODEL_K, MODEL_N),
    (REF_TOKENS[0], MODEL_K, MODEL_N),
    SHORT_K,
    (64, 100, MODEL_N),
    (16, MODEL_K + 1, MODEL_N),
]


def _backward_with_device_gate(shape, checked):
    """Device ``(dx, dW, db)`` plus the reference triple computed with the *device* gate.

    The gate is formed by the device kernel from the same ``pre`` and
    ``grad_output``, then handed to ``mlp_up_gemm_gelu_reference_backward`` as
    ``gate=``: that replaces the reference's transcendental derivative, so what
    the byte comparison isolates is exactly the three contractions.
    """

    from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

    rows, k_dim, n_dim = shape
    x, weight, bias = _inputs(shape)
    grad = _grad_of(shape, checked)
    xr = x[:checked].clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    with _backend("general"):
        CudaMlpUpGemmGeluOp()(xr, wr, bias=br).backward(grad)
        _, pre = _forward_device(x[:checked].reshape(-1, k_dim), weight, bias, emit_pre=True)
    device_gate = _gate_device(grad, pre)
    ref = _reference_backward_with_gate(
        x[:checked], weight, pre, grad, device_gate, ("backward", k_dim, n_dim, checked)
    )
    return (xr.grad, wr.grad, br.grad), ref


@CUDA
class TestTreeContractByteEquality:
    """The general path computes the reference's tree: byte equality, no tolerance.

    Everything here pins ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=general``, so on a
    Hopper host with the wgmma entry points built this is the portable path and
    not the hardware order. The fp32 CPU reference is the expensive side: it costs
    O(M * K * N) element-ops (~1 s per row at the model's K/N), so every comparison
    here runs on the cached ``TREE_REFERENCE_ROWS`` slice -- one cache entry serves
    the 256-token anchor and all three reference tiers, because ``_inputs`` draws
    the slice's operands from seeded generators -- and the full tiers keep their
    shape with their bytes pinned device-vs-device by
    :class:`TestTreePathInvariance`. The ``pre`` anchor itself lives in
    :class:`TestPreContract`; the ``dx``/``dW``/``db`` comparisons here feed the
    reference the device's own gate so the transcendental is out of the comparison.
    Every mismatch count reported on failure is the *number of differing elements*,
    which the contract requires to be zero.
    """

    @pytest.mark.parametrize("shape", TREE_FORWARD_SHAPES)
    def test_forward_value_matches_the_reference_gelu(self, shape):
        rows = shape[0]
        checked = min(rows, TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            got = op_forward(x, weight, bias)
        # The cached `forward` is `gelu_tanh_fp32` of the cached `pre`, i.e. exactly
        # `mlp_up_gemm_gelu_reference_forward` on the same bounded slice.
        (ref,) = _reference_rows(shape, "forward", checked)
        identical, worst = _deviation(got[:checked], ref)
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{shape}: identical={identical}"
        assert worst <= DECLARED_MAX_ULPS, f"{shape}: worst={worst:.2f} ulp"
        assert got.dtype is torch.bfloat16

    @pytest.mark.parametrize("shape", TREE_FORWARD_SHAPES)
    def test_forward_pre_is_byte_equal(self, shape):
        rows = shape[0]
        checked = min(rows, TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        (ref,) = _reference_rows(shape, "pre", checked)
        mismatches = _byte_mismatches(pre[:checked], ref)
        assert (
            mismatches == 0
        ), f"{shape}: {mismatches} of {ref.numel()} fp32 elements differ from the reference"

    @pytest.mark.parametrize("shape", TREE_BACKWARD_SHAPES)
    def test_backward_is_byte_equal_to_the_reference(self, shape):
        """With the device gate fed to the reference, the contractions are byte-equal."""

        checked = min(shape[0], TREE_REFERENCE_ROWS)
        got, ref = _backward_with_device_gate(shape, checked)
        for name, g, r in (("dx", got[0], ref[0]), ("dW", got[1], ref[1]), ("db", got[2], ref[2])):
            want = r.bfloat16()
            mismatches = _byte_mismatches(g, want)
            assert mismatches == 0, (
                f"{shape} {name}: {mismatches} of {want.numel()} bf16 elements differ "
                "from the fp32 reference (device gate)"
            )


@CUDA
class TestTreePathInvariance:
    """The general path's bytes cannot depend on the batch, the tile or a rerun."""

    def test_forward_deterministic(self):
        x, weight, bias = _inputs(IMG_MLP_UP_SHAPE)
        with _backend("general"):
            first = op_forward(x, weight, bias)
            for _ in range(3):
                assert torch.equal(op_forward(x, weight, bias), first)

    def test_forward_row_invariant(self):
        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            full = op_forward(x, weight, bias)
            for rows in (1, 7, x.shape[0] - 1):
                assert torch.equal(
                    op_forward(x[:rows].contiguous(), weight, bias), full[:rows]
                ), f"rows={rows}"

    def test_forward_tiling_invariant(self):
        """Padded tiles (and the partial k leaf) must not leak into a row's bytes."""

        x, weight, bias = _inputs((130, 96, MODEL_N))
        with _backend("general"):
            base = op_forward(x[:64].contiguous(), weight, bias)
            assert torch.equal(op_forward(x, weight, bias)[:64], base)
        x, weight, bias = _inputs((68, MODEL_K + 1, MODEL_N))
        with _backend("general"):
            base = op_forward(x[:64].contiguous(), weight, bias)
            assert torch.equal(op_forward(x, weight, bias)[:64], base)

    def test_backward_deterministic(self):
        x, weight, bias = _inputs(SMALL)
        grad = _grad_of(SMALL)

        def run():
            with _backend("general"):
                xr = x.detach().clone().requires_grad_(True)
                wr = weight.detach().clone().requires_grad_(True)
                br = bias.detach().clone().requires_grad_(True)
                op_forward(xr, wr, br).backward(grad)
                return xr.grad, wr.grad, br.grad

        first = run()
        for _ in range(3):
            assert all(torch.equal(a, b) for a, b in zip(run(), first))


# ---------------------------------------------------------------------------
# the gate and the emit_pre contract
# ---------------------------------------------------------------------------
@CUDA
class TestGateAndEmitPre:
    """The activation gradient at the dtype boundary, and the inference path.

    The gate is ``bf16_rne(f32(grad_y) * gelu_tanh_grad_fp32(pre))``. Device and
    reference agree bit for bit wherever ``tanh`` is saturated (then the
    derivative is the pinned polynomial and the fp32 product is
    correctly-rounded); elsewhere the device's ``tanhf`` (max 2 ulp) is the only
    free operand, so the comparison is the declared gate tolerance.
    """

    GATE_SHAPE = (ANCHOR_TOKENS, MODEL_K, MODEL_N)

    def test_gate_matches_the_reference_gate(self):
        shape = self.GATE_SHAPE
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        got = _gate_device(grad, pre)
        ref = mlp_up_gemm_gelu_reference_gate(pre.float().cpu(), grad.float().cpu())
        assert got.ndim == 2 and got.shape == grad.shape
        identical, worst = _gate_deviation(got, ref)
        assert identical >= GATE_MIN_IDENTICAL_FRACTION, f"gate identical={identical}"
        assert worst <= GATE_MAX_ULPS, f"gate worst={worst:.3f} ulp"

    def test_gate_is_tanh_free_where_the_tanh_saturates(self):
        """On the saturated region the gate is the pinned polynomial, byte for byte."""

        shape = self.GATE_SHAPE
        x, weight, bias = _saturating_inputs(shape)
        grad = _grad_of(shape)
        with _backend("general"):
            _, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        got = _gate_device(grad, pre)
        ref = mlp_up_gemm_gelu_reference_gate(pre.float().cpu(), grad.float().cpu())
        saturated = gelu_tanh_argument(pre.float().cpu()).abs() >= 12.0
        assert bool(saturated.any()), "the saturation mask must be non-empty"
        differ = (got.cpu().bfloat16() != ref) & saturated
        assert int(differ.sum()) == 0, (
            f"{int(differ.sum())} saturated gate elements differ; the deviation is not "
            "confined to the tanh"
        )

    def test_emit_pre_false_does_not_materialize_pre(self):
        x, weight, bias = _inputs(SMALL)
        x2d = x.reshape(-1, MODEL_K)
        out_with, with_pre = _forward_device(x2d, weight, bias, emit_pre=True)
        assert with_pre.numel() == x.shape[0] * MODEL_N
        assert with_pre.dtype is torch.float32
        out_without, without_pre = _forward_device(x2d, weight, bias, emit_pre=False)
        assert without_pre.numel() == 0, "emit_pre=False must not write pre"
        assert without_pre.dtype is torch.float32
        # The store is a pure function of the pre-activation, so skipping pre
        # cannot move a single bf16 byte of the output.
        assert _byte_mismatches(out_without, out_with) == 0

    def test_emit_pre_false_matches_under_no_grad(self):
        """``torch.no_grad()`` drops ``pre`` and cannot move ``y``.

        The ``emit_pre`` decision is taken at this op-level call site, not inside
        the autograd Function (torch runs a custom Function's ``forward`` with grad
        mode disabled): a grad-enabled call whose operands ask for a gradient
        materializes the fp32 ``pre`` the backward's gate needs, a ``no_grad`` call
        does not. The store is a pure function of the pre-activation, so dropping
        ``pre`` cannot move a byte of ``y`` -- compared here through the *routed*
        op, so both sides are the same path.
        """

        from rl_engine.kernels.ops.cuda.linear import mlp_up_gemm_gelu as mod

        x, weight, bias = _inputs(SMALL)
        xr, wr, br = (t.clone().requires_grad_(True) for t in (x, weight, bias))
        x2d = x.reshape(-1, MODEL_K)
        with torch.enable_grad():
            out_grad = op_forward(xr, wr, br)
        assert mod._needs_pre(xr, wr, br) is True, "a gradient-able call must keep pre"
        with torch.no_grad():
            assert mod._needs_pre(x, weight, bias) is False, "no_grad must drop pre"
            out_infer = op_forward(x, weight, bias)
            _, pre = _forward_device(x2d, weight, bias, emit_pre=False)
        assert pre.numel() == 0
        assert _byte_mismatches(out_infer, out_grad.detach()) == 0

    def test_emit_pre_false_is_byte_equal_over_reruns(self):
        x, weight, bias = _inputs(SMALL)
        x2d = x.reshape(-1, MODEL_K)
        first, _ = _forward_device(x2d, weight, bias, emit_pre=False)
        for _ in range(3):
            out, pre = _forward_device(x2d, weight, bias, emit_pre=False)
            assert pre.numel() == 0
            assert _byte_mismatches(out, first) == 0


# ---------------------------------------------------------------------------
# the two CUDA paths and the contract each one publishes
# ---------------------------------------------------------------------------
@CUDA
class TestCudaPathsAndContracts:
    """The general (tree) and Hopper (mma) paths are different contracts.

    Each is verified against its own oracle elsewhere -- the tree path byte-for-
    byte against the fp32 CPU reference above, the Hopper path against Triton in
    ``tests/test_mlp_up_gemm_gelu_triton.py``. What is checked here is the
    routing: which path a call takes, what contract it reports, and that no path
    silently becomes the other.
    """

    PAIR_SHAPES = [(1, 96, MODEL_N), (130, 96, 64), (32, MODEL_K, 64), (7, MODEL_K, MODEL_N)]

    @pytest.mark.parametrize("shape", PAIR_SHAPES)
    def test_general_path_never_falls_back(self, shape):
        """Every shape the row supports is served by the tree, whose pre is the reference's.

        The fp32 CPU reference is bounded to ``TREE_REFERENCE_ROWS`` rows (it costs
        ~1 s per row at K = 3072); the routing assertions cover the whole shape.
        """

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_backend_used,
            mlp_up_gemm_gelu_contract_used,
        )

        checked = min(shape[0], TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        with _backend("general"):
            assert mlp_up_gemm_gelu_backend_used(x, weight) == "general"
            assert mlp_up_gemm_gelu_contract_used(x, weight) == TREE_CONTRACT
            got, pre = _forward_device(x.reshape(-1, shape[1]), weight, bias, emit_pre=True)
        (ref_pre,) = _reference_rows(shape, "pre", checked)
        assert _byte_mismatches(pre[:checked], ref_pre) == 0
        (ref,) = _reference_rows(shape, "forward", checked)
        identical, worst = _deviation(got[:checked], ref)
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION and worst <= DECLARED_MAX_ULPS

    @pytest.mark.parametrize("shape", [(64, 100, MODEL_N), (16, MODEL_K + 1, MODEL_N)])
    def test_odd_reduction_tails_still_correct(self, shape):
        """K that is not a multiple of 8 (or of 32) runs the portable tree, exactly.

        The Hopper entries refuse such a tensor map (the row strides have to be
        multiples of 8 elements); the wrapper then runs the general kernel, whose
        pre-activation is byte-equal to the fp32 reference, and reports the tree
        contract. The reference comparison is bounded to ``TREE_REFERENCE_ROWS``
        rows (a K = 3073 tree costs ~1 s per row); one cache entry serves it and
        :class:`TestTreeContractByteEquality`'s tail shapes.
        """

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_contract_used,
        )

        rows, k_dim, n_dim = shape
        checked = min(rows, TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        assert mlp_up_gemm_gelu_contract_used(x, weight) == TREE_CONTRACT
        got, pre = _forward_device(x.reshape(-1, k_dim), weight, bias, emit_pre=True)
        (ref_pre,) = _reference_rows(shape, "pre", checked)
        assert _byte_mismatches(pre[:checked], ref_pre) == 0, f"K={k_dim}: pre-activation differs"
        (ref,) = _reference_rows(shape, "forward", checked)
        identical, worst = _deviation(got[:checked], ref)
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION and worst <= DECLARED_MAX_ULPS

    def test_below_cc9_the_tree_contract_serves_and_is_unchanged(self, monkeypatch):
        """Below cc 9.0 the auto route is the portable tree, not a changed result."""

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_backend_used,
            mlp_up_gemm_gelu_contract_used,
        )

        x, weight, bias = _inputs(SMALL)
        with _backend("general"):
            forced = op_forward(x, weight, bias)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
        assert mlp_up_gemm_gelu_backend_used(x, weight) == "general"
        assert mlp_up_gemm_gelu_contract_used(x, weight) == TREE_CONTRACT
        assert torch.equal(op_forward(x, weight, bias), forced)

    def test_backend_env_selects_the_contract(self, monkeypatch):
        """`RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND` pins the contract on one machine.

        `general` must produce the fp32 reference's pre-activation (the portable
        tree, ``mlp-up-gemm-gelu-tree``), `hopper` must refuse rather than
        silently change the arithmetic order when it cannot serve the operands,
        and an unknown value is an error.
        """

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_contract_used,
        )

        x, weight, bias = _inputs(SMALL)
        monkeypatch.setenv(BACKEND_ENV, "general")
        assert mlp_up_gemm_gelu_contract_used(x, weight) == TREE_CONTRACT
        _, pre = _forward_device(x.reshape(-1, MODEL_K), weight, bias, emit_pre=True)
        # Bounded: this entry is the shared (pre, K, N, TREE_REFERENCE_ROWS) cache
        # entry that the anchor and every reference tier compare against.
        (ref_pre,) = _reference_rows(SMALL, "pre", TREE_REFERENCE_ROWS)
        assert _byte_mismatches(pre[:TREE_REFERENCE_ROWS], ref_pre) == 0
        monkeypatch.setenv(BACKEND_ENV, "bogus")
        with pytest.raises(ValueError):
            op_forward(x, weight, bias)
        # `hopper` is an error when it cannot serve, never a fallback to the tree
        monkeypatch.setenv(BACKEND_ENV, "hopper")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
        with pytest.raises(RuntimeError):
            op_forward(x, weight, bias)

    def test_route_report_is_emitted_once(self, monkeypatch, capsys):
        """First call emits `[RL-Kernel][route] ... module=mlp_up_gemm_gelu ...` once."""

        from rl_engine.kernels.ops.cuda.linear import mlp_up_gemm_gelu as mod

        monkeypatch.setattr(mod, "_ROUTE_REPORTED", False)
        monkeypatch.delenv(mod._BACKEND_ENV, raising=False)
        monkeypatch.delenv("RL_KERNEL_ROUTE_REPORT", raising=False)
        x, weight, bias = _inputs(SMALL)
        mod.CudaMlpUpGemmGeluOp()(x, weight, bias=bias)
        first = capsys.readouterr().out
        assert "module=mlp_up_gemm_gelu" in first
        # a cc 9.0 device only takes the Hopper path when its symbols are compiled in
        # (`KERNEL_ALIGN_FORCE_SM90=1`); otherwise the route is the portable tree
        expected = "hopper" if _hopper_serves(x.device) else "general"
        order_contract = mod.MMA_CONTRACT if expected == "hopper" else mod.TREE_CONTRACT
        assert "requested=auto" in first and f"actual={expected}" in first
        assert f"contract={order_contract}" in first
        mod.CudaMlpUpGemmGeluOp()(x, weight, bias=bias)
        assert "module=mlp_up_gemm_gelu" not in capsys.readouterr().out

    def test_route_report_names_the_contract_and_the_pin(self, monkeypatch, capsys):
        """A pinned `general` reports the tree contract and how to pin it back."""

        from rl_engine.kernels.ops.cuda.linear import mlp_up_gemm_gelu as mod

        monkeypatch.setattr(mod, "_ROUTE_REPORTED", False)
        monkeypatch.setenv(mod._BACKEND_ENV, "general")
        monkeypatch.delenv("RL_KERNEL_ROUTE_REPORT", raising=False)
        x, weight, bias = _inputs(SMALL)
        mod.CudaMlpUpGemmGeluOp()(x, weight, bias=bias)
        out = capsys.readouterr().out
        assert "requested=general" in out and "actual=general" in out
        assert f"contract={mod.TREE_CONTRACT}" in out
        # the text says the change is a contract change and how to pin it back
        assert "portable_tree_contract" in out
        assert f"{mod._BACKEND_ENV}=hopper" in out

    def test_reports_requested_actual_backend_and_contract(self, monkeypatch):
        """Requested backend, actual backend, fallback state and contract are queryable."""

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_backend,
            mlp_up_gemm_gelu_backend_used,
            mlp_up_gemm_gelu_contract_used,
        )

        x, weight, _ = _inputs(SMALL)
        assert mlp_up_gemm_gelu_backend() == "auto"
        # the cc 9.0 path needs its symbols built in; without them auto falls back
        expected = "hopper" if _hopper_serves(x.device) else "general"
        assert mlp_up_gemm_gelu_backend_used(x, weight) == expected
        assert mlp_up_gemm_gelu_contract_used(x, weight) == (
            MMA_CONTRACT if expected == "hopper" else TREE_CONTRACT
        )
        monkeypatch.setenv(BACKEND_ENV, "general")
        assert mlp_up_gemm_gelu_backend_used(x, weight) == "general"
        assert mlp_up_gemm_gelu_contract_used(x, weight) == TREE_CONTRACT

    def test_rocm_dispatch_uses_triton(self, monkeypatch):
        """On a ROCm torch the registry resolves this row to the Triton backend."""

        from rl_engine.kernels.registry import kernel_registry

        monkeypatch.setattr(torch.version, "hip", "6.0.0")
        op = kernel_registry.get_op("mlp_up_gemm_gelu", device=torch.device("cuda"))
        assert type(op).__name__ in ("TritonMlpUpGemmGeluOp", "NativeMlpUpGemmGeluOp")

    def test_cuda_op_fails_closed_on_rocm(self, monkeypatch):
        """The CUDA op must refuse on ROCm rather than reach for absent symbols."""

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _inputs(SMALL)
        monkeypatch.setattr(torch.version, "hip", "6.0.0")
        with pytest.raises(RuntimeError, match="Triton"):
            CudaMlpUpGemmGeluOp()(x, weight, bias=bias)

    def test_degenerate_and_rank3_shapes(self):
        x, weight, bias = _inputs(SMALL)
        empty = op_forward(x[:0].contiguous(), weight, bias)
        assert empty.shape == (0, SMALL[2])
        x3 = torch.randn(4, 8, SMALL[1], generator=torch.Generator().manual_seed(1))
        x3 = x3.to(torch.bfloat16).cuda()
        flat = x3.reshape(-1, SMALL[1])
        assert torch.equal(
            op_forward(x3, weight, bias),
            op_forward(flat, weight, bias).reshape(4, 8, SMALL[2]),
        )


# ---------------------------------------------------------------------------
# the Hopper path: the hardware-order contract
# ---------------------------------------------------------------------------
@CUDA
class TestHopperPath:
    """``mlp-up-gemm-gelu-mma`` on Hopper: the row's hardware order.

    Byte identity with the Triton backend (which implements the same order) is
    asserted in ``tests/test_mlp_up_gemm_gelu_triton.py``; here the path is
    pinned with ``RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=hopper`` and checked against
    the independent fp32 reference. This is the *mma* contract, whose reduction
    associates the sum differently than the reference's tree, so its ``pre`` is
    compared under a tight fp32-ulp bound (not byte equality -- that is the tree
    contract's claim); its ``db`` is bit-identical (the shared fold of the gate),
    and the GELU value and the two GEMM contractions are declared-tolerance
    comparisons.
    """

    def _skip_without_hopper(self, device) -> None:
        if not _hopper_serves(device):
            pytest.skip("the Hopper path needs KERNEL_ALIGN_FORCE_SM90=1 and a cc 9.x device")

    @pytest.mark.parametrize(
        "shape", [SMALL, (ANCHOR_TOKENS, MODEL_K, MODEL_N), (REF_TOKENS[0], MODEL_K, MODEL_N)]
    )
    def test_forward_pre_within_fp32_ulp_and_value_within_tolerance(self, shape):
        """The hardware order's ``pre`` against the reference tree: fp32 ulps, not bytes.

        The mma order (wgmma k16 chunks chained into one fp32 accumulator) and the
        reference's 32-wide-leaf mid-split tree associate the same sum differently,
        so their fp32 ``pre`` differs in the last bits (measured on the H100 at
        K = 3072: only ~0.5% of elements bit-identical, worst element ~40 fp32
        ulps from the reference -- the size of a sqrt(K)-step accumulation-order
        difference). Byte equality of ``pre`` is the *tree* contract's claim
        (:class:`TestPreContract`) and holds between the two mma implementations
        (Hopper vs Triton); here the bound is the fp32 bound, and the identity
        fraction is reported. The GELU value keeps the declared bf16 tolerance.
        """

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_backend_used,
            mlp_up_gemm_gelu_contract_used,
        )

        x, weight, bias = _inputs(shape)
        self._skip_without_hopper(x.device)
        with _backend("hopper"):
            assert mlp_up_gemm_gelu_backend_used(x, weight) == "hopper"
            assert mlp_up_gemm_gelu_contract_used(x, weight) == MMA_CONTRACT
            got, pre = _forward_device(
                x.reshape(-1, shape[1]), weight, bias, emit_pre=True, sm90=True
            )
        (ref_pre,) = _reference_rows(shape, "pre", TREE_REFERENCE_ROWS)
        pre_identical, worst_pre = _pre_deviation(pre[:TREE_REFERENCE_ROWS], ref_pre)
        assert worst_pre <= FP32_MAX_ULPS, (
            f"{shape}: the Hopper pre-activation is {worst_pre:.2f} fp32 ulps from "
            f"the reference tree (identical={pre_identical:.4f} of elements)"
        )
        (ref,) = _reference_rows(shape, "forward", TREE_REFERENCE_ROWS)
        identical, worst = _deviation(got[:TREE_REFERENCE_ROWS], ref)
        assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{shape}: identical={identical}"
        assert worst <= DECLARED_MAX_ULPS, f"{shape}: worst={worst:.2f} ulp"

    def test_backward_matches_the_reference_within_the_declared_tolerance(self):
        """``dx``/``dW`` are the hardware order; ``db`` is the shared bit-exact fold.

        The reference runs on the bounded ``TREE_REFERENCE_ROWS`` slice (the fp32
        CPU tree costs ~1 s per row at the model's K/N) and the device gradients are
        taken on that same slice, so the row-folding ``dW``/``db`` compare like for
        like. The slice's cache entry is shared with the tree path, because both
        paths publish the same (byte-equal) ``pre`` and hence the same device gate.
        """

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        shape = (32, MODEL_K, MODEL_N)
        checked = TREE_REFERENCE_ROWS
        x, weight, bias = _inputs(shape)
        self._skip_without_hopper(x.device)
        grad = _grad_of(shape, checked)
        xr = x[:checked].clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        br = bias.detach().clone().requires_grad_(True)
        with _backend("hopper"):
            CudaMlpUpGemmGeluOp()(xr, wr, bias=br).backward(grad)
            _, pre = _forward_device(
                x[:checked].reshape(-1, MODEL_K), weight, bias, emit_pre=True, sm90=True
            )
        gate = _gate_device(grad, pre)
        ref_dx, ref_dw, ref_db = _reference_backward_with_gate(
            x[:checked], weight, pre, grad, gate, ("backward", MODEL_K, MODEL_N, checked)
        )
        for name, got, ref in (("dx", xr.grad, ref_dx), ("dW", wr.grad, ref_dw)):
            identical, worst = _deviation(got, ref)
            assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{name} identical={identical}"
            assert worst <= DECLARED_MAX_ULPS, f"{name} worst={worst:.2f} ulp"
        # db takes the (shared) gate as its only operand, so it is byte-exact here
        assert torch.equal(br.grad.cpu(), ref_db.to(torch.bfloat16))

    def test_hopper_is_requestable_and_the_contract_is_reported(self):
        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import (
            mlp_up_gemm_gelu_backend_used,
        )

        x, weight, bias = _inputs(SMALL)
        self._skip_without_hopper(x.device)
        with _backend("hopper"):
            assert mlp_up_gemm_gelu_backend_used(x, weight) == "hopper"
            assert op_forward(x, weight, bias).dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
@CUDA
class TestBackward:
    @pytest.mark.parametrize("shape", [SMALL, (32, MODEL_K, MODEL_N)])
    def test_gradients_match_reference_within_declared_tolerance(self, shape):
        """The routed path's gradients vs the fp32 reference with the device's gate.

        The reference walks K python-level steps plus two folds (~1 s per row at the
        model's K/N), so it runs on the bounded ``TREE_REFERENCE_ROWS`` slice, and
        the device gradients are taken on that same slice: ``dW``/``db`` fold over
        the batch rows, so slicing keeps both sides comparable. The slice's cache
        entry is shared with :class:`TestTreeContractByteEquality` (the gate's first
        row is part of the cache key).
        """

        checked = min(shape[0], TREE_REFERENCE_ROWS)
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, checked)
        xr = x[:checked].clone().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        br = bias.detach().clone().requires_grad_(True)
        op_forward(xr, wr, br).backward(grad)
        pre = _route_pre(x[:checked].reshape(-1, MODEL_K), weight, bias)
        gate = _gate_device(grad, pre)
        ref_dx, ref_dw, ref_db = _reference_backward_with_gate(
            x[:checked], weight, pre, grad, gate, ("backward", MODEL_K, MODEL_N, checked)
        )
        for name, got, ref in (("dx", xr.grad, ref_dx), ("dW", wr.grad, ref_dw)):
            identical, worst = _deviation(got, ref)
            assert identical >= DECLARED_MIN_IDENTICAL_FRACTION, f"{name} identical={identical}"
            assert worst <= DECLARED_MAX_ULPS, f"{name} worst={worst:.2f} ulp"
        # db is the same ascending fp32 left fold of the gate in both
        # implementations, so it is bit-exact after the single bf16 cast.
        assert torch.equal(br.grad.cpu(), ref_db.to(torch.bfloat16))

    def test_db_is_the_left_fold_of_the_gate(self):
        """db must be the ascending-row fp32 fold of the *routed* device gate."""

        shape = (24, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, seed=12)
        bias.requires_grad_(True)
        op_forward(x, weight, bias).backward(grad)
        pre = _route_pre(x.reshape(-1, MODEL_K), weight, bias.detach())
        gate = _gate_device(grad, pre)
        assert torch.equal(bias.grad.cpu(), _gate_fold(gate))

    def test_gradients_deterministic(self):
        x, weight, bias = _inputs(SMALL)
        grad = _grad_of(SMALL, seed=13)

        def run():
            xr = x.clone().requires_grad_(True)
            wr = weight.clone().requires_grad_(True)
            br = bias.clone().requires_grad_(True)
            op_forward(xr, wr, br).backward(grad)
            return xr.grad, wr.grad, br.grad

        first = run()
        for _ in range(3):
            assert all(torch.equal(a, b) for a, b in zip(run(), first))


# ---------------------------------------------------------------------------
# correctness against the exact result, not just the fp32 model
# ---------------------------------------------------------------------------
def _ulp_of_max_magnitude(t: torch.Tensor) -> float:
    """One bf16 ulp of the tensor's largest magnitude (the contract's absolute scale)."""

    return 2.0 ** (math.floor(math.log2(float(t.float().abs().max()))) - 7)


@CUDA
class TestExactTruth:
    """Compare against the exact result; the fp32 model cannot be the only oracle.

    Agreement with the model is self-consistency: it cannot see a mistake that
    the model and the kernel share (a transposed weight, a dropped leaf, a wrong
    cast), and it says nothing about whether the fp32 accumulation order is
    accurate. These checks pin the structural exactness of the anchor directly,
    and then the GELU value and the three contractions against oracles computed
    independently of both implementations.
    """

    def test_integer_inputs_make_the_anchor_bit_exact(self):
        """Small integers make fp32 accumulation exact, so ``pre`` must be exact.

        ``|x|, |W| <= 3`` over ``K = 3072`` keeps every partial sum below
        ``2**24``, i.e. representable, so the sum is order independent: the
        pre-activation has to equal the exact dot product bit for bit, which no
        transposed weight, mis-indexed column or dropped term can survive.
        """

        rows, k_dim, n_dim = 32, MODEL_K, MODEL_N
        gen = torch.Generator().manual_seed(7)
        xi = torch.randint(-3, 4, (rows, k_dim), generator=gen).float()
        wi = torch.randint(-3, 4, (n_dim, k_dim), generator=gen).float()
        bi = torch.randint(-8, 9, (n_dim,), generator=gen).float()
        x, weight, bias = xi.bfloat16().cuda(), wi.bfloat16().cuda(), bi.bfloat16().cuda()
        _, pre = _forward_device(x, weight, bias, emit_pre=True)
        _, pre_nobias = _forward_device(x, weight, None, emit_pre=True)
        exact = (xi.double() @ wi.double().T).float()
        assert _byte_mismatches(pre, (exact.double() + bi.double()).float().cuda()) == 0
        assert _byte_mismatches(pre_nobias, exact.cuda()) == 0

    def test_full_k_identity_and_bias(self):
        k_dim, n_dim = MODEL_K, MODEL_N
        x = torch.ones(3, k_dim, dtype=torch.bfloat16).cuda()
        weight = torch.ones(n_dim, k_dim, dtype=torch.bfloat16).cuda()
        # every term is 1 and every partial sum is an exact small integer, so the
        # total is exactly K in any association order: a dropped or doubled leaf
        # moves it, and the fp32 store of K is itself exact
        _, pre = _forward_device(x, weight, None, emit_pre=True)
        assert torch.equal(pre, torch.full((3, n_dim), float(k_dim), dtype=torch.float32).cuda())
        bias = torch.arange(n_dim, dtype=torch.float32).bfloat16().cuda()
        _, pre_bias = _forward_device(x, weight, bias, emit_pre=True)
        want = (torch.tensor(float(k_dim)) + bias.float()).expand(3, n_dim).contiguous()
        assert _byte_mismatches(pre_bias, want.cuda()) == 0
        # bias alone: x = 0 must reproduce the bias exactly, and only once
        _, pre_only_bias = _forward_device(torch.zeros_like(x), weight, bias, emit_pre=True)
        assert (
            _byte_mismatches(pre_only_bias, bias.float().expand(3, n_dim).contiguous().cuda()) == 0
        )

    @pytest.mark.parametrize("k_dim", [3072, 3071, 3073])
    def test_one_hot_covers_every_leaf_boundary(self, k_dim):
        """Every leaf's first and last reduction index must contribute exactly once.

        At the model's reduction length, so the 96-leaf mid-split tree and its
        short tail leaf are the ones being probed.
        """

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
            _, pre = _forward_device(x, weight, None, emit_pre=True)
            assert pre.float()[0, 0].item() == 1.0, f"k={k} not counted exactly once"

    @pytest.mark.parametrize(
        "shape",
        [SHORT_K, (ANCHOR_TOKENS, MODEL_K, MODEL_N)]
        + [(tokens, MODEL_K, MODEL_N) for tokens in REF_TOKENS],
    )
    def test_gelu_matches_the_fp64_truth(self, shape):
        """Accuracy of the GELU value against an oracle outside both implementations.

        The exact fp64 pre-activation is rounded once to fp32 and then put
        through the *reference's* pinned sequence; the device differs only by its
        ``tanhf``, so the bf16 store must agree except when the value sits within
        a few fp32 ulps of a tie. Measured on this part at ``K = 3072``: every
        element within 1 bf16 ulp of the largest magnitude and >= 99% of elements
        bit-identical; the bounds below keep headroom.
        """

        x, weight, bias = _inputs(shape)
        got = op_forward(x, weight, bias)
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
        bf16 gate the device produced, so the oracle is independent of both
        implementations' association orders and of the GELU derivative.
        """

        shape = (ANCHOR_TOKENS, MODEL_K, MODEL_N)
        x, weight, bias = _inputs(shape)
        grad = _grad_of(shape, seed=11)
        xr, wr, br = (t.clone().requires_grad_(True) for t in (x, weight, bias))
        op_forward(xr, wr, br).backward(grad)
        pre = _route_pre(x.reshape(-1, MODEL_K), weight, bias)
        gate = _gate_device(grad, pre)
        truth_dx = (gate.double() @ weight.double()).bfloat16()
        truth_dw = (gate.double().t() @ x.double()).bfloat16()
        truth_db = _gate_fold(gate).cuda()
        for name, got, want in (
            ("dx", xr.grad, truth_dx),
            ("dW", wr.grad, truth_dw),
            ("db", br.grad, truth_db),
        ):
            deviation = (
                got.float().double() - want.float().double()
            ).abs() / _ulp_of_max_magnitude(want)
            assert deviation.max().item() <= 2.0, f"{name}: worst {deviation.max().item():.3f} ulp"
            assert (
                float((got == want).float().mean()) >= 0.99
            ), f"{name} diverges from the exact gradient"


# ---------------------------------------------------------------------------
# integration: dispatch, fail-closed behaviour, model wiring
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_registry_dispatches_cuda_and_cpu(self):
        """CUDA prefers the Triton path, CPU the fp32 reference.

        Triton lowers the same pinned schedule to wgmma, which is byte-identical
        to the hand-written kernel, so it is the CUDA default; the native kernel
        stays as the no-Triton fallback.
        """

        from rl_engine.kernels.registry import kernel_registry

        cpu_op = kernel_registry.get_op("mlp_up_gemm_gelu", device=torch.device("cpu"))
        assert type(cpu_op).__name__ == "NativeMlpUpGemmGeluOp"
        if torch.cuda.is_available():
            cuda_op = kernel_registry.get_op("mlp_up_gemm_gelu", device=torch.device("cuda"))
            assert type(cuda_op).__name__ in ("TritonMlpUpGemmGeluOp", "CudaMlpUpGemmGeluOp")
            # The native kernel must stay loadable as the fallback path.
            from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

            assert CudaMlpUpGemmGeluOp() is not None

    def test_gtest_spec_registered(self):
        from rl_engine.kernels.gtest.operator_specs import OP_SPECS

        spec = OP_SPECS["mlp_up_gemm_gelu"]
        assert spec.op_class == "reduction"
        assert spec.grad_input_names == ("x", "weight", "bias")
        assert "cuda" in spec.candidate_paths

    def test_gtest_inputs_shapes(self):
        import argparse

        from rl_engine.kernels.gtest.operator_inputs import make_operator_inputs

        args = argparse.Namespace(
            batch=2, seq=16, k_dim=MODEL_K, n_dim=MODEL_N, dtype="float32", seed=0, device="cpu"
        )
        tensors = make_operator_inputs("mlp_up_gemm_gelu", args, torch.float32, torch.device("cpu"))
        assert tensors["x"].shape == (32, MODEL_K)
        assert tensors["weight"].shape == (MODEL_N, MODEL_K)
        assert tensors["bias"].shape == (MODEL_N,)

    @CUDA
    def test_fail_closed_on_fp32_input(self):
        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _inputs(SMALL, dtype=torch.float32)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpUpGemmGeluOp()(x, weight, bias=bias)

    @CUDA
    def test_fail_closed_below_sm80(self, monkeypatch):
        """Below sm80 there is no kernel for either CUDA path: it must raise.

        Below compute capability 8.0 the row's validated support matrix ends, so
        a silent dispatch would run an unvalidated configuration instead of
        failing.
        """

        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _inputs(SMALL)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpUpGemmGeluOp()(x, weight, bias=bias)

    @CUDA
    def test_fail_closed_on_non_bf16_bias(self):
        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _inputs(SMALL)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpUpGemmGeluOp()(x, weight, bias=bias.float())

    @CUDA
    def test_fail_closed_on_bias_shape(self):
        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _inputs(SMALL)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpUpGemmGeluOp()(x, weight, bias=bias[:-1].contiguous())

    @CUDA
    def test_fail_closed_on_k_mismatch(self):
        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _inputs(SMALL)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpUpGemmGeluOp()(x, weight[:, :-16].contiguous(), bias=bias)

    @CUDA
    def test_cc10_does_not_take_the_hopper_path(self, monkeypatch):
        """A cc >= 10 device must not dispatch to the sm_90a-only wgmma body.

        The body only exists under ``__CUDA_ARCH_FEAT_SM90_ALL``: on a
        ``KERNEL_ALIGN_FORCE_SM90=1`` build for a cc >= 10 host it compiles to an
        inert stub, so the runtime gate is cc 9.0 exactly. A cc >= 10 call takes
        the portable tree, and asking for the hardware order raises rather than
        changing the contract silently.
        """

        from rl_engine.kernels.ops.cuda.linear import mlp_up_gemm_gelu as cuda_mod

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
        x, weight, bias = _inputs(SMALL)
        assert cuda_mod.mlp_up_gemm_gelu_backend_used(x, weight) == "general"
        assert cuda_mod.mlp_up_gemm_gelu_contract_used(x, weight) == TREE_CONTRACT
        monkeypatch.setenv(BACKEND_ENV, "hopper")
        with pytest.raises(RuntimeError):
            cuda_mod.CudaMlpUpGemmGeluOp()(x, weight, bias=bias)

    @CUDA
    def test_strided_bias_is_read_at_the_right_elements(self):
        """A view bias must give the same bytes as its contiguous copy.

        The kernels index the bias at unit stride, so the backend has to
        normalize it; before that, a stride-2 bias silently returned wrong values.
        """

        x, weight, _ = _inputs(SMALL, seed=14)
        bias = torch.randn(SMALL[2] * 2, device="cuda", dtype=torch.bfloat16)[::2]
        assert bias.shape == (SMALL[2],) and bias.stride(0) == 2
        assert torch.equal(op_forward(x, weight, bias), op_forward(x, weight, bias.contiguous()))

    def test_fail_closed_on_non_cuda_device(self):
        from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp

        x, weight, bias = _cpu_inputs(SMALL)
        with pytest.raises((ValueError, RuntimeError)):
            CudaMlpUpGemmGeluOp()(x, weight, bias=bias)
