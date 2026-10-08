# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C1 (RFC #428): Qwen3-Next RMSNorm conventions and batch invariance.

Claim levels exercised here (RFC #428 section 2.1):
  * L0 repeatable    -- identical inputs reproduce bitwise identical outputs.
  * L1 batch-invariant -- a row is unaffected by slicing, concurrency and (for the
    PyTorch reference) padding; packing and order are not exercised.

L2 (train-rollout exact against vLLM) is NOT claimed by this file; it needs the
rollout engine on the other side.

The reduction order here is the repo's fixed 32-wide chunked reduction, which
differs from upstream's ``mean(-1)``. What is reproduced exactly is the weight
convention and the cast order, so comparisons against the upstream formula are
tolerance based while the invariance assertions are bitwise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.kernels.ops.pytorch.norm.qwen3_next_rms_norm import (
    Qwen3NextRMSNormGatedHFOp,
    Qwen3NextRMSNormGatedOp,
    Qwen3NextRMSNormOp,
)

# Qwen3-Next-80B-A3B-Instruct config.json
_HIDDEN = 2048  # hidden_size
_HEAD_V_DIM = 128  # linear_value_head_dim -- the gated norm width
_EPS = 1e-6  # rms_norm_eps

_CONTRACT = load_contract()


def _forward_tol(dtype: torch.dtype) -> dict[str, float]:
    """C1 forward_accuracy row for the ``reduction`` op class -- no private thresholds."""
    spec = resolve_tolerance(
        _CONTRACT, judgment="forward_accuracy", op_class="reduction", dtype=dtype
    )
    return {"atol": spec.atol, "rtol": spec.rtol}


def _rand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Upstream formulas, transcribed from transformers/models/qwen3_next.
# These pin the exact semantics; they are deliberately verbatim.
# --------------------------------------------------------------------------- #
def _hf_rms_norm(x, weight, eps=_EPS):
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)


def _vllm_rms_norm_gated(x, weight, gate, eps=_EPS):
    """vLLM ``RMSNormGated.forward_static`` with ``norm_before_gate=True``."""
    orig_dtype = x.dtype
    x, weight, z = x.float(), weight.float(), gate.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out = (x * torch.rsqrt(variance + eps)) * weight
    out = out * F.silu(z)
    return out.to(orig_dtype)


def _hf_rms_norm_gated(x, weight, gate, eps=_EPS):
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    h = weight * h.to(input_dtype)
    h = h * F.silu(gate.to(torch.float32))
    return h.to(input_dtype)


# --------------------------------------------------------------------------- #
# 1. The zero-centred (1 + w) convention
# --------------------------------------------------------------------------- #
def test_zero_weight_is_identity_scaling():
    """weight == 0 must leave the normalized value untouched, bitwise.

    This is what separates the zero-centred convention from the plain one: a
    plain RMSNorm returns all zeros for a zero weight, this one returns the
    bare normalized value.
    """
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp, shape_invariant_rstd

    x = _rand((4, _HIDDEN), seed=0)
    zeros = torch.zeros(_HIDDEN)

    x_f = x.float()
    bare = x_f * shape_invariant_rstd(x_f, _EPS).unsqueeze(-1)
    assert torch.equal(Qwen3NextRMSNormOp().forward_fp32(x, zeros, eps=_EPS), bare)

    # ... and the plain convention really does differ here.
    assert torch.equal(NativeRMSNormOp().forward_fp32(x, zeros, eps=_EPS), torch.zeros_like(bare))


def test_weight_offset_is_one():
    """w and w+1 scaling relationship: out(w) == norm * (1 + w)."""
    op = Qwen3NextRMSNormOp()
    x = _rand((2, _HEAD_V_DIM), seed=1)
    unit = op.forward_fp32(x, torch.zeros(_HEAD_V_DIM))  # scale = 1
    doubled = op.forward_fp32(x, torch.ones(_HEAD_V_DIM))  # scale = 2
    torch.testing.assert_close(doubled, 2.0 * unit, atol=1e-6, rtol=1e-6)


def test_offset_applied_in_fp32_not_folded_into_bf16():
    """The 1 + w offset must not be pre-rounded through bf16.

    A weight one bf16 ULP below zero stays distinguishable from exactly zero
    once the offset is added in fp32; folding (1 + w) into bf16 first would
    collapse both to 1.0 and lose the difference.
    """
    op = Qwen3NextRMSNormOp()
    x = _rand((2, _HEAD_V_DIM), seed=2)
    tiny = torch.full((_HEAD_V_DIM,), -(2**-9), dtype=torch.bfloat16)
    folded = (1.0 + tiny.float()).bfloat16()  # the wrong way
    assert not torch.equal(
        op.forward_fp32(x, tiny),
        op.forward_fp32(x, folded - 1.0),
    )


# --------------------------------------------------------------------------- #
# 2. L1 -- batch invariance, bitwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hidden", [_HIDDEN, _HEAD_V_DIM])
def test_batch_invariance_slice(hidden):
    op = Qwen3NextRMSNormOp()
    w, x = _rand((hidden,), seed=3), _rand((8, 32, hidden), seed=4)
    full = op.forward_fp32(x, w)
    assert torch.equal(op.forward_fp32(x[:1], w), full[:1])
    assert torch.equal(op.forward_fp32(x[3:5], w), full[3:5])


@pytest.mark.parametrize("batch", [1, 2, 8, 16, 32, 48, 64])
def test_batch_invariance_across_concurrency(batch):
    """RFC #428 section 10: batch size / concurrency axis 1..64."""
    op = Qwen3NextRMSNormOp()
    w = _rand((_HIDDEN,), seed=5)
    target = _rand((1, _HIDDEN), seed=6)
    others = _rand((batch - 1, _HIDDEN), seed=7) if batch > 1 else None
    alone = op.forward_fp32(target, w)
    for position in ("first", "last"):
        if others is None:
            batched, index = target, 0
        elif position == "first":
            batched, index = torch.cat([target, others]), 0
        else:
            batched, index = torch.cat([others, target]), batch - 1
        assert torch.equal(op.forward_fp32(batched, w)[index : index + 1], alone)


def test_batch_invariance_with_padding():
    op = Qwen3NextRMSNormOp()
    w = _rand((_HIDDEN,), seed=8)
    x = _rand((4, _HIDDEN), seed=9)
    padded = torch.cat([x, _rand((6, _HIDDEN), seed=10)], dim=0)
    assert torch.equal(op.forward_fp32(padded, w)[:4], op.forward_fp32(x, w))


def test_gated_batch_invariance_slice():
    op = Qwen3NextRMSNormGatedOp()
    w = _rand((_HEAD_V_DIM,), seed=11)
    x = _rand((8, 32, _HEAD_V_DIM), seed=12)
    gate = _rand((8, 32, _HEAD_V_DIM), seed=13)
    full = op.forward_fp32(x, w, gate)
    assert torch.equal(op.forward_fp32(x[:1], w, gate[:1]), full[:1])
    assert torch.equal(op.forward_fp32(x[3:5], w, gate[3:5]), full[3:5])


# --------------------------------------------------------------------------- #
# 2b. The fixed-order reduction is what buys invariance -- on CUDA too
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("hidden", [_HIDDEN, _HEAD_V_DIM])
def test_chunked_reduction_is_slice_invariant_on_device(hidden):
    """The rstd statistic must be slice-invariant on the accelerator, every seed.

    Measured counterpoint on B200/bf16/H=2048: a plain ``mean(-1)`` broke this on
    1 of 20 seeds. We assert only the property we rely on -- asserting that torch's
    ``mean`` is broken would be a brittle test of someone else's kernel.
    """
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import shape_invariant_rstd

    for seed in range(20):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(64, hidden, device="cuda", dtype=torch.bfloat16, generator=g)
        full = shape_invariant_rstd(x.float(), _EPS)
        assert torch.equal(shape_invariant_rstd(x[3:5].float(), _EPS), full[3:5]), seed
        assert torch.equal(shape_invariant_rstd(x[:1].float(), _EPS), full[:1]), seed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_op_batch_invariance_on_device():
    """L1 for the op itself, on the accelerator, in the model's dtype."""
    op = Qwen3NextRMSNormOp()
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(64, _HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    full = op.forward(x, w)
    for n in (1, 2, 8, 16, 32, 48, 64):
        assert torch.equal(op.forward(x[:n], w), full[:n]), n


# --------------------------------------------------------------------------- #
# 3. L0 -- repeatable
# --------------------------------------------------------------------------- #
def test_deterministic_repeat():
    op = Qwen3NextRMSNormOp()
    x, w = _rand((64, _HIDDEN), seed=14), _rand((_HIDDEN,), seed=15)
    first = op.forward_fp32(x, w)
    for _ in range(10):
        assert torch.equal(op.forward_fp32(x, w), first)


# --------------------------------------------------------------------------- #
# 4. Agreement with the upstream formula (tolerance, not bitwise: the reduction
#    order differs by design)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hidden", [_HIDDEN, _HEAD_V_DIM])
def test_matches_upstream_formula_fp32(hidden):
    op = Qwen3NextRMSNormOp()
    x, w = _rand((4, 16, hidden), seed=16), _rand((hidden,), seed=17)
    torch.testing.assert_close(op.forward_fp32(x, w), _hf_rms_norm(x, w), atol=1e-6, rtol=1e-6)


def test_gated_strict_matches_vllm_formula_fp32():
    """The strict gated op follows vLLM, which is what the L2 claim targets."""
    op = Qwen3NextRMSNormGatedOp()
    x = _rand((4, 16, _HEAD_V_DIM), seed=18)
    w = _rand((_HEAD_V_DIM,), seed=19)
    gate = _rand((4, 16, _HEAD_V_DIM), seed=20)
    torch.testing.assert_close(
        op.forward_fp32(x, w, gate),
        _vllm_rms_norm_gated(x, w, gate),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gated_hf_witness_matches_hf_formula_fp32():
    op = Qwen3NextRMSNormGatedHFOp()
    x = _rand((4, 16, _HEAD_V_DIM), seed=18)
    w = _rand((_HEAD_V_DIM,), seed=19)
    gate = _rand((4, 16, _HEAD_V_DIM), seed=20)
    torch.testing.assert_close(
        op.forward_fp32(x, w, gate),
        _hf_rms_norm_gated(x, w, gate),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gated_conventions_diverge_in_low_precision():
    """Pin the HF-vs-vLLM gated divergence (RFC #428 first-divergence boundary).

    With fp32 inputs the two conventions coincide, because the HF round-trip is
    an identity. In bf16 they do not, and the gap is far larger than a ULP: this
    is why the convention is part of the operator identity and not a detail.
    """
    strict, witness = Qwen3NextRMSNormGatedOp(), Qwen3NextRMSNormGatedHFOp()
    x = _rand((64, _HEAD_V_DIM), seed=33).bfloat16()
    w = _rand((_HEAD_V_DIM,), seed=34).bfloat16()
    gate = _rand((64, _HEAD_V_DIM), seed=35).bfloat16()

    a, b = strict.forward(x, w, gate), witness.forward(x, w, gate)
    assert not torch.equal(a, b), "the two gated conventions must be distinguishable"
    assert (a.float() - b.float()).abs().max() > 1e-3

    # ... and in fp32 the round-trip vanishes, so they agree bitwise.
    xf, wf, gf = x.float(), w.float(), gate.float()
    assert torch.equal(strict.forward(xf, wf, gf), witness.forward(xf, wf, gf))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_upstream_formula_low_precision(dtype):
    op = Qwen3NextRMSNormOp()
    x = _rand((4, 16, _HIDDEN), seed=21).to(dtype)
    w = _rand((_HIDDEN,), seed=22).to(dtype)
    got, ref = op.forward(x, w), _hf_rms_norm(x, w)
    assert got.dtype == ref.dtype == dtype
    torch.testing.assert_close(got.float(), ref.float(), **_forward_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_hf_witness_low_precision(dtype):
    """Covers the mid-computation cast back to the input dtype."""
    op = Qwen3NextRMSNormGatedHFOp()
    x = _rand((4, 16, _HEAD_V_DIM), seed=23).to(dtype)
    w = _rand((_HEAD_V_DIM,), seed=24).to(dtype)
    gate = _rand((4, 16, _HEAD_V_DIM), seed=25).to(dtype)
    got, ref = op.forward(x, w, gate), _hf_rms_norm_gated(x, w, gate)
    assert got.dtype == ref.dtype == dtype
    torch.testing.assert_close(got.float(), ref.float(), **_forward_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_strict_low_precision(dtype):
    op = Qwen3NextRMSNormGatedOp()
    x = _rand((4, 16, _HEAD_V_DIM), seed=23).to(dtype)
    w = _rand((_HEAD_V_DIM,), seed=24).to(dtype)
    gate = _rand((4, 16, _HEAD_V_DIM), seed=25).to(dtype)
    got, ref = op.forward(x, w, gate), _vllm_rms_norm_gated(x, w, gate)
    assert got.dtype == ref.dtype == dtype
    torch.testing.assert_close(got.float(), ref.float(), **_forward_tol(dtype))


# --------------------------------------------------------------------------- #
# 5. dtype paths and guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_paths(dtype):
    op = Qwen3NextRMSNormOp()
    x = _rand((2, 16, _HIDDEN), seed=26).to(dtype)
    w = _rand((_HIDDEN,), seed=27).to(dtype)
    assert op.forward(x, w).dtype == dtype
    assert op.forward_fp32(x, w).dtype == torch.float32


def test_eps_inside_sqrt():
    op = Qwen3NextRMSNormOp()
    out = op.forward_fp32(torch.zeros(1, _HIDDEN), torch.zeros(_HIDDEN))
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros(1, _HIDDEN))


def test_bad_weight_shape_raises():
    op = Qwen3NextRMSNormOp()
    with pytest.raises(ValueError, match="weight must be 1-D"):
        op.forward_fp32(_rand((2, _HIDDEN), seed=28), _rand((_HIDDEN - 1,), seed=29))


def test_gated_shape_mismatch_raises():
    op = Qwen3NextRMSNormGatedOp()
    x, w = _rand((2, _HEAD_V_DIM), seed=30), _rand((_HEAD_V_DIM,), seed=31)
    with pytest.raises(ValueError, match="gate must match x"):
        op.forward_fp32(x, w, _rand((3, _HEAD_V_DIM), seed=32))


# --------------------------------------------------------------------------- #
# 6. CUDA kernel: the offset is applied in fp32 inside the kernel
# --------------------------------------------------------------------------- #
_CUDA_RMSNORM = False
if torch.cuda.is_available():  # pragma: no branch - probe only
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        _CUDA_RMSNORM = (
            _EXT_AVAILABLE
            and getattr(_C, "rmsnorm_api_version", None) == 2
            and all(hasattr(_C, name) for name in ("rmsnorm_forward", "rmsnorm_backward_dx"))
        )
    except ImportError:  # pragma: no cover
        _CUDA_RMSNORM = False

requires_cuda_rmsnorm = pytest.mark.skipif(
    not _CUDA_RMSNORM, reason="CUDA RMSNorm extension is not available"
)


@requires_cuda_rmsnorm
def test_cuda_offset_equals_explicit_fp32_weight():
    """``weight_offset=1.0`` must equal passing an fp32 ``1 + w``, bitwise.

    This is the correctness proof for doing the offset inside the kernel: it is
    the same arithmetic as the fp32 reference, not an approximation of it.
    """
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

    torch.manual_seed(0)
    x = torch.randn(512, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)

    offset = rmsnorm_cuda(x, w, eps=_EPS, weight_offset=1.0)
    explicit = rmsnorm_cuda(x, (1.0 + w.float()), eps=_EPS)
    assert torch.equal(offset, explicit)


@requires_cuda_rmsnorm
def test_cuda_offset_is_not_folded_through_bfloat16():
    """Pre-rounding ``1 + w`` to bf16 must give a different answer.

    If this ever becomes equal, the offset has stopped being applied in fp32 and
    the zero-centred contract is silently broken.
    """
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

    torch.manual_seed(0)
    x = torch.randn(512, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)

    offset = rmsnorm_cuda(x, w, eps=_EPS, weight_offset=1.0)
    folded = rmsnorm_cuda(x, (1.0 + w.float()).bfloat16(), eps=_EPS)
    assert not torch.equal(offset, folded)


@requires_cuda_rmsnorm
def test_cuda_default_offset_preserves_plain_convention():
    """An unset offset must leave the existing kernel behaviour untouched."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import RMSNormCudaOp, rmsnorm_cuda

    torch.manual_seed(0)
    x = torch.randn(256, _HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(_HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(RMSNormCudaOp().forward(x, w, eps=_EPS), rmsnorm_cuda(x, w, eps=_EPS))


@requires_cuda_rmsnorm
@pytest.mark.parametrize("batch", [1, 2, 8, 16, 32, 48, 64, 512])
def test_cuda_zero_centred_batch_invariance(batch):
    """L1 for the zero-centred CUDA op across the RFC #428 concurrency axis."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp

    op = Qwen3NextRMSNormCudaOp()
    torch.manual_seed(0)
    x = torch.randn(512, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(op.forward(x[:batch], w, eps=_EPS), op.forward(x, w, eps=_EPS)[:batch])


@requires_cuda_rmsnorm
def test_cuda_zero_centred_within_tolerance_of_golden():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp

    torch.manual_seed(0)
    x = torch.randn(256, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)
    got = Qwen3NextRMSNormCudaOp().forward(x, w, eps=_EPS)
    ref = Qwen3NextRMSNormOp().forward_fp32(x, w, eps=_EPS)
    torch.testing.assert_close(got.float(), ref, **_forward_tol(torch.bfloat16))


@requires_cuda_rmsnorm
def test_cuda_zero_centred_backward_is_offset_aware():
    """dx must see (1 + w); dw is offset-independent since d/dw (1+w) == d/dw w."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

    torch.manual_seed(0)
    x = torch.randn(128, _HEAD_V_DIM, device="cuda", dtype=torch.float32)
    w = torch.randn(_HEAD_V_DIM, device="cuda", dtype=torch.float32)
    dy = torch.randn(128, _HEAD_V_DIM, device="cuda", dtype=torch.float32)

    grads = {}
    for name, off in (("plain", 0.0), ("zero_centred", 1.0)):
        xg = x.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        rmsnorm_cuda(xg, wg, eps=_EPS, weight_offset=off).backward(dy.clone())
        grads[name] = (xg.grad.clone(), wg.grad.clone())

    assert torch.isfinite(grads["zero_centred"][0]).all()
    assert not torch.equal(grads["plain"][0], grads["zero_centred"][0])  # dx differs
    assert torch.equal(grads["plain"][1], grads["zero_centred"][1])  # dw does not


# --------------------------------------------------------------------------- #
# 7. Gated RMSNorm CUDA kernel (the GDN block's norm)
# --------------------------------------------------------------------------- #
_HAS_CUDA_GATED = False
if torch.cuda.is_available():  # pragma: no branch - probe only
    try:
        from rl_engine.kernels.ops.base import _C as _C_probe
        from rl_engine.kernels.ops.base import _EXT_AVAILABLE as _EXT_probe

        _HAS_CUDA_GATED = _EXT_probe and hasattr(_C_probe, "rmsnorm_gated_forward")
    except ImportError:  # pragma: no cover
        _HAS_CUDA_GATED = False

requires_cuda_gated = pytest.mark.skipif(
    not _HAS_CUDA_GATED, reason="gated RMSNorm CUDA extension is not available"
)


def _gated_cuda_inputs(seed=0, rows=512, hidden=_HEAD_V_DIM, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype, generator=g)
    gate = torch.randn(rows, hidden, device="cuda", dtype=dtype, generator=g)
    weight = torch.randn(hidden, device="cuda", dtype=dtype, generator=g)
    return x, weight, gate


@requires_cuda_gated
def test_cuda_gated_matches_golden_within_contract():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    x, w, gate = _gated_cuda_inputs()
    got = Qwen3NextRMSNormGatedCudaOp().forward(x, w, gate, eps=_EPS)
    ref = Qwen3NextRMSNormGatedOp().forward_fp32(x, w, gate, eps=_EPS)
    torch.testing.assert_close(got.float(), ref, **_forward_tol(torch.bfloat16))


@requires_cuda_gated
@pytest.mark.parametrize("offset", [0.0, 1.0])
@pytest.mark.parametrize("activation", [0, 1], ids=["silu", "sigmoid"])
@pytest.mark.parametrize("hidden", [_HEAD_V_DIM, 2048, 5120])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"]
)
def test_cuda_gated_rstd_is_bitwise_identical_to_plain_kernel(dtype, hidden, activation, offset):
    """The gate must not perturb the normalization statistic.

    Same x, same rstd, bitwise -- otherwise the gate has leaked into the
    reduction and the two kernels no longer share a contract. Both kernels
    launch with ``choose_threads(H)``: H=128 runs 128 threads with one column
    each, 2048 and 5120 run 512 threads with 4 and 10 serial columns each.
    Each activation is its own template instantiation, and the offset is
    applied only after the statistic, so neither it nor the gate may move it.
    """
    from rl_engine.kernels.ops.base import _C

    x, w, gate = _gated_cuda_inputs(hidden=hidden, dtype=dtype)
    _, rstd_gated = _C.rmsnorm_gated_forward(x, w, gate, _EPS, offset, activation)
    _, rstd_plain = _C.rmsnorm_forward(x, w, _EPS, offset)
    assert torch.equal(rstd_gated, rstd_plain)


@requires_cuda_gated
@pytest.mark.parametrize("batch", [1, 2, 8, 16, 32, 48, 64, 512])
def test_cuda_gated_batch_invariance(batch):
    """L1 across the RFC #428 concurrency axis, bitwise."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    op = Qwen3NextRMSNormGatedCudaOp()
    x, w, gate = _gated_cuda_inputs()
    full = op.forward(x, w, gate, eps=_EPS)
    assert torch.equal(op.forward(x[:batch], w, gate[:batch], eps=_EPS), full[:batch])


@requires_cuda_gated
def test_cuda_gated_zero_gate_zeroes_output():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    x, w, _ = _gated_cuda_inputs()
    out = Qwen3NextRMSNormGatedCudaOp().forward(x, w, torch.zeros_like(x), eps=_EPS)
    assert torch.equal(out, torch.zeros_like(out))


@requires_cuda_gated
def test_cuda_gated_unit_weight_is_plain_norm_times_silu():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp, rmsnorm_cuda

    x, _, gate = _gated_cuda_inputs()
    ones = torch.ones(_HEAD_V_DIM, device="cuda", dtype=x.dtype)
    got = Qwen3NextRMSNormGatedCudaOp().forward(x, ones, gate, eps=_EPS).float()
    ref = rmsnorm_cuda(x, ones, eps=_EPS).float() * F.silu(gate.float())
    torch.testing.assert_close(got, ref, **_forward_tol(torch.bfloat16))


@requires_cuda_gated
def test_cuda_gated_sigmoid_activation():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda, rmsnorm_gated_cuda

    x, w, gate = _gated_cuda_inputs()
    got = rmsnorm_gated_cuda(x, w, gate, eps=_EPS, activation="sigmoid").float()
    ref = rmsnorm_cuda(x, w, eps=_EPS).float() * torch.sigmoid(gate.float())
    torch.testing.assert_close(got, ref, **_forward_tol(torch.bfloat16))


@requires_cuda_gated
def test_cuda_gated_backward_matches_autograd_golden():
    """dx, dweight and dgate against the fp32 reference's autograd."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x, w, gate = _gated_cuda_inputs()
    x, w, gate = x.float(), w.float(), gate.float()
    dy = torch.randn_like(x)

    got = [t.clone().requires_grad_(True) for t in (x, w, gate)]
    rmsnorm_gated_cuda(*got, eps=_EPS).backward(dy)
    ref = [t.clone().requires_grad_(True) for t in (x, w, gate)]
    Qwen3NextRMSNormGatedOp().forward_fp32(*ref, eps=_EPS).backward(dy)

    for name, a, b in zip(("dx", "dweight", "dgate"), got, ref):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-4, rtol=1e-4, msg=name)


@requires_cuda_gated
@pytest.mark.parametrize("batch", [1, 8, 64, 512])
def test_cuda_gated_dweight_is_batch_invariant(batch):
    """A row's contribution to dweight must not depend on the batch it sits in.

    The ascending-row fp32 left fold makes dweight over the first n rows exactly
    the fold of those n row-contributions -- so the n-row run must reproduce the
    512-row run's prefix, bitwise.
    """
    from rl_engine.kernels.ops.base import _C
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda
    from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32

    x, w, gate = _gated_cuda_inputs()
    x, w, gate = x.float(), w.float(), gate.float()
    dy = torch.randn_like(x)

    wg = w.clone().requires_grad_(True)
    rmsnorm_gated_cuda(x[:batch], wg, gate[:batch], eps=_EPS).backward(dy[:batch])

    # rstd from the full-batch forward: the row statistic is batch-invariant, so
    # its prefix is what the n-row run must have seen. Recomputing it with
    # mean(-1) instead would compare against a different reduction.
    _, rstd_full = _C.rmsnorm_gated_forward(x, w, gate, _EPS, 0.0, 0)
    rows = dy * F.silu(gate) * x * rstd_full.unsqueeze(-1)
    assert torch.equal(wg.grad, reduce_rows_fp32(rows[:batch]))


@requires_cuda_gated
def test_cuda_gated_weight_offset_is_applied_in_fp32():
    """The gated kernel carries the same offset contract as the plain one."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x, w, gate = _gated_cuda_inputs()
    offset = rmsnorm_gated_cuda(x, w, gate, eps=_EPS, weight_offset=1.0)
    explicit = rmsnorm_gated_cuda(x, (1.0 + w.float()), gate, eps=_EPS)
    assert torch.equal(offset, explicit)
    folded = rmsnorm_gated_cuda(x, (1.0 + w.float()).bfloat16(), gate, eps=_EPS)
    assert not torch.equal(offset, folded)


@requires_cuda_gated
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_cuda_gated_parameter_vjp_contributions_match_the_fold(dtype, activation):
    """Compare the harness hook with actual autograd, including its CUDA rstd."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import (
        Qwen3NextRMSNormGatedCudaOp,
        _fold_dweight_rows,
    )

    x, w, gate = (t.to(dtype) for t in _gated_cuda_inputs(rows=512))
    w = w.float().requires_grad_()
    dy = torch.randn_like(x)
    op = Qwen3NextRMSNormGatedCudaOp(activation=activation)
    op.forward(x, w, gate, eps=_EPS).backward(dy)
    rows = op.parameter_vjp_contributions_fp32(x=x, weight=w, gate=gate, grad_output=dy, eps=_EPS)[
        "weight"
    ]
    assert torch.equal(_fold_dweight_rows(rows, torch.float32), w.grad)


@requires_cuda_gated
def test_cuda_gated_wrapper_rejects_equal_numel_wrong_shape():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    x, w, gate = _gated_cuda_inputs(rows=6)
    x = x.reshape(2, 3, -1)
    with pytest.raises(ValueError, match="gate must match x"):
        Qwen3NextRMSNormGatedCudaOp().forward(x, w, gate)


@requires_cuda_gated
def test_cuda_gated_sigmoid_backward_uses_the_sigmoid_derivative():
    """The activation gradient has two branches; only silu was covered."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x, w, gate = _gated_cuda_inputs(rows=64)
    x, w, gate = x.float(), w.float(), gate.float()
    dy = torch.randn_like(x)

    got = [t.clone().requires_grad_(True) for t in (x, w, gate)]
    rmsnorm_gated_cuda(*got, eps=_EPS, activation="sigmoid").backward(dy)

    ref = [t.clone().requires_grad_(True) for t in (x, w, gate)]
    rstd = torch.rsqrt(ref[0].square().mean(dim=-1) + _EPS)
    out = (ref[0] * rstd.unsqueeze(-1) * ref[1]) * torch.sigmoid(ref[2])
    out.backward(dy)
    for name, a, b in zip(("dx", "dweight", "dgate"), got, ref):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-4, rtol=1e-4, msg=name)


@requires_cuda_gated
def test_cuda_gated_rejects_bad_activation():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x, w, gate = _gated_cuda_inputs(rows=8)
    with pytest.raises(ValueError, match="activation must be one of"):
        rmsnorm_gated_cuda(x, w, gate, eps=_EPS, activation="gelu")


def test_gated_cuda_op_rejects_bad_activation_at_construction():
    """The activation is validated before the extension check, so this runs on CPU."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    with pytest.raises(ValueError, match="activation must be one of"):
        Qwen3NextRMSNormGatedCudaOp(activation="gelu")


@requires_cuda_gated
def test_cuda_gated_swish_is_an_alias_for_silu():
    """vLLM maps output_gate_type "swish" to "silu"; the alias must not select sigmoid."""
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    x, w, gate = _gated_cuda_inputs(rows=64)
    swish = Qwen3NextRMSNormGatedCudaOp(activation="swish").forward(x, w, gate, eps=_EPS)
    silu = Qwen3NextRMSNormGatedCudaOp(activation="silu").forward(x, w, gate, eps=_EPS)
    sigmoid = Qwen3NextRMSNormGatedCudaOp(activation="sigmoid").forward(x, w, gate, eps=_EPS)
    assert torch.equal(swish, silu)
    assert not torch.equal(swish, sigmoid)


@requires_cuda_gated
def test_cuda_gated_rejects_mismatched_gate():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x, w, gate = _gated_cuda_inputs(rows=8)
    with pytest.raises((AssertionError, RuntimeError)):
        rmsnorm_gated_cuda(x, w, gate[:4], eps=_EPS)
    with pytest.raises((AssertionError, RuntimeError)):
        rmsnorm_gated_cuda(x, w, gate.float(), eps=_EPS)


# --------------------------------------------------------------------------- #
# 8. `__call__` is the documented entry point and must agree with `forward`
# --------------------------------------------------------------------------- #
def test_call_matches_forward():
    x, w = _rand((4, _HIDDEN), seed=40), _rand((_HIDDEN,), seed=41)
    op = Qwen3NextRMSNormOp()
    assert torch.equal(op(x, w, eps=_EPS), op.forward(x, w, eps=_EPS))


@pytest.mark.parametrize("cls", [Qwen3NextRMSNormGatedOp, Qwen3NextRMSNormGatedHFOp])
def test_gated_call_matches_forward(cls):
    x = _rand((4, _HEAD_V_DIM), seed=42)
    w = _rand((_HEAD_V_DIM,), seed=43)
    gate = _rand((4, _HEAD_V_DIM), seed=44)
    op = cls()
    assert torch.equal(op(x, w, gate, eps=_EPS), op.forward(x, w, gate, eps=_EPS))


def test_zero_centred_op_inherits_the_plain_reference():
    """The only difference from the plain op is the weight convention.

    Pins the inheritance: if the subclass ever grows its own `_rms_norm`, this
    stops being true and the two references can drift apart silently.
    """
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp

    assert issubclass(Qwen3NextRMSNormOp, NativeRMSNormOp)
    assert Qwen3NextRMSNormOp.weight_offset == 1.0
    assert NativeRMSNormOp.weight_offset == 0.0
    assert Qwen3NextRMSNormOp._rms_norm.__func__ is NativeRMSNormOp._rms_norm.__func__


def test_plain_reference_is_unchanged_by_the_offset_plumbing():
    """Adding `weight_offset` to the base must not perturb the plain path.

    `0.0 + w` rewrites -0.0 to +0.0, which torch.equal does not notice, so this
    compares the raw bits.
    """
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp

    x = _rand((2, _HEAD_V_DIM), seed=45)
    w = torch.zeros(_HEAD_V_DIM)
    w[0] = -0.0
    got = NativeRMSNormOp().forward_fp32(x, w, eps=_EPS)
    x_f = x.float()
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import shape_invariant_rstd

    expected = (x_f * shape_invariant_rstd(x_f, _EPS).unsqueeze(-1)) * w.float()
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32))


@requires_cuda_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_plain_signed_zero_is_preserved(dtype):
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

    x = torch.ones(2, 128, device="cuda", dtype=dtype)
    x[1].neg_()
    weight = torch.full((128,), -0.0, device="cuda", dtype=dtype)
    actual = rmsnorm_cuda(x, weight)
    expected = x * weight
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(bits), expected.view(bits))


@requires_cuda_rmsnorm
@pytest.mark.parametrize("offset", [0.0, 1.0])
@pytest.mark.parametrize("shape", [(512, 128), (2, 3, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_parameter_contributions_reproduce_backward(offset, shape, dtype):
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp, RMSNormCudaOp
    from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32

    torch.manual_seed(128)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn_like(x)
    op = RMSNormCudaOp() if offset == 0.0 else Qwen3NextRMSNormCudaOp()
    op(x, weight).backward(upstream)
    rows = op.parameter_vjp_contributions_fp32(x=x, weight=weight, grad_output=upstream)["weight"]
    folded = reduce_rows_fp32(rows.reshape(-1, shape[-1])).to(dtype)
    assert torch.equal(weight.grad, folded)


def test_zero_centred_cuda_constructor_rejects_missing_extension(monkeypatch):
    from rl_engine.kernels.ops.cuda.norm import rmsnorm

    monkeypatch.setattr(rmsnorm, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(rmsnorm, "_C", None)
    with pytest.raises(RuntimeError, match="requires the compiled"):
        rmsnorm.Qwen3NextRMSNormCudaOp()


@requires_cuda_gated
@pytest.mark.parametrize(
    "fault", ["weight_shape", "weight_dtype", "gate_dtype", "dy_dtype", "rstd_dtype"]
)
def test_gated_extension_rejects_invalid_tensor_contract(fault):
    from rl_engine import _C

    x, w, gate = _gated_cuda_inputs(rows=8)
    dy = torch.ones_like(x)
    rstd = torch.ones(8, device=x.device, dtype=torch.float32)
    if fault == "weight_shape":
        w = w[:-1]
    elif fault == "weight_dtype":
        w = w.half()
    elif fault == "gate_dtype":
        gate = gate.float()
    elif fault == "dy_dtype":
        dy = dy.float()
    else:
        rstd = rstd.bfloat16()
    with pytest.raises(RuntimeError):
        _C.rmsnorm_gated_backward_dx(dy, x, w, gate, rstd)


@requires_cuda_gated
def test_gated_extension_handles_empty_batch_and_nondefault_stream():
    from rl_engine import _C

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, w, gate = _gated_cuda_inputs(rows=8)
        actual, rstd = _C.rmsnorm_gated_forward(x, w, gate, _EPS)
        dx = _C.rmsnorm_gated_backward_dx(torch.ones_like(x), x, w, gate, rstd)
        empty, empty_rstd = _C.rmsnorm_gated_forward(x[:0], w, gate[:0], _EPS)
        empty_dx = _C.rmsnorm_gated_backward_dx(x[:0], x[:0], w, gate[:0], empty_rstd)
    stream.synchronize()
    expected, expected_rstd = _C.rmsnorm_gated_forward(x, w, gate, _EPS)
    expected_dx = _C.rmsnorm_gated_backward_dx(torch.ones_like(x), x, w, gate, expected_rstd)
    assert torch.equal(actual, expected)
    assert torch.equal(dx, expected_dx)
    assert empty.shape == empty_dx.shape == (0, x.shape[-1])
    assert empty_rstd.numel() == 0


@requires_cuda_gated
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_gated_signed_zero_weight_is_preserved(dtype):
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x = torch.ones(2, 128, device="cuda", dtype=dtype)
    x[1].neg_()
    weight = torch.full((128,), -0.0, device="cuda", dtype=dtype)
    actual = rmsnorm_gated_cuda(x, weight, torch.ones_like(x))
    expected = x * weight
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(bits), expected.view(bits))


@requires_cuda_gated
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_gated_dgate_keeps_signed_zero_weight(dtype):
    """dgate scales by the weight too, so it must not turn a -0.0 weight into +0.0.

    dgate = dy * x * rstd * w * act'(gate); with dy, rstd, act'(1) > 0 its sign is
    the sign of ``x * w``, zeros included.
    """
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_gated_cuda

    x = torch.ones(2, 128, device="cuda", dtype=dtype)
    x[1].neg_()
    weight = torch.full((128,), -0.0, device="cuda", dtype=dtype)
    gate = torch.ones_like(x).requires_grad_()
    rmsnorm_gated_cuda(x, weight, gate).backward(torch.ones_like(x))
    expected = x * weight
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(gate.grad.view(bits), expected.view(bits))
