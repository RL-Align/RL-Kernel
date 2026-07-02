# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for NativeAttentionOp (ISSUE #108 WS1 ground-truth baseline).

Standard-softmax attention: out = softmax(Q Kᵀ * scale + masks) @ V, hand-written
(NOT F.scaled_dot_product_attention) so the reduction order is fixed. Like
lm_head this is a *reduction* (over the key dim Skv), so:

  * Axis-B (accuracy): the low-precision ``forward`` path accumulates in the
    input dtype and drifts from the fp32 ``forward_fp32`` ground truth. It is
    checked with a tolerance relative to the output peak magnitude, not bitwise
    -- attention outputs are convex combinations of V rows, so many entries sit
    near zero while the accumulated error tracks the reduction length.
  * Axis-A (batch invariance): bitwise within a single dtype, but only once the
    CPU reduction order is pinned. Multi-threaded CPU GEMM splits the matmul
    reduction by the batch dimension, which silently breaks bitwise batch
    invariance. ``_single_thread`` fixes the reduction order; it is the local
    stand-in for the planned testing/determinism.py::deterministic_context.

This op covers ONLY the softmax attention; QK-Norm and RoPE are applied before
the call (see the chain test) -- the q,k here are plain synthetic tensors.
"""

import contextlib
import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp
from rl_engine.kernels.registry import kernel_registry

# Qwen3-8B attention dims (synthetic tensors, no checkpoint). Unlike embedding /
# lm_head (whose multi-GB weight forces shrinking), attention's cost is the
# scores tensor [B, Hq, Sq, Skv], so the *real* head dims are cheap at a SMALL
# (batch, seq) load point and kept real here. Only LARGE (8, 4096) is GPU-only.
_N_HEADS = 32  # Q heads
_N_KV = 8  # KV heads; GQA group g = 32 / 8 = 4
_HEAD_DIM = 128  # 32 * 128 == 4096 == hidden

# Axis-B: max abs error as a fraction of the output peak magnitude. Calibrated
# from measured SMALL drift (bf16 ~1% of peak, fp16 ~0.1%) with headroom.
_DTYPE_REL_PEAK = {torch.bfloat16: 3.0e-2, torch.float16: 5.0e-3}


def _cpu_fp16_matmul_supported() -> bool:
    """Probe whether this CPU backend implements float16 matmul."""
    try:
        _ = torch.randn(2, 2, dtype=torch.float16) @ torch.randn(2, 2, dtype=torch.float16)
        return True
    except RuntimeError:
        return False


# CPU half-precision matmul is backend/ISA-dependent and may be unimplemented on
# some runners -- gate the fp16 axis so a missing kernel skips rather than fails.
_FP16_IF_CPU_MATMUL_SUPPORTED = pytest.param(
    torch.float16,
    marks=pytest.mark.skipif(
        not _cpu_fp16_matmul_supported(),
        reason="CPU float16 matmul unsupported on this backend",
    ),
)
_DTYPES_AXIS_B = (torch.bfloat16, _FP16_IF_CPU_MATMUL_SUPPORTED)
_DTYPES_AXIS_A = (torch.float32, torch.bfloat16, _FP16_IF_CPU_MATMUL_SUPPORTED)


@contextlib.contextmanager
def _single_thread():
    """Pin CPU GEMM to one thread so the matmul reduction order is batch-independent."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# Shared helpers -- fixed-seed Generator for determinism / reproducibility.
def _qkv(batch, sq, skv, *, seed, dtype=torch.float32, n_heads=_N_HEADS, n_kv=_N_KV, d=_HEAD_DIM):
    """Fixed-seed random q [B,Hq,Sq,D], k/v [B,Hkv,Skv,D] for reproducibility."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, n_heads, sq, d, generator=gen, dtype=dtype)
    k = torch.randn(batch, n_kv, skv, d, generator=gen, dtype=dtype)
    v = torch.randn(batch, n_kv, skv, d, generator=gen, dtype=dtype)
    return q, k, v


def _ref_softmax_attn(q, k, v, *, causal, scale=None, key_padding_mask=None):
    """Independent naive-softmax reference mirroring the contract (GQA, masks).

    Dtype-preserving: pass fp32 tensors for a bitwise fp32 check, or .double()
    tensors for a TF32-immune high-precision reference.
    """
    qf, kf, vf = q, k, v
    Hq, Sq, D = qf.shape[1], qf.shape[2], qf.shape[3]
    Hkv, Skv = kf.shape[1], kf.shape[2]
    if Hkv != Hq:
        r = Hq // Hkv
        kf = kf.repeat_interleave(r, dim=1)
        vf = vf.repeat_interleave(r, dim=1)
    s = torch.matmul(qf, kf.transpose(-1, -2)) * (
        scale if scale is not None else 1.0 / math.sqrt(D)
    )
    if causal:
        m = torch.triu(
            torch.ones(Sq, Skv, dtype=torch.bool, device=qf.device), diagonal=Skv - Sq + 1
        )
        s = s.masked_fill(m, float("-inf"))
    if key_padding_mask is not None:
        s = s.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
    return torch.softmax(s, dim=-1) @ vf


# --------------------------------------------------------------------------- #
# fp32 ground-truth correctness
# --------------------------------------------------------------------------- #
# forward_fp32 == the independent naive fp32 reference, bitwise. Both fix the
# same reduction order, so this validates the op's wiring (transpose dims, scale,
# masks) exactly. TF32 is pinned off so the fp32 forward path matches too.
def test_forward_fp32_matches_independent_reference():
    """forward_fp32 (and the fp32 forward path, TF32 off) is bitwise-equal to a
    naive fp32 reference."""
    q, k, v = _qkv(2, 16, 16, seed=1)  # Qwen3 32/8/128, SMALL prefill
    ref = _ref_softmax_attn(q, k, v, causal=True)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        assert torch.equal(NativeAttentionOp().forward_fp32(q, k, v, causal=True), ref)
        assert torch.equal(NativeAttentionOp().forward(q, k, v, causal=True), ref)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def test_forward_fp32_ignores_ambient_autocast_and_restores_tf32():
    """forward_fp32 is a strict fp32 reference under ambient autocast/TF32 settings."""
    op = NativeAttentionOp()
    q, k, v = _qkv(2, 8, 8, seed=11)
    ref = _ref_softmax_attn(q, k, v, causal=True)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = op.forward_fp32(q, k, v, causal=True)
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)
        assert torch.backends.cuda.matmul.allow_tf32 is True  # restored
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU to exercise TF32")
def test_forward_fp32_disables_tf32_on_gpu():
    """On a TF32-enabled GPU, forward_fp32 stays true fp32, no worse than a TF32 path."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(21)
    q = torch.randn(2, _N_HEADS, 64, _HEAD_DIM, generator=gen, device=device)
    k = torch.randn(2, _N_KV, 64, _HEAD_DIM, generator=gen, device=device)
    v = torch.randn(2, _N_KV, 64, _HEAD_DIM, generator=gen, device=device)
    ref = _ref_softmax_attn(q.double(), k.double(), v.double(), causal=True).float()

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # hostile ambient setting
    try:
        strict = NativeAttentionOp().forward_fp32(q, k, v, causal=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    peak = ref.abs().max().item()
    strict_err = (strict - ref).abs().max().item()
    print(f"\n[attention fp32-vs-tf32] strict_err={strict_err:.3g} peak={peak:.3g}")
    assert strict_err <= 1.0e-3 * peak  # fp32-tight, well under TF32 drift floor


# --------------------------------------------------------------------------- #
# Attention-specific correctness (closed-form, independent of the op's code)
# --------------------------------------------------------------------------- #
# With q=k=0 every score is 0, so softmax is uniform over the *visible* keys.
# Under causal masking query i sees keys 0..i, so out[i] == mean(v[0..i]).
def test_causal_uniform_attention_closed_form():
    """Causal masking: with uniform scores, out[i] == mean of keys 0..i."""
    B, H, S, D = 1, 1, 4, 3
    q = torch.zeros(B, H, S, D)
    k = torch.zeros(B, H, S, D)
    v = torch.arange(S * D, dtype=torch.float32).reshape(1, 1, S, D)  # distinct rows
    out = NativeAttentionOp().forward_fp32(q, k, v, causal=True)
    expected = torch.stack([v[0, 0, : i + 1].mean(dim=0) for i in range(S)])  # cumulative mean
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


# Decode special case: a single query (Sq=1) with causal offset Skv-1+1=Skv masks
# nothing -> it must see all keys, i.e. equal the non-causal result.
def test_causal_decode_sees_all_keys():
    """Decode (Sq=1): causal masks nothing, so it equals the non-causal result."""
    op = NativeAttentionOp()
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(2, _N_HEADS, 1, _HEAD_DIM, generator=gen)
    k = torch.randn(2, _N_KV, 40, _HEAD_DIM, generator=gen)
    v = torch.randn(2, _N_KV, 40, _HEAD_DIM, generator=gen)
    assert torch.equal(
        op.forward_fp32(q, k, v, causal=True), op.forward_fp32(q, k, v, causal=False)
    )


# GQA: 32 Q heads share 8 KV heads (g=4). Output keeps 32 heads, and the result
# matches an independent reference that expands KV with repeat_interleave.
def test_gqa_replication():
    """GQA: output keeps Hq=32 heads and matches the repeat_interleave reference."""
    q, k, v = _qkv(2, 8, 8, seed=3)
    out = NativeAttentionOp().forward_fp32(q, k, v, causal=False)
    assert out.shape == (2, _N_HEADS, 8, _HEAD_DIM)
    assert out.shape[1] == 4 * k.shape[1]  # 32 == g * 8
    assert torch.equal(out, _ref_softmax_attn(q, k, v, causal=False))


def test_gqa_requires_divisible_heads():
    """Hq not divisible by Hkv is rejected (no valid GQA grouping)."""
    gen = torch.Generator().manual_seed(31)
    q = torch.randn(1, 6, 4, _HEAD_DIM, generator=gen)  # 6 not divisible by 4
    k = torch.randn(1, 4, 4, _HEAD_DIM, generator=gen)
    v = torch.randn(1, 4, 4, _HEAD_DIM, generator=gen)
    with pytest.raises(ValueError, match="not divisible"):
        NativeAttentionOp().forward_fp32(q, k, v, causal=False)


# scale defaults to 1/sqrt(head_dim); an explicit scale (incl. 0.0) is honored.
def test_scale_default_and_explicit():
    """Default scale is 1/sqrt(D); an explicit scale (incl. 0.0) is used verbatim."""
    op = NativeAttentionOp()
    q, k, v = _qkv(2, 8, 8, seed=4)
    assert torch.equal(
        op.forward_fp32(q, k, v, causal=False),
        op.forward_fp32(q, k, v, causal=False, scale=1.0 / math.sqrt(_HEAD_DIM)),
    )
    # scale=0.0 -> all scores 0 -> uniform attention over all keys (mean of V).
    out0 = op.forward_fp32(q, k, v, causal=False, scale=0.0)
    assert torch.allclose(
        out0,
        v.float().repeat_interleave(4, dim=1).mean(dim=2, keepdim=True).expand_as(out0),
        atol=1e-6,
    )


# key_padding_mask (True=valid): padded key columns get zero weight, so the
# result equals attention computed over only the valid keys.
#
# NOTE: padding changes the softmax reduction width (Skv=10 vs Skv=6).  Even
# though masked positions contribute exp(-inf)=0 to the sum, the internal
# reduction order of torch.softmax over a size-10 row vs a size-6 row may
# differ (vectorisation boundaries, intermediate rounding of partial sums),
# so bitwise equality across different Skv is NOT guaranteed in IEEE 754.
# We assert near-equality (atol=2e-6) which validates the masking semantics
# without over-constraining the floating-point reduction path.  The observed
# drift is ~1.3e-6 and is platform-sensitive, so the threshold carries headroom.
_PADDING_ATOL = 2.0e-6


def test_key_padding_mask_excludes_padded_keys():
    """key_padding_mask: padded keys get zero weight (≈ attending over valid keys only).

    Not bitwise-equal because the softmax reduction width differs (Skv=10 vs 6);
    see comment above for rationale.
    """
    op = NativeAttentionOp()
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(2, _N_HEADS, 6, _HEAD_DIM, generator=gen)
    k_valid = torch.randn(2, _N_KV, 6, _HEAD_DIM, generator=gen)
    v_valid = torch.randn(2, _N_KV, 6, _HEAD_DIM, generator=gen)
    pad_k = torch.randn(2, _N_KV, 4, _HEAD_DIM, generator=gen)  # 4 padding columns
    pad_v = torch.randn(2, _N_KV, 4, _HEAD_DIM, generator=gen)
    k = torch.cat([k_valid, pad_k], dim=2)
    v = torch.cat([v_valid, pad_v], dim=2)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :6] = True  # first 6 valid, last 4 padding

    masked = op.forward_fp32(q, k, v, causal=False, key_padding_mask=mask)
    valid_only = op.forward_fp32(q, k_valid, v_valid, causal=False)

    diff = (masked - valid_only).abs()
    max_err = diff.max().item()
    print(f"\n[padding mask] max_abs_err={max_err:.3g} (threshold={_PADDING_ATOL:.1g})")
    assert torch.allclose(
        masked, valid_only, atol=_PADDING_ATOL, rtol=0.0
    ), f"Padding-masked result diverges from valid-only by {max_err:.3g} > {_PADDING_ATOL}"


# A query whose every key is padded out has an all -inf row; naive softmax would
# emit NaN. The op defines such fully-masked rows as 0, keeping outputs and grads
# finite (NaN would poison both and break alignment against downstream kernels).
def test_fully_masked_query_returns_zero_not_nan():
    """All keys padded out -> the query yields 0 (not NaN), and grads stay finite."""
    gen = torch.Generator().manual_seed(9)
    q = torch.randn(1, _N_HEADS, 4, _HEAD_DIM, generator=gen, requires_grad=True)
    k = torch.randn(1, _N_KV, 4, _HEAD_DIM, generator=gen)
    v = torch.randn(1, _N_KV, 4, _HEAD_DIM, generator=gen)
    mask = torch.zeros(1, 4, dtype=torch.bool)  # all False == every key is padding

    out = NativeAttentionOp().forward_fp32(q, k, v, causal=False, key_padding_mask=mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))

    # NaN would propagate through backward; assert the gradient is finite (zero here).
    out.sum().backward()
    assert torch.isfinite(q.grad).all()


# --------------------------------------------------------------------------- #
# Axis-B accuracy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", _DTYPES_AXIS_B)
def test_dtype_path_accuracy(dtype: torch.dtype):
    """Axis-B: the low-precision path drifts from fp32 by a bounded fraction of the output peak."""
    op = NativeAttentionOp()
    q, k, v = _qkv(2, 16, 16, seed=2)
    ref = op.forward_fp32(q, k, v, causal=True)  # fp32 ground truth
    cand = op.forward(q.to(dtype), k.to(dtype), v.to(dtype), causal=True)
    assert cand.dtype == dtype

    err = (cand.float() - ref).abs()
    peak = ref.abs().max()
    max_abs, mean_abs = err.max().item(), err.mean().item()
    print(f"\n[attention {dtype}] max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} peak={peak:.4g}")
    assert max_abs <= _DTYPE_REL_PEAK[dtype] * peak.item()


def test_output_shape():
    """Output shape is [B, Hq, Sq, D]."""
    q, k, v = _qkv(3, 7, 7, seed=3)
    out = NativeAttentionOp().forward(q, k, v, causal=True)
    assert out.shape == (3, _N_HEADS, 7, _HEAD_DIM)


# --------------------------------------------------------------------------- #
# Axis-A batch invariance (bitwise, single-thread reduction order)
# --------------------------------------------------------------------------- #
# A sequence's attention output must not depend on how many other sequences
# share the batch. Compute on the full batch once, then slice -- never compute a
# slice on its own. Requires the pinned single-thread reduction order.
@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
def test_batch_invariance_slice(dtype: torch.dtype):
    """Axis-A: a sequence's output is bitwise-independent of how many share the batch."""
    op = NativeAttentionOp()
    q, k, v = _qkv(8, 16, 16, seed=5, dtype=dtype)
    with _single_thread():
        full = op.forward(q, k, v, causal=True)
        assert torch.equal(op.forward(q[:1], k[:1], v[:1], causal=True), full[:1])
        assert torch.equal(op.forward(q[3:5], k[3:5], v[3:5], causal=True), full[3:5])


@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
def test_batch_invariance_chunked(dtype: torch.dtype):
    """Axis-A (chunked): processing the batch in chunks and concatenating == one shot."""
    op = NativeAttentionOp()
    q, k, v = _qkv(8, 16, 16, seed=6, dtype=dtype)
    with _single_thread():
        full = op.forward(q, k, v, causal=True)
        c1 = op.forward(q[:3], k[:3], v[:3], causal=True)
        c2 = op.forward(q[3:], k[3:], v[3:], causal=True)
        assert torch.equal(torch.cat([c1, c2], dim=0), full)


@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
def test_backward_batch_invariance_slice(dtype: torch.dtype):
    """Axis-A (backward): q/k/v grads are bitwise-independent of batch size."""
    op = NativeAttentionOp()
    q_full, k_full, v_full = _qkv(8, 16, 16, seed=10, dtype=dtype)

    q_full.requires_grad_(True)
    k_full.requires_grad_(True)
    v_full.requires_grad_(True)

    gen = torch.Generator().manual_seed(10)
    dy = torch.randn(8, _N_HEADS, 16, _HEAD_DIM, generator=gen, dtype=dtype)

    with _single_thread():
        out_full = op.forward(q_full, k_full, v_full, causal=True)
        out_full.backward(dy)

    q_slice = q_full[:1].detach().clone().requires_grad_(True)
    k_slice = k_full[:1].detach().clone().requires_grad_(True)
    v_slice = v_full[:1].detach().clone().requires_grad_(True)

    with _single_thread():
        out_slice = op.forward(q_slice, k_slice, v_slice, causal=True)
        out_slice.backward(dy[:1])

    assert torch.equal(q_slice.grad, q_full.grad[:1])
    assert torch.equal(k_slice.grad, k_full.grad[:1])
    assert torch.equal(v_slice.grad, v_full.grad[:1])


# --------------------------------------------------------------------------- #
# Purity / gradient / registry
# --------------------------------------------------------------------------- #
def test_inputs_not_mutated():
    """Purity: no input tensor is mutated in place."""
    op = NativeAttentionOp()
    q, k, v = _qkv(2, 8, 8, seed=7)
    qc, kc, vc = q.clone(), k.clone(), v.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)
    mc = mask.clone()
    op.forward(q, k, v, causal=True, key_padding_mask=mask)
    op.forward_fp32(q, k, v, causal=True, key_padding_mask=mask)
    assert torch.equal(q, qc) and torch.equal(k, kc) and torch.equal(v, vc)
    assert torch.equal(mask, mc)


def test_gradient_matches_reference():
    """fp32 autograd grads match autograd through the double-precision reference.

    isfinite only rules out NaN/Inf -- it can't tell a correct gradient from a
    wrong-but-finite one, and attention's backward (softmax Jacobian + dQ/dK/dV
    contractions) is the most error-prone in the stack. Backprop a *random*
    cotangent (not .sum(), whose all-ones cotangent collapses the contraction)
    and compare q/k/v grads against autograd through the independent
    _ref_softmax_attn computed in float64 (TF32-immune high-precision golden).
    """
    op = NativeAttentionOp()
    q, k, v = _qkv(2, 8, 8, seed=8)
    q, k, v = q.requires_grad_(True), k.requires_grad_(True), v.requires_grad_(True)

    out = op.forward_fp32(q, k, v, causal=True)
    gen = torch.Generator().manual_seed(8)
    dy = torch.randn(out.shape, generator=gen, dtype=out.dtype)  # seeded for reproducibility
    out.backward(dy)

    qd, kd, vd = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    _ref_softmax_attn(qd, kd, vd, causal=True).backward(dy.double())
    for t, td in ((q, qd), (k, kd), (v, vd)):
        assert t.grad is not None and t.grad.shape == t.shape
        torch.testing.assert_close(t.grad, td.grad.float(), rtol=1e-4, atol=1e-4)


def test_registry_dispatches_native_attention_op():
    """The registry resolves "attention" to the ground-truth NativeAttentionOp."""
    assert isinstance(kernel_registry.get_op("attention"), NativeAttentionOp)


# --------------------------------------------------------------------------- #
# Qwen3-8B LARGE real-scale GPU smoke test
# --------------------------------------------------------------------------- #
# The scores tensor [B=8, Hq=32, Skv=4096, Skv=4096] is ~17 GB in fp32, so the
# LARGE load point is GPU-only and skips without enough memory. SMALL/MEDIUM at
# real head dims already run on CPU above; this validates real prefill scale.
def _enough_gpu_memory(num_bytes: int) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        free, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        return False
    return free > num_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_attention_qwen3_8b_large_real_shape():
    """GPU smoke at LARGE Qwen3-8B prefill (batch=8, seq=4096, 32/8/128)."""
    # Runtime memory check (not a collection-time skipif): free memory at
    # collection time is not representative on a shared GPU. Naive fp32 attention
    # peaks at ~3x the scores tensor (scores + masked copy + softmax probs all
    # live transiently), so budget 3x the ~17 GB scores -> ~50 GB peak. This makes
    # LARGE an H100-class (H-series nightly) test, skipping on smaller GPUs.
    scores_bytes = 8 * _N_HEADS * 4096 * 4096 * 4  # ~17 GB
    if not _enough_gpu_memory(scores_bytes * 3):
        pytest.skip("not enough free GPU memory for the ~50 GB fp32 LARGE attention peak")
    device = torch.device("cuda")
    op = NativeAttentionOp()
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(8, _N_HEADS, 4096, _HEAD_DIM, generator=gen, dtype=torch.float32, device=device)
    k = torch.randn(8, _N_KV, 4096, _HEAD_DIM, generator=gen, dtype=torch.float32, device=device)
    v = torch.randn(8, _N_KV, 4096, _HEAD_DIM, generator=gen, dtype=torch.float32, device=device)
    out = op.forward_fp32(q, k, v, causal=True)
    assert out.shape == (8, _N_HEADS, 4096, _HEAD_DIM)
    assert torch.isfinite(out).all()
    # Axis-A: compute on full batch, then slice (no per-slice recompute).
    assert torch.equal(op.forward_fp32(q[:1], k[:1], v[:1], causal=True), out[:1])
