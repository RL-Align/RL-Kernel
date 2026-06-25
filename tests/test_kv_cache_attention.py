# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for NativeKVCacheAttnOp (ISSUE #108 WS1 ground-truth baseline).

KV-cache (decode/incremental) attention: concat the step's new K/V onto the
cache along the seq axis, then run the *same* standard-softmax attention used for
prefill. Because it delegates to NativeAttentionOp, the central guarantees are:

  * Delegation equivalence: kv_cache(q, cache, new) == standard_attn(q,
    cat([cache,new])) bitwise -- validates the wiring (cat dim, arg order,
    fp32/dtype path pairing).
  * Split-point invariance (Axis-A flavor): where you draw the cache/new
    boundary must not change the result. Same full K/V width -> same reduction
    -> bitwise identical (torch.equal).
  * Prefill<->decode consistency: stepwise decode reproduces full-prefill outputs
    up to a small tolerance. Here the softmax reduction *width* differs (step t
    reduces over t+1 keys vs the full Skv with future positions masked to -inf),
    so -- exactly as with key padding in standard attention -- IEEE 754 does not
    guarantee bitwise equality; we assert allclose(atol=1e-6).
  * Axis-B accuracy: the low-precision forward path drifts from forward_fp32 and
    is checked with a tolerance relative to the output peak.

This op covers ONLY the attention; QK-Norm and RoPE are applied before the call.
"""

import contextlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.kv_cache import NativeKVCacheAttnOp
from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp
from rl_engine.kernels.registry import kernel_registry

# Qwen3-8B attention dims (synthetic tensors, no checkpoint).
_N_HEADS = 32  # Q heads
_N_KV = 8  # KV heads; GQA group g = 32 / 8 = 4
_HEAD_DIM = 128  # 32 * 128 == 4096 == hidden

# Axis-B: max abs error as a fraction of the output peak magnitude (same basis as
# the standard-attention test, since this op shares its reduction).
_DTYPE_REL_PEAK = {torch.bfloat16: 3.0e-2, torch.float16: 5.0e-3}

# Prefill<->decode reduction width differs -> not bitwise; bounded near-equality.
_DECODE_ATOL = 1.0e-6

# key_padding_mask compares a softmax over (S_past+S_new) keys against one over the
# valid-only subset, so the reduction widths differ (same situation as the standard
# attention padding test).  The drift is ~1.3e-6 and platform-sensitive, so this
# cross-width comparison carries extra headroom over the closed-form decode checks.
_PADDING_ATOL = 2.0e-6


def _cpu_fp16_matmul_supported() -> bool:
    """Probe whether this CPU backend implements float16 matmul."""
    try:
        _ = torch.randn(2, 2, dtype=torch.float16) @ torch.randn(2, 2, dtype=torch.float16)
        return True
    except RuntimeError:
        return False


_FP16_IF_CPU_MATMUL_SUPPORTED = pytest.param(
    torch.float16,
    marks=pytest.mark.skipif(
        not _cpu_fp16_matmul_supported(),
        reason="CPU float16 matmul unsupported on this backend",
    ),
)
_DTYPES_AXIS_B = (torch.bfloat16, _FP16_IF_CPU_MATMUL_SUPPORTED)


@contextlib.contextmanager
def _single_thread():
    """Pin CPU GEMM to one thread so the matmul reduction order is stable."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _q(batch, sq, *, seed, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, _N_HEADS, sq, _HEAD_DIM, generator=gen, dtype=dtype)


def _kv(batch, s, *, seed, dtype=torch.float32):
    """K/V of KV-head count, length s."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, _N_KV, s, _HEAD_DIM, generator=gen, dtype=dtype)
    v = torch.randn(batch, _N_KV, s, _HEAD_DIM, generator=gen, dtype=dtype)
    return k, v


# --------------------------------------------------------------------------- #
# Delegation equivalence: kv_cache == standard_attn on the concatenation.
# --------------------------------------------------------------------------- #
def test_forward_fp32_equals_standard_attn_on_concat():
    """forward_fp32 must equal NativeAttentionOp.forward_fp32 on cat([cache,new])."""
    op = NativeKVCacheAttnOp()
    ref = NativeAttentionOp()
    b, s_past, s_new = 2, 6, 3
    q = _q(b, s_new, seed=1)  # Sq == S_new (the new tokens' queries)
    k_cache, v_cache = _kv(b, s_past, seed=2)
    k_new, v_new = _kv(b, s_new, seed=3)

    with _single_thread():
        got = op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True)
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)
        want = ref.forward_fp32(q, k_full, v_full, causal=True)
    assert torch.equal(got, want)
    assert got.shape == (b, _N_HEADS, s_new, _HEAD_DIM)


def test_forward_equals_standard_attn_on_concat():
    """forward (input-dtype path) must equal NativeAttentionOp.forward on cat([cache,new])."""
    op = NativeKVCacheAttnOp()
    ref = NativeAttentionOp()
    b, s_past, s_new = 2, 6, 3
    q = _q(b, s_new, seed=1)
    k_cache, v_cache = _kv(b, s_past, seed=2)
    k_new, v_new = _kv(b, s_new, seed=3)

    with _single_thread():
        got = op.forward(q, k_cache, v_cache, k_new, v_new, causal=True)
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)
        want = ref.forward(q, k_full, v_full, causal=True)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# Split-point invariance: the cache/new boundary must not change the result.
# --------------------------------------------------------------------------- #
def test_cache_split_point_invariance():
    """Splitting the same full K/V at different cache/new boundaries is bitwise identical.

    This is the soul of KV-cache correctness: a token computed during prefill (in
    the cache) vs at decode time (as new) yields the same attention output.
    """
    op = NativeKVCacheAttnOp()
    b, total = 2, 8
    sq = total  # attend the whole sequence (prefill-shaped) so every split is valid
    q = _q(b, sq, seed=10)
    k_full, v_full = _kv(b, total, seed=11)

    outputs = []
    with _single_thread():
        for split in (0, 1, 4, total):  # all-new, ..., all-cache
            k_cache, k_new = k_full[:, :, :split], k_full[:, :, split:]
            v_cache, v_new = v_full[:, :, :split], v_full[:, :, split:]
            outputs.append(op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True))
    for other in outputs[1:]:
        assert torch.equal(outputs[0], other)


# --------------------------------------------------------------------------- #
# Prefill <-> decode consistency (reduction width differs -> near-equal).
# --------------------------------------------------------------------------- #
def test_stepwise_decode_matches_full_prefill():
    """Token-by-token decode reproduces full-prefill outputs (atol=1e-6).

    Not bitwise: at step t the softmax reduces over t+1 keys, whereas prefill
    reduces over the full Skv with future positions masked to -inf -- a different
    reduction width, so IEEE 754 only guarantees near-equality (cf. key padding
    in standard attention).
    """
    op = NativeKVCacheAttnOp()
    ref = NativeAttentionOp()
    b, seq = 2, 7
    q_all = _q(b, seq, seed=20)
    k_all, v_all = _kv(b, seq, seed=21)

    with _single_thread():
        # Full prefill: one shot over the whole sequence, causal.
        prefill = ref.forward_fp32(q_all, k_all, v_all, causal=True)  # [B, Hq, seq, D]

        # Stepwise decode: at step t, cache = positions [0,t), new = position t.
        for t in range(seq):
            q_t = q_all[:, :, t : t + 1]  # the query for position t (Sq=1)
            k_cache, v_cache = k_all[:, :, :t], v_all[:, :, :t]
            k_new, v_new = k_all[:, :, t : t + 1], v_all[:, :, t : t + 1]
            decode_t = op.forward_fp32(q_t, k_cache, v_cache, k_new, v_new, causal=True)
            max_err = (decode_t - prefill[:, :, t : t + 1]).abs().max().item()
            assert torch.allclose(
                decode_t, prefill[:, :, t : t + 1], atol=_DECODE_ATOL, rtol=0.0
            ), f"decode step {t} diverges from prefill by {max_err:.3g} > {_DECODE_ATOL}"


def test_batch_invariance_slice():
    """Axis-A: a row computed in a batch-of-N is bitwise identical to batch-of-1.

    Each query row reduces over its own keys independently of how many sequences
    share the batch, so slicing row i out of the batch-N output must equal running
    row i alone. CPU GEMM is pinned to one thread so the matmul reduction order is
    batch-independent (multi-threaded GEMM can split by batch and break bitwise).
    """
    op = NativeKVCacheAttnOp()
    n, s_past, s_new = 4, 6, 2
    q = _q(n, s_new, seed=100)
    k_cache, v_cache = _kv(n, s_past, seed=101)
    k_new, v_new = _kv(n, s_new, seed=102)

    with _single_thread():
        full = op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True)
        for i in range(n):
            row = op.forward_fp32(
                q[i : i + 1],
                k_cache[i : i + 1],
                v_cache[i : i + 1],
                k_new[i : i + 1],
                v_new[i : i + 1],
                causal=True,
            )
            assert torch.equal(full[i : i + 1], row), f"batch row {i} not invariant"


def test_empty_cache_equals_plain_attention():
    """S_past=0 (pure prefill) delegates to plain attention over k_new/v_new."""
    op = NativeKVCacheAttnOp()
    ref = NativeAttentionOp()
    b, seq = 2, 5
    q = _q(b, seq, seed=30)
    k_new, v_new = _kv(b, seq, seed=31)
    k_cache = k_new[:, :, :0]  # [B, Hkv, 0, D]
    v_cache = v_new[:, :, :0]

    with _single_thread():
        got = op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True)
        want = ref.forward_fp32(q, k_new, v_new, causal=True)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# Decode (Sq=1) sees the whole cache; closed-form uniform check.
# --------------------------------------------------------------------------- #
def test_decode_single_query_uniform_attention():
    """With identical keys, a single decode query attends uniformly -> mean of V."""
    op = NativeKVCacheAttnOp()
    b, s_past, s_new = 1, 4, 1
    q = _q(b, s_new, seed=40)
    # All keys identical -> all scores equal -> softmax uniform over all S_past+S_new.
    k = torch.ones(b, _N_KV, 1, _HEAD_DIM)
    k_cache = k.expand(b, _N_KV, s_past, _HEAD_DIM).contiguous()
    k_new = k.expand(b, _N_KV, s_new, _HEAD_DIM).contiguous()
    gen = torch.Generator().manual_seed(41)
    v_cache = torch.randn(b, _N_KV, s_past, _HEAD_DIM, generator=gen)
    v_new = torch.randn(b, _N_KV, s_new, _HEAD_DIM, generator=gen)

    out = op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True)
    v_full = torch.cat([v_cache, v_new], dim=2)  # [B, Hkv, Skv, D]
    expected_kv = v_full.mean(dim=2, keepdim=True)  # uniform avg over keys
    expected = expected_kv.repeat_interleave(_N_HEADS // _N_KV, dim=1)  # GQA broadcast
    assert torch.allclose(out, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# key_padding_mask over the concatenated length.
# --------------------------------------------------------------------------- #
def test_key_padding_mask_excludes_padded_keys():
    """Padding columns (over S_past+S_new) get zero weight (~ attending valid keys)."""
    op = NativeKVCacheAttnOp()
    b, s_past, s_new = 2, 5, 3
    q = _q(b, s_new, seed=50)
    k_cache, v_cache = _kv(b, s_past, seed=51)
    k_new, v_new = _kv(b, s_new, seed=52)
    skv = s_past + s_new

    # Mask out the last 2 cached keys for one batch row; non-causal to isolate padding.
    mask = torch.ones(b, skv, dtype=torch.bool)
    mask[0, s_past - 2 : s_past] = False

    with _single_thread():
        masked = op.forward_fp32(
            q, k_cache, v_cache, k_new, v_new, causal=False, key_padding_mask=mask
        )
        # Equivalent: drop those keys entirely from the valid row.
        keep = mask[0]
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)
        ref = NativeAttentionOp()
        valid_only_row0 = ref.forward_fp32(
            q[:1], k_full[:1][:, :, keep], v_full[:1][:, :, keep], causal=False
        )
    assert torch.allclose(masked[:1], valid_only_row0, atol=_PADDING_ATOL, rtol=0.0)


# --------------------------------------------------------------------------- #
# Axis-B accuracy: low-precision forward vs fp32 ground truth.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", _DTYPES_AXIS_B)
def test_dtype_path_accuracy(dtype: torch.dtype):
    """forward(dtype) tracks forward_fp32 within the per-dtype peak-relative tolerance."""
    op = NativeKVCacheAttnOp()
    b, s_past, s_new = 2, 12, 4
    q = _q(b, s_new, seed=60, dtype=dtype)
    k_cache, v_cache = _kv(b, s_past, seed=61, dtype=dtype)
    k_new, v_new = _kv(b, s_new, seed=62, dtype=dtype)

    got = op.forward(q, k_cache, v_cache, k_new, v_new, causal=True)
    ref = op.forward_fp32(
        q.float(),
        k_cache.float(),
        v_cache.float(),
        k_new.float(),
        v_new.float(),
        causal=True,
    )
    assert got.dtype == dtype
    peak = ref.abs().max().item()
    max_err = (got.float() - ref).abs().max().item()
    assert (
        max_err <= _DTYPE_REL_PEAK[dtype] * peak
    ), f"{dtype}: max_abs_err={max_err:.3g} > {_DTYPE_REL_PEAK[dtype]:.1%} of peak {peak:.3g}"


# --------------------------------------------------------------------------- #
# Shape / GQA / purity / registry.
# --------------------------------------------------------------------------- #
def test_output_shape_follows_q():
    op = NativeKVCacheAttnOp()
    b, s_past, s_new = 3, 9, 2
    q = _q(b, s_new, seed=70)
    k_cache, v_cache = _kv(b, s_past, seed=71)
    k_new, v_new = _kv(b, s_new, seed=72)
    out = op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True)
    assert out.shape == (b, _N_HEADS, s_new, _HEAD_DIM)


def test_gqa_requires_divisible_heads():
    """q heads not divisible by KV heads raises (propagated from NativeAttentionOp)."""
    op = NativeKVCacheAttnOp()
    b = 1
    q = torch.randn(b, 7, 1, _HEAD_DIM)  # 7 not divisible by _N_KV=8
    k_cache, v_cache = _kv(b, 3, seed=80)
    k_new, v_new = _kv(b, 1, seed=81)
    with pytest.raises(ValueError):
        op.forward_fp32(q, k_cache, v_cache, k_new, v_new)


def test_inputs_not_mutated():
    """Pure op: cache/new tensors are not modified in place."""
    op = NativeKVCacheAttnOp()
    b, s_past, s_new = 2, 5, 2
    q = _q(b, s_new, seed=90)
    k_cache, v_cache = _kv(b, s_past, seed=91)
    k_new, v_new = _kv(b, s_new, seed=92)
    snapshots = [t.clone() for t in (q, k_cache, v_cache, k_new, v_new)]
    op.forward_fp32(q, k_cache, v_cache, k_new, v_new, causal=True)
    for orig, snap in zip((q, k_cache, v_cache, k_new, v_new), snapshots):
        assert torch.equal(orig, snap)


def test_registry_dispatches_kv_cache_attn_op():
    """The registry resolves "kv_cache_attention" to NativeKVCacheAttnOp."""
    assert isinstance(kernel_registry.get_op("kv_cache_attention"), NativeKVCacheAttnOp)
