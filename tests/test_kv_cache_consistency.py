# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""
Prefill / decode KV-cache path-consistency tests (WS1, issue #152).

One representative test per claim across the five planned PRs:

  1. Full-prefill == chunked-prefill, and naive batched SDPA is NOT bitwise.
  2. Decode (one query vs N cached KV) == prefill (incl. padding, GQA).
  3. Stored-KV: matched dtype is bitwise; low precision stays bounded.
  4. "Generate then re-score" equivalence.
  5. CI smoke across short / long / varlen / padded.

Contract: prefill and decode share one fixed reduction order, so matched dtype
is bitwise-equal. CPU/fp32 -- no GPU/kernel build required.
"""

from __future__ import annotations

import pytest
import torch

import torch.nn.functional as F

from rl_engine.testing.kv_consistency import (
    AttentionSpec,
    KVCache,
    TinyCausalLM,
    assert_path_parity,
    attend_single_query,
    expand_kv_heads,
    fixed_order_attention,
    parity_report,
    replay_decode,
)
from rl_engine.testing.reference_ops import selected_logprobs_reference

torch.manual_seed(0)

# One GQA spec is the representative case: it exercises the GQA head expansion
# (num_heads > num_kv_heads) that dense/MQA are degenerate cases of.
SPEC = AttentionSpec(num_heads=8, num_kv_heads=2, head_dim=16)


def _make_qkv(batch, seqlen, spec, *, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seqlen, spec.num_heads, spec.head_dim, generator=gen)
    k = torch.randn(batch, seqlen, spec.num_kv_heads, spec.head_dim, generator=gen)
    v = torch.randn(batch, seqlen, spec.num_kv_heads, spec.head_dim, generator=gen)
    return q, k, v


# --------------------------------------------------------------------------- #
# PR1 -- Full-prefill vs chunked-prefill consistency.
# --------------------------------------------------------------------------- #


def test_full_vs_chunked_prefill_bitwise():
    """Carrying KV across chunk boundaries must not change the reduction."""
    q, k, v = _make_qkv(2, 24, SPEC)
    full = fixed_order_attention(q, k, v, spec=SPEC)

    b, s = q.shape[0], q.shape[1]
    cache = KVCache(b, SPEC, s, dtype=q.dtype, device=q.device)
    chunked = torch.zeros_like(full)
    for start in range(0, s, 8):  # chunk size 8
        for t in range(start, min(start + 8, s)):
            cache.append(k[:, t], v[:, t])
            kc, vc = cache.context()
            chunked[:, t] = attend_single_query(q[:, t], kc, vc, scale=SPEC.scale, spec=SPEC)
    assert_path_parity(chunked, full, require_bitwise=True)


def _naive_batched_sdpa(q, k, v, spec):
    """A naive whole-sequence SDPA -- the path whose reduction order we must NOT use."""
    qh = q.transpose(1, 2).float()  # [B, H, S, D]
    kh = expand_kv_heads(k, spec).transpose(1, 2).float()
    vh = expand_kv_heads(v, spec).transpose(1, 2).float()
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=spec.causal, scale=spec.scale)
    return out.transpose(1, 2)


def test_naive_sdpa_only_matches_within_tolerance():
    """The naive batched path drifts from the contract -- the bug #152 targets."""
    q, k, v = _make_qkv(2, 48, SPEC)
    contract = fixed_order_attention(q, k, v, spec=SPEC)
    report = parity_report(_naive_batched_sdpa(q, k, v, SPEC), contract)
    assert not report.bitwise  # close, but NOT bitwise: reduction-order drift
    assert report.max_abs_error < 1e-4


# --------------------------------------------------------------------------- #
# PR2 -- Decode (one query vs N cached KV) vs prefill.
# --------------------------------------------------------------------------- #


def test_decode_matches_prefill_bitwise():
    q, k, v = _make_qkv(4, 32, SPEC)  # batch > 1
    prefill = fixed_order_attention(q, k, v, spec=SPEC)
    decode = replay_decode(q, k, v, spec=SPEC)
    assert_path_parity(decode, prefill, require_bitwise=True)


@pytest.mark.parametrize("pad_side", ["left", "right"])
def test_decode_matches_prefill_with_padding(pad_side):
    batch, seqlen = 3, 20
    q, k, v = _make_qkv(batch, seqlen, SPEC)
    lengths = [20, 14, 8]
    mask = torch.zeros(batch, seqlen, dtype=torch.bool)
    for b, L in enumerate(lengths):
        if pad_side == "right":
            mask[b, :L] = True
        else:
            mask[b, seqlen - L :] = True
    prefill = fixed_order_attention(q, k, v, spec=SPEC, key_mask=mask)
    decode = replay_decode(q, k, v, spec=SPEC, key_mask=mask)
    assert_path_parity(decode, prefill, require_bitwise=True, msg=f"pad={pad_side}")


# --------------------------------------------------------------------------- #
# PR3 -- Stored-KV layout/dtype: no writer-vs-reader precision drift.
# --------------------------------------------------------------------------- #


def test_stored_kv_matched_dtype_is_bitwise():
    """fp32 writer + fp32 reader -> the cache itself adds zero drift."""
    q, k, v = _make_qkv(2, 24, SPEC)
    prefill = fixed_order_attention(q, k, v, spec=SPEC)
    decode = replay_decode(q, k, v, spec=SPEC, kv_dtype=torch.float32)
    assert_path_parity(decode, prefill, require_bitwise=True)


def test_stored_kv_low_precision_within_tolerance():
    """Low-precision storage is the ONLY drift source, and it is bounded."""
    q, k, v = _make_qkv(2, 32, SPEC)
    prefill = fixed_order_attention(q, k, v, spec=SPEC)
    decode = replay_decode(q, k, v, spec=SPEC, kv_dtype=torch.float16)
    assert_path_parity(decode, prefill, atol=5e-3, rtol=5e-3)


# --------------------------------------------------------------------------- #
# PR4 -- "Generate then re-score" equivalence (full forward chain).
# --------------------------------------------------------------------------- #


def test_generate_then_rescore_equivalence():
    model = TinyCausalLM(vocab_size=64, d_model=48, spec=SPEC, seed=1)
    prompt = torch.randint(0, 64, (2, 5))

    full_ids, gen_step_logprobs = model.generate(prompt, max_new_tokens=7)

    # Re-score the produced sequence through the prefill (training) path.
    prefill_logits = model.prefill_logits(full_ids)
    rescored = selected_logprobs_reference(prefill_logits[:, :-1], full_ids[:, 1:])
    rescored_gen = rescored[:, prompt.shape[1] - 1 :]  # generated positions only
    assert_path_parity(rescored_gen, gen_step_logprobs, require_bitwise=True)


# --------------------------------------------------------------------------- #
# PR5 -- CI smoke test (short / long / varlen / padded).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seqlen", "lengths"),
    [
        pytest.param(4, None, id="short"),
        pytest.param(256, None, id="long"),
        pytest.param(32, [32, 17, 5, 1], id="varlen"),
        pytest.param(24, [24, 24, 10, 3], id="padded"),
    ],
)
def test_decode_smoke(seqlen, lengths):
    batch = 4 if lengths else 2
    q, k, v = _make_qkv(batch, seqlen, SPEC, seed=7)
    mask = None
    if lengths:
        mask = torch.zeros(batch, seqlen, dtype=torch.bool)
        for b, L in enumerate(lengths):
            mask[b, :L] = True
    prefill = fixed_order_attention(q, k, v, spec=SPEC, key_mask=mask)
    decode = replay_decode(q, k, v, spec=SPEC, key_mask=mask)
    assert_path_parity(decode, prefill, require_bitwise=True)
