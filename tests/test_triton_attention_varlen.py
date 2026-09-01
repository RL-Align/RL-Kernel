# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the Triton FlashAttention LSE export and packed varlen path.

Extends `rl_engine.kernels.ops.triton.triton_attn` (the "cross-platform semantic
baseline" the SM90/SM80/ROCm fused-attention kernels will validate against) with:

  * `return_lse=True` on the existing dense kernel: exposes the attention-domain
    LSE (log-sum-exp of the scaled, masked QK^T logits) already accumulated
    online as `M`/`L` for the backward pass, but previously discarded.
  * `triton_flash_attention_varlen`: packed `[total_tokens, H, D]` + `cu_seqlens`
    attention, for RL rollout/training batches that are concatenated rather than
    padded (same convention as `flash_attn_varlen_func` and this repo's `pack`
    op, #182).

Both are checked against an independent per-sequence fp32 reference (`_ref_attn`,
a masked-softmax + logsumexp closed form matching the causal-offset convention
`Skv - Sq` documented for `NativeAttentionOp` in docs/operators/attention.md),
not against the dense Triton kernel itself, so a shared bug in the online-softmax
loop cannot hide from these tests.

The exported LSE is attention-domain (per query row, over the key dimension),
not the vocab-domain LSE produced by the logp/linear_logp kernels.
"""

import math

import pytest
import torch

from rl_engine.kernels.ops.triton.triton_attn import (
    triton_flash_attention,
    triton_flash_attention_varlen,
)

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton attention requires a CUDA device and Triton.",
)

_ATOL_OUT = 0.05  # fp16 accumulation tolerance, dense and varlen paths alike
_ATOL_LSE = 1e-2


def _ref_attn(q, k, v, causal, sm_scale):
    """[H, Sq, D] / [H, Skv, D] fp32 masked-softmax reference with LSE."""
    H, Sq, D = q.shape
    Skv = k.shape[1]
    scores = torch.einsum("hqd,hkd->hqk", q, k) * sm_scale
    if causal:
        mask = torch.triu(
            torch.ones(Sq, Skv, dtype=torch.bool, device=q.device), diagonal=Skv - Sq + 1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,hkd->hqd", probs, v)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _run_varlen_case(seqlens_q, seqlens_k, heads, head_dim, causal, dtype=torch.float16):
    device = "cuda"
    batch = len(seqlens_q)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    cu_q = torch.tensor(
        [0, *torch.tensor(seqlens_q).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(seqlens_k).cumsum(0).tolist()], dtype=torch.int32, device=device
    )

    q = torch.randn(total_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, heads, head_dim, device=device, dtype=dtype, requires_grad=True)

    sm_scale = 1.0 / math.sqrt(head_dim)
    out, lse = triton_flash_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=causal,
        sm_scale=sm_scale,
        return_lse=True,
    )
    do = torch.randn_like(out)
    out.backward(do)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_()
    k_ref = k.detach().clone().float().requires_grad_()
    v_ref = v.detach().clone().float().requires_grad_()
    out_ref = torch.empty(total_q, heads, head_dim, device=device, dtype=torch.float32)
    lse_ref = torch.empty(total_q, heads, device=device, dtype=torch.float32)

    qs = ks = 0
    for b in range(batch):
        sq, sk = seqlens_q[b], seqlens_k[b]
        o, lval = _ref_attn(
            q_ref[qs : qs + sq].transpose(0, 1),
            k_ref[ks : ks + sk].transpose(0, 1),
            v_ref[ks : ks + sk].transpose(0, 1),
            causal,
            sm_scale,
        )
        out_ref[qs : qs + sq] = o.transpose(0, 1)
        lse_ref[qs : qs + sq] = lval.transpose(0, 1)
        qs += sq
        ks += sk
    out_ref.backward(do.float())

    torch.testing.assert_close(out.float(), out_ref, atol=_ATOL_OUT, rtol=0.0)
    torch.testing.assert_close(lse, lse_ref, atol=_ATOL_LSE, rtol=0.0)
    torch.testing.assert_close(dq.float(), q_ref.grad, atol=_ATOL_OUT, rtol=0.0)
    torch.testing.assert_close(dk.float(), k_ref.grad, atol=_ATOL_OUT, rtol=0.0)
    torch.testing.assert_close(dv.float(), v_ref.grad, atol=_ATOL_OUT, rtol=0.0)


@requires_triton_cuda
class TestDenseLSEExport:
    def test_lse_matches_reference(self):
        device = "cuda"
        batch, heads, seq, head_dim = 2, 4, 256, 64
        sm_scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)

        out, lse = triton_flash_attention(q, k, v, causal=True, sm_scale=sm_scale, return_lse=True)
        assert lse.shape == (batch, heads, seq)
        assert lse.dtype == torch.float32

        ref_out = torch.empty_like(q, dtype=torch.float32)
        ref_lse = torch.empty(batch, heads, seq, device=device)
        for b in range(batch):
            o, lval = _ref_attn(q[b].float(), k[b].float(), v[b].float(), True, sm_scale)
            ref_out[b], ref_lse[b] = o, lval

        torch.testing.assert_close(out.float(), ref_out, atol=_ATOL_OUT, rtol=0.0)
        torch.testing.assert_close(lse, ref_lse, atol=_ATOL_LSE, rtol=0.0)

    def test_lse_is_non_differentiable_and_backward_still_works(self):
        device = "cuda"
        q = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16, requires_grad=True)

        out, lse = triton_flash_attention(q, k, v, causal=True, return_lse=True)
        assert not lse.requires_grad

        out.float().sum().backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None

    def test_default_call_is_backward_compatible(self):
        device = "cuda"
        q = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16)
        k = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16)
        v = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16)

        out = triton_flash_attention(q, k, v, causal=True)
        assert isinstance(out, torch.Tensor)


@requires_triton_cuda
class TestVarlenAttention:
    def test_prefill_uneven_seqlens_not_block_aligned(self):
        # BLOCK_M/BLOCK_N default to 64/128; deliberately not multiples of either.
        _run_varlen_case([37, 128, 200, 5], [37, 128, 200, 5], heads=4, head_dim=64, causal=True)

    def test_non_causal(self):
        _run_varlen_case([37, 130, 61], [37, 130, 61], heads=2, head_dim=64, causal=False)

    def test_decode_style_sq_less_than_skv(self):
        # Small new-query chunk (e.g. rollout decode step) against a longer KV cache,
        # varying independently per sequence in the batch.
        _run_varlen_case([1, 3, 1], [50, 91, 17], heads=4, head_dim=64, causal=True)

    def test_head_dim_128(self):
        _run_varlen_case([100, 260], [100, 260], heads=2, head_dim=128, causal=True)

    def test_zero_length_sequence_in_batch(self):
        # A fully-masked / empty response is a real occurrence in packed RL batches.
        _run_varlen_case([0, 128, 0, 37], [50, 128, 0, 37], heads=2, head_dim=64, causal=True)

    def test_non_contiguous_q_k_v_backward(self):
        # Regression test (review finding): `do` was forced `.contiguous()` in
        # backward, but the kernels indexed DO using Q's/Out's strides instead of
        # DO's own -- silently correct only when Q/Out/DO all happen to share the
        # same contiguous layout, as every other case in this file does by
        # construction (fresh `torch.randn(...)`).
        #
        # Building q/k/v as `[H, total, D].transpose(0, 1)` gives a non-contiguous
        # but non-overlapping-and-dense view. `torch.empty_like` (used for `out`
        # and `dq`) preserves that exact stride pattern rather than falling back to
        # a contiguous layout, so `out`'s strides differ from `do`'s (`do` is a
        # plain contiguous tensor here) -- exactly the mismatch the review flagged.
        device = "cuda"
        heads, head_dim = 2, 64
        seqlens = [37, 61]
        total = sum(seqlens)
        cu = torch.tensor(
            [0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device
        )
        sm_scale = 1.0 / math.sqrt(head_dim)

        def make(seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            base = torch.randn(
                heads, total, head_dim, device=device, dtype=torch.float16, generator=gen
            )
            return base.transpose(0, 1).detach().requires_grad_()

        q, k, v = make(1), make(2), make(3)
        assert not (q.is_contiguous() or k.is_contiguous() or v.is_contiguous())

        out = triton_flash_attention_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale
        )
        assert not out.is_contiguous()  # confirms empty_like preserved q's layout

        do = torch.randn(total, heads, head_dim, device=device, dtype=out.dtype)  # plain contiguous
        assert do.stride() != out.stride()
        out.backward(do)
        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        q_ref = q.detach().clone().float().requires_grad_()
        k_ref = k.detach().clone().float().requires_grad_()
        v_ref = v.detach().clone().float().requires_grad_()
        out_ref = torch.empty(total, heads, head_dim, device=device, dtype=torch.float32)
        qs = 0
        for sq in seqlens:
            o, _ = _ref_attn(
                q_ref[qs : qs + sq].transpose(0, 1),
                k_ref[qs : qs + sq].transpose(0, 1),
                v_ref[qs : qs + sq].transpose(0, 1),
                True,
                sm_scale,
            )
            out_ref[qs : qs + sq] = o.transpose(0, 1)
            qs += sq
        out_ref.backward(do.float())

        torch.testing.assert_close(out.float(), out_ref, atol=_ATOL_OUT, rtol=0.0)
        torch.testing.assert_close(dq.float(), q_ref.grad, atol=_ATOL_OUT, rtol=0.0)
        torch.testing.assert_close(dk.float(), k_ref.grad, atol=_ATOL_OUT, rtol=0.0)
        torch.testing.assert_close(dv.float(), v_ref.grad, atol=_ATOL_OUT, rtol=0.0)
