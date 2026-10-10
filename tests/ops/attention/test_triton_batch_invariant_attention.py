# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.shared.triton.attention.standard_attn import (
    TritonBatchInvariantAttentionOp,
    triton_batch_invariant_attention_with_lse,
)
from rl_engine.reference.attention.standard_attn import NativeAttentionOp

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _qkv(
    batch: int,
    q_len: int,
    kv_len: int,
    *,
    q_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, q_heads, q_len, head_dim), generator=gen, device="cuda", dtype=dtype)
    k = torch.randn((batch, kv_heads, kv_len, head_dim), generator=gen, device="cuda", dtype=dtype)
    v = torch.randn((batch, kv_heads, kv_len, head_dim), generator=gen, device="cuda", dtype=dtype)
    return q, k, v


def _run_backward(op, q, k, v, dy, *, causal=True, key_padding_mask=None):
    q_req = q.detach().clone().requires_grad_(True)
    k_req = k.detach().clone().requires_grad_(True)
    v_req = v.detach().clone().requires_grad_(True)
    out = op(q_req, k_req, v_req, causal=causal, key_padding_mask=key_padding_mask)
    out.backward(dy)
    return out.detach(), q_req.grad.detach(), k_req.grad.detach(), v_req.grad.detach()


def _run_chunked_prefill(q, k, v, *, chunk_size: int, key_padding_mask=None):
    outs = []
    lses = []
    for start in range(0, q.size(2), chunk_size):
        end = min(q.size(2), start + chunk_size)
        chunk_mask = None if key_padding_mask is None else key_padding_mask[:, :end]
        out, lse = triton_batch_invariant_attention_with_lse(
            q[:, :, start:end],
            k[:, :, :end],
            v[:, :, :end],
            causal=True,
            key_padding_mask=chunk_mask,
        )
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol", [(torch.float16, 3e-3, 3e-3), (torch.bfloat16, 5e-2, 2e-2)]
)
def test_triton_attention_matches_native_forward(causal, dtype, atol, rtol):
    q, k, v = _qkv(2, 17, 17, dtype=dtype)
    mask = torch.ones((2, 17), device="cuda", dtype=torch.bool)
    mask[1, 13:] = False

    out = TritonBatchInvariantAttentionOp()(q, k, v, causal=causal, key_padding_mask=mask)
    ref = NativeAttentionOp().forward_fp32(q, k, v, causal=causal, key_padding_mask=mask)

    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@requires_cuda
def test_triton_attention_lse_matches_native_softmax_stats():
    q, k, v = _qkv(1, 9, 13, dtype=torch.float32, q_heads=2, kv_heads=1, head_dim=16)
    mask = torch.ones((1, 13), device="cuda", dtype=torch.bool)
    mask[:, 10:] = False

    out, lse = triton_batch_invariant_attention_with_lse(
        q, k, v, causal=True, key_padding_mask=mask
    )
    ref = NativeAttentionOp().forward_fp32(q, k, v, causal=True, key_padding_mask=mask)

    k_expanded = k.repeat_interleave(2, dim=1)
    scores = torch.matmul(q.float(), k_expanded.float().transpose(-1, -2)) * (1.0 / 16**0.5)
    causal_mask = torch.triu(
        torch.ones(9, 13, dtype=torch.bool, device="cuda"),
        diagonal=13 - 9 + 1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    ref_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_triton_attention_lse_is_non_differentiable():
    q, k, v = _qkv(1, 8, 8, dtype=torch.bfloat16, seed=12)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out, lse = triton_batch_invariant_attention_with_lse(q, k, v, causal=True)
    assert lse.requires_grad is False
    (out.float().sum() + lse.float().sum()).backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@requires_cuda
def test_triton_attention_batch_position_invariant():
    dtype = torch.bfloat16
    q_real, k_real, v_real = _qkv(1, 16, 16, dtype=dtype, seed=2)
    op = TritonBatchInvariantAttentionOp()

    out_single = op(q_real, k_real, v_real, causal=True)
    q_batch, k_batch, v_batch = _qkv(4, 16, 16, dtype=dtype, seed=3)
    q_batch[2] = q_real[0]
    k_batch[2] = k_real[0]
    v_batch[2] = v_real[0]
    out_batch = op(q_batch, k_batch, v_batch, causal=True)

    assert torch.equal(out_single[0], out_batch[2])


@requires_cuda
def test_triton_attention_padding_layout_invariant():
    dtype = torch.bfloat16
    q, k_real, v_real = _qkv(1, 8, 8, dtype=dtype, seed=4)
    op = TritonBatchInvariantAttentionOp()

    mask_a = torch.ones((1, 8), device="cuda", dtype=torch.bool)
    out_a = op(q, k_real, v_real, causal=False, key_padding_mask=mask_a)

    _, k_pad, v_pad = _qkv(1, 8, 16, dtype=dtype, seed=5)
    mask_b = torch.zeros((1, 16), device="cuda", dtype=torch.bool)
    real_positions = [2 * i + 1 for i in range(8)]
    for src, dst in enumerate(real_positions):
        k_pad[:, :, dst] = k_real[:, :, src]
        v_pad[:, :, dst] = v_real[:, :, src]
        mask_b[:, dst] = True
    out_b = op(q, k_pad, v_pad, causal=False, key_padding_mask=mask_b)

    torch.testing.assert_close(out_a.float(), out_b.float(), atol=5e-2, rtol=2e-2)


def _c3_style_rows(
    n_rows: int,
    heads: int,
    head_dim: int,
    *,
    offset: int,
    dtype: torch.dtype,
    sample_id: str = "s2",
):
    """Match C3 ``_logical_fill`` so the 1-ULP left-pad case is reproducible."""

    n = heads * head_dim
    sample_ord = sum(ord(ch) for ch in sample_id)
    rows = []
    for position in range(n_rows):
        axis = torch.arange(n, device="cuda", dtype=torch.int64)
        values = ((axis + sample_ord * 17 + position * 13 + offset * 11) % 257) - 128
        rows.append((values.to(torch.float32) / 1024.0).to(dtype).reshape(heads, head_dim))
    stacked = torch.stack(rows)
    return stacked.unsqueeze(0).permute(0, 2, 1, 3).contiguous()


@requires_cuda
def test_triton_attention_causal_left_pad_matches_right_pad_bitwise():
    """C3 BN/padded_left vs packed/right-pad must be bitwise at Qwen3 head_dim=128.

    The kernel rebases a contiguous valid KV interval to logical column zero so
    left padding cannot move values to different reduction lanes.
    """

    dtype = torch.bfloat16
    valid, right_len, left_len, heads, head_dim = 13, 19, 20, 4, 128
    q_real = _c3_style_rows(valid, heads, head_dim, offset=0, dtype=dtype)
    k_real = _c3_style_rows(valid, 1, head_dim, offset=1, dtype=dtype)
    v_real = _c3_style_rows(valid, 1, head_dim, offset=2, dtype=dtype)
    op = TritonBatchInvariantAttentionOp()

    def _place(padded: int, side: str) -> torch.Tensor:
        q = torch.zeros((1, heads, padded, head_dim), device="cuda", dtype=dtype)
        k = torch.zeros((1, 1, padded, head_dim), device="cuda", dtype=dtype)
        v = torch.zeros((1, 1, padded, head_dim), device="cuda", dtype=dtype)
        mask = torch.zeros((1, padded), device="cuda", dtype=torch.bool)
        sl = slice(padded - valid, padded) if side == "left" else slice(0, valid)
        q[:, :, sl] = q_real
        k[:, :, sl] = k_real
        v[:, :, sl] = v_real
        mask[:, sl] = True
        return op(q, k, v, causal=True, key_padding_mask=mask)[:, :, sl]

    assert torch.equal(_place(left_len, "left"), _place(right_len, "right"))


@requires_cuda
def test_triton_attention_lse_padding_layout_invariant():
    dtype = torch.bfloat16
    q, k_real, v_real = _qkv(1, 8, 8, dtype=dtype, seed=8)

    mask_a = torch.ones((1, 8), device="cuda", dtype=torch.bool)
    out_a, lse_a = triton_batch_invariant_attention_with_lse(
        q,
        k_real,
        v_real,
        causal=False,
        key_padding_mask=mask_a,
    )

    _, k_pad, v_pad = _qkv(1, 8, 17, dtype=dtype, seed=9)
    mask_b = torch.zeros((1, 17), device="cuda", dtype=torch.bool)
    real_positions = [2 * i for i in range(8)]
    for src, dst in enumerate(real_positions):
        k_pad[:, :, dst] = k_real[:, :, src]
        v_pad[:, :, dst] = v_real[:, :, src]
        mask_b[:, dst] = True
    out_b, lse_b = triton_batch_invariant_attention_with_lse(
        q,
        k_pad,
        v_pad,
        causal=False,
        key_padding_mask=mask_b,
    )

    torch.testing.assert_close(out_a.float(), out_b.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(lse_a, lse_b, atol=1e-5, rtol=1e-5)


@requires_cuda
@pytest.mark.parametrize(
    "q_heads, kv_heads, head_dim",
    [
        pytest.param(4, 4, 32, id="mha"),
        pytest.param(4, 2, 32, id="gqa-small"),
        pytest.param(8, 2, 64, id="gqa-wide"),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 13])
def test_triton_attention_chunked_prefill_matches_full_prefill(
    q_heads, kv_heads, head_dim, chunk_size
):
    q, k, v = _qkv(
        2,
        13,
        13,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        seed=20,
    )

    full, full_lse = triton_batch_invariant_attention_with_lse(q, k, v, causal=True)
    chunked, chunked_lse = _run_chunked_prefill(q, k, v, chunk_size=chunk_size)

    assert torch.equal(full, chunked)
    assert torch.equal(full_lse, chunked_lse)


@requires_cuda
@pytest.mark.parametrize("chunk_size", [1, 4, 6])
def test_triton_attention_chunked_prefill_with_key_padding_mask(chunk_size):
    q, k, v = _qkv(2, 13, 13, dtype=torch.bfloat16, seed=21)
    mask = torch.ones((2, 13), device="cuda", dtype=torch.bool)
    mask[0, 9:] = False
    mask[1, 11:] = False

    full, full_lse = triton_batch_invariant_attention_with_lse(
        q,
        k,
        v,
        causal=True,
        key_padding_mask=mask,
    )
    chunked, chunked_lse = _run_chunked_prefill(
        q,
        k,
        v,
        chunk_size=chunk_size,
        key_padding_mask=mask,
    )

    assert torch.equal(full, chunked)
    assert torch.equal(full_lse, chunked_lse)


@requires_cuda
def test_triton_attention_decode_matches_prefill_suffix_context():
    dtype = torch.bfloat16
    q_full, k_full, v_full = _qkv(1, 12, 12, dtype=dtype, seed=6)
    op = TritonBatchInvariantAttentionOp()

    prefill = op(q_full, k_full, v_full, causal=True)
    decode = op(q_full[:, :, -1:], k_full, v_full, causal=True)

    assert torch.equal(prefill[:, :, -1:], decode)


@requires_cuda
@pytest.mark.parametrize("s_new", [1, 3])
def test_triton_attention_kv_cache_handoff_matches_prefill_suffix(s_new):
    dtype = torch.bfloat16
    q_full, k_full, v_full = _qkv(2, 12, 12, dtype=dtype, seed=22)
    op = TritonBatchInvariantAttentionOp()

    prefill = op(q_full, k_full, v_full, causal=True)
    decode = op(q_full[:, :, -s_new:], k_full, v_full, causal=True)

    assert torch.equal(prefill[:, :, -s_new:], decode)


@requires_cuda
def test_triton_attention_all_false_key_padding_mask_row_matches_native():
    dtype = torch.bfloat16
    q, k, v = _qkv(2, 5, 5, dtype=dtype, seed=23)
    mask = torch.ones((2, 5), device="cuda", dtype=torch.bool)
    mask[1] = False
    q_req = q.detach().clone().requires_grad_(True)
    k_req = k.detach().clone().requires_grad_(True)
    v_req = v.detach().clone().requires_grad_(True)

    out, lse = triton_batch_invariant_attention_with_lse(
        q_req,
        k_req,
        v_req,
        causal=False,
        key_padding_mask=mask,
    )
    ref = NativeAttentionOp().forward_fp32(q, k, v, causal=False, key_padding_mask=mask)

    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isfinite(out).all()
    assert torch.isneginf(lse[1]).all()
    assert torch.equal(ref[1], torch.zeros_like(ref[1]))
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=2e-2)

    dy = torch.randn_like(out)
    out.backward(dy)

    assert torch.isfinite(q_req.grad).all()
    assert torch.isfinite(k_req.grad).all()
    assert torch.isfinite(v_req.grad).all()
    assert torch.equal(q_req.grad[1], torch.zeros_like(q_req.grad[1]))
    assert torch.equal(k_req.grad[1], torch.zeros_like(k_req.grad[1]))
    assert torch.equal(v_req.grad[1], torch.zeros_like(v_req.grad[1]))


@requires_cuda
def test_triton_attention_backward_matches_native_vjp():
    dtype = torch.bfloat16
    q, k, v = _qkv(1, 8, 8, dtype=dtype, seed=7)
    dy = torch.randn_like(q)
    op = TritonBatchInvariantAttentionOp()
    native = NativeAttentionOp()

    out, dq, dk, dv = _run_backward(op, q, k, v, dy, causal=True)
    ref_out, ref_dq, ref_dk, ref_dv = _run_backward(native, q, k, v, dy, causal=True)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dq.float(), ref_dq.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dk.float(), ref_dk.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dv.float(), ref_dv.float(), atol=5e-2, rtol=2e-2)
    import rl_engine.backends.shared.triton.attention.standard_attn as attn_mod

    assert not hasattr(attn_mod, "NativeAttentionOp")


@requires_cuda
def test_triton_attention_partial_pad_backward_matches_native_vjp():
    """GQA + partial key padding: compare dq/dk/dv against NativeAttentionOp."""
    dtype = torch.bfloat16
    q, k, v = _qkv(2, 8, 8, q_heads=4, kv_heads=2, dtype=dtype, seed=24)
    mask = torch.ones((2, 8), device="cuda", dtype=torch.bool)
    mask[0, 6:] = False
    mask[1, 5:] = False
    dy = torch.randn_like(q)
    op = TritonBatchInvariantAttentionOp()
    native = NativeAttentionOp()

    out, dq, dk, dv = _run_backward(op, q, k, v, dy, causal=True, key_padding_mask=mask)
    ref_out, ref_dq, ref_dk, ref_dv = _run_backward(
        native, q, k, v, dy, causal=True, key_padding_mask=mask
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all() and torch.isfinite(dv).all()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dq.float(), ref_dq.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dk.float(), ref_dk.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dv.float(), ref_dv.float(), atol=5e-2, rtol=2e-2)
    # Padded key positions must not accumulate gradient.
    assert torch.equal(dk[0, :, 6:], torch.zeros_like(dk[0, :, 6:]))
    assert torch.equal(dv[0, :, 6:], torch.zeros_like(dv[0, :, 6:]))
    assert torch.equal(dk[1, :, 5:], torch.zeros_like(dk[1, :, 5:]))
    assert torch.equal(dv[1, :, 5:], torch.zeros_like(dv[1, :, 5:]))


@requires_cuda
def test_triton_attention_backward_batch_position_invariant():
    dtype = torch.bfloat16
    q_real, k_real, v_real = _qkv(1, 8, 8, dtype=dtype, seed=10)
    dy_real = torch.randn_like(q_real)
    op = TritonBatchInvariantAttentionOp()

    _, dq_single, dk_single, dv_single = _run_backward(
        op,
        q_real,
        k_real,
        v_real,
        dy_real,
        causal=True,
    )

    q_batch, k_batch, v_batch = _qkv(4, 8, 8, dtype=dtype, seed=11)
    dy_batch = torch.randn_like(q_batch)
    q_batch[2] = q_real[0]
    k_batch[2] = k_real[0]
    v_batch[2] = v_real[0]
    dy_batch[2] = dy_real[0]
    _, dq_batch, dk_batch, dv_batch = _run_backward(
        op,
        q_batch,
        k_batch,
        v_batch,
        dy_batch,
        causal=True,
    )

    assert torch.equal(dq_single[0], dq_batch[2])
    assert torch.equal(dk_single[0], dk_batch[2])
    assert torch.equal(dv_single[0], dv_batch[2])
