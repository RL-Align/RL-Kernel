# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_N = 64


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _standard_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    lse_ptr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    S_Q: tl.constexpr,
    S_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_KEY_PADDING_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    kv_head = q_head // (H_Q // H_KV)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(
        q_ptr + batch * stride_qb + q_head * stride_qh + row * stride_qs + offs_d * stride_qd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    # Padding must not change which reduction lane owns a logical KV token.
    # With physical columns, left padding shifts every valid value to another
    # lane and changes the floating-point reduction tree even though the mask
    # is semantically correct.  Find the first valid physical column and run
    # both softmax passes in logical-column order.  C2 fixtures require one
    # contiguous valid interval (left or right padding), so this also preserves
    # the causal position of every restored logical token.
    valid_start = 0
    if HAS_KEY_PADDING_MASK:
        valid_start = S_KV
        for start_n in range(0, S_KV, BLOCK_N):
            probe_cols = start_n + tl.arange(0, BLOCK_N)
            probe_in_bounds = probe_cols < S_KV
            probe_keep = tl.load(
                mask_ptr + batch * S_KV + probe_cols,
                mask=probe_in_bounds,
                other=0,
            )
            block_first = tl.min(
                tl.where(probe_in_bounds & (probe_keep != 0), probe_cols, S_KV),
                axis=0,
            )
            valid_start = tl.minimum(valid_start, block_first)
    logical_row = row - valid_start

    max_score = -float("inf")
    for start_n in range(0, S_KV, BLOCK_N):
        logical_cols = start_n + tl.arange(0, BLOCK_N)
        cols = valid_start + logical_cols
        col_mask = cols < S_KV
        k = tl.load(
            k_ptr
            + batch * stride_kb
            + kv_head * stride_kh
            + cols[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(col_mask, scores, -float("inf"))

        if CAUSAL:
            causal_keep = logical_cols <= (logical_row + S_KV - S_Q)
            scores = tl.where(causal_keep, scores, -float("inf"))

        if HAS_KEY_PADDING_MASK:
            keep = tl.load(mask_ptr + batch * S_KV + cols, mask=col_mask, other=0)
            scores = tl.where(keep != 0, scores, -float("inf"))

        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    denom = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start_n in range(0, S_KV, BLOCK_N):
        logical_cols = start_n + tl.arange(0, BLOCK_N)
        cols = valid_start + logical_cols
        col_mask = cols < S_KV
        k = tl.load(
            k_ptr
            + batch * stride_kb
            + kv_head * stride_kh
            + cols[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(col_mask, scores, -float("inf"))

        if CAUSAL:
            causal_keep = logical_cols <= (logical_row + S_KV - S_Q)
            scores = tl.where(causal_keep, scores, -float("inf"))

        if HAS_KEY_PADDING_MASK:
            keep = tl.load(mask_ptr + batch * S_KV + cols, mask=col_mask, other=0)
            scores = tl.where(keep != 0, scores, -float("inf"))

        probs = tl.exp(scores - max_score)
        probs = tl.where(max_score == -float("inf"), 0.0, probs)
        denom += tl.sum(probs, axis=0)

        v = tl.load(
            v_ptr
            + batch * stride_vb
            + kv_head * stride_vh
            + cols[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v, axis=0)

    out = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(
        out_ptr + batch * stride_ob + q_head * stride_oh + row * stride_os + offs_d * stride_od,
        out,
        mask=d_mask,
    )
    tl.store(lse_ptr + (batch * H_Q + q_head) * S_Q + row, max_score + tl.log(denom))


@triton.jit
def _find_valid_start(
    mask_ptr,
    batch,
    S_KV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_KEY_PADDING_MASK: tl.constexpr,
):
    valid_start = 0
    if HAS_KEY_PADDING_MASK:
        valid_start = S_KV
        for start_n in range(0, S_KV, BLOCK_N):
            probe_cols = start_n + tl.arange(0, BLOCK_N)
            probe_in_bounds = probe_cols < S_KV
            probe_keep = tl.load(
                mask_ptr + batch * S_KV + probe_cols,
                mask=probe_in_bounds,
                other=0,
            )
            block_first = tl.min(
                tl.where(probe_in_bounds & (probe_keep != 0), probe_cols, S_KV),
                axis=0,
            )
            valid_start = tl.minimum(valid_start, block_first)
    return valid_start


@triton.jit
def _standard_attn_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    mask_ptr,
    dq_ptr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    S_Q: tl.constexpr,
    S_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dos: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqs: tl.constexpr,
    stride_dqd: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_KEY_PADDING_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    kv_head = q_head // (H_Q // H_KV)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(
        q_ptr + batch * stride_qb + q_head * stride_qh + row * stride_qs + offs_d * stride_qd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + batch * stride_dob + q_head * stride_doh + row * stride_dos + offs_d * stride_dod,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    lse = tl.load(lse_ptr + (batch * H_Q + q_head) * S_Q + row)
    delta = tl.load(delta_ptr + (batch * H_Q + q_head) * S_Q + row)
    row_valid = (lse == lse) & (lse != -float("inf"))

    valid_start = _find_valid_start(mask_ptr, batch, S_KV, BLOCK_N, HAS_KEY_PADDING_MASK)
    logical_row = row - valid_start

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start_n in range(0, S_KV, BLOCK_N):
        logical_cols = start_n + tl.arange(0, BLOCK_N)
        cols = valid_start + logical_cols
        col_mask = cols < S_KV
        k = tl.load(
            k_ptr
            + batch * stride_kb
            + kv_head * stride_kh
            + cols[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr
            + batch * stride_vb
            + kv_head * stride_vh
            + cols[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        keep = col_mask
        if CAUSAL:
            keep = keep & (logical_cols <= (logical_row + S_KV - S_Q))
        if HAS_KEY_PADDING_MASK:
            pad_keep = tl.load(mask_ptr + batch * S_KV + cols, mask=col_mask, other=0)
            keep = keep & (pad_keep != 0)
        probs = tl.exp(scores - lse)
        probs = tl.where(keep & row_valid, probs, 0.0)
        dprob = tl.sum(do[None, :] * v, axis=1)
        dscore = probs * (dprob - delta)
        acc += tl.sum(dscore[:, None] * k, axis=0)

    tl.store(
        dq_ptr + batch * stride_dqb + q_head * stride_dqh + row * stride_dqs + offs_d * stride_dqd,
        acc * sm_scale,
        mask=d_mask,
    )


@triton.jit
def _standard_attn_dkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    mask_ptr,
    dk_ptr,
    dv_ptr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    S_Q: tl.constexpr,
    S_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dos: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dks: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvs: tl.constexpr,
    stride_dvd: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_KEY_PADDING_MASK: tl.constexpr,
):
    col = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    valid_start = _find_valid_start(mask_ptr, batch, S_KV, BLOCK_N, HAS_KEY_PADDING_MASK)
    logical_col = col - valid_start
    col_keep = col < S_KV
    if HAS_KEY_PADDING_MASK:
        pad_keep = tl.load(mask_ptr + batch * S_KV + col)
        col_keep = col_keep & (pad_keep != 0) & (col >= valid_start)

    k = tl.load(
        k_ptr + batch * stride_kb + kv_head * stride_kh + col * stride_ks + offs_d * stride_kd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + batch * stride_vb + kv_head * stride_vh + col * stride_vs + offs_d * stride_vd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    acc_dk = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_D,), dtype=tl.float32)
    group = H_Q // H_KV
    for gi in range(0, group):
        q_head = kv_head * group + gi
        for row in range(0, S_Q):
            logical_row = row - valid_start
            row_keep = col_keep
            if CAUSAL:
                row_keep = row_keep & (logical_col <= (logical_row + S_KV - S_Q))
            q = tl.load(
                q_ptr
                + batch * stride_qb
                + q_head * stride_qh
                + row * stride_qs
                + offs_d * stride_qd,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                do_ptr
                + batch * stride_dob
                + q_head * stride_doh
                + row * stride_dos
                + offs_d * stride_dod,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(lse_ptr + (batch * H_Q + q_head) * S_Q + row)
            delta = tl.load(delta_ptr + (batch * H_Q + q_head) * S_Q + row)
            row_valid = (lse == lse) & (lse != -float("inf"))
            score = tl.sum(q * k, axis=0) * sm_scale
            prob = tl.exp(score - lse)
            keep = row_keep & row_valid
            prob = tl.where(keep, prob, 0.0)
            dprob = tl.sum(do * v, axis=0)
            dscore = prob * (dprob - delta)
            acc_dk += dscore * q
            acc_dv += prob * do

    tl.store(
        dk_ptr + batch * stride_dkb + kv_head * stride_dkh + col * stride_dks + offs_d * stride_dkd,
        acc_dk * sm_scale,
        mask=d_mask,
    )
    tl.store(
        dv_ptr + batch * stride_dvb + kv_head * stride_dvh + col * stride_dvs + offs_d * stride_dvd,
        acc_dv,
        mask=d_mask,
    )


class _TritonBatchInvariantAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        causal: bool,
        scale: float,
        return_lse: bool,
        output_fp32: bool,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.contiguous()

        batch, q_heads, q_len, head_dim = q.shape
        kv_batch, kv_heads, kv_len, k_dim = k.shape
        if v.shape != k.shape:
            raise ValueError("k and v must have the same shape")
        if kv_batch != batch:
            raise ValueError("q, k, and v must have the same batch size")
        if k_dim != head_dim:
            raise ValueError("q, k, and v must have the same head dimension")
        if q_heads % kv_heads != 0:
            raise ValueError("q heads must be divisible by k/v heads for GQA/MQA")
        if head_dim > 256:
            raise ValueError("Triton batch-invariant attention supports head_dim <= 256")
        if key_padding_mask is not None and key_padding_mask.shape != (batch, kv_len):
            raise ValueError("key_padding_mask must have shape [batch, key_seq_len]")

        out = torch.empty_like(q, dtype=torch.float32 if output_fp32 else q.dtype)
        lse = torch.empty((batch, q_heads, q_len), device=q.device, dtype=torch.float32)
        block_d = _next_power_of_2(head_dim)
        grid = (q_len, q_heads, batch)
        dummy_mask = (
            key_padding_mask
            if key_padding_mask is not None
            else q.new_empty((1,), dtype=torch.bool)
        )

        _standard_attn_fwd_kernel[grid](
            q,
            k,
            v,
            dummy_mask,
            out,
            lse,
            batch,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            scale,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=block_d,
            CAUSAL=causal,
            HAS_KEY_PADDING_MASK=key_padding_mask is not None,
            num_warps=8,
        )

        mask_for_save = (
            key_padding_mask
            if key_padding_mask is not None
            else q.new_empty((0,), dtype=torch.bool)
        )
        ctx.save_for_backward(q, k, v, out, lse, mask_for_save)
        ctx.causal = causal
        ctx.scale = scale
        ctx.has_key_padding_mask = key_padding_mask is not None
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        q, k, v, out, lse, mask_for_save = ctx.saved_tensors
        grad_out = grad_outputs[0].contiguous()
        key_padding_mask = mask_for_save if ctx.has_key_padding_mask else None
        batch, q_heads, q_len, head_dim = q.shape
        kv_heads, kv_len = k.shape[1], k.shape[2]
        block_d = _next_power_of_2(head_dim)
        dummy_mask = (
            key_padding_mask
            if key_padding_mask is not None
            else q.new_empty((1,), dtype=torch.bool)
        )
        delta = (grad_out.float() * out.float()).sum(dim=-1).contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        common = dict(
            B=batch,
            H_Q=q_heads,
            H_KV=kv_heads,
            S_Q=q_len,
            S_KV=kv_len,
            D=head_dim,
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qs=q.stride(2),
            stride_qd=q.stride(3),
            stride_kb=k.stride(0),
            stride_kh=k.stride(1),
            stride_ks=k.stride(2),
            stride_kd=k.stride(3),
            sm_scale=float(ctx.scale),
            BLOCK_N=_BLOCK_N,
            BLOCK_D=block_d,
            CAUSAL=ctx.causal,
            HAS_KEY_PADDING_MASK=ctx.has_key_padding_mask,
            num_warps=8,
        )
        _standard_attn_dq_kernel[(q_len, q_heads, batch)](
            q,
            k,
            v,
            grad_out,
            delta,
            lse,
            dummy_mask,
            dq,
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vs=v.stride(2),
            stride_vd=v.stride(3),
            stride_dob=grad_out.stride(0),
            stride_doh=grad_out.stride(1),
            stride_dos=grad_out.stride(2),
            stride_dod=grad_out.stride(3),
            stride_dqb=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dqs=dq.stride(2),
            stride_dqd=dq.stride(3),
            **common,
        )
        _standard_attn_dkv_kernel[(kv_len, kv_heads, batch)](
            q,
            k,
            v,
            grad_out,
            delta,
            lse,
            dummy_mask,
            dk,
            dv,
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vs=v.stride(2),
            stride_vd=v.stride(3),
            stride_dob=grad_out.stride(0),
            stride_doh=grad_out.stride(1),
            stride_dos=grad_out.stride(2),
            stride_dod=grad_out.stride(3),
            stride_dkb=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dks=dk.stride(2),
            stride_dkd=dk.stride(3),
            stride_dvb=dv.stride(0),
            stride_dvh=dv.stride(1),
            stride_dvs=dv.stride(2),
            stride_dvd=dv.stride(3),
            **common,
        )
        return dq, dk, dv, None, None, None, None, None


def _resolve_scale(
    head_dim: int,
    *,
    scale: Optional[float],
    softmax_scale: Optional[float],
) -> float:
    if scale is not None and softmax_scale is not None:
        raise ValueError("set only one of scale or softmax_scale")
    value = scale if scale is not None else softmax_scale
    return float(value) if value is not None else 1.0 / math.sqrt(head_dim)


def triton_batch_invariant_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale_value = _resolve_scale(q.shape[-1], scale=scale, softmax_scale=softmax_scale)
    return _TritonBatchInvariantAttention.apply(
        q,
        k,
        v,
        key_padding_mask,
        causal,
        scale_value,
        False,
        False,
    )


def triton_batch_invariant_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale_value = _resolve_scale(q.shape[-1], scale=scale, softmax_scale=softmax_scale)
    return _TritonBatchInvariantAttention.apply(
        q, k, v, key_padding_mask, causal, scale_value, False, True
    )


def triton_batch_invariant_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_value = _resolve_scale(q.shape[-1], scale=scale, softmax_scale=softmax_scale)
    return _TritonBatchInvariantAttention.apply(
        q,
        k,
        v,
        key_padding_mask,
        causal,
        scale_value,
        True,
        False,
    )


class TritonBatchInvariantAttentionOp:
    """Triton standard-softmax attention with fixed key-order reductions."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        return self.forward(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            softmax_scale=softmax_scale,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if dropout_p != 0.0:
            raise ValueError("batch-invariant attention does not support dropout")
        return triton_batch_invariant_attention(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            softmax_scale=softmax_scale,
            key_padding_mask=key_padding_mask,
        )

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if dropout_p != 0.0:
            raise ValueError("batch-invariant attention does not support dropout")
        return triton_batch_invariant_attention_fp32(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            softmax_scale=softmax_scale,
            key_padding_mask=key_padding_mask,
        )
