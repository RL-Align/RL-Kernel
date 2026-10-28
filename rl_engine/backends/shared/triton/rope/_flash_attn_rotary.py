# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Adapted from Dao-AILab/flash-attention, revision
# 98c1cee0ed59307833841cdbf567fe336ceec13a, flash_attn/ops/triton/rotary.py.
# Changes: explicit shared/per-sample table strides. Launch policy is in the
# sibling wrapper. See FLASH_ATTENTION_LICENSE for the retained BSD-3 terms.

import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    stride_cos_batch,
    stride_cos_seqlen,
    stride_cos_dim,
    stride_sin_batch,
    stride_sin_seqlen,
    stride_sin_dim,
    # Meta-parameters
    # We want ROTARY_DIM to be constexpr, otherwise the triton compiler doesn't know that
    # the mask is constant every 8 elements, and it will generate LDG.16 instead of LDG.128
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = (
        COS
        + pid_batch * stride_cos_batch
        + (rm_cs[:, None] * stride_cos_seqlen + rk_half[None, :] * stride_cos_dim)
    )
    SIN = (
        SIN
        + pid_batch * stride_sin_batch
        + (rm_cs[:, None] * stride_sin_seqlen + rk_half[None, :] * stride_sin_dim)
    )
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk_half[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        )
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(
            X + ROTARY_DIM_HALF * stride_x_headdim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk[None, None, :] < ROTARY_DIM)
        )
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)
