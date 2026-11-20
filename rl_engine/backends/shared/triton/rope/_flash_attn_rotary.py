# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Adapted from Dao-AILab/flash-attention, revision
# 98c1cee0ed59307833841cdbf567fe336ceec13a, flash_attn/ops/triton/rotary.py.
# Specialized for full-head adjacent pairs, with explicit table strides and
# sign-bit conjugation. Launch policy is in the sibling wrapper.
# See FLASH_ATTENTION_LICENSE for the retained BSD-3 terms.

import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    seqlen,
    nheads,
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
    HEAD_DIM: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, HEAD_DIM // 2)
    valid_rows = rm < seqlen

    COS = (
        COS
        + pid_batch * stride_cos_batch
        + rm[:, None] * stride_cos_seqlen
        + rk_half[None, :] * stride_cos_dim
    )
    SIN = (
        SIN
        + pid_batch * stride_sin_batch
        + rm[:, None] * stride_sin_seqlen
        + rk_half[None, :] * stride_sin_dim
    )
    cos = tl.load(COS, mask=valid_rows[:, None], other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=valid_rows[:, None], other=0.0).to(tl.float32)
    if CONJUGATE:
        # Triton 3.5 lowers unary minus to 0 - x, which loses the sign of +0.
        # Flip the FP32 sign bit to match CUDA's transpose, including +/-0.
        sin = (sin.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)

    rk = tl.arange(0, HEAD_DIM)
    X = (
        X
        + pid_batch * stride_x_batch
        + rh[:, None, None] * stride_x_nheads
        + rm[None, :, None] * stride_x_seqlen
        + rk[None, None, :] * stride_x_headdim
    )
    OUT = (
        OUT
        + pid_batch * stride_out_batch
        + rh[:, None, None] * stride_out_nheads
        + rm[None, :, None] * stride_out_seqlen
        + rk[None, None, :] * stride_out_headdim
    )
    mask = (rh[:, None, None] < nheads) & valid_rows[None, :, None]
    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, HEAD_DIM // 2, 2]))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    tl.store(OUT, tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, HEAD_DIM]), mask=mask)
