# SPDX-License-Identifier: Apache-2.0
"""Packed E2M1/E8M0 expert projection with original BF16 activation operands."""

import torch
import triton
import triton.language as tl

from rl_engine.backends.rocm.moe.packed_mx import _backward
from rl_engine.kernels.ops.triton.moe.fused_mlp import (
    _block_prefix,
    _decode_e2m1,
    _e8m0_scale,
    _grid_blocks,
    _locate_token_block,
    _search_steps,
)


@triton.jit
def _forward(
    X,
    W,
    S,
    OFF,
    PREFIX,
    GAMMA,
    Y,
    RAW,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    SAVE_RAW: tl.constexpr,
):
    e, start, end = _locate_token_block(OFF, PREFIX, E, tl.program_id(0), 16, SEARCH)
    if start >= end:
        return
    rows = start.to(tl.int64) + tl.arange(0, 16)
    cols = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(K // 32):
        x = tl.load(X + rows[:, None] * K + block * 32 + kk[None, :], rows[:, None] < end, other=0)
        packed = tl.load(
            W + (e.to(tl.int64) * N + cols[None, :]) * (K // 2) + block * 16 + kk[:, None] // 2,
            cols[None, :] < N,
            other=0,
        )
        nib = tl.where(kk[:, None] % 2 == 0, packed & 15, packed >> 4)
        scale = _e8m0_scale(
            tl.load(S + (e.to(tl.int64) * N + cols) * (K // 32) + block, cols < N, other=127)
        )
        weight = (_decode_e2m1(nib) * scale[None, :]).to(tl.bfloat16)
        acc = tl.dot(x, weight, acc)
    if SAVE_RAW:
        tl.store(
            RAW + rows[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < end) & (cols[None, :] < N),
        )
    if HAS_GAMMA:
        gamma = tl.load(GAMMA + rows, rows < end, other=0)
        acc = acc * gamma[:, None]
    tl.store(
        Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < end) & (cols[None, :] < N)
    )


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scales, offsets, gammas):
        if (
            x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or codes.dtype != torch.uint8
            or scales.dtype != torch.uint8
        ):
            raise ValueError(
                "W4A16 requires BF16 activations and original packed uint8 codes/scales"
            )
        e, n, half_k = codes.shape
        k = half_k * 2
        if x.shape[1] != k or tuple(scales.shape) != (e, n, k // 32) or offsets.shape != (e + 1,):
            raise ValueError("original block32 expert geometry mismatch")
        y = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
        save_raw = gammas is not None and ctx.needs_input_grad[4]
        raw = torch.empty_like(y, dtype=torch.float32) if save_raw else y
        if x.shape[0]:
            _forward[(_grid_blocks(e, x.shape[0], 16), triton.cdiv(n, 32))](
                x,
                codes,
                scales,
                offsets,
                _block_prefix(offsets, 16),
                x if gammas is None else gammas,
                y,
                raw,
                k,
                n,
                e,
                _search_steps(e),
                gammas is not None,
                save_raw,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(codes, scales, offsets, raw, *(() if gammas is None else (gammas,)))
        ctx.geometry = (x.shape[0], k, n, e)
        return y

    @staticmethod
    def backward(ctx, dy):
        codes, scales, offsets, raw, *gamma = ctx.saved_tensors
        m, k, n, e = ctx.geometry
        dgamma = (dy.float() * raw).sum(-1) if gamma and ctx.needs_input_grad[4] else None
        scaled = (dy.float() * gamma[0][:, None]).bfloat16() if gamma else dy
        dx = torch.empty((m, k), device=dy.device, dtype=torch.float32)
        if m:
            _backward[(_grid_blocks(e, m, 16), triton.cdiv(k, 32))](
                scaled.contiguous(),
                codes,
                scales,
                offsets,
                _block_prefix(offsets, 16),
                dx,
                k,
                n,
                e,
                _search_steps(e),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx.bfloat16(), None, None, None, dgamma


def packed_w4a16(x, codes, scales, offsets, gammas=None):
    if not torch.version.hip or not x.is_cuda:
        raise RuntimeError("original W4A16 projection requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    if arch != "gfx942":
        raise ValueError("the original W4A16 profile is validated on gfx942 only")
    if not codes.is_contiguous() or not scales.is_contiguous():
        raise ValueError("original packed expert codes/scales must be contiguous")
    if offsets.dtype != torch.int64 or offsets.device != x.device:
        raise ValueError("expert offsets must be colocated int64")
    if gammas is not None and (
        gammas.shape != (x.shape[0],) or gammas.dtype != torch.float32 or not gammas.is_contiguous()
    ):
        raise ValueError("route gammas must be contiguous original FP32 row weights")
    return _Projection.apply(x, codes, scales, offsets, gammas)
