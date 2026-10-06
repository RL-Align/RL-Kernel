# SPDX-License-Identifier: Apache-2.0
"""Triton FP32 tree reductions with fixed tiles and no atomic reductions."""

import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib

from ..timestep_embed_mlp import TimestepEmbedMLPOp


@tr.jit
def _compensated_add(ah, al, bh, bl):
    s = ah + bh
    v = s - ah
    error = (ah - (s - v)) + (bh - v)
    t = (al + bl) + error
    high = s + t
    low = t - (high - s)
    return high, low


@tr.jit
def _timestep_mm_short(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    out = tl.program_id(0) * 256 + tl.arange(0, 256)
    row, col = out // N, out % N
    acc = tl.full((256,), 0, tl.float32)
    for k in range(K):
        a = tl.load(A + row * A0 + k * A1, row < M, 0)
        b = tl.load(B + k * B0 + col * B1, row < M, 0)
        acc = tl.fma(a, b, acc)
    tl.store(C + out, acc, row < M)


@tr.jit
def _timestep_mm_tree(
    A,
    B,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * 4 + tl.arange(0, 4)
    k = tl.arange(0, BLOCK_K)
    a = tl.load(A + row * A0 + k * A1, k < K, 0)
    b = tl.load(B + k[None, :] * B0 + col[:, None] * B1, (k[None, :] < K) & (col[:, None] < N), 0)
    product = a[None, :] * b
    residual = tl.fma(a[None, :], b, -product)
    high, low = tl.reduce((product, residual), 1, _compensated_add)
    acc = high + low
    tl.store(C + row * N + col, acc, col < N)


@tr.jit
def _timestep_embedding(T, F, E, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    row, col = i // 128, i % 128
    t = tl.load(T + row, row < N, 0)
    f = tl.load(F + col)
    p = (t * f) * 1000.0
    tl.store(E + row * 256 + col, lib.cos(p), row < N)
    tl.store(E + row * 256 + 128 + col, lib.sin(p), row < N)


@tr.jit
def _timestep_silu(X, Y, N: tl.constexpr, DERIVATIVE: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    x = tl.load(X + i, i < N, 0)
    s = lib.div_rn(1.0, 1.0 + lib.exp(-x))
    y = s + (x * s) * (1.0 - s) if DERIVATIVE else x * s
    tl.store(Y + i, y, i < N)


@tr.jit
def _timestep_dt(DE, E, F, DT):
    row = tl.program_id(0)
    i = tl.arange(0, 128)
    dc, ds = tl.load(DE + row * 256 + i), tl.load(DE + row * 256 + 128 + i)
    c, s = tl.load(E + row * 256 + i), tl.load(E + row * 256 + 128 + i)
    f = tl.load(F + i)
    terms = ((-dc * s + ds * c) * 1000.0) * f
    high, low = tl.reduce((terms, tl.full((128,), 0.0, tl.float32)), 0, _compensated_add)
    dt = high + low
    tl.store(DT + row, dt)


class TritonPrimitives:
    def mm(self, a, b, trace):
        trace.kernels.append("triton._timestep_mm")
        m, k = a.shape
        n = b.shape[1]
        out = torch.empty((m, n), device=a.device, dtype=torch.float32)
        if m and n:
            if k <= 32:
                _timestep_mm_short[(tr.cdiv(m * n, 256),)](
                    a, b, out, m, n, k, *a.stride(), *b.stride(), enable_fp_fusion=False
                )
            else:
                if b.stride(0) != 1:
                    b = b.T.contiguous().T
                _timestep_mm_tree[(m, tr.cdiv(n, 4))](
                    a,
                    b,
                    out,
                    n,
                    k,
                    *a.stride(),
                    *b.stride(),
                    tr.next_power_of_2(k),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return out

    def embedding(self, t, f, trace):
        trace.kernels.append("triton._timestep_embedding")
        out = torch.empty((t.numel(), 256), device=t.device, dtype=torch.float32)
        if t.numel():
            _timestep_embedding[(tr.cdiv(t.numel() * 128, 256),)](
                t, f, out, t.numel(), enable_fp_fusion=False
            )
        return out

    def unary(self, x, derivative, trace):
        trace.kernels.append(
            "triton._timestep_silu_grad" if derivative else "triton._timestep_silu"
        )
        out = torch.empty_like(x)
        if x.numel():
            _timestep_silu[(tr.cdiv(x.numel(), 256),)](
                x, out, x.numel(), derivative, enable_fp_fusion=False
            )
        return out

    def timestep_grad(self, de, e, f, trace):
        trace.kernels.append("triton._timestep_dt")
        out = torch.empty((de.shape[0],), device=de.device, dtype=torch.float32)
        if out.numel():
            _timestep_dt[(out.numel(),)](de, e, f, out, enable_fp_fusion=False)
        return out


class TritonTimestepEmbedMLPOp(TimestepEmbedMLPOp):
    def __init__(self, *, allow_fallback=False):
        super().__init__("triton", allow_fallback=allow_fallback)
