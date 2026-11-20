# SPDX-License-Identifier: Apache-2.0
"""Row-local task 46 kernels. No reductions, atomics, or batch-dependent tiling."""

import triton
import triton.language as tl


@triton.jit
def _forward(
    G,
    U,
    P,
    H,
    SIG,
    SILU,
    RAW,
    F: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    U0: tl.constexpr,
    U1: tl.constexpr,
    P0: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = col < F
    g = tl.load(G + row * G0 + col * G1, mask, 0.0).to(tl.float32)
    u = tl.load(U + row * U0 + col * U1, mask, 0.0).to(tl.float32)
    if HAS_WEIGHT:
        g = tl.where(g > 10.0, 10.0, g)
        u = tl.where(u < -10.0, -10.0, tl.where(u > 10.0, 10.0, u))
    sig = tl.div_rn(1.0, 1.0 + tl.exp(-g))
    silu = g * sig
    raw = silu * u
    value = raw
    if HAS_WEIGHT:
        value = raw * tl.load(P + row * P0)
    off = row * F + col
    tl.store(SIG + off, sig, mask)
    tl.store(SILU + off, silu, mask)
    tl.store(RAW + off, raw, mask)
    # Only the final weighted activation is rounded to BF16.
    tl.store(H + off, value, mask)


@triton.jit
def _backward(
    DH,
    G,
    U,
    P,
    SIG,
    SILU,
    DG,
    DU,
    F: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    U0: tl.constexpr,
    U1: tl.constexpr,
    P0: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = col < F
    g0 = tl.load(G + row * G0 + col * G1, mask, 0.0).to(tl.float32)
    u0 = tl.load(U + row * U0 + col * U1, mask, 0.0).to(tl.float32)
    dy = tl.load(DH + row * D0 + col * D1, mask, 0.0).to(tl.float32)
    g, u = g0, u0
    if HAS_WEIGHT:
        g = tl.where(g0 > 10.0, 10.0, g0)
        u = tl.where(u0 < -10.0, -10.0, tl.where(u0 > 10.0, 10.0, u0))
        dy = dy * tl.load(P + row * P0)
    off = row * F + col
    sig = tl.load(SIG + off, mask, 0.0)
    silu = tl.load(SILU + off, mask, 0.0)
    dsilu = sig * (1.0 + g * (1.0 - sig))
    dg = (dy * u) * dsilu
    du = dy * silu
    if HAS_WEIGHT:
        dg = dg * (g0 < 10.0).to(tl.float32)
        du = du * ((u0 > -10.0) & (u0 < 10.0)).to(tl.float32)
    tl.store(DG + off, dg, mask)
    tl.store(DU + off, du, mask)
