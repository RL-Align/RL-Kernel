# SPDX-License-Identifier: Apache-2.0
"""Gather and sequentially add Top-6 rows without a full restored tensor."""

import triton
import triton.language as tl


@triton.jit
def _combine(
    EXPERT,
    ROWS,
    SHARED,
    OUT,
    STEPS,
    BEFORE,
    H: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    DEBUG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = col < H
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for slot in tl.static_range(6):
        row = tl.load(ROWS + token * 6 + slot).to(tl.int64)
        value = tl.load(EXPERT + row * E0 + col * E1, mask, 0.0).to(tl.float32)
        acc = acc + value
        if DEBUG:
            tl.store(STEPS + (token * 6 + slot) * H + col, acc, mask)
    if HAS_SHARED:
        shared = tl.load(SHARED + token * S0 + col * S1, mask, 0.0).to(tl.float32)
        acc = acc + shared
    if DEBUG:
        tl.store(BEFORE + token * H + col, acc, mask)
    # One final round-to-nearest-even conversion; no earlier BF16 rounding.
    output = acc.to(OUT.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(OUT + token * H + col, output, mask)


@triton.jit
def _output_vjp(
    DY, TOKENS, DX, H: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    token = tl.load(TOKENS + row).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    value = tl.load(DY + token * D0 + col * D1, col < H, 0.0)
    tl.store(DX + row * H + col, value, col < H)
