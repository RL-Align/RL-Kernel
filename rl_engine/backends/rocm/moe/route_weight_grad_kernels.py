# SPDX-License-Identifier: Apache-2.0
"""Fixed adjacent-pair partials, followed by ascending chunk accumulation."""

import triton
import triton.language as tl


@triton.jit
def _partials(
    DH,
    SILU,
    UP,
    PARTIAL,
    F: tl.constexpr,
    CHUNKS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    U0: tl.constexpr,
    U1: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    col = chunk * 256 + tl.arange(0, 256).to(tl.int64)
    mask = col < F
    dh = tl.load(DH + row * D0 + col * D1, mask, 0.0).to(tl.float32)
    silu = tl.load(SILU + row * S0 + col * S1, mask, 0.0)
    up = tl.load(UP + row * U0 + col * U1, mask, 0.0).to(tl.float32)
    up = tl.where(up < -10.0, -10.0, tl.where(up > 10.0, 10.0, up))
    # Preserve multiplication association; do not use dh * saved_unweighted.
    terms = (dh * silu) * up
    # Explicit adjacent pairs: (0+1), (2+3), ... at every level. No tl.sum
    # reduction tree chosen implicitly by layout or architecture.
    for level in tl.static_range(8):
        left, right = tl.split(tl.reshape(terms, (256 >> (level + 1), 2)))
        terms = left + right
    partial = tl.reshape(terms, ())
    tl.store(PARTIAL + row * CHUNKS + chunk, partial)


@triton.jit
def _finish(PARTIAL, DP, CHUNKS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.full((), 0.0, tl.float32)
    for chunk in range(CHUNKS):
        acc = acc + tl.load(PARTIAL + row * CHUNKS + chunk)
    tl.store(DP + row, acc)
