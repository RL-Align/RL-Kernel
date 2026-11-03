# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Task 5 Triton kernels. Serial folds, no atomics, no batch-dependent tiling.

Every long reduction walks its axis one element at a time in ascending order
with a single FP32 accumulator (``loop_unroll_factor=1`` keeps the compiler
from re-associating the fold). Launch with ``enable_fp_fusion=False`` so each
multiply and add rounds on its own. Division goes through ``tl.div_rn``
(correctly rounded); ``tl.rsqrt`` is the device's rsqrt, the same call
``torch.rsqrt`` makes on that device.
"""

import triton
import triton.language as tl


@triton.jit
def _forward(
    X,
    GAMMA,
    Y,
    RES,
    R,
    D: tl.constexpr,
    X0,
    X1,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((1,), tl.float32)
    for col in tl.range(0, D, loop_unroll_factor=1):
        value = tl.load(X + row * X0 + col * X1).to(tl.float32)
        acc = acc + value * value
    mean = tl.div_rn(acc, tl.full((1,), D, tl.float32))
    r = tl.rsqrt(mean + EPS)
    tl.store(R + row + tl.arange(0, 1), r)

    out = row * D
    for start in tl.range(0, D, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < D
        x_bf16 = tl.load(X + row * X0 + cols * X1, mask, 0.0)
        x32 = x_bf16.to(tl.float32)
        gamma = tl.load(GAMMA + cols, mask, 0.0)
        y32 = (x32 * r) * gamma
        # The only downcast of the operator; the residual keeps the input bytes.
        tl.store(Y + out + cols, y32.to(tl.bfloat16), mask)
        tl.store(RES + out + cols, x_bf16, mask)


@triton.jit
def _backward_dx(
    DY,
    DRES,
    X,
    GAMMA,
    R,
    DX,
    D: tl.constexpr,
    DY0,
    DY1,
    DR0,
    DR1,
    X0,
    X1,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    r = tl.load(R + row + tl.arange(0, 1))
    q = tl.zeros((1,), tl.float32)
    for col in tl.range(0, D, loop_unroll_factor=1):
        dy = tl.load(DY + row * DY0 + col * DY1).to(tl.float32)
        gamma = tl.load(GAMMA + col)
        x = tl.load(X + row * X0 + col * X1).to(tl.float32)
        q = q + (dy * gamma) * x
    r3 = (r * r) * r
    d32 = tl.full((1,), D, tl.float32)

    out = row * D
    for start in tl.range(0, D, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < D
        dy = tl.load(DY + row * DY0 + cols * DY1, mask, 0.0).to(tl.float32)
        dres = tl.load(DRES + row * DR0 + cols * DR1, mask, 0.0).to(tl.float32)
        x = tl.load(X + row * X0 + cols * X1, mask, 0.0).to(tl.float32)
        gamma = tl.load(GAMMA + cols, mask, 0.0)
        u = dy * gamma
        rhs = tl.div_rn((x * r3) * q, d32)
        dx_norm = (r * u) - rhs
        tl.store(DX + out + cols, dx_norm + dres, mask)


@triton.jit
def _backward_dgamma(
    DY,
    X,
    R,
    DGAMMA,
    T,
    D: tl.constexpr,
    DY0,
    DY1,
    X0,
    X1,
    BLOCK: tl.constexpr,
):
    cols = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < D
    acc = tl.zeros((BLOCK,), tl.float32)
    for row in tl.range(0, T, loop_unroll_factor=1):
        dy = tl.load(DY + row * DY0 + cols * DY1, mask, 0.0).to(tl.float32)
        x = tl.load(X + row * X0 + cols * X1, mask, 0.0).to(tl.float32)
        r = tl.load(R + row)
        acc = acc + (dy * x) * r
    tl.store(DGAMMA + cols, acc, mask)
