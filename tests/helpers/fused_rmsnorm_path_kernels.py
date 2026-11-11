# SPDX-License-Identifier: Apache-2.0
"""Expose production row helpers for fixed-path checks without changing public APIs."""

import triton
import triton.language as tl

from rl_engine.backends.shared.triton.norm.fused_add_rmsnorm import (
    _rmsnorm_row_correction,
    _rmsnorm_row_statistics,
)


@triton.jit
def row_statistics_probe(
    updated_ptr,
    sums_ptr,
    inverse_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    updated = tl.load(updated_ptr + row * N_COLS + cols, mask=cols < N_COLS, other=0.0)
    squared_sum, inverse = _rmsnorm_row_statistics(updated, N_COLS, EPS)
    tl.store(sums_ptr + row, squared_sum)
    tl.store(inverse_ptr + row, inverse)


@triton.jit
def row_correction_probe(
    grad_normalized_ptr,
    normalized_ptr,
    sums_ptr,
    correction_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * N_COLS + cols
    grad_normalized = tl.load(grad_normalized_ptr + offsets, mask=cols < N_COLS, other=0.0)
    normalized = tl.load(normalized_ptr + offsets, mask=cols < N_COLS, other=0.0)
    total, correction = _rmsnorm_row_correction(grad_normalized, normalized, N_COLS)
    tl.store(sums_ptr + row, total)
    tl.store(correction_ptr + row, correction)
