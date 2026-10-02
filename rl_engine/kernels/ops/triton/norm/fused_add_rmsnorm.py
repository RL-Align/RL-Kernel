# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 arithmetic prototype for fused residual addition and RMSNorm.

The caller supplies contiguous x/residual tensors viewed as [rows, n_cols],
a contiguous weight vector [n_cols], and FP32 y/updated_residual output buffers.
Launch one program per row, with a fixed power-of-two BLOCK_SIZE >= n_cols > 0,
fixed num_warps, and enable_fp_fusion=False.

Backward also launches one program per row and recomputes the same statistics.
It writes input gradients and FP32 weight-gradient contributions [rows, n_cols].
A second kernel left-folds those contributions in ascending row order, matching
the accumulation order of the existing reduce_rows_fp32 helper. Both upstream
gradient buffers are required; supply zeros for an unused output branch.
Gradient output buffers select the final storage dtypes. Use the same stream
for both backward launches, and disable FP fusion for both kernels.

This prototype normalizes the unrounded FP32 residual sum. Input/output dtype
support and any BF16 residual rounding point still need a model-level contract
before adding the public wrapper.
"""

import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    updated_residual_ptr,
    n_cols: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute u = x + residual and y = u * rsqrt(mean(u**2) + EPS) * weight."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 1. Add the residual, keeping the intermediate sum in FP32 for now.
    updated_residual = x + residual

    # 2. Reduce across this row only; masked columns contribute zero.
    sum_squares = tl.sum(updated_residual * updated_residual, axis=0, keep_dims=True)
    mean_square = tl.div_rn(sum_squares, n_cols)
    inverse_rms = tl.rsqrt(mean_square + EPS)

    # 3. Normalize each element, then apply its feature weight.
    normalized = updated_residual * inverse_rms
    y = normalized * weight

    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(updated_residual_ptr + offsets, updated_residual, mask=mask)


@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    grad_y_ptr,
    grad_updated_residual_output_ptr,
    grad_x_ptr,
    grad_residual_ptr,
    grad_weight_per_row_ptr,
    n_cols: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute input gradients and each row's FP32 weight-gradient contribution."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_updated_residual_output = tl.load(
        grad_updated_residual_output_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    # Recompute the forward intermediates in the same order as the forward kernel.
    updated_residual = x + residual
    sum_squares = tl.sum(updated_residual * updated_residual, axis=0, keep_dims=True)
    mean_square = tl.div_rn(sum_squares, n_cols)
    inverse_rms = tl.rsqrt(mean_square + EPS)
    normalized = updated_residual * inverse_rms

    # 1. y = normalized * weight. Save this row's weight-gradient contribution.
    grad_normalized = grad_y * weight
    grad_weight_per_row = grad_y * normalized

    # 2. normalized = updated_residual * inverse_rms: follow both input paths.
    grad_updated_residual_direct = grad_normalized * inverse_rms
    grad_inverse_rms = tl.sum(grad_normalized * updated_residual, axis=0, keep_dims=True)

    # 3. inverse_rms = (mean_square + EPS) ** (-0.5).
    inverse_rms_cubed = inverse_rms * inverse_rms * inverse_rms
    grad_mean_square = grad_inverse_rms * (-0.5 * inverse_rms_cubed)

    # 4. mean_square = sum_squares / n_cols.
    grad_sum_squares = tl.div_rn(grad_mean_square, n_cols)

    # 5. The sum's backward broadcasts this per-row gradient over all columns.
    grad_squared = grad_sum_squares

    # 6. squared = updated_residual * updated_residual.
    grad_updated_residual_via_rms = grad_squared * (2.0 * updated_residual)

    # 7. Combine both paths from y and the separate residual-output gradient.
    grad_updated_residual_from_y = grad_updated_residual_direct + grad_updated_residual_via_rms
    grad_updated_residual_total = grad_updated_residual_from_y + grad_updated_residual_output

    # 8. updated_residual = x + residual: both inputs receive this gradient.
    grad_x = grad_updated_residual_total
    grad_residual = grad_updated_residual_total

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_residual_ptr + offsets, grad_residual, mask=mask)
    tl.store(grad_weight_per_row_ptr + offsets, grad_weight_per_row, mask=mask)


@triton.jit
def _fused_add_rmsnorm_bwd_weight_kernel(
    grad_weight_per_row_ptr,
    grad_weight_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Left-fold FP32 row contributions; launch ceil(n_cols / BLOCK_SIZE) programs.

    Each program owns a block of columns. Rows are accumulated sequentially,
    with no atomics or row-count-dependent reduction tree. Zero rows yield zero.
    """
    block = tl.program_id(0).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    grad_weight = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    offsets = cols
    for _ in range(n_rows):
        contribution = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_weight = grad_weight + contribution
        offsets = offsets + n_cols

    tl.store(grad_weight_ptr + cols, grad_weight, mask=mask)
