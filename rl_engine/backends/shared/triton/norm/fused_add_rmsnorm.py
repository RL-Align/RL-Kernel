# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 arithmetic for fused residual addition and RMSNorm.

The caller supplies contiguous x/residual tensors viewed as [rows, n_cols],
a contiguous weight vector [n_cols], and FP32 y/updated_residual output buffers.
Forward also saves one FP32 inverse_rms per row for backward.
Launch one program per row, with a fixed power-of-two BLOCK_SIZE >= n_cols > 0,
fixed num_warps, and enable_fp_fusion=False.

Backward reuses updated_residual and inverse_rms. The unfused strategies launch
one program per row and write input gradients plus FP32 weight-gradient
contributions [rows, n_cols]. A metadata policy chooses the strategy.
SEQUENTIAL left-folds in row
order. TILED uses row lanes within each program. PARALLEL partitions rows and merges
FP32 partials in a second launch. The strategies have different addition orders.
FUSED instead accumulates weight contributions while computing
input gradients for a fixed group of rows, then merges [groups, n_cols] partials.
It never allocates the full [rows, n_cols] contribution matrix.
Both upstream gradient buffers are required; supply zeros for an unused output branch.
Gradient output buffers select the final storage dtypes. Use the same stream
for all backward launches, and disable FP fusion for all kernels.

Normalization uses the unrounded FP32 residual sum. Both forward outputs are
FP32; backward computes in FP32 and casts each input gradient to that input's
dtype on store. Model integration must preserve these declared cast points.
This operator is not yet registered as the model's strict implementation.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Generic, Protocol, TypeAlias, TypeVar

import torch
import triton
import triton.language as tl

_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
"""Accepted device types; ROCm PyTorch also uses the CUDA namespace."""

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
"""Floating dtypes accepted independently for x, residual, and weight."""

_NUM_WARPS = 4
"""Warps per program for forward and per-row input-gradient calculations."""

_WEIGHT_BLOCK_ROWS = 32
"""Fixed row lanes used when merging PARALLEL or FUSED weight partials."""


class RMSNormWeightGradStrategy(Enum):
    """Select how backward computes and combines FP32 weight contributions."""

    SEQUENTIAL = "sequential"
    """Write [M, D] contributions; each column block sums rows one at a time."""

    TILED = "tiled"
    """Write [M, D] contributions; accumulate row tiles, then reduce the row lanes."""

    PARALLEL = "parallel"
    """Write [M, D] contributions; independent row tiles produce partials for a merge."""

    FUSED = "fused"
    """Compute input gradients and group weight partials together, then merge.

    Avoids writing the full [M, D] contribution matrix.
    """


@dataclass(frozen=True)
class _SequentialConfig:
    """Launch settings for accumulating weight contributions one row at a time."""

    block_cols: int = 128
    """Columns owned by one reduction program."""

    num_warps: int = 4
    """Warps in that program; rows are accumulated sequentially."""


@dataclass(frozen=True)
class _TiledConfig:
    """Launch settings for accumulating row tiles within each column program."""

    block_rows: int = 32
    """Independent row lanes accumulated before the final sum."""

    block_cols: int = 128
    """Columns owned by one reduction program."""

    num_warps: int = 4
    """Warps for the tiled reduction."""


@dataclass(frozen=True)
class _ParallelConfig:
    """Launch settings for parallel row partials followed by a separate merge."""

    rows_per_partial: int = 256
    """Rows reduced by each independent partial program."""

    block_cols: int = 128
    """Column tile shared by partial and merge kernels."""

    num_warps: int = 4
    """Warps for both stages; the merge uses fixed 32-row tiles."""


@dataclass(frozen=True)
class _FusedConfig:
    """Row grouping and merge settings for fused input and weight-gradient work."""

    rows_per_group: int = 64
    """Rows whose input gradients and weight partial are fused."""

    merge_block_cols: int = 64
    """Column tile for merging group partials only."""

    merge_num_warps: int = 4
    """Merge warps; the fused kernel always uses _NUM_WARPS."""


@dataclass(frozen=True)
class RMSNormDefaultDevice:
    """Default variant: a shared policy with no device-specific attributes."""


@dataclass(frozen=True)
class RMSNormDevice:
    """Device variant: a real device identified by backend and GPU model."""

    backend: str
    """Real backend name, e.g. 'cuda' or 'rocm'."""

    gpu_name: str
    """GPU model; empty only for backends without a model lookup."""


RMSNormDeviceKey: TypeAlias = RMSNormDefaultDevice | RMSNormDevice
"""A sum type: Default carries no data; Device requires its device metadata."""

_DEFAULT_DEVICE = RMSNormDefaultDevice()


@dataclass(frozen=True)
class RMSNormWeightGradKey:
    """Metadata range for an internal tuning rule."""

    device: RMSNormDeviceKey
    """_DEFAULT_DEVICE matches every device; RMSNormDevice requires an exact match."""

    dtype: torch.dtype | None
    """x dtype; None shares a rule across supported dtypes."""

    n_cols: int | None
    """Exact feature width D; None matches every width."""

    min_rows: int
    """Inclusive lower bound on flattened rows M = x.numel() // D."""

    max_rows: int | None
    """Inclusive upper bound; None also covers larger unseen M."""


@dataclass(frozen=True)
class _RulePriority:
    """Compare matching rules by device, then dtype, then feature width."""

    device_specific: bool
    """A real device takes precedence over the default device."""

    dtype_specific: bool
    """A fixed dtype takes precedence over an unrestricted dtype."""

    width_specific: bool
    """A fixed D takes precedence over an unrestricted width."""

    def __gt__(self, other: object) -> bool:
        """Overload >: compare device first, then dtype, then feature width.

        Specific (True) wins over unrestricted (False). Equal priorities return
        False so the selector keeps the already selected rule.
        """
        if not isinstance(other, _RulePriority):
            return NotImplemented
        if self.device_specific != other.device_specific:
            return self.device_specific
        if self.dtype_specific != other.dtype_specific:
            return self.dtype_specific
        if self.width_specific != other.width_specific:
            return self.width_specific
        return False  # Equal priority keeps the already selected rule.


_ConfigT = TypeVar("_ConfigT")
"""The concrete config type stored by a plan, e.g. _FusedConfig."""

_ConfigT_contra = TypeVar("_ConfigT_contra", contravariant=True)
"""Config type consumed by the launcher protocol.

A callable accepting a broader config type can serve a more specific config type.
This is a separate protocol declaration, not a second config type in a plan.
"""


@dataclass(frozen=True)
class _BackwardInputs:
    """Contiguous saved tensors, upstreams, and preallocated gradient outputs."""

    updated_residual: torch.Tensor
    """FP32 [..., D] residual sum saved by forward."""

    inverse_rms: torch.Tensor
    """FP32 [M] inverse RMS saved by forward."""

    weight: torch.Tensor
    """[D] scale vector, in the original weight dtype."""

    grad_y: torch.Tensor
    """Upstream gradient [..., D]; zeros when y was unused."""

    grad_updated_residual_output: torch.Tensor
    """Upstream gradient [..., D]; zeros when the residual output was unused."""

    grad_x: torch.Tensor
    """Preallocated [..., D] output, in the original x dtype."""

    grad_residual: torch.Tensor
    """Preallocated [..., D] output, in the original residual dtype."""

    grad_weight: torch.Tensor
    """Preallocated [D] output, in the original weight dtype."""

    n_rows: int
    """Flattened row count M."""

    n_cols: int
    """Feature width D."""


class _BackwardLauncher(Protocol[_ConfigT_contra]):
    """Callable interface; Launcher[ConfigT] accepts that plan's config type."""

    def __call__(self, *, inputs: _BackwardInputs, config: _ConfigT_contra) -> None:
        """Write all three gradients on the caller's input-device stream."""
        ...


@dataclass(frozen=True)
class RMSNormWeightGradPlan(Generic[_ConfigT]):
    """One complete static configuration: matching conditions, strategy, and launcher.

    Each entry has its own fixed key and config. Selection returns the entry
    itself; it never copies a plan, replaces its config, or adds matching state.
    The registry groups these entries by strategy. A key of None marks that
    strategy's default for explicit selection and excludes it from automatic
    matching. Other entries define the automatic device/dtype/shape ranges.

    Autograd saves the selected plan. Backward calls
    plan.launch(inputs=inputs, config=plan.config) without selecting again.
    ConfigT ties the stored config to the launcher's accepted config type.
    """

    key: RMSNormWeightGradKey | None
    """Static matching conditions; None marks the strategy's explicit default."""

    strategy: RMSNormWeightGradStrategy
    """Which backward algorithm this plan represents."""

    config: _ConfigT
    """Fixed launch settings for this entry, using the strategy's own config type."""

    launch: _BackwardLauncher[_ConfigT]
    """Function to call with this same config type."""

    def get_priority(self) -> _RulePriority:
        """Derive specificity from this entry's static matching conditions."""
        if self.key is None:
            return _RulePriority(device_specific=False, dtype_specific=False, width_specific=False)
        match self.key.device:
            case RMSNormDefaultDevice():
                device_specific = False
            case RMSNormDevice():
                device_specific = True
        return _RulePriority(
            device_specific=device_specific,
            dtype_specific=self.key.dtype is not None,
            width_specific=self.key.n_cols is not None,
        )


def _default_weight_grad_plan(strategy: RMSNormWeightGradStrategy) -> RMSNormWeightGradPlan:
    """Return the strategy's static default entry, identified by key=None."""
    for plan in _WEIGHT_GRAD_PLANS[strategy]:
        if plan.key is None:
            return plan
    raise ValueError(f"No default weight-gradient plan configured for {strategy.value}.")


@lru_cache(maxsize=1024)
def select_rmsnorm_weight_grad_plan(
    device_key: RMSNormDevice, dtype: torch.dtype, n_rows: int, n_cols: int
) -> RMSNormWeightGradPlan:
    """Prefer device, dtype, then width-specific rules before shared defaults.

    Only immutable metadata is cached. Rules at the same device/dtype/width
    tier must not overlap. Clear this cache after changing rules in a live process.

    Args:
        device_key: Actual input backend and GPU model.
        dtype: Original x dtype, before FP32 intermediate calculations.
        n_rows: Flattened row count M = x.numel() // D.
        n_cols: Feature width D, taken from the last input dimension.

    Returns:
        The matching static plan with highest priority, or the SEQUENTIAL default
        when no entry matches. Entries in the strategy registry remain unchanged.
    """
    selected_plan: RMSNormWeightGradPlan | None = None
    for plans in _WEIGHT_GRAD_PLANS.values():
        for candidate in plans:
            key = candidate.key
            if key is None:
                continue
            match key.device:
                case RMSNormDefaultDevice():
                    # Shared rule: any input device can use this configuration.
                    pass
                case RMSNormDevice() as rule_device:
                    # Device-specific rule: both backend and GPU model must match.
                    if rule_device != device_key:
                        continue

            if key.dtype is not None and key.dtype != dtype:
                continue
            if key.n_cols is not None and key.n_cols != n_cols:
                continue
            if n_rows < key.min_rows:
                continue
            if key.max_rows is not None and n_rows > key.max_rows:
                continue
            if selected_plan is None or candidate.get_priority() > selected_plan.get_priority():
                selected_plan = candidate
    if selected_plan is None:
        return _default_weight_grad_plan(RMSNormWeightGradStrategy.SEQUENTIAL)
    return selected_plan


@lru_cache(maxsize=None)
def _cuda_device_key(backend: str, index: int) -> RMSNormDevice:
    """Cache the GPU name for a resolved CUDA/ROCm device index."""
    return RMSNormDevice(
        backend=backend,
        gpu_name=torch.cuda.get_device_name(index),
    )


def rmsnorm_device_key(device: torch.device) -> RMSNormDevice:
    """Identify the input GPU, including ROCm's torch.cuda device namespace."""
    if device.type == "cuda":
        backend = "rocm" if torch.version.hip is not None else "cuda"
        index = device.index if device.index is not None else torch.cuda.current_device()
        return _cuda_device_key(backend, index)
    backend = "rocm" if device.type == "hip" else device.type
    return RMSNormDevice(backend=backend, gpu_name="")


def _resolve_weight_grad_plan(
    device: torch.device,
    dtype: torch.dtype,
    n_rows: int,
    n_cols: int,
    strategy: RMSNormWeightGradStrategy | None,
) -> RMSNormWeightGradPlan:
    """Resolve an explicit strategy or choose automatically from input metadata.

    Explicit strategies use their internal default configs without a device
    query. Automatic selection always uses the device and shape metadata,
    including inputs with zero rows.
    """
    return (
        _default_weight_grad_plan(strategy)
        if strategy is not None
        else select_rmsnorm_weight_grad_plan(rmsnorm_device_key(device), dtype, n_rows, n_cols)
    )


@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    updated_residual_ptr,
    inverse_rms_ptr,
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

    # 1. Add the residual, keeping the intermediate sum in FP32.
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
    tl.store(inverse_rms_ptr + row, tl.reshape(inverse_rms, ()))


@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    updated_residual_ptr,
    inverse_rms_ptr,
    weight_ptr,
    grad_y_ptr,
    grad_updated_residual_output_ptr,
    grad_x_ptr,
    grad_residual_ptr,
    grad_weight_per_row_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute input gradients and each row's FP32 weight-gradient contribution."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    updated_residual = tl.load(updated_residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row).to(tl.float32)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_updated_residual_output = tl.load(
        grad_updated_residual_output_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    # 1. Reuse the FP32 sum and row statistic saved by forward.
    normalized = updated_residual * inverse_rms

    # 2. y = normalized * weight. Save this row's weight-gradient contribution.
    grad_normalized = grad_y * weight
    grad_weight_per_row = grad_y * normalized

    # 3. RMSNorm backward: rstd * (g - normalized * mean(g * normalized)).
    # One fixed row reduction combines the direct and inverse_rms paths.
    correction_sum = tl.sum(grad_normalized * normalized, axis=0)
    correction = tl.div_rn(correction_sum, n_cols)
    grad_updated_residual_from_y = inverse_rms * (grad_normalized - normalized * correction)

    # 4. Add the gradient from the separate residual output.
    grad_updated_residual_total = grad_updated_residual_from_y + grad_updated_residual_output

    # 5. updated_residual = x + residual: both inputs receive this gradient.
    grad_x = grad_updated_residual_total
    grad_residual = grad_updated_residual_total

    tl.store(grad_x_ptr + offsets, grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask)
    tl.store(
        grad_residual_ptr + offsets,
        grad_residual.to(grad_residual_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(grad_weight_per_row_ptr + offsets, grad_weight_per_row, mask=mask)


@triton.jit
def _fused_add_rmsnorm_bwd_grouped_kernel(
    updated_residual_ptr,
    inverse_rms_ptr,
    weight_ptr,
    grad_y_ptr,
    grad_updated_residual_output_ptr,
    grad_x_ptr,
    grad_residual_ptr,
    grad_weight_partials_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
):
    """Compute input gradients and fold weight contributions within a row group.

    Each program retains one FP32 vector of weight partials in registers. Only
    that vector is written after the loop. Groups are fixed by ROWS_PER_GROUP,
    independent of SM count, scheduling, or the current batch's number of rows.
    Keep each row's arithmetic and four-warps layout identical to the row kernel.
    """
    group = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    grad_weight_partial = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    row_start = group * ROWS_PER_GROUP
    row_end = tl.minimum(row_start + ROWS_PER_GROUP, n_rows)

    for row in range(row_start, row_end):
        offsets = row * n_cols + cols
        updated_residual = tl.load(updated_residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        inverse_rms = tl.load(inverse_rms_ptr + row).to(tl.float32)
        grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_updated_residual_output = tl.load(
            grad_updated_residual_output_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        # Preserve the original per-row expressions, including FP32 rounding.
        normalized = updated_residual * inverse_rms
        grad_normalized = grad_y * weight
        grad_weight_per_row = grad_y * normalized
        correction_sum = tl.sum(grad_normalized * normalized, axis=0)
        correction = tl.div_rn(correction_sum, n_cols)
        grad_updated_residual_from_y = inverse_rms * (grad_normalized - normalized * correction)
        grad_updated_residual_total = grad_updated_residual_from_y + grad_updated_residual_output
        tl.store(
            grad_x_ptr + offsets,
            grad_updated_residual_total.to(grad_x_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            grad_residual_ptr + offsets,
            grad_updated_residual_total.to(grad_residual_ptr.dtype.element_ty),
            mask=mask,
        )
        grad_weight_partial = grad_weight_partial + grad_weight_per_row

    tl.store(grad_weight_partials_ptr + group * n_cols + cols, grad_weight_partial, mask=mask)


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
    for row in range(n_rows):
        offsets = row.to(tl.int64) * n_cols + cols
        contribution = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_weight = grad_weight + contribution

    tl.store(grad_weight_ptr + cols, grad_weight.to(grad_weight_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_bwd_weight_tiled_kernel(
    grad_weight_per_row_ptr,
    grad_weight_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Accumulate row tiles, then reduce the row lanes once.

    Each program still owns distinct output columns; no atomics are used.
    Row lane k accumulates rows k, k + BLOCK_ROWS, ... in FP32. Reducing those
    lanes changes rounding relative to the sequential kernel, even with FP
    fusion disabled. Zero rows yield zero; both row and column tails are masked.
    """
    block = tl.program_id(0).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = tl.arange(0, BLOCK_ROWS).to(tl.int64)

    grad_weight = tl.full((BLOCK_ROWS, BLOCK_SIZE), 0.0, tl.float32)
    for start in range(0, n_rows, BLOCK_ROWS):
        current_rows = start + rows
        offsets = current_rows[:, None] * n_cols + cols[None, :]
        mask = (current_rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        contribution = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_weight = grad_weight + contribution

    grad_weight_sum = tl.sum(grad_weight, axis=0)
    tl.store(
        grad_weight_ptr + cols,
        grad_weight_sum.to(grad_weight_ptr.dtype.element_ty),
        mask=cols < n_cols,
    )


@triton.jit
def _fused_add_rmsnorm_bwd_weight_partial_kernel(
    contributions_ptr,
    partials_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """One independent FP32 sum per fixed row tile and column tile; no atomics."""
    col_block = tl.program_id(0).to(tl.int64)
    row_block = tl.program_id(1).to(tl.int64)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offsets = rows[:, None] * n_cols + cols[None, :]
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    values = tl.load(contributions_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(values, axis=0)
    tl.store(partials_ptr + row_block * n_cols + cols, partial, mask=cols < n_cols)


def _launch_fused_add_rmsnorm_bwd_weight_sequential(
    *,
    grad_weight_per_row: torch.Tensor,
    grad_weight: torch.Tensor,
    n_rows: int,
    n_cols: int,
    config: _SequentialConfig,
) -> None:
    """Write weight gradients with the sequential kernel's launch configuration."""
    grid = (triton.cdiv(n_cols, config.block_cols),)
    _fused_add_rmsnorm_bwd_weight_kernel[grid](
        grad_weight_per_row,
        grad_weight,
        n_rows,
        n_cols,
        BLOCK_SIZE=config.block_cols,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )


def _launch_fused_add_rmsnorm_bwd_weight_tiled(
    *,
    grad_weight_per_row: torch.Tensor,
    grad_weight: torch.Tensor,
    n_rows: int,
    n_cols: int,
    config: _TiledConfig,
) -> None:
    """Write weight gradients with the tiled kernel's launch configuration."""
    grid = (triton.cdiv(n_cols, config.block_cols),)
    _fused_add_rmsnorm_bwd_weight_tiled_kernel[grid](
        grad_weight_per_row,
        grad_weight,
        n_rows,
        n_cols,
        BLOCK_SIZE=config.block_cols,
        BLOCK_ROWS=config.block_rows,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )


def _launch_fused_add_rmsnorm_bwd_weight_parallel(
    *,
    grad_weight_per_row: torch.Tensor,
    grad_weight: torch.Tensor,
    n_rows: int,
    n_cols: int,
    config: _ParallelConfig,
) -> None:
    """Partition rows, then merge on the same stream using a fixed reduction layout.

    Allocate FP32 [max(1, ceil(rows / rows_per_partial)), n_cols] partials on the input
    device. Every element is overwritten on every launch, including tails.
    The split depends on the explicit configuration, never on SM count/occupancy.
    """
    n_partials = max(1, triton.cdiv(n_rows, config.rows_per_partial))
    partials = torch.empty(
        (n_partials, n_cols), device=grad_weight_per_row.device, dtype=torch.float32
    )
    col_blocks = triton.cdiv(n_cols, config.block_cols)
    _fused_add_rmsnorm_bwd_weight_partial_kernel[(col_blocks, n_partials)](
        grad_weight_per_row,
        partials,
        n_rows,
        n_cols,
        BLOCK_ROWS=config.rows_per_partial,
        BLOCK_COLS=config.block_cols,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )
    # Reuse a fixed 32-lane fold, now over partials rather than all input rows.
    _fused_add_rmsnorm_bwd_weight_tiled_kernel[(col_blocks,)](
        partials,
        grad_weight,
        n_partials,
        n_cols,
        BLOCK_SIZE=config.block_cols,
        BLOCK_ROWS=_WEIGHT_BLOCK_ROWS,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )


def _launch_row_backward(inputs: _BackwardInputs) -> torch.Tensor:
    """Write grad_x/grad_residual and return the FP32 [M, D] weight contributions."""
    contributions = torch.empty(
        (inputs.n_rows, inputs.n_cols),
        device=inputs.updated_residual.device,
        dtype=torch.float32,
    )
    _fused_add_rmsnorm_bwd_kernel[(inputs.n_rows,)](
        inputs.updated_residual,
        inputs.inverse_rms,
        inputs.weight,
        inputs.grad_y,
        inputs.grad_updated_residual_output,
        inputs.grad_x,
        inputs.grad_residual,
        contributions,
        inputs.n_cols,
        BLOCK_SIZE=triton.next_power_of_2(inputs.n_cols),
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    return contributions


def _launch_backward_sequential(*, inputs: _BackwardInputs, config: _SequentialConfig) -> None:
    """Write input gradients, then sum each column's weight contributions in row order."""
    contributions = _launch_row_backward(inputs)
    _launch_fused_add_rmsnorm_bwd_weight_sequential(
        grad_weight_per_row=contributions,
        grad_weight=inputs.grad_weight,
        n_rows=inputs.n_rows,
        n_cols=inputs.n_cols,
        config=config,
    )


def _launch_backward_tiled(*, inputs: _BackwardInputs, config: _TiledConfig) -> None:
    """Write input gradients, then accumulate weight contributions with row lanes."""
    contributions = _launch_row_backward(inputs)
    _launch_fused_add_rmsnorm_bwd_weight_tiled(
        grad_weight_per_row=contributions,
        grad_weight=inputs.grad_weight,
        n_rows=inputs.n_rows,
        n_cols=inputs.n_cols,
        config=config,
    )


def _launch_backward_parallel(*, inputs: _BackwardInputs, config: _ParallelConfig) -> None:
    """Write input gradients, reduce independent row tiles, then merge weight partials."""
    contributions = _launch_row_backward(inputs)
    _launch_fused_add_rmsnorm_bwd_weight_parallel(
        grad_weight_per_row=contributions,
        grad_weight=inputs.grad_weight,
        n_rows=inputs.n_rows,
        n_cols=inputs.n_cols,
        config=config,
    )


def _launch_backward_fused(*, inputs: _BackwardInputs, config: _FusedConfig) -> None:
    """Fused backward with only [ceil(M / rows_per_group), D] scratch."""
    n_groups = triton.cdiv(inputs.n_rows, config.rows_per_group)
    partials = torch.empty(
        (n_groups, inputs.n_cols),
        device=inputs.updated_residual.device,
        dtype=torch.float32,
    )
    _fused_add_rmsnorm_bwd_grouped_kernel[(n_groups,)](
        inputs.updated_residual,
        inputs.inverse_rms,
        inputs.weight,
        inputs.grad_y,
        inputs.grad_updated_residual_output,
        inputs.grad_x,
        inputs.grad_residual,
        partials,
        inputs.n_rows,
        inputs.n_cols,
        BLOCK_SIZE=triton.next_power_of_2(inputs.n_cols),
        ROWS_PER_GROUP=config.rows_per_group,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    # Same-stream fixed-order merge; no atomic additions or host synchronization.
    merge_grid = (triton.cdiv(inputs.n_cols, config.merge_block_cols),)
    _fused_add_rmsnorm_bwd_weight_tiled_kernel[merge_grid](
        partials,
        inputs.grad_weight,
        n_groups,
        inputs.n_cols,
        BLOCK_SIZE=config.merge_block_cols,
        BLOCK_ROWS=_WEIGHT_BLOCK_ROWS,
        num_warps=config.merge_num_warps,
        enable_fp_fusion=False,
    )


_WEIGHT_GRAD_PLANS: dict[RMSNormWeightGradStrategy, tuple[RMSNormWeightGradPlan, ...]] = {
    RMSNormWeightGradStrategy.SEQUENTIAL: (
        RMSNormWeightGradPlan(
            key=None,
            strategy=RMSNormWeightGradStrategy.SEQUENTIAL,
            config=_SequentialConfig(),
            launch=_launch_backward_sequential,
        ),
        RMSNormWeightGradPlan(
            key=RMSNormWeightGradKey(
                device=_DEFAULT_DEVICE,
                dtype=None,
                n_cols=None,
                min_rows=0,
                max_rows=8,
            ),
            strategy=RMSNormWeightGradStrategy.SEQUENTIAL,
            config=_SequentialConfig(),
            launch=_launch_backward_sequential,
        ),
    ),
    RMSNormWeightGradStrategy.TILED: (
        RMSNormWeightGradPlan(
            key=None,
            strategy=RMSNormWeightGradStrategy.TILED,
            config=_TiledConfig(),
            launch=_launch_backward_tiled,
        ),
        RMSNormWeightGradPlan(
            key=RMSNormWeightGradKey(
                device=_DEFAULT_DEVICE,
                dtype=None,
                n_cols=None,
                min_rows=9,
                max_rows=32,
            ),
            strategy=RMSNormWeightGradStrategy.TILED,
            config=_TiledConfig(block_rows=32, block_cols=64, num_warps=4),
            launch=_launch_backward_tiled,
        ),
        RMSNormWeightGradPlan(
            key=RMSNormWeightGradKey(
                device=_DEFAULT_DEVICE,
                dtype=None,
                n_cols=None,
                min_rows=33,
                max_rows=None,
            ),
            strategy=RMSNormWeightGradStrategy.TILED,
            config=_TiledConfig(block_rows=64, block_cols=64, num_warps=8),
            launch=_launch_backward_tiled,
        ),
    ),
    RMSNormWeightGradStrategy.PARALLEL: (
        RMSNormWeightGradPlan(
            key=None,
            strategy=RMSNormWeightGradStrategy.PARALLEL,
            config=_ParallelConfig(),
            launch=_launch_backward_parallel,
        ),
    ),
    RMSNormWeightGradStrategy.FUSED: (
        RMSNormWeightGradPlan(
            key=None,
            strategy=RMSNormWeightGradStrategy.FUSED,
            config=_FusedConfig(),
            launch=_launch_backward_fused,
        ),
        RMSNormWeightGradPlan(
            key=RMSNormWeightGradKey(
                device=_DEFAULT_DEVICE,
                dtype=None,
                n_cols=2688,
                min_rows=16384,
                max_rows=None,
            ),
            strategy=RMSNormWeightGradStrategy.FUSED,
            config=_FusedConfig(),
            launch=_launch_backward_fused,
        ),
    ),
}
"""Complete static backward plans, grouped by strategy in a single registry.

Each strategy has one key=None entry for explicit selection. Remaining entries
pair a fixed matching range with its strategy, config, and launcher. Tuple order
does not identify defaults; the key=None marker does.

H100 80 GB defaults are shared by other devices until separately tuned.
Only D=2688 and M>=16384 use FUSED; M>65536 is an explicit extrapolation.
Dtypes share settings, and graph/eager execution never changes selection.
"""


def _validate_inputs(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """Check forward shapes, floating dtypes, shared device, and positive finite eps."""
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have shape [..., D] with D > 0.")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x.")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must have shape [D].")
    if residual.device != x.device or weight.device != x.device:
        raise ValueError("x, residual, and weight must be on the same device.")
    if any(t.dtype not in _SUPPORTED_DTYPES for t in (x, residual, weight)):
        raise TypeError(f"x, residual, and weight must have dtype in {_SUPPORTED_DTYPES}.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if x.device.type not in _SUPPORTED_DEVICES:
        raise ValueError(f"Triton fused add RMSNorm requires a device in {_SUPPORTED_DEVICES}.")


def _validate_backward_inputs(
    updated_residual: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Check saved tensor shapes, shared device, and the weight dtype for backward."""
    if updated_residual.ndim == 0 or updated_residual.shape[-1] == 0:
        raise ValueError("updated_residual must have shape [..., D] with D > 0.")
    if updated_residual.device.type not in _SUPPORTED_DEVICES:
        raise ValueError(f"Triton fused add RMSNorm requires a device in {_SUPPORTED_DEVICES}.")
    n_cols = updated_residual.shape[-1]
    if inverse_rms.shape != (updated_residual.numel() // n_cols,):
        raise ValueError("inverse_rms must have one value per flattened input row.")
    if weight.shape != (n_cols,) or weight.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("weight must have shape [D] and a supported floating dtype.")
    if inverse_rms.device != updated_residual.device or weight.device != updated_residual.device:
        raise ValueError("saved tensors must be on the same device.")


def _launch_fused_add_rmsnorm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch forward on the input device, copying strided tensors if needed.

    Args:
        x: Input activations [..., D] in a supported floating dtype.
        residual: Residual activations with the same shape and device as x.
        weight: Scale vector [D] on the input device.
        eps: Positive finite constant added to the per-row mean square.

    Returns:
        (y, updated_residual, inverse_rms), all FP32. The first two tensors
        retain x.shape; inverse_rms has one value per flattened input row.
    """
    _validate_inputs(x, residual, weight, eps)
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    updated_residual = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    inverse_rms = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return y, updated_residual, inverse_rms

    if x.device.type == "cuda":
        device_context = torch.cuda.device(x.device)
    else:
        device_context = nullcontext()
    with device_context:
        _fused_add_rmsnorm_fwd_kernel[(n_rows,)](
            x.contiguous(),
            residual.contiguous(),
            weight.contiguous(),
            y,
            updated_residual,
            inverse_rms,
            n_cols,
            eps,
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
    return y, updated_residual, inverse_rms


def _launch_fused_add_rmsnorm_bwd(
    updated_residual: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor | None,
    grad_updated_residual_output: torch.Tensor | None,
    *,
    x_dtype: torch.dtype,
    residual_dtype: torch.dtype,
    plan: RMSNormWeightGradPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare backward buffers and execute the plan already selected by forward.

    Allocate the three output buffers, make kernel inputs contiguous, and launch
    the supplied plan on the input device. An unused output contributes zero.
    This function performs no strategy selection. All calculations use FP32;
    each final gradient is stored in its original input dtype.

    Args:
        updated_residual: FP32 residual sum [..., D] saved by forward.
        inverse_rms: FP32 inverse RMS [M] saved by forward.
        weight: Original scale vector [D], retaining its input dtype.
        grad_y: Upstream gradient for y, or None when that output was unused.
        grad_updated_residual_output: Upstream gradient for the residual output,
            or None when that output was unused.
        x_dtype: Original x dtype, used to allocate grad_x.
        residual_dtype: Original residual dtype, used to allocate grad_residual.
        plan: Complete static plan saved by forward, or selected by a direct caller.

    Returns:
        (grad_x, grad_residual, grad_weight), with each gradient stored in its
        corresponding original input dtype and shape.
    """
    _validate_backward_inputs(updated_residual, inverse_rms, weight)
    device = updated_residual.device
    shape = updated_residual.shape
    n_cols = shape[-1]
    n_rows = updated_residual.numel() // n_cols

    # 1. Allocate the three input gradients in their original dtypes.
    grad_x = torch.empty(shape, device=device, dtype=x_dtype)
    grad_residual = torch.empty(shape, device=device, dtype=residual_dtype)
    if n_rows == 0:
        # Empty x/residual have empty gradients; weight receives no contribution.
        grad_weight = torch.zeros_like(weight)
        return grad_x, grad_residual, grad_weight
    grad_weight = torch.empty((n_cols,), device=device, dtype=weight.dtype)

    if device.type == "cuda":
        device_context = torch.cuda.device(device)
    else:
        device_context = nullcontext()
    with device_context:
        # 2. Prepare contiguous saved tensors and upstreams; unused outputs supply zeros.
        updated_residual = updated_residual.contiguous()
        inverse_rms = inverse_rms.contiguous()
        weight = weight.contiguous()

        grad_y = torch.zeros_like(updated_residual) if grad_y is None else grad_y.contiguous()
        grad_updated_residual_output = (
            torch.zeros_like(updated_residual)
            if grad_updated_residual_output is None
            else grad_updated_residual_output.contiguous()
        )

        # 3. Pass the prepared tensors and buffers to the selected launcher.
        inputs = _BackwardInputs(
            updated_residual=updated_residual,
            inverse_rms=inverse_rms,
            weight=weight,
            grad_y=grad_y,
            grad_updated_residual_output=grad_updated_residual_output,
            grad_x=grad_x,
            grad_residual=grad_residual,
            grad_weight=grad_weight,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        plan.launch(inputs=inputs, config=plan.config)
    return grad_x, grad_residual, grad_weight


class _FusedAddRMSNormTritonFunction(torch.autograd.Function):
    """Connect the two outputs and three input gradients to PyTorch autograd."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps, weight_grad_strategy):
        """Return two FP32 outputs and save tensors, input dtypes, and the resolved plan."""
        y, updated_residual, inverse_rms = _launch_fused_add_rmsnorm_fwd(
            x, residual, weight, eps=eps
        )
        plan = _resolve_weight_grad_plan(
            device=x.device,
            dtype=x.dtype,
            n_rows=inverse_rms.numel(),
            n_cols=x.shape[-1],
            strategy=weight_grad_strategy,
        )
        # Keep the existing FP32 output, not copies of x/residual or a rounded sum.
        ctx.save_for_backward(updated_residual, inverse_rms, weight)
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype
        ctx.weight_grad_plan = plan
        # Backward explicitly handles an output that was not used by the loss.
        ctx.set_materialize_grads(False)
        return y, updated_residual

    @staticmethod
    def backward(ctx, grad_y, grad_updated_residual_output):
        """Combine both output paths and return gradients in forward-argument order."""
        if grad_y is None and grad_updated_residual_output is None:
            return None, None, None, None, None
        updated_residual, inverse_rms, weight = ctx.saved_tensors
        grad_x, grad_residual, grad_weight = _launch_fused_add_rmsnorm_bwd(
            updated_residual=updated_residual,
            inverse_rms=inverse_rms,
            weight=weight,
            grad_y=grad_y,
            grad_updated_residual_output=grad_updated_residual_output,
            x_dtype=ctx.x_dtype,
            residual_dtype=ctx.residual_dtype,
            plan=ctx.weight_grad_plan,
        )
        needs_x_grad, needs_residual_grad, needs_weight_grad = ctx.needs_input_grad[:3]
        if not needs_x_grad:
            grad_x = None
        if not needs_residual_grad:
            grad_residual = None
        if not needs_weight_grad:
            grad_weight = None
        # forward inputs: x, residual, weight, eps, weight_grad_strategy.
        return grad_x, grad_residual, grad_weight, None, None


class TritonFusedAddRMSNormOp:
    """FP32-output fused add RMSNorm with first-order autograd on accepted GPUs.

    x/residual have identical shape [..., D], and weight has shape [D]. All
    inputs share a device and may independently use FP16, BF16, or FP32.
    Both outputs retain the input shape and use FP32; each input gradient uses
    that input's dtype. Noncontiguous tensors are copied to contiguous buffers.
    Normalization uses the FP32 residual sum without an intermediate downcast.
    weight_grad_strategy selects how backward accumulates weight gradients.
    FUSED combines row backward and grouped weight accumulation,
    keeping the row's fixed arithmetic and using a smaller partial workspace.
    None selects a conservative metadata-based strategy and configuration.
    An explicit strategy uses its internal default configuration. Tile sizes
    and warp counts are internal tuning details, not constructor arguments.
    The resolved plan belongs to each autograd call, not to this reusable Op.
    """

    op_class = "norm"

    def __init__(
        self,
        *,
        weight_grad_strategy: RMSNormWeightGradStrategy | None = None,
    ):
        """Choose a backward strategy while keeping launch configurations internal.

        Args:
            weight_grad_strategy: None selects automatically for each input;
                an explicit strategy uses its internal default configuration.
        """
        self.weight_grad_strategy = weight_grad_strategy

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 (y, updated_residual), each retaining x.shape."""
        return self.forward(x, residual, weight, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the residual and apply RMSNorm with first-order autograd support.

        Args:
            x: Input activations [..., D]; FP16, BF16, or FP32.
            residual: Same shape/device as x; its dtype may differ from x.
            weight: Scale vector [D] on the same device; FP16, BF16, or FP32.
            eps: Positive finite constant added before taking the inverse RMS.

        Returns:
            (y, updated_residual), both FP32 with x.shape. Backward returns
            gradients in the respective original input dtypes.
        """
        return _FusedAddRMSNormTritonFunction.apply(
            x, residual, weight, eps, self.weight_grad_strategy
        )
