# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 arithmetic for fused residual addition and RMSNorm.

The caller supplies contiguous x/residual tensors viewed as [rows, n_cols],
a contiguous weight vector [n_cols], and FP32 y/updated_residual output buffers.
Forward also saves one FP32 inverse_rms per row for backward.
Launch one program per row, with a fixed power-of-two BLOCK_SIZE >= n_cols > 0,
fixed num_warps, and enable_fp_fusion=False.
Both feature-axis reductions use the shared ordered sum_tile helper. Its
power-of-two width follows D, never the batch size or weight-gradient strategy.

Backward reuses updated_residual and inverse_rms. Both strategies launch one
program per row and write input gradients plus FP32 weight contributions [M, D].
SEQUENTIAL loads contributions one row at a time. TILED loads multiple rows but
adds them individually in ascending order. Both start from positive FP32 zero
and cast only after the complete row fold. Tile sizes change data movement,
never the addition tree. Grouped PARALLEL/FUSED reductions are not supported.
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

from rl_engine.backends.shared.triton.reduction import sum_tile

_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
"""Accepted device types; ROCm PyTorch also uses the CUDA namespace."""

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
"""Floating dtypes accepted independently for x, residual, and weight."""

_NUM_WARPS = 4
"""Warps per program for forward and per-row input-gradient calculations."""


class RMSNormWeightGradStrategy(Enum):
    """Select how backward computes and combines FP32 weight contributions."""

    SEQUENTIAL = "sequential"
    """Write [M, D] contributions; each column block sums rows one at a time."""

    TILED = "tiled"
    """Load row tiles; fold their rows in the same FP32 order as SEQUENTIAL."""


@dataclass(frozen=True)
class _SequentialConfig:
    """Launch settings for accumulating weight contributions one row at a time."""

    block_cols: int = 128
    """Columns owned by one reduction program."""

    num_warps: int = 4
    """Warps in that program; rows are accumulated sequentially."""


@dataclass(frozen=True)
class _TiledConfig:
    """Load-tile settings; neither tile size nor warp count changes row order."""

    block_rows: int = 32
    """Rows loaded together, then added one at a time in ascending order."""

    block_cols: int = 128
    """Columns owned by one reduction program."""

    num_warps: int = 4
    """Warps for the tiled reduction."""


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
"""The concrete config type stored by a plan, e.g. _TiledConfig."""

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
def _rmsnorm_row_statistics(updated_residual, n_cols: tl.constexpr, EPS: tl.constexpr):
    """Return the ordered FP32 squared sum and inverse RMS of one padded row.

    Invalid columns must already be zero. sum_tile fixes the addition tree;
    divide by the real feature width, not the padded vector length. Return the
    sum as well so conformance tests can distinguish reduction from rsqrt error.
    """
    sum_squares = sum_tile(updated_residual * updated_residual)
    mean_square = tl.div_rn(sum_squares, n_cols)
    inverse_rms = tl.rsqrt(mean_square + EPS)
    return sum_squares, inverse_rms


@triton.jit
def _rmsnorm_row_correction(grad_normalized, normalized, n_cols: tl.constexpr):
    """Return the ordered FP32 dot sum and its mean for input-gradient correction.

    Both vectors have the same padded width as forward. Padding contributes
    zero, and every weight-gradient strategy uses this identical row expression.
    """
    correction_sum = sum_tile(grad_normalized * normalized)
    return correction_sum, tl.div_rn(correction_sum, n_cols)


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
    _, inverse_rms = _rmsnorm_row_statistics(updated_residual, n_cols, EPS)

    # 3. Normalize each element, then apply its feature weight.
    normalized = updated_residual * inverse_rms
    y = normalized * weight

    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(updated_residual_ptr + offsets, updated_residual, mask=mask)
    tl.store(inverse_rms_ptr + row, tl.reshape(inverse_rms, ()))


@triton.jit
def _weight_contribution_fp32(grad_y, updated_residual, inverse_rms):
    """One row's parameter VJP, shared by backward and logical-row export."""
    normalized = updated_residual * inverse_rms
    return grad_y * normalized


@triton.jit
def _fused_add_rmsnorm_weight_contributions_kernel(
    updated_residual_ptr,
    inverse_rms_ptr,
    grad_y_ptr,
    contributions_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Export actual FP32 per-row contributions without summing or downcasting."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols
    updated_residual = tl.load(updated_residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row)
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    contributions = _weight_contribution_fp32(grad_y, updated_residual, inverse_rms)
    tl.store(contributions_ptr + offsets, contributions, mask=mask)


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
    grad_weight_per_row = _weight_contribution_fp32(grad_y, updated_residual, inverse_rms)

    # 3. RMSNorm backward: rstd * (g - normalized * mean(g * normalized)).
    # One fixed row reduction combines the direct and inverse_rms paths.
    _, correction = _rmsnorm_row_correction(grad_normalized, normalized, n_cols)
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
    """Batch loads while preserving the exact sequential FP32 row fold.

    A tile is a load buffer, not a set of partial sums. Gather one row from it
    at a time and add to the single accumulator. The tail loop visits only real
    rows, so padding cannot introduce additional additions or signed-zero changes.
    """
    block = tl.program_id(0).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = tl.arange(0, BLOCK_ROWS).to(tl.int64)
    grad_weight = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    for start in range(0, n_rows, BLOCK_ROWS):
        current_rows = start + rows
        offsets = current_rows[:, None] * n_cols + cols[None, :]
        mask = (current_rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        contributions = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        for row in range(tl.minimum(BLOCK_ROWS, n_rows - start)):
            indices = tl.full((1, BLOCK_SIZE), row, tl.int32)
            contribution = tl.gather(contributions, indices, axis=0).reshape((BLOCK_SIZE,))
            grad_weight = grad_weight + contribution

    tl.store(
        grad_weight_ptr + cols, grad_weight.to(grad_weight_ptr.dtype.element_ty), mask=cols < n_cols
    )


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
    """Write input gradients, then load tiles and fold their rows in ascending order."""
    contributions = _launch_row_backward(inputs)
    _launch_fused_add_rmsnorm_bwd_weight_tiled(
        grad_weight_per_row=contributions,
        grad_weight=inputs.grad_weight,
        n_rows=inputs.n_rows,
        n_cols=inputs.n_cols,
        config=config,
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
                max_rows=None,
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
    ),
}
"""One registry of immutable plans; both strategies use the same FP32 row order.

AUTO conservatively uses SEQUENTIAL at every shape until the ordered-load TILED
implementation has new GPU evidence. The former lane/group reduction timings
cannot select this new algorithm. Device-specific tuning may change loads or
launch dimensions, but must not introduce row partial sums or earlier casts.
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
    Both strategies implement the same arithmetic order for all gradients.
    GPU conformance must be qualified on the selected backend/compiler.
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

    def parameter_vjp_contributions_fp32(
        self,
        *,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        grad_output: torch.Tensor | None,
        eps: float = 1e-5,
    ) -> dict[str, torch.Tensor]:
        """Return unreduced weight VJPs for the logical-token aggregation protocol.

        grad_output is the upstream of y. The residual output is independent
        of weight and contributes nothing here; pass None when only it is used.
        The returned weight tensor is FP32 [M, D], even for a low-precision
        weight. This recomputes forward statistics with the production kernel
        and uses backward's exact contribution expression, not a PyTorch oracle.

        For batch/chunk invariance, callers retain token identities, restore
        these rows to the common logical order, and left-fold from FP32 zero.
        Cast to weight.dtype only once after the full logical batch. Adding
        already-reduced microbatch weight.grad tensors is not this protocol.
        Both SEQUENTIAL and TILED backward implement this same ordered fold
        when their input rows are already in logical order.
        """
        if grad_output is not None:
            if grad_output.shape != x.shape or grad_output.device != x.device:
                raise ValueError("grad_output must have the same shape and device as x.")
            if grad_output.dtype not in _SUPPORTED_DTYPES:
                raise TypeError(f"grad_output must have dtype in {_SUPPORTED_DTYPES}.")
        _, updated_residual, inverse_rms = _launch_fused_add_rmsnorm_fwd(
            x, residual, weight, eps=eps
        )
        n_rows, n_cols = inverse_rms.numel(), x.shape[-1]
        if grad_output is None:
            return {"weight": torch.zeros((n_rows, n_cols), device=x.device, dtype=torch.float32)}
        contributions = torch.empty((n_rows, n_cols), device=x.device, dtype=torch.float32)
        if n_rows:
            device_context = (
                torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext()
            )
            with device_context:
                _fused_add_rmsnorm_weight_contributions_kernel[(n_rows,)](
                    updated_residual,
                    inverse_rms,
                    grad_output.contiguous(),
                    contributions,
                    n_cols,
                    BLOCK_SIZE=triton.next_power_of_2(n_cols),
                    num_warps=_NUM_WARPS,
                    enable_fp_fusion=False,
                )
        return {"weight": contributions}
