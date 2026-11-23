# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Select unique rows along an axis and scatter their gradients back with Triton."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.contracts.operators.index_select import require_unique_indices

_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)
_BLOCK_SIZE = 1024
_TILE_ELEMENTS = 512
_MAX_BLOCK_ROWS = 16
_MAX_GRID_X = 2**31 - 1
_MAX_GRID_Y = 65535


@triton.jit
def _strided_offset(flat_index, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    """Map a flattened logical coordinate to its strided element offset.

    Used separately for the dimensions before and after the selected axis.
    An empty shape contributes zero, covering selection along the first or
    last dimension without requiring an input copy.
    """
    # Even an empty suffix needs one zero offset per lane for a masked load.
    offset = tl.full(flat_index.shape, 0, tl.int64)
    # For shape (H, W), a flat index i gives offset (i // W) * stride_h + (i % W) * stride_w.
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (flat_index % SHAPE[axis]) * STRIDES[axis]
        flat_index = flat_index // SHAPE[axis]
    return offset


@triton.jit
def _load_index_row(index_ptr, position, index_stride):
    """Read index[position]. Only a non-contiguous index view passes a stride."""
    if index_stride is None:
        offset = position
    else:
        offset = position * index_stride
    return tl.load(index_ptr + offset).to(tl.int64)


@triton.jit
def _index_select_fwd_kernel(
    x_ptr,
    index_ptr,
    y_ptr,
    indices_valid_ptr,
    n_rows,
    n_selected,
    axis_size,
    inner_size: tl.constexpr,
    X_IS_CONTIGUOUS: tl.constexpr,
    index_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_STEPS: tl.constexpr,
    COL_STEPS: tl.constexpr,
    OUTER_SHAPE: tl.constexpr = None,
    INNER_SHAPE: tl.constexpr = None,
    OUTER_STRIDES: tl.constexpr = None,
    INNER_STRIDES: tl.constexpr = None,
    axis_stride=None,
):
    """Copy a tile from an indexed slice, retaining the input's storage bits.

    Logically view x as [outer, axis, inner] and y as [outer, K, inner].
    Grid axis 0 distributes tiles of output rows and axis 1 their columns.
    A narrow slice shares its program with neighboring rows. A capped grid
    revisits tiles in grid-size steps. Invalid indices flag the asynchronous
    bounds check and never address input storage.
    """
    first_row_tile = tl.program_id(0).to(tl.int64)
    row_stride = tl.num_programs(0).to(tl.int64)
    first_col_tile = tl.program_id(1).to(tl.int64)
    col_stride = tl.num_programs(1).to(tl.int64)

    for step in range(ROW_STEPS):
        row_tile = first_row_tile + step * row_stride
        output_row = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_row = output_row < n_rows
        # Modulo K always addresses the index vector, including padding lanes.
        outer, position = output_row // n_selected, output_row % n_selected
        input_row = _load_index_row(index_ptr, position, index_stride)
        valid_index = (input_row >= 0) & (input_row < axis_size)
        # The flag starts at one. Only tiles with an invalid index write it.
        if tl.sum((valid_row & ~valid_index).to(tl.int32), 0) != 0:
            tl.atomic_xchg(indices_valid_ptr, 0, sem="relaxed")
        input_row = tl.where(valid_index, input_row, 0)
        if X_IS_CONTIGUOUS:
            x_start = (outer * axis_size + input_row) * inner_size
        else:
            outer_offset = _strided_offset(outer, OUTER_SHAPE, OUTER_STRIDES)
            x_start = outer_offset + input_row * axis_stride
        for col_step in range(COL_STEPS):
            col_tile = first_col_tile + col_step * col_stride
            cols = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = valid_row[:, None] & (cols[None, :] < inner_size)
            if X_IS_CONTIGUOUS:
                inner_offsets = cols
            else:
                inner_offsets = _strided_offset(cols, INNER_SHAPE, INNER_STRIDES)
            value = tl.load(
                x_ptr + x_start[:, None] + inner_offsets[None, :],
                mask=mask & valid_index[:, None],
                other=0,
            )
            tl.store(y_ptr + output_row[:, None] * inner_size + cols[None, :], value, mask=mask)


@triton.jit
def _index_select_bwd_unique_kernel(
    grad_y_ptr,
    index_ptr,
    grad_x_ptr,
    n_rows,
    n_selected,
    axis_size,
    inner_size: tl.constexpr,
    GRAD_Y_IS_CONTIGUOUS: tl.constexpr,
    index_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_STEPS: tl.constexpr,
    COL_STEPS: tl.constexpr,
    OUTER_SHAPE: tl.constexpr = None,
    INNER_SHAPE: tl.constexpr = None,
    OUTER_STRIDES: tl.constexpr = None,
    INNER_STRIDES: tl.constexpr = None,
    axis_stride=None,
):
    """Scatter a tile of rows and columns. Unique indices give each element one writer."""
    first_row_tile = tl.program_id(0).to(tl.int64)
    row_stride = tl.num_programs(0).to(tl.int64)
    first_col_tile = tl.program_id(1).to(tl.int64)
    col_stride = tl.num_programs(1).to(tl.int64)

    for step in range(ROW_STEPS):
        row_tile = first_row_tile + step * row_stride
        output_row = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_row = output_row < n_rows
        outer, position = output_row // n_selected, output_row % n_selected
        input_row = _load_index_row(index_ptr, position, index_stride)
        # Forward reports invalid indices asynchronously. Independently guard
        # the scatter address before writing to the input gradient.
        valid_index = (input_row >= 0) & (input_row < axis_size)
        valid_row = valid_row & valid_index
        input_row = tl.where(valid_index, input_row, 0)
        # grad_y has the output shape, so output_row is already its flat row index.
        if GRAD_Y_IS_CONTIGUOUS:
            grad_y_start = output_row * inner_size
        else:
            outer_offset = _strided_offset(outer, OUTER_SHAPE, OUTER_STRIDES)
            grad_y_start = outer_offset + position * axis_stride
        grad_x_start = (outer * axis_size + input_row) * inner_size
        for col_step in range(COL_STEPS):
            col_tile = first_col_tile + col_step * col_stride
            cols = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = valid_row[:, None] & (cols[None, :] < inner_size)
            if GRAD_Y_IS_CONTIGUOUS:
                inner_offsets = cols
            else:
                inner_offsets = _strided_offset(cols, INNER_SHAPE, INNER_STRIDES)
            grad = tl.load(
                grad_y_ptr + grad_y_start[:, None] + inner_offsets[None, :], mask=mask, other=0
            )
            # Match the reference's FP32 sum starting at +0. An upstream -0 gives +0.
            value = 0.0 + grad.to(tl.float32)
            tl.store(
                grad_x_ptr + grad_x_start[:, None] + cols[None, :],
                value.to(grad_x_ptr.dtype.element_ty),
                mask=mask,
            )


_BACKWARD_KERNELS = {True: _index_select_bwd_unique_kernel}


def _validate_inputs(x: Tensor, index: Tensor, dim: int) -> None:
    """Check tensor metadata and handle index bounds when the gather cannot run."""
    if x.device.type not in _SUPPORTED_DEVICES:
        raise RuntimeError(f"x must use a device in {_SUPPORTED_DEVICES}, got {x.device.type!r}.")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if not -x.ndim <= dim < x.ndim:
        raise IndexError(f"dim must be in [{-x.ndim}, {x.ndim}), got {dim}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be one of {_SUPPORTED_DTYPES}, got {x.dtype}")
    if index.ndim != 1:
        raise ValueError("index must have shape [K]")
    if index.dtype not in _INDEX_DTYPES:
        raise TypeError(f"index dtype must be one of {_INDEX_DTYPES}, got {index.dtype}")
    if x.device != index.device:
        raise ValueError("x and index must be on the same device")
    if index.numel() and x.numel() == 0:
        # A zero-size axis or an empty outer/inner dimension leaves no data to
        # gather. The index contract still applies to every supplied entry.
        axis_size = x.shape[dim]
        # The axis length can exceed int32 even when the requested row fits.
        max_valid_index = min(axis_size - 1, torch.iinfo(index.dtype).max)
        torch._assert_async(
            ((index >= 0) & (index <= max_valid_index)).all(),
            f"index values must be in [0, {axis_size})",
        )


@dataclass(frozen=True, slots=True)
class _AxisLayout:
    """Describe a non-contiguous source tensor around the selected dimension ``dim``.

    Strides are measured in elements.

    Attributes:
        outer_shape: Sizes of the axes before ``dim``, in tensor order.
        inner_shape: Sizes of the axes after ``dim``, in tensor order.
        outer_strides: Strides of the axes before ``dim``, in tensor order.
            For a strided view, this tuple is empty when selecting the first axis.
        axis_stride: Distance between consecutive positions along ``dim``.
            A broadcast axis can have stride zero.
        inner_strides: Strides of the axes after ``dim``, in tensor order.
            For a strided view, this tuple is empty when selecting the last axis.
    """

    outer_shape: tuple[int, ...]
    inner_shape: tuple[int, ...]
    outer_strides: tuple[int, ...]
    axis_stride: int
    inner_strides: tuple[int, ...]

    @staticmethod
    def kernel_kwargs(layout: _AxisLayout | None) -> dict:
        """Return the strided kernel arguments, or an empty dict for a contiguous source."""
        if layout is None:
            return {}
        return {
            "OUTER_SHAPE": layout.outer_shape,
            "INNER_SHAPE": layout.inner_shape,
            "OUTER_STRIDES": layout.outer_strides,
            "INNER_STRIDES": layout.inner_strides,
            "axis_stride": layout.axis_stride,
        }

    @staticmethod
    def is_contiguous(layout: _AxisLayout | None) -> bool:
        """Return True when the entire source tensor is contiguous."""
        return layout is None


def _get_axis_layout(
    tensor: Tensor, dim: int, outer_shape: tuple[int, ...], inner_shape: tuple[int, ...]
) -> _AxisLayout | None:
    """Return None for contiguous addressing, or the shapes and strides of the view."""
    if tensor.is_contiguous():
        return None
    strides = tensor.stride()
    return _AxisLayout(
        outer_shape=tuple(outer_shape),
        inner_shape=tuple(inner_shape),
        outer_strides=strides[:dim],
        axis_stride=strides[dim],
        inner_strides=strides[dim:][1:],
    )


def _tile_grid(n_rows: int, inner_size: int) -> tuple[tuple[int, int], dict[str, int]]:
    """Group narrow rows in one tile; split wide rows across column tiles.

    A grid below both launch limits visits its tile once. When an axis is
    capped, the same program visits the remaining tiles along that axis.
    """
    block_size = min(triton.next_power_of_2(inner_size), _BLOCK_SIZE)
    block_rows = min(_MAX_BLOCK_ROWS, max(1, _TILE_ELEMENTS // block_size))
    row_tiles = triton.cdiv(n_rows, block_rows)
    col_tiles = triton.cdiv(inner_size, block_size)
    grid = min(row_tiles, _MAX_GRID_X), min(col_tiles, _MAX_GRID_Y)
    return grid, {
        "BLOCK_ROWS": block_rows,
        "BLOCK_SIZE": block_size,
        "ROW_STEPS": triton.cdiv(row_tiles, grid[0]),
        "COL_STEPS": triton.cdiv(col_tiles, grid[1]),
    }


def _launch_index_select_fwd(x: Tensor, index: Tensor, dim: int) -> Tensor:
    """Allocate the selected shape and copy slices on the input device's stream."""
    # Start at the selected axis, then drop it. For dim=-1 the suffix is empty.
    outer_shape, inner_shape = x.shape[:dim], x.shape[dim:][1:]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    n_selected = index.numel()
    output_shape = (*outer_shape, n_selected, *inner_shape)
    y = x.new_empty(output_shape)
    if y.numel() == 0 or x.numel() == 0:
        return y
    n_rows = outer_size * n_selected
    grid, tile = _tile_grid(n_rows, inner_size)
    x_layout = _get_axis_layout(x, dim, outer_shape, inner_shape)
    # The caller may have another device selected. This guard restores it after the launch.
    device_module = torch.get_device_module(x.device)
    with device_module.device(x.device):
        indices_valid = x.new_ones((), dtype=torch.int32)
        _index_select_fwd_kernel[grid](
            x,
            index,
            y,
            indices_valid,
            n_rows=n_rows,
            n_selected=n_selected,
            axis_size=x.shape[dim],
            inner_size=inner_size,
            X_IS_CONTIGUOUS=_AxisLayout.is_contiguous(x_layout),
            index_stride=None if index.is_contiguous() else index.stride(0),
            num_warps=4,
            **tile,
            **_AxisLayout.kernel_kwargs(x_layout),
        )
        torch._assert_async(indices_valid, f"index values must be in [0, {x.shape[dim]})")
    return y


def _launch_index_select_bwd(
    grad_y: Tensor, index: Tensor, x_shape: tuple[int, ...], dim: int, *, unique_indices: bool
) -> Tensor:
    """Use the forward call's mode to select how gradients are written back."""
    # The kernel writes only selected rows. Unselected rows keep +0, even for an empty index.
    grad_x = grad_y.new_zeros(x_shape)
    if grad_y.numel() == 0:
        return grad_x

    outer_shape, inner_shape = x_shape[:dim], x_shape[dim:][1:]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    n_selected = index.numel()
    n_rows = outer_size * n_selected
    grid, tile = _tile_grid(n_rows, inner_size)
    backward_kernel = _BACKWARD_KERNELS[unique_indices]
    grad_y_layout = _get_axis_layout(grad_y, dim, outer_shape, inner_shape)
    device_module = torch.get_device_module(grad_y.device)
    with device_module.device(grad_y.device):
        backward_kernel[grid](
            grad_y,
            index,
            grad_x,
            n_rows=n_rows,
            n_selected=n_selected,
            axis_size=x_shape[dim],
            inner_size=inner_size,
            GRAD_Y_IS_CONTIGUOUS=_AxisLayout.is_contiguous(grad_y_layout),
            index_stride=None if index.is_contiguous() else index.stride(0),
            num_warps=4,
            enable_fp_fusion=False,
            **tile,
            **_AxisLayout.kernel_kwargs(grad_y_layout),
        )
    return grad_x


class _IndexSelectTritonFunction(torch.autograd.Function):
    """Carry the index tensor, input shape, selected axis and mode into backward."""

    @staticmethod
    def forward(ctx, x: Tensor, index: Tensor, dim: int, unique_indices: bool) -> Tensor:
        y = _launch_index_select_fwd(x, index, dim)
        # Backward needs the indices and input metadata to route gradients back to x.
        ctx.save_for_backward(index)
        ctx.x_shape = x.shape
        ctx.dim = dim
        ctx.unique_indices = unique_indices
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: Tensor) -> tuple[Tensor, None, None, None]:
        (index,) = ctx.saved_tensors
        grad_x = _launch_index_select_bwd(
            grad_y, index, ctx.x_shape, ctx.dim, unique_indices=ctx.unique_indices
        )
        return grad_x, None, None, None


class TritonIndexSelectOp:
    """Select slices along an explicit axis with one writer per selected gradient.

    Args:
        x: Rank >= 1 FP32/FP16/BF16 tensor on a supported Triton device.
        index: Same-device int32/int64 vector of distinct, in-bounds row indices.
        dim: Required keyword axis. Negative axes count from the end of x.
        unique_indices: Pass True when the caller guarantees distinct indices.
            False (the default) raises NotImplementedError. The kernel trusts
            this promise; supplying duplicates with True gives undefined gradients.

    Output replaces x.shape[dim] by K and preserves input dtype and copied bits.
    Indices are shared across all other coordinates; negative indices are invalid.
    Strided inputs, indices and upstream gradients are supported. Forward uses
    a 2-D grid and writes contiguous output. Backward initializes grad_x to +0
    and scatters FP32(+0) + grad_y to each selected row, cast once to x.dtype.
    Only first-order autograd is supported.
    """

    op_class = "reduction"

    def __call__(
        self, x: Tensor, index: Tensor, *, dim: int, unique_indices: bool = False
    ) -> Tensor:
        """Select the supplied rows under the caller's uniqueness guarantee."""
        return self.forward(x, index, dim=dim, unique_indices=unique_indices)

    def forward(
        self, x: Tensor, index: Tensor, *, dim: int, unique_indices: bool = False
    ) -> Tensor:
        """Select the mode and validate the inputs before entering autograd."""
        require_unique_indices(unique_indices)
        _validate_inputs(x, index, dim)
        return _IndexSelectTritonFunction.apply(x, index, dim, unique_indices)
