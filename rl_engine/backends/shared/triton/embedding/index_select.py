# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Select unique rows along an axis and scatter their gradients back with Triton."""

from __future__ import annotations

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
_BLOCK_SIZE = 256
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
def _index_select_fwd_kernel(
    x_ptr,
    index_ptr,
    y_ptr,
    n_rows,
    n_selected,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    X_OUTER_STRIDES: tl.constexpr,
    X_INNER_STRIDES: tl.constexpr,
    x_stride_axis,
    index_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a tile from an indexed slice, retaining the input's storage bits.

    Logically view x as [outer, axis, inner] and y as [outer, K, inner].
    Grid axis 0 distributes the outer * K logical rows, and axis 1 distributes
    the inner tiles of each row. If either axis reaches its launch limit, a
    program advances by the grid size on that axis. Input reads use x's
    original strides, and output writes are contiguous. The kernel copies
    each selected element without floating-point arithmetic or an input reshape.
    """
    first_row = tl.program_id(0).to(tl.int64)
    row_stride = tl.num_programs(0).to(tl.int64)
    row_end = tl.cast(n_rows, tl.int64)
    first_tile = tl.program_id(1)
    n_tiles = tl.cdiv(inner_size, BLOCK_SIZE)
    tile_stride = tl.num_programs(1)

    for output_row in range(first_row, row_end, row_stride):
        # y[outer, position, :] comes from x[outer, index[position], :].
        outer, position = output_row // n_selected, output_row % n_selected
        outer_offset = _strided_offset(outer, OUTER_SHAPE, X_OUTER_STRIDES)
        input_row = tl.load(index_ptr + position * index_stride).to(tl.int64)
        for tile in range(first_tile, n_tiles, tile_stride):
            cols = tl.cast(tile, tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < inner_size
            inner_offsets = _strided_offset(cols, INNER_SHAPE, X_INNER_STRIDES)
            value = tl.load(
                x_ptr + outer_offset + input_row * x_stride_axis + inner_offsets,
                mask=mask,
                other=0,
            )
            tl.store(y_ptr + output_row * inner_size + cols, value, mask=mask)


@triton.jit
def _index_select_bwd_unique_kernel(
    grad_y_ptr,
    index_ptr,
    grad_x_ptr,
    n_rows,
    n_selected,
    axis_size,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    GRAD_OUTER_STRIDES: tl.constexpr,
    GRAD_INNER_STRIDES: tl.constexpr,
    grad_y_stride_axis,
    index_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter a tile of grad_y into grad_x when each selected row occurs once."""
    first_row = tl.program_id(0).to(tl.int64)
    row_stride = tl.num_programs(0).to(tl.int64)
    row_end = tl.cast(n_rows, tl.int64)
    first_tile = tl.program_id(1)
    n_tiles = tl.cdiv(inner_size, BLOCK_SIZE)
    tile_stride = tl.num_programs(1)

    for output_row in range(first_row, row_end, row_stride):
        outer, position = output_row // n_selected, output_row % n_selected
        outer_offset = _strided_offset(outer, OUTER_SHAPE, GRAD_OUTER_STRIDES)
        input_row = tl.load(index_ptr + position * index_stride).to(tl.int64)
        # Read through grad_y's upstream strides. Unique indices give each selected
        # row of the contiguous grad_x a single writer for each tile.
        grad_y_start = outer_offset + position * grad_y_stride_axis
        grad_x_start = (outer * axis_size + input_row) * inner_size
        for tile in range(first_tile, n_tiles, tile_stride):
            cols = tl.cast(tile, tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < inner_size
            inner_offsets = _strided_offset(cols, INNER_SHAPE, GRAD_INNER_STRIDES)
            grad = tl.load(grad_y_ptr + grad_y_start + inner_offsets, mask=mask, other=0)
            # Match the reference's FP32 sum starting at +0. An upstream -0 gives +0.
            value = 0.0 + grad.to(tl.float32)
            tl.store(
                grad_x_ptr + grad_x_start + cols, value.to(grad_x_ptr.dtype.element_ty), mask=mask
            )


_BACKWARD_KERNELS = {True: _index_select_bwd_unique_kernel}


def _validate_inputs(x: Tensor, index: Tensor, dim: int) -> None:
    """Check the device, metadata and index bounds.

    Accept strided tensors and empty dimensions. Check index bounds on device
    when the backend implements PyTorch's asynchronous assertion.
    This backend owns its checks independently of the PyTorch reference.
    """
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
    if index.numel() != 0:
        axis_size = x.shape[dim]
        # The axis length can exceed int32 even when the requested row fits.
        max_valid_index = min(axis_size - 1, torch.iinfo(index.dtype).max)
        torch._assert_async(
            ((index >= 0) & (index <= max_valid_index)).all(),
            f"index values must be in [0, {axis_size})",
        )


def _launch_index_select_fwd(x: Tensor, index: Tensor, dim: int) -> Tensor:
    """Allocate the selected shape and copy slices on the input device's stream."""
    # Start at the selected axis, then drop it. For dim=-1 the suffix is empty.
    outer_shape, inner_shape = x.shape[:dim], x.shape[dim:][1:]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    n_selected = index.numel()
    output_shape = (*outer_shape, n_selected, *inner_shape)
    y = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    if y.numel() == 0:
        return y
    block_size = min(triton.next_power_of_2(inner_size), _BLOCK_SIZE)
    n_rows = outer_size * n_selected
    grid_x = min(n_rows, _MAX_GRID_X)
    grid_y = min(triton.cdiv(inner_size, block_size), _MAX_GRID_Y)
    grid = (grid_x, grid_y)
    # The caller may have another device selected. This guard restores it after the launch.
    device_module = torch.get_device_module(x.device)
    with device_module.device(x.device):
        _index_select_fwd_kernel[grid](
            x,
            index,
            y,
            n_rows=n_rows,
            n_selected=n_selected,
            inner_size=inner_size,
            OUTER_SHAPE=tuple(outer_shape),
            INNER_SHAPE=tuple(inner_shape),
            X_OUTER_STRIDES=x.stride()[:dim],
            X_INNER_STRIDES=x.stride()[dim:][1:],
            x_stride_axis=x.stride(dim),
            index_stride=index.stride(0),
            BLOCK_SIZE=block_size,
        )
    return y


def _launch_index_select_bwd(
    grad_y: Tensor, index: Tensor, x_shape: tuple[int, ...], dim: int, *, unique_indices: bool
) -> Tensor:
    """Use the forward call's mode to select how gradients are written back."""
    # The kernel writes only selected rows. Unselected rows keep +0, even for an empty index.
    grad_x = torch.zeros(x_shape, device=grad_y.device, dtype=grad_y.dtype)
    if grad_y.numel() == 0:
        return grad_x

    outer_shape, inner_shape = x_shape[:dim], x_shape[dim:][1:]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    n_selected = index.numel()
    block_size = min(triton.next_power_of_2(inner_size), _BLOCK_SIZE)
    n_rows = outer_size * n_selected
    grid_x = min(n_rows, _MAX_GRID_X)
    grid_y = min(triton.cdiv(inner_size, block_size), _MAX_GRID_Y)
    grid = (grid_x, grid_y)
    backward_kernel = _BACKWARD_KERNELS[unique_indices]
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
            OUTER_SHAPE=tuple(outer_shape),
            INNER_SHAPE=tuple(inner_shape),
            GRAD_OUTER_STRIDES=grad_y.stride()[:dim],
            GRAD_INNER_STRIDES=grad_y.stride()[dim:][1:],
            grad_y_stride_axis=grad_y.stride(dim),
            index_stride=index.stride(0),
            BLOCK_SIZE=block_size,
            num_warps=4,
            enable_fp_fusion=False,
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
