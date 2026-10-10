# SPDX-License-Identifier: Apache-2.0
"""Copy-only scatters for internally proven unique sampling indices."""
import torch
import triton
import triton.language as tl


@triton.jit
def _columns(
    values,
    indices,
    output,
    rows: tl.constexpr,
    width: tl.constexpr,
    value_stride: tl.constexpr,
    index_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // width, offsets % width
    active = row < rows
    destination = tl.load(indices + row * index_stride + col, active, 0)
    value = tl.load(values + row * value_stride + col, active, 0)
    tl.store(output + row * width + destination, value, active)


@triton.jit
def _rows(
    values,
    indices,
    output,
    rows: tl.constexpr,
    width: tl.constexpr,
    value_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // width, offsets % width
    active = row < rows
    destination = tl.load(indices + row, active, 0)
    value = tl.load(values + row * value_stride + col, active, 0)
    tl.store(output + destination * output_stride + col, value, active)


def scatter_permuted_columns(values, indices):
    """Inverse-copy a complete argsort permutation; each output is written once."""
    assert values.ndim == 2 and values.shape == indices.shape
    assert values.stride(1) == indices.stride(1) == 1
    output = torch.empty(values.shape, dtype=values.dtype, device=values.device)
    if values.numel():
        _columns[(triton.cdiv(values.numel(), 512),)](
            values, indices, output, *values.shape, values.stride(0), indices.stride(0), 512
        )
    return output


def copy_unique_rows_(output, indices, values):
    """Copy rows selected by arange/nonzero without a generic collision sort."""
    assert values.ndim == output.ndim == 2
    assert values.shape == (indices.numel(), output.size(1))
    assert values.stride(1) == output.stride(1) == indices.stride(0) == 1
    if values.numel():
        _rows[(triton.cdiv(values.numel(), 512),)](
            values, indices, output, *values.shape, values.stride(0), output.stride(0), 512
        )
    return output
