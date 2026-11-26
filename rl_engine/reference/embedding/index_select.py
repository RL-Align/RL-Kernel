# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""PyTorch reference for unique index selection."""

from __future__ import annotations

import torch

from rl_engine.contracts.operators.index_select import require_unique_indices

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _validate_inputs(x: torch.Tensor, index: torch.Tensor, dim: int) -> None:
    """Check tensor metadata, the selection axis and index bounds."""
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
    if index.numel():
        axis_size = x.shape[dim]
        # The axis length can exceed int32 even when the requested row fits.
        max_valid_index = min(axis_size - 1, torch.iinfo(index.dtype).max)
        valid = ((index >= 0) & (index <= max_valid_index)).all()
        message = f"index values must be in [0, {axis_size})"
        if x.is_cuda:
            # Queue the check on the CUDA or ROCm stream. Reading bool(valid) would block the host.
            torch._assert_async(valid, message)
        elif not bool(valid):
            # Report IndexError on other devices. On CPU, _assert_async would raise RuntimeError.
            raise IndexError(message)


class NativeIndexSelectOp:
    """Select distinct slices along dim and return a contiguous result in x.dtype.

    x has rank >= 1 and FP32/FP16/BF16 dtype. index is a same-device int32/int64
    vector [K] shared by all remaining coordinates. The required keyword dim
    accepts negative axes. Forward preserves the bits of every selected element.
    Pass unique_indices=True only for distinct, in-bounds indices. False
    (the default) raises NotImplementedError. Duplicate entries with True
    violate the caller's guarantee and give undefined gradients.

    PyTorch autograd routes one upstream gradient to each selected slice of x
    and leaves unselected slices at zero. Strided and empty tensors are supported.
    """

    op_class = "reduction"

    def __call__(
        self, x: torch.Tensor, index: torch.Tensor, *, dim: int, unique_indices: bool = False
    ) -> torch.Tensor:
        """Select the supplied rows under the caller's uniqueness guarantee."""
        return self.forward(x, index, dim=dim, unique_indices=unique_indices)

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, *, dim: int, unique_indices: bool = False
    ) -> torch.Tensor:
        """Select slices in index order using PyTorch's native autograd."""
        require_unique_indices(unique_indices)
        _validate_inputs(x, index, dim)
        return x.index_select(dim, index)

    def forward_fp32(
        self, x: torch.Tensor, index: torch.Tensor, *, dim: int, unique_indices: bool = False
    ) -> torch.Tensor:
        """Select in FP32 so autograd scatters in FP32 before casting the input gradient."""
        require_unique_indices(unique_indices)
        _validate_inputs(x, index, dim)
        return x.float().index_select(dim, index)
