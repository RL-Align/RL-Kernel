# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""PyTorch reference for H3's selection after the FP32 video/audio heads."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
INDEX_DTYPES = (torch.int32, torch.int64)


def validate_packed_index_select(x: torch.Tensor, index: torch.Tensor) -> None:
    """Check [B,S,D] / [K] metadata and range, allowing strided and empty tensors.

    CUDA range errors are asynchronous device assertions, like index_select.
    The CPU reference raises IndexError synchronously.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape [B, S, D]")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be one of {SUPPORTED_DTYPES}, got {x.dtype}")
    if index.ndim != 1:
        raise ValueError("index must have shape [K]")
    if index.dtype not in INDEX_DTYPES:
        raise TypeError(f"index dtype must be one of {INDEX_DTYPES}, got {index.dtype}")
    if x.device != index.device:
        raise ValueError("x and index must be on the same device")
    if index.numel():
        valid = ((index >= 0) & (index < x.shape[1])).all()
        message = f"index values must be in [0, {x.shape[1]})"
        if x.is_cuda:
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise IndexError(message)


class _OrderedSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        ctx.save_for_backward(index)
        ctx.input_shape = x.shape
        return x.index_select(1, index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        # Independent executable contract: visit output positions in increasing j.
        # Use CPU tensors to avoid CUDA atomics and one device launch per position.
        grad32 = grad.detach().to(device="cpu", dtype=torch.float32)
        result = torch.zeros(ctx.input_shape, dtype=torch.float32)
        for position, row in enumerate(index.cpu().tolist()):
            result[:, row, :] += grad32[:, position, :]
        return result.to(device=grad.device, dtype=grad.dtype), None


class _AccuracySelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        ctx.save_for_backward(index)
        ctx.input_shape = x.shape
        return x.float().index_select(1, index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        # An independent CPU FP64 scatter-add checks accuracy, not bitwise order.
        result = torch.zeros(ctx.input_shape, dtype=torch.float64)
        result.index_add_(1, index.cpu().long(), grad.detach().double().cpu())
        return result.to(device=grad.device, dtype=grad.dtype), None


class NativePackedIndexSelectOp:
    """Select ``x[:, index, :]`` with a fixed-order FP32 input VJP.

    Inputs are FP32/FP16/BF16 [B,S,D] and same-device int32/int64 [K].
    Output is contiguous [B,K,D], preserving input dtype and every copied bit.
    Repeated indices accumulate in increasing output position, independently
    for each batch and channel: start at FP32 +0, round each addition to FP32,
    then cast once to the input dtype. Unselected rows receive +0.
    This reference intentionally copies gradients to CPU; it is not a baseline
    for performance. Only first-order autograd is supported.
    """

    op_class = "reduction"

    def __call__(self, x, index):
        return self.forward(x, index)

    def forward(self, x, index):
        validate_packed_index_select(x, index)
        return _OrderedSelect.apply(x, index)

    def forward_fp32(self, x, index):
        """FP32 output with independent CPU FP64 VJP for accuracy validation."""
        validate_packed_index_select(x, index)
        return _AccuracySelect.apply(x, index)
