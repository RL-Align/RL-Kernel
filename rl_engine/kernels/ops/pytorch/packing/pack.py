# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""PyTorch-native fallback for fused masking + variable-length packing.

During RL training (PPO/DPO/GRPO) only the generated, non-padding tokens
contribute to the loss. Materializing full ``[B, S, ...]`` logits for the
masked-out positions wastes VRAM. This op compacts the active rows of a dense
``[B, S, ...]`` tensor into a contiguous ``[Total_Active, ...]`` tensor (and
scatters gradients back on the backward pass), so downstream loss kernels only
ever touch active tokens.

This is the portable reference path that defines the numerical contract for the
Triton / CUDA / ROCm native kernels (issue #42). The packing order is
row-major over the flattened ``[B, S]`` grid, identical to
``x.reshape(-1, *tail)[mask.reshape(-1)]`` and to
``SyntheticRLKernelBatch.compact_completion_values`` used by the tests.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _validate(x: torch.Tensor, mask: torch.Tensor) -> None:
    if mask.dim() < 1:
        raise ValueError("mask must have at least one dimension.")
    if mask.shape != x.shape[: mask.dim()]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match the leading dims of "
            f"x.shape {tuple(x.shape)} (expected {tuple(x.shape[: mask.dim()])})."
        )


class _PackFunction(torch.autograd.Function):
    """forward: gather active rows; backward: scatter grads to active rows."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor):
        _validate(x, mask)
        lead = mask.dim()
        tail_shape = x.shape[lead:]

        flat_mask = mask.reshape(-1).to(torch.bool)
        flat_x = x.reshape(-1, *tail_shape) if tail_shape else x.reshape(-1)

        index = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        packed = flat_x.index_select(0, index)

        # cu_seqlens: prefix-sum of per-row active counts, for varlen consumers.
        # Count from the bool mask so a non-bool mask (e.g. {0, 2}) matches the
        # number of rows actually packed above (nonzero == active).
        bool_mask = flat_mask.reshape(mask.shape[0], -1)
        per_row_active = bool_mask.to(torch.int64).sum(dim=1)
        cu_seqlens = torch.zeros(mask.shape[0] + 1, dtype=torch.int64, device=x.device)
        torch.cumsum(per_row_active, dim=0, out=cu_seqlens[1:])

        ctx.save_for_backward(index)
        ctx.flat_rows = flat_x.shape[0]
        ctx.tail_shape = tail_shape
        ctx.x_shape = tuple(x.shape)
        ctx.x_dtype = x.dtype
        return packed, cu_seqlens

    @staticmethod
    def backward(ctx, grad_packed: torch.Tensor, grad_cu_seqlens):
        (index,) = ctx.saved_tensors
        tail_shape = ctx.tail_shape

        grad_flat = grad_packed.new_zeros(
            (ctx.flat_rows, *tail_shape) if tail_shape else (ctx.flat_rows,)
        )
        grad_flat.index_copy_(0, index, grad_packed)
        grad_x = grad_flat.reshape(ctx.x_shape)
        # grad w.r.t. mask is undefined (boolean selector).
        return grad_x, None


class NativePackOp:
    """PyTorch-native fused masking + variable-length packing (pack-and-pad).

    Forward packs the active rows of ``x`` (selected by ``mask``) into a
    contiguous ``[Total_Active, *tail]`` tensor and returns the per-row
    ``cu_seqlens`` prefix-sum. Backward scatters the upstream gradient back to
    the original ``[B, S, *tail]`` layout, leaving zeros at inactive positions.
    """

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _PackFunction.apply(x, mask)

    @staticmethod
    def unpack(
        packed: torch.Tensor,
        mask: torch.Tensor,
        *,
        tail_shape: Tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Scatter a packed ``[Total_Active, *tail]`` tensor back to a dense
        ``[*mask.shape, *tail]`` tensor with zeros at inactive positions.

        This is the explicit (non-autograd) inverse used by diagnostics; the
        backward pass of :class:`_PackFunction` performs the same scatter.
        """
        flat_mask = mask.reshape(-1).to(torch.bool)
        tail = tuple(packed.shape[1:]) if tail_shape is None else tuple(tail_shape)
        out = packed.new_zeros((flat_mask.numel(), *tail))
        index = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        out.index_copy_(0, index, packed)
        return out.reshape(*mask.shape, *tail)
