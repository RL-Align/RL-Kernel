# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini shared-loop weight-gradient contract (INK-21)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


def accumulate_shared_weight_gradients(
    gradients: Mapping[int, torch.Tensor],
    *,
    visit_order: Iterable[int] = (0, 1, 2),
    shard: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Accumulate repeated physical-weight gradients in a declared order.

    Gradients are folded in FP32 over the requested logical visits. ``shard``
    models the local slice retained after a deterministic reduce-scatter.
    """
    order = tuple(int(visit) for visit in visit_order)
    if not order:
        raise ValueError("visit_order must contain at least one visit")
    missing = [visit for visit in order if visit not in gradients]
    if missing:
        raise ValueError(f"missing gradients for visits {missing}")
    first = gradients[order[0]]
    if not first.is_floating_point():
        raise TypeError("shared-weight gradients must be floating point")
    for visit in order:
        gradient = gradients[visit]
        if gradient.shape != first.shape or gradient.device != first.device:
            raise ValueError("all shared-weight gradients must share shape and device")
        if not gradient.is_floating_point():
            raise TypeError("shared-weight gradients must be floating point")

    accumulated = torch.zeros_like(first, dtype=torch.float32)
    for visit in order:
        accumulated = accumulated + gradients[visit].float()
    if shard is not None:
        start, end = shard
        if start < 0 or end < start or end > accumulated.shape[0]:
            raise ValueError("reduce-scatter shard must fit the first gradient dimension")
        accumulated = accumulated[start:end]
    return accumulated.to(dtype=first.dtype)


class _NativeSharedWeightGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradients, shard_start, shard_end):
        from rl_engine.backends.extension import _C

        output = _C.inkling2_shared_weight_grad_forward(
            gradients.contiguous(), int(shard_start), int(shard_end)
        )
        ctx.save_for_backward(gradients.contiguous())
        ctx.shard_start = int(shard_start)
        ctx.shard_end = int(shard_end)
        ctx.input_shape = tuple(gradients.shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from rl_engine.backends.extension import _C

        (gradients,) = ctx.saved_tensors
        d_gradients = _C.inkling2_shared_weight_grad_backward(
            grad_output.contiguous(),
            int(gradients.shape[0]),
            int(gradients.shape[1]),
            int(gradients.numel() // (gradients.shape[0] * gradients.shape[1])),
            ctx.shard_start,
            ctx.shard_end,
        ).reshape(ctx.input_shape)
        return d_gradients, None, None


def accumulate_shared_weight_gradients_cuda(
    gradients: Mapping[int, torch.Tensor],
    *,
    visit_order: Iterable[int] = (0, 1, 2),
    shard: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Run the native CUDA deterministic shared-weight gradient fold."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_shared_weight_grad_forward"):
        raise RuntimeError("Inkling 2 shared-weight CUDA extension is unavailable")
    order = tuple(int(visit) for visit in visit_order)
    if not order:
        raise ValueError("visit_order must contain at least one visit")
    missing = [visit for visit in order if visit not in gradients]
    if missing:
        raise ValueError(f"missing gradients for visits {missing}")
    first = gradients[order[0]]
    if not first.is_cuda or not first.is_floating_point() or first.ndim < 2:
        raise ValueError(
            "native shared-weight gradients must be CUDA floating tensors [visits, rows, ...]"
        )
    for visit in order:
        gradient = gradients[visit]
        if (
            gradient.shape != first.shape
            or gradient.device != first.device
            or gradient.dtype != first.dtype
        ):
            raise ValueError("all shared-weight gradients must share shape, device, and dtype")
    full_rows = first.shape[0]
    if shard is None:
        shard_start, shard_end = 0, full_rows
    else:
        shard_start, shard_end = (int(shard[0]), int(shard[1]))
        if shard_start < 0 or shard_end < shard_start or shard_end > full_rows:
            raise ValueError("reduce-scatter shard must fit the first gradient dimension")
    stacked = torch.stack([gradients[visit] for visit in order], dim=0)
    return _NativeSharedWeightGradFunction.apply(stacked, shard_start, shard_end)
