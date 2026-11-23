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
