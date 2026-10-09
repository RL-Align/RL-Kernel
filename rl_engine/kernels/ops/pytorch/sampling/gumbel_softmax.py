# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch


def _validate_gumbel_softmax_inputs(
    logits: torch.Tensor,
    tau: float,
    gumbels: Optional[torch.Tensor],
) -> None:
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least 2 dimensions, got shape {tuple(logits.shape)}")
    if logits.size(-1) <= 0:
        raise ValueError("logits vocab dimension must be non-empty")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be a floating-point tensor, got dtype {logits.dtype}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if gumbels is not None:
        if gumbels.shape != logits.shape:
            raise ValueError(
                f"gumbels shape {tuple(gumbels.shape)} must match logits shape "
                f"{tuple(logits.shape)}"
            )
        if gumbels.device != logits.device:
            raise ValueError(
                f"gumbels device {gumbels.device} must match logits device {logits.device}"
            )
        if not gumbels.is_floating_point():
            raise TypeError(f"gumbels must be floating-point, got dtype {gumbels.dtype}")


def _sample_gumbels_like(logits: torch.Tensor) -> torch.Tensor:
    # Matches torch.nn.functional.gumbel_softmax's exponential sampling path.
    return (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )


def gumbel_softmax_reference(
    logits: torch.Tensor,
    *,
    tau: float = 1.0,
    hard: bool = False,
    gumbels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_gumbel_softmax_inputs(logits, tau, gumbels)
    noise = (
        _sample_gumbels_like(logits) if gumbels is None else gumbels.detach().to(dtype=logits.dtype)
    )
    y_soft = torch.softmax((logits + noise) / tau, dim=-1)
    if not hard:
        return y_soft

    index = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class NativeGumbelSoftmaxOp:
    """PyTorch reference implementation for differentiable Gumbel-Softmax sampling."""

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        gumbels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(logits, tau=tau, hard=hard, gumbels=gumbels)

    def apply(
        self,
        logits: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        gumbels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return gumbel_softmax_reference(logits, tau=float(tau), hard=bool(hard), gumbels=gumbels)
