# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared selected-logprob drift summaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogprobDriftStats:
    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    active_count: int


def summarize_logprob_drift(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> LogprobDriftStats:
    """Summarize absolute drift, optionally over active rows only."""

    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match reference shape "
            f"{tuple(reference.shape)}"
        )
    diff = (candidate.float() - reference.float()).abs()
    if mask is None:
        values = diff.reshape(-1)
    else:
        if mask.shape != diff.shape:
            raise ValueError("mask shape must match candidate and reference")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be bool")
        values = diff[mask.to(device=diff.device)]

    count = int(values.numel())
    if count == 0:
        return LogprobDriftStats(0.0, 0.0, 0.0, 0.0, 0)
    return LogprobDriftStats(
        max_abs=float(values.max().item()),
        mean_abs=float(values.mean().item()),
        p95_abs=float(torch.quantile(values, 0.95).item()),
        p99_abs=float(torch.quantile(values, 0.99).item()),
        active_count=count,
    )


__all__ = ["LogprobDriftStats", "summarize_logprob_drift"]
