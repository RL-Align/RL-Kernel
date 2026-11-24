# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 last-axis reduction independent of leading row count."""

import torch


def fixed_sum_last(values: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair tree, right zero padding at odd levels, keep last dim."""
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("expected a nonempty reduction axis")
    work = values.float()
    while work.shape[-1] > 1:
        if work.shape[-1] % 2:
            work = torch.cat((work, torch.zeros_like(work[..., :1])), dim=-1)
        work = work[..., ::2] + work[..., 1::2]
    return work
