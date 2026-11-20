# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5 projection boundary: add the shared LoRA delta to an existing base output."""

from __future__ import annotations

import torch

from rl_engine.ops.moe.lora_delta import shared_grouped_lora_delta


def lora_projection(
    x: torch.Tensor,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    alpha: float,
    *,
    base_output: torch.Tensor | None = None,
    recompute: bool = False,
) -> torch.Tensor:
    """Return base-only, LoRA-only or FP32 base+delta before the next P5 boundary.

    The caller owns the frozen quantized base and its input-gradient path.
    Do not detach base_output: frozen base weights still contribute to dX.
    FC1 leaves the result FP32 for activation; FC2's final cast belongs to
    the enclosing expert module. No route weight is applied here.
    """
    if (a is None) != (b is None):
        raise ValueError("LoRA A and B must either both be present or both be absent")
    if base_output is not None:
        if x.ndim != 2 or base_output.ndim != 2 or base_output.shape[0] != x.shape[0]:
            raise ValueError("base_output must have shape [M,N] with the same M as X")
        if base_output.dtype != torch.float32:
            raise TypeError("base_output must be FP32; do not round before adding LoRA")
        if base_output.device != x.device:
            raise ValueError("base_output and LoRA must share one device")
    if a is None or b is None:
        if base_output is None:
            raise ValueError("projection requires a base output or a LoRA adapter")
        return base_output
    if base_output is not None and (b.ndim != 2 or base_output.shape[1] != b.shape[0]):
        raise ValueError("base_output width must equal the LoRA output width")
    delta = shared_grouped_lora_delta(x, a, b, alpha, recompute=recompute)
    return delta if base_output is None else base_output + delta
