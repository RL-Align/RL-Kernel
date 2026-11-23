# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini attention O-projection contract (INK-17)."""

from __future__ import annotations

import torch


def attention_o_residual(
    attention: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply deterministic O projection, then add the attention residual.

    The projection reduction runs in FP32 with a fixed hidden-dimension left
    fold. The residual add also occurs in FP32, and the final tensor returns to
    the attention input dtype.
    """
    if attention.ndim < 2 or residual.shape != attention.shape:
        raise ValueError("attention and residual must have the same [..., hidden] shape")
    if (
        weight.ndim != 2
        or weight.shape[1] != attention.shape[-1]
        or weight.shape[0] != residual.shape[-1]
    ):
        raise ValueError("O-projection weight must have shape [hidden, hidden]")
    if attention.device != residual.device or attention.device != weight.device:
        raise ValueError("attention, residual and O-projection weight must share a device")
    if (
        not attention.is_floating_point()
        or not residual.is_floating_point()
        or not weight.is_floating_point()
    ):
        raise TypeError("attention O-projection tensors must be floating point")
    if bias is not None:
        if bias.shape != (weight.shape[0],):
            raise ValueError("O-projection bias must have shape [hidden]")
        if bias.device != attention.device:
            raise ValueError("O-projection bias must share the attention device")

    attention_flat = attention.reshape(-1, attention.shape[-1]).float()
    weight_fp32 = weight.float()
    projected = torch.zeros(
        (attention_flat.shape[0], weight.shape[0]), device=attention.device, dtype=torch.float32
    )
    for hidden_index in range(attention.shape[-1]):
        projected = (
            projected
            + attention_flat[:, hidden_index : hidden_index + 1] * weight_fp32[:, hidden_index]
        )
    if bias is not None:
        projected = projected + bias.float()
    output = projected + residual.reshape(-1, residual.shape[-1]).float()
    return output.reshape_as(residual).to(dtype=attention.dtype)
