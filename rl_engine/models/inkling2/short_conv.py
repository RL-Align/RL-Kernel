# SPDX-License-Identifier: Apache-2.0
"""Deterministic causal short-convolution contract for Inkling 2 Mini (INK-10).

The model has four independent kernel-size-four streams for Q, K, V and MLP.
This reference keeps the accumulation order explicit and returns the state needed
for chunked prefill and decode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

import torch

STREAMS = ("q", "k", "v", "mlp")
KERNEL_SIZE = 4


def causal_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a depthwise causal short convolution and return its new history.

    ``x`` has shape ``[batch, time, hidden]`` and ``weight`` has shape
    ``[hidden, 4]``. ``state`` contains the previous three inputs as
    ``[batch, hidden, 3]``. The four tap products are accumulated from oldest
    to newest in FP32 before the result is cast back to the input dtype.
    """
    if x.ndim != 3:
        raise ValueError("short-conv input must have shape [batch, time, hidden]")
    if weight.shape != (x.shape[-1], KERNEL_SIZE):
        raise ValueError(f"short-conv weight must have shape [hidden, {KERNEL_SIZE}]")
    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError("short-conv input and weight must be floating point tensors")
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("short-conv input and weight must share device and dtype")

    batch, time, hidden = x.shape
    if state is None:
        state = torch.zeros((batch, hidden, KERNEL_SIZE - 1), device=x.device, dtype=x.dtype)
    if state.shape != (batch, hidden, KERNEL_SIZE - 1):
        raise ValueError("short-conv state must have shape [batch, hidden, 3]")
    if state.device != x.device or state.dtype != x.dtype:
        raise ValueError("short-conv state must share device and dtype with input")

    history = torch.cat((state, x.transpose(1, 2)), dim=-1)
    output = torch.zeros((batch, hidden, time), device=x.device, dtype=torch.float32)
    weight_fp32 = weight.float()
    for tap in range(KERNEL_SIZE):
        output = output + history[..., tap : tap + time].float() * weight_fp32[:, tap].view(
            1, hidden, 1
        )

    next_state = history[..., -(KERNEL_SIZE - 1) :].contiguous()
    return output.transpose(1, 2).to(dtype=x.dtype), next_state


def causal_short_conv_streams(
    inputs: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    states: Mapping[str, torch.Tensor] | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Apply four independent Q/K/V/MLP short-convolution streams."""
    expected = set(STREAMS)
    if set(inputs) != expected or set(weights) != expected:
        raise ValueError("expected exactly q, k, v and mlp short-conv streams")
    if states is not None and set(states) != expected:
        raise ValueError("short-conv states must contain exactly q, k, v and mlp")

    outputs: Dict[str, torch.Tensor] = {}
    next_states: Dict[str, torch.Tensor] = {}
    for name in STREAMS:
        outputs[name], next_states[name] = causal_short_conv(
            inputs[name], weights[name], None if states is None else states[name]
        )
    return outputs, next_states
