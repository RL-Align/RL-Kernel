# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini expert gate/up arithmetic contract (INK-26)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softcap(value: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply the declared smooth ``cap * tanh(value / cap)`` transform."""
    if cap <= 0:
        raise ValueError("softcap must be positive")
    return float(cap) * torch.tanh(value / float(cap))


def expert_gate_up(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    *,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Compute the FP32 softcapped SwiGLU gate/up projection.

    The public tensor keeps the input dtype, while both linear projections,
    softcaps and elementwise multiplication run in FP32. This gives the CUDA
    and ROCm paths one explicit numerical contract and leaves backward to
    PyTorch autograd.
    """
    if hidden.ndim < 2:
        raise ValueError("expert hidden states must have shape [..., hidden]")
    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate and up weights must share shape [intermediate, hidden]")
    if hidden.shape[-1] != gate_weight.shape[-1]:
        raise ValueError("expert hidden size does not match gate/up weights")
    for name, bias in (("gate", gate_bias), ("up", up_bias)):
        if bias is not None and bias.shape != (gate_weight.shape[0],):
            raise ValueError(f"{name} bias must have shape [intermediate]")
    devices = [hidden.device, gate_weight.device, up_weight.device]
    if gate_bias is not None:
        devices.append(gate_bias.device)
    if up_bias is not None:
        devices.append(up_bias.device)
    if any(device != hidden.device for device in devices):
        raise ValueError("expert gate/up tensors must share a device")

    hidden_fp32 = hidden.float()
    gate = F.linear(
        hidden_fp32, gate_weight.float(), None if gate_bias is None else gate_bias.float()
    )
    up = F.linear(hidden_fp32, up_weight.float(), None if up_bias is None else up_bias.float())
    return (F.silu(softcap(gate, gate_cap)) * softcap(up, up_cap)).to(hidden.dtype)
