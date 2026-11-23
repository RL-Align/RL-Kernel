# SPDX-License-Identifier: Apache-2.0
"""BF16 projection / FP32 blend reference for Inkling 2 Mini cross-pass gates (INK-20).

The released checkpoint has independent rank-64 controllers for attention-Q,
K, V and MoE. TorchTitan calls these attn, k, v and mlp.
This is a reference contract, not a registered CUDA or ROCm kernel.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

STREAMS = ("attn", "k", "v", "mlp")
_RMS_EPS = 1e-6


@dataclass(frozen=True)
class GateWeights:
    down: torch.Tensor  # [rank, 2 * hidden]
    gain: torch.Tensor  # [hidden, rank]
    forget: torch.Tensor  # [hidden, rank]


def blend_cross_pass_message(
    destination: torch.Tensor,
    message: torch.Tensor,
    weights: GateWeights,
) -> torch.Tensor:
    """Match Miles' BF16 controller projections and FP32 output arithmetic.

    (1 - tanh(forget)) * destination + tanh(gain) * message.
    The two inputs are independently RMS-normalized before concatenation,
    and the linear projections execute in BF16 rather than FP32.
    """
    if destination.ndim != 2 or message.shape != destination.shape:
        raise ValueError("destination and message must have matching [tokens, hidden] shape")
    hidden = destination.shape[-1]
    rank = weights.down.shape[0] if weights.down.ndim == 2 else None
    if rank != 64 or weights.down.shape != (64, 2 * hidden):
        raise ValueError("Inkling 2 Mini gate requires rank-64 down projection")
    if weights.gain.shape != (hidden, 64) or weights.forget.shape != (hidden, 64):
        raise ValueError("gate gain/forget must have shape [hidden, 64]")
    if any(
        t.device != destination.device
        for t in (message, weights.down, weights.gain, weights.forget)
    ):
        raise ValueError("gate inputs and weights must share a device")

    pair = torch.stack((destination.float(), message.float()), dim=1)
    normalized = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + _RMS_EPS)
    projected = F.linear(
        normalized.flatten(1).to(torch.bfloat16), weights.down.to(torch.bfloat16)
    )
    gain = F.linear(projected, weights.gain.to(torch.bfloat16)).float()
    forget = F.linear(projected, weights.forget.to(torch.bfloat16)).float()
    return (1.0 - torch.tanh(forget)) * destination.float() + torch.tanh(gain) * message.float()


def blend_loop_streams(
    destinations: Mapping[str, torch.Tensor],
    messages: Mapping[str, torch.Tensor],
    controllers: Mapping[str, GateWeights],
) -> dict[str, torch.Tensor]:
    """Apply four independent controllers; never share their weights."""
    expected = set(STREAMS)
    if set(destinations) != expected or set(messages) != expected or set(controllers) != expected:
        raise ValueError("expected exactly attn, k, v and mlp loop streams")
    return {
        name: blend_cross_pass_message(destinations[name], messages[name], controllers[name])
        for name in STREAMS
    }
