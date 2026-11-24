# SPDX-License-Identifier: Apache-2.0
"""INK-11: learned relative-logit projection and distance lookup."""

import torch

REL_DIM = 16
REL_EXTENT = 1024


def project_relative_logits(
    relative: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Project [tokens, heads, 16] with [16, 1024] weights.

    The input has already passed INK-07/12 clipping and scaling. Cast the
    weight to the compute dtype, fold products in FP32 in channel order, then
    round once to the input dtype. This is an eager portable candidate, not a
    fused kernel or a certified canonical parameter-gradient implementation.
    """
    if relative.ndim != 3 or relative.shape[-1] != REL_DIM or relative.numel() == 0:
        raise ValueError("expected nonempty [tokens, heads, 16] relative input")
    if weight.shape != (REL_DIM, REL_EXTENT) or weight.device != relative.device:
        raise ValueError("expected [16, 1024] weight on the relative-input device")
    if relative.dtype not in (torch.bfloat16, torch.float32) or weight.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("relative projection requires BF16 or FP32 tensors")
    if relative.device.type not in ("cpu", "cuda"):
        raise RuntimeError("relative projection supports CPU and CUDA/ROCm")
    r = relative.float()
    w = weight.to(relative.dtype).float()
    out = r[..., 0:1] * w[0]
    for channel in range(1, REL_DIM):
        out = out + r[..., channel : channel + 1] * w[channel]
    return out.to(relative.dtype)


def lookup_relative_bias(
    logits: torch.Tensor, query_positions: torch.Tensor, key_positions: torch.Tensor
) -> torch.Tensor:
    """Return [queries, heads, keys] bias for a bounded attention tile.

    Distance is query position minus key position. Future keys and distances
    >= extent receive zero bias, NOT the last table entry. Causality and
    packed-document masking belong to the attention operator.
    """
    if logits.ndim != 3 or logits.shape[-1] != REL_EXTENT:
        raise ValueError("expected [queries, heads, 1024] relative logits")
    if query_positions.shape != (logits.shape[0],) or key_positions.ndim != 1:
        raise ValueError("query/key positions must be one-dimensional")
    for positions in (query_positions, key_positions):
        if positions.device != logits.device or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be integer tensors on the logits device")
    distance = query_positions.long()[:, None] - key_positions.long()[None, :]
    valid = (distance >= 0) & (distance < REL_EXTENT)
    index = distance.clamp(0, REL_EXTENT - 1)[:, None, :].expand(-1, logits.shape[1], -1)
    bias = logits.gather(-1, index)
    return torch.where(valid[:, None, :], bias, torch.zeros_like(bias))
