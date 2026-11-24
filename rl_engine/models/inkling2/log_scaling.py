# SPDX-License-Identifier: Apache-2.0
"""INK-12: full-attention position-dependent logarithmic scaling."""

import torch

LOG_SCALING_FLOOR = 8192
LOG_SCALING_ALPHA = 0.1


def full_attention_tau(positions: torch.Tensor) -> torch.Tensor:
    """FP32 tau from zero-based per-document positions, not packed row indices."""
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a one-dimensional integer tensor")
    if positions.device.type not in ("cpu", "cuda"):
        raise RuntimeError("positions must be on CPU or CUDA/ROCm")
    if bool((positions < 0).any()):
        raise ValueError("positions must be nonnegative")
    # Convert to int64 before adding 1 so int32 cannot wrap.
    ratio = (positions.long() + 1).float() / LOG_SCALING_FLOOR
    return 1.0 + LOG_SCALING_ALPHA * torch.log(ratio.clamp(min=1.0))


def apply_full_attention_log_scaling(
    query: torch.Tensor,
    relative: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_local: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply tau to post-clip Q and relative features; leave K/V untouched.

    The input dtype is the intermediate post-clip rounding boundary.
    Local attention bypasses scaling. Return both tensors in their original
    dtype after an FP32 multiply. The input tensors are never mutated.
    """
    if query.ndim != 3 or relative.ndim != 3 or query.shape[:2] != relative.shape[:2]:
        raise ValueError("Q and relative features must share token/head geometry")
    if query.dtype != relative.dtype or query.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Q and relative features must share BF16 or FP32 dtype")
    if positions.shape != (query.shape[0],) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must match the token dimension")
    if query.device != relative.device or query.device != positions.device:
        raise ValueError("Q, relative features and positions must share a device")
    if query.device.type not in ("cpu", "cuda"):
        raise RuntimeError("expected CPU or CUDA/ROCm")
    if is_local:
        return query, relative
    tau = full_attention_tau(positions).view(-1, 1, 1)
    return (query.float() * tau).to(query.dtype), (relative.float() * tau).to(relative.dtype)
