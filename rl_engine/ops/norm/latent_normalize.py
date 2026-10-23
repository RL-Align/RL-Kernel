# SPDX-License-Identifier: Apache-2.0
"""Input contract for the Qwen-Image Phase A latent affine transform."""

import torch

CONTRACT_ID = "latent-normalize-fp32-v1"


def validate_latents(x: torch.Tensor, mean: torch.Tensor, std_inv: torch.Tensor) -> None:
    """Validate metadata without device synchronization or implicit layout copies.

    Statistics are fixed model buffers, not trainable parameters. Their values
    must be finite, with positive std_inv. Validation of model buffer values is
    the caller's responsibility (once at model load, outside the hot path).
    """
    if x.layout != torch.strided or x.ndim not in (4, 5) or x.shape[1] != 16:
        raise ValueError("x must have shape [B,16,H,W] or [B,16,1,H,W]")
    if x.ndim == 5 and x.shape[2] != 1:
        raise ValueError("Phase A supports one latent frame only")
    if any(size <= 0 for size in x.shape):
        raise ValueError("latent dimensions must be positive")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous in channel-first order")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("x must be float32 or bfloat16")
    for name, stat in (("mean", mean), ("std_inv", std_inv)):
        if stat.shape != (16,) or not stat.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [16] vector")
        if stat.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if stat.device != x.device:
            raise ValueError(f"{name} and x must be on the same device")
        if stat.requires_grad:
            raise ValueError(f"{name} must be a fixed model buffer (requires_grad=False)")
