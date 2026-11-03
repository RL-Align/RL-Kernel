# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tensor and rounding contract for the shared P5 LoRA delta (RFC #520, item 45)."""

from __future__ import annotations

import math
from numbers import Real

import torch

ROCM_LORA_PROFILE = "dsv4-lora-rocm-fp32-v1"


def validate_lora_inputs(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> float:
    """Validate BF16 X[M,K], A[R,K], B[N,R]; alpha is the effective multiplier."""
    for name, tensor in (("x", x), ("a", a), ("b", b)):
        if tensor.layout != torch.strided or tensor.ndim != 2:
            raise ValueError(f"LoRA {name} must be a strided 2-D tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"LoRA {name} must be BF16, got {tensor.dtype}")
        if tensor.device != x.device:
            raise ValueError("LoRA operands must share one device")
    if min(x.shape[1], a.shape[0], b.shape[0]) <= 0:
        raise ValueError("LoRA K, rank and output width must be positive")
    if a.shape[1] != x.shape[1] or b.shape[1] != a.shape[0]:
        raise ValueError("LoRA expects X[M,K], A[R,K] and B[N,R]")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("LoRA alpha must be a real scalar, not a trainable tensor")
    value = float(alpha)
    if not math.isfinite(value) or abs(value) > torch.finfo(torch.float32).max:
        raise ValueError("LoRA alpha must be finite and representable in FP32")
    return value


def validate_lora_backward(
    dy: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, u: torch.Tensor
) -> None:
    if dy.layout != torch.strided or dy.shape != (x.shape[0], b.shape[0]):
        raise ValueError("LoRA dY must have shape [M,N]")
    if dy.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("LoRA dY must be BF16 or FP32")
    if u.layout != torch.strided or u.shape != (x.shape[0], a.shape[0]):
        raise ValueError("LoRA saved U must have shape [M,R]")
    if u.dtype != torch.bfloat16:
        raise TypeError("LoRA saved U must be BF16")
    if dy.device != x.device or u.device != x.device:
        raise ValueError("LoRA dY and U must be on the operands' device")
