# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image modulation projection and independent precision reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

HIDDEN = 3072


def validate_projection(temb, weight, bias, multiplier):
    if temb.ndim != 2 or temb.shape[0] < 1 or temb.shape[1] != HIDDEN:
        raise ValueError("temb must be nonempty [B,3072]")
    if tuple(weight.shape) != (multiplier * HIDDEN, HIDDEN):
        raise ValueError(f"weight must be [{multiplier * HIDDEN},3072]")
    if tuple(bias.shape) != (multiplier * HIDDEN,):
        raise ValueError(f"bias must be [{multiplier * HIDDEN}]")
    if temb.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("only bfloat16 and float32 are supported")
    if any(t.dtype != temb.dtype for t in (weight, bias)):
        raise TypeError("temb, weight and bias must share a dtype")
    if any(t.device != temb.device for t in (weight, bias)):
        raise ValueError("all inputs must be on the same device")


class _RoundStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        return x.to(dtype).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def round_boundary(x, dtype):
    """Model a declared storage rounding while retaining an FP32/FP64 VJP."""
    return _RoundStraightThrough.apply(x, dtype)


class _ReferenceLinear(torch.autograd.Function):
    """Independent FP32 reference with a balanced, blocked input-gradient sum.

    A single CPU BLAS dot over 6144/18432 terms can itself miss the gradient
    tolerance under cancellation. The 128-output blocks are independent BLAS
    calls, followed by pairwise FP32 addition; no CUDA kernel is used here.
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        parts = [
            grad[:, start : start + 128] @ weight[start : start + 128]
            for start in range(0, weight.shape[0], 128)
        ]
        while len(parts) > 1:
            combined = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
            if len(parts) % 2:
                combined.append(parts[-1])
            parts = combined
        return parts[0], grad.T @ x, grad.sum(0)


def reference_projection(temb, weight, bias, execution_dtype):
    act = round_boundary(F.silu(temb), execution_dtype)
    return round_boundary(_ReferenceLinear.apply(act, weight, bias), execution_dtype)


class QwenImageAdaLNModProjectionReference:
    """One stream: SiLU + Linear, returning two (shift, scale, gate) triples."""

    op_class = "reduction"
    kernel_id = "pytorch.qwenimage.adaln_mod_projection.reference"

    def forward(self, temb, weight, bias):
        validate_projection(temb, weight, bias, 6)
        return F.linear(F.silu(temb), weight, bias)

    __call__ = forward

    def forward_fp32(self, temb, weight, bias):
        validate_projection(temb, weight, bias, 6)
        return reference_projection(temb.float(), weight.float(), bias.float(), temb.dtype)
