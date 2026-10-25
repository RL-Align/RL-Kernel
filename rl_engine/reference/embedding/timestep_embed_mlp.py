# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 product-sum gold for the standard Qwen-Image timestep MLP."""

import math

import torch
from torch import nn


def validate(timestep, weight1, bias1, weight2, bias2):
    """Validate module-boundary values without changing their dtype or device."""
    values = (timestep, weight1, bias1, weight2, bias2)
    if any(x.device != timestep.device for x in values):
        raise ValueError("all inputs must share a device")
    if timestep.ndim != 1 or weight1.ndim != 2 or weight1.shape[1] != 256:
        raise ValueError("expected timestep [B] and weight1 [H, 256]")
    h = weight1.shape[0]
    if h < 1 or weight2.shape != (h, h) or bias1.shape != (h,) or bias2.shape != (h,):
        raise ValueError("expected weight2 [H,H] and biases [H], H > 0")
    if weight1.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("MLP parameters must be float32 or bfloat16")
    if any(x.dtype != weight1.dtype for x in values[1:]):
        raise TypeError("MLP parameters must share dtype")
    if timestep.dtype not in (torch.float32, torch.bfloat16, torch.int32, torch.int64):
        raise TypeError("timestep must be fp32, bf16, int32 or int64")
    if torch.is_autocast_enabled(timestep.device.type):
        raise ValueError("explicit FP32 accumulation requires autocast disabled")


def _compensated_sum(terms):
    """Independent pure-torch FP32 two-sum tree along the final dimension."""
    size = terms.shape[-1]
    padded = 1 << max(0, size - 1).bit_length()
    hi = torch.nn.functional.pad(terms, (0, padded - size))
    lo = torch.zeros_like(hi)
    while hi.shape[-1] > 1:
        a, b = hi[..., ::2], hi[..., 1::2]
        total = a + b
        recovered = total - a
        error = (a - (total - recovered)) + (b - recovered)
        tail = (lo[..., ::2] + lo[..., 1::2]) + error
        hi = total + tail
        lo = tail - (hi - total)
    return (hi + lo).squeeze(-1)


class _ReferenceLinear(torch.autograd.Function):
    """FP32 reference linear with explicit independent VJP reductions.

    Compensated sums avoid making a CPU BLAS reduction accident the oracle for
    ill-conditioned timestep gradients. No CUDA/Triton primitives are shared.
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        if not x.shape[0]:
            return x.new_empty((0, weight.shape[0]))
        return torch.stack([_compensated_sum(weight * row) + bias for row in x])

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            transposed = weight.T.contiguous()
            gx = (
                torch.stack([_compensated_sum(transposed * row) for row in grad])
                if len(grad)
                else torch.zeros_like(x)
            )
        if ctx.needs_input_grad[1]:
            gw = torch.zeros_like(weight)
            for g, row in zip(grad, x, strict=True):
                gw = gw + g[:, None] * row[None, :]
        if ctx.needs_input_grad[2]:
            gb = grad.sum(0)
        return gx, gw, gb


class NativeTimestepEmbedMLPOp(nn.Module):
    """Gold with independent compensated FP32 linear VJPs and torch sigmoid autograd.

    BF16 describes parameter/output storage; fused intermediates remain FP32.
    No fixed-order parameter-gradient guarantee is made for this eager gold.
    """

    def forward_fp32(self, timestep, weight1, bias1, weight2, bias2):
        validate(timestep, weight1, bias1, weight2, bias2)
        # Constants are ordinary FP32 construction, not a platform fingerprint.
        exponent = -math.log(10000.0) * torch.arange(128, dtype=torch.float32)
        frequency = torch.exp(exponent / 128).to(timestep.device)
        phase = (timestep.float()[:, None] * frequency[None, :]) * 1000.0
        embedding = torch.cat((phase.cos(), phase.sin()), dim=1)
        w1, b1, w2, b2 = (x.float() for x in (weight1, bias1, weight2, bias2))
        if timestep.numel() == 0:
            # Keep all five floating leaves connected for empty-batch backward.
            zero = sum(x.sum() * 0 for x in (embedding, w1, b1, w2, b2))
            return embedding.new_empty((0, w1.shape[0])) + zero
        z = _ReferenceLinear.apply(embedding, w1, b1)
        h = z * torch.sigmoid(z)
        return _ReferenceLinear.apply(h, w2, b2)

    def forward(self, timestep, weight1, bias1, weight2, bias2):
        return self.forward_fp32(timestep, weight1, bias1, weight2, bias2).to(weight1.dtype)
