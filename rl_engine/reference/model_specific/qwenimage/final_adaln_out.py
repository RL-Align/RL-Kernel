# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image final LayerNorm modulation; excludes the output projection."""

import torch
import torch.nn.functional as F

from .adaln_mod_projection import HIDDEN, reference_projection, round_boundary, validate_projection

EPS = 1e-6


def validate_final(x, temb, weight, bias):
    validate_projection(temb, weight, bias, 2)
    if x.ndim != 3 or x.shape[0] != temb.shape[0] or x.shape[1] < 1 or x.shape[2] != HIDDEN:
        raise ValueError("x must be [B,S,3072], S > 0, with the temb batch size")
    if x.dtype != temb.dtype:
        raise TypeError("x and temb must share a dtype")
    if x.device != temb.device:
        raise ValueError("all inputs must be on the same device")


def _reference_token_sum(values):
    # Independent sequential Neumaier sum, not the CUDA 32-token tree.
    # Keep the requested FP32/FP64 arithmetic; long cancellation-heavy sums
    # otherwise make the reference itself an unreliable tolerance oracle.
    total = torch.zeros_like(values[:, 0])
    correction = torch.zeros_like(total)
    for value in values.unbind(1):
        next_total = total + value
        error = torch.where(
            total.abs() >= value.abs(),
            (total - next_total) + value,
            (value - next_total) + total,
        )
        correction = correction + error
        total = next_total
    return total + correction


class _ReferenceModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, norm, scale, shift, dtype):
        gain = round_boundary(1 + scale, dtype)
        ctx.save_for_backward(norm, gain)
        product = round_boundary(norm * gain[:, None, :], dtype)
        return round_boundary(product + shift[:, None, :], dtype)

    @staticmethod
    def backward(ctx, grad):
        norm, gain = ctx.saved_tensors
        return (
            grad * gain[:, None, :],
            _reference_token_sum(grad * norm),
            _reference_token_sum(grad),
            None,
        )


def reference_final(x, temb, weight, bias, execution_dtype):
    scale, shift = reference_projection(temb, weight, bias, execution_dtype).chunk(2, -1)
    norm = round_boundary(F.layer_norm(x, (HIDDEN,), eps=EPS), execution_dtype)
    return _ReferenceModulate.apply(norm, scale, shift, execution_dtype)


class QwenImageFinalAdaLNOutReference:
    op_class = "reduction"
    kernel_id = "pytorch.qwenimage.final_adaln_out.reference"

    def forward(self, x, temb, weight, bias):
        validate_final(x, temb, weight, bias)
        scale, shift = F.linear(F.silu(temb), weight, bias).chunk(2, -1)
        return F.layer_norm(x, (HIDDEN,), eps=EPS) * (1 + scale)[:, None, :] + shift[:, None, :]

    __call__ = forward

    def forward_fp32(self, x, temb, weight, bias):
        validate_final(x, temb, weight, bias)
        return reference_final(x.float(), temb.float(), weight.float(), bias.float(), x.dtype)
