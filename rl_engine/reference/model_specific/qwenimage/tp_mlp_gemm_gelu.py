# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image MLP geometry, channel sharding and independent CPU reference."""

from __future__ import annotations

import torch

from rl_engine.reference.gemm.mlp_up_gemm_gelu import (
    gelu_tanh_fp32,
    left_fold_bias_gradient,
    left_fold_weight_gradient,
    mlp_up_gemm_gelu_reference_gate,
    tree_gemm,
)
from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import HIDDEN
from rl_engine.reference.model_specific.qwenimage.tp_qkv_bias_gemm import TP_SIZES

INTERMEDIATE = 4 * HIDDEN


def validate_mlp(x, up_weight, up_bias, down_weight, down_bias, tp_world=1):
    if type(tp_world) is not int or tp_world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    width = INTERMEDIATE // tp_world
    if x.ndim != 2 or not 1 <= x.shape[0] <= 65535 * 64 or x.shape[1] != HIDDEN:
        raise ValueError("x must have shape [M,3072] with 1 <= M <= 4194240")
    shapes = ((width, HIDDEN), (width,), (HIDDEN, width), (HIDDEN,))
    for tensor, shape in zip((up_weight, up_bias, down_weight, down_bias), shapes):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"MLP parameter must have shape {shape} for TP={tp_world}")
        if tensor.device != x.device or tensor.dtype != x.dtype:
            raise ValueError("MLP inputs and parameters must share a device and dtype")
    if x.dtype != torch.bfloat16:
        raise TypeError("Qwen-Image TP MLP requires bfloat16 operands")


def shard_mlp_parameters(up_weight, up_bias, down_weight, down_bias, *, tp_world, tp_rank):
    """Copy matching up rows/down columns into compact per-rank parameters."""
    validate_mlp(up_weight.new_empty((1, HIDDEN)), up_weight, up_bias, down_weight, down_bias)
    if type(tp_world) is not int or tp_world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_world:
        raise ValueError("TP rank must be in [0, TP size)")
    width = INTERMEDIATE // tp_world
    owned = slice(tp_rank * width, (tp_rank + 1) * width)
    return (
        up_weight[owned].clone(memory_format=torch.contiguous_format),
        up_bias[owned].clone(memory_format=torch.contiguous_format),
        down_weight[:, owned].clone(memory_format=torch.contiguous_format),
        down_bias.clone(memory_format=torch.contiguous_format),
    )


class _Reference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_weight, up_bias, down_weight, down_bias):
        pre = tree_gemm(x, up_weight.t()) + up_bias.float()
        hidden = gelu_tanh_fp32(pre).to(x.dtype)
        ctx.save_for_backward(x, up_weight, down_weight, hidden, pre)
        return (tree_gemm(hidden, down_weight.t()) + down_bias.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, up_weight, down_weight, hidden, pre = ctx.saved_tensors
        dh = tree_gemm(grad, down_weight).to(x.dtype)
        gate = mlp_up_gemm_gelu_reference_gate(pre, dh)
        return (
            tree_gemm(gate, up_weight).to(x.dtype),
            left_fold_weight_gradient(gate, x).to(x.dtype),
            left_fold_bias_gradient(gate).to(x.dtype),
            left_fold_weight_gradient(grad, hidden).to(x.dtype),
            left_fold_bias_gradient(grad).to(x.dtype),
        )


class QwenImageTPMlpGemmGeluReference:
    """Evaluate the complete single-rank MLP on CPU with explicit FP32 arithmetic."""

    op_class = "reduction"
    kernel_id = "pytorch.cpu.qwenimage.tp_mlp_gemm_gelu.reference"

    def forward(self, x, up_weight, up_bias, down_weight, down_bias):
        validate_mlp(x, up_weight, up_bias, down_weight, down_bias)
        if x.device.type != "cpu":
            raise ValueError("MLP reference requires CPU tensors")
        return _Reference.apply(x, up_weight, up_bias, down_weight, down_bias)

    __call__ = forward
