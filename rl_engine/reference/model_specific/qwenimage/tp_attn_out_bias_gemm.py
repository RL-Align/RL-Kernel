# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image row-parallel output geometry and independent CPU arithmetic."""

from __future__ import annotations

import torch

from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import HIDDEN
from rl_engine.reference.model_specific.qwenimage.tp_qkv_bias_gemm import TP_SIZES


def validate_attn_out(x, weight, bias, tp_world=1):
    if type(tp_world) is not int or tp_world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    width = HIDDEN // tp_world
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != width:
        raise ValueError(f"x must be nonempty [M,{width}] for TP={tp_world}")
    if tuple(weight.shape) != (HIDDEN, width):
        raise ValueError(f"weight must be [3072,{width}] for TP={tp_world}")
    if bias is not None and tuple(bias.shape) != (HIDDEN,):
        raise ValueError("bias must be [3072]")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("only bfloat16 and float32 are supported")
    tensors = (weight,) if bias is None else (weight, bias)
    if any(t.dtype != x.dtype for t in tensors):
        raise TypeError("x, weight and bias must share a dtype")
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs must be on the same device")


def shard_attn_out_parameters(weight, bias, *, tp_world, tp_rank):
    """Slice contiguous whole-head input columns; bias stays replicated."""
    validate_attn_out(weight.new_empty((1, HIDDEN)), weight, bias)
    if type(tp_world) is not int or tp_world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_world:
        raise ValueError("TP rank must be in [0, TP size)")
    width = HIDDEN // tp_world
    return weight[:, tp_rank * width : (tp_rank + 1) * width].contiguous(), bias


def tree_gemm_fp32(a, b):
    """CPU gold: 32 ascending FMA steps per leaf, midpoint merges in FP32."""
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("attention output reference requires CPU tensors")
    a, b = a.float(), b.float()

    def subtree(begin, end):
        if end - begin == 1:
            acc = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
            for index in range(begin * 32, min((begin + 1) * 32, a.shape[1])):
                acc = torch.addcmul(acc, a[:, index, None], b[None, :, index])
            return acc
        middle = (begin + end) // 2
        return subtree(begin, middle) + subtree(middle, end)

    return subtree(0, (a.shape[1] + 31) // 32)


class _Reference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        result = tree_gemm_fp32(x, weight)
        if bias is not None:
            result = result + bias.float()
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        from rl_engine.ops.autograd.vjp_fp32 import row_local_bias_fp32

        x, weight = ctx.saved_tensors
        g, rows = grad.float(), x.float()
        dx = tree_gemm_fp32(g, weight.t()).to(x.dtype)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        for row in range(x.shape[0]):
            dw = torch.addcmul(dw, g[row, :, None], rows[row, None, :])
        db = row_local_bias_fp32(g).to(x.dtype) if ctx.has_bias else None
        return dx, dw.to(weight.dtype), db


class QwenImageTPAttnOutBiasGemmReference:
    """Full single-GPU contract evaluated on CPU, without CUDA implementation calls.

    CPU addcmul is separately checked against the scalar libm fmaf oracle.
    The explicit tree and row fold specify the FP32 rounding sequence.
    """

    op_class = "reduction"
    kernel_id = "pytorch.cpu.qwenimage.tp_attn_out_bias_gemm.reference"

    def forward(self, x, weight, bias=None):
        validate_attn_out(x, weight, bias)
        if x.device.type != "cpu":
            raise ValueError("attention output reference requires CPU tensors")
        return _Reference.apply(x, weight, bias)

    __call__ = forward

    def forward_fp32(self, x, weight, bias=None):
        validate_attn_out(x, weight, bias)
        result = tree_gemm_fp32(x, weight)
        return result if bias is None else result + bias.float()


__all__ = [
    "QwenImageTPAttnOutBiasGemmReference",
    "shard_attn_out_parameters",
    "validate_attn_out",
]
