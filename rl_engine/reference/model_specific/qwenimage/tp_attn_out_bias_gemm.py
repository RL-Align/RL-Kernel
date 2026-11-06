# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image row-parallel output geometry and WS1 CPU tree reference."""

from __future__ import annotations

import torch

from rl_engine.reference.gemm.attn_out_bias_gemm import NativeAttnOutBiasGemmOp
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


class QwenImageTPAttnOutBiasGemmReference:
    """Full CPU WS1 tree reference; GPU strict execution uses native kernels.

    The CPU restriction avoids PyTorch 2.4 CUDA addcmul, which fails the WS1
    FMA oracle in the qualified CI environment. The CPU primitive is checked
    independently against libm fmaf.
    """

    op_class = "reduction"
    kernel_id = "pytorch.cpu.qwenimage.tp_attn_out_bias_gemm.reference"

    def __init__(self):
        self.reference = NativeAttnOutBiasGemmOp()

    def forward(self, x, weight, bias=None):
        validate_attn_out(x, weight, bias)
        if x.device.type != "cpu":
            raise ValueError("attention output reference requires CPU tensors")
        return self.reference(x, weight, bias=bias)

    __call__ = forward

    def forward_fp32(self, x, weight, bias=None):
        validate_attn_out(x, weight, bias)
        if x.device.type != "cpu":
            raise ValueError("attention output reference requires CPU tensors")
        return self.reference.forward_fp32(x, weight, bias=bias)


__all__ = [
    "QwenImageTPAttnOutBiasGemmReference",
    "shard_attn_out_parameters",
    "validate_attn_out",
]
