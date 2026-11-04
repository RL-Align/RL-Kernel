# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image QKV geometry and independent PyTorch reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import HIDDEN

TP_SIZES = (1, 2, 4, 8)


def validate_qkv(x, weight, bias, tp_world=1):
    if type(tp_world) is not int or tp_world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != HIDDEN:
        raise ValueError("x must be nonempty [M,3072]")
    local_width = 3 * HIDDEN // tp_world
    if tuple(weight.shape) != (local_width, HIDDEN):
        raise ValueError(f"weight must be [{local_width},3072] for TP={tp_world}")
    if bias is not None and tuple(bias.shape) != (local_width,):
        raise ValueError(f"bias must be [{local_width}] for TP={tp_world}")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("only bfloat16 and float32 are supported")
    tensors = (weight,) if bias is None else (weight, bias)
    if any(t.dtype != x.dtype for t in tensors):
        raise TypeError("x, weight and bias must share a dtype")
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs must be on the same device")


def shard_qkv_parameters(weight, bias, *, tp_world, tp_rank):
    """Select the same contiguous head range from each of Q, K and V."""
    validate_qkv(weight.new_empty((1, HIDDEN)), weight, bias)
    if type(tp_world) is not int or tp_world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_world:
        raise ValueError("TP rank must be in [0, TP size)")
    width = HIDDEN // tp_world
    start = tp_rank * width
    weights = [part[start : start + width] for part in weight.chunk(3, 0)]
    local_bias = None
    if bias is not None:
        local_bias = torch.cat([part[start : start + width] for part in bias.chunk(3, 0)])
    return torch.cat(weights, dim=0), local_bias


class QwenImageTPQKVBiasGemmReference:
    """Full single-rank QKV reference; it does not emulate strict CUDA arithmetic."""

    op_class = "reduction"
    kernel_id = "pytorch.qwenimage.tp_qkv_bias_gemm.reference"

    def forward(self, x, weight, bias=None):
        validate_qkv(x, weight, bias)
        return F.linear(x, weight, bias)

    __call__ = forward

    def forward_fp32(self, x, weight, bias=None):
        validate_qkv(x, weight, bias)
        return F.linear(x.float(), weight.float(), None if bias is None else bias.float())
