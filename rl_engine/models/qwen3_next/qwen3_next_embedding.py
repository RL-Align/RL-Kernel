# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared TP4 vocabulary embedding with a fixed-order, non-atomic VJP."""

import torch
import torch.distributed as dist

from rl_engine.models.qwen3_next.qwen3_next_tp import _reduce


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight, rank, group):
        flat = tokens.reshape(-1)
        local = flat - rank * 37984
        valid = (local >= 0) & (local < 37984)
        rows = local.clamp(0, 37983)
        value = weight.index_select(0, rows)
        value = torch.where(valid[:, None], value, torch.zeros_like(value))
        ctx.save_for_backward(local, valid)
        ctx.weight_shape, ctx.weight_dtype = weight.shape, weight.dtype
        return _reduce(value, group).reshape(*tokens.shape, 2048)

    @staticmethod
    def backward(ctx, dy):
        local, valid = ctx.saved_tensors
        local = local[valid]
        values = dy.reshape(-1, 2048)[valid]
        gradient = torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=dy.device)
        order = torch.argsort(local, stable=True)
        local, values = local[order], values[order]
        unique, counts = torch.unique_consecutive(local, return_counts=True)
        start = 0
        # Every row is written once. Pairwise FP32 additions use a fixed tree
        # over source token order; there is no floating-point scatter/atomic.
        for row, count in zip(unique.cpu().tolist(), counts.cpu().tolist(), strict=True):
            part = values[start : start + count].float()
            while part.shape[0] > 1:
                pairs = part.shape[0] // 2
                merged = part[: 2 * pairs : 2] + part[1 : 2 * pairs : 2]
                part = torch.cat((merged, part[-1:]), dim=0) if part.shape[0] % 2 else merged
            gradient[row].copy_(part[0].to(ctx.weight_dtype))
            start += count
        return None, gradient, None, None


def shared_embedding(tokens, weight, group):
    if not dist.is_initialized() or group is None or dist.get_world_size(group) != 4:
        raise ValueError("Shared Qwen embedding requires an explicit TP4 group")
    if tokens.dtype != torch.int64 or tokens.device != weight.device:
        raise ValueError("Embedding token IDs must be int64 on the weight device")
    if weight.shape != (37984, 2048) or weight.dtype != torch.bfloat16:
        raise ValueError("Each rank must own a BF16 [37984, 2048] embedding shard")
    if weight.device.type != "cuda" or bool(((tokens < 0) | (tokens >= 151936)).any()):
        raise ValueError("Embedding requires valid official vocabulary IDs on CUDA")
    return _Embedding.apply(tokens, weight, dist.get_rank(group), group)
