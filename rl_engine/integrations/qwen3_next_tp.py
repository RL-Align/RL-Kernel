# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Differentiable TP4 boundaries and parameter metadata shared by Qwen3-Next blocks.

The forward copy and reduction follow Megatron's column/row-parallel autograd
pair. Reductions go through ``collective_for_group`` so a strict deployment
can pin the collective algorithm; nothing here selects one implicitly.
"""

import torch
import torch.distributed as dist


def _reduce(value, group):
    from rl_engine.distributed.collectives import collective_for_group

    collective = collective_for_group(group, min_size_bytes=value.numel() * value.element_size())
    return collective.all_reduce(value.contiguous())


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient):
        return _reduce(gradient, ctx.group), None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        return _reduce(value, group)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def _rank(rank):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(4):
        raise ValueError("TP4 rank must be an integer in [0, 4)")


def _weight(value, shape, name):
    if value.dtype != torch.bfloat16 or value.shape != shape:
        raise ValueError(f"{name} must be BF16 {shape}")


def _tp_group(group):
    if not dist.is_initialized() or dist.get_world_size(group) != 4:
        raise ValueError("Qwen3-Next blocks require an initialized four-rank TP group")
    return dist.group.WORLD if group is None else group


def _parallel_parameter(parameter, dimension, *, stride=1, duplicate=False):
    # MCore uses these attributes when computing the global optimizer gradient
    # norm. A replicated copy updates normally but must not count twice.
    parameter.tensor_model_parallel = True
    parameter.partition_dim = dimension
    parameter.partition_stride = stride
    parameter.shared = duplicate
