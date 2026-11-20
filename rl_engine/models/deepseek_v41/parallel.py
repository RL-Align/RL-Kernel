# SPDX-License-Identifier: Apache-2.0
"""Attention tensor parallelism; replicated dense/FFN and source-owned caches.

Q/output-group parameters are column sharded, wo_b is row sharded. BF16
projection outputs are reduced as BF16, matching the released row-parallel
boundary. This differs numerically from one unsplit GEMM. Quantized parameters
remain frozen; autograd collectives sum replicated-input activation gradients.
"""

from dataclasses import dataclass
import torch
import torch.distributed as dist


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous().clone()
        dist.all_reduce(dy, group=ctx.group)
        return dy, None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, dy):
        return dy, None


@dataclass(frozen=True)
class AttentionTP:
    size: int = 1
    rank: int = 0
    group: object = None

    def __post_init__(self):
        if self.size not in (1, 2, 4) or not 0 <= self.rank < self.size:
            raise ValueError("attention TP supports sizes 1,2,4")
        if self.size > 1 and (
            not dist.is_initialized()
            or dist.get_world_size(self.group) != self.size
            or dist.get_rank(self.group) != self.rank
        ):
            raise ValueError("TP process group does not match the requested topology")

    def copy(self, x):
        return _Copy.apply(x, self.group) if self.size > 1 else x

    def reduce(self, x):
        return _Reduce.apply(x, self.group) if self.size > 1 else x

    def shard(self, x, dim=0):
        if x.shape[dim] % self.size:
            raise ValueError("TP dimension is not evenly divisible")
        return x.chunk(self.size, dim=dim)[self.rank].contiguous()
