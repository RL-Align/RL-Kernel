# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Cross-node deterministic collective: node-local IPC fixed tree, then one exact gather.

``DeterministicCollective`` (``cuda_ipc_fixed_tree``) reads peers' memory through CUDA IPC,
so it is single node. This wrapper (``hierarchical_ipc_nccl_tree``) reaches the same bits on
several nodes without relying on any NCCL reduction order:

1. every group of ``local_size`` consecutive ranks reduces with the node-local fixed tree;
2. the group partials move with NCCL ``all_gather`` (bit-exact data movement, no arithmetic);
3. each rank adds the group partials in one fixed, balanced order.

With ``world_size = nodes * local_size`` and ``local_size``, ``nodes`` powers of two, the
tree ``((0+1)+(2+3)) + ((4+5)+(6+7))`` of an 8-way node-local reduction is exactly "the 4-way
tree of each half, then the two halves added". The input dtype is the accumulation dtype, as in
``DeterministicCollective`` (FP32 inputs keep every partial in FP32, which is what the H3
parallel operators use), so the result is bitwise the single-node tree of the same world size.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from rl_engine.distributed.algorithms.collectives import DeterministicCollective

_GROUP_COUNTS = (1, 2, 4, 8)


def tree_add(parts: list[torch.Tensor]) -> torch.Tensor:
    """Balanced pairwise sum ``((p0+p1)+(p2+p3))+...`` of a power-of-two number of tensors."""

    if len(parts) & (len(parts) - 1) or not parts:
        raise ValueError(f"tree reduction needs a power-of-two part count, got {len(parts)}")
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


class HierarchicalCollective:
    """TP-invariant all-reduce / reduce-scatter over ``world_size`` ranks on several nodes."""

    backend_id = "hierarchical_ipc_nccl_tree"

    def __init__(
        self,
        local_size: int,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = 64 << 20,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before collectives")
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)
        if self.world_size % local_size or local_size not in _GROUP_COUNTS:
            raise ValueError(
                f"local_size {local_size} must be 1/2/4/8 and divide world {self.world_size}"
            )
        self.local_size = local_size
        self.num_groups = self.world_size // local_size
        if self.num_groups not in _GROUP_COUNTS:
            raise ValueError(f"{self.num_groups} node groups; need 1, 2, 4 or 8")
        self.group_index, self.local_rank = divmod(self.rank, local_size)
        # Every rank creates every group in the same order (a torch.distributed requirement).
        ranks = dist.get_process_group_ranks(self.group)  # global rank of each group member
        local_groups = [
            dist.new_group(ranks[g * local_size : (g + 1) * local_size])
            for g in range(self.num_groups)
        ]
        cross_groups = [
            dist.new_group([ranks[g * local_size + r] for g in range(self.num_groups)])
            for r in range(local_size)
        ]
        self._cross = cross_groups[self.local_rank]
        self._local = DeterministicCollective(
            local_groups[self.group_index], device, max_size_bytes=max_size_bytes
        )

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        """The fixed-tree sum of every rank's ``input`` on every rank (new tensor)."""

        if input.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError(f"unsupported dtype {input.dtype}")
        wide = (
            input.contiguous()
        )  # the input dtype is the accumulation dtype, as in the node-local tree
        partial = self._local.all_reduce(wide) if self.local_size > 1 else wide.clone()
        if self.num_groups > 1:
            gathered = [torch.empty_like(partial) for _ in range(self.num_groups)]
            dist.all_gather(gathered, partial, group=self._cross)
            partial = tree_add(gathered)
        return partial

    def reduce_scatter(self, input: torch.Tensor) -> torch.Tensor:
        """``all_reduce`` then this rank's equal share of dimension 0."""

        if input.shape[0] % self.world_size:
            raise ValueError("dimension 0 must divide by the world size")
        rows = input.shape[0] // self.world_size
        return self.all_reduce(input)[self.rank * rows : (self.rank + 1) * rows].contiguous()

    def all_gather(self, input: torch.Tensor) -> torch.Tensor:
        """Rank-ordered concatenation along dimension 0 (exact data movement)."""

        out = [torch.empty_like(input) for _ in range(self.world_size)]
        dist.all_gather(out, input.contiguous(), group=self.group)
        return torch.cat(out)

    def close(self) -> None:
        self._local.close()
