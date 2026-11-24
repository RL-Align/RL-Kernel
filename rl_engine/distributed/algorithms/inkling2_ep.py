# SPDX-License-Identifier: Apache-2.0
"""INK-35: ordered EP dispatch and fixed-topology gradient reduction."""

import torch
import torch.distributed as dist


def expert_dispatch_order(ids: torch.Tensor, ep_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable expert-major permutation and owning EP rank for each routed slot."""
    if ep_size not in (1, 2, 4, 8) or ids.ndim != 2 or ids.shape[1] != 8:
        raise ValueError("expected top-8 routes and EP1/2/4/8")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be integers")
    if bool(((ids < 0) | (ids >= 128)).any()):
        raise ValueError("expert ID outside 128-expert range")
    sorted_ids = ids.sort(-1).values
    if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
        raise ValueError("duplicate route expert")
    order = ids.flatten().argsort(stable=True)
    owners = ids.flatten()[order].long() // (128 // ep_size)
    return order, owners


def fold_rank_gradients(parts: torch.Tensor) -> torch.Tensor:
    """Left-fold FP32 rank contributions in ascending group-rank order."""
    if parts.ndim < 2 or not parts.shape[0] or parts.dtype != torch.float32:
        raise ValueError("expected nonempty FP32 rank contributions")
    total = torch.zeros_like(parts[0])
    for part in parts.unbind(0):
        total = total + part
    return total


@torch.no_grad()
def ordered_reduce_scatter(
    gradient: torch.Tensor, group, *, platform: str, average: bool = False
) -> torch.Tensor:
    """Debug FSDP2 gradient-buffer candidate, not an installed FSDP hook.

    NCCL/RCCL transport only; all-gather contributions and fold locally before
    slicing the rank's shard. Fixed topology only, full-buffer memory cost.
    All group members must call with the same metadata and collective order.
    """
    if platform not in ("cuda", "rocm") or not gradient.is_cuda:
        raise RuntimeError("requires CUDA or ROCm GPU gradient")
    if (torch.version.hip is not None) != (platform == "rocm"):
        raise RuntimeError("requested platform does not match PyTorch build")
    if not dist.is_initialized() or group is None or dist.get_backend(group) != "nccl":
        raise RuntimeError("requires explicit NCCL/RCCL process group")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError("expected a participating rank in group size 1/2/4/8")
    if gradient.dtype != torch.float32 or not gradient.numel() or gradient.numel() % world:
        raise ValueError("FP32 gradient must divide evenly across ranks")
    local = gradient.contiguous().flatten()
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local, group=group)
    total = fold_rank_gradients(torch.stack(gathered))
    if average:
        total = total / world
    return total.chunk(world)[rank].contiguous()
