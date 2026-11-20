# SPDX-License-Identifier: Apache-2.0
"""Version-preserving native PP state export with exact DP replica checks.

Every actor rank must consume the generator in the same order. Transfer one
original-dtype tensor at a time; do not gather pickled model weights on GPUs.
"""

import hashlib

import torch
import torch.distributed as dist


def _descriptor(value):
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        sha256=hashlib.sha256(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    )


def export_pipeline_state(local_state, *, pp_group, dp_group, weight_version=0):
    """Verify live DP copies, then broadcast each disjoint PP-owned state tensor."""
    if not local_state or not all(isinstance(v, torch.Tensor) for v in local_state.values()):
        raise ValueError("pipeline export requires nonempty actual stage tensors")
    device = next(iter(local_state.values())).device
    local = {name: _descriptor(value) for name, value in local_state.items()}
    envelope = dict(weight_version=weight_version, tensors=local)
    replicas = [None] * dist.get_world_size(dp_group)
    dist.all_gather_object(replicas, envelope, group=dp_group)
    disagree = any(replica != envelope for replica in replicas)
    failed = torch.tensor(int(disagree), dtype=torch.int32, device=device)
    # Propagate failures across stages before entering any PP broadcast.
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError("actual DP replicas differ before pipeline state synchronization")
    stages = [None] * dist.get_world_size(pp_group)
    dist.all_gather_object(stages, envelope, group=pp_group)
    if len({stage["weight_version"] for stage in stages}) != 1:
        raise RuntimeError("pipeline ranks disagree on the actual weight version")
    owners = {}
    for stage_rank, manifest in enumerate(stages):
        for name, description in manifest["tensors"].items():
            if name in owners:
                raise ValueError(f"pipeline state has duplicate ownership: {name}")
            owners[name] = (stage_rank, description)
    ranks = dist.get_process_group_ranks(pp_group)
    for name, (owner, description) in sorted(owners.items()):
        if dist.get_rank(pp_group) == owner:
            value = local_state[name].detach().contiguous().clone()
        else:
            dtype = getattr(torch, description["dtype"].removeprefix("torch."))
            value = torch.empty(description["shape"], dtype=dtype, device=device)
        dist.broadcast(value, src=ranks[owner], group=pp_group)
        if _descriptor(value) != description:
            raise RuntimeError(f"pipeline state transfer changed original bytes: {name}")
        yield name, value
