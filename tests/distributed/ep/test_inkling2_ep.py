# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.distributed as dist

from rl_engine.distributed.algorithms.inkling2_ep import (
    expert_dispatch_order,
    fold_rank_gradients,
    ordered_reduce_scatter,
)


def test_order_and_rank_fold():
    ids = torch.tensor([[64, 1, 0, 33, 127, 12, 4, 2]])
    for ep in (1, 2, 4, 8):
        order, owner = expert_dispatch_order(ids, ep)
        assert torch.equal(ids.flatten()[order].long() // (128 // ep), owner)
    assert torch.equal(fold_rank_gradients(torch.tensor([[1e8], [-1e8], [1.0]])), torch.ones(1))


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_distributed_gpu(platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} required")
    if int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8):
        pytest.skip("run under torchrun with 2/4/8 GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        x = torch.full((world * 2,), float(rank + 1), device="cuda")
        result = ordered_reduce_scatter(x, dist.group.WORLD, platform=platform)
        assert torch.equal(result, torch.full_like(result, world * (world + 1) / 2))
    finally:
        dist.destroy_process_group()
