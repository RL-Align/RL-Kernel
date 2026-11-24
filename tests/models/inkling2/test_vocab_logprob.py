# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.distributed as dist

from rl_engine.models.inkling2.vocab_logprob import PADDED, VALID, vocab_bounds, vocab_logprob


def test_global_tiles():
    for tp in (1, 2, 4, 8):
        bounds = vocab_bounds(tp)
        assert bounds[-1][1] == PADDED
        assert all((end - start) % (PADDED // 64) == 0 for start, end in bounds)
    assert VALID < PADDED


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_gpu_vocab(platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} required")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    try:
        rank = dist.get_rank() if world > 1 else 0
        full = torch.arange(PADDED, device="cuda", dtype=torch.float32)[None] / 1000
        full[:, VALID:] = float("nan")
        target = torch.tensor([VALID - 1], device="cuda")
        expected, expected_lse = vocab_logprob(full, target, platform=platform)
        start, end = vocab_bounds(world)[rank]
        local = full[:, start:end].clone().requires_grad_()
        actual, lse = vocab_logprob(local, target, platform=platform,
                                   tp_group=dist.group.WORLD if world > 1 else None)
        assert torch.equal(actual, expected) and torch.equal(lse, expected_lse)
        actual.sum().backward()
        assert torch.isfinite(local.grad).all()
        assert torch.count_nonzero(local.grad[:, max(0, VALID - start):]) == 0
    finally:
        if world > 1:
            dist.destroy_process_group()
