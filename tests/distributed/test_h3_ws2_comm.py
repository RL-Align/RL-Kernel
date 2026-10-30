# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``gather_rows`` is a pure copy: padding past the fast path must not change a single byte."""

from __future__ import annotations

import pytest
import torch

from rl_engine.validation.distributed.h3_ws2 import run_world

needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")


def _gather_target(collective, dummy, *, rows, width, dtype):
    from rl_engine.backends.cuda.model_specific.minimax_h3.ws2_comm import gather_rows

    del dummy
    gen = torch.Generator().manual_seed(11 + collective.rank)
    shard = torch.randn(rows, width, generator=gen).to(dtype).cuda()
    ours = gather_rows(collective, shard)
    plain = collective.all_gather(shard.contiguous())
    return {"equal": torch.tensor(int(torch.equal(ours, plain))), "shape": torch.tensor(ours.shape)}


@needs_two_gpus
@pytest.mark.parametrize(
    ("rows", "width"),
    [
        (1, 4),  # a few bytes: plain gather
        (3, 16),  # below the pad threshold
        (4, 512),  # 2 ranks: 8 KiB of bf16 output, exactly at the threshold
        (5, 512),  # just above it: padded
        (7, 33),  # odd width, padded or not
        (600, 128),  # well above the single-block limit: padded shard equals natural size
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_rows_equals_the_plain_all_gather(rows, width, dtype):
    ranks = run_world(
        2, _gather_target, {"dummy": torch.zeros(1)}, rows=rows, width=width, dtype=dtype
    )
    for r in ranks:
        assert int(r["equal"]) == 1
        assert r["shape"].tolist() == [2 * rows, width]
