# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.validation.moe.ep_integrity_runner import _exchange


def _uneven_exchange_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        # Independent fixed oracle: source, local row, destination. Rank 2
        # starts empty but receives two rows; rank 1 receives none.
        planned = torch.tensor([[0, 0, 2], [0, 1, 0], [0, 2, 2], [1, 0, 0]])
        payload = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])
        local = planned[:, 0] == rank
        routes, values = _exchange(planned[local], payload[local], planned[local, 2], 3)
        owned = planned[:, 2] == rank
        assert torch.equal(routes, planned[owned])
        assert torch.equal(values, payload[owned])
        # Reverse arrival order and return using the actual local receive rows.
        routes, values = routes.flip(0), values.flip(0)
        returned, outputs = _exchange(routes, values, routes[:, 0], 3)
        order = torch.argsort(returned[:, 1])
        assert torch.equal(returned[order], planned[local])
        assert torch.equal(outputs[order], payload[local])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_uneven_local_rows_and_empty_rank_round_trip(tmp_path):
    context = mp.spawn(
        _uneven_exchange_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=3,
        join=False,
    )
    try:
        deadline = time.monotonic() + 60
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("EP exchange workers exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
