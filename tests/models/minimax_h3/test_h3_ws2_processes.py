# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU process lifecycle regressions for the multi-GPU evidence runner."""

import os
import time

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.validation.models import h3_ws2


def _worker(rank, world, init, queue, path, target, kwargs):
    mode = kwargs["mode"]
    if mode == "success":
        torch.save({"value": torch.tensor(rank)}, path.with_name(f"rank{rank}.pt"))
        queue.put({"rank": rank})
    elif rank == 0 and mode == "error":
        queue.put({"rank": rank, "error": "original rank traceback"})
    elif rank == 0 and mode == "crash":
        os._exit(7)
    else:
        time.sleep(120)


@pytest.mark.parametrize("mode", ["success", "error", "crash", "timeout"])
def test_run_world_reaps_workers(monkeypatch, mode):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(h3_ws2, "_bootstrap", _worker)
    before = {p.pid for p in mp.active_children()}
    start = time.monotonic()
    if mode == "success":
        ranks = h3_ws2.run_world(2, None, {}, timeout=30, mode=mode)
        assert [r["rank"] for r in ranks] == [0, 1]
        assert [r["value"].item() for r in ranks] == [0, 1]
    elif mode == "timeout":
        with pytest.raises(TimeoutError, match="timed out"):
            h3_ws2.run_world(2, None, {}, timeout=0.2, mode=mode)
    else:
        message = "original rank traceback" if mode == "error" else "rank 0 exited with code 7"
        with pytest.raises(RuntimeError, match=message):
            h3_ws2.run_world(2, None, {}, timeout=30, mode=mode)
        assert time.monotonic() - start < 25
    assert {p.pid for p in mp.active_children()} == before
