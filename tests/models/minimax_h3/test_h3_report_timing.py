# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Timing harness of the H3 evidence: competing callables are timed in a shuffled order.

Run on a real CUDA device with real kernels (torch matmuls of two sizes); the check is about
the executed order and the medians, not about absolute times.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.validation.models.h3_report import measure, time_interleaved

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _callables():
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    return {"small": lambda: a @ a, "large": lambda: b @ b}


def test_every_callable_runs_once_per_iteration_in_both_orders():
    fns = _callables()
    medians, executed = time_interleaved(fns, warmup=2, iters=40, seed=0)
    assert set(medians) == {"small", "large"}
    assert all(value > 0 for value in medians.values())
    assert len(executed) == 40
    # digits index list(fns): each iteration runs both callables exactly once
    assert {"".join(sorted(order)) for order in executed} == {"01"}
    # the order is shuffled, so neither callable always goes first
    assert {"01", "10"} <= set(executed)


def test_the_executed_order_is_reproducible_from_the_seed():
    fns = _callables()
    _, first = time_interleaved(fns, warmup=1, iters=25, seed=3)
    _, second = time_interleaved(fns, warmup=1, iters=25, seed=3)
    _, other = time_interleaved(fns, warmup=1, iters=25, seed=4)
    assert first == second
    assert first != other


def test_measure_records_the_order_next_to_the_timings():
    case = {
        "op": "matmul",
        "case": "T=1",
        "backend": "torch",
        "bytes": 1 << 20,
        **{key: fn for key, fn in zip(("candidate", "provider"), _callables().values())},
    }
    row = measure(case, warmup=2, iters=20)
    assert row["candidate_us"] > 0 and row["provider_us"] > 0
    assert row["timing"]["callables"] == ["candidate", "provider"]
    assert len(row["timing"]["executed_order"]) == 20
    assert {"01", "10"} <= set(row["timing"]["executed_order"])
