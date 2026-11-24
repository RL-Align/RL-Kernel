# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Output lifetime and allocated-memory accounting, independent of kernel numerics."""

import weakref

import pytest
import torch

from benchmarks.operators.gemm.benchmark_projection_gemm import measure


def test_each_output_is_released_before_the_next_call(monkeypatch):
    refs = []
    events = []
    state = {"allocated": 100, "peak": 100, "calls": 0, "reset_at": []}

    class Output:
        dtype = torch.float32

        def __init__(self):
            state["allocated"] += 32
            state["peak"] = max(state["peak"], state["allocated"])

        def __del__(self):
            state["allocated"] -= 32

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
            self.is_end = len(events) % 2 == 1
            events.append(self)

        def record(self):
            assert any(ref() is not None for ref in refs) == self.is_end

        def elapsed_time(self, end):
            assert end.is_end
            return 1.0

    def call():
        assert all(ref() is None for ref in refs), "previous output retained at next API call"
        assert state["allocated"] == 100
        state["calls"] += 1
        state["allocated"] += 200  # A temporary that dies inside the public API.
        output = Output()
        refs.append(weakref.ref(output))
        state["allocated"] -= 200
        return output

    def reset():
        state["reset_at"].append(state["calls"])
        state["peak"] = state["allocated"]

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: state["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: state["peak"])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    for _ in range(2):  # Includes the transition between two measured APIs.
        events.clear()
        result = measure(call, warmup=3, repeats=5)
        assert result["dtype"] == "torch.float32"
        assert result["baseline_allocated_bytes"] == 100
        assert result["peak_allocated_bytes"] == 332
        assert result["peak_increment_bytes"] == 232
        assert result["post_release_allocated_bytes"] == 100
        assert all(ref() is None for ref in refs)
    assert state["calls"] == 18  # 2 * (3 warmup + 5 timed + 1 memory call).
    assert state["reset_at"] == [8, 17]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_single_call_cuda_memory_has_exactly_one_output():
    refs = []

    def call():
        assert all(ref() is None for ref in refs)
        output = torch.empty(1024 * 1024, device="cuda", dtype=torch.float32)
        refs.append(weakref.ref(output))
        return output

    result = measure(call, warmup=2, repeats=4)
    assert result["peak_increment_bytes"] == 4 * 1024 * 1024
    assert result["peak_allocated_bytes"] == result["baseline_allocated_bytes"] + 4 * 1024 * 1024
    assert result["post_release_allocated_bytes"] == result["baseline_allocated_bytes"]
    assert all(ref() is None for ref in refs)
