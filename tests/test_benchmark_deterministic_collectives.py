# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for collective benchmark timing boundaries."""

import pytest

from benchmarks import benchmark_deterministic_collectives as benchmark


@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("with_prepare", [False, True])
def test_median_us_excludes_preparation(monkeypatch, warmup, with_prepare):
    """Reset every call before its timing event without timing the reset."""
    clock_us = 0
    calls = []
    value = 1

    class Event:
        """Model CUDA events on a stream with a deterministic virtual clock."""

        def __init__(self, *, enable_timing):
            """Accept the same timing flag as a CUDA event."""
            assert enable_timing
            self.timestamp = None

        def record(self):
            """Record the stream's current virtual timestamp."""
            self.timestamp = clock_us
            calls.append("event")

        def synchronize(self):
            """Represent completion of the recorded event."""
            calls.append("end_sync")

        def elapsed_time(self, end):
            """Return elapsed milliseconds, matching the CUDA event API."""
            return (end.timestamp - self.timestamp) / 1e3

    def prepare():
        """Spend time resetting the mutable collective input."""
        nonlocal clock_us, value
        clock_us += 1000
        value = 1
        calls.append("prepare")

    def collective():
        """Model an in-place reduction with seven microseconds of work."""
        nonlocal clock_us, value
        if with_prepare:
            assert value == 1
        value *= 2
        clock_us += 7
        calls.append("collective")

    monkeypatch.setattr(benchmark.torch.cuda, "Event", Event)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: calls.append("sync"))
    measured = benchmark._median_us(
        collective, warmup, 3, prepare=prepare if with_prepare else None
    )

    assert measured == pytest.approx(7)
    preparation = ["prepare"] if with_prepare else []
    assert (
        calls
        == (preparation + ["collective"]) * warmup
        + ["sync"]
        + (preparation + ["event", "collective", "event", "end_sync"]) * 3
    )
