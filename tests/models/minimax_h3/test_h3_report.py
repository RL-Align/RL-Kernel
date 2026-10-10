# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Comparative timing balances backend order and preserves its raw evidence."""

from __future__ import annotations

import pytest

from rl_engine.validation.models import h3_report


def test_measure_interleaves_backends_and_records_samples(monkeypatch):
    """Balance candidate/provider timing order and preserve samples and derived bandwidth."""

    elapsed_us = 0.0
    calls = []

    class Event:
        def __init__(self, *, enable_timing):
            """Require timing-enabled events in the deterministic timing stub."""

            assert enable_timing

        def record(self):
            """Capture the synthetic elapsed clock at this event."""

            self.timestamp = elapsed_us

        def synchronize(self):
            """Leave synchronization inert because the timing stub has no pending GPU work."""

            pass

        def elapsed_time(self, end):
            """Return elapsed synthetic event time in CUDA milliseconds."""

            return (end.timestamp - self.timestamp) / 1e3

    def candidate():
        """Record a candidate call and advance the synthetic clock by two microseconds."""

        nonlocal elapsed_us
        calls.append("candidate")
        elapsed_us += 2.0

    def provider():
        """Record a provider call and advance the synthetic clock by four microseconds."""

        nonlocal elapsed_us
        calls.append("provider")
        elapsed_us += 4.0

    monkeypatch.setattr(h3_report.torch.cuda, "Event", Event)
    monkeypatch.setattr(h3_report.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(h3_report, "peak_mib", lambda fn: 0.0)
    report = h3_report.measure(
        {"bytes": 1000, "candidate": candidate, "provider": provider}, warmup=1, iters=4
    )
    assert calls == [
        "candidate",
        "provider",  # warmup
        "candidate",
        "provider",
        "provider",
        "candidate",
        "candidate",
        "provider",
        "provider",
        "candidate",
    ]
    assert report["timing_order"] == [
        ["candidate", "provider"],
        ["provider", "candidate"],
        ["candidate", "provider"],
        ["provider", "candidate"],
    ]
    assert report["timing_samples_us"] == {"candidate": [2.0] * 4, "provider": [4.0] * 4}
    assert report["candidate_us"] == pytest.approx(2.0)
    assert report["provider_us"] == pytest.approx(4.0)
    assert report["candidate_gbps"] == pytest.approx(0.5)
    assert report["provider_gbps"] == pytest.approx(0.25)
