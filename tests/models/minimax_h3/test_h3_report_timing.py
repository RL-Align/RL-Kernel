# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``measure`` on a real CUDA device with real kernels.

``test_h3_report.py`` checks the bookkeeping with a stubbed clock; this one runs two real matmuls
of different size and checks that every callable is timed in every iteration, that the first
callable rotates, and that the stored samples reproduce the reported medians.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from rl_engine.validation.models.h3_report import measure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _case():
    small = torch.randn(256, 256, device="cuda")
    large = torch.randn(1024, 1024, device="cuda")
    return {
        "op": "matmul",
        "case": "T=1",
        "backend": "torch",
        "bytes": 1 << 20,
        "candidate": lambda: small @ small,
        "provider": lambda: large @ large,
    }


def test_every_callable_is_timed_in_every_iteration_with_a_rotating_first_callable():
    row = measure(_case(), warmup=2, iters=20)
    assert len(row["timing_samples_us"]["candidate"]) == 20
    assert len(row["timing_samples_us"]["provider"]) == 20
    assert row["timing_order"][0] == ["candidate", "provider"]
    assert row["timing_order"][1] == ["provider", "candidate"]
    assert all(sorted(order) == ["candidate", "provider"] for order in row["timing_order"])
    assert {order[0] for order in row["timing_order"]} == {"candidate", "provider"}


def test_the_reported_medians_come_from_the_stored_samples():
    row = measure(_case(), warmup=2, iters=20)
    for key in ("candidate", "provider"):
        assert row[f"{key}_us"] == statistics.median(row["timing_samples_us"][key])
        assert row[f"{key}_us"] > 0
        assert row[f"{key}_gbps"] == pytest.approx(row["bytes"] / (row[f"{key}_us"] * 1e-6) / 1e9)
        assert row[f"{key}_peak_mib"] >= 0
    # the larger matmul is slower than the smaller one in the same run
    assert row["provider_us"] > row["candidate_us"]


def test_backward_timings_of_every_perf_case_exclude_the_forward_pass(monkeypatch):
    """A case with a timed backward must prepare the graph outside the CUDA-event window."""

    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.validation.models import h3_report

    monkeypatch.setattr(h3_report, "NORM_SEQ_LENS", (257,))
    registry = KernelRegistry()
    checked = 0
    for op, build in h3_report.PERF_CASES.items():
        for case in build(registry):
            for side in ("candidate", "provider"):
                if f"{side}_backward" in case:
                    checked += 1
                    assert f"{side}_backward_setup" in case, f"{op} {case['case']} {side}"
    assert checked
