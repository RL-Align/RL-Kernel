# SPDX-License-Identifier: Apache-2.0
"""Validate diagnostic statistics without weakening original FP32 byte checks."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from examples.operators.dsv4_4l import cycle, measurement


def test_difference_statistics_do_not_replace_original_fp32_bits(tmp_path, monkeypatch):
    monkeypatch.setattr(cycle, "ROOT", tmp_path)
    case = dict(id="A", prompt_token_ids=[3], response_token_ids=[4, 5])
    x = torch.tensor([-0.0, -2.0], dtype=torch.float32)
    y = torch.tensor([0.0, -2.5], dtype=torch.float32)
    row = cycle.compare_values("test", [case], [x], [y], 0)[0]
    assert row["expected_tokens"] == row["compared_tokens"] == 2
    assert row["mismatch_count"] == 2 and row["mismatch_rate"] == 1
    assert row["max_abs_diff"] == 0.5 and row["mean_abs_diff"] == 0.25
    assert row["first_mismatch"]["rollout_bits"] == "80000000"
    assert row["first_mismatch"]["actor_bits"] == "00000000"


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([1.0]).double(),
    ],
)
def test_incomplete_nonfinite_or_cast_scores_still_fail(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(cycle, "ROOT", tmp_path)
    case = dict(id="A", prompt_token_ids=[3], response_token_ids=[4])
    with pytest.raises(AssertionError):
        cycle.compare_values("bad", [case], [bad], [torch.tensor([1.0])], 0)


def test_server_timer_awaits_completion_and_does_not_change_outputs(monkeypatch):
    monkeypatch.setenv("DSV4_MEASURE", "1")
    events = []
    monkeypatch.setattr(measurement, "emit", lambda phase, **kw: events.append((phase, kw)))
    response = SimpleNamespace(model_dump=lambda: dict(token_ids=[7, 8]))

    async def generate(*args, **kwargs):
        await asyncio.sleep(0)
        return response

    server = SimpleNamespace(generate=generate, launch_server=generate, collective_rpc=generate)
    measurement.instrument_server(server)
    got = asyncio.run(server.generate([1, 2, 3], {}, request_id="A-v2"))
    assert got is response
    phase, fields = events[0]
    assert phase == "rollout_generate" and fields["success"] is True
    assert fields["prompt_tokens"] == 3 and fields["seconds"] >= 0
    assert events[1][0] == "generation_workload" and events[1][1]["response_tokens"] == 2


def test_timer_preserves_failure_and_does_not_report_success(monkeypatch):
    monkeypatch.setenv("DSV4_MEASURE", "1")
    events = []
    monkeypatch.setattr(measurement, "emit", lambda phase, **kw: events.append(kw))
    with pytest.raises(RuntimeError, match="worker failed"):
        with measurement.measure("test"):
            raise RuntimeError("worker failed")
    assert len(events) == 1 and not events[0]["success"]
    json.dumps(events, allow_nan=False)


def test_report_rejects_missing_version_and_nonfinite_or_incomplete_rows():
    from examples.operators.dsv4_4l.aggregate_comparison import KINDS, summarize_scores

    rows = [
        dict(
            comparison_kind=k,
            weight_version=v,
            expected_tokens=2,
            compared_tokens=2,
            mismatch_count=0,
            mismatch_rate=0,
            dtype="torch.float32",
            all_finite=True,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
        )
        for v in range(4)
        for k in KINDS
    ]
    assert len(summarize_scores(rows)) == 12
    with pytest.raises(ValueError, match="missing"):
        summarize_scores(rows[:-1])
    rows[0]["mean_abs_diff"] = float("nan")
    with pytest.raises(ValueError, match="diagnostic"):
        summarize_scores(rows)
    rows[0]["mean_abs_diff"] = 0.0
    rows[0]["compared_tokens"] = 0
    with pytest.raises(ValueError, match="incomplete"):
        summarize_scores(rows)


def test_report_keeps_finite_native_mismatch_statistics():
    from examples.operators.dsv4_4l.aggregate_comparison import KINDS, summarize_scores

    first = dict(token_id=4, rollout_bits="80000000", actor_bits="00000000")
    rows = [
        dict(
            comparison_kind=k,
            weight_version=v,
            expected_tokens=2,
            compared_tokens=2,
            mismatch_count=1,
            mismatch_rate=0.5,
            dtype="torch.float32",
            all_finite=True,
            max_abs_diff=0.125,
            mean_abs_diff=0.0625,
            first_mismatch=first,
        )
        for v in range(4)
        for k in KINDS
    ]
    result = summarize_scores(rows)
    assert all(x["mismatch_count"] == 1 and x["mismatch_rate"] == 0.5 for x in result)
    assert all(x["mean_abs_diff"] == 0.0625 and not x["batch_invariance"] for x in result)
