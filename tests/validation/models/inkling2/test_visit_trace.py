# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.validation.models.inkling2.visit_trace import VisitTrace, tensor_digest, visits


def test_schedule_repeated_weights_distinct_visits():
    v = visits()
    assert len(v) == 48
    assert [(v[j].physical, v[j].pass_index) for j in (6, 18, 30)] == [(6, 0), (6, 1), (6, 2)]
    assert v[47].physical == 23 and v[47].pass_index is None


def test_raw_bytes_and_snapshot():
    assert tensor_digest(torch.tensor([0.0])) != tensor_digest(torch.tensor([-0.0]))
    x = torch.ones(2, 3).bfloat16()
    trace = VisitTrace({"profile": "fixture"})
    trace.record(6, "partial", x, backend="fixture", kernel="test")
    old = trace.report()["records"][0]["sha256"]
    x.zero_()
    assert trace.report()["records"][0]["sha256"] == old
    assert tensor_digest(x)["sha256"] != old
    with pytest.raises(ValueError, match="duplicate"):
        trace.record(6, "partial", x, backend="fixture", kernel="test")


def test_order_and_backward_identity():
    trace = VisitTrace({})
    for j in (30, 6, 18):
        trace.record(j, "partial", torch.ones(1), backend="fixture", kernel="test")
    trace.record(
        6, "partial", torch.zeros(1), direction="backward", backend="fixture", kernel="test"
    )
    assert [r["logical"] for r in trace.report()["records"]] == [6, 6, 18, 30]
    with pytest.raises(ValueError, match="outside"):
        trace.record(48, "partial", torch.ones(1), backend="fixture", kernel="test")


def test_selected_logprob_bits_and_record_order():
    trace = VisitTrace({})
    for role in ("z", "a"):
        trace.record(0, role, torch.ones(1), backend="fixture", kernel="test")
    trace.record_logprob(torch.tensor([0.0, -0.0]))
    report = trace.report()
    assert [r["role"] for r in report["records"]] == ["z", "a"]
    assert report["selected_logprob_bits"] == [0, -2147483648]
    with pytest.raises(ValueError, match="already recorded"):
        trace.record_logprob(torch.ones(1))
