# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch

from rl_engine.validation.models.inkling2.visit_trace import VisitTrace
from rl_engine.validation.models.inkling2.ws1_chain import compare_chain


def report_fixture():
    # Fabricated metadata exercises the comparator only; not real-model evidence.
    meta = dict(
        revision="fixture",
        profile="bf16-dequant",
        capture_mode="real-weight",
        physical_layers=24,
        logical_visits=48,
        hardware="fixture",
        token_count=2,
        engine="fixture",
        runtime_revision="fixture",
        trace_scope="block-boundary",
    )
    for name in ("config", "quant_metadata", "weights", "input"):
        meta[f"{name}_sha256"] = "a" * 64
    trace = VisitTrace(meta)
    for j in range(48):
        trace.record(j, "partial", torch.ones(2, 3), backend="fixture", kernel="test")
    trace.record_logprob(torch.tensor([-1.0, -2.0]))
    return trace.report()


def test_exact_match_and_first_drift():
    a = report_fixture()
    b = copy.deepcopy(a)
    result = compare_chain(a, b)
    assert result["trace_equal"] and not result["ws1_certified"]
    assert result["compared_boundaries"] == 48
    b["records"][18]["sha256"] = "b" * 64
    b["records"][30]["sha256"] = "c" * 64
    b["selected_logprob_bits"][1] += 1
    result = compare_chain(a, b)
    assert not result["trace_equal"] and result["boundary_mismatches"] == 2
    assert result["first_boundary_drift"]["logical"] == 18
    assert result["selected_token_mismatches"] == 1
    assert result["first_selected_token_drift"] == 1


def test_fail_closed_on_incomplete_or_incomparable_trace():
    a = report_fixture()
    b = copy.deepcopy(a)
    b["records"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        compare_chain(a, b)
    b = copy.deepcopy(a)
    b["metadata"]["weights_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="incomparable"):
        compare_chain(a, b)
    with pytest.raises(ValueError, match="backward"):
        compare_chain(a, a, require_backward=True)


def test_reject_wrong_pass_and_nonfinite_logprob():
    a = report_fixture()
    b = copy.deepcopy(a)
    b["records"][18]["pass_index"] = 0
    with pytest.raises(ValueError, match="pass mapping"):
        compare_chain(a, b)
    b = copy.deepcopy(a)
    b["selected_logprob_bits"][0] = 0x7F800000
    with pytest.raises(ValueError, match="nonfinite"):
        compare_chain(a, b)
