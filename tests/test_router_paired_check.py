# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Torch paired-check tests.

Until the fixture manifest is published (anchor_pending), manifest-based
tests skip; behavioral tests use synthetic entries to pin the rules:
- paired diff is diagnostic, never flips strict verdicts
- incompatible tensors and non-finite diagnostics are invalid evidence
- missing evidence -> MISSING_PROVENANCE (67)
- malformed manifest entries are skipped, never guessed
"""

from __future__ import annotations

import json

import pytest
import torch

from rl_engine.moe.t01_verdicts import RouterVerdict
from rl_engine.moe.validation.paired_check import (
    GOLDEN_MANIFEST_PATH,
    GoldenEntry,
    PairedRecord,
    load_golden_manifest,
    paired_gate_for_goldens,
    run_paired_check,
)


def _golden(name="g1"):
    return GoldenEntry(
        name=name,
        fixture_path="fixtures/router/g1.pt",
        checksum="ab" * 32,
        kind="random",
        router_mode="learned",
    )


# --- manifest -----------------------------------------------------------------
def test_manifest_absent_means_anchor_pending_empty():
    """No manifest -> [] (documented anchor_pending), never a guess."""
    entries = load_golden_manifest(GOLDEN_MANIFEST_PATH)
    # in-repo state today: start kit unpublished
    if not GOLDEN_MANIFEST_PATH.is_file():
        assert entries == []


def test_manifest_malformed_entries_skipped(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps(
            {
                "goldens": [
                    {"name": "ok", "fixture_path": "f", "checksum": "c"},
                    {"no_name": True},  # malformed -> skipped
                    "a-string",  # malformed -> skipped
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = load_golden_manifest(p)
    assert [e.name for e in entries] == ["ok"]


def test_manifest_corrupt_json_returns_empty(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_golden_manifest(p) == []


@pytest.mark.parametrize("payload", [42, {"goldens": 42}])
def test_manifest_wrong_root_type_returns_empty(tmp_path, payload):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert load_golden_manifest(p) == []


# --- paired run ---------------------------------------------------------------
def test_paired_run_records_diagnostics():
    gold = {"w": torch.tensor([1.0, 2.0, 3.0])}
    inputs = {"x": 1}
    ref = lambda inputs: {"w": torch.tensor([1.0, 2.0, 3.5])}  # noqa: E731
    rep = run_paired_check("cx", _golden(), gold, ref, inputs)
    assert rep.passed
    rec = rep.extra["paired_record"]
    assert rec["n_tensors_compared"] == 1
    assert rec["torch_max_abs_diff"] == pytest.approx(0.5)
    assert rep.extra["diagnostic_exact"] is False
    assert rep.extra["diagnostic_warning"] is not None


def test_paired_diff_is_diagnostic_never_flips_strict():
    """Even a huge Torch diff keeps the check PASS — strict gates rule."""
    gold = {"w": torch.tensor([1.0])}
    ref = lambda inputs: {"w": torch.tensor([99.0])}  # noqa: E731
    rep = run_paired_check("cx", _golden(), gold, ref, {})
    assert rep.passed  # diagnostic-only by design
    assert rep.extra["paired_record"]["torch_max_abs_diff"] > 1.0
    assert "strict L3a/L3b verdict unchanged" in rep.extra["diagnostic_warning"]


def test_paired_missing_execution_evidence_is_67():
    gold = {"w": torch.tensor([1.0])}
    rep = run_paired_check("cx", _golden(), gold, torch_reference=None, case_inputs={})
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE


def test_paired_reference_crash_is_missing_evidence_not_pass():
    """A crashing Torch run is absent evidence — never silently green."""

    def bad_ref(inputs):
        raise RuntimeError("boom")

    rep = run_paired_check("cx", _golden(), {"w": torch.tensor([1.0])}, bad_ref, {})
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
    assert "boom" in rep.detail


def test_paired_reference_missing_required_tensor_fails_67():
    rep = run_paired_check(
        "cx",
        _golden(),
        {"w": torch.tensor([1.0])},
        lambda inputs: {},
        {},
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
    assert rep.extra["missing"] == ["w"]


def test_paired_empty_golden_cannot_pass_with_zero_evidence():
    rep = run_paired_check("cx", _golden(), {}, lambda inputs: {}, {})
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE


def test_paired_empty_tensor_is_not_comparison_evidence():
    empty = torch.empty(0)
    rep = run_paired_check(
        "cx",
        _golden(),
        {"w": empty},
        lambda inputs: {"w": empty.clone()},
        {},
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE


def test_paired_reference_must_return_tensor_mapping():
    rep = run_paired_check(
        "cx",
        _golden(),
        {"w": torch.ones(1)},
        lambda inputs: [torch.ones(1)],
        {},
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE


def test_paired_shape_or_dtype_mismatch_is_invalid_evidence():
    gold = {"w": torch.zeros(4, dtype=torch.float32)}
    incompatible = (
        torch.zeros(5, dtype=torch.float32),
        torch.zeros(4, dtype=torch.float64),
    )
    for ref_tensor in incompatible:
        rep = run_paired_check(
            "cx",
            _golden(),
            gold,
            lambda inputs, value=ref_tensor: {"w": value},
            {},
        )
        assert not rep.passed
        assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
        assert rep.extra["tensor"] == "w"
        assert "not comparable" in rep.detail


def test_paired_existing_record_short_circuits():
    """Pre-shipped paired evidence counts; no local Torch run needed."""
    record = PairedRecord(
        golden_name="g1", torch_max_abs_diff=0.0, torch_mean_abs_diff=0.0, n_tensors_compared=2
    )
    rep = run_paired_check("cx", _golden(), {}, None, {}, existing_record=record)
    assert rep.passed
    assert rep.extra["paired_record"]["n_tensors_compared"] == 2
    assert rep.extra["diagnostic_exact"] is True
    assert rep.extra["diagnostic_warning"] is None


def test_paired_existing_record_must_match_golden_and_compare_tensors():
    wrong = PairedRecord(
        golden_name="other", torch_max_abs_diff=0.0, torch_mean_abs_diff=0.0, n_tensors_compared=1
    )
    rep = run_paired_check("cx", _golden(), {}, None, {}, existing_record=wrong)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE

    empty = PairedRecord(
        golden_name="g1", torch_max_abs_diff=0.0, torch_mean_abs_diff=0.0, n_tensors_compared=0
    )
    rep = run_paired_check("cx", _golden(), {}, None, {}, existing_record=empty)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE

    nonfinite = PairedRecord(
        golden_name="g1",
        torch_max_abs_diff=float("inf"),
        torch_mean_abs_diff=0.0,
        n_tensors_compared=1,
    )
    rep = run_paired_check("cx", _golden(), {}, None, {}, existing_record=nonfinite)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
    assert "non-finite" in rep.detail

    incoherent = PairedRecord(
        golden_name="g1", torch_max_abs_diff=0.1, torch_mean_abs_diff=0.2, n_tensors_compared=1
    )
    rep = run_paired_check("cx", _golden(), {}, None, {}, existing_record=incoherent)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE


# --- gate ---------------------------------------------------------------------
def test_gate_all_goldens_covered_passes():
    goldens = [_golden("a"), _golden("b")]
    records = {"a": PairedRecord("a", 0.0, 0.0, 1), "b": PairedRecord("b", 0.0, 0.0, 1)}
    rep = paired_gate_for_goldens("cx", goldens, records)
    assert rep.passed


def test_gate_missing_golden_evidence_fails_67():
    goldens = [_golden("a"), _golden("b")]
    records = {"a": PairedRecord("a", 0.0, 0.0, 1)}
    rep = paired_gate_for_goldens("cx", goldens, records)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
    assert rep.extra["missing"] == ["b"]


def test_gate_rejects_nonfinite_paired_record():
    goldens = [_golden("a")]
    records = {"a": PairedRecord("a", float("nan"), 0.0, 1)}
    rep = paired_gate_for_goldens("cx", goldens, records)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
    assert rep.extra["missing"] == ["a"]


def test_gate_empty_manifest_is_missing_provenance_not_vacuous_pass():
    rep = paired_gate_for_goldens("cx", [], {})
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE


# --- anchor_pending integration -------------------------------------------------
@pytest.mark.skipif(
    not GOLDEN_MANIFEST_PATH.is_file(),
    reason="start kit unpublished (anchor_pending); "
    "revisit when fixtures/router/manifest.json lands",
)
def test_real_manifest_goldens_all_have_gate_slots():
    goldens = load_golden_manifest()
    assert goldens, "manifest exists but is empty — manifest contract violation"
    rep = paired_gate_for_goldens("real", goldens, records={})
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_PROVENANCE
    assert set(rep.extra["missing"]) == {g.name for g in goldens}
