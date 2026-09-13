# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Torch paired-check tests (T09 step 7, contract §2.5).

Until T01 publishes the fixture manifest (anchor_pending), manifest-based
tests skip; behavioral tests use synthetic entries to pin the rules:
- paired diff is diagnostic, never flips strict verdicts
- missing evidence -> MISSING_PROVENANCE (67)
- malformed manifest entries are skipped, never guessed
"""

from __future__ import annotations

import json

import pytest
import torch

from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.paired_check import (
    GOLDEN_MANIFEST_PATH,
    GoldenEntry,
    PairedRecord,
    load_golden_manifest,
    paired_gate_for_goldens,
    run_paired_check,
)


def _golden(name="g1"):
    return GoldenEntry(name=name, fixture_path="fixtures/p3/g1.pt",
                       checksum="ab" * 32, kind="random", router_mode="learned")


# --- manifest -----------------------------------------------------------------

def test_manifest_absent_means_anchor_pending_empty():
    """No manifest -> [] (documented anchor_pending), never a guess."""
    entries = load_golden_manifest(GOLDEN_MANIFEST_PATH)
    # in-repo state today: start kit unpublished
    if not GOLDEN_MANIFEST_PATH.is_file():
        assert entries == []


def test_manifest_malformed_entries_skipped(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"goldens": [
        {"name": "ok", "fixture_path": "f", "checksum": "c"},
        {"no_name": True},                       # malformed -> skipped
        "a-string",                              # malformed -> skipped
    ]}), encoding="utf-8")
    entries = load_golden_manifest(p)
    assert [e.name for e in entries] == ["ok"]


def test_manifest_corrupt_json_returns_empty(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{not json", encoding="utf-8")
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


def test_paired_diff_is_diagnostic_never_flips_strict():
    """Even a huge Torch diff keeps the check PASS — strict gates rule."""
    gold = {"w": torch.tensor([1.0])}
    ref = lambda inputs: {"w": torch.tensor([99.0])}  # noqa: E731
    rep = run_paired_check("cx", _golden(), gold, ref, {})
    assert rep.passed                      # diagnostic-only by design
    assert rep.extra["paired_record"]["torch_max_abs_diff"] > 1.0


def test_paired_missing_execution_evidence_is_67():
    gold = {"w": torch.tensor([1.0])}
    rep = run_paired_check("cx", _golden(), gold, torch_reference=None, case_inputs={})
    assert not rep.passed
    assert rep.verdict is P3Verdict.MISSING_PROVENANCE


def test_paired_reference_crash_is_missing_evidence_not_pass():
    """A crashing Torch run is absent evidence — never silently green."""
    def bad_ref(inputs):
        raise RuntimeError("boom")

    rep = run_paired_check("cx", _golden(), {"w": torch.tensor([1.0])}, bad_ref, {})
    assert not rep.passed
    assert rep.verdict is P3Verdict.MISSING_PROVENANCE
    assert "boom" in rep.detail


def test_paired_shape_mismatch_records_inf_diff():
    gold = {"w": torch.zeros(4)}
    ref = lambda inputs: {"w": torch.zeros(5)}  # noqa: E731
    rep = run_paired_check("cx", _golden(), gold, ref, {})
    assert rep.passed
    assert rep.extra["paired_record"]["torch_max_abs_diff"] == float("inf")


def test_paired_existing_record_short_circuits():
    """T01-shipped paired evidence counts; no local Torch run needed."""
    record = PairedRecord(golden_name="g1", torch_max_abs_diff=0.0,
                          torch_mean_abs_diff=0.0, n_tensors_compared=2)
    rep = run_paired_check("cx", _golden(), {}, None, {}, existing_record=record)
    assert rep.passed
    assert rep.extra["paired_record"]["n_tensors_compared"] == 2


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
    assert rep.verdict is P3Verdict.MISSING_PROVENANCE
    assert rep.extra["missing"] == ["b"]


# --- anchor_pending integration -------------------------------------------------

@pytest.mark.skipif(not GOLDEN_MANIFEST_PATH.is_file(),
                    reason="T01 start kit unpublished (anchor_pending); "
                           "revisit when fixtures/p3/manifest.json lands")
def test_real_manifest_goldens_all_have_gate_slots():
    goldens = load_golden_manifest()
    assert goldens, "manifest exists but is empty — T01 contract violation"
    rep = paired_gate_for_goldens("real", goldens, records={})
    assert not rep.passed
    assert rep.verdict is P3Verdict.MISSING_PROVENANCE
    assert set(rep.extra["missing"]) == {g.name for g in goldens}
