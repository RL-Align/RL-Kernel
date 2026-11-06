# SPDX-License-Identifier: Apache-2.0
"""Check plot data preparation without a plotting runtime or GPU."""

import json
import math
import sys

import pytest

from benchmarks.operators.gemm.plot_h3_projections import (
    _accuracy_notes,
    _accuracy_series,
    _load_cases,
    _load_reports,
    main,
)


def test_accuracy_uses_all_families_and_distinguishes_failed_ratios():
    def accuracy(ratio):
        return {"tensor": "y", "transform": "fp32_reference", **ratio}

    cases = [
        {
            "rows": 8,
            "input_family": "random",
            "checks": [accuracy({"max_error_over_tolerance": 0.2})],
        },
        {
            "rows": 8,
            "input_family": "checkpoint_fixture",
            "checks": [accuracy({"passed": False, "error": "nonfinite tensor"})],
        },
        {
            "rows": 16,
            "input_family": "checkpoint_fixture",
            "checks": [accuracy({"max_error_over_tolerance": 1.4, "passed": False})],
        },
        {"rows": 32, "input_family": "cancellation", "checks": []},
        {
            "rows": 64,
            "input_family": "random",
            "checks": [accuracy({"max_error_over_tolerance": math.inf, "passed": False})],
        },
    ]

    sizes, values, failed, unavailable = _accuracy_series(cases, "y")

    assert sizes == [8, 16, 32, 64]
    assert values[:2] == [0.2, 1.4]
    assert all(math.isnan(value) for value in values[2:])
    assert failed == {8, 64}
    assert unavailable == {32}


def _timing_case(rows):
    return {
        "semantic_op": "h3_attention_o_gemm",
        "input_family": "random",
        "rows": rows,
        "checks": [
            {
                "tensor": "y",
                "transform": "fp32_reference",
                "max_error_over_tolerance": 0.2,
                "tolerance": {"atol": 0.05, "rtol": 0.02},
            }
        ],
        "benchmarks": {"candidate_forward": {"median_ms": 0.3}},
        "execution_traces": [{"forward_launch": {"triton_version": "3.2.0"}}],
    }


def _timing_report(cases):
    return {
        "schema": "h3-projections-qualification-v1",
        "cases": cases,
        "provenance": {
            "source_revision": "source",
            "source_hashes": {"operator.py": "source-hash"},
            "working_tree_clean": True,
            "python": "3.10.12",
            "torch": "2.6.0",
            "cuda": "12.4",
            "seed": 420,
            "repeats": 5,
            "checkpoint_revision": "checkpoint",
            "contract_sha256": "contract-hash",
            "method": "FP32 reference",
        },
        "hardware": {
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "architecture": [9, 0],
            "device_count_used": 1,
            "memory_bytes": 85017493504,
        },
    }


@pytest.mark.parametrize("wrapped", [False, True])
def test_reports_keep_failed_task_accuracy_and_prefer_full_timing(wrapped):
    size = 32768
    full = _timing_case(size)
    failed_check = {**full["checks"][0], "max_error_over_tolerance": 1.4, "passed": False}
    overlap = {**_timing_case(size), "checks": [failed_check]}
    task = _timing_case(2 * size)
    full_report = _timing_report([full])
    tasks_report = _timing_report([overlap, task])
    if wrapped:
        bundle = {
            "artifacts": {
                "tasks.json": {"content": tasks_report},
                "full.json": {"content": full_report},
            }
        }
        reports = _load_reports(bundle)
    else:
        reports = _load_reports(full_report) + _load_reports(tasks_report)

    cases, timing, timing_unverified = _load_cases(reports)

    assert not timing_unverified
    assert _accuracy_series(cases, "y") == ([size, 2 * size], [1.4, 0.2], set(), set())
    assert timing == {full["semantic_op"]: {size: full, 2 * size: task}}


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("provenance", "source_revision", "other-source"),
        ("provenance", "torch", "2.4.1"),
        ("provenance", "seed", 123),
        ("provenance", "repeats", 1),
        ("hardware", "gpu_model", "NVIDIA H20"),
    ],
)
def test_combined_timings_reject_mismatched_provenance(section, field, value):
    first = _timing_report([_timing_case(33)])
    second = _timing_report([_timing_case(129)])
    second[section][field] = value

    with pytest.raises(ValueError, match="different provenance"):
        _load_cases([first, second])


def test_combined_timings_require_provenance_even_if_both_reports_lack_it():
    first = {"cases": [_timing_case(33)]}
    second = {"cases": [_timing_case(129)]}

    with pytest.raises(ValueError, match="lack complete provenance"):
        _load_cases([first, second])


def test_accuracy_only_report_keeps_completed_failures_without_timing_provenance():
    timed = _timing_case(33)
    failed = _timing_case(129)
    failed["benchmarks"] = {}
    failed["checks"][0].update(passed=False, max_error_over_tolerance=None)
    accuracy_only = {"status": "error", "cases": [failed]}

    cases, timing, timing_unverified = _load_cases([_timing_report([timed]), accuracy_only])

    assert not timing_unverified
    assert timing == {timed["semantic_op"]: {33: timed}}
    sizes, values, failed_sizes, unavailable = _accuracy_series(cases, "y")
    assert sizes == [33, 129]
    assert values[0] == 0.2 and math.isnan(values[1])
    assert failed_sizes == {129}
    assert unavailable == set()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"artifacts": []},
        {"artifacts": {"full.json": None}},
        {"artifacts": {"tasks.json": {"content": []}}},
        {"artifacts": {"tasks.json": {"content": {"cases": {}}}}},
        {"cases": None},
        {"cases": [None]},
    ],
)
def test_malformed_evidence_is_a_cli_error(payload, tmp_path, monkeypatch, capsys):
    evidence = tmp_path / "malformed.json"
    evidence.write_text(json.dumps(payload))
    output = tmp_path / "plot.svg"
    monkeypatch.setattr(sys, "argv", ["plot", "--evidence", str(evidence), "--output", str(output)])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "error: malformed.json:" in capsys.readouterr().err
    assert not output.exists()


def test_single_report_can_render_with_unverified_timing_provenance():
    report = {"cases": [_timing_case(33)]}

    cases, timing, timing_unverified = _load_cases([report])

    assert cases == report["cases"]
    assert timing[report["cases"][0]["semantic_op"]][33] == cases[0]
    assert timing_unverified


@pytest.mark.parametrize(
    ("provenance", "tolerance", "expected_note"),
    [
        ({}, None, None),
        ({"source_revision": None}, None, "incomplete"),
        ({"source_revision": "other-source"}, None, "differ"),
        ({}, {"atol": 0.06, "rtol": 0.02}, "differ"),
    ],
)
def test_accuracy_notes_distinguish_incomplete_and_mixed_reports(
    provenance, tolerance, expected_note
):
    first = _timing_report([_timing_case(33)])
    second = _timing_report([_timing_case(129)])
    second["provenance"].update(provenance)
    if tolerance is not None:
        second["cases"][0]["checks"][0]["tolerance"] = tolerance

    notes = _accuracy_notes([first, second])

    if expected_note is None:
        assert not notes
    else:
        assert len(notes) == 1 and expected_note in notes[0]
