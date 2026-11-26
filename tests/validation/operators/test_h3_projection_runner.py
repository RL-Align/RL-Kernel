# SPDX-License-Identifier: Apache-2.0
"""Exercise local refusal evidence and the full matrix orchestration with a CPU surrogate."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.ops.gemm import h3_projections as ops
from rl_engine.reference.gemm.h3_projections import reference_projection

ROOT = Path(__file__).resolve().parents[3]
REPLAY_TOOL = ROOT / "tools/validation/operators/compare_h3_projection_replays.py"
spec = importlib.util.spec_from_file_location(
    "h3_runner", ROOT / "tools/validation/operators/validate_h3_projections.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_cpu_refusal_preserves_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "report.json"
    assert runner.main(["--output", str(path)]) == 2
    report = json.loads(path.read_text())
    assert report["status"] == "error"
    assert "SM90 GPU is required" in report["error"]
    assert not report["acceptance_complete"] and not report["cases"]
    for name, digest in report["provenance"]["source_hashes"].items():
        assert runner.sha256(ROOT / name) == digest
        assert "Compatibility import" not in (ROOT / name).read_text()


@pytest.mark.parametrize("kind", tuple(ops.SPECS))
@pytest.mark.parametrize(
    "rows,max_rows,inapplicable",
    [
        (4, 21, set()),
        (3, 21, {"batch2_reshape"}),
        (4, 4, {"inactive_padding"}),
        (3, 3, {"inactive_padding", "batch2_reshape"}),
    ],
)
def test_matrix_shapes_with_independent_cpu_reference(
    kind, rows, max_rows, inapplicable, monkeypatch
):
    monkeypatch.setitem(ops.SPECS, kind, (4, 6))
    monkeypatch.setattr(runner, "H3_MAX_ROWS", max_rows)
    monkeypatch.setattr(
        runner,
        "training_projection",
        lambda op, x, w, dy, keys, **kw: (reference_projection(x, w, dy), []),
    )
    monkeypatch.setattr(
        runner, "row_local", lambda op, x, w, dy: reference_projection(x, w, dy)[:2]
    )
    monkeypatch.setattr(runner, "timed", lambda fn, repeats: {"not_executed": True})
    monkeypatch.setattr(runner, "_require_backend", lambda *args: None)
    # Powers of two keep the independent CPU reductions exactly representable.
    x = torch.ones(rows, 4).bfloat16()
    w = torch.ones(6, 4).bfloat16()
    dy = torch.ones(rows, 6).bfloat16()
    case = runner.run_case(x, w, dy, "cpu_orchestration_only", 1, kind=kind)
    assert case["passed"]
    assert {row["transform"] for row in case["checks"]} == {
        "fp32_reference",
        "repeat",
        "permuted_logical_rows",
        "chunk31_single_backward",
        "strided_input_and_weight",
        "inactive_padding",
        "batch2_reshape",
        "singleton_vs_batch",
        "other_rows_mutated",
    } - inapplicable
    assert {item["transform"] for item in case["inapplicable_transforms"]} == inapplicable
    assert all(item["reason"] for item in case["inapplicable_transforms"])


@pytest.mark.parametrize("tensor", ("y", "dx", "dw"))
def test_cpu_surrogate_detects_reference_failures(tensor, monkeypatch):
    kind = "h3_attention_o_gemm"
    monkeypatch.setitem(ops.SPECS, kind, (4, 6))

    def inaccurate_projection(op, x, w, dy, keys, **kwargs):
        values = list(reference_projection(x, w, dy))
        values[("y", "dx", "dw").index(tensor)] += 1
        return tuple(values), []

    monkeypatch.setattr(runner, "training_projection", inaccurate_projection)
    monkeypatch.setattr(
        runner, "row_local", lambda op, x, w, dy: reference_projection(x, w, dy)[:2]
    )
    monkeypatch.setattr(runner, "timed", lambda fn, repeats: {"not_executed": True})
    monkeypatch.setattr(runner, "_require_backend", lambda *args: None)
    x = torch.ones(4, 4).bfloat16()
    w = torch.ones(6, 4).bfloat16()
    dy = torch.ones(4, 6).bfloat16()
    case = runner.run_case(x, w, dy, "cpu_orchestration_only", 1, kind=kind)
    assert not case["passed"]
    reference_check = next(
        check
        for check in case["checks"]
        if check["transform"] == "fp32_reference" and check["tensor"] == tensor
    )
    assert not reference_check["passed"] and reference_check["failing_elements"] > 0


@pytest.fixture
def replay_report():
    return {
        "status": "passed",
        "provenance": {
            name: "test-only"
            for name in (
                "source_revision",
                "source_hashes",
                "checkpoint_revision",
                "seed",
                "contract_sha256",
            )
        },
        "cases": [
            {
                "semantic_op": "h3_attention_o_gemm",
                "input_family": "test",
                "rows": 1,
                "output_digests": {name: {"sha256": "test-only"} for name in ("y", "dx", "dw")},
            }
        ],
    }


def test_replay_rejects_failed_sources_and_detects_drift(replay_report):
    replay_spec = importlib.util.spec_from_file_location("h3_replay", REPLAY_TOOL)
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    assert replay.compare(replay_report, replay_report)["status"] == "passed"
    changed = copy.deepcopy(replay_report)
    changed["cases"][0]["output_digests"]["dw"]["sha256"] = "drift"
    assert replay.compare(replay_report, changed)["status"] == "failed"
    changed["provenance"]["source_revision"] = "different"
    with pytest.raises(ValueError, match="provenance mismatch"):
        replay.compare(replay_report, changed)
    changed["status"] = "failed"
    with pytest.raises(ValueError, match="both reports"):
        replay.compare(replay_report, changed)


@pytest.mark.parametrize(
    "scenario,exit_code,status,error",
    [
        ("same", 0, "passed", None),
        ("digest", 1, "failed", None),
        ("status", 2, "error", "both reports"),
        ("provenance", 2, "error", "replay provenance mismatch"),
        ("ambiguous", 2, "error", "ambiguous duplicate replay case"),
        ("subset", 2, "error", "nonempty subset"),
        ("invalid_json", 2, "error", "invalid original report JSON"),
        ("unreadable", 2, "error", "could not read replay report"),
    ],
)
def test_replay_cli_preserves_input_hashes(
    replay_report, tmp_path, scenario, exit_code, status, error
):
    original, replay, output = (tmp_path / f"{name}.json" for name in ("original", "replay", "out"))
    second = copy.deepcopy(replay_report)
    if scenario == "digest":
        second["cases"][0]["output_digests"]["dw"]["sha256"] = "drift"
    elif scenario == "status":
        second["status"] = "failed"
    elif scenario == "provenance":
        second["provenance"]["source_revision"] = "different"
    elif scenario == "ambiguous":
        second["cases"].append(second["cases"][0])
    elif scenario == "subset":
        second["cases"][0]["rows"] += 1
    original_raw = b"{" if scenario == "invalid_json" else json.dumps(replay_report).encode()
    replay_raw = None if scenario == "unreadable" else json.dumps(second).encode()
    original.write_bytes(original_raw)
    if replay_raw is not None:
        replay.write_bytes(replay_raw)

    result = subprocess.run(
        [sys.executable, str(REPLAY_TOOL), str(original), str(replay), "--output", str(output)],
        check=False,
    )

    assert result.returncode == exit_code
    report = json.loads(output.read_text())
    assert report["status"] == status
    assert report["report_sha256"] == [
        hashlib.sha256(raw).hexdigest() if raw is not None else None
        for raw in (original_raw, replay_raw)
    ]
    if error is not None:
        assert error in report["error"] and report["checks"] == []
    else:
        assert "error" not in report
        assert len(report["checks"]) == 3
        assert [check["passed"] for check in report["checks"]] == [True, True, status == "passed"]


def test_replay_cli_wont_overwrite_an_input_report(tmp_path):
    original = tmp_path / "original.json"
    replay = tmp_path / "replay.json"
    original.write_bytes(b"{")
    replay.write_bytes(b"{}")
    output = tmp_path / "alias.json"
    output.hardlink_to(original)

    result = subprocess.run(
        [sys.executable, str(REPLAY_TOOL), str(original), str(replay), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--output must differ from both input reports" in result.stderr
    assert original.read_bytes() == b"{"
    assert replay.read_bytes() == b"{}"


def test_pinned_task_geometry():
    layouts = runner.task_layouts()
    assert [row["latent_frames"] for row in layouts] == [37, 72, 102]
    assert [row["rows"] for row in layouts] == [39982, 75658, 106238]
    assert all(row["activation_kind"] == "synthetic" for row in layouts)
    assert all(row["rows"] <= ops.H3_MAX_ROWS for row in layouts)
