# SPDX-License-Identifier: Apache-2.0
"""Exercise local refusal evidence and the full matrix orchestration with a CPU surrogate."""
import importlib.util
from pathlib import Path

import pytest
import torch

from rl_engine.ops.gemm import h3_projections as ops
from rl_engine.reference.gemm.h3_projections import reference_projection

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "h3_runner", ROOT / "tools/validation/operators/validate_h3_projections.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_cpu_refusal_preserves_provenance(tmp_path, monkeypatch):
    import json

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
def test_matrix_shapes_with_independent_cpu_reference(kind, monkeypatch):
    monkeypatch.setitem(ops.SPECS, kind, (4, 6))
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
    x = torch.ones(4, 4).bfloat16()
    w = torch.ones(6, 4).bfloat16()
    dy = torch.ones(4, 6).bfloat16()
    case = runner.run_case(x, w, dy, "cpu_orchestration_only", 1, kind=kind)
    assert case["passed"]
    assert {row["transform"] for row in case["checks"]} >= {
        "fp32_reference",
        "repeat",
        "permuted_logical_rows",
        "chunk31_single_backward",
        "strided_input_and_weight",
        "inactive_padding",
        "batch2_reshape",
        "singleton_vs_batch",
        "other_rows_mutated",
    }


def test_replay_rejects_failed_sources_and_detects_drift():
    import copy

    replay_spec = importlib.util.spec_from_file_location(
        "h3_replay",
        ROOT / "tools/validation/operators/compare_h3_projection_replays.py",
    )
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    report = {
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
    assert replay.compare(report, report)["status"] == "passed"
    changed = copy.deepcopy(report)
    changed["cases"][0]["output_digests"]["dw"]["sha256"] = "drift"
    assert replay.compare(report, changed)["status"] == "failed"
    changed["provenance"]["source_revision"] = "different"
    with pytest.raises(ValueError, match="provenance mismatch"):
        replay.compare(report, changed)
    changed["status"] = "failed"
    with pytest.raises(ValueError, match="both reports"):
        replay.compare(report, changed)


def test_pinned_task_geometry():
    layouts = runner.task_layouts()
    assert [row["latent_frames"] for row in layouts] == [37, 72, 102]
    assert [row["rows"] for row in layouts] == [39982, 75658, 106238]
    assert all(row["activation_kind"] == "synthetic" for row in layouts)
    assert all(row["rows"] <= ops.H3_MAX_ROWS for row in layouts)
