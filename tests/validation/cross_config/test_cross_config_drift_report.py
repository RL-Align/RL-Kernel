from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import rl_engine.validation.cross_config.__main__ as cli_main
from rl_engine.validation.cross_config.artifacts import ArtifactStore
from rl_engine.validation.reports.drift_report import (
    build_cross_config_attempt_report,
    build_drift_report,
    build_drift_trace,
    load_drift_bundle,
    render_drift_report,
    render_drift_report_image,
    write_drift_bundle,
    write_drift_report,
    write_drift_report_image,
    write_drift_trace,
)


def _artifacts(*, with_timestamp: bool = False):
    sample = {
        "sample_position": 0,
        "sample_index": 17,
        "rollout_id": 23,
        "batch_layout_fingerprint": "layout-1",
    }
    if with_timestamp:
        sample.update(start_ts=10.0, end_ts=10.5)
    manifest = {
        "mode": "audit",
        "samples": [sample],
        "batch_invariance_cases": [{"case": "same_sample_alone"}],
        "validation": {"warnings": [], "failures": []},
        "runtime_provenance": {"operator": "linear_logp"},
    }
    cube = {
        "mode": "audit",
        "rank": 0,
        "axes": {"dtype": "bf16", "cp": 1, "logp_backend": "rlk.linear_logp.fast"},
        "metrics": {
            "active_token_count": 2,
            "max_abs_dlogp": 0.125,
            "warning_count": 1,
            "metadata_warning_count": 0,
            "metadata_failure_count": 0,
        },
        "worst_token": {"abs_dlogp": 0.125, "sample_position": 0, "token_position": 4},
        "metadata_validation": {"warnings": [], "failures": []},
        "runtime_provenance": {
            "actual_backend": "rl_engine.linear_logp",
            "fallback": False,
        },
    }
    return manifest, cube


def _completed_attempt(root: Path) -> Path:
    store = ArtifactStore(root)
    attempt = store.create_attempt("drift-report", "case-1")
    envelope = {"case_id": "case-1", "attempt_id": attempt.name}
    requested = {
        "rollout": {
            "tensor_parallel_size": 2,
            "context_parallel_size": 2,
            "dtype": "bfloat16",
        },
        "training": {"compute_dtype": "bfloat16", "sharding": "tp2-cp2"},
    }
    store.write_json(
        attempt,
        "requested",
        {
            **envelope,
            "schema_version": "cross_config.requested.v1",
            "case": {"requested": requested},
        },
    )
    store.write_json(
        attempt,
        "materialized",
        {
            **envelope,
            "schema_version": "cross_config.materialized_envelope.v1",
            "materialized_case": {"case": {"requested": requested}},
        },
    )
    store.write_json(
        attempt,
        "identity",
        {
            **envelope,
            "schema_version": "cross_config.identity_envelope.v1",
            "identity": {"token_ids": [[11, 12], [21, 22]]},
        },
    )
    store.write_json(
        attempt,
        "actual",
        {
            **envelope,
            "schema_version": "cross_config.actual.v1",
            "operator_source": "exact_resolution_and_instance",
            "execution_fingerprint": "execution-sha",
            "environment_fingerprint": "environment-sha",
            "rollout": {"backend_id": "rlkernel.attention.deterministic.v1"},
            "training": {"backend_id": "rlkernel.ffn.qwen3.deterministic.v1"},
        },
    )
    store.write_json(
        attempt,
        "comparison",
        {
            **envelope,
            "schema_version": "cross_config.alignment_result.v1",
            "status": "pass",
            "comparable": True,
            "passed": True,
            "mismatch_count": 0,
            "fixed_threshold": 0.0,
            "contract_fingerprint": "contract-sha",
            "diagnostics": {},
        },
    )
    scores = {
        "selected_logprobs": torch.zeros((2, 2)),
        "active_mask": torch.tensor([[False, True], [False, True]]),
    }

    def write_tensor_bundle(name: str, tensors: dict[str, torch.Tensor]) -> None:
        torch.save(
            {"schema_version": 1, "tensors": tensors, "metadata": envelope},
            attempt / f"{name}.pt",
        )

    for name in ("score_rollout", "score_training"):
        write_tensor_bundle(name, scores)
    write_tensor_bundle(
        "token_diffs",
        {
            "rollout_logprobs": torch.zeros((2, 2)),
            "training_logprobs": torch.zeros((2, 2)),
            "active_mask": scores["active_mask"],
            "absolute_diff": torch.zeros((2, 2)),
            "mismatch_mask": torch.zeros((2, 2), dtype=torch.bool),
        },
    )
    store.complete_attempt(
        attempt,
        summary={
            **envelope,
            "schema_version": "cross_config.complete.v1",
            "status": "pass",
        },
    )
    return attempt


@pytest.mark.unit
def test_report_uses_ordinal_timeline_without_fabricating_timestamps():
    manifest, cube = _artifacts()

    report = build_drift_report(replay_manifest=manifest, result_cube=cube)

    assert report["timeline_mode"] == "ordinal_diagnostic"
    assert "not elapsed time" in report["timeline_note"]
    assert {event["lane"] for event in report["events"]} == {
        "Training audit",
        "Rollout samples",
        "Operator / backend",
        "Drift markers",
    }
    assert report["status"] == "warning"


@pytest.mark.unit
def test_report_prefers_actual_backend_and_timestamp_mode():
    manifest, cube = _artifacts(with_timestamp=True)
    cube["runtime_provenance"] = {
        "requested_backend": "registry",
        "actual_backend": "native.linear_logp",
        "fallback": True,
    }

    report = build_drift_report(replay_manifest=manifest, result_cube=cube)
    operator = next(event for event in report["events"] if event["id"] == "operator-backend")

    assert report["timeline_mode"] == "timestamp"
    assert operator["label"] == "native.linear_logp"
    assert operator["status"] == "warning"


@pytest.mark.unit
def test_rendered_report_is_self_contained_and_escapes_details(tmp_path: Path):
    manifest, cube = _artifacts()
    manifest["validation"]["warnings"] = [{"code": "bad<&", "message": "value </script>"}]
    cube["metadata_validation"] = manifest["validation"]

    report = build_drift_report(
        replay_manifest=manifest,
        result_cube=cube,
        title="<diagnostic>",
    )
    html = render_drift_report(report)
    output = write_drift_report(report, tmp_path / "drift.html")

    assert output.exists()
    assert "<diagnostic>" not in html
    assert "value </script>" not in html
    assert "ordinal_diagnostic" in html
    assert "Operator / backend" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "detail.innerHTML" not in html
    assert "detail.replaceChildren" in html


@pytest.mark.unit
def test_static_report_image_is_shareable_png(tmp_path: Path):
    pytest.importorskip("PIL")
    manifest, cube = _artifacts()
    report = build_drift_report(replay_manifest=manifest, result_cube=cube)

    image = render_drift_report_image(report)
    assert image.size == (2400, 1680)

    output = write_drift_report_image(report, tmp_path / "drift.png")
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_consistency_trace_is_expandable_chrome_trace_json(tmp_path: Path):
    manifest, cube = _artifacts()
    report = build_drift_report(replay_manifest=manifest, result_cube=cube)

    trace = build_drift_trace(report)
    assert trace["metadata"]["timeline_mode"] == "ordinal_diagnostic"
    assert any(
        event.get("ph") == "M" and event.get("name") == "thread_name"
        for event in trace["traceEvents"]
    )
    assert any(
        event.get("ph") == "X" and event.get("cat") == "consistency.audit"
        for event in trace["traceEvents"]
    )
    assert any(
        event.get("ph") == "I" and event.get("cat") == "consistency.drift"
        for event in trace["traceEvents"]
    )

    output = write_drift_trace(report, tmp_path / "drift.json")
    assert output.read_text(encoding="utf-8").startswith('{\n  "traceEvents"')


@pytest.mark.unit
def test_consistency_bundle_contains_report_trace_and_preview(tmp_path: Path):
    pytest.importorskip("PIL")
    manifest, cube = _artifacts()
    report = build_drift_report(replay_manifest=manifest, result_cube=cube)

    output = write_drift_bundle(report, tmp_path / "drift.rlk-drift")
    assert output.exists()

    import zipfile

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "report.json",
            "trace.json",
            "preview.png",
        }

    loaded = load_drift_bundle(output)
    assert loaded["manifest"]["format"] == "rl_kernel.cross_config_drift"
    assert loaded["manifest"]["preview_included"] is True
    assert loaded["report"]["status"] == "warning"
    assert loaded["trace"]["metadata"]["timeline_mode"] == "ordinal_diagnostic"


@pytest.mark.unit
def test_consistency_bundle_can_omit_preview(tmp_path: Path):
    manifest, cube = _artifacts()
    report = build_drift_report(replay_manifest=manifest, result_cube=cube)

    output = write_drift_bundle(
        report,
        tmp_path / "drift-no-preview.rlk-drift",
        include_preview=False,
    )

    import zipfile

    with zipfile.ZipFile(output) as archive:
        assert "preview.png" not in archive.namelist()
        bundle_manifest = json.loads(archive.read("manifest.json"))
    assert bundle_manifest["preview_included"] is False
    assert bundle_manifest["files"] == ["manifest.json", "report.json", "trace.json"]


@pytest.mark.unit
def test_attempt_report_reads_only_sealed_cross_config_artifacts(tmp_path: Path):
    attempt_dir = _completed_attempt(tmp_path)

    report = build_cross_config_attempt_report(attempt_dir)

    assert report["status"] == "pass"
    assert report["axes"]["rollout_tp"] == 2
    assert report["axes"]["rollout_cp"] == 2
    assert report["metrics"]["active_token_count"] == 2
    assert report["runtime_provenance"]["operator_source"] == "exact_resolution_and_instance"


@pytest.mark.unit
def test_attempt_report_rejects_unsealed_artifact_directory(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    attempt = store.create_attempt("unfinished", "case-1")

    with pytest.raises(Exception, match="COMPLETE"):
        build_cross_config_attempt_report(attempt)


@pytest.mark.unit
def test_cli_writes_offline_bundle_from_sealed_attempt(tmp_path: Path, capsys):
    attempt = _completed_attempt(tmp_path)
    output = tmp_path / "report.rlk-drift"

    assert cli_main.main(["report", str(attempt), "--output", str(output), "--no-preview"]) == 0

    assert output.is_file()
    assert "drift report: pass" in capsys.readouterr().err
