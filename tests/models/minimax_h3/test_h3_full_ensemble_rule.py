# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The ensemble rule is calibrated on provider comparisons and applied to candidate ones."""

from __future__ import annotations

import argparse
import json

import pytest

from tools.validation.models import h3_full_ensemble_rule as rule

REFERENCE = "ref-sha"


def _comparison(status: str, scale: float = 1.0, **extra) -> dict:
    # The gradient error is above 1, as the recorded provider spread of the real model is.
    errors = {
        "video": {"relative_l2": 0.010 * scale, "max_abs": 1.0},
        "parameter.w": {"relative_l2": 1.1 * scale, "max_abs": 2.0},
    }
    report = {"status": status, "reference_report_sha256": REFERENCE, "errors": errors, **extra}
    if status in rule.CANDIDATE_STATUSES:
        report["tensor_l2"] = {t: {"reference": 1.0, "other": 1.0} for t in errors}
    return report


def _write(path, report) -> str:
    path.write_text(json.dumps(report))
    return str(path)


def _calibrate(tmp_path, reports: dict[str, dict]):
    providers = [f"{name}={_write(tmp_path / f'{name}.json', r)}" for name, r in reports.items()]
    out = tmp_path / "rule.json"
    rule.calibrate(argparse.Namespace(provider=providers, output=out))
    return out


def _providers(count=3):
    return {f"p{i}": _comparison(rule.PROVIDER_STATUS, 1.0 + 0.1 * i) for i in range(count)}


def test_providers_calibrate_a_rule_a_candidate_can_be_checked_against(tmp_path):
    rule_path = _calibrate(tmp_path, _providers())
    _write(tmp_path / "candidate.json", _comparison("passed", 1.05))
    out = tmp_path / "out.json"
    rule.check(
        argparse.Namespace(rule=rule_path, candidate=tmp_path / "candidate.json", output=out)
    )
    assert json.loads(out.read_text())["status"] == "passed"


@pytest.mark.parametrize("status", ["passed", "failed", "diagnostic_only", None])
def test_calibration_rejects_anything_that_is_not_a_provider_comparison(tmp_path, status):
    reports = _providers()
    reports["p0"] = _comparison(status)
    if status is None:
        del reports["p0"]["status"]
    with pytest.raises(ValueError, match="p0 is (not a provider comparison|a hybrid)"):
        _calibrate(tmp_path, reports)


def test_candidate_check_rejects_a_provider_comparison(tmp_path):
    rule_path = _calibrate(tmp_path, _providers())
    _write(tmp_path / "provider.json", _comparison(rule.PROVIDER_STATUS))
    with pytest.raises(ValueError, match="not a candidate comparison"):
        rule.check(
            argparse.Namespace(
                rule=rule_path, candidate=tmp_path / "provider.json", output=tmp_path / "o.json"
            )
        )


def _check(tmp_path, report):
    """Run ``rule.check``; return the written output and the process exit code (0 if it returns)."""

    rule_path = _calibrate(tmp_path, _providers())
    _write(tmp_path / "candidate.json", report)
    out = tmp_path / "out.json"
    args = argparse.Namespace(rule=rule_path, candidate=tmp_path / "candidate.json", output=out)
    code = 0
    try:
        rule.check(args)
    except SystemExit as exit_:
        code = exit_.code
    assert out.exists(), "check must write its output before it exits"
    return json.loads(out.read_text()), code


def test_a_zeroed_gradient_is_rejected_although_its_relative_error_is_within_the_bound(tmp_path):
    report = _comparison("passed", 0.9)
    report["errors"]["parameter.w"]["relative_l2"] = 1.0  # what an all-zero gradient measures
    report["tensor_l2"]["parameter.w"]["other"] = 0.0
    out, code = _check(tmp_path, report)
    assert out["status"] == "failed"
    assert out["zeroed_gradients"] == ["parameter.w"]
    assert out["failures"] == ["parameter.w"]
    assert code == 1  # the process-failure contract that CI consumes


def test_the_same_error_with_a_nonzero_gradient_passes(tmp_path):
    report = _comparison("passed", 0.9)
    report["errors"]["parameter.w"]["relative_l2"] = 1.0
    out, code = _check(tmp_path, report)
    assert out["status"] == "passed"
    assert out["zeroed_gradients"] == []
    assert code == 0


def test_a_gradient_that_is_zero_in_the_reference_too_is_not_a_zeroed_gradient(tmp_path):
    report = _comparison("passed", 0.9)
    report["tensor_l2"]["parameter.w"] = {"reference": 0.0, "other": 0.0}
    out, code = _check(tmp_path, report)
    assert out["zeroed_gradients"] == [] and code == 0


def test_a_comparison_without_tensor_norms_is_refused(tmp_path):
    report = _comparison("passed", 0.9)
    del report["tensor_l2"]
    rule_path = _calibrate(tmp_path, _providers())
    _write(tmp_path / "candidate.json", report)
    with pytest.raises(ValueError, match="per-tensor norms"):
        rule.check(
            argparse.Namespace(
                rule=rule_path, candidate=tmp_path / "candidate.json", output=tmp_path / "o.json"
            )
        )


def _evidence_root(path, tensors):
    import torch

    from tools.validation.models.h3_full_model import digest

    path.mkdir()
    payload = {
        "video": tensors["video"],
        "audio": tensors["audio"],
        "nodes": {},
        "input_gradients": {},
    }
    torch.save(payload, path / "tensors.pt")
    grads = {}
    for name, value in tensors["parameters"].items():
        torch.save(value, path / f"{name}.pt")
        grads[name] = {"path": f"{name}.pt", "sha256": digest(path / f"{name}.pt")}
    (path / "gradients.json").write_text(json.dumps(grads))
    (path / "report.json").write_text(json.dumps({"tensors_sha256": digest(path / "tensors.pt")}))


def test_compare_reports_the_norms_of_both_sides(tmp_path):
    import torch

    from tools.validation.models import h3_full_compare as cmp

    ref = {"video": torch.ones(4), "audio": torch.ones(2), "parameters": {"w": torch.ones(3) * 2}}
    other = {"video": torch.ones(4), "audio": torch.ones(2), "parameters": {"w": torch.zeros(3)}}
    _evidence_root(tmp_path / "ref", ref)
    _evidence_root(tmp_path / "other", other)
    norms: dict = {}
    rows = dict(cmp.compare(tmp_path / "ref", tmp_path / "other", norms))
    assert rows["parameter.w"]["relative_l2"] == 1.0  # the zeroed gradient: error exactly 1
    assert norms["parameter.w"] == {"reference": pytest.approx(12**0.5), "other": 0.0}
    assert norms["video"]["reference"] == norms["video"]["other"] == 2.0
    assert dict(cmp.compare(tmp_path / "ref", tmp_path / "other")) == rows  # norms are optional


def test_the_reference_manifest_and_root_flags_go_together(tmp_path, monkeypatch):
    from tools.validation.models import h3_full_compare as cmp

    base = ["h3_full_compare", "--reference", "r", "--other", "o", "--output", str(tmp_path / "x")]
    for extra in (["--reference-manifest", "m.json"], ["--reference-root", "root"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(ValueError, match="given together"):
            cmp.main()


def test_candidate_prepare_creates_a_nested_output_but_refuses_an_existing_one(
    tmp_path, monkeypatch
):
    from tools.validation.models import h3_candidate_prepare as prep

    target = tmp_path / "evidence" / "run1" / "prepare"
    monkeypatch.setattr(
        "sys.argv",
        ["prep", "--checkpoint", str(tmp_path), "--reference-root", str(tmp_path)]
        + ["--output", str(target)],
    )
    with pytest.raises(Exception):  # it fails later, on the missing reference evidence
        prep.main()
    assert target.is_dir()  # the nested output directory was created before that
    with pytest.raises(FileExistsError):
        prep.main()
