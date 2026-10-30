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
    rule_path = _calibrate(tmp_path, _providers())
    _write(tmp_path / "candidate.json", report)
    out = tmp_path / "out.json"
    args = argparse.Namespace(rule=rule_path, candidate=tmp_path / "candidate.json", output=out)
    try:
        rule.check(args)
    except SystemExit:
        pass
    return json.loads(out.read_text())


def test_a_zeroed_gradient_is_rejected_although_its_relative_error_is_within_the_bound(tmp_path):
    report = _comparison("passed", 0.9)
    report["errors"]["parameter.w"]["relative_l2"] = 1.0  # what an all-zero gradient measures
    report["tensor_l2"]["parameter.w"]["other"] = 0.0
    out = _check(tmp_path, report)
    assert out["status"] == "failed"
    assert out["zeroed_gradients"] == ["parameter.w"]
    assert out["failures"] == ["parameter.w"]


def test_the_same_error_with_a_nonzero_gradient_passes(tmp_path):
    report = _comparison("passed", 0.9)
    report["errors"]["parameter.w"]["relative_l2"] = 1.0
    out = _check(tmp_path, report)
    assert out["status"] == "passed"
    assert out["zeroed_gradients"] == []


def test_a_gradient_that_is_zero_in_the_reference_too_is_not_a_zeroed_gradient(tmp_path):
    report = _comparison("passed", 0.9)
    report["tensor_l2"]["parameter.w"] = {"reference": 0.0, "other": 0.0}
    assert _check(tmp_path, report)["zeroed_gradients"] == []


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
