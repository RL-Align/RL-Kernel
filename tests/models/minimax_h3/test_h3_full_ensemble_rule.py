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
    errors = {
        "video": {"relative_l2": 0.010 * scale, "max_abs": 1.0},
        "block.0.grad": {"relative_l2": 0.020 * scale, "max_abs": 2.0},
    }
    return {"status": status, "reference_report_sha256": REFERENCE, "errors": errors, **extra}


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
