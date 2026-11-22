"""Fixed-rollout contract guards: thresholds files, bounds sources and control reports."""

import hashlib
import json

import pytest

from tools.validation.models.qwen3_ppo.compare_fixed import expected_thresholds, load_contract
from tools.validation.models.qwen3_ppo.freeze_fixed_reference import (
    check_bounds_source,
    check_report_inputs,
)


@pytest.mark.parametrize(
    "content",
    [{}, [], {"bounds": {}}, {"bounds": {"0/loss": {"max_abs": 1.0}}}],
)
def test_load_contract_rejects_anything_but_frozen_bounds(tmp_path, content):
    path = tmp_path / "bounds.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="not a frozen bounds contract"):
        load_contract(path)


def test_load_contract_hashes_the_bytes_it_parsed(tmp_path):
    path = tmp_path / "bounds.json"
    raw = json.dumps({"reference_report_sha256": "r", "bounds": {"0/loss": {"max_abs": 1.0}}})
    path.write_text(raw)
    contract, digest = load_contract(path)
    assert contract["bounds"]["0/loss"]["max_abs"] == 1.0
    assert digest == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize(
    ("role", "precision", "tp_comparison", "name"),
    [
        ("actor", "fp32", False, "actor-fp32-repeat-bounds.json"),
        ("critic", "bf16", False, "critic-cross-precision-bounds.json"),
        ("actor", "fp32", True, "actor-fp32-repeat-bounds.json"),
        ("critic", "bf16", True, "critic-bf16-repeat-bounds.json"),
    ],
)
def test_each_comparison_names_its_bounds_file(role, precision, tp_comparison, name):
    """The names qualify_candidate.sbatch passes, so its own selections still pass."""
    other = {"role": role, "precision": precision}
    assert expected_thresholds(other, tp_comparison=tp_comparison) == name


def _bounds(reference="r0", other="r1", controls=("impl",)):
    return {
        "status": "frozen_reference_bounds",
        "reference_report_sha256": reference,
        "other_report_sha256": other,
        "extra_controls": [{"report_sha256": c} for c in controls],
    }


def test_bounds_must_come_from_the_selected_reports():
    check_bounds_source(_bounds(), reference="r0", other="r1", controls=["impl"], name="b")
    for wrong in (_bounds(other="x"), _bounds(controls=("other-impl",)), _bounds(controls=())):
        with pytest.raises(ValueError, match="not frozen from the selected control reports"):
            check_bounds_source(wrong, reference="r0", other="r1", controls=["impl"], name="b")


def test_control_reports_must_match_role_precision_and_inputs():
    report = dict(
        role="actor",
        precision="fp32",
        rollout_manifest_sha256="roll",
        targets_manifest_sha256="tgt",
    )
    kwargs = dict(role="actor", precision="fp32", rollout_sha256="roll", targets_sha256="tgt")
    check_report_inputs(report, **kwargs)
    with pytest.raises(ValueError, match="is not actor fp32"):
        check_report_inputs({**report, "precision": "bf16"}, **kwargs)
    with pytest.raises(ValueError, match="another rollout or targets"):
        check_report_inputs({**report, "rollout_manifest_sha256": "other"}, **kwargs)
