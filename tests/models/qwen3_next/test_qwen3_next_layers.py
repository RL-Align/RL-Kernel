# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU failure-injection tests; none constitutes native model evidence."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/validation/models"))

from qwen3_next_layer_contract import (  # noqa: E402
    checked_record,
    layer_requirements,
    validate_layer_report,
    validate_rank_result,
)
from qwen3_next_runtime import STAGES  # noqa: E402
from qwen3_next_runtime import identity_digest  # noqa: E402


def rank_row(requirement, rank):
    row = {
        "status": "passed",
        "rank": rank,
        "scope": requirement["scope"],
        "full_model_l2": "not_evaluated",
        **requirement.get("fields", {}),
    }
    for path, cases in requirement["groups"].items():
        target = row
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = list(cases)
    return row


def test_layers_are_required_between_preflight_and_l2():
    assert STAGES["layers"][:3] == ("512G", "04:00:00", "preflight")
    assert STAGES["l2"][2] == "layers"


@pytest.mark.parametrize("gate", ["gdn", "attention", "moe", "adapters", "heads"])
def test_rank_gate_requires_complete_unduplicated_inventory(gate):
    requirement = layer_requirements(ROOT)["gates"][gate]
    row = rank_row(requirement, 2)
    validate_rank_result(row, requirement, 2)
    for field in ("rank", "scope", "status", "full_model_l2"):
        altered = {**row, field: "changed"}
        with pytest.raises(ValueError, match="identity"):
            validate_rank_result(altered, requirement, 2)
    for skipped in (1, 3):
        with pytest.raises(ValueError, match="skipped"):
            validate_rank_result({**row, "skipped": skipped}, requirement, 2)
    for path in requirement["groups"]:
        for duplicate in (False, True):
            changed = copy.deepcopy(row)
            value = changed
            for part in path.split("."):
                value = value[part]
            value.append(value[0]) if duplicate else value.pop()
            with pytest.raises(ValueError, match="mandatory"):
                validate_rank_result(changed, requirement, 2)


def test_artifact_hash_and_attempt_escape_are_rejected(tmp_path):
    from qwen3_next_runtime import file_digest

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    path = attempt / "rank.json"
    path.write_text("{}")
    record = {"path": str(path), "sha256": file_digest(path)}
    assert checked_record(record, root=attempt) == path
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="digest"):
        checked_record(record, root=attempt)
    record["sha256"] = file_digest(path)
    with pytest.raises(ValueError, match="escaped"):
        checked_record(record, root=attempt / "different")


def test_complete_component_flags_cannot_replace_native_scheduler(tmp_path):
    from rl_engine.validation.models.qwen3_next_binary import (
        archive_provider_binary,
        validate_binary_record,
    )

    source = tmp_path / "source"
    (source / "rl_engine").mkdir(parents=True)
    (source / "rl_engine/_C.cpython-unit.so").write_bytes(b"unit native binary")
    provider_binary = archive_provider_binary(source, tmp_path, source_tree="unit")
    actual_binary = validate_binary_record(provider_binary)
    identity = {"source_tree": "unit"}
    requirement = layer_requirements(ROOT)
    report = {
        "schema_version": 1,
        "status": "passed",
        "phase": "complete",
        "scope": "qwen3_next_layer_and_scheduler_gates",
        "identity": identity,
        "requirements_sha256": identity_digest(requirement),
        "full_model_l2": "not_evaluated",
        "gates": {},
        "provider_binary": provider_binary,
    }
    from qwen3_next_runtime import file_digest

    for name, expected in requirement["gates"].items():
        records = {}
        for rank in range(4):
            path = tmp_path / f"{name}-{rank}.json"
            path.write_text(
                json.dumps({**rank_row(expected, rank), "provider_binary": actual_binary})
            )
            records[str(rank)] = {"path": str(path), "sha256": file_digest(path)}
        report["gates"][name] = {"exit_code": 0, "gpu_release_verified": True, "ranks": records}
    with pytest.raises(ValueError, match="artifact path"):
        validate_layer_report(report, requirement, identity, root=tmp_path)
    report["phase"] = "native_scheduler"
    with pytest.raises(ValueError, match="completeness"):
        validate_layer_report(report, requirement, identity, root=tmp_path)


def test_missing_native_scheduler_is_explicitly_incomplete(monkeypatch):
    import qwen3_next_layers as runner

    def missing(_):
        raise ModuleNotFoundError("native scheduler missing")

    monkeypatch.setattr(runner.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="native_scheduler_not_ready"):
        runner.check_scheduler_ready()
