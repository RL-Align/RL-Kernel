# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pinned layer component inventory and original tensor scheduler acceptance."""

import hashlib
import json
from pathlib import Path

from qwen3_next_runtime import file_digest, identity_digest


def layer_requirements(root, ep=1):
    """Mandatory layer gates; the MoE gate follows the expert parallelism (1 or 4)."""
    root = Path(root)
    path = root / "rl_engine/validation/models/qwen3_next_layer_cases.json"
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1 or set(registry.get("gates", {})) != {
        "gdn",
        "attention",
        "moe",
        "moe_ep",
        "adapters",
        "heads",
    }:
        raise ValueError("Incomplete layer gate registry")
    if ep not in (1, 4):
        raise ValueError("Layer gates exist for EP 1 and EP 4 only")
    registry["gates"].pop("moe_ep" if ep == 1 else "moe")
    for gate in registry["gates"].values():
        gate["script_sha256"] = file_digest(root / gate["script"])
        for cases in gate["groups"].values():
            if not cases or len(cases) != len(set(cases)):
                raise ValueError("Mandatory layer case list is empty or duplicated")
    cases = sorted(
        f"{name}/{group}/{case}"
        for name, gate in registry["gates"].items()
        for group, values in gate["groups"].items()
        for case in values
    )
    return {
        **registry,
        "registry_sha256": file_digest(path),
        "case_ids_per_rank": cases,
        "case_ids_sha256": hashlib.sha256(("\n".join(cases) + "\n").encode()).hexdigest(),
    }


def validate_rank_result(row, requirement, rank):
    if any(
        row.get(name) != value
        for name, value in {
            "status": "passed",
            "rank": rank,
            "scope": requirement["scope"],
            "full_model_l2": "not_evaluated",
            **requirement.get("fields", {}),
        }.items()
    ):
        raise ValueError("Layer rank evidence has a wrong identity, scope or status")
    if any(row.get(name, 0) for name in ("skipped", "failures", "errors")):
        raise ValueError("Layer rank evidence includes skipped or failed checks")
    for field, expected in requirement["groups"].items():
        value = row
        for key in field.split("."):
            value = value.get(key, {}) if isinstance(value, dict) else None
        if not isinstance(value, list) or sorted(value) != sorted(expected):
            raise ValueError(f"Layer rank evidence changed mandatory cases: {field}")


def checked_record(record, *, root=None):
    if not isinstance(record, dict) or not {"path", "sha256"} <= record.keys():
        raise ValueError("Missing artifact path/content digest")
    path = Path(record["path"])
    if not path.is_absolute():
        if root is None:
            raise ValueError("Artifact path must be absolute without an evidence root")
        path = Path(root) / path
    path = path.resolve(strict=True)
    if root is not None and not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escaped its current attempt")
    if file_digest(path) != record["sha256"]:
        raise ValueError("Artifact content digest changed")
    return path


def validate_layer_report(report, requirement, identity, *, root):
    if any(
        report.get(key) != value
        for key, value in {
            "schema_version": 1,
            "status": "passed",
            "phase": "complete",
            "scope": "qwen3_next_layer_and_scheduler_gates",
            "identity": identity,
            "requirements_sha256": identity_digest(requirement),
            "full_model_l2": "not_evaluated",
        }.items()
    ):
        raise ValueError("Layer acceptance identity or completeness mismatch")
    if set(report.get("gates", {})) != set(requirement["gates"]):
        raise ValueError("Layer acceptance omitted a mandatory gate")
    from rl_engine.validation.models.qwen3_next_binary import validate_binary_record

    binary = validate_binary_record(
        report.get("provider_binary"), source_tree=identity["source_tree"], artifact_root=root
    )
    for name, expected in requirement["gates"].items():
        gate = report["gates"][name]
        if gate.get("exit_code") != 0 or gate.get("gpu_release_verified") is not True:
            raise ValueError("Layer process failed or did not release GPU memory")
        ranks = gate.get("ranks", {})
        if set(ranks) != {"0", "1", "2", "3"}:
            raise ValueError("Layer acceptance requires each of the four TP ranks")
        for rank, record in ranks.items():
            row = json.loads(checked_record(record, root=root).read_text())
            validate_rank_result(row, expected, int(rank))
            if row.get("provider_binary") != binary:
                raise ValueError("Layer ranks did not import the archived CUDA provider")
    scheduler_path = checked_record(report.get("scheduler"), root=root)
    scheduler = json.loads(scheduler_path.read_text())
    from rl_engine.validation.models.qwen3_next_l2 import validate_scheduler_report

    validate_scheduler_report(scheduler, expected_identity=identity, artifact_root=root)
    if scheduler.get("provider_binary") != report["provider_binary"]:
        raise ValueError("Layer and scheduler gates used different CUDA providers")
    if scheduler["case_ids"] != requirement["scheduler_case_ids"]:
        raise ValueError("Scheduler evidence differs from the pinned layer registry")
