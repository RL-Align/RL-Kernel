# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Versioned acceptance checks shared by the stage runners and the launcher."""

import hashlib
import json
import re
from pathlib import Path

from qwen3_next_runtime import atomic_json, file_digest, identity_digest  # noqa: F401

CONTRACT_VERSION = 4
SCOPES = {
    "components": "qwen3_next_components_only",
    "diagnostics": "qwen3_next_operator_diagnostics",
    "preflight": "four_gpu_infrastructure_and_operators_only",
    "layers": "qwen3_next_layer_and_scheduler_gates",
    "l2": "qwen3_next_full_model_l2",
    "train1": "qwen3_next_training_update1",
    "train2": "qwen3_next_training_update2",
    "train": "qwen3_next_training_and_reload",
    "bi": "qwen3_next_batch_invariance",
}
# Stages that evaluate the full model against updated weights (an L2 matrix each).
TRAIN_STAGES = ("train1", "train2", "train")
IDENTITY_FIELDS = (
    "contract_version",
    "source_tree",
    "environment_sha256",
    "cuda_identity",
    "checkpoint_identity",
    "provider_config",
    "topology",
    "operator_requirements",
    "layer_requirements",
)


def checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint)
    metadata = {
        str(path.relative_to(checkpoint)): file_digest(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file() and path.suffix != ".safetensors"
    }
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    reference = json.loads(
        (
            Path(__file__).resolve().parents[4]
            / "rl_engine/validation/models/qwen3_next_norm_manifest.json"
        ).read_text()
    )["model_identity"]
    snapshot = reference["weight_snapshot"]
    if (
        checkpoint.name != reference["revision"]
        or metadata.get(snapshot["index_file"]) != snapshot["index_sha256"]
    ):
        raise ValueError("Checkpoint metadata does not match the official pinned revision")
    expected_shards = {item["filename"]: item for item in snapshot["shards"]}
    if set(index["weight_map"].values()) != expected_shards.keys():
        raise ValueError("Checkpoint shard inventory differs from the official revision")
    shards = {}
    for filename, expected in sorted(expected_shards.items()):
        stat = (checkpoint / filename).stat()
        if stat.st_size != expected["size_bytes"]:
            raise ValueError(f"Checkpoint shard size mismatch: {filename}")
        shards[filename] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "expected_sha256": expected["sha256"],
        }
    return {
        "revision": checkpoint.name,
        "metadata": metadata,
        "shards": shards,
        "expected_content_hash": snapshot["content_hash"],
        "full_hash_verification": "required_before_model_load",
    }


def operator_requirements(root, suite):
    registry = json.loads(
        (
            Path(root) / "rl_engine/validation/models/qwen3_next/qwen3_next_operator_cases.json"
        ).read_text()
    )
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported mandatory operator case registry")
    suites = registry["suites"]
    cases = sorted(suites["norm"] + suites["core"] if suite == "all" else suites[suite])
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("Mandatory operator case registry is empty or duplicated")
    gates = (
        sorted(
            f"{op}-{gate}.json"
            for op in ("qwen3_next_rms_norm", "rms_norm_gated")
            for gate in ("forward", "gradient")
        )
        if suite in ("all", "norm")
        else []
    )
    return {
        "suite": suite,
        "case_ids": cases,
        "case_ids_sha256": hashlib.sha256(("\n".join(cases) + "\n").encode()).hexdigest(),
        "norm_gates": gates,
    }


def latest_attempt(run_dir, scheduler_attempt=None):
    values = [
        int(path.name)
        for path in (Path(run_dir) / "attempts").glob("*")
        if path.is_dir() and path.name.isdigit()
    ]
    if scheduler_attempt is not None:
        if not str(scheduler_attempt).isdigit():
            raise ValueError("Invalid scheduler attempt identity")
        values.append(int(scheduler_attempt))
    return str(max(values)) if values else None


def acceptance_identity(manifest):
    return {field: manifest[field] for field in IDENTITY_FIELDS}


def result_envelope(manifest, attempt, *, evidence):
    return {
        **acceptance_identity(manifest),
        "source_commit": manifest["source_commit"],
        "run_id": manifest["run_id"],
        "stage": manifest["stage"],
        "suite": manifest.get("suite", "all"),
        "execution_id": manifest["execution_id"],
        "attempt": str(attempt),
        "status": "diagnostic_completed" if manifest["stage"] == "diagnostics" else "passed",
        "phase": "complete",
        "scope": SCOPES[manifest["stage"]],
        "full_model_l2": "not_evaluated",
        "evidence": evidence,
    }


def evidence_record(path, root):
    return {"path": str(Path(path).relative_to(root)), "sha256": file_digest(path)}


def validate_partial_training_evidence(manifest, root, evidence, *, attempt_root):
    """train1/train2: the progress record, its last phase's worker export and L2 matrix agree.

    A training phase's L2 runs on that phase's exported weights, so its matrix
    identity is the run's model identity with the export as the checkpoint (the
    pinned base recorded as ``base_checkpoint``), exactly as the stage built it.
    """
    from qwen3_next_l2 import model_identity
    from qwen3_next_training import STAGE_LAST_PHASE, read_record, stage_phases

    from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import validate_matrix_report

    stage = manifest["stage"]
    progress = json.loads((root / evidence["updates"]["path"]).read_text())
    phases = list(stage_phases(stage))
    last = STAGE_LAST_PHASE[stage]
    if (
        progress.get("status") != "passed"
        or progress.get("scope") != SCOPES[stage]
        or progress.get("identity") != model_identity(manifest)
        or progress.get("phases") != phases
        or any(phase not in progress.get("workers", {}) for phase in phases)
    ):
        raise ValueError("Training progress identity, scope or phases mismatch")
    matrix_record = progress.get("l2_reports", {}).get(last, {})
    if matrix_record.get("sha256") != evidence["l2_matrix"]["sha256"]:
        raise ValueError("Training progress does not reference the accepted L2 matrix")
    worker = read_record(progress["workers"][last])
    export = worker.get("export") or {}
    if (
        export.get("weight_version") is None
        or export.get("base_checkpoint") != manifest["checkpoint"]
    ):
        raise ValueError("Training phase export does not derive from the pinned base checkpoint")
    report = json.loads((root / evidence["l2_matrix"]["path"]).read_text())
    validate_matrix_report(
        report,
        expected_identity=model_identity({**manifest, **export}),
        artifact_root=attempt_root,
    )


def validate_result(manifest, result, *, run_dir=None, attempt=None):
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Historical acceptance contract cannot authorize a new stage")
    expected = {
        **acceptance_identity(manifest),
        "source_commit": manifest["source_commit"],
        "run_id": manifest["run_id"],
        "stage": manifest["stage"],
        "suite": manifest.get("suite", "all"),
        "status": "diagnostic_completed" if manifest["stage"] == "diagnostics" else "passed",
        "phase": "complete",
        "scope": SCOPES[manifest["stage"]],
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"Acceptance identity mismatch: {field}")
    if not re.fullmatch(r"[0-9a-f]{32}", str(manifest.get("execution_id", ""))):
        raise ValueError("Run has no launcher execution identity")
    if result.get("execution_id") != manifest["execution_id"]:
        raise ValueError("Acceptance execution identity mismatch")
    if not str(result.get("attempt", "")).isdigit():
        raise ValueError("Acceptance has no attempt identity")
    if run_dir is not None:
        attempt = latest_attempt(run_dir, attempt)
    if attempt is not None and result["attempt"] != str(attempt):
        raise ValueError("Acceptance belongs to a stale attempt")
    if manifest["stage"] in ("l2", *TRAIN_STAGES) and result.get("full_model_l2") != "passed":
        raise ValueError("Full-model L2 evidence is required")
    if (
        manifest["stage"] not in ("l2", *TRAIN_STAGES)
        and result.get("full_model_l2") != "not_evaluated"
    ):
        raise ValueError("Component and layer acceptance cannot claim full-model L2")
    if manifest["stage"] == "train" and result.get("training_and_reload") != "passed":
        raise ValueError("Two updates and reload evidence is required")
    evidence = result.get("evidence", {})
    required = {"operators", "environment"}
    if manifest["stage"] == "preflight":
        required |= {"device_guards", "tp4", "checkpoint", "rollout_smoke"}
    if manifest["stage"] == "components" and manifest.get("suite") in ("norm", "all"):
        required.add("device_guards")
    if manifest["stage"] in ("l2", *TRAIN_STAGES):
        required |= {"l2_matrix"}
    if manifest["stage"] == "l2":
        required.add("provider_binary")
    if manifest["stage"] == "layers":
        required |= {"layers", "checkpoint", "provider_binary"}
    if manifest["stage"] == "bi":
        required |= {"bi_matrix", "provider_binary"}
    if manifest["stage"] == "train":
        required |= {"updates", "reload"}
    if manifest["stage"] in ("train1", "train2"):
        required |= {"updates"}
    if not required <= evidence.keys():
        raise ValueError(f"Incomplete acceptance evidence: {sorted(required - evidence.keys())}")
    if run_dir is not None:
        root = Path(run_dir).resolve()
        attempt_root = root / "attempts" / result["attempt"]
        for record in evidence.values():
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(attempt_root) or file_digest(path) != record["sha256"]:
                raise ValueError("Acceptance evidence content/attempt mismatch")
        operators = json.loads((root / evidence["operators"]["path"]).read_text())
        if (
            operators.get("status") != expected["status"]
            or operators.get("git_commit") != manifest["source_commit"]
        ):
            raise ValueError("Operator evidence must match the completed source run")
        requirement = manifest["operator_requirements"]
        if manifest["stage"] == "diagnostics":
            if (
                operators.get("diagnostic_only") is not True
                or operators.get("case_ids_sha256") != requirement["case_ids_sha256"]
            ):
                raise ValueError("Diagnostic evidence case identity mismatch")
            return identity_digest(expected)
        case_ids = operators.get("mandatory_case_ids", [])
        if (
            case_ids != requirement["case_ids"]
            or operators.get("case_ids_sha256") != requirement["case_ids_sha256"]
        ):
            raise ValueError("Evidence differs from the pinned mandatory case registry")
        gates = operators.get("norm_gates", {})
        if sorted(gates) != requirement["norm_gates"] or any(
            gate.get("passed") is not True for gate in gates.values()
        ):
            raise ValueError("Norm C3/C4 evidence is missing or failed")
        expected_digest = hashlib.sha256(("\n".join(sorted(case_ids)) + "\n").encode()).hexdigest()
        counts = operators.get("provider_tests", {})
        if (
            not case_ids
            or len(case_ids) != len(set(case_ids))
            or len(case_ids) != counts.get("tests")
            or operators.get("case_ids_sha256") != expected_digest
        ):
            raise ValueError("Mandatory operator case identities do not match the evidence")
        if not counts.get("tests") or any(
            counts.get(key, 0) for key in ("skipped", "errors", "failures")
        ):
            raise ValueError("Required operator tests must pass without skips")
        if manifest["stage"] == "bi":
            from qwen3_next_l2 import model_identity

            from rl_engine.validation.models import qwen3_next_bi_cases as bi

            matrix = json.loads((root / evidence["bi_matrix"]["path"]).read_text())
            suite = bi.quick_cells() if manifest.get("bi_suite") == "quick" else bi.cells()
            if (
                matrix.get("status") != "passed"
                or matrix.get("identity") != model_identity(manifest)
                or matrix.get("cell_ids") != [cell.cell_id for cell in suite]
                or any(not matrix["cells"][cell.cell_id]["passed"] for cell in suite)
            ):
                raise ValueError("Batch-invariance matrix is incomplete or failed")
        if manifest["stage"] in ("layers", "l2", *TRAIN_STAGES):
            from qwen3_next_l2 import model_identity

            if manifest["stage"] == "layers":
                from qwen3_next_layer_contract import validate_layer_report

                checkpoint = json.loads((root / evidence["checkpoint"]["path"]).read_text())
                if any(
                    checkpoint.get(key) != value
                    for key, value in {
                        "revision": manifest["checkpoint_identity"]["revision"],
                        "content_hash": manifest["checkpoint_identity"]["expected_content_hash"],
                        "shards_verified": len(manifest["checkpoint_identity"]["shards"]),
                    }.items()
                ):
                    raise ValueError(
                        "Layer checkpoint verification differs from the pinned snapshot"
                    )
                report = json.loads((root / evidence["layers"]["path"]).read_text())
                validate_layer_report(
                    report,
                    manifest["layer_requirements"],
                    model_identity(manifest),
                    root=attempt_root,
                )
            else:
                from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import (
                    validate_matrix_report,
                )

                certificate = manifest.get("scheduler_profile_certificate", {})
                certificate_path = Path(certificate.get("path", ""))
                if not certificate_path.is_absolute() or file_digest(
                    certificate_path
                ) != certificate.get("sha256"):
                    raise ValueError("Strict L2 requires its pinned native scheduler certificate")
                certified_manifest = json.loads((certificate_path.parent / "run.json").read_text())
                certified_result = json.loads(certificate_path.read_text())
                if certified_manifest.get("stage") != "layers" or acceptance_identity(
                    certified_manifest
                ) != acceptance_identity(manifest):
                    raise ValueError("Strict L2 certificate belongs to another provider identity")
                validate_result(
                    certified_manifest, certified_result, run_dir=certificate_path.parent
                )
                if manifest["stage"] == "train":
                    from qwen3_next_training import validate_training_report

                    if evidence["updates"]["sha256"] != evidence["reload"]["sha256"]:
                        raise ValueError(
                            "Update and reload evidence must name the same complete report"
                        )
                    report = json.loads((root / evidence["updates"]["path"]).read_text())
                    sections = {
                        "baseline",
                        "update1",
                        "update2",
                        "reload",
                        "continuous_step3",
                        "resumed_step3",
                        "checkpoint",
                        "fault_injection",
                    }
                    if (
                        report.get("identity") != model_identity(manifest)
                        or set(report.get("required_sections", [])) != sections
                    ):
                        raise ValueError("Training report identity or required sections mismatch")
                    if (
                        report.get("l2_reports", {}).get("reload", {}).get("sha256")
                        != evidence["l2_matrix"]["sha256"]
                    ):
                        raise ValueError(
                            "Training reload does not reference the accepted full L2 matrix"
                        )
                    validate_training_report(report, verify_files=True)
                elif manifest["stage"] in ("train1", "train2"):
                    validate_partial_training_evidence(
                        manifest, root, evidence, attempt_root=attempt_root
                    )
                else:
                    report = json.loads((root / evidence["l2_matrix"]["path"]).read_text())
                    validate_matrix_report(
                        report,
                        expected_identity=model_identity(manifest),
                        artifact_root=attempt_root,
                    )
    return identity_digest(expected)
