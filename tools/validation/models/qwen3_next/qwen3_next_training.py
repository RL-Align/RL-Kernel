#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native RL update, full-model L2 and atomic restart acceptance on four GPUs."""

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from qwen3_next_checkpoint import (  # noqa: E402
    newest_complete_checkpoint,
    publish_checkpoint,
    resume_interrupted_publications,
    validate_checkpoint,
    write_rank_manifest,
)
from qwen3_next_contract import (  # noqa: E402
    SCOPES,
    atomic_json,
    evidence_record,
    result_envelope,
    validate_result,
)
from qwen3_next_l2 import check_ready as check_l2_ready  # noqa: E402
from qwen3_next_l2 import (  # noqa: E402
    model_identity,
    run_matrix,
    run_process,
    verify_before_model_load,
    wait_gpu_release,
)
from qwen3_next_runtime import failure_fields  # noqa: E402
from qwen3_next_runtime import (  # noqa: E402
    file_digest,
    owned_root,
    recover_storage,
    retire_directory,
    training_storage_root,
    verify_environment,
)

from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next.native_training import (  # noqa: E402,E501
    policy_fixture,
    run_training_phase,
    validate_active_parameter_names,
)
from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_training import (  # noqa: E402
    state_digest,
)
from rl_engine.validation.models.qwen3_next.qwen3_next_binary import (  # noqa: E402
    archive_provider_binary,
    validate_binary_record,
)
from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import (  # noqa: E402
    case_matrix,
    checked_artifact_path,
    compare_artifacts,
    json_digest,
    load_tensor_artifact,
    validate_matrix_report,
)

SECTIONS = (
    "baseline",
    "update1",
    "update2",
    "reload",
    "continuous_step3",
    "resumed_step3",
    "checkpoint",
    "fault_injection",
)
FAULTS = ("interrupted_writer", "missing_rank", "hash_error", "unpublished_load")
PHASES = ("update1", "update2", "reload", "continuous", "resumed")
# Each training stage runs the phases up to its last one and leaves the durable
# state (checkpoints, journal) for the next stage to continue from.
STAGE_LAST_PHASE = {"train1": "update1", "train2": "update2", "train": "resumed"}


def stage_phases(stage):
    """Phases a training stage executes, in order."""
    last = STAGE_LAST_PHASE[stage]
    return PHASES[: PHASES.index(last) + 1]


def continued_training_storage(previous):
    """Durable state a continuing stage resumes: the train1 run's, however many stages back.

    train2 continues train1's storage instead of creating its own, so the final
    stage must follow the predecessor chain back to the run that owns the state.
    """
    run = Path(previous)
    for _ in range(len(STAGE_LAST_PHASE)):
        manifest = json.loads((run / "run.json").read_text())
        if manifest.get("stage") == "train1":
            return training_storage_root(manifest, run)
        if manifest.get("stage") not in STAGE_LAST_PHASE or not manifest.get("previous"):
            break
        run = Path(manifest["previous"])
    raise ValueError("Training continuation does not lead back to a train1 run")


def continuation_identity_matches(journal_identity, model_identity_value, fixture_sha256):
    """A later training stage continues a journal only for the same model/env/fixture."""
    expected = {**model_identity_value, "fixture_sha256": fixture_sha256}
    actual = {k: v for k, v in journal_identity.items() if k not in ("run_id", "execution_id")}
    return actual == expected


EVENTS = (
    "l2:baseline",
    "worker:update1",
    "l2:update1",
    "worker:update2",
    "l2:update2",
    "worker:reload",
    "l2:reload",
    "worker:continuous",
    "worker:resumed",
)


def record(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": file_digest(path)}


def read_record(value):
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or file_digest(path) != value["sha256"]
    ):
        raise ValueError(f"Training evidence changed or is missing: {path}")
    return json.loads(path.read_text())


def require_digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("Training state contains an invalid original-byte digest")


def validate_state(state, *, rank, step, identity, parameters):
    expected = {
        "schema_version": 1,
        "scope": "native_vime_training_state",
        "rank": rank,
        "identity": identity,
        "step": step,
        "sample_position": step * 4,
        "consumed_train_samples": step * 4,
        "fixture_sha256": identity["fixture_sha256"],
    }
    hash_fields = {
        "model_sha256",
        "optimizer_parameter_sha256",
        "optimizer_header_sha256",
        "scheduler_sha256",
        "rng_sha256",
        "state_sha256",
    }
    if set(state) != set(expected) | hash_fields or any(
        state.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Native state omits or changes rank/step/sample/fixture identity")
    if (
        state_digest({key: value for key, value in state.items() if key != "state_sha256"})
        != state["state_sha256"]
    ):
        raise ValueError("Native state digest does not bind the recorded recovery state")
    if not isinstance(state["model_sha256"], list) or len(state["model_sha256"]) != 1:
        raise ValueError("Native state requires the complete single PP1 model chunk")
    for value in state["model_sha256"] + [
        state[key] for key in ("optimizer_header_sha256", "scheduler_sha256", "rng_sha256")
    ]:
        require_digest(value)
    moments = state["optimizer_parameter_sha256"]
    if set(moments) != set(parameters):
        raise ValueError("Native optimizer state does not cover every trainable parameter")
    for values in moments.values():
        if not {"param", "exp_avg", "exp_avg_sq"}.issubset(values):
            raise ValueError("Native state omitted FP32 masters or Adam moments")
        for value in values.values():
            require_digest(value)


def validate_journal(journal, identity, *, verify_files=True):
    if journal.get("schema_version") != 1 or journal.get("identity") != identity:
        raise ValueError("Training journal belongs to different source/environment/fixture")
    events = journal.get("events", [])
    if events != list(EVENTS[: len(events)]) or not events:
        raise ValueError("Training journal did not preserve update/L2/reload/control order")
    if set(journal.get("workers", {})) != {
        event.split(":")[1] for event in events if event.startswith("worker:")
    }:
        raise ValueError("Training journal worker records disagree with durable phase order")
    if set(journal.get("l2_reports", {})) != {
        event.split(":")[1] for event in events if event.startswith("l2:")
    }:
        raise ValueError("Training journal L2 records disagree with durable phase order")
    if verify_files:
        for value in journal["workers"].values():
            read_record(value)
        for value in journal["l2_reports"].values():
            validate_matrix_report(read_record(value))


def repeat_reload_in_current_attempt(journal, attempt):
    """Restore durable updates, but never pass with an older attempt's final reload."""
    if "reload" not in journal["workers"]:
        return False
    original = read_record(journal["workers"]["reload"])
    if original.get("attempt") == str(attempt):
        return False
    if not str(original.get("attempt", "")).isdigit():
        raise ValueError("Recovery source has no valid attempt identity")
    cutoff = journal["events"].index("worker:reload")
    retired = {
        "reason": "repeat_final_reload_and_controls_in_current_attempt",
        "from_attempt": original["attempt"],
        "to_attempt": str(attempt),
        "events": journal["events"][cutoff:],
        "workers": {
            phase: journal["workers"].pop(phase)
            for phase in ("reload", "continuous", "resumed")
            if phase in journal["workers"]
        },
        "l2_report": journal["l2_reports"].pop("reload", None),
    }
    journal["events"] = journal["events"][:cutoff]
    journal.setdefault("recovery", []).append(retired)
    return True


def validate_update(audit, *, step, fixture):
    if (audit.get("step"), audit.get("sample_position"), audit.get("fixture")) != (
        step,
        step * 4,
        fixture,
    ):
        raise ValueError("Native policy update progress/fixture mismatch")
    norm = audit.get("grad_norm", float("nan"))
    if audit.get("policy_loss_calls") != 4 or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Native policy update needs four losses and a finite nonzero norm")
    before, after = (
        audit.get("parameter_hashes_before", {}),
        audit.get("parameter_hashes_after", {}),
    )
    if not before or before.keys() != after.keys():
        raise ValueError("Native update must hash every trainable parameter before and after")
    validate_active_parameter_names(before)
    for value in [*before.values(), *after.values()]:
        require_digest(value)
    if audit.get("progress") != {
        "consumed_before": (step - 1) * 4,
        "consumed_after": step * 4,
        "scheduler_before": (step - 1) * 4,
        "scheduler_after": step * 4,
        "sample_counter_owner": "qwen3_next_native_training.policy_step",
    }:
        raise ValueError("Native update did not commit matching sample/scheduler progress")
    changed = [name for name in before if before[name] != after[name]]
    if not changed or sorted(changed) != sorted(audit.get("changed_parameters", [])):
        raise ValueError("Native optimizer did not prove an actual parameter change")
    gradients = audit.get("gradients", {})
    parameters, groups = gradients.get("parameters", {}), gradients.get("active_groups", {})
    if parameters.keys() != before.keys() or groups.keys() != before.keys():
        raise ValueError("Every active parameter tensor must have predeclared gradient evidence")
    for name in before:
        if (
            parameters[name].get("present") is not True
            or parameters[name].get("nonzero") is not True
        ):
            raise ValueError(f"Active parameter tensor has no valid gradient: {name}")
        if parameters[name].get("dtype") not in ("torch.float32", "torch.bfloat16"):
            raise ValueError("Active gradient evidence omitted or changed its storage dtype")
        if groups[name] != [name]:
            raise ValueError("Active gradient group was substituted after execution")
    metrics = audit.get("metrics", {})
    if not metrics or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("Native policy loss must retain finite scalar training metrics")
    return True


def validate_worker(value, *, phase, identity):
    if (
        value.get("schema_version") != 1
        or value.get("status") != "passed"
        or value.get("phase") != phase
        or value.get("identity") != identity
        or value.get("all_rank_processes_exited") is not True
        or value.get("gpu_release_verified") is not True
        or [rank.get("rank") for rank in value.get("ranks", [])] != list(range(4))
    ):
        raise ValueError("Incomplete native four-rank worker evidence")
    pids = [rank.get("pid") for rank in value["ranks"]]
    if any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != 4:
        raise ValueError("Native worker requires four distinct rank processes")
    binary = validate_binary_record(
        value.get("provider_binary"), source_tree=identity["source_tree"]
    )
    for rank, row in enumerate(value["ranks"]):
        if row.get("provider_binary") != binary:
            raise ValueError("Native training rank differs from its archived CUDA provider")
        if row.get("identity") != identity or row.get("phase") != phase:
            raise ValueError("Native rank identity mismatch")
        if row.get("native_entrypoints") != [
            "vime.backends.megatron_utils.model.setup_model_and_optimizer",
            "vime.backends.megatron_utils.model.train_one_step",
            "vime.backends.megatron_utils.loss.policy_loss_function",
        ]:
            raise ValueError("Worker did not identify the required native training entrypoints")
        if phase in ("continuous", "resumed"):
            controls = row.get("controls", {})
            if set(controls) != ({"1", "2", "3"} if phase == "continuous" else {"3"}):
                raise ValueError("Native recovery controls omitted or added an update")
            updates = [(int(step), control) for step, control in controls.items()]
        elif phase == "reload":
            restored = row.get("restored", {})
            audit = restored.get("rank_evidence", {}).get("update_evidence", {})
            updates = [(2, {"update": audit, "state": row.get("state", {})})]
        else:
            updates = [(1 if phase == "update1" else 2, row)]
        for step, control in updates:
            audit = control["update"]
            validate_update(audit, step=step, fixture=identity["fixture_sha256"])
            validate_state(
                control["state"],
                rank=rank,
                step=step,
                identity=identity,
                parameters=audit["parameter_hashes_after"],
            )


def validate_export_evidence(worker, identity):
    from qwen3_next_training_export import validate_training_export_record

    evidence = worker["export_evidence"]
    complete = read_record(evidence["complete"])
    checkpoint_identity = validate_training_export_record(
        complete, worker["export"]["checkpoint_identity"]
    )
    if checkpoint_identity != worker["export"]["checkpoint_identity"]:
        raise ValueError("Archived export proof differs from the weights actually tested by L2")
    if complete["training_identity"] != identity:
        raise ValueError("Archived export belongs to another native training identity")
    read_record(evidence["index"])
    # The original completion record binds the exact index bytes, including HF
    # tensor-to-shard mapping. Historical proof never authorizes loading a file.
    if complete["files"]["model.safetensors.index.json"]["sha256"] != evidence["index"]["sha256"]:
        raise ValueError("Archived export index differs from its completion manifest")
    audits = [
        (
            row["restored"]["rank_evidence"]["update_evidence"]
            if worker["phase"] == "reload"
            else row["update"]
        )
        for row in worker["ranks"]
    ]
    if complete.get("model_identity_method") != "native_parameter_hashes_after" or complete[
        "model_rank_sha256"
    ] != [state_digest(audit["parameter_hashes_after"]) for audit in audits]:
        raise ValueError("HF export bytes do not belong to the actual optimizer-updated actor")


def archive_export(worker, directory, identity):
    from qwen3_next_training_export import validate_training_export

    export = worker["export"]
    root = Path(export["checkpoint"])
    validate_training_export(root, export["checkpoint_identity"])
    evidence = {}
    for kind, name in (
        ("complete", "EXPORT_COMPLETE.json"),
        ("index", "model.safetensors.index.json"),
    ):
        destination = Path(directory) / name
        # Atomic JSON preserves the parsed content, while recording the index's
        # original bytes is required for the completion manifest hash binding.
        with destination.open("xb") as output, (root / name).open("rb") as source:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        evidence[kind] = record(destination)
    worker["export_evidence"] = evidence
    validate_export_evidence(worker, identity)


def compare_reload_matrices(updated, reloaded):
    """Reopen every original phase artifact; a saved passed boolean is insufficient."""
    ids = [case.case_id for case in case_matrix()]
    if any(
        matrix.get("case_ids") != ids or set(matrix.get("cases", {})) != set(ids)
        for matrix in (updated, reloaded)
    ):
        raise ValueError("Reload comparison requires both complete sixteen-case matrices")
    result = {"status": "passed", "raw_bit_comparison": True, "case_ids": ids, "cases": {}}
    for case in ids:
        result["cases"][case] = {}
        for role in ("rollout", "recompute"):
            left = load_tensor_artifact(
                checked_artifact_path(updated["cases"][case]["artifacts"][role])
            )
            right = load_tensor_artifact(
                checked_artifact_path(reloaded["cases"][case]["artifacts"][role])
            )
            compared = compare_artifacts(left, right)
            if not compared["passed"]:
                raise ValueError(
                    f"Reload changed original {role} tensor bits/state at {case}: {compared}"
                )
            result["cases"][case][role] = compared["comparisons"]
    return result


def validate_training_report(report, *, verify_files=True):
    """No partial matrix, optimizer header or stale worker can close training."""
    if (
        report.get("schema_version") != 1
        or report.get("status") != "passed"
        or report.get("scope") != "qwen3_next_training_and_reload"
        or report.get("required_sections") != list(SECTIONS)
    ):
        raise ValueError("Incomplete/unsupported native training acceptance")
    if not verify_files:
        raise ValueError("Training acceptance requires verification of original evidence files")
    if report.get("events") != list(EVENTS):
        raise ValueError("Training acceptance omitted or reordered a required phase")
    l2 = report.get("l2_reports", {})
    workers = report.get("workers", {})
    if set(l2) != {"baseline", "update1", "update2", "reload"} or set(workers) != {
        "update1",
        "update2",
        "reload",
        "continuous",
        "resumed",
    }:
        raise ValueError("Training omitted an update, full L2 matrix or recovery control")
    matrices = {}
    versions = {
        "baseline": "baseline",
        "update1": "update-1",
        "update2": "update-2",
        "reload": "reload-2",
    }
    for phase, item in l2.items():
        matrices[phase] = matrix = read_record(item)
        validate_matrix_report(matrix)
        identity = matrix["identity"]
        if identity.get("weight_version") != versions[phase]:
            raise ValueError("Training phase used the wrong model weight version")
        immutable = set(report["identity"]) - {
            "checkpoint",
            "checkpoint_identity",
            "weight_version",
            "base_checkpoint",
            "base_checkpoint_identity",
        }
        if any(identity.get(key) != report["identity"][key] for key in immutable):
            raise ValueError("Training phase changed source, environment, provider or topology")
        if phase == "baseline":
            if identity != report["identity"]:
                raise ValueError("Training baseline must retain the exact original model identity")
        elif (identity.get("base_checkpoint"), identity.get("base_checkpoint_identity")) != (
            report["identity"]["checkpoint"],
            report["identity"]["checkpoint_identity"],
        ):
            raise ValueError("Updated actor export changed its official base checkpoint identity")
        if matrix["fixture"]["content_sha256"] != report["l2_fixture_sha256"]:
            raise ValueError("Training L2 changed its frozen input tokens")
    identity = report["training_identity"]
    if {key: identity.get(key) for key in report["identity"]} != report["identity"]:
        raise ValueError("Training checkpoint is detached from the baseline model identity")
    if report.get("reload_matches_update2") != compare_reload_matrices(
        matrices["update2"], matrices["reload"]
    ):
        raise ValueError("Reload report does not match the reopened original phase artifacts")
    phases = {}
    for phase, item in workers.items():
        value = phases[phase] = read_record(item)
        validate_worker(value, phase=phase, identity=identity)
        if phase in ("update1", "update2", "reload"):
            validate_export_evidence(value, identity)
            export = value["export"]
            if any(rank["export"] != export for rank in value["ranks"]):
                raise ValueError("Native ranks disagree on exported weights")
            for key in ("checkpoint", "checkpoint_identity", "weight_version"):
                if export[key] != matrices[phase]["identity"][key]:
                    raise ValueError("L2 did not test the actual updated/reloaded actor export")
        if phase in ("update1", "update2"):
            step = 1 if phase == "update1" else 2
            for rank in value["ranks"]:
                validate_update(rank["update"], step=step, fixture=identity["fixture_sha256"])
    for rank in range(4):
        if (
            phases["update2"]["ranks"][rank]["update"]["parameter_hashes_before"]
            != phases["update1"]["ranks"][rank]["update"]["parameter_hashes_after"]
        ):
            raise ValueError("Step2 did not start from the actual completed step1 parameters")
        if phases["update2"]["ranks"][rank]["state"] != phases["reload"]["ranks"][rank]["state"]:
            raise ValueError("Step2 model/optimizer/scheduler/RNG/progress changed on restart")
        continuous = phases["continuous"]["ranks"][rank]["controls"]
        resumed = phases["resumed"]["ranks"][rank]["controls"]
        if set(continuous) != {"1", "2", "3"} or set(resumed) != {"3"}:
            raise ValueError("Continuous/restarted controls omitted a required update")
        for step in (1, 2, 3):
            validate_update(
                continuous[str(step)]["update"], step=step, fixture=identity["fixture_sha256"]
            )
            if (
                step < 3
                and continuous[str(step)]["state"]
                != phases[f"update{step}"]["ranks"][rank]["state"]
            ):
                raise ValueError("Continuous reference changed an earlier update")
        validate_update(resumed["3"]["update"], step=3, fixture=identity["fixture_sha256"])
        if continuous["3"]["state"] != resumed["3"]["state"]:
            raise ValueError("Continuous and restarted step3 states are not exactly equal")
    checkpoint = report["checkpoint"]
    complete = validate_checkpoint(checkpoint["path"], identity, verify_files=False)
    if (
        complete["step"] != 2
        or file_digest(Path(checkpoint["path"]) / "COMPLETE.json") != checkpoint["sha256"]
    ):
        raise ValueError("Training acceptance requires the verified complete step2 checkpoint")
    for rank in range(4):
        progress = json.loads(
            (Path(checkpoint["path"]) / f"rank-{rank}" / "progress.json").read_text()
        )
        state = phases["update2"]["ranks"][rank]["state"]
        for key in state.keys() - {"scope", "state_sha256"}:
            if progress.get(key) != state[key]:
                raise ValueError(
                    "Published checkpoint bytes contradict the step2 recovery identity"
                )
        if progress.get("update_evidence") != phases["update2"]["ranks"][rank]["update"]:
            raise ValueError("Published checkpoint detached the committed optimizer update")
    validate_fault_evidence(read_record(report["fault_injection"]), identity)
    return json_digest(report)


def validate_fault_evidence(faults, identity):
    from qwen3_next_checkpoint import _validate_ranks

    if faults.get("identity") != identity or set(faults.get("cases", {})) != set(FAULTS):
        raise ValueError("Checkpoint publication fault matrix is incomplete")
    if any(value.get("passed") is not True for value in faults["cases"].values()):
        raise ValueError("An interrupted/corrupted checkpoint was not rejected")
    interrupted = faults["cases"]["interrupted_writer"]
    if interrupted.get("exit_code") != -signal.SIGTERM:
        raise ValueError("Interrupted writer did not record its actual termination")
    checked_artifact_path(interrupted["writer_file"])
    for case in FAULTS:
        path = Path(faults["cases"][case]["checkpoint"])
        if path.is_symlink() or path.suffix != ".partial" or not path.is_dir():
            raise ValueError("Fault evidence did not preserve its original unpublished directory")
        try:
            if case in ("missing_rank", "hash_error"):
                _validate_ranks(path, world_size=4, identity=identity)
            else:
                validate_checkpoint(path, identity)
        except (OSError, ValueError) as exc:
            if case == "missing_rank" and not isinstance(exc, FileNotFoundError):
                raise ValueError("Missing-rank fixture failed for an unrelated reason") from exc
            if case == "hash_error" and not re.search("checksum mismatch|size mismatch", str(exc)):
                raise ValueError("Corruption fixture failed for an unrelated reason") from exc
        else:
            raise ValueError(f"Fault fixture no longer reproduces its rejection: {case}")


def checkpoint_faults(output, identity):
    """Kill a real writer and corrupt protocol files; never touch model checkpoints."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    partial = output / "interrupted.partial"
    partial.mkdir()
    code = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "p.write_bytes(b'incomplete checkpoint'); time.sleep(300)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(partial / "writing")], start_new_session=True
    )
    deadline = time.monotonic() + 10
    try:
        while not (partial / "writing").exists():
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Fault injection writer did not start")
            time.sleep(0.01)
        os.killpg(child.pid, signal.SIGTERM)
        code = child.wait(timeout=10)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
    recovery = newest_complete_checkpoint(output, identity)
    cases = {
        "interrupted_writer": {
            "passed": code != 0 and recovery["checkpoint"] is None,
            "exit_code": code,
            "checkpoint": str(partial),
            "writer_file": record(partial / "writing"),
        }
    }
    try:
        validate_checkpoint(partial, identity)
    except (OSError, ValueError) as exc:
        cases["unpublished_load"] = {"passed": True, "error": str(exc), "checkpoint": str(partial)}
    partial = output / "missing.partial"
    partial.mkdir()
    for rank in range(4):
        files = {}
        for role in ("model", "optimizer", "rng", "progress"):
            path = partial / f"rank-{rank}-{role}.bin"
            path.write_bytes(f"{rank}:{role}".encode())
            files[role] = [path.name]
        write_rank_manifest(
            partial,
            rank=rank,
            world_size=4,
            identity=identity,
            step=2,
            sample_position=8,
            files=files,
        )
    hidden = partial / "rank-3.manifest.json"
    corrupted = output / "corrupted.partial"
    shutil.copytree(partial, corrupted)
    hidden.unlink()
    try:
        publish_checkpoint(partial, output / "missing", world_size=4, identity=identity)
    except (OSError, ValueError) as exc:
        cases["missing_rank"] = {"passed": True, "error": str(exc), "checkpoint": str(partial)}
    (corrupted / "rank-2-optimizer.bin").write_bytes(b"corrupt optimizer bytes")
    try:
        publish_checkpoint(corrupted, output / "corrupted", world_size=4, identity=identity)
    except (OSError, ValueError) as exc:
        cases["hash_error"] = {"passed": True, "error": str(exc), "checkpoint": str(corrupted)}
    result = {
        "identity": identity,
        "scope": "actual_checkpoint_publication_protocol",
        "cases": cases,
    }
    if set(cases) != set(FAULTS) or not all(value["passed"] for value in cases.values()):
        raise ValueError("Checkpoint fault injection accepted incomplete state")
    validate_fault_evidence(result, identity)
    atomic_json(output / "faults.json", result)
    return record(output / "faults.json")


def _owned_remove(path, root, identity=None, *, proof=None):
    path, root = Path(path), Path(root)
    if path.is_symlink() or root.is_symlink() or not path.is_dir():
        raise ValueError("Cleanup target is not a directory owned by this training run")
    path, root = path.resolve(), root.resolve()
    if path.parent != root:
        raise ValueError("Cleanup target is not a direct child of this training run")
    if identity is None or proof is None:
        raise ValueError("Cleanup requires explicit ownership and retirement proof")
    return retire_directory(path, root, identity, proof=proof)


def remove_interrupted(root, identity):
    return recover_storage(root, identity)


def retire_export(worker, root, identity):
    validate_export_evidence(worker, identity)
    path = Path(worker["export"]["checkpoint"])
    if not path.exists():
        return  # A prior attempt may have retired it after durable L2 publication.
    complete = path / "EXPORT_COMPLETE.json"
    if (
        complete.is_symlink()
        or json.loads(complete.read_text()) != worker["export"]["checkpoint_identity"]
    ):
        raise ValueError("Cleanup export is not the artifact whose complete L2 was retained")
    _owned_remove(
        path,
        root,
        identity,
        proof={
            "export": worker["export"]["checkpoint_identity"],
            "evidence": worker["export_evidence"],
        },
    )


def remove_uncommitted_exports(root, identity, journal):
    """A crash after export publication cannot retain an unbounded extra copy."""
    from qwen3_next_training_export import validate_training_export_record

    keep = {
        str(Path(read_record(item)["export"]["checkpoint"]).resolve())
        for phase, item in journal["workers"].items()
        if phase in ("update1", "update2", "reload") and phase not in journal["l2_reports"]
    }
    removed = []
    for path in Path(root).iterdir():
        if (
            not path.is_dir()
            or path.suffix == ".partial"
            or path.name.startswith(".")
            or str(path.resolve()) in keep
        ):
            continue
        marker = path / "EXPORT_COMPLETE.json"
        if path.is_symlink() or marker.is_symlink() or not marker.is_file():
            raise ValueError("Completed export cleanup has no safe ownership proof")
        complete = json.loads(marker.read_text())
        validate_training_export_record(complete, complete)
        if complete["training_identity"] != identity:
            raise ValueError("Completed export belongs to another training run")
        removed.append(
            {"path": str(path.absolute()), "complete": complete, "action": "retire_unneeded_export"}
        )
        _owned_remove(path, root, identity, proof={"export": complete})
    return removed


def execute(manifest_path, output):
    from qwen3_next_runtime import require_predecessor

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    run_root = manifest_path.parent
    stage = manifest["stage"]
    if stage not in STAGE_LAST_PHASE:
        raise ValueError("Native training runner requires a training stage")
    previous = Path(manifest["previous"])
    require_predecessor(stage, previous, Path(manifest["checkpoint"]), manifest)
    prior_result = json.loads((previous / "result.json").read_text())
    continuation = stage != "train1"
    if continuation:
        # Continue the predecessor training run's durable state under its own identity.
        if manifest.get("training_storage"):
            raise ValueError("A continuing training stage uses its predecessor's storage")
        durable = continued_training_storage(previous)
        journal_path = durable / "journal.json"
        journal = json.loads(journal_path.read_text())
        baseline_path = Path(journal["l2_reports"]["baseline"]["path"])
    else:
        baseline_path = previous / prior_result["evidence"]["l2_matrix"]["path"]
    baseline = json.loads(baseline_path.read_text())
    validate_matrix_report(baseline, expected_identity=model_identity(manifest))
    fixture_path = Path(baseline["fixture"]["path"])
    if file_digest(fixture_path) != baseline["fixture"]["sha256"]:
        raise ValueError("Baseline L2 fixture was changed")
    fixture = policy_fixture(fixture_path)
    if continuation:
        identity = journal["identity"]
        if not continuation_identity_matches(
            identity, model_identity(manifest), fixture.content_sha256
        ):
            raise ValueError("Training continuation identity differs from the journal")
        if journal["l2_reports"]["baseline"] != record(baseline_path):
            raise ValueError("Training continuation baseline L2 changed")
    else:
        identity = {
            **model_identity(manifest),
            "run_id": manifest["run_id"],
            "execution_id": manifest["execution_id"],
            "fixture_sha256": fixture.content_sha256,
        }
        durable = training_storage_root(manifest, run_root)
    if not durable.is_absolute():
        raise ValueError("Training storage must be an absolute path")
    durable.mkdir(parents=True, exist_ok=True)
    checkpoint_root, export_root = durable / "checkpoints", durable / "exports"
    owned_root(checkpoint_root, identity)
    owned_root(export_root, identity)
    journal_path = durable / "journal.json"
    journal = {
        "schema_version": 1,
        "identity": identity,
        "workers": {},
        "l2_reports": {"baseline": record(baseline_path)},
        "events": ["l2:baseline"],
        "recovery": [],
    }
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
    validate_journal(journal, identity)
    for phase, value in journal["workers"].items():
        worker = read_record(value)
        validate_worker(worker, phase=phase, identity=identity)
        if phase in ("update1", "update2", "reload"):
            validate_export_evidence(worker, identity)
    if repeat_reload_in_current_attempt(journal, os.environ["QWEN3_NEXT_ATTEMPT"]):
        validate_journal(journal, identity, verify_files=False)
        atomic_json(journal_path, journal)
    verify_environment(manifest["environment"], require_venv=True)
    check_l2_ready()
    run_process(
        [manifest["python"], str(ROOT / "setup.py"), "build_ext", "--inplace"],
        log_path=output / "build.log",
        timeout=1800,
    )
    provider_binary = archive_provider_binary(ROOT, output, source_tree=manifest["source_tree"])
    probe = output / "gpu-probe.json"
    run_process(
        [
            manifest["python"],
            str(ROOT / "tools/validation/models/qwen3_next/qwen3_next_l2.py"),
            "--gpu-probe",
            "--output",
            str(probe),
        ],
        log_path=output / "gpu-probe.log",
        timeout=180,
    )
    uuids = json.loads(probe.read_text())["uuids"]
    wait_gpu_release(uuids)
    # A preempted attempt may have written a whole checkpoint without publishing it:
    # publish it (full validation) before interrupted state is retired.
    journal["recovery"].extend(resume_interrupted_publications(checkpoint_root, identity))
    for root in (checkpoint_root, export_root):
        journal["recovery"].extend(remove_interrupted(root, identity))
    journal["recovery"].extend(remove_uncommitted_exports(export_root, identity, journal))
    atomic_json(journal_path, journal)
    export_bytes = sum(
        value["size"] for value in manifest["checkpoint_identity"]["shards"].values()
    )
    for phase in stage_phases(stage):
        if phase not in journal["workers"]:
            recovery = newest_complete_checkpoint(checkpoint_root, identity, verify_files=False)
            checkpoint = None if phase == "continuous" else recovery["checkpoint"]
            directory = output / f"{phase}-{uuid4().hex}"
            directory.mkdir()
            request = {
                "manifest": manifest,
                "phase": phase,
                "identity": identity,
                "fixture": str(fixture_path),
                "checkpoint": checkpoint,
                "checkpoint_root": str(checkpoint_root),
                "export_root": str(export_root),
                "export_bytes": export_bytes,
                "provider_binary": provider_binary,
                "expected_states": {
                    str(step): [
                        rank["state"]
                        for rank in read_record(journal["workers"][f"update{step}"])["ranks"]
                    ]
                    for step in (1, 2)
                    if f"update{step}" in journal["workers"]
                },
            }
            request_path, worker_path = directory / "request.json", directory / "worker.json"
            atomic_json(request_path, request)
            process = run_process(
                [
                    manifest["python"],
                    str(Path(__file__)),
                    "--worker-request",
                    str(request_path),
                    "--output",
                    str(worker_path),
                ],
                log_path=directory / "worker.log",
                timeout=8 * 3600,
            )
            wait_gpu_release(uuids)
            worker = json.loads(worker_path.read_text())
            worker.update(
                {
                    "gpu_release_verified": True,
                    "process": process,
                    "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
                }
            )
            validate_worker(worker, phase=phase, identity=identity)
            if phase in ("update1", "update2", "reload"):
                archive_export(worker, directory, identity)
            atomic_json(worker_path, worker)
            journal["workers"][phase] = record(worker_path)
            journal["events"].append(f"worker:{phase}")
            validate_journal(journal, identity, verify_files=False)
            atomic_json(journal_path, journal)
        worker = read_record(journal["workers"][phase])
        if phase in ("update1", "update2", "reload") and phase not in journal["l2_reports"]:
            directory = output / f"{phase}-l2-{uuid4().hex}"
            directory.mkdir()
            phase_manifest = {
                **manifest,
                **worker["export"],
                "memory": "512G",
                "input_fixture": str(fixture_path),
                "input_fixture_sha256": file_digest(fixture_path),
                "prebuilt_provider_binary": provider_binary,
            }
            phase_manifest_path = directory / "manifest.json"
            atomic_json(phase_manifest_path, phase_manifest)
            matrix = run_matrix(phase_manifest_path, directory, fixture_path)
            validate_matrix_report(matrix, expected_identity=model_identity(phase_manifest))
            journal["l2_reports"][phase] = record(directory / "l2-matrix.json")
            journal["events"].append(f"l2:{phase}")
            validate_journal(journal, identity, verify_files=False)
            atomic_json(journal_path, journal)
        if phase == "reload":
            journal["reload_matches_update2"] = compare_reload_matrices(
                read_record(journal["l2_reports"]["update2"]),
                read_record(journal["l2_reports"]["reload"]),
            )
            atomic_json(journal_path, journal)
        # Native workers retire the previous full checkpoint immediately after
        # verified successor publication; export retirement waits for durable L2.
        if phase in ("update1", "update2", "reload"):
            retire_export(worker, export_root, identity)
    if stage != "train":
        _close_partial_training_stage(
            manifest, output, run_root, previous, prior_result, journal, identity, baseline, uuids
        )
        return
    last = read_record(journal["workers"]["reload"])["checkpoint"]
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "qwen3_next_training_and_reload",
        "identity": model_identity(manifest),
        "training_identity": identity,
        "required_sections": list(SECTIONS),
        "l2_fixture_sha256": baseline["fixture"]["content_sha256"],
        "l2_reports": journal["l2_reports"],
        "workers": journal["workers"],
        "events": journal["events"],
        "recovery": journal["recovery"],
        "reload_matches_update2": journal["reload_matches_update2"],
        "checkpoint": {"path": last, "sha256": file_digest(Path(last) / "COMPLETE.json")},
        "fault_injection": checkpoint_faults(output / "checkpoint-faults", identity),
        "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
    }
    validate_training_report(report)
    atomic_json(output / "training-report.json", report)
    atomic_json(output / "reload.json", report)
    shutil.copyfile(journal["l2_reports"]["reload"]["path"], output / "l2-matrix.json")
    shutil.copyfile(
        previous / prior_result["evidence"]["operators"]["path"], output / "operators.json"
    )
    atomic_json(
        output / "environment.json",
        {"environment": manifest["environment"], "gpu_uuids": uuids, "execution_gpus": 4},
    )
    evidence = {
        name: evidence_record(output / filename, run_root)
        for name, filename in {
            "updates": "training-report.json",
            "reload": "reload.json",
            "l2_matrix": "l2-matrix.json",
            "operators": "operators.json",
            "environment": "environment.json",
        }.items()
    }
    result = result_envelope(manifest, os.environ["QWEN3_NEXT_ATTEMPT"], evidence=evidence)
    result.update(full_model_l2="passed", training_and_reload="passed")
    validate_result(manifest, result, run_dir=run_root, attempt=os.environ["QWEN3_NEXT_ATTEMPT"])
    atomic_json(output / "result.json", result)


def _close_partial_training_stage(
    manifest, output, run_root, previous, prior_result, journal, identity, baseline, uuids
):
    """Record a train1/train2 result: its phases, L2 matrix and the checkpoint left behind."""
    stage = manifest["stage"]
    last_phase = STAGE_LAST_PHASE[stage]
    checkpoint = read_record(journal["workers"][last_phase])["checkpoint"]
    phases = list(stage_phases(stage))
    if any(
        phase not in journal["workers"] or phase not in journal["l2_reports"] for phase in phases
    ):
        raise ValueError("Partial training stage closed before its phases completed")
    progress = {
        "schema_version": 1,
        "status": "passed",
        "scope": SCOPES[stage],
        "identity": model_identity(manifest),
        "training_identity": identity,
        "phases": phases,
        "l2_fixture_sha256": baseline["fixture"]["content_sha256"],
        "l2_reports": journal["l2_reports"],
        "workers": journal["workers"],
        "events": journal["events"],
        "recovery": journal["recovery"],
        "checkpoint": {
            "path": checkpoint,
            "sha256": file_digest(Path(checkpoint) / "COMPLETE.json"),
        },
        "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
    }
    atomic_json(output / "training-progress.json", progress)
    shutil.copyfile(journal["l2_reports"][last_phase]["path"], output / "l2-matrix.json")
    shutil.copyfile(
        previous / prior_result["evidence"]["operators"]["path"], output / "operators.json"
    )
    atomic_json(
        output / "environment.json",
        {"environment": manifest["environment"], "gpu_uuids": uuids, "execution_gpus": 4},
    )
    evidence = {
        name: evidence_record(output / filename, run_root)
        for name, filename in {
            "updates": "training-progress.json",
            "l2_matrix": "l2-matrix.json",
            "operators": "operators.json",
            "environment": "environment.json",
        }.items()
    }
    result = result_envelope(manifest, os.environ["QWEN3_NEXT_ATTEMPT"], evidence=evidence)
    result.update(full_model_l2="passed", training_phases=phases)
    validate_result(manifest, result, run_dir=run_root, attempt=os.environ["QWEN3_NEXT_ATTEMPT"])
    atomic_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--check-ready", action="store_true")
    args = parser.parse_args()
    if args.check_ready:
        check_l2_ready()
        from qwen3_next_training_checkpoint import save_training_checkpoint  # noqa: F401
        from qwen3_next_training_export import export_training_weights  # noqa: F401

        print(json.dumps({"status": "ready", "native_training": True}))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.worker_request:
        request = json.loads(args.worker_request.read_text())
        verify_environment(request["manifest"]["environment"], require_venv=True)
        verify_before_model_load(request["manifest"], args.output.parent)
        run_training_phase(request, args.output)
        return 0
    if args.manifest is None:
        parser.error("--manifest is required")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        execute(args.manifest, args.output)
    except Exception as exc:
        atomic_json(
            args.output / "result.json",
            {
                "run_id": json.loads(args.manifest.read_text())["run_id"],
                "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
                "status": "failed",
                "training_and_reload": "not_passed",
                **failure_fields(exc),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
