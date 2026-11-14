#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent native rollout/recompute processes; no numerical fallback."""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from qwen3_next_contract import (  # noqa: E402
    atomic_json,
    checkpoint_identity,
    evidence_record,
    result_envelope,
    validate_result,
)
from qwen3_next_runtime import (  # noqa: E402
    failure_fields,
    file_digest,
    memory_limits,
    parse_memory_request,
    subprocess_failure,
    verify_checkpoint,
    verify_environment,
)

from rl_engine.validation.models.qwen3_next.qwen3_next_binary import (  # noqa: E402
    archive_provider_binary,
    validate_binary_record,
    verify_imported_provider,
)
from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import (  # noqa: E402
    NATIVE_WORKER_MODULE,
    PROMPT_LENGTHS,
    L2Case,
    case_matrix,
    compare_artifacts,
    compare_batch_prefill,
    first_divergent_layer,
    first_divergent_node,
    json_digest,
    load_tensor_artifact,
    prompt_fixture,
    save_tensor_artifact,
    validate_debug_capture,
    validate_fixture,
    validate_matrix_report,
    validate_model_artifact,
    worker_request,
)


def check_ready():
    try:
        module = importlib.import_module(NATIVE_WORKER_MODULE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"native_workers_not_ready: {exc}") from exc
    required = {
        "rollout": True,
        "recompute": True,
        "debug_layers": True,
        "architecture_layers": 48,
        "tp": 4,
        "batched_cases": True,
    }
    actual = module.capabilities()
    if any(actual.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"native_workers_not_ready: required {required}, received {actual}")
    if any(
        not callable(getattr(module, name, None))
        for name in (
            "run_rollout_case",
            "run_recompute_case",
            "run_rollout_cases",
            "run_recompute_cases",
        )
    ):
        raise RuntimeError("native_workers_not_ready: actual model worker entrypoints missing")
    return module, actual


def model_identity(manifest):
    identity = {
        **(
            {"scheduler_profile_certificate": manifest["scheduler_profile_certificate"]}
            if "scheduler_profile_certificate" in manifest
            else {}
        ),
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "environment_sha256": manifest["environment_sha256"],
        "checkpoint": manifest["checkpoint"],
        "weight_version": manifest.get("weight_version", "baseline"),
        "checkpoint_identity": manifest["checkpoint_identity"],
        "provider_config": manifest["provider_config"],
        "topology": manifest["topology"],
        "architecture_layers": 48,
        "weights_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
    }
    if manifest.get("weight_version", "baseline") != "baseline":
        identity.update(
            base_checkpoint=manifest["base_checkpoint"],
            base_checkpoint_identity=manifest["base_checkpoint_identity"],
        )
    from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import validate_weight_identity

    validate_weight_identity(identity)
    return identity


def verify_before_model_load(manifest, output):
    if manifest.get("weight_version", "baseline") != "baseline":
        verified = verify_updated_weights(manifest)
    else:
        if checkpoint_identity(manifest["checkpoint"]) != manifest["checkpoint_identity"]:
            raise RuntimeError("checkpoint_mismatch: official weight snapshot changed")
        reference = json.loads(
            (ROOT / "rl_engine/validation/models/qwen3_next_norm_manifest.json").read_text()
        )["model_identity"]
        verified = verify_checkpoint(Path(manifest["checkpoint"]), reference)
    atomic_json(output / "weight-verification.json", verified)
    return verified


def verify_updated_weights(manifest):
    from qwen3_next_training_export import validate_training_export

    verified = validate_training_export(manifest["checkpoint"], manifest["checkpoint_identity"])
    expected = {
        "weight_version": manifest["weight_version"],
        "base_checkpoint": manifest["base_checkpoint"],
        "base_checkpoint_identity": manifest["base_checkpoint_identity"],
    }
    if any(verified.get(key) != value for key, value in expected.items()):
        raise RuntimeError("checkpoint_mismatch: updated weights/base/version provenance differs")
    for key in (
        "source_commit",
        "source_tree",
        "environment_sha256",
        "provider_config",
        "topology",
    ):
        if verified["training_identity"].get(key) != manifest[key]:
            raise RuntimeError(f"checkpoint_mismatch: updated weights changed {key}")
    return verified


def run_one(module, role, request, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output.with_suffix(".request.json"), request)
    getattr(module, f"run_{role}_case")(request, output)
    if not output.is_file():
        raise RuntimeError("Native worker did not produce a full-vocabulary tensor artifact")
    artifact = load_tensor_artifact(output)
    validate_model_artifact(artifact, request)
    validate_debug_capture(artifact, request)
    return artifact


def worker_batch(
    manifest,
    output,
    role,
    fixture_path,
    *,
    debug_request=None,
    native_worker=None,
    provider_binary=None,
):
    if manifest.get("weight_version", "baseline") != "baseline" and not fixture_path.is_file():
        raise RuntimeError("Updated weights must reuse the already frozen baseline input fixture")
    module, capabilities = check_ready() if native_worker is None else native_worker
    verify_environment(manifest["environment"], require_venv=True)
    output.mkdir(parents=True, exist_ok=debug_request is not None)
    verify_before_model_load(manifest, output)
    identity = model_identity(manifest)
    actual_binary = None
    if provider_binary is not None:
        actual_binary = verify_imported_provider(
            provider_binary, source_tree=manifest["source_tree"]
        )
    elif manifest["provider_config"].get("profile") == "qwen3_next_tp4_shared_eager_v1":
        raise RuntimeError("Native worker requires the stage's pinned CUDA binary evidence")
    if debug_request is not None:
        request = json.loads(debug_request.read_text())
        if request["identity"] != identity or request["role"] != role:
            raise ValueError("Debug request identity mismatch")
        artifact = run_one(module, role, request, output / "debug.pt")
        if not artifact.get("debug_tensors"):
            raise RuntimeError("First divergence rerun did not save layer/recurrent-state tensors")
        return
    fixture = None
    if fixture_path.is_file():
        fixture = load_tensor_artifact(fixture_path)
        validate_fixture(fixture)
    elif role != "rollout":
        raise RuntimeError("Recompute requires the already frozen baseline token fixture")
    records, pending = {}, []

    def cases():
        nonlocal fixture
        if fixture is None:
            from transformers import AutoTokenizer

            from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import freeze_responses

            tokenizer = AutoTokenizer.from_pretrained(
                manifest["checkpoint"], local_files_only=True, trust_remote_code=False
            )
            prompts = prompt_fixture(tokenizer, manifest["checkpoint_identity"]["metadata"])
            baseline, baseline_records = {}, {}
            for length in PROMPT_LENGTHS:
                case = L2Case(length, 4, "full")
                request = worker_request(case, prompts, role=role, identity=identity, generate=True)
                path = output / "baseline" / f"{case.case_id}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(path.with_suffix(".request.json"), request)
                atomic_json(
                    output / "active-case.json",
                    {
                        "case_id": case.case_id,
                        "phase": "baseline_generation",
                        "request": str(path.with_suffix(".request.json")),
                    },
                )
                yield request, path
                artifact = load_tensor_artifact(path)
                validate_model_artifact(artifact, request)
                if actual_binary is not None and artifact.get("provider_binary") != actual_binary:
                    raise ValueError("Baseline generation used a different CUDA provider")
                baseline[str(length)] = artifact
                baseline_records[str(length)] = {"path": str(path), "sha256": file_digest(path)}
            fixture = freeze_responses(
                prompts,
                baseline,
                baseline_identity=json_digest(identity),
                baseline_records=baseline_records,
            )
            validate_fixture(fixture)
            save_tensor_artifact(fixture_path, fixture)
        tokenizer_identity = manifest.get(
            "base_checkpoint_identity", manifest["checkpoint_identity"]
        )
        if fixture["prompt_identity"]["tokenizer"] != tokenizer_identity["metadata"]:
            raise RuntimeError("Frozen input tokenizer identity mismatch")
        for case in case_matrix():
            request = worker_request(case, fixture, role=role, identity=identity)
            path = output / f"{case.case_id}.pt"
            atomic_json(path.with_suffix(".request.json"), request)
            atomic_json(
                output / "active-case.json",
                {
                    "case_id": case.case_id,
                    "phase": (
                        "preparing_recompute_requests" if role == "recompute" else "frozen_inputs"
                    ),
                    "request": str(path.with_suffix(".request.json")),
                },
            )
            # Recompute materializes this iterator before one four-rank model
            # lifetime. Its artifacts therefore exist only after the batch returns.
            # Rollout's baseline loop above stays lazy to freeze generated tokens.
            pending.append((request, path))
            yield request, path

    getattr(module, f"run_{role}_cases")(cases())
    if [request["case_id"] for request, _ in pending] != [case.case_id for case in case_matrix()]:
        raise RuntimeError("Native batch worker did not consume every required case")
    for request, path in pending:
        artifact = load_tensor_artifact(path)
        validate_model_artifact(artifact, request)
        if actual_binary is not None and artifact.get("provider_binary") != actual_binary:
            raise ValueError("Native rank traces used a different CUDA provider")
        records[request["case_id"]] = {
            "path": str(path),
            "sha256": file_digest(path),
            "request_sha256": request["request_sha256"],
        }
    if manifest.get("weight_version", "baseline") != "baseline":
        verify_updated_weights(manifest)
    elif checkpoint_identity(manifest["checkpoint"]) != manifest["checkpoint_identity"]:
        raise RuntimeError("checkpoint_mismatch: weights changed during model execution")
    if provider_binary is not None:
        verify_imported_provider(provider_binary, source_tree=manifest["source_tree"])
    atomic_json(
        output / "worker-result.json",
        {
            "status": "passed",
            "role": role,
            "pid": os.getpid(),
            "identity": identity,
            "capabilities": capabilities,
            "fixture_sha256": fixture["content_sha256"],
            "cases": records,
            "provider_binary": actual_binary,
        },
    )


def run_process(command, *, log_path, timeout=4 * 3600, cleanup_grace_seconds=10):
    with Path(log_path).open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            from qwen3_next_runtime import wait_owned_child

            # The leader may exit on TERM while workers in its group ignore it.
            # Finish the bounded group cleanup even after the leader has exited.
            wait_owned_child(child, lambda: True, grace_seconds=cleanup_grace_seconds)
            raise
    if code:
        raise subprocess_failure(code, log_path)
    return {"pid": child.pid, "exit_code": code, "log": str(log_path)}


def gpu_probe(output):
    import torch

    if torch.cuda.device_count() != 4 or any(
        "B200" not in torch.cuda.get_device_name(index) for index in range(4)
    ):
        raise RuntimeError("topology_mismatch: four allocated B200 GPUs required")
    atomic_json(
        output, {"uuids": [str(torch.cuda.get_device_properties(index).uuid) for index in range(4)]}
    )


def wait_gpu_release(uuids, *, timeout=30):
    wanted = {uuid.lower().removeprefix("gpu-") for uuid in uuids}
    deadline = time.monotonic() + timeout
    while True:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        remaining = [
            line
            for line in output.splitlines()
            if len(line.split(",")) == 3
            and line.split(",")[1].strip().lower().removeprefix("gpu-") in wanted
        ]
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"gpu_memory_not_released: {remaining}")
        time.sleep(0.25)


def run_matrix(manifest_path, output, fixture_path):
    """Return evidence after two independent processes; never mark training complete."""
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    check_ready()
    verify_environment(manifest["environment"], require_venv=True)
    memory = memory_limits()
    requested = parse_memory_request(manifest["memory"])
    if (
        memory["effective_bytes"] is None
        or memory["effective_bytes"] < requested
        or memory["available_bytes"] is None
        or memory["available_bytes"] < requested
    ):
        raise RuntimeError("insufficient_memory: L2 allocation cannot satisfy its pinned budget")
    prebuilt = manifest.get("prebuilt_provider_binary")
    if prebuilt is None:
        run_process(
            [manifest["python"], str(ROOT / "setup.py"), "build_ext", "--inplace"],
            log_path=output / "build.log",
            timeout=1800,
        )
    else:
        expected = validate_binary_record(prebuilt, source_tree=manifest["source_tree"])
        path = Path(expected["path"])
        if path.parent != ROOT / "rl_engine" or file_digest(path) != expected["sha256"]:
            raise ValueError("Training's prebuilt CUDA provider changed before L2")
    provider_binary = archive_provider_binary(ROOT, output, source_tree=manifest["source_tree"])
    probe_path = output / "gpu-probe.json"
    run_process(
        [manifest["python"], str(Path(__file__)), "--gpu-probe", "--output", str(probe_path)],
        log_path=output / "gpu-probe.log",
        timeout=180,
    )
    uuids = json.loads(probe_path.read_text())["uuids"]
    wait_gpu_release(uuids)
    atomic_json(
        output / "environment.json",
        {
            "environment": manifest["environment"],
            "memory": memory,
            "gpu_uuids": uuids,
            "execution_gpus": 4,
        },
    )
    if manifest.get("input_fixture"):
        if (
            not manifest.get("input_fixture_sha256")
            or file_digest(fixture_path) != manifest["input_fixture_sha256"]
        ):
            raise RuntimeError(
                "input_fixture_mismatch: external baseline fixture must be pinned by file SHA256"
            )
    processes = {}
    for role in ("rollout", "recompute"):
        command = [
            manifest["python"],
            str(Path(__file__)),
            "--worker",
            role,
            "--manifest",
            str(manifest_path),
            "--output",
            str(output / role),
            "--fixture",
            str(fixture_path),
            "--provider-binary",
            json.dumps(provider_binary),
        ]
        processes[role] = run_process(command, log_path=output / f"{role}.log")
        wait_gpu_release(uuids)
        processes[role]["gpu_release_verified"] = True
        worker_result = json.loads((output / role / "worker-result.json").read_text())
        if (
            worker_result.get("pid") != processes[role]["pid"]
            or worker_result.get("status") != "passed"
            or worker_result.get("identity") != model_identity(manifest)
            or sorted(worker_result.get("cases", {}))
            != sorted(case.case_id for case in case_matrix())
        ):
            raise RuntimeError("Independent native worker did not finish the required matrix")
        atomic_json(output / "processes.json", processes)
    fixture = load_tensor_artifact(fixture_path)
    validate_fixture(fixture)
    records, first_failure = {}, None
    baseline_outputs = {}
    for length, record in fixture["baseline_artifacts"].items():
        if file_digest(record["path"]) != record["sha256"]:
            raise RuntimeError("Baseline generation artifact was changed")
        baseline_outputs[length] = load_tensor_artifact(record["path"])
        original_identity = baseline_outputs[length]["metadata"]["identity"]
        if json_digest(original_identity) != fixture["baseline_identity"]:
            raise RuntimeError("Baseline generation provenance was changed")
        baseline_request = worker_request(
            L2Case(int(length), 4, "full"),
            fixture,
            role="rollout",
            identity=original_identity,
            generate=True,
        )
        validate_model_artifact(baseline_outputs[length], baseline_request)
    identity = model_identity(manifest)
    for case in case_matrix():
        artifacts = {}
        paths = {}
        for role in ("rollout", "recompute"):
            path = output / role / f"{case.case_id}.pt"
            artifact = load_tensor_artifact(path)
            validate_model_artifact(
                artifact, worker_request(case, fixture, role=role, identity=identity)
            )
            artifacts[role], paths[role] = artifact, str(path)
        paired = compare_artifacts(artifacts["rollout"], artifacts["recompute"])
        invariance = {}
        for role in ("rollout", "recompute"):
            reference = load_tensor_artifact(
                output / role / f"{L2Case(case.prompt_length, 4, 'full').case_id}.pt"
            )
            invariance[role] = compare_batch_prefill(reference, artifacts[role])
        baseline_checks = {}
        if identity.get("weight_version") == "baseline":
            baseline_checks = {
                role: compare_batch_prefill(baseline_outputs[str(case.prompt_length)], artifact)
                for role, artifact in artifacts.items()
            }
        passed = (
            paired["passed"]
            and all(all(checks.values()) for checks in invariance.values())
            and all(all(checks.values()) for checks in baseline_checks.values())
        )
        records[case.case_id] = {
            **paired,
            "passed": passed,
            "batch_prefill": invariance,
            "baseline_generation": baseline_checks,
            "artifacts": {
                role: {"path": path, "sha256": file_digest(path)} for role, path in paths.items()
            },
        }
        if not passed and first_failure is None:
            if not paired["passed"]:
                first_failure = {
                    "case": case,
                    "paired": paired,
                    "contexts": [("rollout", case, False), ("recompute", case, False)],
                }
            elif baseline_checks and not all(
                all(checks.values()) for checks in baseline_checks.values()
            ):
                failed_role = next(
                    role
                    for role in ("rollout", "recompute")
                    if not all(baseline_checks[role].values())
                )
                reference_case = L2Case(case.prompt_length, 4, "full")
                reference = baseline_outputs[str(case.prompt_length)]
                first_failure = {
                    "case": case,
                    "paired": {
                        **paired,
                        "first_divergent_layer": first_divergent_layer(
                            reference, artifacts[failed_role]
                        ),
                        "first_divergent_node": first_divergent_node(
                            reference, artifacts[failed_role]
                        ),
                    },
                    "contexts": [("rollout", reference_case, True), (failed_role, case, False)],
                }
            else:
                failed_role = next(
                    role for role in ("rollout", "recompute") if not all(invariance[role].values())
                )
                reference_case = L2Case(case.prompt_length, 4, "full")
                reference = load_tensor_artifact(
                    output / failed_role / f"{reference_case.case_id}.pt"
                )
                first_failure = {
                    "case": case,
                    "paired": {
                        **paired,
                        "first_divergent_layer": first_divergent_layer(
                            reference, artifacts[failed_role]
                        ),
                        "first_divergent_node": first_divergent_node(
                            reference, artifacts[failed_role]
                        ),
                    },
                    "contexts": [(failed_role, reference_case, False), (failed_role, case, False)],
                }
    report = {
        "schema_version": 1,
        "status": "passed" if first_failure is None else "failed",
        "scope": "qwen3_next_full_model_l2",
        "identity": identity,
        "fixture": {
            "path": str(fixture_path),
            "sha256": file_digest(fixture_path),
            "content_sha256": fixture["content_sha256"],
        },
        "case_ids": [case.case_id for case in case_matrix()],
        "cases": records,
        "processes": processes,
        "worker_results": {
            role: {
                "path": str(output / role / "worker-result.json"),
                "sha256": file_digest(output / role / "worker-result.json"),
            }
            for role in ("rollout", "recompute")
        },
        "raw_bit_comparison": True,
        "logits_dtype": "bfloat16",
        "logprobs_dtype": "float32",
        "response_length": 16,
        "vocab_size": 151936,
        "provider_binary": provider_binary,
    }
    atomic_json(output / "l2-matrix.json", report)
    if first_failure:
        case = first_failure["case"]
        layer = first_failure["paired"]["first_divergent_layer"]
        capture = {
            "case_id": case.case_id,
            "first_divergent_layer": layer,
            "first_divergent_node": first_failure["paired"].get("first_divergent_node"),
            "first_divergent_component": "lm_head_or_logprob" if layer is None else "decoder_layer",
            "first_difference": first_failure["paired"]["first_difference"],
            "captures": {},
        }
        for label, (role, context_case, generate) in zip(
            ("reference", "candidate"), first_failure["contexts"]
        ):
            directory = output / "first-divergence" / label
            directory.mkdir(parents=True, exist_ok=True)
            request = worker_request(
                context_case,
                fixture,
                role=role,
                identity=identity,
                trace_layers=[47] if layer is None else [layer],
                generate=generate,
            )
            request_path = directory / "request.json"
            atomic_json(request_path, request)
            try:
                command = [
                    manifest["python"],
                    str(Path(__file__)),
                    "--worker",
                    role,
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(directory),
                    "--fixture",
                    str(fixture_path),
                    "--debug-request",
                    str(request_path),
                    "--provider-binary",
                    json.dumps(provider_binary),
                ]
                run_process(command, log_path=directory / "worker.log")
                wait_gpu_release(uuids)
            except (RuntimeError, subprocess.SubprocessError) as exc:
                capture["captures"][label] = {"status": "failed", "error": str(exc)}
                break
            capture["captures"][label] = {
                "status": "saved",
                "role": role,
                "case_id": context_case.case_id,
                "path": str(directory / "debug.pt"),
                "sha256": file_digest(directory / "debug.pt"),
            }
        atomic_json(output / "first-divergence.json", capture)
    if report["status"] == "passed":
        validate_matrix_report(report, expected_identity=identity, artifact_root=output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--worker", choices=("rollout", "recompute"))
    parser.add_argument("--debug-request", type=Path)
    parser.add_argument("--provider-binary", type=json.loads)
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--gpu-probe", action="store_true")
    args = parser.parse_args()
    if args.check_ready:
        try:
            _, capabilities = check_ready()
        except (RuntimeError, AttributeError) as exc:
            print(json.dumps({"status": "not_ready", "error": str(exc)}))
            return 2
        print(json.dumps({"status": "ready", "capabilities": capabilities}))
        return 0
    if args.gpu_probe:
        gpu_probe(args.output)
        return 0
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required")
    manifest = json.loads(args.manifest.read_text())
    fixture_path = args.fixture or Path(
        manifest.get("input_fixture", args.manifest.parent / "inputs/frozen.pt")
    )
    if args.worker:
        worker_batch(
            manifest,
            args.output,
            args.worker,
            fixture_path,
            debug_request=args.debug_request,
            provider_binary=args.provider_binary,
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if manifest["stage"] != "l2":
            raise RuntimeError("Standalone L2 runner cannot close training acceptance")
        from qwen3_next_runtime import require_predecessor

        previous = Path(manifest["previous"])
        require_predecessor("l2", previous, Path(manifest["checkpoint"]), manifest)
        predecessor = json.loads((previous / "run.json").read_text())
        previous_result = json.loads((previous / "result.json").read_text())
        validate_result(predecessor, previous_result, run_dir=previous)
        report = run_matrix(args.manifest, args.output, fixture_path)
        if report["status"] != "passed":
            raise RuntimeError("Full-model raw-bit L2 failed; first divergence artifacts retained")
        prior_operators = previous / previous_result["evidence"]["operators"]["path"]
        shutil.copyfile(prior_operators, args.output / "operators.json")
        atomic_json(
            args.output / "predecessor-evidence.json",
            {
                "run_id": predecessor["run_id"],
                "execution_id": predecessor["execution_id"],
                "result_sha256": file_digest(previous / "result.json"),
            },
        )
        evidence = {
            name: evidence_record(args.output / filename, args.manifest.parent)
            for name, filename in {
                "operators": "operators.json",
                "environment": "environment.json",
                "l2_matrix": "l2-matrix.json",
                "predecessor": "predecessor-evidence.json",
                "provider_binary": "provider-binary.json",
            }.items()
        }
        result = result_envelope(manifest, os.environ["QWEN3_NEXT_ATTEMPT"], evidence=evidence)
        result["full_model_l2"] = "passed"
        validate_result(
            manifest, result, run_dir=args.manifest.parent, attempt=os.environ["QWEN3_NEXT_ATTEMPT"]
        )
        atomic_json(args.output / "result.json", result)
    except Exception as exc:
        atomic_json(
            args.output / "result.json",
            {
                "run_id": manifest["run_id"],
                "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
                "status": "failed",
                "full_model_l2": "not_passed",
                **failure_fields(exc),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
