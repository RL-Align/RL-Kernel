#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4 layer gates and actual native scheduler acceptance, before independent L2."""

import argparse
import importlib
import json
import os
import shutil
import sys
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
from qwen3_next_l2 import (  # noqa: E402
    model_identity,
    run_process,
    verify_before_model_load,
    wait_gpu_release,
    worker_batch,
)
from qwen3_next_layer_contract import (  # noqa: E402
    layer_requirements,
    validate_layer_report,
    validate_rank_result,
)
from qwen3_next_runtime import failure_fields  # noqa: E402
from qwen3_next_runtime import (  # noqa: E402
    file_digest,
    identity_digest,
    memory_limits,
    parse_memory_request,
    verify_environment,
)

from rl_engine.validation.models.qwen3_next.qwen3_next_binary import (  # noqa: E402
    archive_provider_binary,
    validate_binary_record,
)
from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import (  # noqa: E402
    L2Case,
    case_matrix,
    compare_batch_prefill,
    first_divergent_layer,
    load_tensor_artifact,
    validate_fixture,
    validate_scheduler_report,
)

SCHEDULER_MODULE = "rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout"


def check_scheduler_ready():
    try:
        module = importlib.import_module(SCHEDULER_MODULE)
        actual = module.capabilities()
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(f"native_scheduler_not_ready: {exc}") from exc
    required = {
        "rollout": True,
        "debug_layers": True,
        "architecture_layers": 48,
        "tp": 4,
        "batched_cases": True,
        "native_scheduler": True,
    }
    if any(actual.get(key) != value for key, value in required.items()) or not callable(
        getattr(module, "run_rollout_cases", None)
    ):
        raise RuntimeError(f"native_scheduler_not_ready: required {required}, received {actual}")
    return module, actual


def scheduler_report(manifest, output, fixture_path, process, provider_binary):
    fixture = load_tensor_artifact(fixture_path)
    validate_fixture(fixture)
    records = {}
    for case in case_matrix():
        path = output / "rollout" / f"{case.case_id}.pt"
        candidate = load_tensor_artifact(path)
        reference = load_tensor_artifact(
            output / "rollout" / f"{L2Case(case.prompt_length, 4, 'full').case_id}.pt"
        )
        baseline = load_tensor_artifact(
            fixture["baseline_artifacts"][str(case.prompt_length)]["path"]
        )
        batch = compare_batch_prefill(reference, candidate)
        generated = compare_batch_prefill(baseline, candidate)
        records[case.case_id] = {
            "passed": all(batch.values()) and all(generated.values()),
            "batch_prefill": {"rollout": batch},
            "baseline_generation": {"rollout": generated},
            "artifacts": {"rollout": {"path": str(path), "sha256": file_digest(path)}},
        }
        if not records[case.case_id]["passed"] and not (output / "first-divergence.json").exists():
            divergent = reference if not all(batch.values()) else baseline
            atomic_json(
                output / "first-divergence.json",
                {
                    "case_id": case.case_id,
                    "first_divergent_layer": first_divergent_layer(divergent, candidate),
                    "candidate_artifact": str(path),
                    "reference_worker_traces": divergent.get("worker_traces", []),
                    "candidate_worker_traces": candidate.get("worker_traces", []),
                    "capture": "all_48_layers_and_owned_states_in_original_worker_traces",
                },
            )
    worker_path = output / "rollout/worker-result.json"
    report = {
        "schema_version": 1,
        "status": "passed" if all(row["passed"] for row in records.values()) else "failed",
        "scope": "qwen3_next_native_scheduler_only",
        "full_model_l2": "not_evaluated",
        "raw_bit_comparison": True,
        "identity": model_identity(manifest),
        "case_ids": [case.case_id for case in case_matrix()],
        "cases": records,
        "fixture": {
            "path": str(fixture_path),
            "sha256": file_digest(fixture_path),
            "content_sha256": fixture["content_sha256"],
        },
        "processes": {"rollout": process},
        "worker_results": {
            "rollout": {"path": str(worker_path), "sha256": file_digest(worker_path)}
        },
        "provider_binary": provider_binary,
    }
    atomic_json(output / "scheduler-report.json", report)
    validate_scheduler_report(
        report, expected_identity=model_identity(manifest), artifact_root=output
    )
    return report


def run_layers(manifest_path, output):
    from qwen3_next_runtime import require_predecessor

    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "layers":
        raise ValueError("Layer runner can only close the layers stage")
    previous = Path(manifest["previous"])
    require_predecessor("layers", previous, Path(manifest["checkpoint"]), manifest)
    verify_environment(manifest["environment"], require_venv=True)
    requirement = layer_requirements(ROOT)
    if requirement != manifest["layer_requirements"]:
        raise ValueError("Layer registry differs from the submitted source snapshot")
    memory = memory_limits()
    requested = parse_memory_request(manifest["memory"])
    if any(
        value is None or value < requested
        for value in (memory["effective_bytes"], memory["available_bytes"])
    ):
        raise RuntimeError("insufficient_memory: layer allocation cannot satisfy its pinned budget")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "incomplete",
        "phase": "build",
        "scope": "qwen3_next_layer_and_scheduler_gates",
        "full_model_l2": "not_evaluated",
        "identity": model_identity(manifest),
        "requirements_sha256": identity_digest(requirement),
        "gates": {},
    }

    def progress(phase):
        report["phase"] = phase
        atomic_json(output / "layers.json", report)

    run_process(
        [manifest["python"], str(ROOT / "setup.py"), "build_ext", "--inplace", "--force"],
        log_path=output / "build.log",
        timeout=1800,
    )
    provider_binary = archive_provider_binary(ROOT, output, source_tree=manifest["source_tree"])
    report["provider_binary"] = provider_binary
    actual_binary = validate_binary_record(provider_binary, source_tree=manifest["source_tree"])
    progress("checkpoint")
    verify_before_model_load(manifest, output)
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
    atomic_json(
        output / "environment.json",
        {
            "environment": manifest["environment"],
            "memory": memory,
            "gpu_uuids": uuids,
            "execution_gpus": 4,
        },
    )
    for name, expected in requirement["gates"].items():
        progress(name)
        directory = output / name
        process = run_process(
            [
                manifest["python"],
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                str(ROOT / expected["script"]),
                "--checkpoint",
                manifest["checkpoint"],
                expected["output_option"],
                str(directory),
            ],
            log_path=output / f"{name}.log",
        )
        wait_gpu_release(uuids)
        ranks = {}
        for rank in range(4):
            path = directory / f"rank-{rank}.json"
            validate_rank_result(json.loads(path.read_text()), expected, rank)
            if json.loads(path.read_text()).get("provider_binary") != actual_binary:
                raise ValueError("Layer rank used a different CUDA provider than its build")
            ranks[str(rank)] = {"path": str(path), "sha256": file_digest(path)}
        report["gates"][name] = {**process, "gpu_release_verified": True, "ranks": ranks}
        progress(name + "_passed")
    progress("native_scheduler")
    check_scheduler_ready()
    fixture = output / "inputs/frozen.pt"
    process = run_process(
        [
            manifest["python"],
            str(Path(__file__)),
            "--scheduler-worker",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output / "rollout"),
            "--fixture",
            str(fixture),
            "--provider-binary",
            json.dumps(provider_binary),
        ],
        log_path=output / "scheduler.log",
    )
    wait_gpu_release(uuids)
    process["gpu_release_verified"] = True
    scheduler_report(manifest, output, fixture, process, provider_binary)
    report["scheduler"] = {
        "path": str(output / "scheduler-report.json"),
        "sha256": file_digest(output / "scheduler-report.json"),
    }
    if checkpoint_identity(manifest["checkpoint"]) != manifest["checkpoint_identity"]:
        raise RuntimeError("checkpoint_mismatch: official snapshot changed during layer execution")
    report.update(status="passed", phase="complete")
    validate_layer_report(report, requirement, model_identity(manifest), root=output)
    atomic_json(output / "layers.json", report)
    previous_result = json.loads((previous / "result.json").read_text())
    shutil.copyfile(
        previous / previous_result["evidence"]["operators"]["path"], output / "operators.json"
    )
    names = {
        "operators": "operators.json",
        "environment": "environment.json",
        "layers": "layers.json",
        "checkpoint": "weight-verification.json",
        "provider_binary": "provider-binary.json",
    }
    evidence = {
        name: evidence_record(output / filename, manifest_path.parent)
        for name, filename in names.items()
    }
    result = result_envelope(manifest, os.environ["QWEN3_NEXT_ATTEMPT"], evidence=evidence)
    validate_result(manifest, result, run_dir=manifest_path.parent)
    atomic_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--scheduler-worker", action="store_true")
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--provider-binary", type=json.loads)
    args = parser.parse_args()
    if args.check_ready:
        _, capabilities = check_scheduler_ready()
        print(json.dumps({"status": "ready", "capabilities": capabilities}))
        return 0
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required")
    manifest = json.loads(args.manifest.read_text())
    if args.scheduler_worker:
        worker_batch(
            manifest,
            args.output,
            "rollout",
            args.fixture,
            native_worker=check_scheduler_ready(),
            provider_binary=args.provider_binary,
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run_layers(args.manifest, args.output)
    except Exception as exc:
        kind = str(exc).split(":", 1)[0]
        atomic_json(
            args.output / "result.json",
            {
                "run_id": manifest["run_id"],
                "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
                "status": "incomplete" if kind == "native_scheduler_not_ready" else "failed",
                "phase": (
                    "native_scheduler" if kind == "native_scheduler_not_ready" else "layer_gate"
                ),
                **failure_fields(exc),
                "full_model_l2": "not_evaluated",
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
