# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate and optionally seal one append-only 200-rollout experiment arm."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from rl_engine.integrations.runtime import _contains_triton, _runtime_platform

RECORD_RE = re.compile(r"\b(rollout|step|perf)\s+(\d+):\s+(\{.*\})\s*$")
FRAMEWORKS = (("megatron", "training"), ("vllm", "rollout"))
MODULES = ("attention", "ffn", "logp")
EXPECTED_FIXED_TOPOLOGY = {
    "gpus": 8,
    "actor_gpus": 8,
    "rollout_gpus": 8,
    "tp": 4,
    "cp": 2,
    "pp": 1,
    "colocate": True,
    "offload_train": False,
    "offload_rollout": True,
}
CASE_FIELDS = {
    "attention": "attention_case",
    "ffn": "ffn_case",
    "logp": "logp_case",
}
RL_KERNEL_LINEAR_LOGP_PROVIDER = "rl_engine.integrations.vime.linear_logp_provider.provider"
VIME_NATIVE_LINEAR_LOGP_MARKER = (
    "linear_logp native active: "
    "backend_id=vime.utils.ppo_utils.calculate_log_probs_and_entropy "
    "contract_id=vime.native.linear_logp.v1 route=unconfigured device=cuda"
)
RL_KERNEL_MISMATCH_SIDECAR_MARKER = "rlkernel mismatch sidecar active: logp_case="
CUDA_GRAPH_LAUNCHER_MARKERS = (
    "required vLLM full-decode CUDA Graph capture sizes",
    "strict vLLM full-decode CUDA Graph capture sizes",
)


def _validate_topology(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["manifest does not contain the required TP4/CP2 colocated topology"]
    errors = [
        f"manifest topology {key}={value.get(key)!r}, expected {expected!r}"
        for key, expected in EXPECTED_FIXED_TOPOLOGY.items()
        if value.get(key) != expected
    ]
    rollout_gpus = value.get("rollout_gpus")
    rollout_gpus_per_engine = value.get("rollout_gpus_per_engine")
    rollout_cp = value.get("rollout_cp", 1)
    rollout_tp = value.get("rollout_tp", rollout_gpus_per_engine)
    rollout_engines = value.get("rollout_engines")
    if (
        not isinstance(rollout_gpus, int)
        or not isinstance(rollout_tp, int)
        or not isinstance(rollout_cp, int)
        or not isinstance(rollout_gpus_per_engine, int)
        or rollout_tp <= 0
        or rollout_cp <= 0
        or rollout_gpus_per_engine != rollout_tp * rollout_cp
        or rollout_gpus % rollout_gpus_per_engine
    ):
        errors.append(
            "manifest rollout_gpus_per_engine must equal rollout_tp * rollout_cp "
            "and divide rollout_gpus"
        )
    elif rollout_engines != rollout_gpus // rollout_gpus_per_engine:
        errors.append(
            "manifest rollout_engines does not match " "rollout_gpus // rollout_gpus_per_engine"
        )
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _compare_mismatch_sidecars(
    directory: Path,
    *,
    require_exact: bool,
    tensor_parallel_size: int,
    context_parallel_size: int,
) -> dict[str, Any]:
    import sys

    examples_root = Path(__file__).resolve().parents[1]
    if str(examples_root) not in sys.path:
        sys.path.insert(0, str(examples_root))
    from vime_rocm_attention_ablation.validate_artifacts import compare_train_rollout_logps

    return compare_train_rollout_logps(
        directory,
        require_exact=require_exact,
        tensor_parallel_size=tensor_parallel_size,
        context_parallel_size=context_parallel_size,
    )


def _parse_runtime_records(log_text: str) -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {
        "rollout": {},
        "step": {},
        "perf": {},
    }
    for line in log_text.splitlines():
        match = RECORD_RE.search(line)
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group(3))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            records[match.group(1)][int(match.group(2))] = value
    return records


def _validate_cudagraph(log_text: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    execution = manifest.get("vllm_execution", {})
    capture_sizes = execution.get("capture_sizes", []) if isinstance(execution, Mapping) else []
    compact_sizes = "[" + ",".join(str(value) for value in capture_sizes) + "]"
    checks = {
        "launcher_marker": any(
            f"{marker}: {compact_sizes}" in log_text for marker in CUDA_GRAPH_LAUNCHER_MARKERS
        ),
        "engine_mode": bool(re.search(r"cudagraph_mode.*FULL_DECODE_ONLY", log_text)),
        "not_eager": "enforce_eager=False" in log_text,
        "capture_sizes": (
            f"'cudagraph_capture_sizes': {capture_sizes}" in log_text
            or f'"cudagraph_capture_sizes":{compact_sizes}' in log_text
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_capture_sizes": capture_sizes,
    }


def _side(case_id: str, target: str) -> str:
    training, rollout = case_id.split("/", 1)
    selected = training if target == "training" else rollout
    return "rl_kernel" if selected == "R" else "production"


def _load_readbacks(directory: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(directory.glob("*.json")):
        value = _load_json(path)
        value["_path"] = str(path)
        values.append(value)
    if not values:
        raise ValueError(f"no framework readbacks found in {directory}")
    return values


def _reported_backend_ids(value: Any) -> set[str]:
    backend_ids: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"actual_backend", "backend_id"} and isinstance(nested, str):
                backend_ids.add(nested)
            backend_ids.update(_reported_backend_ids(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            backend_ids.update(_reported_backend_ids(nested))
    return backend_ids


def _validate_readbacks(
    readbacks: list[dict[str, Any]], arm: Mapping[str, Any], log_text: str
) -> dict[str, Any]:
    errors: list[str] = []
    frameworks: dict[str, Any] = {}
    for framework, target in FRAMEWORKS:
        matching = [
            value
            for value in readbacks
            if value.get("framework") == framework and value.get("target") == target
        ]
        label = f"{framework}/{target}"
        if not matching:
            errors.append(f"missing {label} readback")
            continue
        for value in matching:
            if value.get("fallbacks"):
                errors.append(f"{label} recorded fallback in {value['_path']}")

        module_summary: dict[str, Any] = {}
        for module in MODULES:
            case_id = str(arm[CASE_FIELDS[module]])
            expected = _side(case_id, target)
            records = [
                value["operators"][module]
                for value in matching
                if isinstance(value.get("operators"), Mapping) and module in value["operators"]
            ]
            installed_count = sum(module in value.get("installed_hooks", {}) for value in matching)
            call_count = sum(int(record.get("call_count", 0)) for record in records)
            implementations = sorted({str(record.get("implementation", "")) for record in records})
            backend_ids = sorted({str(record.get("backend_id", "")) for record in records})
            case_ids = sorted({str(record.get("case_id", "")) for record in records})
            native_megatron_logp = (
                framework == "megatron"
                and target == "training"
                and module == "logp"
                and expected == "production"
            )
            if native_megatron_logp:
                marker_present = (
                    VIME_NATIVE_LINEAR_LOGP_MARKER in log_text
                    or f"{RL_KERNEL_MISMATCH_SIDECAR_MARKER}{case_id}" in log_text
                )
                if installed_count:
                    errors.append(
                        f"{label} production logp unexpectedly installed an RL-Kernel hook"
                    )
                if records:
                    errors.append(f"{label} production logp unexpectedly entered provider readback")
                if not marker_present:
                    errors.append(
                        f"{label} production logp did not report Vime's native backend marker"
                    )
                module_summary[module] = {
                    "case_id": case_id,
                    "expected_implementation": expected,
                    "installed_processes": installed_count,
                    "call_count": call_count,
                    "implementations": implementations,
                    "backend_ids": backend_ids,
                    "native_marker_present": marker_present,
                    "native_backend_id": ("vime.utils.ppo_utils.calculate_log_probs_and_entropy"),
                }
                continue
            if installed_count == 0:
                errors.append(f"{label} {module} hook was not installed")
            if call_count == 0:
                errors.append(f"{label} {module} had zero calls")
            if any(value != expected for value in implementations):
                errors.append(
                    f"{label} {module} implementation {implementations!r} != {expected!r}"
                )
            if any(value != case_id for value in case_ids):
                errors.append(f"{label} {module} case IDs {case_ids!r} != {case_id!r}")
            for record in records:
                if _contains_triton(record):
                    errors.append(f"{label} {module} used Triton")
                if _runtime_platform(record.get("provenance")) != "cuda":
                    errors.append(f"{label} {module} did not report CUDA execution")
                if record.get("provenance", {}).get("fallback") is True:
                    errors.append(f"{label} {module} provenance recorded fallback")
                if expected == "rl_kernel" and not str(record.get("backend_id", "")).startswith(
                    "rlkernel."
                ):
                    errors.append(f"{label} {module} did not use an RL-Kernel backend")
                provenance = record.get("provenance", {})
                reported_backend_ids = _reported_backend_ids(record)
                strict_execution = isinstance(provenance, Mapping) and (
                    provenance.get("deterministic_linear_logp") is True
                    or (
                        isinstance(provenance.get("execution"), Mapping)
                        and provenance["execution"].get("strict_backend") is True
                    )
                )
                if expected == "production" and (
                    any(value.startswith("rlkernel.") for value in reported_backend_ids)
                    or strict_execution
                ):
                    errors.append(
                        f"{label} {module} production route executed an RL-Kernel backend: "
                        f"{sorted(reported_backend_ids)!r}"
                    )
            module_summary[module] = {
                "case_id": case_id,
                "expected_implementation": expected,
                "installed_processes": installed_count,
                "call_count": call_count,
                "implementations": implementations,
                "backend_ids": backend_ids,
            }
        frameworks[label] = {"readback_count": len(matching), "modules": module_summary}
    return {"passed": not errors, "errors": errors, "frameworks": frameworks}


def _numeric(value: Any, label: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{label} is missing or non-numeric")
        return None
    result = float(value)
    if not math.isfinite(result):
        errors.append(f"{label} is non-finite")
        return None
    return result


def _validate_runtime_logprobs(
    records: Mapping[int, Mapping[str, Any]],
    expected_rounds: int,
    global_batch_size: int,
    require_zero: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for step in sorted(records):
        record = records[step]
        mismatch = _numeric(
            record.get("train/train_current_rollout_logprob_mismatch_count"),
            f"step {step} mismatch_count",
            errors,
        )
        maximum = _numeric(
            record.get("train/train_current_rollout_logprob_max_abs_diff"),
            f"step {step} max_abs_diff",
            errors,
        )
        mean_abs = _numeric(
            record.get("train/train_rollout_logprob_abs_diff"),
            f"step {step} mean_abs_diff",
            errors,
        )
        numel = _numeric(
            record.get("train/train_current_rollout_logprob_numel"),
            f"step {step} active_token_count",
            errors,
        )
        if numel is not None and numel <= 0:
            errors.append(f"step {step} has no active tokens")
        total_mismatches = None if mismatch is None else mismatch * global_batch_size
        total_active_tokens = None if numel is None else numel * global_batch_size
        rows.append(
            {
                "step": step,
                "bitwise_mismatch_count": total_mismatches,
                "max_abs_dlogp": maximum,
                "mean_abs_dlogp": mean_abs,
                "active_token_count": total_active_tokens,
                "vime_mean_mismatch_count_per_sample": mismatch,
                "vime_mean_active_tokens_per_sample": numel,
            }
        )
    if len(rows) != expected_rounds:
        errors.append(f"observed {len(rows)} train steps, expected {expected_rounds}")
    bitwise_zero = bool(rows) and all(
        row["bitwise_mismatch_count"] == 0.0 and row["max_abs_dlogp"] == 0.0 for row in rows
    )
    if require_zero and not bitwise_zero:
        errors.append("R/R arm did not achieve bitwise-zero runtime metrics")
    return {
        "passed": not errors,
        "errors": errors,
        "evidence_source": (
            "VIME runtime torch.ne/max metrics; VIME reports sample means, converted "
            "to counts with global_batch_size"
        ),
        "bitwise_zero": bitwise_zero,
        "rows": rows,
        "total_active_token_exposure": sum(row["active_token_count"] or 0.0 for row in rows),
    }


def _inspect_offline_dumps(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("*.pt"))
    comparable = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rollout_data = payload.get("rollout_data", {}) if isinstance(payload, Mapping) else {}
        if isinstance(rollout_data, Mapping) and "log_probs" in rollout_data:
            comparable += 1
    return {
        "status": "available" if paths and comparable == len(paths) else "unavailable",
        "artifact_count": len(paths),
        "comparable_artifact_count": comparable,
        "reason": (
            None
            if paths and comparable == len(paths)
            else (
                "current VIME dump lacks captured training log_probs; "
                "runtime exact metrics are used"
            )
        ),
    }


def validate_run(run_dir: Path) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    arm = manifest.get("arm")
    if not isinstance(arm, Mapping):
        raise ValueError("manifest.arm is missing")
    log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    records = _parse_runtime_records(log_text)
    require_zero = all(str(arm[CASE_FIELDS[module]]) == "R/R" for module in MODULES)
    cudagraph = _validate_cudagraph(log_text, manifest)
    readbacks = _validate_readbacks(_load_readbacks(run_dir / "readbacks"), arm, log_text)
    runtime_logprobs = _validate_runtime_logprobs(
        records["step"],
        int(manifest["num_rollout"]),
        int(manifest["batching"]["global_batch_size"]),
        False,
    )
    logprobs = _compare_mismatch_sidecars(
        run_dir / "mismatch-sidecars",
        require_exact=require_zero,
        tensor_parallel_size=int(manifest["topology"]["tp"]),
        context_parallel_size=int(manifest["topology"]["cp"]),
    )
    global_errors = []
    algorithm = manifest.get("algorithm", {})
    if not isinstance(algorithm, Mapping) or algorithm.get("advantage_estimator") != "grpo":
        global_errors.append("manifest does not explicitly select GRPO")
    train_command = manifest.get("train_command", [])
    expected_algorithm_pair = ["--advantage-estimator", "grpo"]
    if not isinstance(train_command, list) or not any(
        train_command[index : index + 2] == expected_algorithm_pair
        for index in range(max(0, len(train_command) - 1))
    ):
        global_errors.append("train command does not explicitly select GRPO")
    topology = manifest.get("topology")
    global_errors.extend(_validate_topology(topology))
    topology = topology if isinstance(topology, Mapping) else {}
    required_command_pairs = [
        ("--actor-num-gpus-per-node", "8"),
        ("--rollout-num-gpus", "8"),
        ("--tensor-model-parallel-size", "4"),
        ("--context-parallel-size", "2"),
        (
            "--rollout-num-gpus-per-engine",
            str(topology.get("rollout_gpus_per_engine", "")),
        ),
    ]
    if "rollout_cp" in topology:
        required_command_pairs.append(
            (
                "--vllm-prefill-context-parallel-size",
                str(topology["rollout_cp"]),
            )
        )
    if isinstance(train_command, list):
        for flag, value in required_command_pairs:
            if not any(
                train_command[index : index + 2] == [flag, value]
                for index in range(max(0, len(train_command) - 1))
            ):
                global_errors.append(f"train command does not contain {flag} {value}")
        if "--colocate" not in train_command:
            global_errors.append("train command does not enable colocated execution")
        if "--no-offload-train" not in train_command:
            global_errors.append("train command does not keep the TP4 actor resident")
        if "--offload-rollout" not in train_command:
            global_errors.append("train command does not offload rollout during training")
        training_logp_implementation = _side(str(arm["logp_case"]), "training")
        provider_pair = ["--linear-logp-provider", RL_KERNEL_LINEAR_LOGP_PROVIDER]
        strict_mode_pair = ["--linear-logp-provider-mode", "strict"]
        has_provider = any(
            train_command[index : index + 2] == provider_pair
            for index in range(max(0, len(train_command) - 1))
        )
        has_strict_mode = any(
            train_command[index : index + 2] == strict_mode_pair
            for index in range(max(0, len(train_command) - 1))
        )
        if training_logp_implementation == "production":
            if "--linear-logp-provider" in train_command:
                global_errors.append(
                    "production Megatron logp must not configure a linear_logp provider"
                )
            if "--linear-logp-provider-mode" in train_command:
                global_errors.append("production Megatron logp must not configure provider mode")
        elif not has_provider or not has_strict_mode:
            global_errors.append(
                "RL-Kernel Megatron logp must configure the strict RL-Kernel provider"
            )
    expected_recompute = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
    }
    if manifest.get("training_memory") != expected_recompute:
        global_errors.append("manifest does not contain the required recompute configuration")
    if re.search(r"fallback=true", log_text, re.IGNORECASE):
        global_errors.append("run log contains fallback=true")
    if "Traceback (most recent call last)" in log_text:
        global_errors.append("run log contains a traceback")
    report = {
        "schema_version": "rlkernel.vime_qwen3_8b_tp4_cp2_200.validation.v1",
        "run_id": manifest.get("run_id"),
        "group": arm.get("group"),
        "passed": bool(
            cudagraph["passed"] and readbacks["passed"] and logprobs["passed"] and not global_errors
        ),
        "errors": global_errors,
        "cudagraph": cudagraph,
        "runtime_readbacks": readbacks,
        "train_rollout_logprob": logprobs,
        "runtime_scalar_logprob": runtime_logprobs,
        "offline_tensor_comparison": _inspect_offline_dumps(run_dir / "train-data"),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "run-validation.json"
    try:
        report = validate_run(run_dir)
    except Exception as exc:
        report = {
            "schema_version": "rlkernel.vime_qwen3_8b_tp4_cp2_200.validation.v1",
            "passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.seal and report["passed"]:
        (run_dir / "COMPLETE").touch(exist_ok=False)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
