# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate CUDA-only framework readbacks and Vime train/rollout Logp dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from rl_engine.integrations.runtime import _contains_triton, _runtime_platform

_FRAMEWORKS = (("megatron", "training"), ("vllm", "rollout"))
_MODULES = ("attention", "ffn", "logp")
_STRICT_LOGP_BACKEND = "rlkernel.linear_logp.bitwise.v1"
_BACKEND_PREFIXES = ("rlkernel.", "pytorch-vocab-parallel-logp")


def load_readbacks(directory: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"readback must contain an object: {path}")
        value["_path"] = str(path)
        values.append(value)
    if not values:
        raise ValueError(f"no framework readbacks found in {directory}")
    return values


def validate_readbacks(readbacks: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    frameworks: dict[str, Any] = {}
    for framework, target in _FRAMEWORKS:
        matching = [
            value
            for value in readbacks
            if value.get("framework") == framework and value.get("target") == target
        ]
        label = f"{framework}/{target}"
        if not matching:
            errors.append(f"missing {label} readback")
            continue
        module_summary: dict[str, Any] = {}
        for value in matching:
            if value.get("fallbacks"):
                errors.append(f"{label} recorded fallback: {value['fallbacks']}")
        for module in _MODULES:
            hook_count = sum(module in value.get("installed_hooks", {}) for value in matching)
            records = [
                value["operators"][module]
                for value in matching
                if isinstance(value.get("operators"), Mapping) and module in value["operators"]
            ]
            call_count = sum(int(record.get("call_count", 0)) for record in records)
            if hook_count == 0:
                errors.append(f"{label} {module} hook was not installed")
            if call_count == 0:
                errors.append(f"{label} {module} had zero calls")
            backends = sorted({str(record.get("backend_id", "")) for record in records})
            for record in records:
                backend = str(record.get("backend_id", ""))
                if module == "logp" and backend != _STRICT_LOGP_BACKEND:
                    errors.append(
                        f"{label} logp used {backend!r}, expected {_STRICT_LOGP_BACKEND!r}"
                    )
                elif not backend.startswith(_BACKEND_PREFIXES):
                    errors.append(f"{label} {module} used unexpected backend {backend!r}")
                if _contains_triton(record):
                    errors.append(f"{label} {module} used Triton")
                if _runtime_platform(record.get("provenance")) != "cuda":
                    errors.append(f"{label} {module} did not report CUDA execution")
            module_summary[module] = {
                "installed_processes": hook_count,
                "call_count": call_count,
                "backend_ids": backends,
            }
        frameworks[label] = {
            "readback_count": len(matching),
            "modules": module_summary,
        }
    return {"passed": not errors, "errors": errors, "frameworks": frameworks}


def _load_train_dump(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"train dump must contain a mapping: {path}")
    return value


def compare_train_rollout_logps(paths: list[Path]) -> dict[str, Any]:
    sample_count = 0
    element_count = 0
    mismatch_count = 0
    max_abs_diff = 0.0
    errors: list[str] = []
    for path in paths:
        payload = _load_train_dump(path)
        samples = payload.get("samples")
        if not isinstance(samples, list):
            rollout_data = payload.get("rollout_data")
            if not isinstance(rollout_data, Mapping):
                errors.append(f"{path} has neither samples nor rollout_data")
                continue
            training_values = rollout_data.get("log_probs")
            rollout_values = rollout_data.get("rollout_log_probs")
            if not isinstance(training_values, (list, tuple)) or not isinstance(
                rollout_values, (list, tuple)
            ):
                errors.append(f"{path} rollout_data lacks list log_probs/rollout_log_probs")
                continue
            if len(training_values) != len(rollout_values):
                errors.append(
                    f"{path} logprob list length mismatch: "
                    f"{len(training_values)} != {len(rollout_values)}"
                )
            samples = [
                {"log_probs": training, "rollout_log_probs": rollout}
                for training, rollout in zip(training_values, rollout_values, strict=False)
            ]
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                errors.append(f"{path} sample {sample_index} is not a mapping")
                continue
            training = sample.get("log_probs")
            rollout = sample.get("rollout_log_probs")
            if not isinstance(training, torch.Tensor) or not isinstance(rollout, torch.Tensor):
                errors.append(
                    f"{path} sample {sample_index} lacks tensor log_probs/rollout_log_probs"
                )
                continue
            sample_count += 1
            if training.shape != rollout.shape:
                errors.append(
                    f"{path} sample {sample_index} shape mismatch: "
                    f"{tuple(training.shape)} != {tuple(rollout.shape)}"
                )
                continue
            if training.dtype != rollout.dtype:
                errors.append(
                    f"{path} sample {sample_index} dtype mismatch: "
                    f"{training.dtype} != {rollout.dtype}"
                )
            element_count += training.numel()
            mismatch_count += int(torch.ne(training, rollout).sum().item())
            if training.numel():
                diff = (training.float() - rollout.float()).abs()
                if not bool(torch.isfinite(diff).all().item()):
                    errors.append(f"{path} sample {sample_index} has non-finite drift")
                else:
                    max_abs_diff = max(max_abs_diff, float(diff.max().item()))
    if not paths:
        errors.append("no Vime train dump was found")
    if sample_count == 0:
        errors.append("no comparable train/rollout samples were found")
    torch_equal = not errors and mismatch_count == 0
    return {
        "passed": torch_equal and max_abs_diff == 0.0,
        "torch_equal": torch_equal,
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff if math.isfinite(max_abs_diff) else None,
        "sample_count": sample_count,
        "element_count": element_count,
        "errors": errors,
        "artifacts": [str(path) for path in paths],
    }


def validate_artifacts(readback_dir: Path, train_data_dir: Path) -> dict[str, Any]:
    readbacks = validate_readbacks(load_readbacks(readback_dir))
    train_paths = sorted(train_data_dir.glob("*.pt"))
    bitwise = compare_train_rollout_logps(train_paths)
    return {
        "schema_version": "rlkernel.vime_cuda_bitwise_validation.v1",
        "passed": bool(readbacks["passed"] and bitwise["passed"]),
        "runtime_policy": {"platform": "cuda", "triton_allowed": False},
        "readbacks": readbacks,
        "train_rollout_logp": bitwise,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback-dir", type=Path, required=True)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = validate_artifacts(args.readback_dir, args.train_data_dir)
    except Exception as exc:
        report = {
            "schema_version": "rlkernel.vime_cuda_bitwise_validation.v1",
            "passed": False,
            "runtime_policy": {"platform": "cuda", "triton_allowed": False},
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
