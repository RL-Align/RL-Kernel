# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate ROCm Vime matrices with independently selected operator routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .validate_artifacts import (
    CASE_IMPLEMENTATIONS,
    FRAMEWORK_TARGETS,
    _case_id,
    _validate_production_record,
    _validate_rlkernel_record,
    _validate_strict_dense_record,
)

MODULES = ("attention", "ffn", "logp")
RL_KERNEL_MISMATCH_SIDECAR_MARKER = "rlkernel mismatch sidecar active: logp_case="


def validate_module_readbacks(
    readbacks: Sequence[Mapping[str, Any]],
    cases: Mapping[str, str],
    *,
    log_text: str = "",
) -> dict[str, Any]:
    """Validate a ROCm module matrix with independently selected operator routes."""

    normalized_cases = {module: _case_id(str(cases[module])) for module in MODULES}
    errors: list[str] = []
    frameworks: dict[str, Any] = {}

    for framework, target in FRAMEWORK_TARGETS:
        label = f"{framework}/{target}"
        matching = [
            value
            for value in readbacks
            if value.get("framework") == framework and value.get("target") == target
        ]
        if not matching:
            errors.append(f"missing {label} readback")
            continue

        for value in matching:
            plan = value.get("plan")
            plan_cases = plan.get("cases") if isinstance(plan, Mapping) else None
            if not isinstance(plan_cases, Mapping):
                errors.append(f"{label}: readback does not contain an integration plan")
            else:
                for module, case_id in normalized_cases.items():
                    item = plan_cases.get(module)
                    if not isinstance(item, Mapping) or item.get("case_id") != case_id:
                        errors.append(f"{label}: readback {module} plan is not {case_id}")
            if value.get("fallbacks"):
                errors.append(f"{label} recorded fallback: {value['fallbacks']}")

        module_summary: dict[str, Any] = {}
        for module, case_id in normalized_cases.items():
            module_label = f"{label} {module}"
            expected = CASE_IMPLEMENTATIONS[case_id][target]
            records = [
                value["operators"][module]
                for value in matching
                if isinstance(value.get("operators"), Mapping)
                and isinstance(value["operators"].get(module), Mapping)
            ]
            hook_count = sum(
                isinstance(value.get("installed_hooks"), Mapping)
                and bool(value["installed_hooks"].get(module))
                for value in matching
            )
            call_count = sum(int(record.get("call_count", 0)) for record in records)
            native_megatron_logp = (
                framework == "megatron"
                and target == "training"
                and module == "logp"
                and expected == "production"
            )
            if native_megatron_logp:
                marker_present = f"{RL_KERNEL_MISMATCH_SIDECAR_MARKER}{case_id}" in log_text
                if hook_count:
                    errors.append(f"{module_label} production route installed an RL-Kernel hook")
                if records:
                    errors.append(f"{module_label} production route entered provider readback")
                if not marker_present:
                    errors.append(
                        f"{module_label} production route lacks mismatch-sidecar evidence"
                    )
                module_summary[module] = {
                    "case_id": case_id,
                    "expected_implementation": expected,
                    "installed_processes": hook_count,
                    "call_count": call_count,
                    "implementations": [],
                    "backend_ids": [],
                    "native_marker_present": marker_present,
                }
                continue

            if hook_count == 0:
                errors.append(f"{module_label} hook was not installed")
            if not records:
                errors.append(f"missing {module_label} execution record")
            if call_count <= 0:
                errors.append(f"{module_label} had zero calls")

            implementations: set[str] = set()
            backend_ids: set[str] = set()
            for record in records:
                implementations.add(str(record.get("implementation", "")))
                backend_ids.add(str(record.get("backend_id", "")))
                if record.get("case_id") != case_id:
                    errors.append(f"{module_label} record has the wrong case_id")
                if record.get("implementation") != expected:
                    errors.append(
                        f"{module_label} implementation={record.get('implementation')!r}, "
                        f"expected {expected!r}"
                    )
                    continue
                execution_mode = record.get("execution_mode", "eager")
                if framework == "vllm":
                    if execution_mode not in {"eager", "compiled_hip_graph"}:
                        errors.append(f"{module_label} has invalid HIP execution mode")
                elif execution_mode != "eager":
                    errors.append(f"{module_label} did not execute in eager mode")
                if expected == "production":
                    _validate_production_record(record, label=module_label, errors=errors)
                elif module == "attention":
                    _validate_rlkernel_record(
                        record,
                        label=module_label,
                        framework=framework,
                        errors=errors,
                    )
                else:
                    _validate_strict_dense_record(
                        record,
                        module=module,
                        framework=framework,
                        label=module_label,
                        errors=errors,
                    )

            module_summary[module] = {
                "case_id": case_id,
                "expected_implementation": expected,
                "installed_processes": hook_count,
                "call_count": call_count,
                "implementations": sorted(implementations),
                "backend_ids": sorted(backend_ids),
            }
        frameworks[label] = {
            "readback_count": len(matching),
            "modules": module_summary,
        }

    return {
        "passed": not errors,
        "errors": errors,
        "frameworks": frameworks,
    }
