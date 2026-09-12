# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate one ROCm Vime Attention operator arm and the four-arm matrix.

The executable matrix in this directory is the Attention implementation matrix
(``P/P``, ``P/R``, ``R/P``, and ``R/R``).  It is intentionally not the compact
``A0``-``A7`` Attention root-cause manifest: those rows need separate, real
mutation hooks before they can be claimed as executed experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

try:
    from .tis_metrics import SIDECAR_SCHEMA_VERSION
except ImportError:  # Direct ``python examples/.../validate_artifacts.py`` execution.
    from tis_metrics import SIDECAR_SCHEMA_VERSION  # type: ignore[no-redef]

SCHEMA_VERSION = "rlkernel.vime_rocm_attention_operator_arm.v1"
MATRIX_SCHEMA_VERSION = "rlkernel.vime_rocm_attention_operator_matrix.v1"
LAUNCH_SCHEMA_VERSION = "rlkernel.vime_rocm_attention_arm_launch.v1"

CASE_IMPLEMENTATIONS = {
    "P/P": {"training": "production", "rollout": "production"},
    "P/R": {"training": "production", "rollout": "rl_kernel"},
    "R/P": {"training": "rl_kernel", "rollout": "production"},
    "R/R": {"training": "rl_kernel", "rollout": "rl_kernel"},
}
FRAMEWORK_TARGETS = (("megatron", "training"), ("vllm", "rollout"))

STRICT_ROCM_BACKEND_ID = "rlkernel.rocm.attention.aiter_ck_ag_rs.v1"
STRICT_ROCM_CORE_BACKEND_ID = "aiter.rocm.ck_dense_mha"
STRICT_ROCM_CORE_ID = "rlkernel.attention.rocm.aiter_ck_dense_mha.v1"
STRICT_ROCM_SCHEDULE_ID = "single_batch_aiter_ck_dense_mha_no_splitkv"
ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID = "rlkernel.rocm.triton_det_gemm"
ROCM_PAGED_ATTENTION_BACKEND_ID = "aiter_mha_batch_prefill_non_split_ck"
ROCM_TRITON_PAGED_ATTENTION_BACKEND_ID = "rlkernel.rocm.triton_chunked_flash_attention.v1"
ROCM_TRITON_DET_GEMM_BACKEND_ID = "rlkernel.det_gemm.triton_mfma_rocm.v1"
ROCM_DETERMINISTIC_COLLECTIVE_BACKEND_ID = "rocm_ipc_fixed_tree"
ROCM_FFN_BACKEND_ID = "rlkernel.rocm.det_gemm_swiglu"
STRICT_FFN_BACKEND_ID = "rlkernel.ffn.qwen3.deterministic.v1"
STRICT_LINEAR_LOGP_BACKEND_ID = "rlkernel.linear_logp.bitwise.v1"
ROCM_LOGP_KERNEL_BACKEND_ID = "rocm-vocab-parallel-logp-ws2"

_FALLBACK_KEYS = {
    "attention_fallback",
    "fallback",
    "fallback_used",
    "split_kv_fallback",
    "used_fallback",
}
_TRITON_KEYS = {"triton_used", "uses_triton"}
_REFERENCE_KEYS = {"reference_only"}


def _case_id(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in CASE_IMPLEMENTATIONS:
        raise ValueError(f"unknown Attention operator case {value!r}")
    return normalized


def _walk_key_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            yield normalized, item
            yield from _walk_key_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_key_values(item)


def _values_for_keys(value: Any, keys: set[str]) -> list[Any]:
    return [item for key, item in _walk_key_values(value) if key in keys]


def _truthy_flag(value: Any, keys: set[str]) -> bool:
    return any(item not in (False, None, "", 0) for item in _values_for_keys(value, keys))


def _contains_string(value: Any, needle: str) -> bool:
    normalized = needle.lower()
    if isinstance(value, str):
        return normalized in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_string(item, normalized) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_string(item, normalized) for item in value)
    return False


def _runtime_platform(provenance: Any) -> str | None:
    values = _values_for_keys(provenance, {"runtime_platform", "platform"})
    normalized = {
        str(value).strip().lower() for value in values if isinstance(value, str) and value.strip()
    }
    if normalized & {"rocm", "hip"}:
        return "rocm"
    if "cuda" in normalized:
        return "cuda"
    return None


def _has_exact_value(value: Any, keys: set[str], expected: str) -> bool:
    return any(str(item).strip() == expected for item in _values_for_keys(value, keys))


def load_readbacks(directory: Path) -> list[dict[str, Any]]:
    """Load framework readbacks emitted into one arm-local directory."""

    values: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"readback must contain an object: {path}")
        value = dict(value)
        value["_path"] = str(path)
        values.append(value)
    if not values:
        raise ValueError(f"no framework readbacks found in {directory}")
    return values


def validate_launch_manifest(path: Path, case_id: str) -> dict[str, Any]:
    """Validate the arm-local frozen configuration emitted before execution."""

    normalized_case = _case_id(case_id)
    errors: list[str] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "passed": False,
            "errors": [f"cannot load launch manifest {path}: {exc}"],
            "frozen_input_fingerprint": None,
        }
    if not isinstance(value, Mapping):
        return {
            "passed": False,
            "errors": [f"launch manifest must contain an object: {path}"],
            "frozen_input_fingerprint": None,
        }
    if value.get("schema_version") != LAUNCH_SCHEMA_VERSION:
        errors.append(f"launch manifest does not use {LAUNCH_SCHEMA_VERSION}")
    if value.get("case_id") != normalized_case:
        errors.append("launch manifest carries the wrong case_id")
    if value.get("expected_implementations") != CASE_IMPLEMENTATIONS[normalized_case]:
        errors.append("launch manifest carries the wrong implementation mapping")
    fingerprint = value.get("frozen_input_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        errors.append("launch manifest lacks a frozen input fingerprint")

    environment = value.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("launch manifest lacks its public environment")
        environment = {}
    expected_environment = {
        "RL_KERNEL_ATTENTION_CASE": normalized_case,
        "RL_KERNEL_FFN_CASE": "R/R",
        "RL_KERNEL_LOGP_CASE": "R/R",
        "RL_KERNEL_VLLM_REAL_VOCAB_SIZE": "151936",
        "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE": "152064",
        "RL_KERNEL_VLLM_INTEGRATION": "1",
        "VLLM_ATTENTION_BACKEND": "ROCM_AITER_FA",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "0",
        "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE": "32",
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            errors.append(f"launch environment {name} is not frozen to {expected!r}")
    hip_visible = environment.get("HIP_VISIBLE_DEVICES")
    if not isinstance(hip_visible, str) or not hip_visible:
        errors.append("launch environment lacks HIP_VISIBLE_DEVICES")
    if environment.get("CUDA_VISIBLE_DEVICES") != hip_visible:
        errors.append("HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES differ")
    sidecar_directory = environment.get("RL_KERNEL_MISMATCH_SIDECAR_DIR")
    if not isinstance(sidecar_directory, str) or not sidecar_directory:
        errors.append("launch environment lacks RL_KERNEL_MISMATCH_SIDECAR_DIR")
    for name in ("RLK_ABLATION_TP_SIZE", "RLK_ABLATION_CP_SIZE"):
        try:
            size = int(environment.get(name))
        except (TypeError, ValueError):
            size = 0
        if size <= 0:
            errors.append(f"launch environment {name} is not a positive integer")
    return {
        "passed": not errors,
        "errors": errors,
        "frozen_input_fingerprint": fingerprint,
        "environment": dict(environment),
    }


def _readback_plan_error(readback: Mapping[str, Any], case_id: str) -> str | None:
    plan = readback.get("plan")
    cases = plan.get("cases") if isinstance(plan, Mapping) else None
    if not isinstance(cases, Mapping):
        return "readback does not contain an integration plan"
    attention = cases.get("attention")
    ffn = cases.get("ffn")
    logp = cases.get("logp")
    if not isinstance(attention, Mapping) or attention.get("case_id") != case_id:
        return f"readback Attention plan is not {case_id}"
    for module, item in (("ffn", ffn), ("logp", logp)):
        if not isinstance(item, Mapping) or item.get("case_id") != "R/R":
            return f"readback changed frozen {module} case away from R/R"
    return None


def _validate_rlkernel_record(
    record: Mapping[str, Any],
    *,
    label: str,
    framework: str,
    errors: list[str],
) -> None:
    backend_id = str(record.get("backend_id", ""))
    provenance = record.get("provenance")
    if not backend_id.startswith("rlkernel."):
        errors.append(f"{label} selected RL-Kernel but reported backend {backend_id!r}")
    if _runtime_platform(provenance) != "rocm":
        errors.append(f"{label} RL-Kernel route did not prove ROCm execution")
    if _truthy_flag(provenance, _FALLBACK_KEYS) or _truthy_flag(provenance, _REFERENCE_KEYS):
        errors.append(f"{label} RL-Kernel route reported a fallback/reference path")
    fallback_values = _values_for_keys(provenance, {"fallback"})
    reference_values = _values_for_keys(provenance, _REFERENCE_KEYS)
    if not fallback_values or not any(value is False for value in fallback_values):
        errors.append(f"{label} did not explicitly prove fallback=false")
    if not reference_values or not any(value is False for value in reference_values):
        errors.append(f"{label} did not explicitly prove reference_only=false")
    if not _has_exact_value(
        provenance,
        {"actual_backend", "backend_id"},
        STRICT_ROCM_BACKEND_ID,
    ):
        errors.append(f"{label} did not prove strict ROCm backend {STRICT_ROCM_BACKEND_ID!r}")
    if not _has_exact_value(
        provenance,
        {"strict_core_id", "core_id"},
        STRICT_ROCM_CORE_ID,
    ):
        errors.append(f"{label} did not prove strict ROCm core {STRICT_ROCM_CORE_ID!r}")
    if not _has_exact_value(
        provenance,
        {"strict_schedule", "schedule_id"},
        STRICT_ROCM_SCHEDULE_ID,
    ):
        errors.append(f"{label} did not prove strict ROCm schedule {STRICT_ROCM_SCHEDULE_ID!r}")
    production_ready = _values_for_keys(provenance, {"production_ready"})
    if not production_ready or not any(value is True for value in production_ready):
        errors.append(f"{label} strict ROCm provenance is not production-ready")
    native_arithmetic = _values_for_keys(provenance, {"native_attention_arithmetic"})
    if not native_arithmetic or not any(value is True for value in native_arithmetic):
        errors.append(f"{label} did not prove native Attention arithmetic")
    if not _has_exact_value(provenance, {"split_kv"}, "disabled"):
        errors.append(f"{label} did not prove split_kv=disabled")
    core_backends = _values_for_keys(provenance, {"core_actual_backends"})
    if not any(
        isinstance(value, (list, tuple)) and STRICT_ROCM_CORE_BACKEND_ID in value
        for value in core_backends
    ):
        errors.append(
            f"{label} did not prove AITER/CK core backend {STRICT_ROCM_CORE_BACKEND_ID!r}"
        )

    projections = _values_for_keys(provenance, {"deterministic_projection"})
    projection_ok = any(
        isinstance(value, Mapping)
        and value.get("backend_id") == ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID
        and value.get("deterministic") is True
        and value.get("split_k") is False
        and value.get("accumulation_dtype") == "fp32"
        and value.get("reduction_order") == "k_ascending"
        and value.get("triton_used") is True
        and isinstance(value.get("roles"), (list, tuple))
        and set(value["roles"]) == {"qkv", "o_proj"}
        for value in projections
    )
    if not projection_ok:
        errors.append(
            f"{label} did not prove deterministic ROCm QKV/O projection "
            f"{ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID!r}"
        )

    # Triton is permitted for deterministic QKV/O projection. Paged Attention
    # arithmetic and page-table reads remain inside native AITER/CK.
    allowed_triton_backends = {
        ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID,
        ROCM_TRITON_PAGED_ATTENTION_BACKEND_ID,
        ROCM_TRITON_DET_GEMM_BACKEND_ID,
    }
    for key, item in _walk_key_values(provenance):
        if key in _TRITON_KEYS and item is True:
            continue
        if isinstance(item, str) and "triton" in item.lower():
            if item not in allowed_triton_backends:
                errors.append(f"{label} reported an unapproved Triton backend {item!r}")

    if framework == "vllm":
        layouts = _values_for_keys(provenance, {"framework_layout"})
        if "vllm_paged_kv" not in layouts:
            errors.append(f"{label} did not prove the vLLM paged-KV execution boundary")
        if not (
            _has_exact_value(provenance, {"paged_kernel"}, ROCM_PAGED_ATTENTION_BACKEND_ID)
            or _has_exact_value(
                provenance, {"paged_kernel"}, ROCM_TRITON_PAGED_ATTENTION_BACKEND_ID
            )
        ):
            errors.append(
                f"{label} did not prove direct fixed-schedule paged execution "
                "(CK non-Split-K or the Triton chunked contract)"
            )
        if any(_values_for_keys(provenance, {"dense_kv_materialized"})):
            errors.append(f"{label} materialized dense KV during paged decode")
        tp_values = _values_for_keys(provenance, {"tp_world_size"})
        try:
            tp_world_size = max(int(value) for value in tp_values)
        except (TypeError, ValueError):
            tp_world_size = 0
        collective_values = _values_for_keys(provenance, {"deterministic_all_reduce_backend"})
        expected_collective = (
            "none" if tp_world_size == 1 else ROCM_DETERMINISTIC_COLLECTIVE_BACKEND_ID
        )
        if tp_world_size <= 0 or expected_collective not in collective_values:
            errors.append(f"{label} did not prove the deterministic ROCm O-projection collective")
    elif framework == "megatron":
        cp_values = _values_for_keys(provenance, {"cp_world_size"})
        try:
            cp_world_size = max(int(value) for value in cp_values)
        except (TypeError, ValueError):
            cp_world_size = 0
        communication = _values_for_keys(provenance, {"communication_backend"})
        expected_communication = "none" if cp_world_size == 1 else "rccl_ag_rs"
        if cp_world_size <= 0 or expected_communication not in communication:
            errors.append(f"{label} did not prove the expected ROCm CP communication path")
        tp_values = _values_for_keys(provenance, {"tp_world_size"})
        try:
            tp_world_size = max(int(value) for value in tp_values)
        except (TypeError, ValueError):
            tp_world_size = 0
        expected_tp_collective = (
            "none" if tp_world_size == 1 else ROCM_DETERMINISTIC_COLLECTIVE_BACKEND_ID
        )
        qkv_collectives = _values_for_keys(provenance, {"tp_qkv_dgrad_collective"})
        output_collectives = _values_for_keys(provenance, {"tp_output_projection_collective"})
        if (
            tp_world_size <= 0
            or expected_tp_collective not in qkv_collectives
            or expected_tp_collective not in output_collectives
        ):
            errors.append(f"{label} did not prove deterministic ROCm TP projection collectives")


def _validate_production_record(
    record: Mapping[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    backend_id = str(record.get("backend_id", ""))
    provenance = record.get("provenance")
    if backend_id.startswith("rlkernel.") or _contains_string(provenance, "rlkernel."):
        errors.append(f"{label} selected production but executed an RL-Kernel backend")
    if _truthy_flag(provenance, _FALLBACK_KEYS):
        errors.append(f"{label} production route reported fallback")


def _validate_strict_dense_record(
    record: Mapping[str, Any],
    *,
    module: str,
    framework: str,
    label: str,
    errors: list[str],
) -> None:
    provenance = record.get("provenance")
    expected_backend = STRICT_FFN_BACKEND_ID if module == "ffn" else STRICT_LINEAR_LOGP_BACKEND_ID
    if record.get("case_id") != "R/R" or record.get("implementation") != "rl_kernel":
        errors.append(f"{label} did not execute the fixed R/R route")
    if record.get("backend_id") != expected_backend:
        errors.append(f"{label} reported backend {record.get('backend_id')!r}")
    expected_mode = "compiled_hip_graph" if framework == "vllm" and module == "ffn" else "eager"
    if record.get("execution_mode", "eager") != expected_mode:
        errors.append(f"{label} did not execute in {expected_mode} mode")
    if int(record.get("call_count", 0)) <= 0:
        errors.append(f"{label} had zero executed calls")
    if _runtime_platform(provenance) != "rocm":
        errors.append(f"{label} did not prove ROCm execution")
    if _truthy_flag(provenance, _FALLBACK_KEYS):
        errors.append(f"{label} reported a fallback")

    if module == "ffn":
        if not _has_exact_value(provenance, {"actual_backend"}, ROCM_FFN_BACKEND_ID):
            errors.append(f"{label} did not prove the strict ROCm FFN backend")
        if not _has_exact_value(
            provenance,
            {"deterministic_all_reduce_backend"},
            ROCM_DETERMINISTIC_COLLECTIVE_BACKEND_ID,
        ):
            errors.append(f"{label} did not prove the ROCm fixed-tree FFN reduction")
        return

    expected_entrypoint = (
        "rocm_deterministic_linear_logp_tp"
        if framework == "megatron"
        else "rocm_vocab_parallel_logp_from_local_logits_tp"
    )
    if not _has_exact_value(provenance, {"logprob_kernel_backend"}, ROCM_LOGP_KERNEL_BACKEND_ID):
        errors.append(f"{label} did not prove the ROCm WS2 logp kernel")
    if not _has_exact_value(provenance, {"strict_entrypoint"}, expected_entrypoint):
        errors.append(f"{label} did not prove strict entrypoint {expected_entrypoint!r}")
    deterministic_values = _values_for_keys(provenance, {"deterministic_linear_logp"})
    if not deterministic_values or not any(value is True for value in deterministic_values):
        errors.append(f"{label} did not prove deterministic linear logp")


def validate_attention_readbacks(
    readbacks: Sequence[Mapping[str, Any]],
    case_id: str,
) -> dict[str, Any]:
    """Validate exact training/rollout routing and strict ROCm provenance."""

    normalized_case = _case_id(case_id)
    expected = CASE_IMPLEMENTATIONS[normalized_case]
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
            plan_error = _readback_plan_error(value, normalized_case)
            if plan_error:
                errors.append(f"{label}: {plan_error}")
            if value.get("fallbacks"):
                errors.append(f"{label} recorded fallback: {value['fallbacks']}")

        hook_count = sum(
            isinstance(value.get("installed_hooks"), Mapping)
            and bool(value["installed_hooks"].get("attention"))
            for value in matching
        )
        if hook_count == 0:
            errors.append(f"{label} Attention hook was not installed")

        records = [
            value["operators"]["attention"]
            for value in matching
            if isinstance(value.get("operators"), Mapping)
            and isinstance(value["operators"].get("attention"), Mapping)
        ]
        call_count = sum(int(record.get("call_count", 0)) for record in records)
        if call_count <= 0:
            errors.append(f"{label} Attention had zero executed calls")

        expected_implementation = expected[target]
        backend_ids: set[str] = set()
        for record in records:
            record_label = f"{label} Attention"
            backend_ids.add(str(record.get("backend_id", "")))
            if record.get("case_id") != normalized_case:
                errors.append(f"{record_label} record has the wrong case_id")
            execution_mode = record.get("execution_mode", "eager")
            if framework == "vllm":
                if execution_mode not in {"eager", "compiled_hip_graph"}:
                    errors.append(f"{record_label} has invalid HIP execution mode")
            elif execution_mode != "eager":
                errors.append(f"{record_label} did not execute in eager mode")
            if record.get("implementation") != expected_implementation:
                errors.append(
                    f"{record_label} implementation={record.get('implementation')!r}, "
                    f"expected {expected_implementation!r}"
                )
                continue
            if expected_implementation == "rl_kernel":
                _validate_rlkernel_record(
                    record,
                    label=record_label,
                    framework=framework,
                    errors=errors,
                )
            else:
                _validate_production_record(record, label=record_label, errors=errors)

        frameworks[label] = {
            "readback_count": len(matching),
            "installed_processes": hook_count,
            "call_count": call_count,
            "expected_implementation": expected_implementation,
            "backend_ids": sorted(backend_ids),
        }

        for module in ("ffn", "logp"):
            dense_label = f"{label} {module.upper()}"
            dense_records = [
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
            if hook_count == 0:
                errors.append(f"{dense_label} hook was not installed")
            if not dense_records:
                errors.append(f"missing {dense_label} execution record")
            for record in dense_records:
                _validate_strict_dense_record(
                    record,
                    module=module,
                    framework=framework,
                    label=dense_label,
                    errors=errors,
                )
            frameworks[f"{label}/{module}"] = {
                "readback_count": len(dense_records),
                "installed_processes": hook_count,
                "call_count": sum(int(record.get("call_count", 0)) for record in dense_records),
                "expected_implementation": "rl_kernel",
                "backend_ids": sorted(
                    {str(record.get("backend_id", "")) for record in dense_records}
                ),
            }

    return {
        "passed": not errors,
        "errors": errors,
        "frameworks": frameworks,
    }


def _tensor(value: Any, *, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    else:
        try:
            result = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - message is exercised through callers
            raise ValueError(f"{label} is not tensor-like") from exc
    return result.reshape(-1).contiguous()


def _scalar_key(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("sample identity key tensors must be scalar")
        return value.detach().cpu().item()
    return value


def _positive_int(value: Any, *, label: str) -> int:
    value = _scalar_key(value)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def slice_response_mask_for_cp(
    loss_mask: Any,
    *,
    total_length: int,
    response_length: int,
    context_parallel_size: int,
    context_parallel_rank: int,
) -> torch.Tensor:
    """Apply Vime's two-ended CP zigzag response slice to a full loss mask."""

    mask = _tensor(loss_mask, label="loss_masks")
    if response_length < 0 or total_length < response_length:
        raise ValueError(f"invalid total/response lengths: {total_length}/{response_length}")
    if mask.numel() != response_length:
        raise ValueError(
            "full loss mask length does not match response_length: "
            f"{mask.numel()} != {response_length}"
        )
    if context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be positive")
    if not 0 <= context_parallel_rank < context_parallel_size:
        raise ValueError("context_parallel_rank is outside the CP world")
    if context_parallel_size == 1:
        return mask

    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * context_parallel_size - 1) // (2 * context_parallel_size)
    chunks = (
        (
            context_parallel_rank * chunk_size,
            (context_parallel_rank + 1) * chunk_size,
        ),
        (
            (2 * context_parallel_size - context_parallel_rank - 1) * chunk_size,
            (2 * context_parallel_size - context_parallel_rank) * chunk_size,
        ),
    )
    response_chunks: list[torch.Tensor] = []
    for start, end in chunks:
        logit_start = max(start, prompt_length - 1)
        logit_end = min(end, total_length - 1)
        if logit_start >= logit_end:
            continue
        response_start = logit_start - (prompt_length - 1)
        response_end = logit_end - (prompt_length - 1)
        response_chunks.append(mask[response_start:response_end])
    if not response_chunks:
        return mask.new_empty((0,))
    return torch.cat(response_chunks, dim=0)


def _hash_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(f"I{value};".encode())
    elif isinstance(value, float):
        digest.update(f"F{value.hex()};".encode())
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"S{len(encoded)}:".encode())
        digest.update(encoded)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"T{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
    else:
        _hash_value(digest, str(value))


def _value_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def load_rollout_identity(directory: Path) -> dict[str, Any]:
    """Hash the frozen sample/token identity emitted by Vime rollout dumps."""

    errors: list[str] = []
    identities: list[dict[str, Any]] = []
    paths = sorted(directory.glob("*.pt"))
    for path in paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            samples = payload.get("samples") if isinstance(payload, Mapping) else None
            if not isinstance(samples, list):
                raise ValueError("payload does not contain a samples list")
            outer_rollout_id = payload.get("rollout_id")
            for ordinal, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    raise ValueError(f"sample {ordinal} is not a mapping")
                tokens = _tensor(sample.get("tokens"), label="tokens")
                response_length = sample.get("response_length")
                if isinstance(response_length, bool) or not isinstance(response_length, int):
                    raise ValueError(f"sample {ordinal} has an invalid response_length")
                identity = {
                    "dump_file": path.name,
                    "ordinal": ordinal,
                    "outer_rollout_id": outer_rollout_id,
                    "rollout_id": sample.get("rollout_id"),
                    "group_index": sample.get("group_index"),
                    "index": sample.get("index"),
                    "prompt": sample.get("prompt"),
                    "tokens": tokens,
                    "response_length": response_length,
                    "loss_mask": sample.get("loss_mask"),
                }
                identities.append(
                    {
                        "sort_key": (
                            path.name,
                            ordinal,
                        ),
                        "identity": identity,
                    }
                )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    identities.sort(key=lambda item: item["sort_key"])
    digest = hashlib.sha256()
    token_count = 0
    for item in identities:
        identity = item["identity"]
        token_count += int(identity["tokens"].numel())
        _hash_value(digest, identity)
    if not paths:
        errors.append(f"no rollout dumps found in {directory}")
    if not identities:
        errors.append("no rollout sample identity was found")
    return {
        "passed": not errors,
        "errors": errors,
        "fingerprint": digest.hexdigest() if identities else None,
        "sample_count": len(identities),
        "token_count": token_count,
        "artifacts": [str(path) for path in paths],
    }


def _sidecar_samples(
    path: Path,
    *,
    tensor_parallel_size: int,
    context_parallel_size: int,
) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("mismatch sidecar must contain a mapping")
    if payload.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError(f"mismatch sidecar does not use {SIDECAR_SCHEMA_VERSION}")
    if payload.get("tensor_parallel_size") != tensor_parallel_size:
        raise ValueError("mismatch sidecar tensor_parallel_size differs from launch")
    if payload.get("context_parallel_size") != context_parallel_size:
        raise ValueError("mismatch sidecar context_parallel_size differs from launch")

    training_values = payload.get("train_log_probs")
    rollout_values = payload.get("rollout_log_probs")
    loss_masks = payload.get("loss_masks")
    total_lengths = payload.get("total_lengths")
    response_lengths = payload.get("response_lengths")
    required_lists = {
        "train_log_probs": training_values,
        "rollout_log_probs": rollout_values,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }
    if any(not isinstance(value, (list, tuple)) for value in required_lists.values()):
        missing = [
            name for name, value in required_lists.items() if not isinstance(value, (list, tuple))
        ]
        raise ValueError(f"mismatch sidecar lacks list fields: {', '.join(missing)}")
    assert isinstance(training_values, (list, tuple))
    assert isinstance(rollout_values, (list, tuple))
    assert isinstance(loss_masks, (list, tuple))
    assert isinstance(total_lengths, (list, tuple))
    assert isinstance(response_lengths, (list, tuple))
    count = len(training_values)
    if not all(
        len(value) == count
        for value in (rollout_values, loss_masks, total_lengths, response_lengths)
    ):
        raise ValueError("mismatch sidecar sample lists have different lengths")

    rank = _scalar_key(payload.get("rank"))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("mismatch sidecar rank must be a non-negative integer")
    call_index = _scalar_key(payload.get("call_index"))
    if isinstance(call_index, bool) or not isinstance(call_index, int) or call_index < 0:
        raise ValueError("mismatch sidecar call_index must be a non-negative integer")
    # The launcher fixes Megatron's parallel order to its default
    # tp-cp-ep-dp-pp, with EP=PP=1. TP ranks are replicas for these values;
    # CP ranks own distinct zigzag response shards and must remain distinct.
    context_parallel_rank = (rank // tensor_parallel_size) % context_parallel_size
    data_parallel_rank = rank // (tensor_parallel_size * context_parallel_size)

    values: list[dict[str, Any]] = []
    for index in range(count):
        logical_key = (
            data_parallel_rank,
            call_index,
            index,
        )
        key = (*logical_key, context_parallel_rank)
        training = _tensor(training_values[index], label="train_log_probs")
        rollout = _tensor(rollout_values[index], label="rollout_log_probs")
        mask = _tensor(loss_masks[index], label="loss_masks").to(torch.bool)
        total_length = _positive_int(total_lengths[index], label="mismatch sidecar total_lengths")
        response_length = _positive_int(
            response_lengths[index], label="mismatch sidecar response_lengths"
        )
        mask = slice_response_mask_for_cp(
            mask,
            total_length=total_length,
            response_length=response_length,
            context_parallel_size=context_parallel_size,
            context_parallel_rank=context_parallel_rank,
        ).to(torch.bool)
        values.append(
            {
                "key": key,
                "logical_key": logical_key,
                "training": training,
                "rollout": rollout,
                "mask": mask,
                "total_length": total_length,
                "response_length": response_length,
            }
        )
    return values


def compare_train_rollout_logps(
    directory: Path,
    *,
    require_exact: bool,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
) -> dict[str, Any]:
    """Compute selected-token metrics from the custom hook's rank/call sidecars."""

    if tensor_parallel_size <= 0 or context_parallel_size <= 0:
        raise ValueError("tensor/context parallel sizes must be positive")
    errors: list[str] = []
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    paths = sorted(directory.glob("*.pt"))
    for path in paths:
        try:
            for sample in _sidecar_samples(
                path,
                tensor_parallel_size=tensor_parallel_size,
                context_parallel_size=context_parallel_size,
            ):
                key = sample["key"]
                fingerprint = _value_fingerprint(
                    {
                        "training": sample["training"],
                        "rollout": sample["rollout"],
                        "mask": sample["mask"],
                        "total_length": sample["total_length"],
                        "response_length": sample["response_length"],
                    }
                )
                previous = unique.get(key)
                if previous is not None:
                    if previous["fingerprint"] != fingerprint:
                        errors.append(f"replicated model-parallel sample {key!r} is inconsistent")
                    continue
                unique[key] = {**sample, "fingerprint": fingerprint}
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if context_parallel_size > 1:
        shards_by_sample: dict[tuple[Any, ...], set[int]] = {}
        lengths_by_sample: dict[tuple[Any, ...], set[tuple[int, int]]] = {}
        for key, sample in unique.items():
            shards_by_sample.setdefault(sample["logical_key"], set()).add(int(key[-1]))
            lengths_by_sample.setdefault(sample["logical_key"], set()).add(
                (sample["total_length"], sample["response_length"])
            )
        expected_shards = set(range(context_parallel_size))
        for logical_key, actual_shards in shards_by_sample.items():
            if actual_shards != expected_shards:
                errors.append(
                    f"sample {logical_key!r} is missing context-parallel shards: "
                    f"expected {sorted(expected_shards)}, got {sorted(actual_shards)}"
                )
            if len(lengths_by_sample[logical_key]) != 1:
                errors.append(
                    f"sample {logical_key!r} has inconsistent lengths across "
                    "context-parallel shards"
                )

    mismatch_count = 0
    element_count = 0
    sum_abs_diff = 0.0
    sum_mismatch_kl = 0.0
    sum_mismatch_k3_kl = 0.0
    max_abs_diff = 0.0
    for key, sample in unique.items():
        training = sample["training"]
        rollout = sample["rollout"]
        mask = sample["mask"]
        if training.shape != rollout.shape:
            errors.append(
                f"sample {key!r} train/rollout shape mismatch: "
                f"{tuple(training.shape)} != {tuple(rollout.shape)}"
            )
            continue
        if training.dtype != rollout.dtype:
            errors.append(
                f"sample {key!r} train/rollout dtype mismatch: "
                f"{training.dtype} != {rollout.dtype}"
            )
            continue
        if mask.numel() != training.numel():
            errors.append(
                f"sample {key!r} mask/logprob length mismatch: "
                f"{mask.numel()} != {training.numel()}"
            )
            continue
        active_training = training[mask]
        active_rollout = rollout[mask]
        if active_training.numel() == 0:
            continue
        if not bool(torch.isfinite(active_training).all()) or not bool(
            torch.isfinite(active_rollout).all()
        ):
            errors.append(f"sample {key!r} contains non-finite log probabilities")
            continue
        mismatch_count += int(torch.ne(active_training, active_rollout).sum().item())
        delta = active_training.to(torch.float64) - active_rollout.to(torch.float64)
        absolute = delta.abs()
        k3 = torch.exp(delta) - delta - 1.0
        if not bool(torch.isfinite(k3).all()):
            errors.append(f"sample {key!r} produced a non-finite mismatch_k3_kl")
            continue
        element_count += int(delta.numel())
        sum_abs_diff += float(absolute.sum().item())
        sum_mismatch_kl += float((-delta).sum().item())
        sum_mismatch_k3_kl += float(k3.sum().item())
        max_abs_diff = max(max_abs_diff, float(absolute.max().item()))

    if not paths:
        errors.append(f"no mismatch sidecars found in {directory}")
    if element_count == 0:
        errors.append("no active train/rollout logprob elements were found")
    mean_abs_diff = sum_abs_diff / element_count if element_count else None
    mismatch_kl = sum_mismatch_kl / element_count if element_count else None
    mismatch_k3_kl = sum_mismatch_k3_kl / element_count if element_count else None
    exact = element_count > 0 and mismatch_count == 0 and max_abs_diff == 0.0
    if require_exact and not exact:
        errors.append("R/R requires bitwise-equal training and rollout log probabilities")
    return {
        "passed": not errors,
        "errors": errors,
        "require_exact": require_exact,
        "torch_equal": exact,
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff if element_count else None,
        "train_rollout_logprob_abs_diff": mean_abs_diff,
        "mismatch_kl": mismatch_kl,
        "mismatch_k3_kl": mismatch_k3_kl,
        "sample_count": len({sample["logical_key"] for sample in unique.values()}),
        "element_count": element_count,
        "artifacts": [str(path) for path in paths],
    }


def validate_arm(
    arm_dir: Path,
    case_id: str,
    *,
    launcher_returncode: int = 0,
) -> dict[str, Any]:
    """Validate one completed Vime arm without accepting configured-only evidence."""

    normalized_case = _case_id(case_id)
    errors: list[str] = []
    launch = validate_launch_manifest(arm_dir / "launch.json", normalized_case)
    try:
        readbacks = validate_attention_readbacks(
            load_readbacks(arm_dir / "readbacks"),
            normalized_case,
        )
    except Exception as exc:
        readbacks = {"passed": False, "errors": [str(exc)], "frameworks": {}}
    rollout_identity = load_rollout_identity(arm_dir / "dump" / "rollout_data")
    environment = launch.get("environment", {})
    try:
        tensor_parallel_size = int(environment.get("RLK_ABLATION_TP_SIZE", 1))
        context_parallel_size = int(environment.get("RLK_ABLATION_CP_SIZE", 1))
    except (TypeError, ValueError):
        tensor_parallel_size = 0
        context_parallel_size = 0
    try:
        metrics = compare_train_rollout_logps(
            arm_dir / "mismatch_sidecars",
            require_exact=normalized_case == "R/R",
            tensor_parallel_size=tensor_parallel_size,
            context_parallel_size=context_parallel_size,
        )
    except Exception as exc:
        metrics = {
            "passed": False,
            "errors": [f"cannot validate mismatch sidecars: {exc}"],
        }
    if launcher_returncode != 0:
        errors.append(f"Vime launcher exited with status {launcher_returncode}")
    errors.extend(launch["errors"])
    errors.extend(readbacks["errors"])
    errors.extend(rollout_identity["errors"])
    errors.extend(metrics["errors"])
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": normalized_case,
        "passed": not errors,
        "launcher_returncode": launcher_returncode,
        "errors": errors,
        "expected_implementations": CASE_IMPLEMENTATIONS[normalized_case],
        "launch": launch,
        "attention_readbacks": readbacks,
        "rollout_identity": rollout_identity,
        "metrics": metrics,
        "claim_boundary": {
            "matrix_kind": "attention_operator_implementation_cross_config",
            "executed_case": normalized_case,
            "a0_a7_mutation_matrix_executed": False,
        },
    }


def validate_matrix(
    arm_reports: Mapping[str, Mapping[str, Any]],
    *,
    frozen_before: Mapping[str, Any],
    frozen_after: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all four arms, frozen sources, and cross-arm sample identity."""

    errors: list[str] = []
    normalized_reports = {_case_id(case): report for case, report in arm_reports.items()}
    if set(normalized_reports) != set(CASE_IMPLEMENTATIONS):
        missing = sorted(set(CASE_IMPLEMENTATIONS) - set(normalized_reports))
        extra = sorted(set(normalized_reports) - set(CASE_IMPLEMENTATIONS))
        errors.append(
            f"matrix cases differ from the four-arm contract; missing={missing}, extra={extra}"
        )

    before_fingerprint = frozen_before.get("fingerprint")
    after_fingerprint = frozen_after.get("fingerprint")
    frozen_sources_match = (
        isinstance(before_fingerprint, str)
        and before_fingerprint
        and before_fingerprint == after_fingerprint
    )
    if not frozen_sources_match:
        errors.append("frozen model/checkpoint/data/revision fingerprint changed during the matrix")

    rollout_fingerprints: dict[str, str | None] = {}
    metrics: dict[str, Any] = {}
    for case_id, report in sorted(normalized_reports.items()):
        if report.get("case_id") != case_id:
            errors.append(f"{case_id} report carries the wrong case_id")
        if report.get("passed") is not True:
            errors.append(f"{case_id} arm did not pass strict validation")
        launch = report.get("launch")
        arm_fingerprint = (
            launch.get("frozen_input_fingerprint") if isinstance(launch, Mapping) else None
        )
        if arm_fingerprint != before_fingerprint:
            errors.append(f"{case_id} arm was not launched with the sealed frozen inputs")
        identity = report.get("rollout_identity")
        rollout_fingerprints[case_id] = (
            identity.get("fingerprint") if isinstance(identity, Mapping) else None
        )
        arm_metrics = report.get("metrics")
        if isinstance(arm_metrics, Mapping):
            metrics[case_id] = {
                key: arm_metrics.get(key)
                for key in (
                    "mismatch_count",
                    "max_abs_diff",
                    "train_rollout_logprob_abs_diff",
                    "mismatch_kl",
                    "mismatch_k3_kl",
                    "sample_count",
                    "element_count",
                )
            }

    identity_values = list(rollout_fingerprints.values())
    rollout_identity_match = (
        len(identity_values) == len(CASE_IMPLEMENTATIONS)
        and all(isinstance(value, str) and value for value in identity_values)
        and len(set(identity_values)) == 1
    )
    if not rollout_identity_match:
        errors.append("rollout sample/token identity is not frozen across all four arms")

    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "passed": not errors,
        "errors": errors,
        "cases": list(CASE_IMPLEMENTATIONS),
        "frozen_sources": {
            "matched": frozen_sources_match,
            "before": before_fingerprint,
            "after": after_fingerprint,
        },
        "rollout_identity": {
            "matched": rollout_identity_match,
            "fingerprints": rollout_fingerprints,
        },
        "metrics": metrics,
        "claim_boundary": {
            "matrix_kind": "attention_operator_implementation_cross_config",
            "implemented_cases": list(CASE_IMPLEMENTATIONS),
            "a0_a7_mutation_matrix_executed": False,
            "note": (
                "A0-A7 remain a diagnostic taxonomy until each row has a real "
                "runtime mutation and restoration hook"
            ),
        },
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASE_IMPLEMENTATIONS), required=True)
    parser.add_argument("--launcher-returncode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = validate_arm(
        args.arm_dir,
        args.case,
        launcher_returncode=args.launcher_returncode,
    )
    output = args.output or args.arm_dir / "validation.json"
    write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
