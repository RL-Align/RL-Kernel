# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""End-to-end ROCm rollout/training Attention ablation runner.

PR230 defines a four-cell production/RL-Kernel matrix.  This module executes
that matrix against a real orchestration command (normally Vime), one fresh
process per cell, and validates the runtime readbacks emitted independently by
Megatron training and vLLM rollout workers.  It deliberately does not execute
synthetic tensors or manufacture a checked-in result payload.

The replay identity is frozen before the first case.  Only case selection and
case-local artifact paths may differ between subprocesses.  A successful exit
without both framework readbacks is a failure, as is an R-side readback that
does not prove ROCm execution through the strict AITER/CK runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from rl_engine.integrations.ablation import Implementation, IntegrationPlan
from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
)

ROCM_ABLATION_SCHEMA_VERSION = "rlkernel.rocm.e2e_attention_ablation.v1"
ROCM_ATTENTION_CASE_IDS = ("P/P", "P/R", "R/P", "R/R")
STRICT_ROCM_ATTENTION_RUNTIME = "rlkernel.rocm.attention.aiter_ck_ag_rs.v1"
STRICT_ROCM_ATTENTION_CORE = STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
STRICT_ROCM_ATTENTION_SCHEDULE = STRICT_ATTENTION_ROCM_SCHEDULE_ID

# These values identify the immutable input and sampling stream.  They must be
# present for an actual matrix execution; a dry run may omit them so launchers
# can be reviewed before expensive GPU allocation.
REQUIRED_REPLAY_ENV = (
    "MODEL_ROOT",
    "TORCH_DIST_ROOT",
    "PROMPT_DATA",
)
FROZEN_REPLAY_ENV = (
    *REQUIRED_REPLAY_ENV,
    "VIME_CKPT",
    "TOKENIZER_PATH",
    "DATASET_SEED",
    "PYTHONHASHSEED",
    "ROLLOUT_SEED",
    "TRAIN_SEED",
    "NUM_ROLLOUT",
    "ROLLOUT_BATCH_SIZE",
    "N_SAMPLES_PER_PROMPT",
    "MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH",
)

_FRAMEWORK_TARGETS = MappingProxyType(
    {
        "training": ("megatron", "training"),
        "rollout": ("vllm", "rollout"),
    }
)


@dataclass(frozen=True)
class RocmAttentionAblationCase:
    """One PR230 Attention implementation pairing."""

    case_id: str
    plan: IntegrationPlan

    def __post_init__(self) -> None:
        if self.case_id not in ROCM_ATTENTION_CASE_IDS:
            raise ValueError(f"unknown ROCm Attention ablation case {self.case_id!r}")
        if self.plan.cases["attention"].case_id != self.case_id:
            raise ValueError("case_id must match the Attention case in the integration plan")
        if self.plan.cases["ffn"].case_id != "P/P" or self.plan.cases["logp"].case_id != "P/P":
            raise ValueError("the Attention matrix must hold FFN and Logp at P/P")

    @property
    def slug(self) -> str:
        return self.case_id.lower().replace("/", "-")

    def implementation_for(self, target: str) -> Implementation:
        return self.plan.implementation_for("attention", target)


@dataclass(frozen=True)
class FrameworkRouteEvidence:
    """Aggregated route evidence from one framework side of one case."""

    framework: str
    target: str
    implementation: str
    backend_ids: tuple[str, ...]
    call_count: int
    readback_count: int
    runtime_platforms: tuple[str, ...]
    actual_backends: tuple[str, ...]


@dataclass(frozen=True)
class RocmAblationCaseResult:
    """Outcome of one real Vime subprocess."""

    case_id: str
    status: str
    returncode: int | None
    log_path: Path
    readback_dir: Path
    routes: Mapping[str, FrameworkRouteEvidence] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"not_run", "passed", "failed"}:
            raise ValueError(f"unknown case status {self.status!r}")
        object.__setattr__(self, "routes", MappingProxyType(dict(self.routes)))


def rocm_attention_ablation_matrix(
    case_ids: Iterable[str] = ROCM_ATTENTION_CASE_IDS,
) -> tuple[RocmAttentionAblationCase, ...]:
    """Build the PR230 matrix while freezing non-Attention modules."""

    normalized = tuple(str(case_id).strip().upper() for case_id in case_ids)
    if not normalized:
        raise ValueError("at least one Attention ablation case is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Attention ablation case IDs must be unique")
    unknown = [case_id for case_id in normalized if case_id not in ROCM_ATTENTION_CASE_IDS]
    if unknown:
        raise ValueError(f"unknown Attention ablation cases: {', '.join(unknown)}")
    return tuple(
        RocmAttentionAblationCase(
            case_id=case_id,
            plan=IntegrationPlan.from_case_ids(
                attention=case_id,
                ffn="P/P",
                logp="P/P",
            ),
        )
        for case_id in normalized
    )


def validate_rocm_host() -> dict[str, Any]:
    """Verify that a real ROCm device is available before allocating a run."""

    import torch

    hip = getattr(torch.version, "hip", None)
    if hip is None:
        raise RuntimeError("end-to-end ROCm ablation requires a ROCm PyTorch build")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("end-to-end ROCm ablation requires at least one visible AMD GPU")
    devices = tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
    return {
        "hip_runtime": str(hip),
        "pytorch": str(torch.__version__),
        "device_count": len(devices),
        "devices": devices,
    }


def validate_replay_environment(environment: Mapping[str, str]) -> None:
    """Reject a run that cannot identify its immutable replay inputs."""

    missing = [name for name in REQUIRED_REPLAY_ENV if not str(environment.get(name, "")).strip()]
    if missing:
        raise RuntimeError("missing frozen replay environment: " + ", ".join(missing))
    num_rollout = str(environment.get("NUM_ROLLOUT", "1")).strip()
    try:
        count = int(num_rollout)
    except ValueError as exc:
        raise RuntimeError("NUM_ROLLOUT must be an integer") from exc
    if count != 1:
        raise RuntimeError(
            "the ablation matrix requires NUM_ROLLOUT=1 so every cell starts from "
            "the same pre-update state"
        )


def replay_identity(
    command: Sequence[str],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Return one stable identity shared by every matrix cell."""

    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("command must contain non-empty argument strings")
    frozen = {
        name: str(environment[name])
        for name in FROZEN_REPLAY_ENV
        if str(environment.get(name, "")).strip()
    }
    payload = {
        "command": list(command),
        "environment": frozen,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_case_environment(
    base_environment: Mapping[str, str],
    case: RocmAttentionAblationCase,
    *,
    case_dir: Path,
) -> dict[str, str]:
    """Materialize one worker environment without mutating the parent process."""

    environment = {str(key): str(value) for key, value in base_environment.items()}
    readback_dir = case_dir / "readbacks"
    environment.update(
        {
            "RL_KERNEL_ATTENTION_CASE": case.case_id,
            "RL_KERNEL_FFN_CASE": "P/P",
            "RL_KERNEL_LOGP_CASE": "P/P",
            "RL_KERNEL_VLLM_INTEGRATION": "1",
            "RL_KERNEL_READBACK_DIR": str(readback_dir),
            "RL_KERNEL_ROUTE_REPORT": "1",
            "RL_KERNEL_PLATFORM": "rocm",
            "RL_KERNEL_ROCM_STRICT_ATTENTION": "1",
            "VLLM_ATTENTION_BACKEND": "ROCM_AITER_FA",
            "RL_KERNEL_ABLATION_CASE": case.case_id,
            "RL_KERNEL_ABLATION_OUTPUT_DIR": str(case_dir),
            "NUM_ROLLOUT": "1",
        }
    )
    return environment


def load_runtime_readbacks(directory: Path) -> list[dict[str, Any]]:
    """Load framework-owned evidence and reject malformed artifacts."""

    payloads: list[dict[str, Any]] = []
    if not directory.is_dir():
        return payloads
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid runtime readback {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"runtime readback must contain an object: {path}")
        value["_source_path"] = str(path)
        payloads.append(value)
    return payloads


def _nested_strings(value: Any, key: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        direct = value.get(key)
        if isinstance(direct, str) and direct.strip():
            found.add(direct.strip())
        for item in value.values():
            found.update(_nested_strings(item, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_nested_strings(item, key))
    return found


def _contains_truthy_fallback(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() in {
                "fallback",
                "fallback_used",
                "used_fallback",
                "split_kv_fallback",
            } and item not in (False, None, "", 0):
                return True
            if _contains_truthy_fallback(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_truthy_fallback(item) for item in value)
    return False


def _contains_triton(value: Any) -> bool:
    if isinstance(value, str):
        return "triton" in value.lower()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() == "triton_used" and item is True:
                return True
            if _contains_triton(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_triton(item) for item in value)
    return False


def _attention_plan_case(payload: Mapping[str, Any]) -> str | None:
    plan = payload.get("plan")
    cases = plan.get("cases") if isinstance(plan, Mapping) else None
    attention = cases.get("attention") if isinstance(cases, Mapping) else None
    value = attention.get("case_id") if isinstance(attention, Mapping) else None
    return str(value) if isinstance(value, str) else None


def _validate_route(
    case: RocmAttentionAblationCase,
    payloads: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> tuple[FrameworkRouteEvidence | None, list[str]]:
    framework, target = _FRAMEWORK_TARGETS[side]
    expected = case.implementation_for(target)
    label = f"{framework}/{target}"
    matching = [
        payload
        for payload in payloads
        if payload.get("framework") == framework and payload.get("target") == target
    ]
    if not matching:
        return None, [f"missing {label} runtime readback"]

    errors: list[str] = []
    records: list[Mapping[str, Any]] = []
    for payload in matching:
        if _attention_plan_case(payload) != case.case_id:
            errors.append(f"{label} readback used a different Attention case")
        if payload.get("fallbacks"):
            errors.append(f"{label} recorded fallback: {payload['fallbacks']}")
        operators = payload.get("operators")
        record = operators.get("attention") if isinstance(operators, Mapping) else None
        if isinstance(record, Mapping):
            try:
                call_count = int(record.get("call_count", 0))
            except (TypeError, ValueError):
                errors.append(f"{label} Attention has an invalid call count")
                continue
            if call_count > 0:
                records.append(record)
    if not records:
        errors.append(f"{label} Attention had zero executed calls")
        return None, errors

    implementations = {str(record.get("implementation", "")) for record in records}
    if implementations != {expected.value}:
        errors.append(
            f"{label} implementation mismatch: expected {expected.value}, "
            f"observed {sorted(implementations)}"
        )
    backend_ids = tuple(sorted({str(record.get("backend_id", "")) for record in records}))
    provenance = [record.get("provenance", {}) for record in records]
    runtime_platforms = tuple(
        sorted(set().union(*(_nested_strings(item, "runtime_platform") for item in provenance)))
    )
    actual_backends = tuple(
        sorted(set().union(*(_nested_strings(item, "actual_backend") for item in provenance)))
    )
    strict_core_ids = set().union(*(_nested_strings(item, "strict_core_id") for item in provenance))
    strict_schedules = set().union(
        *(_nested_strings(item, "strict_schedule") for item in provenance)
    )

    if any(_contains_truthy_fallback(item) for item in provenance):
        errors.append(f"{label} Attention provenance contains a fallback")
    if expected is Implementation.PRODUCTION:
        native_backend = f"{framework}.production.attention"
        if any(backend != native_backend for backend in backend_ids):
            errors.append(f"{label} did not execute framework-native Attention: {backend_ids}")
    else:
        if any(not backend.startswith("rlkernel.attention.") for backend in backend_ids):
            errors.append(f"{label} did not execute the RL-Kernel Attention wrapper")
        if runtime_platforms != ("rocm",):
            errors.append(f"{label} did not prove ROCm execution: {runtime_platforms}")
        if STRICT_ROCM_ATTENTION_RUNTIME not in actual_backends:
            errors.append(
                f"{label} did not prove strict AITER/CK runtime "
                f"{STRICT_ROCM_ATTENTION_RUNTIME!r}"
            )
        if STRICT_ROCM_ATTENTION_CORE not in strict_core_ids:
            errors.append(
                f"{label} did not prove strict AITER/CK core " f"{STRICT_ROCM_ATTENTION_CORE!r}"
            )
        if STRICT_ROCM_ATTENTION_SCHEDULE not in strict_schedules:
            errors.append(
                f"{label} did not prove fixed no-Split-KV schedule "
                f"{STRICT_ROCM_ATTENTION_SCHEDULE!r}"
            )
        if any(_contains_triton(item) for item in provenance):
            errors.append(f"{label} strict ROCm Attention used Triton")

    evidence = FrameworkRouteEvidence(
        framework=framework,
        target=target,
        implementation=expected.value,
        backend_ids=backend_ids,
        call_count=sum(int(record.get("call_count", 0)) for record in records),
        readback_count=len(matching),
        runtime_platforms=runtime_platforms,
        actual_backends=actual_backends,
    )
    return evidence, errors


def validate_case_readbacks(
    case: RocmAttentionAblationCase,
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, FrameworkRouteEvidence], tuple[str, ...]]:
    """Require executed and correctly routed training and rollout evidence."""

    routes: dict[str, FrameworkRouteEvidence] = {}
    errors: list[str] = []
    for side in _FRAMEWORK_TARGETS:
        route, route_errors = _validate_route(case, payloads, side=side)
        if route is not None:
            routes[side] = route
        errors.extend(route_errors)
    return routes, tuple(errors)


def run_rocm_attention_ablation(
    command: Sequence[str],
    *,
    output_dir: Path,
    base_environment: Mapping[str, str] | None = None,
    case_ids: Iterable[str] = ROCM_ATTENTION_CASE_IDS,
    execute: bool = False,
) -> tuple[RocmAblationCaseResult, ...]:
    """Execute each case in a fresh process and validate its real readbacks."""

    matrix = rocm_attention_ablation_matrix(case_ids)
    environment = dict(os.environ if base_environment is None else base_environment)
    identity = replay_identity(command, environment)
    if execute:
        validate_replay_environment(environment)
        occupied = [output_dir / case.slug for case in matrix if (output_dir / case.slug).exists()]
        if occupied:
            joined = ", ".join(str(path) for path in occupied)
            raise FileExistsError(
                "refusing to mix ROCm ablation evidence with existing case directories: " + joined
            )
        validate_rocm_host()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RocmAblationCaseResult] = []
    for case in matrix:
        case_dir = output_dir / case.slug
        readback_dir = case_dir / "readbacks"
        log_path = case_dir / "run.log"
        case_environment = build_case_environment(environment, case, case_dir=case_dir)
        if not execute:
            results.append(
                RocmAblationCaseResult(
                    case_id=case.case_id,
                    status="not_run",
                    returncode=None,
                    log_path=log_path,
                    readback_dir=readback_dir,
                )
            )
            continue

        case_dir.mkdir(parents=True, exist_ok=False)
        readback_dir.mkdir(parents=True, exist_ok=False)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                list(command),
                env=case_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        try:
            payloads = load_runtime_readbacks(readback_dir)
            routes, validation_errors = validate_case_readbacks(case, payloads)
        except ValueError as exc:
            routes = {}
            validation_errors = (str(exc),)
        errors = list(validation_errors)
        if process.returncode != 0:
            errors.insert(0, f"orchestration command exited with {process.returncode}")
        results.append(
            RocmAblationCaseResult(
                case_id=case.case_id,
                status="passed" if not errors else "failed",
                returncode=process.returncode,
                log_path=log_path,
                readback_dir=readback_dir,
                routes=routes,
                errors=tuple(errors),
            )
        )

    write_markdown_summary(
        output_dir / "summary.md",
        results,
        replay=identity,
        executed=execute,
    )
    return tuple(results)


def write_markdown_summary(
    path: Path,
    results: Sequence[RocmAblationCaseResult],
    *,
    replay: Mapping[str, Any],
    executed: bool,
) -> None:
    """Write a compact human-readable report; raw JSON remains worker evidence."""

    lines = [
        "# ROCm rollout/training Attention ablation",
        "",
        f"- Schema: `{ROCM_ABLATION_SCHEMA_VERSION}`",
        f"- Replay identity: `{replay['sha256']}`",
        f"- Executed: `{'yes' if executed else 'no'}`",
        "- FFN / Logp: fixed at `P/P`",
        "",
        "| Attention | Megatron training | vLLM rollout | Calls | Result |",
        "|---|---|---|---:|---|",
    ]
    for result in results:
        training = result.routes.get("training")
        rollout = result.routes.get("rollout")
        calls = sum(route.call_count for route in result.routes.values())
        lines.append(
            "| {case} | {training} | {rollout} | {calls} | {status} |".format(
                case=result.case_id,
                training="—" if training is None else training.implementation,
                rollout="—" if rollout is None else rollout.implementation,
                calls=calls,
                status=result.status,
            )
        )
    failures = [result for result in results if result.errors]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            lines.append(f"- `{result.case_id}`: {'; '.join(result.errors)}")
    lines.extend(
        [
            "",
            "The report is derived from Megatron and vLLM runtime readbacks. A zero-exit",
            "command with missing route evidence is reported as failed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "FROZEN_REPLAY_ENV",
    "REQUIRED_REPLAY_ENV",
    "ROCM_ABLATION_SCHEMA_VERSION",
    "ROCM_ATTENTION_CASE_IDS",
    "STRICT_ROCM_ATTENTION_CORE",
    "STRICT_ROCM_ATTENTION_RUNTIME",
    "STRICT_ROCM_ATTENTION_SCHEDULE",
    "RocmAblationCaseResult",
    "RocmAttentionAblationCase",
    "build_case_environment",
    "load_runtime_readbacks",
    "replay_identity",
    "rocm_attention_ablation_matrix",
    "run_rocm_attention_ablation",
    "validate_case_readbacks",
    "validate_replay_environment",
    "validate_rocm_host",
    "write_markdown_summary",
]
