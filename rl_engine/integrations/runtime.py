# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fail-closed operator routing shared by framework integrations."""

from __future__ import annotations

import atexit
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from rl_engine.integrations.ablation import Implementation, IntegrationPlan


@dataclass(frozen=True)
class OperatorReadback:
    framework: str
    target: str
    module: str
    case_id: str
    implementation: str
    backend_id: str
    call_count: int
    execution_mode: str = "eager"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["provenance"] = dict(self.provenance)
        return payload


class FrameworkOperatorIntegration:
    """Route framework calls without importing or modifying framework packages."""

    def __init__(
        self,
        *,
        framework: str,
        target: str,
        plan: IntegrationPlan,
        rl_kernel_operators: Mapping[str, Callable[..., Any]],
    ) -> None:
        self.framework = framework
        self.target = target
        self.plan = plan
        self._rl_kernel_operators = dict(rl_kernel_operators)
        self._counts: Counter[str] = Counter()
        self._profile_counts: Counter[str] = Counter()
        self._readbacks: dict[str, OperatorReadback] = {}
        self._installed_hooks: dict[str, str] = {}
        self._fallbacks: list[dict[str, str]] = []
        self._reported_routes: set[str] = set()
        self._persisted_route_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._lock = Lock()
        if os.getenv("RL_KERNEL_READBACK_DIR", "").strip():
            atexit.register(self._persist_readback)

    def install_operator(self, module: str, operator: Callable[..., Any]) -> None:
        normalized = module.strip().lower()
        if normalized not in self.plan.cases:
            raise ValueError(f"unknown integration module {module!r}")
        if not callable(operator):
            raise TypeError("operator must be callable")
        with self._lock:
            self._rl_kernel_operators[normalized] = operator

    def record_installed_hook(self, module: str, hook_id: str) -> None:
        normalized = module.strip().lower()
        if normalized not in self.plan.cases:
            raise ValueError(f"unknown integration module {module!r}")
        if not isinstance(hook_id, str) or not hook_id.strip():
            raise ValueError("hook_id must be a non-empty string")
        with self._lock:
            self._installed_hooks[normalized] = hook_id.strip()
        self._persist_readback()

    def record_fallback(self, module: str, reason: str) -> None:
        normalized = module.strip().lower()
        if normalized not in self.plan.cases:
            raise ValueError(f"unknown integration module {module!r}")
        with self._lock:
            self._fallbacks.append({"module": normalized, "reason": str(reason)})
        self._persist_readback()

    def execute(
        self,
        module: str,
        native: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized = module.strip().lower()
        if not callable(native):
            raise TypeError("native operator must be callable")
        implementation = self.plan.implementation_for(normalized, self.target)
        selected: Callable[..., Any]
        if implementation is Implementation.PRODUCTION:
            selected = native
        else:
            rl_kernel_operator = self._rl_kernel_operators.get(normalized)
            if rl_kernel_operator is None:
                raise RuntimeError(
                    f"{self.framework} {normalized} selected RL-Kernel "
                    "but no operator was installed"
                )
            selected = rl_kernel_operator
        result = selected(*args, **kwargs)
        # vLLM's model forward is captured with a Dynamo fullgraph. Readback
        # bookkeeping performs Python locking and JSON I/O, so defer it until
        # the non-compiled execution path while keeping the selected operator
        # inside the captured graph.
        if torch._dynamo.is_compiling():
            return result
        raw_provenance = getattr(selected, "provenance", {})
        if _contains_profile_only(raw_provenance):
            self.record_profile_call(normalized)
        else:
            self.record_execution(
                normalized,
                selected,
                execution_provenance=_execution_provenance(
                    selected=selected,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                ),
            )
        return result

    def record_profile_call(self, module: str) -> None:
        """Record framework shape/profile work without treating it as execution."""

        normalized = module.strip().lower()
        if normalized not in self.plan.cases:
            raise ValueError(f"unknown integration module {module!r}")
        with self._lock:
            self._profile_counts[normalized] += 1
            first_call = self._profile_counts[normalized] == 1
        if first_call:
            self._persist_readback()

    def record_execution(
        self,
        module: str,
        selected: Callable[..., Any],
        *,
        execution_mode: str = "eager",
        execution_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one eager or captured custom-op execution outside Dynamo tracing."""

        normalized = module.strip().lower()
        implementation = self.plan.implementation_for(normalized, self.target)
        if implementation is Implementation.RL_KERNEL and (
            selected is not self._rl_kernel_operators.get(normalized)
        ):
            raise RuntimeError(
                f"{self.framework} {normalized} execution evidence did not use "
                "the installed RL-Kernel operator"
            )
        if execution_mode not in {"eager", "compiled_cuda_graph", "compiled_hip_graph"}:
            raise ValueError(f"unknown execution mode {execution_mode!r}")
        backend_id = getattr(selected, "backend_id", None)
        if not isinstance(backend_id, str) or not backend_id.strip():
            backend_id = (
                f"{self.framework}.production.{normalized}"
                if implementation is Implementation.PRODUCTION
                else f"rlkernel.{normalized}.unidentified"
            )
        if execution_provenance is None:
            raw_provenance = getattr(selected, "provenance", {})
            provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
        else:
            provenance = dict(execution_provenance)
        actual_backend = _actual_backend(provenance) or backend_id
        fallback = _contains_fallback(provenance)
        should_report = False
        route_changed = False
        route_fingerprint = (
            implementation.value,
            backend_id,
            actual_backend,
            fallback,
            _runtime_platform(provenance),
        )
        with self._lock:
            self._counts[normalized] += 1
            case = self.plan.cases[normalized]
            self._readbacks[normalized] = OperatorReadback(
                framework=self.framework,
                target=self.target,
                module=normalized,
                case_id=case.case_id,
                implementation=implementation.value,
                backend_id=backend_id,
                call_count=self._counts[normalized],
                execution_mode=execution_mode,
                provenance=provenance,
            )
            if normalized not in self._reported_routes:
                self._reported_routes.add(normalized)
                should_report = _route_report_enabled()
            if self._persisted_route_fingerprints.get(normalized) != route_fingerprint:
                self._persisted_route_fingerprints[normalized] = route_fingerprint
                route_changed = True
        if route_changed:
            self._persist_readback()
        if should_report:
            print(
                "[RL-Kernel][route] "
                f"framework={self.framework} target={self.target} module={normalized} "
                f"requested={implementation.value} actual={actual_backend} "
                f"fallback={str(fallback).lower()}",
                flush=True,
            )
        if implementation is Implementation.RL_KERNEL and fallback:
            self.record_fallback(normalized, f"operator provenance selected {actual_backend}")
            raise RuntimeError(
                f"{self.framework} {normalized} strict RL-Kernel route reported fallback"
            )

    def readback(self) -> dict[str, Any]:
        with self._lock:
            return {
                "framework": self.framework,
                "target": self.target,
                "plan": self.plan.to_dict(),
                "installed_hooks": dict(self._installed_hooks),
                "fallbacks": list(self._fallbacks),
                "profile_calls": dict(self._profile_counts),
                "operators": {
                    module: readback.to_dict() for module, readback in self._readbacks.items()
                },
            }

    def assert_strict_ready(self) -> None:
        """Fail unless every selected RL-Kernel route was installed and executed."""

        payload = self.readback()
        missing_hooks: list[str] = []
        missing_calls: list[str] = []
        wrong_backends: list[str] = []
        for module in self.plan.cases:
            selected = self.plan.implementation_for(module, self.target)
            if selected is not Implementation.RL_KERNEL:
                continue
            if module not in payload["installed_hooks"]:
                missing_hooks.append(module)
            operator = payload["operators"].get(module)
            if operator is None or int(operator["call_count"]) <= 0:
                missing_calls.append(module)
            elif not str(operator["backend_id"]).startswith(
                ("rlkernel.", "pytorch-vocab-parallel-logp")
            ):
                wrong_backends.append(f"{module}={operator['backend_id']}")
            elif _contains_triton(operator):
                wrong_backends.append(f"{module}=triton")
            elif _runtime_platform(operator.get("provenance")) not in {"cuda", "rocm"}:
                wrong_backends.append(f"{module}=non-cuda/rocm")
        failures: list[str] = []
        if missing_hooks:
            failures.append("missing hooks: " + ", ".join(missing_hooks))
        if missing_calls:
            failures.append("zero calls: " + ", ".join(missing_calls))
        if wrong_backends:
            failures.append("unexpected backends: " + ", ".join(wrong_backends))
        if payload["fallbacks"]:
            failures.append(f"fallbacks: {payload['fallbacks']}")
        if failures:
            raise RuntimeError(
                f"{self.framework} {self.target} integration is not strict-ready: "
                + "; ".join(failures)
            )

    def _persist_readback(self) -> None:
        directory = os.getenv("RL_KERNEL_READBACK_DIR", "").strip()
        if not directory:
            return
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{self.framework}-{self.target}-{os.getpid()}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.readback(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)


def _contains_triton(value: Any) -> bool:
    if isinstance(value, str):
        return "triton" in value.lower()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {"triton_used", "uses_triton"}:
                if item is True:
                    return True
                continue
            if _contains_triton(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_triton(item) for item in value)
    return False


def _contains_fallback(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {
                "fallback",
                "fallback_used",
                "split_kv_fallback",
                "used_fallback",
            } and item not in (False, None, "", 0):
                return True
            if _contains_fallback(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_fallback(item) for item in value)
    return False


def _contains_profile_only(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("profile_only") is True:
        return True
    return any(_contains_profile_only(item) for item in value.values())


def _actual_backend(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("actual_backend")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for item in value.values():
        nested = _actual_backend(item)
        if nested is not None:
            return nested
    return None


def _route_report_enabled() -> bool:
    enabled = os.getenv("RL_KERNEL_ROUTE_REPORT", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return False
    all_ranks = os.getenv("RL_KERNEL_ROUTE_REPORT_ALL_RANKS", "0").strip().lower()
    if all_ranks in {"1", "true", "yes", "on"}:
        return True
    for name in ("RANK", "LOCAL_RANK"):
        value = os.getenv(name, "").strip()
        if value:
            try:
                return int(value) == 0
            except ValueError:
                continue
    return True


def _runtime_platform(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("runtime_platform")
    if isinstance(direct, str):
        return direct
    for item in value.values():
        nested = _runtime_platform(item)
        if nested is not None:
            return nested
    return None


def _execution_provenance(
    *,
    selected: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Capture backend evidence from the callable, result, and real tensors."""

    raw = getattr(selected, "provenance", {})
    provenance = dict(raw) if isinstance(raw, Mapping) else {}
    result_provenance = getattr(result, "provenance", {})
    if isinstance(result_provenance, Mapping) and result_provenance:
        if not provenance:
            provenance.update(result_provenance)
        else:
            provenance.setdefault("result", dict(result_provenance))
    if _runtime_platform(provenance) is None:
        inferred = _infer_runtime_platform((args, kwargs, result))
        if inferred is not None:
            provenance["runtime_platform"] = inferred
    return provenance


def _infer_runtime_platform(value: Any) -> str | None:
    """Infer the device from bounded traversal of execution inputs and outputs."""

    platforms: set[str] = set()
    seen: set[int] = set()
    structural_fields = (
        "context",
        "hidden",
        "hidden_states",
        "logits",
        "target_ids",
        "logp",
        "entropy",
    )

    def visit(item: Any, depth: int) -> None:
        if depth > 4 or len(seen) > 256:
            return
        if isinstance(item, torch.Tensor):
            platforms.add(item.device.type)
            return
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested, depth + 1)
            return
        for field_name in structural_fields:
            try:
                nested = getattr(item, field_name)
            except (AttributeError, RuntimeError):
                continue
            visit(nested, depth + 1)

    visit(value, 0)
    if "cuda" in platforms:
        return "cuda"
    if len(platforms) == 1:
        return next(iter(platforms))
    return None


__all__ = ["FrameworkOperatorIntegration", "OperatorReadback"]
