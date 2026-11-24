# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini Miles/SGLang integration and runtime gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


class RuntimeContractError(ValueError):
    """Raised when an integration contract is malformed."""


@dataclass(frozen=True)
class RuntimeGateReport:
    name: str
    status: str
    blockers: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready" and not self.blockers


@dataclass(frozen=True)
class RuntimeGate:
    name: str
    required_modules: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    require_cuda: bool = False
    require_rocm: bool = False

    def probe(self) -> RuntimeGateReport:
        blockers = []
        modules = {}
        for module in self.required_modules:
            available = importlib.util.find_spec(module) is not None
            modules[module] = available
            if not available:
                blockers.append(f"missing module {module}")
        environment = {}
        for variable in self.required_env:
            present = bool(os.getenv(variable, "").strip())
            environment[variable] = present
            if not present:
                blockers.append(f"missing environment variable {variable}")
        if self.require_cuda and not torch.cuda.is_available():
            blockers.append("CUDA device is unavailable")
        if self.require_rocm and (torch.version.hip is None or not torch.cuda.is_available()):
            blockers.append("ROCm device is unavailable")
        return RuntimeGateReport(
            self.name,
            "ready" if not blockers else "gated",
            tuple(blockers),
            {"modules": modules, "environment": environment},
        )


@dataclass(frozen=True)
class WeightKeyMapping:
    prefixes: tuple[tuple[str, str], ...]

    def map_key(self, source_key: str) -> str:
        matches = [
            target + source_key[len(source) :]
            for source, target in self.prefixes
            if source_key.startswith(source)
        ]
        if len(matches) != 1:
            raise RuntimeContractError(
                f"weight key does not have exactly one mapping: {source_key}"
            )
        return matches[0]

    def map_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped: dict[str, torch.Tensor] = {}
        for source_key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"weight {source_key} is not a tensor")
            target_key = self.map_key(source_key)
            if target_key in mapped:
                raise RuntimeContractError(f"multiple source weights map to {target_key}")
            mapped[target_key] = tensor
        if not mapped:
            raise RuntimeContractError("state dict mapping produced no tensors")
        return mapped


@dataclass(frozen=True)
class CheckpointExport:
    revision: str
    tensors: Mapping[str, torch.Tensor]
    dtype: torch.dtype = torch.bfloat16

    def convert(self, mapping: WeightKeyMapping) -> dict[str, torch.Tensor]:
        converted = {}
        for name, tensor in mapping.map_state_dict(self.tensors).items():
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise RuntimeContractError(
                    f"unsupported checkpoint dtype for {name}: {tensor.dtype}"
                )
            converted[name] = tensor.to(dtype=self.dtype).contiguous()
        return converted

    def fingerprint(self, mapping: WeightKeyMapping) -> str:
        converted = self.convert(mapping)
        digest = hashlib.sha256()
        digest.update(self.revision.encode())
        for name in sorted(converted):
            tensor = converted[name]
            digest.update(name.encode())
            digest.update(str(tensor.shape).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.cpu().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class RouteReplayRecord:
    token_id: int
    expert_ids: tuple[int, ...]
    weights: tuple[float, ...]
    visit_id: int


class VersionedWeightSync:
    """Coordinate model-version publication, cache reset, and route replay."""

    def __init__(
        self,
        *,
        reset_cache: Callable[[], None],
        replay_routes: Callable[[Sequence[RouteReplayRecord]], None],
    ):
        self._reset_cache = reset_cache
        self._replay_routes = replay_routes
        self._version = 0
        self._fingerprint = ""

    @property
    def version(self) -> int:
        return self._version

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def publish(self, *, version: int, fingerprint: str) -> None:
        if version <= self._version:
            raise RuntimeContractError("weight versions must increase monotonically")
        if not fingerprint:
            raise RuntimeContractError("weight fingerprint must be nonempty")
        self._reset_cache()
        self._version = version
        self._fingerprint = fingerprint

    def replay(self, routes: Sequence[RouteReplayRecord]) -> None:
        if self._version == 0:
            raise RuntimeContractError("publish a weight version before route replay")
        for route in routes:
            if route.token_id < 0 or route.visit_id < 0 or not route.expert_ids:
                raise RuntimeContractError("route replay contains invalid identity")
            if len(route.expert_ids) != len(route.weights):
                raise RuntimeContractError("route replay expert IDs and weights must align")
        self._replay_routes(routes)


@dataclass(frozen=True)
class MilesRunSpec:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    platform: str
    strict: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, default=list)


def build_miles_native_spec(*, platform: str, strict: bool = False) -> MilesRunSpec:
    if platform not in {"cuda", "rocm"}:
        raise RuntimeContractError("platform must be cuda or rocm")
    return MilesRunSpec(
        command=("python", "scripts/run_inkling2_mini.py", "--mode", "native"),
        environment={"RL_KERNEL_PLATFORM": platform, "RL_KERNEL_STRICT": str(strict).lower()},
        platform=platform,
        strict=strict,
    )


def rocm_runtime_gate() -> RuntimeGateReport:
    return RuntimeGate(
        "INK-74",
        required_modules=("torch",),
        required_env=("MILES_CHECKPOINT", "SGLANG_CONFIG"),
        require_rocm=True,
    ).probe()
