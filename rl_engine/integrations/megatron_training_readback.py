# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Read-only, per-call evidence capture for the Megatron training logp route (P7 T05).

The observer wraps the existing ``MegatronLogpOperator.__call__`` at class level
(instance identity is preserved for ``FrameworkOperatorIntegration``) and, the
moment each call returns, freezes an immutable :class:`TrainingCallSnapshot`.
It never changes arguments, results, autograd, RNG or the execution path, and
never re-runs a kernel. Every field records where its value came from:

``observed``    value actually passed to or returned by this call
``declared``    self-reported by code (provider/operator/wrapper provenance)
``configured``  caller-supplied configuration (e.g. VIME request metadata)
``upstream``    producer evidence forwarded verbatim

Only ``observed`` and ``upstream`` evidence may certify a strict result; the
other sources are kept for cross-checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

EVIDENCE_SCHEMA_VERSION = "rlkernel.p7.t05.training_call_evidence.v1-draft"
_OBSERVER_MARKER = "_rlk_t05_training_observer"
_CALL = "__call__"  # patched via setattr: mypy rejects assigning to a method


class EvidenceSource(str, Enum):
    OBSERVED = "observed"
    DECLARED = "declared"
    CONFIGURED = "configured"
    UPSTREAM = "upstream"


def tensor_identity(tensor: torch.Tensor, *, digest: bool = True) -> dict[str, Any]:
    """Layout plus raw-byte SHA-256 of a detached copy; the input tensor is untouched."""

    detached = tensor.detach()
    identity: dict[str, Any] = {
        "kind": "tensor",
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "requires_grad": bool(tensor.requires_grad),
    }
    if digest:
        # Hash broadcast (stride-0) dimensions once instead of materializing them:
        # a zero-stride placeholder must not allocate [N, V_local] on the host.
        broadcast = [
            dim
            for dim, (size, step) in enumerate(zip(detached.shape, detached.stride(), strict=True))
            if step == 0 and size > 1
        ]
        reduced = detached
        for dim in broadcast:
            reduced = reduced.narrow(dim, 0, 1)
        # contiguous() keeps stride 0 on size-1 dims, so copy into a fresh dense buffer.
        source = reduced.cpu().reshape(-1)
        flat = torch.empty(source.shape, dtype=source.dtype).copy_(source)
        identity["sha256"] = hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()
        if broadcast:
            identity["broadcast_dims"] = broadcast
    return identity


def json_safe(value: Any, *, digest: bool = True, depth: int = 0) -> Any:
    """Deep, frozen-by-copy JSON representation; tensors become identities, never values."""

    if depth > 16:
        return {"kind": "truncated"}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        return tensor_identity(value, digest=digest)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item, digest=digest, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item, digest=digest, depth=depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item, digest=digest, depth=depth + 1) for item in value), key=repr)
    return {"kind": "object", "type": f"{type(value).__module__}.{type(value).__qualname__}"}


@dataclass(frozen=True)
class EvidenceField:
    name: str
    source: EvidenceSource
    origin: str
    value: Any
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.value,
            "origin": self.origin,
            "present": self.present,
            "value": copy.deepcopy(self.value),
        }


@dataclass(frozen=True)
class TrainingCallSnapshot:
    """Immutable evidence for exactly one training logp call."""

    call_index: int
    fields: tuple[EvidenceField, ...]
    capture_error: Optional[str] = None
    operator_error: Optional[str] = None
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    # Host copy of this call's selected logprob (values, not just identity): the
    # M10 output and the non-finite gate need them. Its identity is in ``fields``.
    selected_logprob: Optional[torch.Tensor] = dataclass_field(
        default=None, compare=False, repr=False
    )
    # Host copy of this call's target ids (values): selected-target ownership is derived from them.
    target_ids: Optional[torch.Tensor] = dataclass_field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        names = [item.name for item in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate evidence field names: {sorted(names)}")

    def field(self, name: str) -> EvidenceField:
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "call_index": self.call_index,
            "capture_error": self.capture_error,
            "operator_error": self.operator_error,
            "fields": [item.to_dict() for item in self.fields],
        }

    @property
    def fingerprint(self) -> str:
        """Content fingerprint; excludes ``call_index`` so identical calls compare equal."""

        payload = self.to_dict()
        payload.pop("call_index")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_ABSENT = object()


def _read(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        current = getattr(current, part, _ABSENT)
        if current is _ABSENT:
            return _ABSENT
    return current


def _field(
    root: Any,
    path: str,
    source: EvidenceSource,
    *,
    origin: str,
    digest: bool,
    name: Optional[str] = None,
) -> EvidenceField:
    label = name or path
    value = _read(root, path)
    if value is _ABSENT:
        return EvidenceField(label, source, origin, None, present=False)
    return EvidenceField(label, source, origin, json_safe(value, digest=digest))


_REQUEST_OBSERVED = (
    "logits",
    "target_ids",
    "log_prob_keep_mask",
    "temperature",
    "with_entropy",
    "with_entropy_grad",
    "chunk_size",
    "token_layout.world_size",
    "token_layout.rank",
    "token_layout.layout",
    "context.hidden",
    "context.projection.weight",
    "context.projection.bias",
    "context.vocab_partition.local_start",
    "context.vocab_partition.local_size",
    "context.vocab_partition.real_size",
    "context.vocab_partition.padded_size",
)


def _group_coordinates(group: Any) -> dict[str, Any]:
    if group is None:
        return {"group": None}
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return {
            "group": "present",
            "rank": None,
            "world_size": None,
            "reason": "dist not initialized",
        }
    return {
        "group": "present",
        "rank": int(torch.distributed.get_rank(group=group)),
        "world_size": int(torch.distributed.get_world_size(group=group)),
    }


CANONICAL_TP_ENV = "RL_KERNEL_STRICT_CANONICAL_TP"
CANONICAL_VOCAB_ENV = "RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE"


def _bytes_sha256(tensor: torch.Tensor) -> str:
    flat = tensor.detach().cpu().reshape(-1)
    dense = torch.empty(flat.shape, dtype=flat.dtype).copy_(flat)
    return hashlib.sha256(dense.view(torch.uint8).numpy().tobytes()).hexdigest()


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _call_device(request: Any) -> Optional[torch.device]:
    for path in ("context.hidden", "logits"):
        value = _read(request, path)
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def rng_identity(device: Optional[torch.device]) -> dict[str, Any]:
    """Digests of RNG states; reads only, never initializes CUDA or reseeds."""

    identity: dict[str, Any] = {"cpu": _bytes_sha256(torch.get_rng_state()), "cuda": None}
    if device is not None and device.type == "cuda" and torch.cuda.is_initialized():
        identity["cuda"] = _bytes_sha256(torch.cuda.get_rng_state(device))
    try:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        tracker = get_cuda_rng_tracker()
        if tracker.is_initialized():
            identity["megatron_tracker"] = {
                str(name): _bytes_sha256(state)
                for name, state in sorted(tracker.get_states().items())
            }
        else:
            identity["megatron_tracker"] = "uninitialized"
    except Exception:
        identity["megatron_tracker"] = None
    return identity


def stream_identity(device: Optional[torch.device]) -> dict[str, Any]:
    if device is not None and device.type == "cuda" and torch.cuda.is_initialized():
        stream = torch.cuda.current_stream(device)
        return {
            "device": str(device),
            "stream": int(stream.cuda_stream),
            "device_name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {"device": None if device is None else str(device), "stream": None}


def build_identity_fields() -> tuple[EvidenceField, ...]:
    """Code/binary identity of the strict route; computed once per observer."""

    fields: list[EvidenceField] = []
    try:
        from rl_engine.integrations import linear_logp as wrapper_module

        fields.append(
            EvidenceField(
                "build.wrapper_module_sha256",
                EvidenceSource.DECLARED,
                "rl_engine.integrations.linear_logp source",
                _file_sha256(getattr(wrapper_module, "__file__", None)),
            )
        )
    except Exception as exc:
        fields.append(
            EvidenceField(
                "build.wrapper_module_sha256", EvidenceSource.DECLARED, repr(exc), None, False
            )
        )
    try:
        from rl_engine.kernels.ops.cuda.loss import linear_logp as kernel_module

        fields += [
            EvidenceField(
                "build.kernel_module_sha256",
                EvidenceSource.DECLARED,
                "rl_engine.kernels.ops.cuda.loss.linear_logp source",
                _file_sha256(getattr(kernel_module, "__file__", None)),
            ),
            EvidenceField(
                "build.strict_contract_version",
                EvidenceSource.DECLARED,
                "kernel module constant",
                getattr(kernel_module, "STRICT_LINEAR_LOGP_CONTRACT_VERSION", None),
            ),
            EvidenceField(
                "build.sm90_n_split_contract",
                EvidenceSource.DECLARED,
                "kernel module constant (vocab split count, not the BN tile width)",
                getattr(kernel_module, "SM90_N_SPLIT_CONTRACT", None),
            ),
        ]
    except Exception as exc:
        fields.append(
            EvidenceField(
                "build.kernel_module_sha256", EvidenceSource.DECLARED, repr(exc), None, False
            )
        )
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        path = getattr(_C, "__file__", None) if _EXT_AVAILABLE else None
        extension: Any = {
            "loaded": bool(_EXT_AVAILABLE),
            "path": path,
            "sha256": _file_sha256(path),
        }
    except Exception as exc:
        extension = {"loaded": False, "error": repr(exc)}
    fields.append(
        EvidenceField(
            "build.extension", EvidenceSource.OBSERVED, "loaded rl_engine._C binary", extension
        )
    )
    fields.append(
        EvidenceField(
            "build.torch",
            EvidenceSource.OBSERVED,
            "running torch",
            {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "hip": torch.version.hip,
                "cxx11abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            },
        )
    )
    return tuple(fields)


def temperature_identity(value: Any) -> dict[str, Any]:
    """Whether the temperature passed to this call is the identity (1.0 or absent)."""

    if value is None:
        return {"kind": "none", "identity": True}
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        return {"kind": "tensor", "identity": bool(torch.all(detached == 1).item())}
    return {"kind": "scalar", "identity": float(value) == 1.0, "value": float(value)}


def capture_request(request: Any, *, digest: bool = True) -> list[EvidenceField]:
    fields = [
        _field(request, path, EvidenceSource.OBSERVED, origin=f"request.{path}", digest=digest)
        for path in _REQUEST_OBSERVED
    ]
    temperature = _read(request, "temperature")
    fields.append(
        EvidenceField(
            "temperature.identity",
            EvidenceSource.OBSERVED,
            "request.temperature (value check)",
            None if temperature is _ABSENT else temperature_identity(temperature),
            present=temperature is not _ABSENT,
        )
    )
    group = _read(request, "tensor_parallel_group")
    fields.append(
        EvidenceField(
            "tensor_parallel_group.coordinates",
            EvidenceSource.OBSERVED,
            "torch.distributed(request.tensor_parallel_group)",
            None if group is _ABSENT else _group_coordinates(group),
            present=group is not _ABSENT,
        )
    )
    fields.append(
        _field(
            request, "metadata", EvidenceSource.CONFIGURED, origin="request.metadata", digest=digest
        )
    )
    return fields


def capture_result(result: Any, *, digest: bool = True) -> list[EvidenceField]:
    observed = ("logp", "entropy")
    return [
        _field(
            result,
            path,
            EvidenceSource.OBSERVED if path in observed else EvidenceSource.DECLARED,
            origin=f"result.{path}",
            digest=digest,
            name=f"result.{path}",
        )
        for path in ("logp", "entropy", "backend_id", "contract_id", "provenance")
    ]


def capture_operator(operator: Any, *, digest: bool = True) -> list[EvidenceField]:
    fields = [
        _field(
            operator,
            "provenance",
            EvidenceSource.DECLARED,
            origin="operator.provenance",
            digest=digest,
            name="operator.provenance",
        )
    ]
    linear = getattr(operator, "_linear_logp", None)
    fields.append(
        EvidenceField(
            "linear_logp.provenance",
            EvidenceSource.DECLARED,
            "operator._linear_logp.provenance",
            (
                None
                if linear is None
                else json_safe(getattr(linear, "provenance", None), digest=digest)
            ),
            present=linear is not None,
        )
    )
    return fields


def capture_runtime(request: Any, before: Mapping[str, Any]) -> list[EvidenceField]:
    """RNG before/after (consumption check), stream, and configured canonical-TP env."""

    device = _call_device(request)
    after = rng_identity(device)
    rng_before = before.get("rng")
    return [
        EvidenceField(
            "runtime.rng_before", EvidenceSource.OBSERVED, "pre-call RNG digests", rng_before
        ),
        EvidenceField("runtime.rng_after", EvidenceSource.OBSERVED, "post-call RNG digests", after),
        EvidenceField(
            "runtime.rng_consumed",
            EvidenceSource.OBSERVED,
            "pre/post RNG digest comparison",
            rng_before != after,
        ),
        EvidenceField(
            "runtime.stream",
            EvidenceSource.OBSERVED,
            "current stream at call",
            before.get("stream"),
        ),
        EvidenceField(
            "configured.canonical_tp",
            EvidenceSource.CONFIGURED,
            f"env {CANONICAL_TP_ENV} at call time",
            os.environ.get(CANONICAL_TP_ENV),
        ),
        EvidenceField(
            "configured.canonical_vocab_size",
            EvidenceSource.CONFIGURED,
            f"env {CANONICAL_VOCAB_ENV} at call time",
            os.environ.get(CANONICAL_VOCAB_ENV),
        ),
    ]


class TrainingLogpObserver:
    """Install/uninstall a read-only per-call observer on the Megatron logp operator class."""

    def __init__(
        self,
        *,
        sink: Optional[Callable[[TrainingCallSnapshot], None]] = None,
        max_snapshots: int = 1024,
        digest_tensors: bool = True,
    ) -> None:
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be positive")
        self._sink = sink
        self._max = max_snapshots
        self._digest = digest_tensors
        self._lock = threading.Lock()
        self._snapshots: list[TrainingCallSnapshot] = []
        self._next_index = 0
        self._installed_on: Optional[type[Any]] = None
        self.sink_errors: list[str] = []
        self._build: Optional[tuple[EvidenceField, ...]] = None

    @property
    def snapshots(self) -> tuple[TrainingCallSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots)

    @property
    def installed(self) -> bool:
        return self._installed_on is not None

    def install(self, operator_cls: Optional[type[Any]] = None) -> None:
        if operator_cls is None:
            from rl_engine.integrations.framework_operators import MegatronLogpOperator

            operator_cls = MegatronLogpOperator
        existing = operator_cls.__dict__.get(_OBSERVER_MARKER)
        if existing is not None:
            if existing[0] is self:
                return
            raise RuntimeError(f"another T05 observer is already installed on {operator_cls!r}")
        if self._installed_on is not None:
            raise RuntimeError("this observer is already installed on another class")
        original = operator_cls.__dict__[_CALL]
        observer = self

        def observed_call(operator: Any, request: Any) -> Any:
            index = observer._claim_index()
            before = observer._before_call(request)
            try:
                result = original(operator, request)
            except BaseException as exc:
                observer._record(
                    index, operator, request, None, before, operator_error=type(exc).__name__
                )
                raise
            observer._record(index, operator, request, result, before)
            return result

        observed_call.__name__ = getattr(original, "__name__", "__call__")
        observed_call.__doc__ = getattr(original, "__doc__", None)
        observed_call.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(operator_cls, _OBSERVER_MARKER, (self, original))
        setattr(operator_cls, _CALL, observed_call)
        self._installed_on = operator_cls

    def uninstall(self) -> None:
        operator_cls = self._installed_on
        if operator_cls is None:
            return
        marker = operator_cls.__dict__.get(_OBSERVER_MARKER)
        if marker is None or marker[0] is not self:
            raise RuntimeError("observer marker was replaced; refusing to restore __call__")
        setattr(operator_cls, _CALL, marker[1])
        delattr(operator_cls, _OBSERVER_MARKER)
        self._installed_on = None

    def _before_call(self, request: Any) -> dict[str, Any]:
        try:
            device = _call_device(request)
            return {"rng": rng_identity(device), "stream": stream_identity(device)}
        except Exception as exc:  # never alter the training call
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _build_fields(self) -> tuple[EvidenceField, ...]:
        with self._lock:
            if self._build is None:
                self._build = build_identity_fields()
            return self._build

    def _claim_index(self) -> int:
        with self._lock:
            index = self._next_index
            self._next_index += 1
            return index

    def _record(
        self,
        index: int,
        operator: Any,
        request: Any,
        result: Any,
        before: Mapping[str, Any],
        *,
        operator_error: Optional[str] = None,
    ) -> None:
        try:
            if "error" in before:
                raise RuntimeError(f"pre-call capture failed: {before['error']}")
            fields = capture_request(request, digest=self._digest)
            if result is not None:
                fields += capture_result(result, digest=self._digest)
            fields += capture_operator(operator, digest=self._digest)
            fields += capture_runtime(request, before)
            fields += list(self._build_fields())
            logp = getattr(result, "logp", None) if result is not None else None
            targets = _read(request, "target_ids")
            snapshot = TrainingCallSnapshot(
                call_index=index,
                fields=tuple(fields),
                operator_error=operator_error,
                selected_logprob=(
                    logp.detach().cpu().clone() if isinstance(logp, torch.Tensor) else None
                ),
                target_ids=(
                    targets.detach().cpu().clone() if isinstance(targets, torch.Tensor) else None
                ),
            )
        except Exception as exc:  # evidence capture must never alter the training call
            snapshot = TrainingCallSnapshot(
                call_index=index,
                fields=(),
                capture_error=f"{type(exc).__name__}: {exc}",
                operator_error=operator_error,
            )
        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max:
                del self._snapshots[: len(self._snapshots) - self._max]
        if self._sink is not None:
            try:
                self._sink(snapshot)
            except Exception as exc:  # a broken sink must not alter the training call
                with self._lock:
                    self.sink_errors.append(f"call {index}: {type(exc).__name__}: {exc}")


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceField",
    "EvidenceSource",
    "TrainingCallSnapshot",
    "TrainingLogpObserver",
    "capture_operator",
    "capture_request",
    "build_identity_fields",
    "capture_result",
    "capture_runtime",
    "json_safe",
    "rng_identity",
    "stream_identity",
    "temperature_identity",
    "tensor_identity",
]
