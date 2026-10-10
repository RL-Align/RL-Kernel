# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exact semantic-operator catalog with case-local instantiation state."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from types import CodeType, MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, cast


class OperatorLifecycle(str, Enum):
    REQUEST = "request"
    ENGINE_CONSTRUCTION = "engine_construction"
    DISTRIBUTED_CONTEXT = "distributed_context"
    PROCESS = "process"


class OperatorFallbackPolicy(str, Enum):
    ERROR = "error"
    DECLARED = "declared"
    RUNTIME_MANAGED = "runtime_managed"


@dataclass(frozen=True)
class OperatorResolutionPolicy:
    strict: bool = True
    allow_test_backends: bool = False


_Policy = Optional[OperatorResolutionPolicy]


class _JsonRecord:
    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(cast(Any, self))
        }


@dataclass(frozen=True)
class OperatorRequirements(_JsonRecord):
    device: str
    dtype: str
    topology: Mapping[str, Any] = field(default_factory=dict)
    alignment_properties: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "rlkernel.semantic_operator.requirements.v1"

    def __post_init__(self) -> None:
        normalized = (
            ("device", _normalize_device(self.device)),
            ("dtype", _normalize_dtype(self.dtype)),
            ("topology", _freeze(self.topology)),
            ("alignment_properties", _freeze(self.alignment_properties)),
        )
        for name, value in normalized:
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class OperatorBackendDescriptor(_JsonRecord):
    semantic_op: str
    backend_id: str
    supported_targets: frozenset[str]
    supported_devices: frozenset[str]
    supported_dtypes: frozenset[str]
    supported_topologies: Mapping[str, Any]
    determinism_or_alignment_properties: Mapping[str, Any]
    lifecycle: OperatorLifecycle
    implementation_class_or_factory: Optional[str | Callable[..., Any]]
    fallback_policy: OperatorFallbackPolicy
    version_or_build_fingerprint: str
    is_smoke_only: bool = False
    schema_version: str = "rlkernel.semantic_operator.backend_descriptor.v1"

    def __post_init__(self) -> None:
        values = {
            "semantic_op": self.semantic_op.strip(),
            "backend_id": self.backend_id.strip(),
            "supported_targets": _normalized_values(self.supported_targets, str),
            "supported_devices": _normalized_values(self.supported_devices, _normalize_device),
            "supported_dtypes": _normalized_values(self.supported_dtypes, _normalize_dtype),
        }
        for name, value in values.items():
            if not value:
                raise ValueError(f"{name} must not be empty")
        if not self.version_or_build_fingerprint.strip():
            raise ValueError("version_or_build_fingerprint must not be empty")
        values.update(
            supported_topologies=_freeze(self.supported_topologies),
            determinism_or_alignment_properties=_freeze(self.determinism_or_alignment_properties),
            lifecycle=OperatorLifecycle(self.lifecycle),
            fallback_policy=OperatorFallbackPolicy(self.fallback_policy),
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def implementation_reference(self) -> Optional[str]:
        return _reference(self.implementation_class_or_factory)

    @property
    def is_strictly_observable(self) -> bool:
        return bool(
            self.determinism_or_alignment_properties.get(
                "strict_observable", self.implementation_class_or_factory is not None
            )
        )

    @property
    def descriptor_fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_descriptor_fingerprint=False))

    def to_dict(self, *, include_descriptor_fingerprint: bool = True) -> dict[str, Any]:
        result = super().to_dict()
        result["implementation_class_or_factory"] = self.implementation_reference
        if include_descriptor_fingerprint:
            result["descriptor_fingerprint"] = self.descriptor_fingerprint
        return result


@dataclass(frozen=True)
class OperatorCapabilityDecision(_JsonRecord):
    capability: str
    requested: Any
    supported: Any
    passed: bool
    reason: str


@dataclass(frozen=True)
class OperatorResolutionTrace(_JsonRecord):
    semantic_op: str
    requested_backend: str
    target: str
    strict: bool
    status: str
    concrete_backend: Optional[str]
    implementation_reference: Optional[str]
    descriptor_fingerprint: Optional[str]
    capability_decisions: tuple[OperatorCapabilityDecision, ...]
    fallback_attempts: tuple[str, ...] = ()
    schema_version: str = "rlkernel.semantic_operator.resolution_trace.v1"


@dataclass(frozen=True)
class OperatorResolution(_JsonRecord):
    descriptor: OperatorBackendDescriptor
    requirements: OperatorRequirements
    target: str
    strict: bool
    trace: OperatorResolutionTrace
    schema_version: str = "rlkernel.semantic_operator.resolution.v1"


@dataclass(frozen=True)
class OperatorInstanceProvenance(_JsonRecord):
    semantic_op: str
    backend_id: str
    target: str
    factory_reference: str
    concrete_implementation: str
    descriptor_fingerprint: str
    implementation_fingerprint: str
    instance_fingerprint: str
    factory_options: Mapping[str, Any] = field(default_factory=dict)
    factory_options_fingerprint: str = ""
    schema_version: str = "rlkernel.semantic_operator.instance_provenance.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "factory_options", _freeze(self.factory_options))


@dataclass(frozen=True)
class _InstanceRecord:
    instance: Any
    descriptor_fingerprint: str
    target: str
    factory: Callable[..., Any]
    factory_options: Mapping[str, Any]


class OperatorRegistrationError(ValueError):
    pass


class OperatorResolutionError(RuntimeError):
    def __init__(self, message: str, trace: OperatorResolutionTrace):
        super().__init__(message)
        self.trace = trace


class OperatorInstantiationError(RuntimeError):
    pass


class SemanticOperatorCatalog:
    def __init__(self, descriptors: Iterable[OperatorBackendDescriptor] = ()):
        self._descriptors: dict[tuple[str, str], OperatorBackendDescriptor] = {}
        for descriptor in descriptors:
            self.register_backend(descriptor)

    def register_backend(
        self,
        descriptor: OperatorBackendDescriptor,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(descriptor, OperatorBackendDescriptor):
            raise TypeError("descriptor must be an OperatorBackendDescriptor")
        key = (descriptor.semantic_op, descriptor.backend_id)
        if key in self._descriptors and not replace:
            raise OperatorRegistrationError(f"operator backend is already registered: {key!r}")
        self._descriptors[key] = descriptor

    def backend_descriptor(
        self,
        semantic_op: str,
        backend_id: str,
    ) -> Optional[OperatorBackendDescriptor]:
        return self._descriptors.get((semantic_op.strip(), backend_id.strip()))

    def backend_descriptors(
        self,
        semantic_op: Optional[str] = None,
    ) -> tuple[OperatorBackendDescriptor, ...]:
        values: Iterable[OperatorBackendDescriptor] = self._descriptors.values()
        if semantic_op is not None:
            normalized = semantic_op.strip()
            values = (value for value in values if value.semantic_op == normalized)
        return tuple(sorted(values, key=lambda value: (value.semantic_op, value.backend_id)))

    def session(self, policy: _Policy = None) -> OperatorSession:
        return OperatorSession(self, policy=policy)

    def _resolve(
        self,
        *,
        semantic_op: str,
        requested_backend: str,
        target: str,
        requirements: OperatorRequirements,
        policy: OperatorResolutionPolicy,
    ) -> OperatorResolution:
        semantic_op = semantic_op.strip()
        requested_backend = requested_backend.strip()
        target = target.strip().lower()
        if not semantic_op or not requested_backend or not target:
            raise ValueError("semantic_op, requested_backend, and target must not be empty")
        if not isinstance(requirements, OperatorRequirements):
            raise TypeError("requirements must be an OperatorRequirements")

        descriptor = self.backend_descriptor(semantic_op, requested_backend)
        if descriptor is None:
            decision = _decision(
                "registration",
                requested_backend,
                [item.backend_id for item in self.backend_descriptors(semantic_op)],
                passed=False,
            )
            trace = _trace(
                semantic_op,
                requested_backend,
                target,
                policy,
                "unsupported",
                (decision,),
            )
            raise OperatorResolutionError(
                f"exact operator backend {requested_backend!r} is not registered; "
                "no fallback was attempted",
                trace,
            )

        topology_capabilities = _target_topology_capabilities(
            descriptor.supported_topologies,
            target,
        )
        topology_ok = topology_capabilities is not None and _supports_complete_mapping(
            topology_capabilities,
            requirements.topology,
        )
        decisions = (
            _decision("target", target, descriptor.supported_targets),
            _decision(
                "smoke_opt_in",
                descriptor.is_smoke_only,
                policy.allow_test_backends,
                not descriptor.is_smoke_only or policy.allow_test_backends,
            ),
            _decision("device", requirements.device, descriptor.supported_devices),
            _decision("dtype", requirements.dtype, descriptor.supported_dtypes),
            _decision(
                "topology",
                requirements.topology,
                topology_capabilities,
                topology_ok,
            ),
            _decision(
                "alignment_properties",
                requirements.alignment_properties,
                descriptor.determinism_or_alignment_properties,
            ),
            _decision(
                "strict_observability",
                policy.strict,
                descriptor.is_strictly_observable,
                not policy.strict or descriptor.is_strictly_observable,
            ),
            _decision(
                "fallback_policy",
                "error" if policy.strict else "declared",
                descriptor.fallback_policy.value,
                not policy.strict or descriptor.fallback_policy is OperatorFallbackPolicy.ERROR,
            ),
        )
        failed = tuple(item for item in decisions if not item.passed)
        observable = descriptor.is_strictly_observable
        status = "unsupported" if failed else ("resolved" if observable else "unobservable")
        trace = _trace(
            semantic_op,
            requested_backend,
            target,
            policy,
            status,
            decisions,
            descriptor,
        )
        if failed:
            raise OperatorResolutionError(
                f"exact operator backend {requested_backend!r} is unsupported: "
                + "; ".join(item.reason for item in failed),
                trace,
            )
        return OperatorResolution(descriptor, requirements, target, policy.strict, trace)


class OperatorSession:
    def __init__(self, catalog: SemanticOperatorCatalog, policy: _Policy = None):
        if not isinstance(catalog, SemanticOperatorCatalog):
            raise TypeError("catalog must be a SemanticOperatorCatalog")
        self.catalog = catalog
        self.policy = policy or OperatorResolutionPolicy()
        self._cache: dict[str, Any] = {}
        self._records: dict[int, _InstanceRecord] = {}

    def resolve(
        self,
        *,
        semantic_op: str,
        requested_backend: str,
        target: str,
        requirements: OperatorRequirements,
        policy: _Policy = None,
        strict: Optional[bool] = None,
    ) -> OperatorResolution:
        return self.catalog._resolve(
            semantic_op=semantic_op,
            requested_backend=requested_backend,
            target=target,
            requirements=requirements,
            policy=_resolve_policy(policy or self.policy, strict),
        )

    def instantiate(
        self,
        resolution: OperatorResolution,
        *,
        factory_kwargs: Optional[Mapping[str, Any]] = None,
        cache: bool = False,
    ) -> Any:
        if not isinstance(resolution, OperatorResolution):
            raise TypeError("resolution must be an OperatorResolution")
        descriptor = resolution.descriptor
        implementation = descriptor.implementation_class_or_factory
        if resolution.trace.status != "resolved" or implementation is None:
            raise OperatorInstantiationError(
                f"backend {descriptor.backend_id!r} has no exact implementation"
            )
        options = dict(factory_kwargs or {})
        cache_key = _fingerprint(
            {
                "descriptor": descriptor.descriptor_fingerprint,
                "target": resolution.target,
                "requirements": resolution.requirements.to_dict(),
                "options": options,
            }
        )
        if cache and cache_key in self._cache:
            return self._cache[cache_key]
        factory = _load_factory(implementation)
        try:
            instance = factory(**options)
        except Exception as exc:
            raise OperatorInstantiationError(
                f"failed to instantiate backend {descriptor.backend_id!r}: {exc}"
            ) from exc
        if instance is None:
            raise OperatorInstantiationError("operator factory returned None")
        self._records[id(instance)] = _InstanceRecord(
            instance,
            descriptor.descriptor_fingerprint,
            resolution.target,
            factory,
            _freeze(options),
        )
        if cache:
            self._cache[cache_key] = instance
        return instance

    def instance_provenance(
        self,
        resolution: OperatorResolution,
        instance: Any,
    ) -> OperatorInstanceProvenance:
        descriptor = resolution.descriptor
        record = self._records.get(id(instance))
        if (
            record is None
            or record.instance is not instance
            or record.descriptor_fingerprint != descriptor.descriptor_fingerprint
            or record.target != resolution.target
        ):
            raise OperatorInstantiationError(
                "operator instance does not match this session resolution"
            )
        factory_reference = descriptor.implementation_reference
        concrete = _reference(type(instance))
        if factory_reference is None or concrete is None:
            raise OperatorInstantiationError("operator implementation is not observable")
        options_fingerprint = _fingerprint(record.factory_options)
        implementation_fingerprint = operator_implementation_fingerprint(
            record.factory,
            instance,
        )
        instance_fingerprint = operator_instance_fingerprint(
            descriptor_fingerprint=descriptor.descriptor_fingerprint,
            factory_reference=factory_reference,
            concrete_implementation=concrete,
            implementation_fingerprint=implementation_fingerprint,
            factory_options_fingerprint=options_fingerprint,
        )
        return OperatorInstanceProvenance(
            descriptor.semantic_op,
            descriptor.backend_id,
            resolution.target,
            factory_reference,
            concrete,
            descriptor.descriptor_fingerprint,
            implementation_fingerprint,
            instance_fingerprint,
            record.factory_options,
            options_fingerprint,
        )

    def clear_instance_cache(self) -> None:
        self._cache.clear()


def operator_implementation_fingerprint(
    implementation: str | Callable[..., Any],
    instance: Any,
) -> str:
    return implementation_fingerprint(
        implementation,
        instance=instance,
        entrypoints=("apply_fp32", "__call__"),
    )


def implementation_fingerprint(
    implementation: str | Callable[..., Any],
    *,
    instance: Any = None,
    entrypoints: Sequence[str] = (),
) -> str:
    """Fingerprint executable code, not only its import reference.

    The identity includes source or bytecode for the resolved factory, its
    concrete class, the defining modules, and explicitly named runtime entry
    points. Module content covers helper functions called by an entry point;
    callable identities additionally make in-process replacements observable.
    """

    factory = _load_factory(implementation)
    concrete_type = type(instance) if instance is not None else None
    runtime_entrypoints = {}
    if instance is not None:
        for name in sorted(set(entrypoints)):
            value = getattr(instance, name, None)
            if callable(value):
                runtime_entrypoints[name] = _callable_identity(value)
    return _fingerprint(
        {
            "factory": _implementation_identity(factory),
            "concrete_type": (
                _implementation_identity(concrete_type) if concrete_type is not None else None
            ),
            "runtime_entrypoints": runtime_entrypoints,
        }
    )


def operator_instance_fingerprint(**identity: str) -> str:
    return _fingerprint(identity)


def _trace(
    semantic_op: str,
    backend: str,
    target: str,
    policy: OperatorResolutionPolicy,
    status: str,
    decisions: tuple[OperatorCapabilityDecision, ...],
    descriptor: Optional[OperatorBackendDescriptor] = None,
) -> OperatorResolutionTrace:
    observable = descriptor is not None and descriptor.is_strictly_observable
    return OperatorResolutionTrace(
        semantic_op,
        backend,
        target,
        policy.strict,
        status,
        (
            descriptor.backend_id
            if descriptor is not None and observable and status != "unsupported"
            else None
        ),
        descriptor.implementation_reference if descriptor else None,
        descriptor.descriptor_fingerprint if descriptor else None,
        decisions,
    )


def _decision(
    capability: str,
    requested: Any,
    supported: Any,
    passed: Optional[bool] = None,
) -> OperatorCapabilityDecision:
    passed = _supports(supported, requested) if passed is None else passed
    actionable = {
        "smoke_opt_in": "smoke backend use requires explicit opt-in",
        "strict_observability": "runtime-native implementation is not exactly observable",
        "fallback_policy": "strict resolution forbids declared or runtime fallback",
    }
    return OperatorCapabilityDecision(
        capability,
        requested,
        supported,
        passed,
        (
            f"{capability} is supported"
            if passed
            else actionable.get(capability, f"{capability} is unsupported")
        ),
    )


def _supports(supported: Any, requested: Any) -> bool:
    if isinstance(supported, str) and supported in {"*", "any"}:
        return True
    if isinstance(supported, Mapping):
        if not isinstance(requested, Mapping):
            return False
        wildcard = supported.get("*")
        return all(
            _supports(supported.get(key, wildcard), value)
            for key, value in requested.items()
            if key in supported or wildcard is not None
        ) and all(key in supported or wildcard is not None for key in requested)
    if isinstance(supported, (set, frozenset, tuple, list)):
        if isinstance(requested, (set, frozenset, tuple, list)):
            return all(any(_supports(item, value) for item in supported) for value in requested)
        return any(_supports(item, requested) for item in supported)
    return supported == requested


def _target_topology_capabilities(supported: Any, target: str) -> Any:
    if not isinstance(supported, Mapping):
        return supported
    targeted = any(key in supported for key in ("rollout", "training"))
    if not targeted:
        return supported
    return supported.get(target, supported.get("*"))


def _supports_complete_mapping(supported: Any, requested: Any) -> bool:
    if isinstance(supported, Mapping) and "*" not in supported:
        if not isinstance(requested, Mapping) or any(key not in requested for key in supported):
            return False
    return _supports(supported, requested)


def _resolve_policy(policy: _Policy, strict: Optional[bool]) -> OperatorResolutionPolicy:
    policy = policy or OperatorResolutionPolicy()
    return policy if strict is None else replace(policy, strict=strict)


def _load_factory(value: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(value):
        return value
    try:
        module_name, attribute = value.rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError, ModuleNotFoundError) as exc:
        raise OperatorInstantiationError(f"operator factory {value!r} is unavailable") from exc
    if not callable(factory):
        raise OperatorInstantiationError(f"operator factory {value!r} is not callable")
    return factory


def _reference(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _normalize_device(value: Any) -> str:
    value = str(value).strip().lower()
    if value.startswith("torch.device("):
        value = value.removeprefix("torch.device(").removesuffix(")").strip("'\"")
    if value.startswith("cuda:"):
        return "cuda"
    return {"gpu": "cuda", "hip": "rocm"}.get(value, value)


def _normalize_dtype(value: Any) -> str:
    value = str(value).strip().lower().replace("torch.", "")
    return {
        "fp32": "float32",
        "float": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
    }.get(value, value)


def _normalized_values(values: Iterable[Any], normalize: Callable[[Any], str]) -> frozenset[str]:
    return frozenset(value for item in values if (value := normalize(item).strip().lower()))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, _JsonRecord):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if callable(value):
        return _reference(value)
    return value


def _implementation_identity(value: Any) -> Mapping[str, Any]:
    reference = _reference(value)
    identity: dict[str, Any] = {
        "reference": reference,
        "kind": "class" if inspect.isclass(value) else "callable",
        "callable": _callable_identity(value),
        "module": _module_identity(getattr(value, "__module__", None)),
    }
    if inspect.isclass(value):
        identity["members"] = {
            name: _callable_identity(member)
            for name, raw_member in sorted(vars(value).items())
            if (member := _descriptor_callable(raw_member)) is not None
        }
    return identity


def _descriptor_callable(value: Any) -> Optional[Callable[..., Any]]:
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    elif isinstance(value, property):
        return None
    return value if callable(value) else None


def _callable_identity(value: Any) -> Mapping[str, Any]:
    if inspect.ismethod(value):
        value = value.__func__
    try:
        unwrapped = inspect.unwrap(value)
    except (TypeError, ValueError):
        unwrapped = value
    code = getattr(unwrapped, "__code__", None)
    try:
        source = inspect.getsource(unwrapped)
    except (OSError, TypeError):
        source = None
    identity: dict[str, Any] = {
        "reference": _reference(unwrapped),
        "source_sha256": (
            hashlib.sha256(source.encode("utf-8")).hexdigest() if source is not None else None
        ),
        "code_sha256": _code_fingerprint(code) if isinstance(code, CodeType) else None,
    }
    if isinstance(code, CodeType):
        identity["defaults"] = _code_value(getattr(unwrapped, "__defaults__", None))
        identity["keyword_defaults"] = _code_value(getattr(unwrapped, "__kwdefaults__", None))
    return identity


def _code_fingerprint(code: CodeType) -> str:
    return _fingerprint(
        {
            "bytecode": code.co_code.hex(),
            "constants": tuple(_code_value(value) for value in code.co_consts),
            "names": code.co_names,
            "variables": code.co_varnames,
            "free_variables": code.co_freevars,
            "cell_variables": code.co_cellvars,
            "positional_arguments": code.co_argcount,
            "positional_only_arguments": code.co_posonlyargcount,
            "keyword_only_arguments": code.co_kwonlyargcount,
            "flags": code.co_flags,
        }
    )


def _code_value(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"nested_code_sha256": _code_fingerprint(value)}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(key): _code_value(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_code_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_code_value(item) for item in value), key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": _reference(type(value)), "repr": repr(value)}


def _module_identity(module_name: Optional[str]) -> Optional[Mapping[str, Any]]:
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return {"name": module_name, "content_sha256": None}
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return {"name": module_name, "content_sha256": None}
    path = Path(module_file)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {"name": module_name, "content_sha256": None}
    return {
        "name": module_name,
        "content_sha256": digest.hexdigest(),
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
