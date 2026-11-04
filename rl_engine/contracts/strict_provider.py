# SPDX-License-Identifier: Apache-2.0
"""Versioned, immutable data contracts for strict provider execution.

This module describes evidence. It neither selects providers nor asserts that
matching identities prove numerical parity. All nested data is copied/frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType

SCHEMA_VERSION = "rl_kernel.strict_provider.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDERS = {"unknown", "unresolved", "todo", "tbd", "n/a", "???"}
_PHASES = ("prefill", "chunked_prefill", "decode", "forward", "backward")
_DTYPES = (
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "fp32",
    "float32",
    "fp64",
    "float64",
    "int8",
    "uint8",
    "int32",
    "int64",
    "bool",
)


class StrictProviderError(ValueError):
    """Stable machine-readable rejection with a useful field-level diagnosis."""

    def __init__(self, code, message, field=None, expected=None, observed=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.expected = expected
        self.observed = observed

    def to_dict(self):
        result = {"code": self.code, "message": self.message}
        for name in ("field", "expected", "observed"):
            value = getattr(self, name)
            if value is not None:
                result[name] = _plain(value)
        return result


def _error(field, message, expected=None, observed=None):
    raise StrictProviderError("invalid_contract", message, field, expected, observed)


def _plain(value):
    if isinstance(value, _Record):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_value(value, path="$"):
    """Accept only unambiguous JSON data; preserve semantically ordered lists."""
    if isinstance(value, _Record):
        value = value.to_dict()
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                _error(path, "Object keys must be nonempty strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if value is None:
        _error(path, "Null is not an identity; omit optional fields or use an explicit NA record")
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _error(path, "Non-finite numbers are not valid contract values")
        return value
    if isinstance(value, str):
        if (
            not value.strip()
            or value.strip().lower() in _PLACEHOLDERS
            or (value.startswith("<") and value.endswith(">"))
        ):
            _error(path, "Empty or unresolved identity value", observed=value)
        return value
    _error(path, "Only JSON values are accepted", observed=type(value).__name__)


def canonical_json(value) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_json(text: str):
    """Parse contract files without silently accepting duplicate object keys."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _error(key, "Duplicate JSON key")
            result[key] = value
        return result

    try:
        return _json_value(json.loads(text, object_pairs_hook=pairs))
    except (json.JSONDecodeError, TypeError) as exc:
        raise StrictProviderError("invalid_contract", "Invalid JSON") from exc


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _fields(data, required, optional=(), path="$"):
    if not isinstance(data, Mapping):
        _error(path, "Expected an object")
    missing = set(required) - data.keys()
    extra = data.keys() - set(required) - set(optional)
    if missing:
        name = sorted(missing)[0]
        _error(f"{path}.{name}", "Missing required field")
    if extra:
        name = sorted(extra)[0]
        _error(f"{path}.{name}", "Unknown field")


def _object(value, path, required=()):
    if not isinstance(value, dict):
        _error(path, "Expected an object")
    for key in required:
        if key not in value:
            _error(f"{path}.{key}", "Missing required field")


def _text(value, path):
    if not isinstance(value, str):
        _error(path, "Expected a string")


def _integer(value, path, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(path, f"Expected an integer >= {minimum}")


def _enum(value, allowed, path):
    if value not in allowed:
        _error(path, "Unsupported value", list(allowed), value)


def _digest(value, path):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _error(path, "Expected a lowercase SHA-256 digest")


def _list(value, path, *, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        _error(path, "Expected a nonempty list" if nonempty else "Expected a list")


def _dtype(value, path):
    if isinstance(value, dict):
        if not value:
            _error(path, "Dtype mapping cannot be empty")
        for key, item in value.items():
            _dtype(item, f"{path}.{key}")
    else:
        _enum(value, _DTYPES, path)


def _description(value, path):
    if not isinstance(value, (str, dict, list)) or not value:
        _error(path, "Expected a nonempty description or explicit not-applicable record")


def _identity_sections(data):
    _object(
        data["model"],
        "$.model",
        ("model_id", "revision", "config_sha256", "weight_index_sha256", "shard_inventory_sha256"),
    )
    for key in ("config_sha256", "weight_index_sha256", "shard_inventory_sha256"):
        _digest(data["model"][key], f"$.model.{key}")
    for key in ("model_id", "revision"):
        _text(data["model"][key], f"$.model.{key}")
    _object(
        data["platform"],
        "$.platform",
        ("kind", "gpu_arch", "torch_build", "hip_runtime", "driver_identity", "triton_build"),
    )
    _enum(data["platform"]["kind"], ("rocm", "cuda", "cpu"), "$.platform.kind")
    _text(data["platform"]["gpu_arch"], "$.platform.gpu_arch")
    for key in ("torch_build", "hip_runtime", "driver_identity", "triton_build"):
        _description(data["platform"][key], f"$.platform.{key}")
    _object(
        data["execution"],
        "$.execution",
        (
            "topology",
            "activation_dtype",
            "graph_mode",
            "tuning_policy",
            "fallback_policy",
            "runtime_settings",
        ),
    )
    execution = data["execution"]
    _object(execution["topology"], "$.execution.topology", ("tp", "ep", "cp", "pp"))
    for key in ("tp", "ep", "cp", "pp"):
        _integer(execution["topology"][key], f"$.execution.topology.{key}", 1)
    _enum(execution["fallback_policy"], ("error",), "$.execution.fallback_policy")
    _enum(execution["tuning_policy"], ("fixed",), "$.execution.tuning_policy")
    _enum(execution["graph_mode"], ("eager", "graph"), "$.execution.graph_mode")
    _dtype(execution["activation_dtype"], "$.execution.activation_dtype")
    _object(execution["runtime_settings"], "$.execution.runtime_settings")


class _Record(Mapping):
    __slots__ = ("_data",)

    def __init__(self, data):
        normalized = _json_value(data)
        self._validate(normalized)
        object.__setattr__(self, "_data", _freeze(normalized))

    def __setattr__(self, name, value):
        raise AttributeError("Contract records are immutable")

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def to_dict(self):
        return _plain(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @staticmethod
    def _validate(data):
        raise NotImplementedError


class KernelPlan(_Record):
    """A fully resolved numerical plan, not a backend preference.

    v1 alignment permits a different outer implementation wrapper only. Actual
    entrypoints, artifact identities, IO, arithmetic and launch settings remain
    part of alignment; heterogeneous paths need another qualified contract.
    """

    @staticmethod
    def _validate(data):
        _fields(
            data,
            ("node", "phase", "implementation", "entrypoint", "code", "io", "numerics", "launch"),
            ("schema_version", "plan_id", "alignment_id"),
        )
        data.setdefault("schema_version", SCHEMA_VERSION)
        _enum(data["schema_version"], (SCHEMA_VERSION,), "$.schema_version")
        for key in ("node", "phase", "implementation", "entrypoint"):
            _text(data[key], f"$.{key}")
        _enum(data["phase"], _PHASES, "$.phase")
        _object(
            data["code"],
            "$.code",
            (
                "source_inventory_sha256",
                "binary_inventory_sha256",
                "compiled_kernel_id",
                "compile_key",
            ),
        )
        for key in ("source_inventory_sha256", "binary_inventory_sha256"):
            _digest(data["code"][key], f"$.code.{key}")
        for key in ("compiled_kernel_id", "compile_key"):
            _description(data["code"][key], f"$.code.{key}")
        _object(
            data["io"],
            "$.io",
            ("input_dtype", "output_dtype", "layout", "shape_class", "workspace_policy"),
        )
        for key in ("input_dtype", "output_dtype"):
            _dtype(data["io"][key], f"$.io.{key}")
        for key in ("layout", "shape_class", "workspace_policy"):
            _description(data["io"][key], f"$.io.{key}")
        _object(
            data["numerics"],
            "$.numerics",
            (
                "accumulator_dtype",
                "split_policy",
                "split_count",
                "reduction_spec",
                "cast_spec",
                "atomic_policy",
            ),
        )
        _integer(data["numerics"]["split_count"], "$.numerics.split_count", 1)
        _dtype(data["numerics"]["accumulator_dtype"], "$.numerics.accumulator_dtype")
        for key in ("reduction_spec", "cast_spec", "atomic_policy"):
            _description(data["numerics"][key], f"$.numerics.{key}")
        _enum(data["numerics"]["split_policy"], ("disabled", "fixed"), "$.numerics.split_policy")
        if data["numerics"]["split_policy"] == "disabled" and data["numerics"]["split_count"] != 1:
            _error("$.numerics.split_count", "Disabled splitting requires one split")
        _object(data["launch"], "$.launch", ("tile_config", "graph_state", "tuning"))
        _enum(data["launch"]["graph_state"], ("eager", "graph"), "$.launch.graph_state")
        _enum(data["launch"]["tuning"], ("fixed",), "$.launch.tuning")
        _description(data["launch"]["tile_config"], "$.launch.tile_config")
        base = {key: value for key, value in data.items() if key not in ("plan_id", "alignment_id")}
        alignment = {key: value for key, value in base.items() if key != "implementation"}
        for key, expected in (
            ("plan_id", fingerprint(base)),
            ("alignment_id", fingerprint(alignment)),
        ):
            if key in data and data[key] != expected:
                _error(f"$.{key}", "Supplied identity does not match content", expected, data[key])
            data[key] = expected


class ProviderLock(_Record):
    @staticmethod
    def _validate(data):
        _fields(
            data,
            ("schema_version", "profile_id", "scope", "model", "platform", "execution", "nodes"),
            ("lock_id",),
        )
        _enum(data["schema_version"], (SCHEMA_VERSION,), "$.schema_version")
        for key in ("profile_id", "scope"):
            _text(data[key], f"$.{key}")
        _identity_sections(data)
        _object(data["nodes"], "$.nodes")
        if not data["nodes"]:
            _error("$.nodes", "At least one node is required")
        for node, spec in data["nodes"].items():
            path = f"$.nodes.{node}"
            _fields(spec, ("semantics_ref", "bindings", "allowed_plans", "alignment"), path=path)
            _text(spec["semantics_ref"], f"{path}.semantics_ref")
            _object(spec["bindings"], f"{path}.bindings")
            if not spec["bindings"]:
                _error(f"{path}.bindings", "At least one role binding is required")
            _object(spec["allowed_plans"], f"{path}.allowed_plans")
            if spec["bindings"].keys() != spec["allowed_plans"].keys():
                _error(f"{path}.allowed_plans", "Allowed plan roles must equal binding roles")
            for role, binding in spec["bindings"].items():
                _enum(role, ("rollout", "replay"), f"{path}.bindings.{role}")
                _object(binding, f"{path}.bindings.{role}", ("implementation", "artifacts_ref"))
                _text(binding["implementation"], f"{path}.bindings.{role}.implementation")
                _digest(binding["artifacts_ref"], f"{path}.bindings.{role}.artifacts_ref")
                plans = spec["allowed_plans"][role]
                _list(plans, f"{path}.allowed_plans.{role}")
                for i, plan_id in enumerate(plans):
                    _digest(plan_id, f"{path}.allowed_plans.{role}[{i}]")
                if len(plans) != len(set(plans)):
                    _error(f"{path}.allowed_plans.{role}", "Duplicate allowed plan")
            _fields(spec["alignment"], ("policy",), path=f"{path}.alignment")
            _enum(spec["alignment"]["policy"], ("identical_plan",), f"{path}.alignment.policy")
        base = {key: value for key, value in data.items() if key != "lock_id"}
        expected = fingerprint(base)
        if "lock_id" in data and data["lock_id"] != expected:
            _error(
                "$.lock_id",
                "Supplied lock identity does not match content",
                expected,
                data["lock_id"],
            )
        data["lock_id"] = expected


class WorkerObservation(_Record):
    @staticmethod
    def _validate(data):
        _fields(
            data,
            ("worker_id", "role", "model", "platform", "execution", "bindings"),
            ("rank", "diagnostics", "weight_snapshot_id"),
        )
        _text(data["worker_id"], "$.worker_id")
        _enum(data["role"], ("rollout", "replay"), "$.role")
        _identity_sections(data)
        _object(data["bindings"], "$.bindings")
        if not data["bindings"]:
            _error("$.bindings", "At least one observed binding is required")
        for node, binding in data["bindings"].items():
            _object(binding, f"$.bindings.{node}", ("implementation", "artifacts_ref"))
            _text(binding["implementation"], f"$.bindings.{node}.implementation")
            _digest(binding["artifacts_ref"], f"$.bindings.{node}.artifacts_ref")
        if "rank" in data:
            _integer(data["rank"], "$.rank")
        if "diagnostics" in data:
            _object(data["diagnostics"], "$.diagnostics")


class CallContext(_Record):
    @staticmethod
    def _validate(data):
        required = (
            "case_id",
            "call_id",
            "node",
            "phase",
            "role",
            "worker_id",
            "run_id",
            "attempt",
            "weight_snapshot_id",
            "layer_id",
        )
        _fields(data, required, ("metadata",))
        for key in required:
            if key != "layer_id":
                _text(data[key], f"$.{key}")
        if isinstance(data["layer_id"], int):
            _integer(data["layer_id"], "$.layer_id")
        else:
            _text(data["layer_id"], "$.layer_id")
        _enum(data["role"], ("rollout", "replay"), "$.role")
        _enum(data["phase"], _PHASES, "$.phase")
        if "metadata" in data:
            _object(data["metadata"], "$.metadata")


class ScopeSpec(_Record):
    @staticmethod
    def _validate(data):
        _fields(data, ("scope_id", "kind", "required_nodes", "inventory_fingerprint"))
        _text(data["scope_id"], "$.scope_id")
        _enum(data["kind"], ("component", "full_model"), "$.kind")
        _list(data["required_nodes"], "$.required_nodes")
        for i, node in enumerate(data["required_nodes"]):
            _text(node, f"$.required_nodes[{i}]")
        if len(set(data["required_nodes"])) != len(data["required_nodes"]):
            _error("$.required_nodes", "Duplicate required node")
        _digest(data["inventory_fingerprint"], "$.inventory_fingerprint")


class CaseSpec(_Record):
    """Predeclared execution coverage and shared input/state identifiers.

    Numerical output policies belong to the operator/model validation harness.
    Adapters record comparison_context in CallContext.metadata so provenance
    can be checked without reading or comparing numerical output tensors.
    """

    @staticmethod
    def _validate(data):
        _fields(data, ("case_id", "comparison_context", "required_calls"))
        _text(data["case_id"], "$.case_id")
        _object(data["comparison_context"], "$.comparison_context")
        if not data["comparison_context"]:
            _error("$.comparison_context", "Comparison context cannot be empty")
        _list(data["required_calls"], "$.required_calls")
        identities = set()
        for i, call in enumerate(data["required_calls"]):
            path = f"$.required_calls[{i}]"
            _fields(
                call, ("node", "phase", "layer_id", "call_id", "role", "rank", "count"), path=path
            )
            for key in ("node", "phase", "call_id"):
                _text(call[key], f"{path}.{key}")
            _enum(call["phase"], _PHASES, f"{path}.phase")
            if isinstance(call["layer_id"], int):
                _integer(call["layer_id"], f"{path}.layer_id")
            else:
                _text(call["layer_id"], f"{path}.layer_id")
            _enum(call["role"], ("rollout", "replay"), f"{path}.role")
            _integer(call["rank"], f"{path}.rank")
            _integer(call["count"], f"{path}.count", 1)
            identity = canonical_json({key: value for key, value in call.items() if key != "count"})
            if identity in identities:
                _error(path, "Duplicate required call; use its count field")
            identities.add(identity)


class SnapshotActivationReceipt(_Record):
    """Load-boundary evidence, distinct from a disk checkpoint inventory.

    This record cannot make arbitrary in-process Python assertions trustworthy.
    Engine adapters must create it after reading the actual loaded shards and
    derived caches, using the identity collector's verification helper.
    """

    @staticmethod
    def _validate(data):
        required = (
            "snapshot_id",
            "generation",
            "worker_id",
            "expected_shards",
            "verified_shards",
            "required_refresh",
            "refreshed_components",
            "content_digest",
            "completed",
            "evidence_kind",
        )
        _fields(data, required, ("receipt_id",))
        _text(data["snapshot_id"], "$.snapshot_id")
        _text(data["worker_id"], "$.worker_id")
        _integer(data["generation"], "$.generation")
        for key in (
            "expected_shards",
            "verified_shards",
            "required_refresh",
            "refreshed_components",
        ):
            _list(data[key], f"$.{key}", nonempty=key == "expected_shards")
            for i, name in enumerate(data[key]):
                _text(name, f"$.{key}[{i}]")
            if len(set(data[key])) != len(data[key]):
                _error(f"$.{key}", "Duplicate receipt component")
        _digest(data["content_digest"], "$.content_digest")
        if not isinstance(data["completed"], bool):
            _error("$.completed", "Expected a boolean")
        _enum(data["evidence_kind"], ("worker_load", "disk_checkpoint"), "$.evidence_kind")
        if set(data["verified_shards"]) - set(data["expected_shards"]):
            _error("$.verified_shards", "Unexpected verified shard")
        if set(data["refreshed_components"]) - set(data["required_refresh"]):
            _error("$.refreshed_components", "Unexpected refreshed component")
        if data["completed"] and (
            set(data["expected_shards"]) != set(data["verified_shards"])
            or set(data["required_refresh"]) != set(data["refreshed_components"])
        ):
            _error("$.completed", "Incomplete load or refresh cannot be marked completed")
        base = {key: value for key, value in data.items() if key != "receipt_id"}
        expected = fingerprint(base)
        if "receipt_id" in data and data["receipt_id"] != expected:
            _error("$.receipt_id", "Receipt identity does not match content")
        data["receipt_id"] = expected

    @property
    def load_complete(self):
        return self.completed and set(self.expected_shards) == set(self.verified_shards)

    @property
    def cache_refresh_complete(self):
        return self.completed and set(self.required_refresh) == set(self.refreshed_components)

    def validate(self):
        self._validate(self.to_dict())
        if not self.completed:
            raise StrictProviderError("snapshot_inactive", "Snapshot load/refresh is incomplete")
        return self
