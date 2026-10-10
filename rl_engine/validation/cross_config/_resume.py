# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Private validation for append-only attempt resume."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from rl_engine.contracts.numerical import resolve_logprob_threshold, tolerance_contract_fingerprint
from rl_engine.runtime.semantic_registry import OperatorInstanceProvenance
from rl_engine.validation.cross_config._execution import (
    canonical_fingerprint,
    json_safe,
    torch_dtype,
)
from rl_engine.validation.cross_config._json import strict_json_loads
from rl_engine.validation.cross_config._provenance import (
    PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT,
    concrete_scorer_spec,
    effective_runtime_status,
    factory_options_fingerprint,
    target_factory_options,
)
from rl_engine.validation.cross_config.artifacts import REQUIRED_CASE_ARTIFACTS
from rl_engine.validation.cross_config.comparison import recompute_mismatch_mask
from rl_engine.validation.cross_config.runtime import RuntimeMaterialization
from rl_engine.validation.cross_config.schema import (
    CanonicalScoringBatch,
    ExperimentCase,
    ScorerSpec,
    ScoreSide,
)

_COMPLETE_KEYS = frozenset(
    {
        "schema_version",
        "case_id",
        "attempt_id",
        "status",
        "comparable",
        "passed",
        "active_token_count",
        "mismatch_count",
        "worst_token_index",
        "max_abs_diff",
        "rollout_backend",
        "training_backend",
        "execution_fingerprint",
        "environment_fingerprint",
        "runner_implementation_fingerprint",
        "artifact_sha256",
    }
)


def completed_attempt_matches(
    attempt_dir: Path,
    case: ExperimentCase,
    batch: CanonicalScoringBatch,
    *,
    materialization: RuntimeMaterialization,
    specs: Mapping[str, ScorerSpec],
    instance_provenance: Mapping[str, OperatorInstanceProvenance],
    operator_factory_options: Optional[Mapping[str | ScoreSide, Mapping[str, Any]]],
    model_state_fingerprints: Mapping[str, Optional[str]],
    scorer_implementation_fingerprints: Mapping[str, str],
    environment: Mapping[str, Any],
    execution_fingerprint: str,
) -> bool:
    try:
        identity = read_json_object(attempt_dir / "identity.json")
        requested = read_json_object(attempt_dir / "requested.json")
        actual = read_json_object(attempt_dir / "actual.json")
        marker = read_json_object(attempt_dir / "COMPLETE")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not (
        set(marker) == _COMPLETE_KEYS
        and identity.get("schema_version") == "cross_config.identity_envelope.v1"
        and requested.get("schema_version") == "cross_config.requested.v1"
        and actual.get("schema_version") == "cross_config.actual.v1"
        and marker.get("schema_version") == "cross_config.complete.v1"
        and actual.get("runner_implementation_fingerprint")
        == PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT
        and actual.get("environment") == environment
        and actual.get("environment_fingerprint") == canonical_fingerprint(environment)
        and marker.get("runner_implementation_fingerprint")
        == PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT
        and marker.get("environment_fingerprint") == canonical_fingerprint(environment)
        and isinstance(marker.get("artifact_sha256"), Mapping)
        and set(marker["artifact_sha256"]) == set(REQUIRED_CASE_ARTIFACTS)
    ):
        return False
    if not (
        identity.get("case_id") == case.case_id
        and identity.get("identity") == batch.identity.to_dict()
        and requested.get("case") == case.to_dict()
        and marker.get("execution_fingerprint") == execution_fingerprint
        and actual.get("execution_fingerprint") == execution_fingerprint
        and marker.get("rollout_backend") == instance_provenance["rollout"].backend_id
        and marker.get("training_backend") == instance_provenance["training"].backend_id
    ):
        return False

    runtime = materialization.provenance.to_dict()
    effective_status = effective_runtime_status(materialization).value
    score_tensors: dict[str, Mapping[str, torch.Tensor]] = {}
    for target in ("rollout", "training"):
        prior = actual.get(target)
        if not isinstance(prior, Mapping):
            return False
        instance = instance_provenance[target]
        options = target_factory_options(operator_factory_options, target)
        expected_operator = {
            "backend_id": instance.backend_id,
            "descriptor_fingerprint": instance.descriptor_fingerprint,
            "implementation_fingerprint": instance.implementation_fingerprint,
            "instance_fingerprint": instance.instance_fingerprint,
            "concrete_implementation": instance.concrete_implementation,
            "factory_options": json_safe(options),
            "factory_options_fingerprint": factory_options_fingerprint(options),
        }
        prior_actual = prior.get("actual")
        if not isinstance(prior_actual, Mapping):
            return False
        if any(prior_actual.get(key) != value for key, value in runtime["actual"].items()):
            return False
        if prior_actual.get("operators", {}).get("selected_logprob") != expected_operator:
            return False
        if prior_actual.get("model_state_fingerprint") != model_state_fingerprints[target]:
            return False
        if (
            prior_actual.get("scorer_implementation_fingerprint")
            != scorer_implementation_fingerprints[target]
        ):
            return False
        expected_implementation = hashlib.sha256(
            (
                f"{materialization.provenance.implementation_fingerprint}:"
                f"{instance.instance_fingerprint}"
            ).encode("utf-8")
        ).hexdigest()
        for key, expected in (
            ("requested", runtime["requested"]),
            ("normalized", runtime["normalized"]),
            ("materialized", runtime["materialized"]),
            ("status", effective_status),
            (
                "construction_fingerprint",
                materialization.provenance.construction_fingerprint,
            ),
            (
                "distributed_context_fingerprint",
                materialization.provenance.distributed_context_fingerprint,
            ),
            ("process_fingerprint", materialization.provenance.process_fingerprint),
            ("implementation_fingerprint", expected_implementation),
            ("world_size", specs[target].world_size),
        ):
            if prior.get(key) != expected:
                return False
        try:
            score_payload = _load_resume_tensor_bundle(attempt_dir / f"score_{target}.pt")
            score_artifact = score_payload["metadata"]["score_artifact"]
            prior_scorer = score_artifact["scorer"]
        except (OSError, KeyError, TypeError, RuntimeError, ValueError):
            return False
        expected_scorer = concrete_scorer_spec(specs[target], instance).to_dict()
        if prior_scorer != expected_scorer:
            return False
        if (
            score_artifact.get("schema_version") != "cross_config.score_artifact.v1"
            or score_artifact.get("case_id") != case.case_id
            or score_artifact.get("attempt_id") != attempt_dir.name
            or score_artifact.get("side") != target
            or score_artifact.get("identity") != batch.identity.to_dict()
            or score_artifact.get("provenance") != prior
        ):
            return False
        tensors = score_payload["tensors"]
        selected = tensors.get("selected_logprobs")
        active_mask = tensors.get("active_mask")
        expected_dtype = torch_dtype(specs[target].dtype)
        if (
            not isinstance(selected, torch.Tensor)
            or not isinstance(active_mask, torch.Tensor)
            or selected.shape != batch.input_ids.shape
            or active_mask.shape != batch.input_ids.shape
            or selected.dtype != expected_dtype
            or active_mask.dtype != torch.bool
            or not torch.equal(active_mask, batch.active_mask.to(device="cpu"))
        ):
            return False
        metadata = score_payload["metadata"]
        if (
            metadata.get("case_id") != case.case_id
            or metadata.get("attempt_id") != attempt_dir.name
            or metadata.get("side") != target
        ):
            return False
        score_tensors[target] = tensors
    return _resume_comparison_matches(
        attempt_dir,
        case,
        batch,
        marker,
        score_tensors,
        specs,
    )


def _load_resume_tensor_bundle(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid tensor bundle schema: {path}")
    tensors = payload.get("tensors")
    metadata = payload.get("metadata")
    if not isinstance(tensors, Mapping) or not all(
        isinstance(tensor, torch.Tensor) for tensor in tensors.values()
    ):
        raise ValueError(f"invalid tensor bundle payload: {path}")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"invalid tensor bundle metadata: {path}")
    return payload


def _resume_comparison_matches(
    attempt_dir: Path,
    case: ExperimentCase,
    batch: CanonicalScoringBatch,
    marker: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, torch.Tensor]],
    specs: Mapping[str, ScorerSpec],
) -> bool:
    try:
        comparison = read_json_object(attempt_dir / "comparison.json")
        token_bundle = _load_resume_tensor_bundle(attempt_dir / "token_diffs.pt")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    required_token_keys = {
        "rollout_logprobs",
        "training_logprobs",
        "active_mask",
        "absolute_diff",
        "mismatch_mask",
    }
    token_tensors = token_bundle["tensors"]
    if not required_token_keys.issubset(token_tensors):
        return False
    rollout = scores["rollout"]["selected_logprobs"]
    training = scores["training"]["selected_logprobs"]
    active_mask = batch.active_mask.to(device="cpu", dtype=torch.bool)
    if not bool(torch.isfinite(rollout[active_mask]).all().item()) or not bool(
        torch.isfinite(training[active_mask]).all().item()
    ):
        return False
    rollout_threshold = resolve_logprob_threshold(specs["rollout"].dtype)
    training_threshold = resolve_logprob_threshold(specs["training"].dtype)
    if rollout_threshold != training_threshold:
        return False
    fixed_threshold = rollout_threshold
    rollout = rollout.masked_fill(~active_mask, 0.0)
    training = training.masked_fill(~active_mask, 0.0)
    absolute_diff = torch.abs(training - rollout)
    mismatch_mask = recompute_mismatch_mask(
        rollout,
        training,
        active_mask,
        fixed_threshold,
    )
    expected_tensors = {
        "rollout_logprobs": rollout,
        "training_logprobs": training,
        "active_mask": active_mask,
        "absolute_diff": absolute_diff,
        "mismatch_mask": mismatch_mask,
    }
    if any(
        token_tensors[name].dtype != expected.dtype
        or token_tensors[name].shape != expected.shape
        or not torch.equal(token_tensors[name], expected)
        for name, expected in expected_tensors.items()
    ):
        return False
    token_metadata = token_bundle["metadata"]
    active_count = int(active_mask.sum().item())
    if active_count == 0:
        return False
    mismatch_count = int(mismatch_mask.sum().item())
    passed = mismatch_count == 0
    status = "pass" if passed else "fail"
    if (
        token_metadata.get("case_id") != case.case_id
        or token_metadata.get("attempt_id") != attempt_dir.name
        or token_metadata.get("status") != status
        or token_metadata.get("fixed_threshold") != fixed_threshold
    ):
        return False
    diagnostics = _comparison_diagnostics(
        rollout,
        training,
        active_mask,
        absolute_diff,
        mismatch_count,
    )
    expected_comparison = {
        "schema_version": "cross_config.alignment_result.v1",
        "case_id": case.case_id,
        "attempt_id": attempt_dir.name,
        "status": status,
        "comparable": True,
        "passed": passed,
        "active_token_count": active_count,
        "mismatch_count": mismatch_count,
        "contract_fingerprint": tolerance_contract_fingerprint(),
        "fixed_threshold": fixed_threshold,
        "identity_errors": [],
        "artifact_errors": [],
        "diagnostics": diagnostics,
        "token_artifact": {
            **{name: _serialized_tensor(tensor) for name, tensor in expected_tensors.items()},
            "fixed_threshold": fixed_threshold,
            "schema_version": "cross_config.token_comparison.v1",
        },
    }
    if comparison != expected_comparison:
        return False
    if (
        marker.get("case_id") != case.case_id
        or marker.get("attempt_id") != attempt_dir.name
        or marker.get("status") != status
        or marker.get("comparable") is not True
        or marker.get("passed") is not passed
        or marker.get("active_token_count") != active_count
        or marker.get("mismatch_count") != mismatch_count
        or marker.get("max_abs_diff") != diagnostics["max_abs_diff"]
        or marker.get("worst_token_index") != diagnostics["worst_token_index"]
    ):
        return False
    return True


def _comparison_diagnostics(
    rollout: torch.Tensor,
    training: torch.Tensor,
    active_mask: torch.Tensor,
    absolute_diff: torch.Tensor,
    mismatch_count: int,
) -> dict[str, Any]:
    active_diff = absolute_diff[active_mask].float()
    delta = (training[active_mask] - rollout[active_mask]).float()
    worst_index = int(torch.argmax(active_diff).item())
    coordinates = torch.nonzero(active_mask, as_tuple=False)
    worst_token = [int(item) for item in coordinates[worst_index].tolist()]
    approximate_kl = torch.exp(delta.double()) - delta.double() - 1.0
    approximate_kl_mean = _finite_float(approximate_kl.mean())
    active_count = int(active_diff.numel())
    return {
        "mean_abs_diff": _finite_float(active_diff.mean()),
        "p95_abs_diff": _finite_float(torch.quantile(active_diff, 0.95)),
        "p99_abs_diff": _finite_float(torch.quantile(active_diff, 0.99)),
        "max_abs_diff": _finite_float(active_diff.max()),
        "mismatch_ratio": mismatch_count / active_count,
        "approximate_kl_mean": approximate_kl_mean,
        "approximate_kl_finite": approximate_kl_mean is not None,
        "worst_token_index": worst_token,
    }


def _finite_float(value: torch.Tensor) -> Optional[float]:
    result = float(value.item())
    return result if math.isfinite(result) else None


def _serialized_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def read_json_object(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
