# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Private execution identity and provenance construction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from rl_engine.runtime.semantic_registry import OperatorInstanceProvenance, OperatorResolution
from rl_engine.validation.cross_config._execution import (
    OperatorExecutionError,
    PairedRunnerError,
    canonical_fingerprint,
    device_type,
    json_safe,
)
from rl_engine.validation.cross_config.runtime import RuntimeMaterialization
from rl_engine.validation.cross_config.schema import (
    MaterializationStatus,
    RuntimeProvenance,
    ScoreArtifact,
    ScorerSpec,
    ScoreSide,
)

PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT = "cross_config.paired_runner.v2"


def effective_runtime_status(
    materialization: RuntimeMaterialization,
) -> MaterializationStatus:
    """Aggregate runtime status after exact resolution supersedes logp readback."""

    statuses = [
        application.status
        for application in materialization.applications
        if not (
            application.path == "logp.backend"
            and application.status is MaterializationStatus.UNOBSERVABLE
        )
    ]
    precedence = (
        MaterializationStatus.ERROR,
        MaterializationStatus.UNSUPPORTED,
        MaterializationStatus.UNOBSERVABLE,
        MaterializationStatus.FALLBACK,
        MaterializationStatus.APPLIED,
    )
    return next(
        (status for status in precedence if status in statuses),
        MaterializationStatus.APPLIED,
    )


def side_provenance(
    base: RuntimeProvenance,
    resolution: OperatorResolution,
    instance: OperatorInstanceProvenance,
    child_payload: Mapping[str, Any],
    spec: ScorerSpec,
    *,
    status: MaterializationStatus,
    factory_options: Mapping[str, Any],
    model_state_fingerprint: Optional[str],
    scorer_implementation_fingerprint: str,
) -> RuntimeProvenance:
    payload = base.to_dict()
    actual = dict(payload["actual"])
    actual["operators"] = {
        "selected_logprob": {
            "backend_id": instance.backend_id,
            "descriptor_fingerprint": instance.descriptor_fingerprint,
            "implementation_fingerprint": instance.implementation_fingerprint,
            "instance_fingerprint": instance.instance_fingerprint,
            "concrete_implementation": instance.concrete_implementation,
            "factory_options": json_safe(factory_options),
            "factory_options_fingerprint": factory_options_fingerprint(factory_options),
        }
    }
    actual["model_state_fingerprint"] = model_state_fingerprint
    actual["scorer_implementation_fingerprint"] = scorer_implementation_fingerprint
    evidence = dict(payload["evidence"])
    evidence.update(
        {
            "operator_resolution": resolution.to_dict(),
            "operator_instance": instance.to_dict(),
            "operator_factory_options": json_safe(factory_options),
            "scoring_guard": json_safe(child_payload.get("guard_evidence", {})),
            "rank_metadata": [
                json_safe(rank.get("metadata", {}))
                for rank in child_payload.get("ranks", ())
                if isinstance(rank, Mapping)
            ],
            "model_state_fingerprint": model_state_fingerprint,
            "scorer_implementation_fingerprint": scorer_implementation_fingerprint,
        }
    )
    implementation_fingerprint = hashlib.sha256(
        f"{base.implementation_fingerprint}:{instance.instance_fingerprint}".encode("utf-8")
    ).hexdigest()
    return RuntimeProvenance(
        requested=payload["requested"],
        normalized=payload["normalized"],
        materialized=payload["materialized"],
        actual=actual,
        status=status,
        construction_fingerprint=base.construction_fingerprint,
        distributed_context_fingerprint=base.distributed_context_fingerprint,
        process_fingerprint=base.process_fingerprint,
        implementation_fingerprint=implementation_fingerprint,
        evidence=evidence,
        rank=0,
        world_size=spec.world_size,
    )


def concrete_scorer_spec(
    spec: ScorerSpec,
    instance: OperatorInstanceProvenance,
) -> ScorerSpec:
    overrides = dict(spec.operator_overrides)
    overrides["selected_logprob"] = instance.backend_id
    return replace(spec, operator_overrides=overrides)


def score_metadata(artifact: ScoreArtifact) -> dict[str, Any]:
    value = artifact.to_dict()
    value.pop("selected_logprobs", None)
    value.pop("active_mask", None)
    return {
        "case_id": artifact.case_id,
        "attempt_id": artifact.attempt_id,
        "side": artifact.side.value,
        "score_artifact": value,
    }


def execution_fingerprint(
    materialization: RuntimeMaterialization,
    *,
    specs: Mapping[str, ScorerSpec],
    instance_provenance: Mapping[str, OperatorInstanceProvenance],
    operator_factory_options: Optional[Mapping[str | ScoreSide, Mapping[str, Any]]],
    model_state_fingerprints: Mapping[str, Optional[str]],
    scorer_implementation_fingerprints: Mapping[str, str],
    environment: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": "cross_config.execution_identity.v1",
        "runner_implementation_fingerprint": PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT,
        "environment": environment,
        "materialized_case": materialization.materialized_case.to_dict(),
        "runtime_provenance": materialization.provenance.to_dict(),
        "runtime_binding": materialization.binding.to_dict(),
        "applications": [application.to_dict() for application in materialization.applications],
        "targets": {
            target: {
                "scorer": concrete_scorer_spec(
                    specs[target],
                    instance_provenance[target],
                ).to_dict(),
                "operator_instance": instance_provenance[target].to_dict(),
                "operator_factory_options": json_safe(
                    target_factory_options(operator_factory_options, target)
                ),
                "model_state_fingerprint": model_state_fingerprints[target],
                "scorer_implementation_fingerprint": (scorer_implementation_fingerprints[target]),
            }
            for target in ("rollout", "training")
        },
    }
    return canonical_fingerprint(payload)


def mapping_target(mapping: Mapping[str | ScoreSide, Any], target: str) -> Any:
    if target in mapping:
        return mapping[target]
    side = ScoreSide(target)
    return mapping.get(side)


def target_factory_options(
    options: Optional[Mapping[str | ScoreSide, Mapping[str, Any]]],
    target: str,
) -> Mapping[str, Any]:
    if options is None:
        return {}
    value = mapping_target(options, target)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise OperatorExecutionError(f"{target} operator factory options must be a mapping")
    return dict(value)


def factory_options_fingerprint(options: Mapping[str, Any]) -> str:
    return canonical_fingerprint(options)


def runtime_adapter_fingerprint(materialization: RuntimeMaterialization) -> str:
    observed = materialization.provenance.evidence.get("adapter_implementation_fingerprint")
    if isinstance(observed, str) and observed:
        return observed
    return materialization.provenance.implementation_fingerprint


def execution_environment_provenance(
    specs: Mapping[str, ScorerSpec],
    *,
    runtime_adapter_fingerprint: str,
    operator_implementation_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[3]
    try:
        package_version = importlib.metadata.version("rl-kernel")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    torch_config = torch.__config__.show()
    execution_devices = {target: device_type(spec.device) for target, spec in sorted(specs.items())}
    return {
        "schema_version": "cross_config.environment.v1",
        "execution_devices": execution_devices,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": {
            "version": str(torch.__version__),
            "git_version": getattr(torch.version, "git_version", None),
            "cuda_build": getattr(torch.version, "cuda", None),
            "hip_build": getattr(torch.version, "hip", None),
            "debug_build": bool(getattr(torch.version, "debug", False)),
            "config_fingerprint": hashlib.sha256(torch_config.encode("utf-8")).hexdigest(),
        },
        "host_runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "mkl_available": bool(torch.backends.mkl.is_available()),
        },
        "rl_kernel": {
            "package_version": package_version,
            "git_revision": _git_revision(source_root),
            "source_tree_fingerprint": _cross_config_source_tree_fingerprint(
                source_root,
                implementation_fingerprints={
                    "runtime_adapter": runtime_adapter_fingerprint,
                    "operators": dict(operator_implementation_fingerprints),
                },
            ),
        },
    }


def _git_revision(source_root: Path) -> Optional[str]:
    git_dir = source_root / ".git"
    try:
        if git_dir.is_file():
            marker = git_dir.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir: "):
                return None
            resolved = Path(marker.removeprefix("gitdir: "))
            git_dir = resolved if resolved.is_absolute() else source_root / resolved
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        reference = head.removeprefix("ref: ")
        loose_ref = git_dir / reference
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip() or None
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            suffix = f" {reference}"
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.endswith(suffix):
                    return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _cross_config_source_tree_fingerprint(
    source_root: Path,
    *,
    implementation_fingerprints: Mapping[str, Any],
) -> str:
    paths = list((source_root / "rl_engine/alignment/cross_config").glob("*.py"))
    paths.extend(
        source_root / relative
        for relative in (
            "rl_engine/runtime/executor.py",
            "rl_engine/contracts/numerical.py",
            "rl_engine/runtime/registry.py",
            "rl_engine/runtime/semantic_registry.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise PairedRunnerError(
                f"cannot fingerprint cross-configuration source file {path}: {exc}"
            ) from exc
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    digest.update(
        json.dumps(
            json_safe(implementation_fingerprints),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()
