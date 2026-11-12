# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Target-specific semantic operator selection for alignment cases."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Optional, cast

import torch

from rl_engine.contracts.operators.logprob import LogprobContract
from rl_engine.runtime.semantic_registry import (
    OperatorInstanceProvenance,
    OperatorRequirements,
    OperatorResolution,
    OperatorResolutionPolicy,
    OperatorSession,
    SemanticOperatorCatalog,
)

OperatorTarget = Literal["rollout", "training", "both"]
ConcreteOperatorTarget = Literal["rollout", "training"]


@dataclass(frozen=True)
class OperatorOverride:
    """Backend overrides for one semantic operator on either or both sides."""

    semantic_op: str
    rollout_backend: Optional[str] = None
    training_backend: Optional[str] = None

    def __post_init__(self) -> None:
        semantic_op = self.semantic_op.strip()
        if not semantic_op:
            raise ValueError("semantic_op must not be empty")
        rollout_backend = _normalized_optional_backend(self.rollout_backend)
        training_backend = _normalized_optional_backend(self.training_backend)
        if rollout_backend is None and training_backend is None:
            raise ValueError("operator override must select rollout, training, or both")
        object.__setattr__(self, "semantic_op", semantic_op)
        object.__setattr__(self, "rollout_backend", rollout_backend)
        object.__setattr__(self, "training_backend", training_backend)

    @classmethod
    def for_target(
        cls,
        *,
        semantic_op: str,
        backend_id: str,
        target: OperatorTarget,
    ) -> OperatorOverride:
        """Create a rollout-only, training-only, or dual-side override."""

        normalized_target = target.strip().lower()
        if normalized_target == "rollout":
            return cls(semantic_op=semantic_op, rollout_backend=backend_id)
        if normalized_target == "training":
            return cls(semantic_op=semantic_op, training_backend=backend_id)
        if normalized_target == "both":
            return cls(
                semantic_op=semantic_op,
                rollout_backend=backend_id,
                training_backend=backend_id,
            )
        raise ValueError("target must be 'rollout', 'training', or 'both'")

    def backend_for(self, target: ConcreteOperatorTarget) -> Optional[str]:
        normalized_target = _concrete_target(target)
        if normalized_target == "rollout":
            return self.rollout_backend
        return self.training_backend

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_op": self.semantic_op,
            "rollout_backend": self.rollout_backend,
            "training_backend": self.training_backend,
        }


@dataclass(frozen=True)
class ResolvedOperatorOverride:
    """Target-specific exact resolutions produced from an operator override."""

    semantic_op: str
    rollout: Optional[OperatorResolution] = None
    training: Optional[OperatorResolution] = None

    def for_target(self, target: ConcreteOperatorTarget) -> Optional[OperatorResolution]:
        normalized_target = _concrete_target(target)
        return self.rollout if normalized_target == "rollout" else self.training

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_op": self.semantic_op,
            "rollout": None if self.rollout is None else self.rollout.to_dict(),
            "training": None if self.training is None else self.training.to_dict(),
        }


class OperatorBridge:
    """Resolve and instantiate semantic operator overrides without planner branches."""

    def __init__(
        self,
        catalog: Optional[SemanticOperatorCatalog | OperatorSession] = None,
        *,
        policy: Optional[OperatorResolutionPolicy] = None,
    ):
        """Create a bridge backed by one case-local operator session."""

        if isinstance(catalog, OperatorSession):
            self.catalog = catalog.catalog
            self.session = catalog
            self.policy = policy or catalog.policy
        else:
            if catalog is None:
                # Built-in descriptors are repository integration details; the
                # generic semantic catalog itself remains backend-neutral.
                from rl_engine.runtime.registry import kernel_registry

                catalog = kernel_registry.semantic
            if not isinstance(catalog, SemanticOperatorCatalog):
                raise TypeError("catalog must be a SemanticOperatorCatalog or OperatorSession")
            self.catalog = catalog
            self.policy = policy or OperatorResolutionPolicy()
            self.session = self.catalog.session(self.policy)

    def resolve_override(
        self,
        override: OperatorOverride,
        *,
        requirements: Mapping[str, OperatorRequirements],
        strict: bool = True,
    ) -> ResolvedOperatorOverride:
        """Resolve only the sides explicitly selected by ``override``."""

        resolved: dict[ConcreteOperatorTarget, OperatorResolution] = {}
        targets: tuple[ConcreteOperatorTarget, ...] = ("rollout", "training")
        for target in targets:
            backend_id = override.backend_for(target)
            if backend_id is None:
                continue
            target_requirements = requirements.get(target)
            if target_requirements is None:
                raise ValueError(f"missing operator requirements for target {target!r}")
            target_policy = replace(self.policy, strict=strict)
            resolved[target] = self.session.resolve(
                semantic_op=override.semantic_op,
                requested_backend=backend_id,
                target=target,
                requirements=target_requirements,
                policy=target_policy,
            )
        return ResolvedOperatorOverride(
            semantic_op=override.semantic_op,
            rollout=resolved.get("rollout"),
            training=resolved.get("training"),
        )

    def instantiate(
        self,
        resolved: ResolvedOperatorOverride,
        *,
        target: ConcreteOperatorTarget,
        factory_kwargs: Optional[Mapping[str, Any]] = None,
        cache: bool = False,
    ) -> Any:
        """Instantiate one resolved side; rollout and training remain independent."""

        resolution = resolved.for_target(target)
        if resolution is None:
            raise ValueError(f"operator override does not select target {target!r}")
        return self.session.instantiate(
            resolution,
            factory_kwargs=factory_kwargs,
            cache=cache,
        )

    def instance_provenance(
        self,
        resolved: ResolvedOperatorOverride,
        *,
        target: ConcreteOperatorTarget,
        instance: Any,
    ) -> OperatorInstanceProvenance:
        resolution = resolved.for_target(target)
        if resolution is None:
            raise ValueError(f"operator override does not select target {target!r}")
        return self.session.instance_provenance(resolution, instance)


def selected_logprobs_with_operator(
    operator: Any,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    active_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
    contract: Optional[LogprobContract] = None,
    tp_group: Any = None,
    num_vocab_tiles: Optional[int] = None,
    deterministic: bool = True,
    validate: bool = True,
) -> torch.Tensor:
    """Apply legacy or contract-aware selected-logprob operators uniformly."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            f"logits leading shape {tuple(logits.shape[:-1])} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )
    mask: Optional[torch.Tensor] = None
    safe_token_ids = token_ids.to(device=logits.device, dtype=torch.long)
    if active_mask is not None:
        if active_mask.shape != token_ids.shape:
            raise ValueError("active_mask shape must match token_ids shape")
        mask = active_mask.to(device=logits.device, dtype=torch.bool)
        safe_token_ids = safe_token_ids.masked_fill(~mask, 0)

    if contract is not None:
        if not isinstance(contract, LogprobContract):
            raise TypeError("contract must be a LogprobContract")
        contract_mask = torch.tensor(contract.mask.active_mask, dtype=torch.bool)
        if contract_mask.numel() != token_ids.numel():
            raise ValueError("contract active_mask size must match token_ids")
        if mask is not None and not torch.equal(contract_mask, mask.detach().cpu().reshape(-1)):
            raise ValueError("active_mask must match contract.mask.active_mask")
        kwargs: dict[str, Any] = {
            "contract": contract,
            "tp_group": tp_group,
            "deterministic": deterministic,
            "validate": validate,
        }
        if num_vocab_tiles is not None:
            kwargs["num_vocab_tiles"] = num_vocab_tiles
        if not callable(operator):
            raise TypeError("contract-aware selected-logprob operator must be callable")
        result = operator(
            (logits / float(temperature)).reshape(-1, logits.shape[-1]),
            safe_token_ids.reshape(-1),
            **kwargs,
        )
        if isinstance(result, tuple):
            if len(result) != 2:
                raise TypeError("contract-aware selected-logprob operator must return (logp, lse)")
            selected, lse = result
            if not isinstance(selected, torch.Tensor) or not isinstance(lse, torch.Tensor):
                raise TypeError("contract-aware selected-logprob operator must return (logp, lse)")
            if lse.shape != safe_token_ids.reshape(-1).shape:
                raise ValueError("vocab LSE output shape must match token_ids")
        else:
            selected = result
        if isinstance(selected, torch.Tensor):
            selected = selected.reshape(token_ids.shape)
    else:
        scaled_logits = logits.float() / float(temperature)
        if hasattr(operator, "apply_fp32") and callable(operator.apply_fp32):
            selected = operator.apply_fp32(scaled_logits, safe_token_ids)
        elif callable(operator):
            selected = operator(scaled_logits, safe_token_ids)
        else:
            raise TypeError("selected-logprob operator must be callable or expose apply_fp32")
    if not isinstance(selected, torch.Tensor):
        raise TypeError("selected-logprob operator must return a torch.Tensor")
    if selected.shape != token_ids.shape:
        raise ValueError(
            f"selected-logprob output shape {tuple(selected.shape)} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )
    selected = selected.to(device=logits.device, dtype=output_dtype)
    if mask is not None:
        selected = selected.masked_fill(~mask, 0.0)
    return selected


def _normalized_optional_backend(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("backend_id must not be empty")
    return normalized


def _concrete_target(target: ConcreteOperatorTarget) -> ConcreteOperatorTarget:
    normalized = target.strip().lower()
    if normalized not in {"rollout", "training"}:
        raise ValueError("target must be 'rollout' or 'training'")
    return cast(ConcreteOperatorTarget, normalized)


__all__ = [
    "ConcreteOperatorTarget",
    "OperatorBridge",
    "OperatorOverride",
    "OperatorTarget",
    "ResolvedOperatorOverride",
    "selected_logprobs_with_operator",
]
