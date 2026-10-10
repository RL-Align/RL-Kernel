# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Training-side (Megatron) runtime adapter for WS2 attention cross-config.

Two things live here:

``MegatronProvenanceAdapter``
    Read-only. Turns a Megatron config object into the construction and
    distributed-context fingerprints the cross-config framework already expects,
    plus the determinism probe. It never imports ``megatron`` -- every accessor is
    duck-typed -- so this module is importable and testable on a laptop.

``MegatronAttentionMaterializer``
    Implements the ``RuntimeMaterializer`` protocol. Before this PR the only
    implementation in the repository was ``CpuSmokeMaterializer`` over a synthetic
    CPU model, so nothing had ever materialized a real distributed runtime.

Scope boundary: materialization builds and validates the training-side
:class:`AttentionContract` and reports what would be constructed. Without an
``AttentionRuntimeReadback`` it reports ``UNOBSERVABLE``, never ``APPLIED``. It
does not launch ``torchrun``, initialize process groups, or execute attention;
the 2-node x 2-GPU launcher must inject readback collected after execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Optional

from rl_engine.contracts.operators.attention import (
    AttentionContract,
    AttentionContractError,
    AttentionMode,
    AttentionRole,
    RoPEFusionBoundary,
    RoPESpec,
    RoPEState,
)
from rl_engine.integrations.common.cross_config import (
    QWEN3_8B,
    AttentionRuntimeReadback,
    Qwen3ModelSpec,
    application,
    attention_dtype,
    build_reduction_spec,
    build_sharding_spec,
    causal_offsets_for,
    flatten,
    split_kv_spec,
    unsupported_reduction_reason,
)
from rl_engine.runtime.semantic_registry import implementation_fingerprint
from rl_engine.validation.cross_config.determinism import (
    DeterminismProbe,
    megatron_probe_from_config,
)
from rl_engine.validation.cross_config.runtime import (
    AdapterMaterialization,
    KnobApplication,
    RuntimeBinding,
)
from rl_engine.validation.cross_config.schema import KnobDescriptor, MaterializationStatus

__all__ = [
    "MEGATRON_CONSTRUCTION_KEYS",
    "MEGATRON_DISTRIBUTED_KEYS",
    "MegatronAttentionMaterializer",
    "MegatronProvenanceAdapter",
]


#: ``TransformerConfig`` fields that change attention arithmetic. Hashed into the
#: construction fingerprint. Deliberately excludes MoE, Mamba, MLA and sparse
#: attention fields: the frozen target is Qwen3-8B dense, and those are asserted
#: off rather than recorded.
MEGATRON_CONSTRUCTION_KEYS: tuple[str, ...] = (
    "attention_backend",
    "attention_softmax_in_fp32",
    "apply_query_key_layer_scaling",
    "apply_rope_fusion",
    "masked_softmax_fusion",
    "bias_activation_fusion",
    "bias_dropout_fusion",
    "gradient_accumulation_fusion",
    "cross_entropy_loss_fusion",
    "cross_entropy_fusion_impl",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "recompute_modules",
    "rotary_base",
    "rotary_percent",
    "rotary_interleaved",
    "rotary_scaling_factor",
    "qk_layernorm",
    "hidden_dropout",
    "attention_dropout",
    "params_dtype",
    "bf16",
    "fp16",
    "fp8",
    "deterministic_mode",
)


#: ``ModelParallelConfig`` fields that define the distributed context.
MEGATRON_DISTRIBUTED_KEYS: tuple[str, ...] = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "context_parallel_size",
    "hierarchical_context_parallel_sizes",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "sequence_parallel",
    "cp_comm_type",
    "tp_comm_overlap",
    "use_te_rng_tracker",
)


#: Fields that must hold these values for the frozen dense target. A mismatch is a
#: hard stop, not a recorded difference -- see the exclusion list in the WS2 scope.
MEGATRON_FROZEN_ASSERTIONS: Mapping[str, Any] = {
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "sequence_parallel": False,
    "fp8": None,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


def _value(config: Any, name: str) -> Any:
    raw = getattr(config, name, None)
    return getattr(raw, "value", raw) if raw is not None else None


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class MegatronProvenanceAdapter:
    """Extract fingerprints and determinism evidence from a Megatron config.

    ``config`` may be a real ``TransformerConfig``/``ModelParallelConfig``, a merged
    namespace, or a test double. Missing attributes read as ``None`` and are
    recorded as such rather than raising: an absent field is itself provenance.
    """

    framework = "megatron"

    def __init__(self, config: Any, *, env: Optional[Mapping[str, str]] = None):
        self.config = config
        self.env = dict(env or {})

    def construction_view(self) -> dict[str, Any]:
        return {name: _value(self.config, name) for name in MEGATRON_CONSTRUCTION_KEYS}

    def distributed_view(self) -> dict[str, Any]:
        return {name: _value(self.config, name) for name in MEGATRON_DISTRIBUTED_KEYS}

    @property
    def construction_fingerprint(self) -> str:
        return _fingerprint(self.construction_view())

    @property
    def distributed_context_fingerprint(self) -> str:
        return _fingerprint(self.distributed_view())

    def determinism_probe(self) -> DeterminismProbe:
        return megatron_probe_from_config(self.config, env=self.env)

    def frozen_scope_violations(self) -> tuple[str, ...]:
        """Return the frozen-scope assertions this config violates."""

        violations: list[str] = []
        for name, expected in MEGATRON_FROZEN_ASSERTIONS.items():
            actual = _value(self.config, name)
            if actual is None:
                # Not declared. Treated as unknown rather than as satisfied, because
                # a silently-absent MoE or FP8 setting is exactly the case that would
                # otherwise slip past a dense-only claim.
                violations.append(f"{name} is not declared (expected {expected!r})")
            elif actual != expected:
                violations.append(f"{name}={actual!r} (expected {expected!r})")
        return tuple(violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "construction": self.construction_view(),
            "distributed_context": self.distributed_view(),
            "construction_fingerprint": self.construction_fingerprint,
            "distributed_context_fingerprint": self.distributed_context_fingerprint,
            "frozen_scope_violations": list(self.frozen_scope_violations()),
            "determinism": self.determinism_probe().to_dict(),
        }


class MegatronAttentionMaterializer:
    """Materialize the training-side attention runtime for the WS2 target."""

    runtime_kind = "megatron_attention"

    def __init__(
        self,
        *,
        model: Qwen3ModelSpec = QWEN3_8B,
        global_sequence_length: int = 4096,
        tp_rank: int = 0,
        cp_rank: int = 0,
        backend_id: str = "rlkernel.cp_attention_reference",
        provenance: Optional[MegatronProvenanceAdapter] = None,
        runtime_readback: Optional[AttentionRuntimeReadback] = None,
    ):
        self.model = model
        self.global_sequence_length = global_sequence_length
        self.tp_rank = tp_rank
        self.cp_rank = cp_rank
        self.backend_id = backend_id
        self.provenance = provenance
        self.runtime_readback = runtime_readback

    @property
    def implementation_fingerprint(self) -> str:
        return implementation_fingerprint(
            type(self),
            instance=self,
            entrypoints=("materialize", "build_contract"),
        )

    def build_contract(self, flat: Mapping[str, Any]) -> AttentionContract:
        """Build the training-side contract. Raises on an unusable request."""

        tp_world_size = int(flat.get("training.tensor_parallel_size", 1))
        cp_world_size = int(flat.get("training.context_parallel_size", 1))
        sharding = build_sharding_spec(
            model=self.model,
            tp_rank=self.tp_rank,
            tp_world_size=tp_world_size,
            cp_rank=self.cp_rank,
            cp_world_size=cp_world_size,
            global_sequence_length=self.global_sequence_length,
        )
        fusion = RoPEFusionBoundary(
            flat.get(
                "attention.fusion_boundary",
                RoPEFusionBoundary.UNFUSED_ROPE_ATTENTION.value,
            )
        )
        rope = RoPESpec(
            q_state=RoPEState.POST_ROPE,
            k_state=RoPEState.POST_ROPE,
            k_cache_state=RoPEState.POST_ROPE,
            theta=self.model.rope_theta,
            rotary_dim=self.model.rotary_dim,
            rope_scaling=self.model.rope_scaling,
            fusion_boundary=fusion,
        )
        batch_size = int(flat.get("batch.size", 1))
        return AttentionContract(
            role=AttentionRole.TRAIN,
            mode=AttentionMode.PREFILL,
            dtype=attention_dtype(
                flat.get("training.compute_dtype", "bf16"), field="training.compute_dtype"
            ),
            batch_size=batch_size,
            query_sequence_length=sharding.local_sequence_length,
            head_dim=self.model.head_dim,
            causal=True,
            causal_offsets=causal_offsets_for(sharding, batch_size),
            sharding=sharding,
            reduction=build_reduction_spec(flat),
            split_kv=split_kv_spec(flat),
            rope=rope,
            export_lse=True,
        )

    def materialize(
        self,
        normalized: Mapping[str, Any],
        descriptors: Mapping[str, KnobDescriptor],
    ) -> AdapterMaterialization:
        flat = flatten(normalized)
        applications: list[KnobApplication] = []

        blocked = unsupported_reduction_reason(flat)
        scope_violations = (
            self.provenance.frozen_scope_violations() if self.provenance is not None else ()
        )

        contract: AttentionContract | None = None
        contract_error: str | None = None
        if blocked is None:
            try:
                contract = self.build_contract(flat)
            except (AttentionContractError, ValueError) as exc:
                contract_error = str(exc)

        for path, requested in flat.items():
            descriptor = descriptors.get(path)
            if descriptor is None or "training" not in descriptor.targets:
                continue
            if blocked is not None and path.startswith("attention.reduction"):
                applications.append(
                    application(
                        descriptor,
                        requested,
                        None,
                        None,
                        MaterializationStatus.UNSUPPORTED,
                        blocked,
                    )
                )
                continue
            if contract_error is not None:
                applications.append(
                    application(
                        descriptor,
                        requested,
                        None,
                        None,
                        MaterializationStatus.ERROR,
                        contract_error,
                    )
                )
                continue
            if contract is None:
                applications.append(
                    application(
                        descriptor,
                        requested,
                        None,
                        None,
                        MaterializationStatus.ERROR,
                        f"attention contract is unavailable: {blocked}",
                    )
                )
                continue
            applications.append(
                self._runtime_application(
                    descriptor,
                    requested,
                    contract=contract,
                    scope_violations=scope_violations,
                )
            )

        tp_world_size = int(flat.get("training.tensor_parallel_size", 1))
        cp_world_size = int(flat.get("training.context_parallel_size", 1))
        side_config: dict[str, Any] = {
            "framework": "megatron",
            "attention_backend": flat.get("training.attention_backend"),
            "compute_dtype": flat.get("training.compute_dtype"),
            "deterministic_mode": flat.get("training.deterministic_mode"),
            "cp_comm_type": flat.get("training.cp_comm_type"),
            "contract": contract.to_dict() if contract is not None else None,
            "contract_error": contract_error or blocked,
            "runtime_readback": (
                None if self.runtime_readback is None else self.runtime_readback.to_dict()
            ),
            "frozen_scope_violations": list(scope_violations),
        }
        if self.provenance is not None:
            side_config["provenance"] = self.provenance.to_dict()

        return AdapterMaterialization(
            applications=tuple(applications),
            binding=RuntimeBinding(
                batch_size=int(flat.get("batch.size", 1)),
                side_configs={"training": side_config, "rollout": {}},
                topology={
                    "training": {
                        "world_size": tp_world_size * cp_world_size,
                        "tensor_parallel_size": tp_world_size,
                        "context_parallel_size": cp_world_size,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                    },
                    "rollout": {"world_size": 1},
                },
                scorer={
                    "mode": "teacher_forcing",
                    "framework": "megatron",
                    "export_lse": True,
                },
                operator_backends={
                    "training": self.backend_id,
                    "rollout": self.backend_id,
                },
                runtime_kind=self.runtime_kind,
            ),
        )

    def _runtime_application(
        self,
        descriptor: KnobDescriptor,
        requested: Any,
        *,
        contract: AttentionContract,
        scope_violations: tuple[str, ...],
    ) -> KnobApplication:
        readback = self.runtime_readback
        if readback is None:
            return application(
                descriptor,
                requested,
                requested,
                None,
                MaterializationStatus.UNOBSERVABLE,
                (
                    "configured in the training contract, but no Megatron runtime "
                    "readback was supplied"
                ),
                frozen_scope_violations=list(scope_violations),
            )
        if scope_violations or not readback.frozen_scope_verified:
            return application(
                descriptor,
                requested,
                requested,
                readback.actual_knobs.get(descriptor.path),
                MaterializationStatus.UNOBSERVABLE,
                "Megatron frozen-scope assertions were not all verified by runtime readback",
                runtime_readback_source=readback.source,
                frozen_scope_violations=list(scope_violations),
            )
        if readback.split_kv_fallback:
            return application(
                descriptor,
                requested,
                requested,
                readback.actual_knobs.get(descriptor.path),
                MaterializationStatus.FALLBACK,
                "Megatron runtime reported a Split-KV fallback",
                runtime_readback_source=readback.source,
            )
        if readback.contract != contract:
            return application(
                descriptor,
                requested,
                requested,
                readback.actual_knobs.get(descriptor.path),
                MaterializationStatus.FALLBACK,
                "Megatron runtime contract differs from the requested contract",
                runtime_readback_source=readback.source,
            )
        if descriptor.path not in readback.actual_knobs:
            return application(
                descriptor,
                requested,
                requested,
                None,
                MaterializationStatus.UNOBSERVABLE,
                "Megatron runtime readback does not expose this knob",
                runtime_readback_source=readback.source,
            )
        actual = readback.actual_knobs[descriptor.path]
        status = (
            MaterializationStatus.APPLIED if actual == requested else MaterializationStatus.FALLBACK
        )
        reason = (
            "verified from the executed Megatron runtime"
            if status is MaterializationStatus.APPLIED
            else "Megatron runtime value differs from the requested value"
        )
        return application(
            descriptor,
            requested,
            requested,
            actual,
            status,
            reason,
            runtime_readback_source=readback.source,
            frozen_scope_violations=list(scope_violations),
        )
