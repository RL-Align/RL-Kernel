# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Rollout-side (vLLM) runtime adapter for WS2 attention cross-config.

Mirrors :mod:`.megatron`, with three differences that come straight from what vLLM
actually is:

* vLLM's context parallelism is ``prefill_context_parallel_size`` -- it applies to
  prefill only, so a decode-mode contract must declare ``cp_world_size == 1``
  regardless of what the prefill knob says.
* ``CacheConfig.block_size`` is the paged-KV page size, and it feeds
  ``KVCacheSpec.page_size`` directly rather than being invented here.
* Determinism comes from the ``VLLM_BATCH_INVARIANT`` environment variable rather
  than from a config field, because vLLM applies it inside
  ``init_batch_invariance()`` at worker startup.

Like the Megatron adapter, nothing here imports ``vllm``; configs are duck-typed so
the module is importable anywhere. Configured-only values remain ``UNOBSERVABLE``;
``APPLIED`` requires an explicit post-execution ``AttentionRuntimeReadback``.
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
from rl_engine.validation.cross_config.determinism import DeterminismProbe, vllm_probe_from_env
from rl_engine.validation.cross_config.runtime import (
    AdapterMaterialization,
    KnobApplication,
    RuntimeBinding,
)
from rl_engine.validation.cross_config.schema import KnobDescriptor, MaterializationStatus

__all__ = [
    "VLLM_ATTENTION_KEYS",
    "VLLM_CACHE_KEYS",
    "VLLM_FROZEN_ASSERTIONS",
    "VLLM_MODEL_KEYS",
    "VLLM_PARALLEL_KEYS",
    "VllmProvenanceAdapter",
    "VllmRolloutMaterializer",
]


VLLM_MODEL_KEYS: tuple[str, ...] = (
    "dtype",
    "seed",
    "quantization",
    "enforce_eager",
    "max_logprobs",
    "disable_cascade_attn",
    "max_model_len",
)

VLLM_CACHE_KEYS: tuple[str, ...] = (
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "prefix_caching_hash_algo",
    "calculate_kv_scales",
    "sliding_window",
)

VLLM_ATTENTION_KEYS: tuple[str, ...] = (
    "backend",
    "flash_attn_version",
    "use_prefill_decode_attention",
    "flash_attn_max_num_splits_for_cuda_graph",
    "use_cudnn_prefill",
    "disable_flashinfer_prefill",
    "use_non_causal",
)

VLLM_PARALLEL_KEYS: tuple[str, ...] = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "prefill_context_parallel_size",
    "data_parallel_size",
)


#: Frozen dense-target assertions on the rollout side. ``cache_dtype`` must stay
#: ``auto`` because an FP8 KV cache is a representation-drift problem tracked
#: separately, and ``disable_cascade_attn`` must stay ``True`` because cascade
#: attention changes the block-merge structure the contract pins down.
VLLM_FROZEN_ASSERTIONS: Mapping[str, Any] = {
    "quantization": None,
    "cache_dtype": "auto",
    "calculate_kv_scales": False,
    "disable_cascade_attn": True,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "sliding_window": None,
}


def _value(config: Any, name: str) -> Any:
    raw = getattr(config, name, None)
    return getattr(raw, "value", raw) if raw is not None else None


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class VllmProvenanceAdapter:
    """Extract fingerprints and determinism evidence from vLLM configs."""

    framework = "vllm"

    def __init__(
        self,
        *,
        model_config: Any = None,
        cache_config: Any = None,
        attention_config: Any = None,
        parallel_config: Any = None,
        env: Optional[Mapping[str, str]] = None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.attention_config = attention_config
        self.parallel_config = parallel_config
        self.env = dict(env or {})

    def construction_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {}
        for prefix, config, keys in (
            ("model", self.model_config, VLLM_MODEL_KEYS),
            ("cache", self.cache_config, VLLM_CACHE_KEYS),
            ("attention", self.attention_config, VLLM_ATTENTION_KEYS),
        ):
            for name in keys:
                view[f"{prefix}.{name}"] = _value(config, name)
        return view

    def distributed_view(self) -> dict[str, Any]:
        return {
            f"parallel.{name}": _value(self.parallel_config, name) for name in VLLM_PARALLEL_KEYS
        }

    @property
    def construction_fingerprint(self) -> str:
        return _fingerprint(self.construction_view())

    @property
    def distributed_context_fingerprint(self) -> str:
        return _fingerprint(self.distributed_view())

    def determinism_probe(self) -> DeterminismProbe:
        return vllm_probe_from_env(self.env, model_config=self.model_config)

    def frozen_scope_violations(self) -> tuple[str, ...]:
        sources = {
            "quantization": self.model_config,
            "disable_cascade_attn": self.model_config,
            "cache_dtype": self.cache_config,
            "calculate_kv_scales": self.cache_config,
            "sliding_window": self.cache_config,
            "pipeline_parallel_size": self.parallel_config,
            "data_parallel_size": self.parallel_config,
        }
        violations: list[str] = []
        for name, expected in VLLM_FROZEN_ASSERTIONS.items():
            config = sources.get(name)
            if config is None:
                continue
            actual = _value(config, name)
            if actual != expected:
                violations.append(f"{name}={actual!r} (expected {expected!r})")
        return tuple(violations)

    @property
    def kv_page_size(self) -> Optional[int]:
        """vLLM's paged-KV block size, which is the contract's ``page_size``."""

        block_size = _value(self.cache_config, "block_size")
        return int(block_size) if block_size is not None else None

    @property
    def split_kv_policy(self) -> Optional[int]:
        """Diagnostic vLLM maximum split count, not the logical chunk-size contract."""

        splits = _value(self.attention_config, "flash_attn_max_num_splits_for_cuda_graph")
        return int(splits) if splits is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "construction": self.construction_view(),
            "distributed_context": self.distributed_view(),
            "construction_fingerprint": self.construction_fingerprint,
            "distributed_context_fingerprint": self.distributed_context_fingerprint,
            "frozen_scope_violations": list(self.frozen_scope_violations()),
            "kv_page_size": self.kv_page_size,
            "flash_attn_max_num_splits_for_cuda_graph": self.split_kv_policy,
            "determinism": self.determinism_probe().to_dict(),
        }


class VllmRolloutMaterializer:
    """Materialize the rollout-side attention runtime for the WS2 target."""

    runtime_kind = "vllm_attention"

    def __init__(
        self,
        *,
        model: Qwen3ModelSpec = QWEN3_8B,
        global_sequence_length: int = 4096,
        tp_rank: int = 0,
        cp_rank: int = 0,
        mode: AttentionMode = AttentionMode.CHUNKED_PREFILL,
        backend_id: str = "vllm.flash_attn",
        provenance: Optional[VllmProvenanceAdapter] = None,
        runtime_readback: Optional[AttentionRuntimeReadback] = None,
    ):
        self.model = model
        self.global_sequence_length = global_sequence_length
        self.tp_rank = tp_rank
        self.cp_rank = cp_rank
        self.mode = mode
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

    def effective_cp_world_size(self, flat: Mapping[str, Any]) -> int:
        """CP applies to prefill only; decode always runs at CP=1."""

        requested = int(flat.get("rollout.context_parallel_size", 1))
        if self.mode is AttentionMode.DECODE:
            return 1
        return requested

    def build_contract(self, flat: Mapping[str, Any]) -> AttentionContract:
        if self.mode is AttentionMode.DECODE:
            # Decode replay needs a validated KVCacheSpec (cache positions, page
            # ownership, prefix-cache identity). That is #235 PR6's contract surface,
            # and inventing a placeholder here would let an unvalidated decode case
            # look bound. Fail instead.
            raise AttentionContractError(
                "decode-mode materialization requires KV-cache identity from #235 PR6; "
                "this adapter covers prefill and chunked prefill"
            )
        tp_world_size = int(flat.get("rollout.tensor_parallel_size", 1))
        cp_world_size = self.effective_cp_world_size(flat)
        sharding = build_sharding_spec(
            model=self.model,
            tp_rank=self.tp_rank,
            tp_world_size=tp_world_size,
            cp_rank=self.cp_rank if cp_world_size > 1 else 0,
            cp_world_size=cp_world_size,
            global_sequence_length=self.global_sequence_length,
        )
        fusion = RoPEFusionBoundary(
            flat.get(
                "attention.fusion_boundary",
                RoPEFusionBoundary.FUSED_ROPE_ATTENTION.value,
            )
        )
        rope = RoPESpec(
            q_state=RoPEState.POST_ROPE,
            k_state=RoPEState.POST_ROPE,
            # vLLM stores post-RoPE K in the cache; recorded, not asserted equal to
            # the training side, because it is a materialization fact.
            k_cache_state=RoPEState.POST_ROPE,
            theta=self.model.rope_theta,
            rotary_dim=self.model.rotary_dim,
            rope_scaling=self.model.rope_scaling,
            fusion_boundary=fusion,
        )
        batch_size = int(flat.get("batch.size", 1))
        return AttentionContract(
            role=AttentionRole.INFER,
            mode=self.mode,
            dtype=attention_dtype(flat.get("rollout.dtype", "bf16"), field="rollout.dtype"),
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

        requested_cp = int(flat.get("rollout.context_parallel_size", 1))
        effective_cp = self.effective_cp_world_size(flat)

        for path, requested in flat.items():
            descriptor = descriptors.get(path)
            if descriptor is None or "rollout" not in descriptor.targets:
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
            if path == "rollout.context_parallel_size" and effective_cp != requested_cp:
                applications.append(
                    application(
                        descriptor,
                        requested,
                        effective_cp,
                        effective_cp,
                        MaterializationStatus.FALLBACK,
                        (
                            "vLLM context parallelism covers prefill only; a decode-mode "
                            f"contract runs at cp_world_size=1, not {requested_cp}"
                        ),
                        vllm_field="ParallelConfig.prefill_context_parallel_size",
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

        tp_world_size = int(flat.get("rollout.tensor_parallel_size", 1))
        side_config: dict[str, Any] = {
            "framework": "vllm",
            "dtype": flat.get("rollout.dtype"),
            "enforce_eager": flat.get("rollout.enforce_eager"),
            "enable_prefix_caching": flat.get("rollout.enable_prefix_caching"),
            "batch_invariant": flat.get("rollout.batch_invariant"),
            "kv_block_size": flat.get("rollout.kv_block_size"),
            "split_kv_policy": flat.get("attention.split_kv_policy"),
            "attention_mode": self.mode.value,
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
                side_configs={"rollout": side_config, "training": {}},
                topology={
                    "rollout": {
                        "world_size": tp_world_size * effective_cp,
                        "tensor_parallel_size": tp_world_size,
                        "context_parallel_size": effective_cp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                    },
                    "training": {"world_size": 1},
                },
                scorer={
                    "mode": "rollout_logprob",
                    "framework": "vllm",
                    "export_lse": True,
                },
                operator_backends={
                    "rollout": self.backend_id,
                    "training": self.backend_id,
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
                "configured in the rollout contract, but no vLLM runtime readback was supplied",
                frozen_scope_violations=list(scope_violations),
            )
        if scope_violations or not readback.frozen_scope_verified:
            return application(
                descriptor,
                requested,
                requested,
                readback.actual_knobs.get(descriptor.path),
                MaterializationStatus.UNOBSERVABLE,
                "vLLM frozen-scope assertions were not all verified by runtime readback",
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
                "vLLM runtime reported a Split-KV fallback",
                runtime_readback_source=readback.source,
            )
        if readback.contract != contract:
            return application(
                descriptor,
                requested,
                requested,
                readback.actual_knobs.get(descriptor.path),
                MaterializationStatus.FALLBACK,
                "vLLM runtime contract differs from the requested contract",
                runtime_readback_source=readback.source,
            )
        if descriptor.path not in readback.actual_knobs:
            return application(
                descriptor,
                requested,
                requested,
                None,
                MaterializationStatus.UNOBSERVABLE,
                "vLLM runtime readback does not expose this knob",
                runtime_readback_source=readback.source,
            )
        actual = readback.actual_knobs[descriptor.path]
        status = (
            MaterializationStatus.APPLIED if actual == requested else MaterializationStatus.FALLBACK
        )
        reason = (
            "verified from the executed vLLM runtime"
            if status is MaterializationStatus.APPLIED
            else "vLLM runtime value differs from the requested value"
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
