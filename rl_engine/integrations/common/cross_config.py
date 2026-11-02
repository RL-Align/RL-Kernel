# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared pieces for the Megatron and vLLM WS2 attention adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rl_engine.contracts.operators.attention import (
    AttentionDType,
    AttentionMerge,
    DowncastPoint,
    ReductionEngine,
    ReductionOrder,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
)
from rl_engine.integrations.common.attention_binding import AttentionRuntimeReadback
from rl_engine.validation.cross_config.runtime import KnobApplication
from rl_engine.validation.cross_config.schema import (
    IsolationScope,
    KnobDescriptor,
    MaterializationStatus,
)

__all__ = [
    "QWEN3_8B",
    "Qwen3ModelSpec",
    "AttentionRuntimeReadback",
    "application",
    "attention_dtype",
    "build_reduction_spec",
    "build_sharding_spec",
    "causal_offsets_for",
    "flatten",
    "split_kv_spec",
    "unsupported_reduction_reason",
]


@dataclass(frozen=True)
class Qwen3ModelSpec:
    """Architecture constants for the frozen dense target.

    These are *not* knobs. #235/#239/#241 all fix Qwen3-8B dense, so they belong to
    the scenario, and both sides must agree on them or the comparison is void.
    """

    name: str = "qwen3-8b"
    hidden_size: int = 4096
    ffn_hidden_size: int = 12288
    num_layers: int = 36
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    real_vocab_size: int = 151936
    rope_theta: float = 1.0e6
    rotary_dim: int = 128
    rope_scaling: str | None = None
    qk_layernorm: bool = True

    def identity_fields(self) -> dict[str, Any]:
        """The subset of :data:`IDENTITY_FIELDS` this spec is responsible for."""

        return {
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "rotary_dim": self.rotary_dim,
            "qk_layernorm": self.qk_layernorm,
        }


QWEN3_8B = Qwen3ModelSpec()


#: The planner normalizes dtype knobs to torch spellings (``bfloat16``), while
#: :class:`AttentionDType` uses short spellings (``bf16``). Passing a normalized knob
#: straight into the enum raises, so every adapter must translate here rather than
#: each inventing its own mapping.
_DTYPE_ALIASES: Mapping[str, AttentionDType] = {
    "bf16": AttentionDType.BF16,
    "bfloat16": AttentionDType.BF16,
    "fp16": AttentionDType.FP16,
    "float16": AttentionDType.FP16,
    "half": AttentionDType.FP16,
    "fp32": AttentionDType.FP32,
    "float32": AttentionDType.FP32,
    "float": AttentionDType.FP32,
}


def attention_dtype(value: Any, *, field: str) -> AttentionDType:
    """Translate a normalized knob dtype into an :class:`AttentionDType`."""

    if isinstance(value, AttentionDType):
        return value
    key = str(value).strip().lower().replace("torch.", "")
    try:
        return _DTYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"{field}={value!r} is not a supported attention dtype; "
            f"expected one of {sorted(set(_DTYPE_ALIASES))}"
        ) from exc


def split_kv_spec(flat: Mapping[str, Any]) -> SplitKVSpec:
    """Build the first-class logical Split-KV request.

    The integer is a fixed logical KV chunk size in tokens. It is intentionally
    not vLLM's ``flash_attn_max_num_splits_for_cuda_graph``: that setting is only
    an upper bound and cannot prove which runtime boundaries executed.
    """

    split_size = flat.get("attention.split_kv_policy")
    if split_size is None:
        return SplitKVSpec.disabled()
    return SplitKVSpec.fixed(int(split_size))


def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested knob mappings into dotted paths."""

    flat: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}{key}"
        if isinstance(child, Mapping):
            flat.update(flatten(child, f"{path}."))
        else:
            flat[path] = child
    return flat


def application(
    descriptor: KnobDescriptor,
    requested: Any,
    materialized: Any,
    actual: Any,
    status: MaterializationStatus,
    reason: str,
    **evidence: Any,
) -> KnobApplication:
    return KnobApplication(
        path=descriptor.path,
        requested=requested,
        materialized=materialized,
        actual=actual,
        lifecycle=descriptor.lifecycle,
        status=status,
        evidence={"reason": reason, **evidence},
        critical=descriptor.critical,
    )


def unsupported_reduction_reason(flat: Mapping[str, Any]) -> str | None:
    """Return why the requested reduction cannot be materialized, if it cannot.

    #236 declares single-member enums for merge order, downcast point and reduction
    engine, so the alternative knob values exist only as control groups. Requesting
    one must fail loudly rather than quietly collapse onto the supported value --
    silently substituting ``global_block_index`` for a requested ``arrival`` would
    make the control group indistinguishable from the treatment.
    """

    order = flat.get("attention.reduction_order")
    if order is not None and order != ReductionOrder.GLOBAL_BLOCK_INDEX.value:
        return (
            f"attention.reduction_order={order!r} has no backend; #236 ReductionOrder "
            f"declares only {ReductionOrder.GLOBAL_BLOCK_INDEX.value!r}"
        )
    downcast = flat.get("attention.reduction_downcast_at")
    if downcast is not None and downcast != DowncastPoint.FINAL_WRITE.value:
        return (
            f"attention.reduction_downcast_at={downcast!r} has no backend; #236 "
            f"DowncastPoint declares only {DowncastPoint.FINAL_WRITE.value!r}"
        )
    engine = flat.get("attention.reduction_engine")
    if engine is not None and engine != ReductionEngine.IN_OP_REFERENCE.value:
        return (
            f"attention.reduction_engine={engine!r} has no backend; the Transformer "
            "Engine merge oracle lands in #235 PR2/PR3, not here"
        )
    acc_dtype = flat.get("attention.reduction_acc_dtype")
    if (
        acc_dtype is not None
        and attention_dtype(acc_dtype, field="attention.reduction_acc_dtype")
        is not AttentionDType.FP32
    ):
        return (
            f"attention.reduction_acc_dtype={acc_dtype!r} violates the WS2 mandate; "
            "the CP (out, lse) merge accumulates in fp32"
        )
    return None


def build_reduction_spec(flat: Mapping[str, Any]) -> ReductionSpec:
    """Build the reduction spec, having already rejected unsupported requests."""

    return ReductionSpec(
        merge=AttentionMerge.ONLINE_SOFTMAX_LSE,
        acc_dtype=AttentionDType.FP32,
        order=ReductionOrder.GLOBAL_BLOCK_INDEX,
        downcast_at=DowncastPoint.FINAL_WRITE,
        engine=ReductionEngine.IN_OP_REFERENCE,
    )


def build_sharding_spec(
    *,
    model: Qwen3ModelSpec,
    tp_rank: int,
    tp_world_size: int,
    cp_rank: int,
    cp_world_size: int,
    global_sequence_length: int,
) -> ShardingSpec:
    """Build a CP/TP sharding spec for one rank of the frozen layout.

    TP splits heads, CP splits the sequence. The #239 rank layout fixes
    ``rank = cp_rank * tp_world_size + tp_rank`` for a 2-node x 2-GPU deployment,
    but nothing here depends on that mapping: ownership is derived from the ranks
    themselves so the same builder serves CP=1 baselines.
    """

    if model.q_heads % tp_world_size or model.kv_heads % tp_world_size:
        raise ValueError(
            f"Qwen3 GQA heads ({model.q_heads}/{model.kv_heads}) must divide evenly "
            f"across tp_world_size={tp_world_size}"
        )
    if global_sequence_length % cp_world_size:
        raise ValueError(
            f"global_sequence_length={global_sequence_length} must divide evenly "
            f"across cp_world_size={cp_world_size}"
        )

    local_q_heads = model.q_heads // tp_world_size
    local_kv_heads = model.kv_heads // tp_world_size
    local_sequence_length = global_sequence_length // cp_world_size

    return ShardingSpec(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        global_q_heads=model.q_heads,
        global_kv_heads=model.kv_heads,
        local_q_head_start=tp_rank * local_q_heads,
        local_q_heads=local_q_heads,
        local_kv_head_start=tp_rank * local_kv_heads,
        local_kv_heads=local_kv_heads,
        global_sequence_length=global_sequence_length,
        local_sequence_length=local_sequence_length,
        # One contiguous CP block per rank. The merge order key is the global block
        # index, never the arrival order of the CP exchange.
        global_block_indices=(cp_rank,),
        global_block_token_starts=(cp_rank * local_sequence_length,),
        local_block_offsets=(0, local_sequence_length),
    )


def causal_offsets_for(sharding: ShardingSpec, batch_size: int) -> tuple[int, ...]:
    """Causal offsets for one CP shard, one entry per batch entry.

    Under CP the local query block does not start at global position zero, so the
    causal mask has to be shifted by the number of preceding global tokens. Taking
    that from ``global_block_token_starts`` rather than recomputing
    ``cp_rank * local_sequence_length`` keeps uneven CP splits correct.
    """

    offset = sharding.global_block_token_starts[0]
    return (offset,) * batch_size


_PROCESS_SCOPES = (IsolationScope.PROCESS, IsolationScope.DISTRIBUTED_CONTEXT)
