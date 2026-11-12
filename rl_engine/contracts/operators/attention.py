# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Typed WS2 contract for context-parallel standard softmax attention.

The objects in this module describe a distributed attention invocation.  They
do not shard tensors, launch collectives, or implement the ``(out, lse)``
merge.  Keeping description and materialization separate lets dispatch reject
an incompatible backend before any numerically different path is launched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)


# Stable identities for Attention arithmetic shared by training and rollout.
# The FA4 core is the strict production path on CUDA; AITER/CK dense MHA is its
# ROCm counterpart. The materializing RL-Kernel core remains available as an
# explicit reference and capability-gap fallback.
STRICT_ATTENTION_PRODUCTION_CORE_ID = "rlkernel.attention.flash_attention4.num_splits1.v1"
STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID = "rlkernel.attention.rocm.aiter_ck_dense_mha.v1"
STRICT_ATTENTION_REFERENCE_CORE_ID = "rlkernel.attention.deterministic_core.v1"
# Compatibility alias for callers that explicitly select the original core.
STRICT_ATTENTION_CORE_ID = STRICT_ATTENTION_REFERENCE_CORE_ID
STRICT_ATTENTION_FA4_SCHEDULE_ID = "single_batch_flash_attention4_num_splits1"
STRICT_ATTENTION_ROCM_SCHEDULE_ID = "single_batch_aiter_ck_dense_mha_no_splitkv"
STRICT_ATTENTION_SCHEDULE_ID = "single_batch_single_query_global_kv_blocks"
# The distributed strict path executes the full-KV core above. This identifies
# the fixed, pre-overlap communication schedule around that core.
STRICT_ATTENTION_RING_SCHEDULE_ID = "rlkernel.attention.strict_ring_state.v1"


class AttentionContractError(ValueError):
    """Raised when attention metadata does not describe a valid invocation."""


class AttentionRole(str, Enum):
    TRAIN = "train"
    INFER = "infer"


class AttentionMode(str, Enum):
    PREFILL = "prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    DECODE = "decode"


class AttentionDType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class AttentionMerge(str, Enum):
    ONLINE_SOFTMAX_LSE = "online_softmax_lse"


class ReductionOrder(str, Enum):
    GLOBAL_BLOCK_INDEX = "global_block_index"


class DowncastPoint(str, Enum):
    FINAL_WRITE = "final_write"


class ReductionEngine(str, Enum):
    IN_OP_REFERENCE = "in_op_reference"


class SplitKVMode(str, Enum):
    DISABLED = "disabled"
    FIXED = "fixed"
    AUTO = "auto"


class ProjectionCollective(str, Enum):
    NONE = "none"
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"


class RoPEState(str, Enum):
    PRE_ROPE = "pre_rope"
    POST_ROPE = "post_rope"


class RoPECastPoint(str, Enum):
    NONE = "none"
    BEFORE_ROPE = "before_rope"
    AFTER_ROPE = "after_rope"


class RoPEFusionBoundary(str, Enum):
    UNFUSED_ROPE_ATTENTION = "unfused_rope_attention"
    FUSED_ROPE_ATTENTION = "fused_rope_attention"


def _enum_value(enum_type: type[_EnumT], value: Any, field: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise AttentionContractError(f"{field} must be one of: {allowed}; got {value!r}") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttentionContractError(f"{field} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AttentionContractError(f"{field} must be a non-negative integer; got {value!r}")
    return value


def _integer_tuple(values: Iterable[int], field: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise AttentionContractError(f"{field} must be an iterable of integers") from exc
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, int):
            raise AttentionContractError(f"{field}[{index}] must be an integer; got {value!r}")
    return result


@dataclass(frozen=True)
class ShardingSpec:
    """Logical TP/CP ownership for one attention invocation.

    TP head shards are currently required to be equal and contiguous.  CP
    sequence ownership may be uneven, but every local block must carry a stable
    logical global index so a later implementation can merge by logical order
    instead of collective arrival order.
    """

    tp_rank: int
    tp_world_size: int
    cp_rank: int
    cp_world_size: int
    global_q_heads: int
    global_kv_heads: int
    local_q_head_start: int
    local_q_heads: int
    local_kv_head_start: int
    local_kv_heads: int
    global_sequence_length: int
    local_sequence_length: int
    global_block_indices: tuple[int, ...]
    global_block_token_starts: tuple[int, ...]
    local_block_offsets: tuple[int, ...]
    packed_sequence_offsets: tuple[int, ...] | None = None
    sp_rank: int = 0
    sp_world_size: int = 1

    def __post_init__(self) -> None:
        tp_world_size = _positive_int(self.tp_world_size, "tp_world_size")
        cp_world_size = _positive_int(self.cp_world_size, "cp_world_size")
        sp_world_size = _positive_int(self.sp_world_size, "sp_world_size")
        tp_rank = _non_negative_int(self.tp_rank, "tp_rank")
        cp_rank = _non_negative_int(self.cp_rank, "cp_rank")
        sp_rank = _non_negative_int(self.sp_rank, "sp_rank")
        tp_rank = _non_negative_int(self.tp_rank, "tp_rank")
        cp_rank = _non_negative_int(self.cp_rank, "cp_rank")
        if tp_rank >= tp_world_size:
            raise AttentionContractError(
                f"tp_rank={tp_rank} must be smaller than tp_world_size={tp_world_size}"
            )
        if cp_rank >= cp_world_size:
            raise AttentionContractError(
                f"cp_rank={cp_rank} must be smaller than cp_world_size={cp_world_size}"
            )
        if sp_rank >= sp_world_size:
            raise AttentionContractError(
                f"sp_rank={sp_rank} must be smaller than sp_world_size={sp_world_size}"
            )

        global_q_heads = _positive_int(self.global_q_heads, "global_q_heads")
        global_kv_heads = _positive_int(self.global_kv_heads, "global_kv_heads")
        if global_q_heads % global_kv_heads != 0:
            raise AttentionContractError(
                f"global_q_heads={global_q_heads} must be divisible by "
                f"global_kv_heads={global_kv_heads} for GQA"
            )
        if global_q_heads % tp_world_size != 0 or global_kv_heads % tp_world_size != 0:
            raise AttentionContractError(
                "global Q/KV heads must be evenly divisible by tp_world_size; "
                f"got Hq={global_q_heads}, Hkv={global_kv_heads}, TP={tp_world_size}"
            )

        local_q_heads = _positive_int(self.local_q_heads, "local_q_heads")
        local_kv_heads = _positive_int(self.local_kv_heads, "local_kv_heads")
        expected_q_heads = global_q_heads // tp_world_size
        expected_kv_heads = global_kv_heads // tp_world_size
        if local_q_heads != expected_q_heads or local_kv_heads != expected_kv_heads:
            raise AttentionContractError(
                "local TP head counts do not preserve the global Q/KV mapping; "
                f"expected ({expected_q_heads}, {expected_kv_heads}), got "
                f"({local_q_heads}, {local_kv_heads})"
            )

        local_q_head_start = _non_negative_int(self.local_q_head_start, "local_q_head_start")
        local_kv_head_start = _non_negative_int(self.local_kv_head_start, "local_kv_head_start")
        expected_q_start = tp_rank * expected_q_heads
        expected_kv_start = tp_rank * expected_kv_heads
        if local_q_head_start != expected_q_start or local_kv_head_start != expected_kv_start:
            raise AttentionContractError(
                "local TP head starts do not match contiguous rank ownership; "
                f"expected ({expected_q_start}, {expected_kv_start}), got "
                f"({local_q_head_start}, {local_kv_head_start})"
            )

        global_sequence_length = _positive_int(
            self.global_sequence_length, "global_sequence_length"
        )
        local_sequence_length = _positive_int(self.local_sequence_length, "local_sequence_length")

        block_indices = _integer_tuple(self.global_block_indices, "global_block_indices")
        if not block_indices:
            raise AttentionContractError("global_block_indices must not be empty")
        if any(index < 0 for index in block_indices):
            raise AttentionContractError("global_block_indices must be non-negative")
        if any(
            left >= right for left, right in zip(block_indices, block_indices[1:], strict=False)
        ):
            raise AttentionContractError(
                "global_block_indices must be unique and strictly increasing"
            )
        object.__setattr__(self, "global_block_indices", block_indices)

        block_token_starts = _integer_tuple(
            self.global_block_token_starts, "global_block_token_starts"
        )
        local_block_offsets = _integer_tuple(self.local_block_offsets, "local_block_offsets")
        if len(block_token_starts) != len(block_indices):
            raise AttentionContractError(
                "global_block_token_starts must contain one entry per global_block_indices entry"
            )
        if any(start < 0 for start in block_token_starts):
            raise AttentionContractError("global_block_token_starts must be non-negative")
        if len(local_block_offsets) != len(block_indices) + 1:
            raise AttentionContractError(
                "local_block_offsets must contain one boundary more than global_block_indices"
            )
        if local_block_offsets[0] != 0 or local_block_offsets[-1] != local_sequence_length:
            raise AttentionContractError(
                "local_block_offsets must start at 0 and end at local_sequence_length"
            )
        if any(
            left >= right
            for left, right in zip(local_block_offsets, local_block_offsets[1:], strict=False)
        ):
            raise AttentionContractError("local_block_offsets must be strictly increasing")

        previous_global_end = 0
        for index, (global_start, local_start, local_end) in enumerate(
            zip(
                block_token_starts,
                local_block_offsets[:-1],
                local_block_offsets[1:],
                strict=True,
            )
        ):
            global_end = global_start + (local_end - local_start)
            if global_end > global_sequence_length:
                raise AttentionContractError(
                    f"global block {block_indices[index]} exceeds global_sequence_length"
                )
            if index > 0 and global_start < previous_global_end:
                raise AttentionContractError(
                    "global block token ranges must be non-overlapping and ordered"
                )
            previous_global_end = global_end
        object.__setattr__(self, "global_block_token_starts", block_token_starts)
        object.__setattr__(self, "local_block_offsets", local_block_offsets)

        if self.packed_sequence_offsets is not None:
            offsets = _integer_tuple(self.packed_sequence_offsets, "packed_sequence_offsets")
            if len(offsets) < 2 or offsets[0] != 0:
                raise AttentionContractError(
                    "packed_sequence_offsets must start at 0 and contain an end offset"
                )
            if any(left >= right for left, right in zip(offsets, offsets[1:], strict=False)):
                raise AttentionContractError("packed_sequence_offsets must be strictly increasing")
            if offsets[-1] != local_sequence_length:
                raise AttentionContractError(
                    "the final packed_sequence_offsets value must equal local_sequence_length; "
                    f"got {offsets[-1]} and {local_sequence_length}"
                )
            object.__setattr__(self, "packed_sequence_offsets", offsets)


@dataclass(frozen=True)
class ReductionSpec:
    """Deterministic CP ``(out, lse)`` merge semantics."""

    merge: AttentionMerge = AttentionMerge.ONLINE_SOFTMAX_LSE
    acc_dtype: AttentionDType = AttentionDType.FP32
    order: ReductionOrder = ReductionOrder.GLOBAL_BLOCK_INDEX
    downcast_at: DowncastPoint = DowncastPoint.FINAL_WRITE
    engine: ReductionEngine = ReductionEngine.IN_OP_REFERENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "merge", _enum_value(AttentionMerge, self.merge, "merge"))
        object.__setattr__(
            self, "acc_dtype", _enum_value(AttentionDType, self.acc_dtype, "acc_dtype")
        )
        object.__setattr__(self, "order", _enum_value(ReductionOrder, self.order, "order"))
        object.__setattr__(
            self, "downcast_at", _enum_value(DowncastPoint, self.downcast_at, "downcast_at")
        )
        object.__setattr__(self, "engine", _enum_value(ReductionEngine, self.engine, "engine"))
        if self.acc_dtype is not AttentionDType.FP32:
            raise AttentionContractError(
                f"CP attention accumulation must be fp32; got {self.acc_dtype.value}"
            )


@dataclass(frozen=True)
class SplitKVExecutionPlan:
    """Actual backend-local Split-KV schedule emitted by a runtime.

    CP ownership is intentionally separate from this plan.  ``boundaries``
    describe the canonical logical KV ranges reduced by one backend invocation;
    CP may transport those partial states between ranks but must not change the
    FP32 merge contract recorded here.
    """

    requested_mode: SplitKVMode
    requested_split_size: int | None
    actual_mode: SplitKVMode | None
    actual_split_size: int | None
    boundaries: tuple[tuple[int, int], ...]
    merge_order: ReductionOrder = ReductionOrder.GLOBAL_BLOCK_INDEX
    acc_dtype: AttentionDType = AttentionDType.FP32
    downcast_at: DowncastPoint = DowncastPoint.FINAL_WRITE
    backend: str = "reference"
    source: str = "contract_exact"
    fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_mode",
            _enum_value(SplitKVMode, self.requested_mode, "requested_mode"),
        )
        if self.actual_mode is not None:
            object.__setattr__(
                self,
                "actual_mode",
                _enum_value(SplitKVMode, self.actual_mode, "actual_mode"),
            )
        for field_name in ("requested_split_size", "actual_split_size"):
            value = getattr(self, field_name)
            if value is not None:
                _positive_int(value, field_name)
        if self.requested_mode is SplitKVMode.FIXED:
            if self.requested_split_size is None:
                raise AttentionContractError(
                    "requested fixed Split-KV mode requires requested_split_size"
                )
        elif self.requested_split_size is not None:
            raise AttentionContractError(
                "requested_split_size is only valid for requested fixed Split-KV mode"
            )
        try:
            boundaries = tuple(tuple(boundary) for boundary in self.boundaries)
        except TypeError as exc:
            raise AttentionContractError(
                "Split-KV boundaries must be an iterable of (start, end) pairs"
            ) from exc
        previous_end = 0
        for index, boundary in enumerate(boundaries):
            if len(boundary) != 2:
                raise AttentionContractError(
                    f"Split-KV boundary {index} must contain exactly start and end"
                )
            start, end = boundary
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                raise AttentionContractError("Split-KV boundaries must satisfy 0 <= start < end")
            if index > 0 and start != previous_end:
                raise AttentionContractError(
                    "Split-KV boundaries must be contiguous and in logical KV order"
                )
            previous_end = end
        if self.actual_mode is None and boundaries:
            raise AttentionContractError("unknown actual Split-KV plan cannot declare boundaries")
        if self.actual_mode is not None and not boundaries:
            raise AttentionContractError("actual Split-KV plan must declare logical boundaries")
        if self.actual_mode is SplitKVMode.FIXED:
            if self.actual_split_size is None:
                raise AttentionContractError(
                    "actual fixed Split-KV mode requires actual_split_size"
                )
            widths = tuple(end - start for start, end in boundaries)
            if any(width != self.actual_split_size for width in widths[:-1]) or (
                widths and widths[-1] > self.actual_split_size
            ):
                raise AttentionContractError(
                    "fixed Split-KV boundaries must use actual_split_size except "
                    "for a shorter final split"
                )
        elif self.actual_split_size is not None:
            raise AttentionContractError(
                "actual_split_size is only valid for actual fixed Split-KV mode"
            )
        if self.actual_mode is SplitKVMode.DISABLED and len(boundaries) != 1:
            raise AttentionContractError(
                "disabled Split-KV execution must contain exactly one boundary"
            )
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(
            self,
            "merge_order",
            _enum_value(ReductionOrder, self.merge_order, "merge_order"),
        )
        object.__setattr__(
            self,
            "acc_dtype",
            _enum_value(AttentionDType, self.acc_dtype, "acc_dtype"),
        )
        object.__setattr__(
            self,
            "downcast_at",
            _enum_value(DowncastPoint, self.downcast_at, "downcast_at"),
        )
        if self.acc_dtype is not AttentionDType.FP32:
            raise AttentionContractError("Split-KV partial states must be merged in fp32")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise AttentionContractError("Split-KV backend must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise AttentionContractError("Split-KV plan source must be a non-empty string")
        if not isinstance(self.fallback, bool):
            raise AttentionContractError("Split-KV fallback must be a bool")
        if self.fallback and not self.fallback_reason:
            raise AttentionContractError("Split-KV fallback_reason is required for a fallback")
        if not self.fallback and self.fallback_reason is not None:
            raise AttentionContractError(
                "Split-KV fallback_reason must be None when fallback=False"
            )
        if not self.fallback and self.actual_mode is not None:
            if self.actual_mode is not self.requested_mode:
                raise AttentionContractError(
                    "actual Split-KV mode may differ from requested mode only for a fallback"
                )
            if self.actual_split_size != self.requested_split_size:
                raise AttentionContractError(
                    "actual Split-KV size may differ from requested size only for a fallback"
                )

    @property
    def actual_split_count(self) -> int | None:
        return None if self.actual_mode is None else len(self.boundaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_split_kv_policy": self.requested_mode.value,
            "requested_split_kv_size": self.requested_split_size,
            "actual_split_kv_policy": (
                None if self.actual_mode is None else self.actual_mode.value
            ),
            "actual_split_kv_size": self.actual_split_size,
            "actual_split_kv_count": self.actual_split_count,
            "actual_split_boundaries": [list(boundary) for boundary in self.boundaries],
            "split_kv_merge_order": self.merge_order.value,
            "split_kv_accum_dtype": self.acc_dtype.value,
            "split_kv_downcast_at": self.downcast_at.value,
            "split_kv_backend": self.backend,
            "split_kv_plan_source": self.source,
            "split_kv_fallback": self.fallback,
            "split_kv_fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class SplitKVSpec:
    """Requested Split-KV policy shared by training and rollout paths."""

    mode: SplitKVMode = SplitKVMode.DISABLED
    fixed_split_size: int | None = None
    strict_consistency: bool = True

    @classmethod
    def disabled(cls, *, strict_consistency: bool = True) -> "SplitKVSpec":
        return cls(mode=SplitKVMode.DISABLED, strict_consistency=strict_consistency)

    @classmethod
    def fixed(cls, split_size: int, *, strict_consistency: bool = True) -> "SplitKVSpec":
        return cls(
            mode=SplitKVMode.FIXED,
            fixed_split_size=split_size,
            strict_consistency=strict_consistency,
        )

    @classmethod
    def auto(cls, *, strict_consistency: bool = False) -> "SplitKVSpec":
        return cls(mode=SplitKVMode.AUTO, strict_consistency=strict_consistency)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _enum_value(SplitKVMode, self.mode, "split_kv.mode"))
        if not isinstance(self.strict_consistency, bool):
            raise AttentionContractError("split_kv.strict_consistency must be a bool")
        if self.mode is SplitKVMode.FIXED:
            if self.fixed_split_size is None:
                raise AttentionContractError("fixed Split-KV policy requires fixed_split_size")
            _positive_int(self.fixed_split_size, "split_kv.fixed_split_size")
        elif self.fixed_split_size is not None:
            raise AttentionContractError("fixed_split_size is only valid for fixed Split-KV policy")
        if self.strict_consistency and self.mode is SplitKVMode.AUTO:
            raise AttentionContractError(
                "auto Split-KV is runtime-shape dependent and is not allowed in strict consistency"
            )

    def resolve(self, total_kv_tokens: int, *, backend: str) -> SplitKVExecutionPlan:
        """Resolve policies whose logical schedule is fully known by contract."""

        total_kv_tokens = _positive_int(total_kv_tokens, "total_kv_tokens")
        if self.mode is SplitKVMode.AUTO:
            return SplitKVExecutionPlan(
                requested_mode=self.mode,
                requested_split_size=None,
                actual_mode=None,
                actual_split_size=None,
                boundaries=(),
                backend=backend,
                source="runtime_required",
            )
        split_size = total_kv_tokens if self.mode is SplitKVMode.DISABLED else self.fixed_split_size
        assert split_size is not None
        boundaries = tuple(
            (start, min(start + split_size, total_kv_tokens))
            for start in range(0, total_kv_tokens, split_size)
        )
        return SplitKVExecutionPlan(
            requested_mode=self.mode,
            requested_split_size=self.fixed_split_size,
            actual_mode=self.mode,
            actual_split_size=self.fixed_split_size,
            boundaries=boundaries,
            backend=backend,
            source="contract_exact",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "fixed_split_size": self.fixed_split_size,
            "strict_consistency": self.strict_consistency,
        }


def validate_split_kv_alignment(
    training: SplitKVExecutionPlan,
    rollout: SplitKVExecutionPlan,
) -> None:
    """Fail closed unless train and rollout executed the same logical plan."""

    if training.actual_mode is None or rollout.actual_mode is None:
        raise AttentionContractError(
            "strict Split-KV alignment requires actual runtime plans from both sides"
        )
    fields = (
        "requested_mode",
        "requested_split_size",
        "actual_mode",
        "actual_split_size",
        "boundaries",
        "merge_order",
        "acc_dtype",
        "downcast_at",
        "fallback",
    )
    mismatches = [
        field_name
        for field_name in fields
        if getattr(training, field_name) != getattr(rollout, field_name)
    ]
    if mismatches:
        raise AttentionContractError(
            "training/rollout Split-KV execution plans differ: " + ", ".join(mismatches)
        )


# What a bitwise cross-config comparison actually requires.
#
# TP degree and batch size are deliberately absent.  The qualified ROCm vendor
# core (AITER/CK dense MHA) has a launch-shape-dependent reduction order, so
# both would otherwise change the bits.  Measured on MI300X (BF16, Hq=32/
# Hkv=8/D=128, causal), raw AITER differs by up to 1.5625e-02 between a batch
# and its rows submitted singly, and by up to 7.8125e-03 between TP degrees.
# The Vime provider removes the dependence by pinning every launch to one batch
# row and one KV group, at roughly 3x forward time, so a head shard's result no
# longer depends on the batch size or TP degree that produced it.  Requiring
# those to match would therefore reject comparisons that are in fact bitwise
# equal.  What remains here is the set that genuinely changes the arithmetic.
CROSS_CONFIG_BOUND_FIELDS = ("dtype", "head_dim", "causal", "export_lse")


def validate_cross_config_alignment(
    training: "AttentionContract",
    rollout: "AttentionContract",
) -> None:
    """Fail closed unless both sides describe one comparable attention invocation.

    Names the field that diverged, so a cross-config drift investigation does
    not start from one opaque fingerprint mismatch.
    """

    if not isinstance(training, AttentionContract) or not isinstance(rollout, AttentionContract):
        raise AttentionContractError("both sides must be AttentionContract instances")

    layout_mismatches = [
        name
        for name in ("global_q_heads", "global_kv_heads")
        if getattr(training.sharding, name) != getattr(rollout.sharding, name)
    ]
    if layout_mismatches:
        raise AttentionContractError(
            "training and rollout describe different global head layouts: "
            + ", ".join(layout_mismatches)
        )

    scalar_mismatches = [
        name
        for name in CROSS_CONFIG_BOUND_FIELDS
        if getattr(training, name) != getattr(rollout, name)
    ]
    if scalar_mismatches:
        raise AttentionContractError(
            "training and rollout attention contracts differ: " + ", ".join(scalar_mismatches)
        )

    if training.split_kv != rollout.split_kv:
        raise AttentionContractError("training and rollout Split-KV policies differ")
    if training.reduction != rollout.reduction:
        raise AttentionContractError("training and rollout reduction specs differ")


@dataclass(frozen=True, order=True)
class SplitKVRuntimeCoordinate:
    """Identity of one batch/rank/owner Split-KV runtime plan."""

    batch_index: int
    tp_rank: int
    cp_rank: int
    owner_cp_rank: int

    def validate(
        self,
        *,
        batch_size: int,
        tp_world_size: int,
        cp_world_size: int,
    ) -> None:
        batch_index = _non_negative_int(self.batch_index, "batch_index")
        tp_rank = _non_negative_int(self.tp_rank, "tp_rank")
        cp_rank = _non_negative_int(self.cp_rank, "cp_rank")
        owner_cp_rank = _non_negative_int(self.owner_cp_rank, "owner_cp_rank")
        if batch_index >= batch_size:
            raise AttentionContractError("Split-KV batch_index is outside batch_size")
        if tp_rank >= tp_world_size:
            raise AttentionContractError("Split-KV tp_rank is outside tp_world_size")
        if cp_rank >= cp_world_size:
            raise AttentionContractError("Split-KV cp_rank is outside cp_world_size")
        if owner_cp_rank >= cp_world_size:
            raise AttentionContractError("Split-KV owner_cp_rank is outside cp_world_size")

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_index": self.batch_index,
            "tp_rank": self.tp_rank,
            "cp_rank": self.cp_rank,
            "owner_cp_rank": self.owner_cp_rank,
        }


@dataclass(frozen=True)
class SplitKVRuntimePlanEntry:
    """Actual plan for one batch/TP/CP consumer and logical KV owner."""

    coordinate: SplitKVRuntimeCoordinate
    expected_kv_range: tuple[int, int]
    execution: SplitKVExecutionPlan

    def validate(
        self,
        *,
        batch_size: int,
        tp_world_size: int,
        cp_world_size: int,
        total_kv_tokens: int,
    ) -> None:
        self.coordinate.validate(
            batch_size=batch_size,
            tp_world_size=tp_world_size,
            cp_world_size=cp_world_size,
        )
        try:
            start, end = self.expected_kv_range
        except (TypeError, ValueError) as exc:
            raise AttentionContractError(
                "expected_kv_range must contain exactly (start, end)"
            ) from exc
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > total_kv_tokens
        ):
            raise AttentionContractError(
                "expected_kv_range must satisfy 0 <= start < end <= total_kv_tokens"
            )
        if self.execution.actual_mode is None:
            raise AttentionContractError("complete Split-KV plan sets require actual runtime plans")
        if self.execution.boundaries[0][0] != start or self.execution.boundaries[-1][1] != end:
            raise AttentionContractError(
                "Split-KV execution boundaries must exactly cover expected_kv_range"
            )
        if any(
            boundary_start < start or boundary_end > end
            for boundary_start, boundary_end in self.execution.boundaries
        ):
            raise AttentionContractError("Split-KV execution boundary escapes expected_kv_range")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.coordinate.to_dict(),
            "expected_kv_range": list(self.expected_kv_range),
            **self.execution.to_dict(),
        }


@dataclass(frozen=True)
class SplitKVRuntimePlanSet:
    """Complete actual Split-KV plans across batch, TP, CP, and KV owners.

    Every CP consumer must report the plan used for every CP-owned KV range.
    This duplicates owner plans across consumers intentionally: it detects one
    rank silently choosing a different Split-K schedule or merge policy.
    """

    batch_size: int
    tp_world_size: int
    cp_world_size: int
    total_kv_tokens: tuple[int, ...]
    entries: tuple[SplitKVRuntimePlanEntry, ...]

    def __post_init__(self) -> None:
        batch_size = _positive_int(self.batch_size, "batch_size")
        tp_world_size = _positive_int(self.tp_world_size, "tp_world_size")
        cp_world_size = _positive_int(self.cp_world_size, "cp_world_size")
        totals = _integer_tuple(self.total_kv_tokens, "total_kv_tokens")
        if len(totals) != batch_size or any(total <= 0 for total in totals):
            raise AttentionContractError(
                "total_kv_tokens must contain one positive length per batch item"
            )
        object.__setattr__(self, "total_kv_tokens", totals)
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        expected_coordinates = {
            SplitKVRuntimeCoordinate(batch_index, tp_rank, cp_rank, owner_cp_rank)
            for batch_index in range(batch_size)
            for tp_rank in range(tp_world_size)
            for cp_rank in range(cp_world_size)
            for owner_cp_rank in range(cp_world_size)
        }
        actual_coordinates = [entry.coordinate for entry in entries]
        if len(set(actual_coordinates)) != len(actual_coordinates):
            raise AttentionContractError("Split-KV runtime plan set contains duplicate coordinates")
        missing = expected_coordinates.difference(actual_coordinates)
        extra = set(actual_coordinates).difference(expected_coordinates)
        if missing or extra:
            raise AttentionContractError(
                "Split-KV runtime plan set coordinate coverage is incomplete; "
                f"missing={_format_split_kv_coordinates(missing)}, "
                f"extra={_format_split_kv_coordinates(extra)}"
            )
        for entry in entries:
            entry.validate(
                batch_size=batch_size,
                tp_world_size=tp_world_size,
                cp_world_size=cp_world_size,
                total_kv_tokens=totals[entry.coordinate.batch_index],
            )
        self._validate_owner_coverage()
        self._validate_rank_invariance()

    def _validate_owner_coverage(self) -> None:
        for batch_index, total in enumerate(self.total_kv_tokens):
            for tp_rank in range(self.tp_world_size):
                ranges = []
                for owner_cp_rank in range(self.cp_world_size):
                    matches = [
                        entry
                        for entry in self.entries
                        if entry.coordinate.batch_index == batch_index
                        and entry.coordinate.tp_rank == tp_rank
                        and entry.coordinate.cp_rank == 0
                        and entry.coordinate.owner_cp_rank == owner_cp_rank
                    ]
                    ranges.append(matches[0].expected_kv_range)
                previous_end = 0
                for start, end in ranges:
                    if start != previous_end:
                        raise AttentionContractError(
                            "Split-KV owner ranges must be gap-free in CP owner order"
                        )
                    previous_end = end
                if previous_end != total:
                    raise AttentionContractError(
                        "Split-KV owner ranges do not cover total_kv_tokens"
                    )

    def _validate_rank_invariance(self) -> None:
        for batch_index in range(self.batch_size):
            for owner_cp_rank in range(self.cp_world_size):
                entries = sorted(
                    (
                        entry
                        for entry in self.entries
                        if entry.coordinate.batch_index == batch_index
                        and entry.coordinate.owner_cp_rank == owner_cp_rank
                    ),
                    key=lambda entry: (
                        entry.coordinate.tp_rank,
                        entry.coordinate.cp_rank,
                    ),
                )
                reference = entries[0]
                for entry in entries[1:]:
                    if entry.expected_kv_range != reference.expected_kv_range:
                        raise AttentionContractError(
                            "Split-KV owner range differs across TP/CP consumers"
                        )
                    try:
                        validate_split_kv_alignment(
                            reference.execution,
                            entry.execution,
                        )
                    except AttentionContractError as exc:
                        raise AttentionContractError(
                            "Split-KV plan differs across TP/CP consumers for "
                            f"batch={batch_index}, owner_cp={owner_cp_rank}: {exc}"
                        ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "tp_world_size": self.tp_world_size,
            "cp_world_size": self.cp_world_size,
            "total_kv_tokens": list(self.total_kv_tokens),
            "entries": [
                entry.to_dict()
                for entry in sorted(self.entries, key=lambda entry: entry.coordinate)
            ],
            "coverage": "complete_batch_tp_cp_owner_cartesian_product",
        }


def validate_split_kv_plan_set_alignment(
    training: SplitKVRuntimePlanSet,
    rollout: SplitKVRuntimePlanSet,
) -> None:
    """Fail closed unless complete train/rollout runtime plan sets align."""

    topology_fields = (
        "batch_size",
        "tp_world_size",
        "cp_world_size",
        "total_kv_tokens",
    )
    topology_mismatches = [
        field_name
        for field_name in topology_fields
        if getattr(training, field_name) != getattr(rollout, field_name)
    ]
    if topology_mismatches:
        raise AttentionContractError(
            "training/rollout Split-KV plan-set topology differs: " + ", ".join(topology_mismatches)
        )
    training_by_coordinate = {entry.coordinate: entry for entry in training.entries}
    rollout_by_coordinate = {entry.coordinate: entry for entry in rollout.entries}
    if training_by_coordinate.keys() != rollout_by_coordinate.keys():
        raise AttentionContractError("training/rollout Split-KV plan-set coordinates differ")
    for coordinate in sorted(training_by_coordinate):
        train_entry = training_by_coordinate[coordinate]
        rollout_entry = rollout_by_coordinate[coordinate]
        if train_entry.expected_kv_range != rollout_entry.expected_kv_range:
            raise AttentionContractError(
                f"training/rollout expected KV range differs at {coordinate}"
            )
        try:
            validate_split_kv_alignment(
                train_entry.execution,
                rollout_entry.execution,
            )
        except AttentionContractError as exc:
            raise AttentionContractError(
                f"training/rollout Split-KV plan differs at {coordinate}: {exc}"
            ) from exc


def build_split_kv_runtime_plan_set(
    total_kv_tokens: Iterable[int],
    *,
    tp_world_size: int,
    cp_world_size: int,
    split_kv: SplitKVSpec,
    backend: str = "contract_reference",
) -> SplitKVRuntimePlanSet:
    """Build a complete owner-local plan set for contract tests and adapters."""

    totals = _integer_tuple(total_kv_tokens, "total_kv_tokens")
    if not totals or any(total < cp_world_size for total in totals):
        raise AttentionContractError(
            "contract plan sets require at least one KV token per CP owner"
        )
    tp_world_size = _positive_int(tp_world_size, "tp_world_size")
    cp_world_size = _positive_int(cp_world_size, "cp_world_size")
    if not isinstance(split_kv, SplitKVSpec):
        raise AttentionContractError("split_kv must be a SplitKVSpec")

    entries: list[SplitKVRuntimePlanEntry] = []
    for batch_index, total in enumerate(totals):
        base = total // cp_world_size
        remainder = total % cp_world_size
        owner_ranges: list[tuple[int, int]] = []
        start = 0
        for owner_cp_rank in range(cp_world_size):
            end = start + base + (1 if owner_cp_rank < remainder else 0)
            owner_ranges.append((start, end))
            start = end
        for tp_rank in range(tp_world_size):
            for cp_rank in range(cp_world_size):
                for owner_cp_rank, (owner_start, owner_end) in enumerate(owner_ranges):
                    local_total = owner_end - owner_start
                    local = split_kv.resolve(local_total, backend=backend)
                    execution = SplitKVExecutionPlan(
                        requested_mode=local.requested_mode,
                        requested_split_size=local.requested_split_size,
                        actual_mode=local.actual_mode,
                        actual_split_size=local.actual_split_size,
                        boundaries=tuple(
                            (owner_start + start, owner_start + end)
                            for start, end in local.boundaries
                        ),
                        merge_order=local.merge_order,
                        acc_dtype=local.acc_dtype,
                        downcast_at=local.downcast_at,
                        backend=local.backend,
                        source=local.source,
                        fallback=local.fallback,
                        fallback_reason=local.fallback_reason,
                    )
                    entries.append(
                        SplitKVRuntimePlanEntry(
                            coordinate=SplitKVRuntimeCoordinate(
                                batch_index=batch_index,
                                tp_rank=tp_rank,
                                cp_rank=cp_rank,
                                owner_cp_rank=owner_cp_rank,
                            ),
                            expected_kv_range=(owner_start, owner_end),
                            execution=execution,
                        )
                    )
    return SplitKVRuntimePlanSet(
        batch_size=len(totals),
        tp_world_size=tp_world_size,
        cp_world_size=cp_world_size,
        total_kv_tokens=totals,
        entries=tuple(entries),
    )


def _format_split_kv_coordinates(
    coordinates: Iterable[SplitKVRuntimeCoordinate],
) -> list[dict[str, int]]:
    return [coordinate.to_dict() for coordinate in sorted(coordinates)]


@dataclass(frozen=True)
class KVCacheSpec:
    """Logical identity of the paged/block KV cache used for replay."""

    cache_positions: tuple[int, ...]
    kv_seq_lens: tuple[int, ...]
    block_table: tuple[tuple[int, ...], ...]
    global_token_positions: tuple[int, ...]
    page_size: int
    prefix_cache_enabled: bool = False
    prefix_cache_key: str | None = None
    shared_prefix_page_count: int = 0

    def __post_init__(self) -> None:
        cache_positions = _integer_tuple(self.cache_positions, "cache_positions")
        kv_seq_lens = _integer_tuple(self.kv_seq_lens, "kv_seq_lens")
        page_size = _positive_int(self.page_size, "page_size")
        global_token_positions = _integer_tuple(
            self.global_token_positions, "global_token_positions"
        )
        if not cache_positions or any(position < 0 for position in cache_positions):
            raise AttentionContractError("cache_positions must contain non-negative positions")
        if not kv_seq_lens or any(length <= 0 for length in kv_seq_lens):
            raise AttentionContractError("kv_seq_lens must contain positive sequence lengths")
        if len(cache_positions) != len(kv_seq_lens):
            raise AttentionContractError(
                "cache_positions must contain one entry per kv_seq_lens entry"
            )
        if not global_token_positions or any(position < 0 for position in global_token_positions):
            raise AttentionContractError(
                "global_token_positions must contain non-negative positions"
            )
        if len(global_token_positions) != sum(kv_seq_lens):
            raise AttentionContractError(
                "global_token_positions must describe every logical cached token; "
                f"expected {sum(kv_seq_lens)}, got {len(global_token_positions)}"
            )
        token_offset = 0
        sequence_position_rows: list[tuple[int, ...]] = []
        for sequence_index, sequence_length in enumerate(kv_seq_lens):
            sequence_positions = global_token_positions[
                token_offset : token_offset + sequence_length
            ]
            if any(
                left >= right
                for left, right in zip(sequence_positions, sequence_positions[1:], strict=False)
            ):
                raise AttentionContractError(
                    "global_token_positions must be strictly increasing within each sequence; "
                    f"sequence {sequence_index} is invalid"
                )
            sequence_position_rows.append(sequence_positions)
            token_offset += sequence_length

        for sequence_index, (cache_position, sequence_positions) in enumerate(
            zip(cache_positions, sequence_position_rows, strict=True)
        ):
            terminal_position = sequence_positions[-1]
            if cache_position != terminal_position:
                raise AttentionContractError(
                    "cache_positions must equal the terminal global token position for each "
                    f"sequence; sequence {sequence_index} expected {terminal_position}, "
                    f"got {cache_position}"
                )

        try:
            block_table = tuple(tuple(row) for row in self.block_table)
        except TypeError as exc:
            raise AttentionContractError(
                "block_table must be a two-dimensional integer table"
            ) from exc
        if len(block_table) != len(kv_seq_lens) or any(not row for row in block_table):
            raise AttentionContractError(
                "block_table must contain one non-empty row per kv_seq_lens entry"
            )
        active_block_rows: list[tuple[int, ...]] = []
        for row_index, (row, sequence_length) in enumerate(
            zip(block_table, kv_seq_lens, strict=True)
        ):
            row_active_blocks: list[int] = []
            saw_padding = False
            for column_index, block in enumerate(row):
                if isinstance(block, bool) or not isinstance(block, int) or block < -1:
                    raise AttentionContractError(
                        "block_table entries must be integer block ids or -1 padding; "
                        f"got block_table[{row_index}][{column_index}]={block!r}"
                    )
                if block == -1:
                    saw_padding = True
                    continue
                if saw_padding:
                    raise AttentionContractError(
                        "block_table -1 padding must be trailing; "
                        f"row {row_index} contains an active block after padding"
                    )
                row_active_blocks.append(block)

            expected_blocks = (sequence_length + page_size - 1) // page_size
            if len(row_active_blocks) != expected_blocks:
                raise AttentionContractError(
                    "block_table active page count must match kv_seq_lens and page_size; "
                    f"row {row_index} expected {expected_blocks}, got {len(row_active_blocks)}"
                )
            if len(set(row_active_blocks)) != len(row_active_blocks):
                raise AttentionContractError(
                    f"block_table row {row_index} contains duplicate active page ids"
                )
            active_block_rows.append(tuple(row_active_blocks))

        if not isinstance(self.prefix_cache_enabled, bool):
            raise AttentionContractError("prefix_cache_enabled must be a bool")
        shared_prefix_page_count = _non_negative_int(
            self.shared_prefix_page_count, "shared_prefix_page_count"
        )
        if self.prefix_cache_enabled and not self.prefix_cache_key:
            raise AttentionContractError(
                "prefix_cache_key is required when prefix_cache_enabled=True"
            )
        if not self.prefix_cache_enabled and self.prefix_cache_key is not None:
            raise AttentionContractError(
                "prefix_cache_key must be None when prefix_cache_enabled=False"
            )
        if not self.prefix_cache_enabled and shared_prefix_page_count != 0:
            raise AttentionContractError(
                "shared_prefix_page_count must be 0 when prefix_cache_enabled=False"
            )

        shared_prefix_pages: tuple[int, ...] = ()
        if shared_prefix_page_count > 0:
            if any(len(row_blocks) < shared_prefix_page_count for row_blocks in active_block_rows):
                raise AttentionContractError(
                    "shared_prefix_page_count exceeds an active block-table row"
                )
            shared_prefix_token_count = shared_prefix_page_count * page_size
            if any(length < shared_prefix_token_count for length in kv_seq_lens):
                raise AttentionContractError(
                    "shared prefix pages must be fully populated and read-only"
                )

            shared_prefix_pages = active_block_rows[0][:shared_prefix_page_count]
            shared_prefix_positions = sequence_position_rows[0][:shared_prefix_token_count]
            for sequence_index, (row_blocks, positions) in enumerate(
                zip(active_block_rows[1:], sequence_position_rows[1:], strict=True), start=1
            ):
                if row_blocks[:shared_prefix_page_count] != shared_prefix_pages:
                    raise AttentionContractError(
                        "shared prefix page ids must match across every sequence; "
                        f"sequence {sequence_index} is inconsistent"
                    )
                if positions[:shared_prefix_token_count] != shared_prefix_positions:
                    raise AttentionContractError(
                        "shared prefix token positions must match across every sequence; "
                        f"sequence {sequence_index} is inconsistent"
                    )

        exclusive_page_owners: dict[int, int] = {}
        shared_prefix_page_ids = set(shared_prefix_pages)
        for sequence_index, row_blocks in enumerate(active_block_rows):
            for page_id in row_blocks[shared_prefix_page_count:]:
                if page_id in shared_prefix_page_ids:
                    raise AttentionContractError(
                        "a writable suffix page cannot alias a read-only shared prefix page"
                    )
                previous_owner = exclusive_page_owners.get(page_id)
                if previous_owner is not None:
                    raise AttentionContractError(
                        "active pages may be shared across sequences only when declared as "
                        "read-only prefix pages; "
                        f"page {page_id} is used by sequences {previous_owner} and "
                        f"{sequence_index}"
                    )
                exclusive_page_owners[page_id] = sequence_index

        object.__setattr__(self, "cache_positions", cache_positions)
        object.__setattr__(self, "kv_seq_lens", kv_seq_lens)
        object.__setattr__(self, "block_table", block_table)
        object.__setattr__(self, "global_token_positions", global_token_positions)


@dataclass(frozen=True)
class RoPESpec:
    """Qwen3 RoPE identity for attention inputs and KV-cache replay.

    The CP attention reference consumes attention inputs.  This object records
    whether those inputs are pre- or post-RoPE and pins the position/cache
    metadata needed to compare fused ``RoPE+Attention`` and unfused
    ``RoPE -> Attention`` materializations.
    """

    q_state: RoPEState = RoPEState.POST_ROPE
    k_state: RoPEState = RoPEState.POST_ROPE
    k_cache_state: RoPEState = RoPEState.POST_ROPE
    theta: float = 1.0e6
    rotary_dim: int = 128
    rope_scaling: str | None = None
    position_ids: tuple[int, ...] | None = None
    query_position_offsets: tuple[int, ...] | None = None
    key_position_offsets: tuple[int, ...] | None = None
    cast_at: RoPECastPoint = RoPECastPoint.AFTER_ROPE
    output_dtype: AttentionDType = AttentionDType.BF16
    fusion_boundary: RoPEFusionBoundary = RoPEFusionBoundary.UNFUSED_ROPE_ATTENTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_state", _enum_value(RoPEState, self.q_state, "q_state"))
        object.__setattr__(self, "k_state", _enum_value(RoPEState, self.k_state, "k_state"))
        object.__setattr__(
            self, "k_cache_state", _enum_value(RoPEState, self.k_cache_state, "k_cache_state")
        )
        if isinstance(self.theta, bool) or not isinstance(self.theta, (float, int)):
            raise AttentionContractError(f"theta must be a positive number; got {self.theta!r}")
        theta = float(self.theta)
        if theta <= 0.0:
            raise AttentionContractError(f"theta must be a positive number; got {self.theta!r}")
        object.__setattr__(self, "theta", theta)
        _positive_int(self.rotary_dim, "rotary_dim")
        if self.rope_scaling is not None and (
            not isinstance(self.rope_scaling, str) or not self.rope_scaling.strip()
        ):
            raise AttentionContractError("rope_scaling must be a non-empty string when provided")
        for position_field in ("position_ids", "query_position_offsets", "key_position_offsets"):
            values = getattr(self, position_field)
            if values is None:
                continue
            normalized = _integer_tuple(values, position_field)
            if not normalized or any(value < 0 for value in normalized):
                raise AttentionContractError(
                    f"{position_field} must contain non-negative positions"
                )
            object.__setattr__(self, position_field, normalized)
        object.__setattr__(self, "cast_at", _enum_value(RoPECastPoint, self.cast_at, "cast_at"))
        object.__setattr__(
            self, "output_dtype", _enum_value(AttentionDType, self.output_dtype, "output_dtype")
        )
        object.__setattr__(
            self,
            "fusion_boundary",
            _enum_value(RoPEFusionBoundary, self.fusion_boundary, "fusion_boundary"),
        )


@dataclass(frozen=True)
class AttentionProjectionSpec:
    """Deterministic QKV or output-projection execution contract."""

    name: str
    input_dtype: AttentionDType = AttentionDType.BF16
    output_dtype: AttentionDType = AttentionDType.BF16
    acc_dtype: AttentionDType = AttentionDType.FP32
    split_kv: SplitKVMode = SplitKVMode.DISABLED
    k_order: str = "ascending"
    backend_policy: str = "native_verified_then_common_deterministic"
    deterministic_backend: str = "rlkernel.cuda.det_gemm"
    tp_forward_collective: ProjectionCollective = ProjectionCollective.NONE
    tp_backward_dgrad_collective: ProjectionCollective = ProjectionCollective.NONE
    sp_forward_collective: ProjectionCollective = ProjectionCollective.NONE
    sp_backward_collective: ProjectionCollective = ProjectionCollective.NONE
    qkv_split_order: tuple[str, ...] = ()
    require_runtime_readback: bool = True

    @classmethod
    def qkv(cls) -> "AttentionProjectionSpec":
        return cls(
            name="qkv",
            tp_backward_dgrad_collective=ProjectionCollective.ALL_REDUCE,
            sp_forward_collective=ProjectionCollective.ALL_GATHER,
            sp_backward_collective=ProjectionCollective.REDUCE_SCATTER,
            qkv_split_order=("q", "k", "v"),
        )

    @classmethod
    def output(cls) -> "AttentionProjectionSpec":
        return cls(
            name="o_proj",
            tp_forward_collective=ProjectionCollective.ALL_REDUCE,
            sp_forward_collective=ProjectionCollective.REDUCE_SCATTER,
            sp_backward_collective=ProjectionCollective.ALL_GATHER,
        )

    def __post_init__(self) -> None:
        if self.name not in {"qkv", "o_proj"}:
            raise AttentionContractError("projection name must be qkv or o_proj")
        for field_name in ("input_dtype", "output_dtype", "acc_dtype"):
            object.__setattr__(
                self,
                field_name,
                _enum_value(AttentionDType, getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "split_kv",
            _enum_value(SplitKVMode, self.split_kv, "projection.split_kv"),
        )
        for field_name in (
            "tp_forward_collective",
            "tp_backward_dgrad_collective",
            "sp_forward_collective",
            "sp_backward_collective",
        ):
            object.__setattr__(
                self,
                field_name,
                _enum_value(
                    ProjectionCollective,
                    getattr(self, field_name),
                    field_name,
                ),
            )
        if self.input_dtype is not AttentionDType.BF16:
            raise AttentionContractError("Attention projections require BF16 input")
        if self.output_dtype is not AttentionDType.BF16:
            raise AttentionContractError("Attention projections require BF16 output")
        if self.acc_dtype is not AttentionDType.FP32:
            raise AttentionContractError("Attention projections require FP32 accumulation")
        if self.split_kv is not SplitKVMode.DISABLED:
            raise AttentionContractError("Attention projection GEMMs must disable Split-K")
        if self.k_order != "ascending":
            raise AttentionContractError("Attention projection GEMMs require ascending K order")
        for field_name in ("backend_policy", "deterministic_backend"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AttentionContractError(f"{field_name} must be a non-empty string")
        if not isinstance(self.require_runtime_readback, bool) or not self.require_runtime_readback:
            raise AttentionContractError("projection runtime readback must be required")
        split_order = tuple(self.qkv_split_order)
        if self.name == "qkv":
            if split_order != ("q", "k", "v"):
                raise AttentionContractError("QKV projection must split in Q, K, V order")
        elif split_order:
            raise AttentionContractError("o_proj must not declare a QKV split order")
        object.__setattr__(self, "qkv_split_order", split_order)


@dataclass(frozen=True)
class AttentionContract:
    """Complete semantic request consumed by contract-aware dispatch."""

    role: AttentionRole
    mode: AttentionMode
    dtype: AttentionDType
    batch_size: int
    query_sequence_length: int
    head_dim: int
    causal: bool
    causal_offsets: tuple[int, ...] | None
    sharding: ShardingSpec
    reduction: ReductionSpec
    split_kv: SplitKVSpec = field(default_factory=SplitKVSpec.disabled)
    kv_cache: KVCacheSpec | None = None
    rope: RoPESpec | None = None
    export_lse: bool = True
    qkv_projection: AttentionProjectionSpec = field(default_factory=AttentionProjectionSpec.qkv)
    output_projection: AttentionProjectionSpec = field(
        default_factory=AttentionProjectionSpec.output
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum_value(AttentionRole, self.role, "role"))
        object.__setattr__(self, "mode", _enum_value(AttentionMode, self.mode, "mode"))
        object.__setattr__(self, "dtype", _enum_value(AttentionDType, self.dtype, "dtype"))
        batch_size = _positive_int(self.batch_size, "batch_size")
        query_sequence_length = _positive_int(self.query_sequence_length, "query_sequence_length")
        _positive_int(self.head_dim, "head_dim")
        if not isinstance(self.sharding, ShardingSpec):
            raise AttentionContractError("sharding must be a ShardingSpec")
        if not isinstance(self.reduction, ReductionSpec):
            raise AttentionContractError("reduction must be a ReductionSpec")
        if not isinstance(self.split_kv, SplitKVSpec):
            raise AttentionContractError("split_kv must be a SplitKVSpec")
        if not isinstance(self.qkv_projection, AttentionProjectionSpec):
            raise AttentionContractError("qkv_projection must be an AttentionProjectionSpec")
        if self.qkv_projection.name != "qkv":
            raise AttentionContractError("qkv_projection must use name='qkv'")
        if not isinstance(self.output_projection, AttentionProjectionSpec):
            raise AttentionContractError("output_projection must be an AttentionProjectionSpec")
        if self.output_projection.name != "o_proj":
            raise AttentionContractError("output_projection must use name='o_proj'")
        if (
            self.mode is AttentionMode.PREFILL
            and query_sequence_length != self.sharding.local_sequence_length
        ):
            raise AttentionContractError(
                "prefill query_sequence_length must equal sharding.local_sequence_length; "
                f"got {query_sequence_length} and {self.sharding.local_sequence_length}"
            )
        if self.sharding.packed_sequence_offsets is not None:
            packed_sequence_count = len(self.sharding.packed_sequence_offsets) - 1
            if packed_sequence_count != batch_size:
                raise AttentionContractError(
                    "packed sequence count must equal logical batch_size; "
                    f"got {packed_sequence_count} packed sequences and batch_size={batch_size}"
                )
        if not isinstance(self.causal, bool):
            raise AttentionContractError("causal must be a bool")
        if self.causal:
            if self.causal_offsets is None:
                raise AttentionContractError("causal_offsets are required for causal attention")
        if self.causal_offsets is not None:
            causal_offsets = _integer_tuple(self.causal_offsets, "causal_offsets")
            if not causal_offsets or any(offset < 0 for offset in causal_offsets):
                raise AttentionContractError("causal_offsets must contain non-negative offsets")
            if self.sharding.packed_sequence_offsets is not None:
                expected_causal_offsets = batch_size
                offset_owner = "packed sequence"
            else:
                expected_causal_offsets = batch_size
                offset_owner = "batch entry"
            if len(causal_offsets) != expected_causal_offsets:
                raise AttentionContractError(
                    f"causal_offsets must contain one entry per {offset_owner}"
                )
            object.__setattr__(self, "causal_offsets", causal_offsets)

        if not isinstance(self.export_lse, bool) or not self.export_lse:
            raise AttentionContractError(
                "export_lse must be True for the WS2 attention-domain LSE contract"
            )

        if self.mode is AttentionMode.DECODE and self.kv_cache is None:
            raise AttentionContractError("kv_cache metadata is required for decode attention")
        if self.kv_cache is not None and not isinstance(self.kv_cache, KVCacheSpec):
            raise AttentionContractError("kv_cache must be a KVCacheSpec when provided")
        if self.rope is not None:
            if not isinstance(self.rope, RoPESpec):
                raise AttentionContractError("rope must be a RoPESpec when provided")
            if self.rope.rotary_dim > self.head_dim:
                raise AttentionContractError(
                    f"rotary_dim={self.rope.rotary_dim} must not exceed head_dim={self.head_dim}"
                )
            if self.rope.position_ids is not None and len(self.rope.position_ids) not in {
                query_sequence_length,
                self.sharding.local_sequence_length,
            }:
                raise AttentionContractError(
                    "position_ids must describe the local query sequence or full local "
                    "sequence length"
                )
            for position_field in ("query_position_offsets", "key_position_offsets"):
                offsets = getattr(self.rope, position_field)
                if offsets is not None and len(offsets) != batch_size:
                    raise AttentionContractError(
                        f"{position_field} must contain one entry per logical batch entry"
                    )
        if self.mode is AttentionMode.DECODE and self.kv_cache is not None:
            if len(self.kv_cache.kv_seq_lens) != batch_size:
                raise AttentionContractError(
                    "decode kv_seq_lens must contain one entry per batch entry"
                )
            if len(self.kv_cache.cache_positions) != batch_size:
                raise AttentionContractError(
                    "decode cache_positions must contain one entry per batch entry"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-compatible requested-contract provenance."""

        sharding = {
            "tp_rank": self.sharding.tp_rank,
            "tp_world_size": self.sharding.tp_world_size,
            "cp_rank": self.sharding.cp_rank,
            "cp_world_size": self.sharding.cp_world_size,
            "sp_rank": self.sharding.sp_rank,
            "sp_world_size": self.sharding.sp_world_size,
            "global_q_heads": self.sharding.global_q_heads,
            "global_kv_heads": self.sharding.global_kv_heads,
            "local_q_head_start": self.sharding.local_q_head_start,
            "local_q_heads": self.sharding.local_q_heads,
            "local_kv_head_start": self.sharding.local_kv_head_start,
            "local_kv_heads": self.sharding.local_kv_heads,
            "global_sequence_length": self.sharding.global_sequence_length,
            "local_sequence_length": self.sharding.local_sequence_length,
            "global_block_indices": list(self.sharding.global_block_indices),
            "global_block_token_starts": list(self.sharding.global_block_token_starts),
            "local_block_offsets": list(self.sharding.local_block_offsets),
            "packed_sequence_offsets": (
                list(self.sharding.packed_sequence_offsets)
                if self.sharding.packed_sequence_offsets is not None
                else None
            ),
        }
        reduction = {
            "merge": self.reduction.merge.value,
            "acc_dtype": self.reduction.acc_dtype.value,
            "order": self.reduction.order.value,
            "downcast_at": self.reduction.downcast_at.value,
            "engine": self.reduction.engine.value,
        }
        kv_cache = None
        if self.kv_cache is not None:
            kv_cache = {
                "cache_positions": list(self.kv_cache.cache_positions),
                "kv_seq_lens": list(self.kv_cache.kv_seq_lens),
                "block_table": [list(row) for row in self.kv_cache.block_table],
                "global_token_positions": list(self.kv_cache.global_token_positions),
                "page_size": self.kv_cache.page_size,
                "prefix_cache_enabled": self.kv_cache.prefix_cache_enabled,
                "prefix_cache_key": self.kv_cache.prefix_cache_key,
                "shared_prefix_page_count": self.kv_cache.shared_prefix_page_count,
            }
        rope = None
        if self.rope is not None:
            rope = {
                "q_state": self.rope.q_state.value,
                "k_state": self.rope.k_state.value,
                "k_cache_state": self.rope.k_cache_state.value,
                "theta": self.rope.theta,
                "rotary_dim": self.rope.rotary_dim,
                "rope_scaling": self.rope.rope_scaling,
                "position_ids": (
                    list(self.rope.position_ids) if self.rope.position_ids is not None else None
                ),
                "query_position_offsets": (
                    list(self.rope.query_position_offsets)
                    if self.rope.query_position_offsets is not None
                    else None
                ),
                "key_position_offsets": (
                    list(self.rope.key_position_offsets)
                    if self.rope.key_position_offsets is not None
                    else None
                ),
                "cast_at": self.rope.cast_at.value,
                "output_dtype": self.rope.output_dtype.value,
                "fusion_boundary": self.rope.fusion_boundary.value,
            }
        projections = {
            spec.name: {
                "input_dtype": spec.input_dtype.value,
                "output_dtype": spec.output_dtype.value,
                "acc_dtype": spec.acc_dtype.value,
                "split_kv": spec.split_kv.value,
                "k_order": spec.k_order,
                "backend_policy": spec.backend_policy,
                "deterministic_backend": spec.deterministic_backend,
                "tp_forward_collective": spec.tp_forward_collective.value,
                "tp_backward_dgrad_collective": spec.tp_backward_dgrad_collective.value,
                "sp_forward_collective": spec.sp_forward_collective.value,
                "sp_backward_collective": spec.sp_backward_collective.value,
                "qkv_split_order": list(spec.qkv_split_order),
                "require_runtime_readback": spec.require_runtime_readback,
            }
            for spec in (self.qkv_projection, self.output_projection)
        }
        return {
            "semantic_operator": "standard_softmax_attention",
            "role": self.role.value,
            "mode": self.mode.value,
            "dtype": self.dtype.value,
            "batch_size": self.batch_size,
            "query_sequence_length": self.query_sequence_length,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "causal_offsets": (
                list(self.causal_offsets) if self.causal_offsets is not None else None
            ),
            "export_lse": self.export_lse,
            "lse_domain": "attention",
            "sharding": sharding,
            "reduction": reduction,
            "split_kv": self.split_kv.to_dict(),
            "kv_cache": kv_cache,
            "rope": rope,
            "projections": projections,
        }

    def cross_rank_fingerprint(self) -> str:
        """Rank-independent identity for preflight agreement across ranks.

        Excludes ``tp_rank``/``cp_rank`` and the local head/sequence bounds they
        derive, so every rank of one logical attention invocation computes the
        same value.  All-gathering this fingerprint together with the resolved
        backend id and aborting on mismatch is the documented preflight for
        distributed dispatch; ``requested_backend="auto"`` is not
        distributed-safe without it.
        """

        payload = self.to_dict()
        payload["sharding"] = {
            key: value
            for key, value in payload["sharding"].items()
            if key
            not in {
                "tp_rank",
                "cp_rank",
                "local_q_head_start",
                "local_q_heads",
                "local_kv_head_start",
                "local_kv_heads",
                "local_sequence_length",
                "global_block_indices",
                "global_block_token_starts",
                "local_block_offsets",
            }
        }
        # Note: Any future extensions to this payload MUST maintain strict JSON
        # serialization determinism across environments to prevent cross-rank
        # hashing mismatches.
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AttentionBackendCapability:
    """Capabilities a concrete backend declares to contract-aware dispatch."""

    backend_id: str
    roles: frozenset[AttentionRole]
    modes: frozenset[AttentionMode]
    dtypes: frozenset[AttentionDType]
    cp_world_sizes: tuple[int, ...]
    tp_world_sizes: tuple[int, ...] | None = None
    exports_attention_lse: bool = False
    deterministic_cp_merge: bool = False
    supports_packed_varlen: bool = False
    supports_kv_cache: bool = False
    supports_rope_metadata: bool = False
    supports_fused_rope_attention: bool = False
    supports_split_kv_disabled: bool = True
    supports_split_kv_fixed: bool = False
    supports_split_kv_auto: bool = False
    reports_actual_split_kv_plan: bool = False
    implementation_kind: str = "production"

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise AttentionContractError("backend_id must be a non-empty string")
        roles = frozenset(_enum_value(AttentionRole, value, "roles") for value in self.roles)
        modes = frozenset(_enum_value(AttentionMode, value, "modes") for value in self.modes)
        dtypes = frozenset(_enum_value(AttentionDType, value, "dtypes") for value in self.dtypes)
        if not roles or not modes or not dtypes:
            raise AttentionContractError("backend roles, modes, and dtypes must not be empty")
        cp_world_sizes = _integer_tuple(self.cp_world_sizes, "cp_world_sizes")
        if not cp_world_sizes or any(size <= 0 for size in cp_world_sizes):
            raise AttentionContractError("cp_world_sizes must contain positive values")
        if len(set(cp_world_sizes)) != len(cp_world_sizes):
            raise AttentionContractError("cp_world_sizes must not contain duplicates")
        tp_world_sizes = None
        if self.tp_world_sizes is not None:
            tp_world_sizes = _integer_tuple(self.tp_world_sizes, "tp_world_sizes")
            if not tp_world_sizes or any(size <= 0 for size in tp_world_sizes):
                raise AttentionContractError("tp_world_sizes must contain positive values")
            if len(set(tp_world_sizes)) != len(tp_world_sizes):
                raise AttentionContractError("tp_world_sizes must not contain duplicates")
        for capability_field in (
            "exports_attention_lse",
            "deterministic_cp_merge",
            "supports_packed_varlen",
            "supports_kv_cache",
            "supports_rope_metadata",
            "supports_fused_rope_attention",
            "supports_split_kv_disabled",
            "supports_split_kv_fixed",
            "supports_split_kv_auto",
            "reports_actual_split_kv_plan",
        ):
            if not isinstance(getattr(self, capability_field), bool):
                raise AttentionContractError(f"{capability_field} must be a bool")
        if self.implementation_kind not in {"production", "reference", "deterministic"}:
            raise AttentionContractError(
                "implementation_kind must be production, reference, or deterministic"
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(self, "cp_world_sizes", cp_world_sizes)
        object.__setattr__(self, "tp_world_sizes", tp_world_sizes)

    def incompatibilities(self, contract: AttentionContract) -> tuple[str, ...]:
        """Explain every reason this backend cannot materialize ``contract``."""

        reasons: list[str] = []
        if contract.role not in self.roles:
            reasons.append(f"role={contract.role.value} is unsupported")
        if contract.mode not in self.modes:
            reasons.append(f"mode={contract.mode.value} is unsupported")
        if contract.dtype not in self.dtypes:
            reasons.append(f"dtype={contract.dtype.value} is unsupported")
        tp_size = contract.sharding.tp_world_size
        cp_size = contract.sharding.cp_world_size
        if self.tp_world_sizes is not None and tp_size not in self.tp_world_sizes:
            reasons.append(f"TP={tp_size} is unsupported")
        if cp_size not in self.cp_world_sizes:
            reasons.append(f"CP={cp_size} is unsupported")
        if contract.export_lse and not self.exports_attention_lse:
            reasons.append("attention-domain LSE export is unsupported")
        if cp_size > 1 and not self.deterministic_cp_merge:
            reasons.append("deterministic CP (out, lse) merge is unsupported")
        if (
            contract.sharding.packed_sequence_offsets is not None
            and not self.supports_packed_varlen
        ):
            reasons.append("packed varlen layout is unsupported")
        if contract.kv_cache is not None and not self.supports_kv_cache:
            reasons.append("KV-cache identity materialization is unsupported")
        if contract.rope is not None and not self.supports_rope_metadata:
            reasons.append("RoPE/position metadata is unsupported")
        if (
            contract.rope is not None
            and contract.rope.fusion_boundary is RoPEFusionBoundary.FUSED_ROPE_ATTENTION
            and not self.supports_fused_rope_attention
        ):
            reasons.append("fused RoPE+Attention boundary is unsupported")
        split_support = {
            SplitKVMode.DISABLED: self.supports_split_kv_disabled,
            SplitKVMode.FIXED: self.supports_split_kv_fixed,
            SplitKVMode.AUTO: self.supports_split_kv_auto,
        }
        if not split_support[contract.split_kv.mode]:
            reasons.append(f"Split-KV policy={contract.split_kv.mode.value} is unsupported")
        if contract.split_kv.strict_consistency and not self.reports_actual_split_kv_plan:
            reasons.append("actual Split-KV execution-plan provenance is unsupported")
        return tuple(reasons)

    def supports(self, contract: AttentionContract) -> bool:
        return not self.incompatibilities(contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "roles": sorted(role.value for role in self.roles),
            "modes": sorted(mode.value for mode in self.modes),
            "dtypes": sorted(dtype.value for dtype in self.dtypes),
            "tp_world_sizes": list(self.tp_world_sizes) if self.tp_world_sizes else None,
            "cp_world_sizes": list(self.cp_world_sizes),
            "exports_attention_lse": self.exports_attention_lse,
            "deterministic_cp_merge": self.deterministic_cp_merge,
            "supports_packed_varlen": self.supports_packed_varlen,
            "supports_kv_cache": self.supports_kv_cache,
            "supports_rope_metadata": self.supports_rope_metadata,
            "supports_fused_rope_attention": self.supports_fused_rope_attention,
            "supports_split_kv_disabled": self.supports_split_kv_disabled,
            "supports_split_kv_fixed": self.supports_split_kv_fixed,
            "supports_split_kv_auto": self.supports_split_kv_auto,
            "reports_actual_split_kv_plan": self.reports_actual_split_kv_plan,
            "implementation_kind": self.implementation_kind,
        }


@dataclass(frozen=True)
class AttentionDispatchResult:
    """A concrete backend plus the actual provenance bound to the request."""

    op: Any
    capability: AttentionBackendCapability
    provenance: dict[str, Any]


__all__ = [
    "AttentionContract",
    "AttentionContractError",
    "AttentionBackendCapability",
    "AttentionDispatchResult",
    "AttentionDType",
    "AttentionProjectionSpec",
    "AttentionMerge",
    "AttentionMode",
    "AttentionRole",
    "DowncastPoint",
    "KVCacheSpec",
    "ReductionEngine",
    "ReductionOrder",
    "ReductionSpec",
    "ProjectionCollective",
    "RoPECastPoint",
    "RoPEFusionBoundary",
    "RoPESpec",
    "RoPEState",
    "ShardingSpec",
    "build_split_kv_runtime_plan_set",
    "SplitKVExecutionPlan",
    "SplitKVMode",
    "SplitKVRuntimeCoordinate",
    "SplitKVRuntimePlanEntry",
    "SplitKVRuntimePlanSet",
    "SplitKVSpec",
    "STRICT_ATTENTION_CORE_ID",
    "STRICT_ATTENTION_FA4_SCHEDULE_ID",
    "STRICT_ATTENTION_PRODUCTION_CORE_ID",
    "STRICT_ATTENTION_REFERENCE_CORE_ID",
    "STRICT_ATTENTION_RING_SCHEDULE_ID",
    "STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID",
    "STRICT_ATTENTION_ROCM_SCHEDULE_ID",
    "STRICT_ATTENTION_SCHEDULE_ID",
    "CROSS_CONFIG_BOUND_FIELDS",
    "validate_cross_config_alignment",
    "validate_split_kv_alignment",
    "validate_split_kv_plan_set_alignment",
]
