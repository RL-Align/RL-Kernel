# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic context-parallel attention reference.

This module is the correctness-first WS2 reference for CP-aware standard
softmax attention. It intentionally stays in PyTorch and uses fp32 partial
states so fused CUDA/Triton backends can validate their CP/LSE merge semantics
against a small, inspectable implementation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_RING_SCHEDULE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    SplitKVExecutionPlan,
    SplitKVMode,
    SplitKVRuntimeCoordinate,
    SplitKVRuntimePlanEntry,
    SplitKVRuntimePlanSet,
    SplitKVSpec,
)
from rl_engine.reference.attention.standard_attn import NativeAttentionOp


@dataclass(frozen=True)
class AttentionPartialState:
    """One KV block's attention state before deterministic LSE merge.

    ``out`` is already normalized within the local KV block and has shape
    ``[B, Hq, Sq, D]``. ``lse`` is the local attention-domain log-sum-exp with
    shape ``[B, Hq, Sq]``. ``block_start`` / ``block_end`` are logical global KV
    positions and define the canonical merge order.
    """

    out: torch.Tensor
    lse: torch.Tensor
    block_start: int
    block_end: int

    def __post_init__(self) -> None:
        if self.out.ndim != 4:
            raise ValueError("partial attention out must have shape [B, Hq, Sq, D]")
        if self.lse.shape != self.out.shape[:3]:
            raise ValueError("partial attention lse must have shape [B, Hq, Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("partial attention out/lse must be on the same device")
        if self.out.dtype is not torch.float32 or self.lse.dtype is not torch.float32:
            raise ValueError("partial attention out/lse must remain FP32 before merge")
        if self.block_start < 0:
            raise ValueError("block_start must be non-negative")
        if self.block_end < self.block_start:
            raise ValueError("block_end must be >= block_start")


@dataclass(frozen=True)
class AttentionRingBlock:
    """One logical KV block in the decoupled Ring Attention schedule."""

    global_block_index: int
    block_start: int
    block_end: int
    owner_cp_rank: int


@dataclass(frozen=True)
class AttentionRingSchedule:
    """Static compute order separated from the fixed arithmetic merge order."""

    schedule_id: str
    total_kv_tokens: int
    cp_world_size: int
    kv_chunk_size: Optional[int]
    blocks: tuple[AttentionRingBlock, ...]
    compute_order: tuple[int, ...]
    merge_order: tuple[int, ...]
    compute_communication: str = "decoupled"
    overlap: str = "disabled"

    @classmethod
    def build(
        cls,
        total_kv_tokens: int,
        *,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> "AttentionRingSchedule":
        if (
            isinstance(cp_world_size, bool)
            or not isinstance(cp_world_size, int)
            or cp_world_size < 1
        ):
            raise ValueError("cp_world_size must be >= 1")
        if kv_chunk_size is not None and (
            isinstance(kv_chunk_size, bool)
            or not isinstance(kv_chunk_size, int)
            or kv_chunk_size < 1
        ):
            raise ValueError("kv_chunk_size must be >= 1 when provided")
        if (
            isinstance(total_kv_tokens, bool)
            or not isinstance(total_kv_tokens, int)
            or total_kv_tokens < 1
        ):
            raise ValueError("total_kv_tokens must be an integer >= 1")
        if total_kv_tokens < cp_world_size:
            raise ValueError("Ring Attention requires at least one KV token per CP rank")
        blocks: list[AttentionRingBlock] = []
        for owner_cp_rank, (owner_start, owner_end) in enumerate(
            _split_bounds(total_kv_tokens, cp_world_size)
        ):
            cursor = owner_start
            while cursor < owner_end:
                block_end = (
                    owner_end if kv_chunk_size is None else min(cursor + kv_chunk_size, owner_end)
                )
                blocks.append(
                    AttentionRingBlock(
                        global_block_index=len(blocks),
                        block_start=cursor,
                        block_end=block_end,
                        owner_cp_rank=owner_cp_rank,
                    )
                )
                cursor = block_end
        compute_order: list[int] = []
        left, right = 0, len(blocks) - 1
        while left <= right:
            compute_order.append(left)
            if left != right:
                compute_order.append(right)
            left += 1
            right -= 1
        return cls(
            schedule_id=STRICT_ATTENTION_RING_SCHEDULE_ID,
            total_kv_tokens=total_kv_tokens,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            blocks=tuple(blocks),
            compute_order=tuple(compute_order),
            merge_order=tuple(range(len(blocks))),
        )

    def provenance(self) -> dict[str, object]:
        return {
            "compute_communication": self.compute_communication,
            "compute_schedule": self.schedule_id,
            "compute_order": list(self.compute_order),
            "merge_order_indices": list(self.merge_order),
            "communication_overlap": self.overlap,
        }


@dataclass(frozen=True)
class DeterministicAttentionCoreResult:
    """Strict-core output and the exact arithmetic plan used for it."""

    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, object]


class DeterministicAttentionCore:
    """Common FP32 Attention arithmetic used by both train and rollout."""

    core_id = STRICT_ATTENTION_CORE_ID
    backend_id = "rlkernel.attention.cp_reference"
    merge_order = "global_block_index"
    accum_dtype = "fp32"
    downcast_at = "final_write"

    def __init__(self, *, split_kv: SplitKVSpec | None = None) -> None:
        self.split_kv = SplitKVSpec.disabled() if split_kv is None else split_kv
        if not isinstance(self.split_kv, SplitKVSpec):
            raise TypeError("split_kv must be a SplitKVSpec")
        if self.split_kv.mode is not SplitKVMode.DISABLED:
            raise ValueError("strict deterministic Attention core requires Split-KV to be disabled")
        self._reference = DeterministicCPAttentionReferenceOp(strict_bitwise=True)

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_position_offsets: torch.Tensor | None = None,
        key_position_offsets: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        out, lse = self._reference.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=1,
            kv_chunk_size=None,
        )
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict Attention output_dtype must be FP16 or BF16")
        plan = self.split_kv.resolve(k.size(2), backend=self.backend_id)
        return DeterministicAttentionCoreResult(
            out=out.to(dtype=resolved_dtype),
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "attention_backend": self.backend_id,
                "split_kv": plan.to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "fallback": False,
                "fallback_reason": None,
                "native_attention_arithmetic": False,
                "strict_schedule": STRICT_ATTENTION_SCHEDULE_ID,
            },
        )


@dataclass(frozen=True)
class AttentionBackwardGradients:
    """Training-side gradients emitted by the CP attention backward reference."""

    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


@dataclass(frozen=True)
class AttentionSavedForwardState:
    """Exact FP32 forward state consumed by the PR8 backward reference."""

    out: torch.Tensor
    lse: torch.Tensor
    causal: bool
    scale: float
    key_padding_mask: Optional[torch.Tensor]
    query_position_offsets: torch.Tensor
    key_position_offsets: torch.Tensor
    cp_world_size: int
    kv_chunk_size: Optional[int]
    query_bounds: tuple[tuple[int, int], ...]
    kv_block_bounds: tuple[tuple[int, int], ...]
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    q_fingerprint: str
    k_fingerprint: str
    v_fingerprint: str
    out_fingerprint: str
    lse_fingerprint: str
    key_padding_mask_fingerprint: Optional[str]
    query_position_offsets_fingerprint: str
    key_position_offsets_fingerprint: str
    strict_bitwise: bool
    strict_schedule: Optional[str]

    def __post_init__(self) -> None:
        if self.out.dtype is not torch.float32 or self.lse.dtype is not torch.float32:
            raise ValueError("saved attention out/lse must be FP32")
        if self.out.ndim != 4 or self.lse.shape != self.out.shape[:3]:
            raise ValueError("saved attention out/lse shapes are invalid")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("saved attention scale must be positive and finite")
        expected_schedule = STRICT_ATTENTION_SCHEDULE_ID if self.strict_bitwise else None
        if self.strict_schedule != expected_schedule:
            raise ValueError("saved attention strict schedule does not match strict_bitwise")


@dataclass(frozen=True)
class AttentionBackwardPathResult:
    """One materialized CP attention backward path."""

    name: str
    out: torch.Tensor
    lse: torch.Tensor
    gradients: AttentionBackwardGradients
    saved_forward_state: AttentionSavedForwardState
    provenance: dict[str, object]


@dataclass(frozen=True)
class GradientDriftStats:
    """Shape-aware absolute drift summary for backward validation reports."""

    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    active_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p95_abs": self.p95_abs,
            "p99_abs": self.p99_abs,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class AttentionBackwardRankDrift:
    """Backward drift for one logical CP rank's sequence ownership."""

    rank: int
    dq: GradientDriftStats
    dk: GradientDriftStats
    dv: GradientDriftStats

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "dq": self.dq.to_dict(),
            "dk": self.dk.to_dict(),
            "dv": self.dv.to_dict(),
        }


@dataclass(frozen=True)
class AttentionBackwardPathDrift:
    """Candidate-vs-reference backward drift for one CP path."""

    candidate_name: str
    dq: GradientDriftStats
    dk: GradientDriftStats
    dv: GradientDriftStats
    out: GradientDriftStats
    lse: GradientDriftStats
    per_rank: tuple[AttentionBackwardRankDrift, ...]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "dq": self.dq.to_dict(),
            "dk": self.dk.to_dict(),
            "dv": self.dv.to_dict(),
            "out": self.out.to_dict(),
            "lse": self.lse.to_dict(),
            "per_rank": [item.to_dict() for item in self.per_rank],
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AttentionBackwardComparisonReport:
    """Structured PR8 report for CP attention gradient drift validation."""

    reference_name: str
    drifts: tuple[AttentionBackwardPathDrift, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_name": self.reference_name,
            "drifts": [drift.to_dict() for drift in self.drifts],
        }


def merge_attention_partial_states(
    states: Sequence[AttentionPartialState],
) -> AttentionPartialState:
    """Merge CP/chunk partial states in logical block order.

    The merge is the online-softmax/LSE merge used by attention, not a plain
    sum. The input order is deliberately ignored: states are sorted by logical
    ``block_start`` so the result depends on global block indices rather than
    arrival order.
    """

    if not states:
        raise ValueError("at least one attention partial state is required")

    ordered = sorted(states, key=lambda item: (item.block_start, item.block_end))
    _validate_merge_shapes_and_ranges(ordered)

    merged = ordered[0]
    merged_out = merged.out.float()
    merged_lse = merged.lse.float()
    for state in ordered[1:]:
        merged_out, merged_lse = _merge_two_states(
            merged_out,
            merged_lse,
            state.out.float(),
            state.lse.float(),
        )

    return AttentionPartialState(
        out=merged_out,
        lse=merged_lse,
        block_start=ordered[0].block_start,
        block_end=ordered[-1].block_end,
    )


class DeterministicCPAttentionReferenceOp:
    """Correctness-first CP attention reference for prefill and chunked prefill.

    The reference consumes attention-ready Q/K. For Qwen3 WS2 this means Q/K
    have already passed QK-Norm and RoPE unless an outer contract explicitly
    marks them as pre-RoPE. RoPE is intentionally kept outside this CP merge
    implementation so fused and unfused ``RoPE+Attention`` paths can compare the
    same post-RoPE Q/K boundary before validating CP communication.

    The op emulates CP by splitting query and KV sequence dimensions into
    logical CP shards. Each query shard computes one partial attention state per
    KV block, then merges those states in fixed global-block order using fp32
    LSE arithmetic. ``forward`` returns the input dtype after the final write;
    ``forward_fp32`` keeps the fp32 merged output.
    """

    op_class = "attention"

    def __init__(self, *, strict_bitwise: bool = False) -> None:
        """Create the reference op.

        ``strict_bitwise`` uses a canonical schedule that is independent of
        batch size and CP ownership.  The regular path remains vectorized for
        drift/performance experiments; the strict path is intentionally slower
        because it is the shared arithmetic reference for train and rollout.
        """

        if not isinstance(strict_bitwise, bool):
            raise TypeError("strict_bitwise must be a bool")
        self.strict_bitwise = strict_bitwise

    @staticmethod
    def split_kv_execution_plans(
        total_kv_tokens: int,
        *,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> list[dict[str, object]]:
        """Export the actual logical Split-KV plan before execution."""

        return split_kv_execution_plan_provenance(
            total_kv_tokens,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            backend="deterministic_cp_reference",
        )

    @staticmethod
    def ring_schedule(
        total_kv_tokens: int,
        *,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> AttentionRingSchedule:
        """Build the production-default pre-overlap Ring schedule."""

        return AttentionRingSchedule.build(
            total_kv_tokens,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )

    @staticmethod
    def execution_provenance(
        total_kv_tokens: int,
        *,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> dict[str, object]:
        """Describe the reference boundary without claiming production communication."""

        plans = split_kv_execution_plan_provenance(
            total_kv_tokens,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            backend="deterministic_cp_reference",
        )
        return {
            "execution_scope": "logical_single_process_cp_reference",
            "runtime_verified": False,
            "input_boundary": "projected_post_qk_norm_post_rope_qkv",
            "query_scope": "logical_global_query_reference",
            "kv_scope": "logical_owner_local_cp_shards",
            "production_cp_protocol": "ag_query_local_kv_rs_out_lse",
            "communication_executed": "none",
            "partial_state": "fp32_out_attention_lse",
            "merge_order": "global_block_index",
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "requested_split_kv_policy": "disabled" if kv_chunk_size is None else "fixed",
            "requested_split_kv_size": kv_chunk_size,
            "actual_split_kv_plans": plans,
        }

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute CP attention with fp32 accumulation and final input-dtype write."""

        out, _ = self.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=q.dtype,
        )
        return out

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute CP attention with fp32 accumulation and fp32 output."""

        out, _ = self.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )
        return out

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(out, lse)`` for the CP reference path.

        ``lse`` is always fp32 and in the attention domain. ``out`` is fp32
        until the final write, then downcast to ``output_dtype``. When omitted,
        ``output_dtype`` defaults to the input dtype.
        """

        resolved_output_dtype = q.dtype if output_dtype is None else output_dtype
        _validate_output_dtype(resolved_output_dtype)
        if self.strict_bitwise:
            out, lse = self._forward_strict_bitwise(
                q,
                k,
                v,
                causal=causal,
                scale=scale,
                key_padding_mask=key_padding_mask,
                query_position_offsets=query_position_offsets,
                key_position_offsets=key_position_offsets,
                cp_world_size=cp_world_size,
                kv_chunk_size=kv_chunk_size,
            )
        else:
            out, lse = self._forward_impl(
                q,
                k,
                v,
                causal=causal,
                scale=scale,
                key_padding_mask=key_padding_mask,
                query_position_offsets=query_position_offsets,
                key_position_offsets=key_position_offsets,
                cp_world_size=cp_world_size,
                kv_chunk_size=kv_chunk_size,
            )
        out = out.to(resolved_output_dtype)
        return out, lse

    def forward_fp32_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fp32 ``(out, lse)`` for the CP reference path."""

        return self.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=torch.float32,
        )

    def _forward_strict_bitwise(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: Optional[float],
        key_padding_mask: Optional[torch.Tensor],
        query_position_offsets: Optional[torch.Tensor],
        key_position_offsets: Optional[torch.Tensor],
        cp_world_size: int,
        kv_chunk_size: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one canonical arithmetic schedule for every caller.

        The regular implementation changes GEMM shapes when batch/CP changes;
        that is numerically valid but cannot be bitwise invariant.  Strict mode
        fixes every matmul to ``[1, H, 1, D]`` by processing one batch row and
        one query position at a time.  KV blocks are global and independent of
        ``cp_world_size``; communication only determines ownership outside this
        reference.  Both training and rollout therefore execute the same
        score, softmax, value and LSE-merge operations in the same order.
        """

        _validate_qkv(q, k, v)
        _validate_scale(scale)
        _validate_partition_args(cp_world_size, kv_chunk_size)
        batch, hq, sq, dim = q.shape
        skv = k.size(2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, skv):
                raise ValueError("key_padding_mask must have shape [B, Skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")

        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            batch,
            q.device,
            default=skv - sq,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            batch,
            q.device,
            default=0,
            name="key_position_offsets",
        )
        if sq == 0:
            zero_dep = _zero_dependency(q.float(), k.float(), v.float())
            return (
                torch.empty(batch, hq, 0, dim, device=q.device, dtype=torch.float32) + zero_dep,
                torch.empty(batch, hq, 0, device=q.device, dtype=torch.float32) + zero_dep,
            )

        # The canonical schedule is global.  CP ownership and arrival order
        # must not change which partial states are generated or merged.
        out_rows: list[torch.Tensor] = []
        lse_rows: list[torch.Tensor] = []
        for batch_index in range(batch):
            q_batch = q[batch_index : batch_index + 1].contiguous()
            k_batch = k[batch_index : batch_index + 1].contiguous()
            v_batch = v[batch_index : batch_index + 1].contiguous()
            pad_batch = (
                None
                if key_padding_mask is None
                else key_padding_mask[batch_index : batch_index + 1].contiguous()
            )
            query_offset = query_offsets[batch_index : batch_index + 1]
            key_offset = key_offsets[batch_index : batch_index + 1]
            query_rows: list[torch.Tensor] = []
            lse_query_rows: list[torch.Tensor] = []
            for query_index in range(sq):
                q_row = q_batch[:, :, query_index : query_index + 1, :].contiguous()
                state = self.local_partial_state(
                    q_row,
                    k_batch,
                    v_batch,
                    q_start=query_index,
                    k_start=0,
                    total_kv_len=skv,
                    total_query_len=sq,
                    causal=causal,
                    scale=scale,
                    key_padding_mask=pad_batch,
                    query_position_offsets=query_offset,
                    key_position_offsets=key_offset,
                )
                query_rows.append(state.out)
                lse_query_rows.append(state.lse)
            out_rows.append(torch.cat(query_rows, dim=2))
            lse_rows.append(torch.cat(lse_query_rows, dim=2))
        return torch.cat(out_rows, dim=0), torch.cat(lse_rows, dim=0)

    def backward_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = torch.float32,
        name: Optional[str] = None,
        saved_forward_state: Optional[AttentionSavedForwardState] = None,
    ) -> AttentionBackwardPathResult:
        """Run the deterministic training-side backward validation path.

        The semantic backward input is ``dout`` plus the forward attention state
        produced from the same Q/K/V, masks, position offsets, CP world, and KV
        block order. The reference keeps the softmax/merge math in fp32 and
        records the final-write dtype in provenance; decode backward is
        intentionally out of scope for PR8.
        """

        _validate_qkv(q, k, v)
        if dout.shape != q.shape:
            raise ValueError("dout must have shape [B, Hq, Sq, D], matching q")
        if not torch.is_floating_point(dout) or torch.is_complex(dout):
            raise ValueError("dout must be a real floating-point tensor")
        if dout.device != q.device:
            raise ValueError("dout must be on the same device as q, k, and v")
        if dout.dtype != q.dtype:
            raise ValueError("dout must have the same dtype as q")
        resolved_output_dtype = q.dtype if output_dtype is None else output_dtype
        _validate_output_dtype(resolved_output_dtype)
        state = saved_forward_state or self.save_forward_state(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )
        _validate_saved_forward_state(
            state,
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            strict_bitwise=self.strict_bitwise,
        )
        gradients = _backward_from_saved_state(
            q,
            k,
            v,
            dout,
            state,
            strict_bitwise=self.strict_bitwise,
        )
        out = state.out.to(resolved_output_dtype)
        lse = state.lse
        ring_schedule = self.ring_schedule(
            k.size(2),
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )

        return AttentionBackwardPathResult(
            name=name
            or _backward_path_name(cp_world_size=cp_world_size, kv_chunk_size=kv_chunk_size),
            out=out.detach(),
            lse=lse.detach(),
            gradients=AttentionBackwardGradients(
                dq=gradients.dq,
                dk=gradients.dk,
                dv=gradients.dv,
            ),
            saved_forward_state=state,
            provenance={
                "attention_mode": "prefill" if kv_chunk_size is None else "chunked_prefill",
                "gradient_mode": "training_backward",
                "gradient_inputs": ["q", "k", "v"],
                "gradient_outputs": ["out"],
                "saved_forward_state": [
                    "out",
                    "attention_lse",
                    "causal_mask",
                    "key_padding_mask",
                    "query_position_offsets",
                    "key_position_offsets",
                    "global_block_index",
                ],
                "cp_world_size": cp_world_size,
                "kv_chunk_size": kv_chunk_size,
                "requested_split_kv_policy": ("disabled" if kv_chunk_size is None else "fixed"),
                "requested_split_kv_size": kv_chunk_size,
                "actual_split_kv_plans": (
                    _strict_no_split_plan_provenance(
                        k.size(2),
                        cp_world_size=cp_world_size,
                        backend="deterministic_cp_backward_strict_reference",
                    )
                    if self.strict_bitwise
                    else split_kv_execution_plan_provenance(
                        k.size(2),
                        cp_world_size=cp_world_size,
                        kv_chunk_size=kv_chunk_size,
                        backend="deterministic_cp_backward_reference",
                    )
                ),
                "merge_order": "global_block_index",
                "accum_dtype": "fp32",
                "downcast_at": "final_write",
                **ring_schedule.provenance(),
                "ring_schedule_default": True,
                "ring_partial_arithmetic": False,
                "strict_bitwise": self.strict_bitwise,
                "strict_core_id": (STRICT_ATTENTION_CORE_ID if self.strict_bitwise else None),
                "strict_schedule": (STRICT_ATTENTION_SCHEDULE_ID if self.strict_bitwise else None),
                "actual_split_kv_policy": (
                    "disabled"
                    if self.strict_bitwise
                    else ("disabled" if kv_chunk_size is None else "fixed")
                ),
                "output_dtype": str(resolved_output_dtype).replace("torch.", ""),
                "q_dtype": str(q.dtype).replace("torch.", ""),
                "k_dtype": str(k.dtype).replace("torch.", ""),
                "v_dtype": str(v.dtype).replace("torch.", ""),
                "dout_dtype": str(dout.dtype).replace("torch.", ""),
                "saved_forward_state_source": (
                    "caller" if saved_forward_state is not None else "captured_reference"
                ),
                "backward_algorithm": (
                    "saved_out_lse_canonical_row_reference"
                    if self.strict_bitwise
                    else "saved_out_lse_block_order_reference"
                ),
                "te_backward_oracle": "not_used",
                "decode_backward": "not_supported",
                "projection_scope": "attention_core_only",
                "qkv_projection_backward_dgrad_collective": "all_reduce",
                "qkv_projection_sp_backward_collective": "reduce_scatter",
                "o_proj_backward_dgrad_collective": "none",
                "o_proj_sp_backward_collective": "all_gather",
                "projection_collectives_executed": False,
                "projection_collectives_source": "attention_contract_runtime_adapter",
            },
        )

    def save_forward_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> AttentionSavedForwardState:
        """Capture the exact state a production training backward must consume."""

        out, lse = self.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )
        batch, _, sq, dim = q.shape
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            batch,
            q.device,
            default=k.size(2) - sq,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            batch,
            q.device,
            default=0,
            name="key_position_offsets",
        )
        mask = None if key_padding_mask is None else key_padding_mask.detach().clone()
        return AttentionSavedForwardState(
            out=out.detach().clone(),
            lse=lse.detach().clone(),
            causal=causal,
            scale=float(scale if scale is not None else 1.0 / math.sqrt(dim)),
            key_padding_mask=mask,
            query_position_offsets=query_offsets.detach().clone(),
            key_position_offsets=key_offsets.detach().clone(),
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            query_bounds=tuple(_split_bounds(sq, cp_world_size)),
            kv_block_bounds=tuple(_kv_block_bounds(k.size(2), cp_world_size, kv_chunk_size)),
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            q_fingerprint=_tensor_fingerprint(q),
            k_fingerprint=_tensor_fingerprint(k),
            v_fingerprint=_tensor_fingerprint(v),
            out_fingerprint=_tensor_fingerprint(out),
            lse_fingerprint=_tensor_fingerprint(lse),
            key_padding_mask_fingerprint=(None if mask is None else _tensor_fingerprint(mask)),
            query_position_offsets_fingerprint=_tensor_fingerprint(query_offsets),
            key_position_offsets_fingerprint=_tensor_fingerprint(key_offsets),
            strict_bitwise=self.strict_bitwise,
            strict_schedule=(STRICT_ATTENTION_SCHEDULE_ID if self.strict_bitwise else None),
        )

    def local_partial_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q_start: int,
        k_start: int,
        total_kv_len: int,
        total_query_len: Optional[int] = None,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
    ) -> AttentionPartialState:
        """Compute one query shard against one logical KV block.

        ``query_position_offsets`` and ``key_position_offsets`` are optional
        per-batch-row base positions. They let the reference express varlen or
        packed metadata while retaining the dense [B, H, S, D] tensor layout.
        For post-RoPE Q/K, these offsets must describe the same absolute token
        positions used when RoPE was applied.
        """

        _validate_qkv(q, k, v)
        _validate_scale(scale)
        if q_start < 0 or k_start < 0:
            raise ValueError("q_start and k_start must be non-negative")
        if total_kv_len < k_start + k.size(2):
            raise ValueError("total_kv_len must cover the local KV block")
        if total_query_len is None:
            total_query_len = q.size(2)
        if total_query_len < q_start + q.size(2):
            raise ValueError("total_query_len must cover the local query block")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (q.size(0), k.size(2)):
                raise ValueError("local key_padding_mask must have shape [B, local_skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("local key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            q.size(0),
            q.device,
            default=total_kv_len - total_query_len,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            q.size(0),
            q.device,
            default=0,
            name="key_position_offsets",
        )

        ctx = NativeAttentionOp._strict_fp32_math(q.device.type)
        with ctx:
            qf = q.float()
            kf = k.float()
            vf = v.float()
            hq, sq, dim = qf.shape[1], qf.shape[2], qf.shape[3]
            hkv, skv = kf.shape[1], kf.shape[2]
            if hkv != hq:
                repeat = hq // hkv
                kf = kf.repeat_interleave(repeat, dim=1)
                vf = vf.repeat_interleave(repeat, dim=1)

            if skv == 0:
                zero_dep = _zero_dependency(qf, kf, vf)
                return AttentionPartialState(
                    out=torch.zeros(q.size(0), hq, sq, dim, device=q.device, dtype=torch.float32)
                    + zero_dep,
                    lse=torch.full(
                        (q.size(0), hq, sq),
                        float("-inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    + zero_dep,
                    block_start=k_start,
                    block_end=k_start,
                )

            scale_value = scale if scale is not None else (1.0 / math.sqrt(dim))
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale_value
            if causal:
                query_base = query_offsets[:, None] + q_start
                key_base = key_offsets[:, None] + k_start
                q_pos = torch.arange(sq, device=q.device, dtype=torch.long) + query_base
                k_pos = torch.arange(skv, device=q.device, dtype=torch.long) + key_base
                causal_mask = k_pos[:, None, :] > q_pos[:, :, None]
                scores = scores.masked_fill(causal_mask[:, None, :, :], float("-inf"))
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

            lse = torch.logsumexp(scores, dim=-1)
            finite_lse = torch.isfinite(lse)
            weights = torch.exp(scores - lse.unsqueeze(-1))
            weights = torch.where(finite_lse.unsqueeze(-1), weights, torch.zeros_like(weights))
            out = torch.matmul(weights, vf)
            return AttentionPartialState(
                out=out,
                lse=lse,
                block_start=k_start,
                block_end=k_start + skv,
            )

    def _forward_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: Optional[float],
        key_padding_mask: Optional[torch.Tensor],
        query_position_offsets: Optional[torch.Tensor],
        key_position_offsets: Optional[torch.Tensor],
        cp_world_size: int,
        kv_chunk_size: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_qkv(q, k, v)
        _validate_scale(scale)
        if (
            isinstance(cp_world_size, bool)
            or not isinstance(cp_world_size, int)
            or cp_world_size < 1
        ):
            raise ValueError("cp_world_size must be >= 1")
        if kv_chunk_size is not None and (
            isinstance(kv_chunk_size, bool)
            or not isinstance(kv_chunk_size, int)
            or kv_chunk_size < 1
        ):
            raise ValueError("kv_chunk_size must be >= 1 when provided")

        batch, hq, sq, dim = q.shape
        skv = k.size(2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, skv):
                raise ValueError("key_padding_mask must have shape [B, Skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            batch,
            q.device,
            default=skv - sq,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            batch,
            q.device,
            default=0,
            name="key_position_offsets",
        )

        q_bounds = _split_bounds(sq, cp_world_size)
        kv_bounds = _kv_block_bounds(skv, cp_world_size, kv_chunk_size)
        out_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for q_start, q_end in q_bounds:
            if q_start == q_end:
                continue
            q_block = q[:, :, q_start:q_end, :]
            states = [
                self.local_partial_state(
                    q_block,
                    k[:, :, k_start:k_end, :],
                    v[:, :, k_start:k_end, :],
                    q_start=q_start,
                    k_start=k_start,
                    total_kv_len=skv,
                    total_query_len=sq,
                    causal=causal,
                    scale=scale,
                    key_padding_mask=(
                        None if key_padding_mask is None else key_padding_mask[:, k_start:k_end]
                    ),
                    query_position_offsets=query_offsets,
                    key_position_offsets=key_offsets,
                )
                for k_start, k_end in kv_bounds
                if k_start != k_end
            ]
            if states:
                merged = merge_attention_partial_states(states)
                out_chunks.append(merged.out)
                lse_chunks.append(merged.lse)
            else:
                zero_dep = _zero_dependency(q_block.float(), k.float(), v.float())
                out_chunks.append(
                    torch.zeros(batch, hq, q_end - q_start, dim, device=q.device) + zero_dep
                )
                lse_chunks.append(
                    torch.full(
                        (batch, hq, q_end - q_start),
                        float("-inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    + zero_dep
                )

        if not out_chunks:
            zero_dep = _zero_dependency(q.float(), k.float(), v.float())
            return (
                torch.empty(batch, hq, 0, dim, device=q.device, dtype=torch.float32) + zero_dep,
                torch.empty(batch, hq, 0, device=q.device, dtype=torch.float32) + zero_dep,
            )
        return torch.cat(out_chunks, dim=2), torch.cat(lse_chunks, dim=2)


def compare_cp_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    query_position_offsets: Optional[torch.Tensor] = None,
    key_position_offsets: Optional[torch.Tensor] = None,
    candidate_cp_world_size: int = 2,
    candidate_kv_chunk_size: Optional[int] = None,
    output_dtype: Optional[torch.dtype] = torch.float32,
) -> AttentionBackwardComparisonReport:
    """Compare CP=1 backward with a CP/chunked-prefill candidate.

    The report includes whole-tensor ``dq/dk/dv`` drift and per-logical-CP-rank
    slices. It is a validation/reporting helper, not a separate production
    backward kernel.
    """

    op = DeterministicCPAttentionReferenceOp()
    reference = op.backward_reference(
        q,
        k,
        v,
        dout,
        causal=causal,
        scale=scale,
        key_padding_mask=key_padding_mask,
        query_position_offsets=query_position_offsets,
        key_position_offsets=key_position_offsets,
        cp_world_size=1,
        kv_chunk_size=None,
        output_dtype=output_dtype,
        name="cp1_backward_reference",
    )
    candidate = op.backward_reference(
        q,
        k,
        v,
        dout,
        causal=causal,
        scale=scale,
        key_padding_mask=key_padding_mask,
        query_position_offsets=query_position_offsets,
        key_position_offsets=key_position_offsets,
        cp_world_size=candidate_cp_world_size,
        kv_chunk_size=candidate_kv_chunk_size,
        output_dtype=output_dtype,
    )
    return AttentionBackwardComparisonReport(
        reference_name=reference.name,
        drifts=(_compare_backward_path(candidate, reference),),
    )


def _backward_from_saved_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    state: AttentionSavedForwardState,
    *,
    strict_bitwise: bool,
) -> AttentionBackwardGradients:
    """Apply standard-softmax backward in canonical global KV-block order."""

    if strict_bitwise:
        return _backward_strict_from_saved_state(q, k, v, dout, state)

    batch, hq, _, dim = q.shape
    hkv = k.size(1)
    group_size = hq // hkv
    with NativeAttentionOp._strict_fp32_math(q.device.type):
        qf = q.float()
        kf = k.float()
        vf = v.float()
        doutf = dout.float()
        k_expanded = kf.repeat_interleave(group_size, dim=1)
        v_expanded = vf.repeat_interleave(group_size, dim=1)
        dq = torch.zeros_like(qf)
        dk_expanded = torch.zeros(
            batch,
            hq,
            k.size(2),
            dim,
            dtype=torch.float32,
            device=q.device,
        )
        dv_expanded = torch.zeros_like(dk_expanded)

        for q_start, q_end in state.query_bounds:
            if q_start == q_end:
                continue
            q_block = qf[:, :, q_start:q_end, :]
            dout_block = doutf[:, :, q_start:q_end, :]
            out_block = state.out[:, :, q_start:q_end, :]
            lse_block = state.lse[:, :, q_start:q_end]
            dq_block = torch.zeros_like(q_block)
            for k_start, k_end in state.kv_block_bounds:
                if k_start == k_end:
                    continue
                k_block = k_expanded[:, :, k_start:k_end, :]
                v_block = v_expanded[:, :, k_start:k_end, :]
                scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * state.scale
                if state.causal:
                    query_base = state.query_position_offsets[:, None] + q_start
                    key_base = state.key_position_offsets[:, None] + k_start
                    q_pos = (
                        torch.arange(
                            q_end - q_start,
                            device=q.device,
                            dtype=torch.long,
                        )
                        + query_base
                    )
                    k_pos = (
                        torch.arange(
                            k_end - k_start,
                            device=q.device,
                            dtype=torch.long,
                        )
                        + key_base
                    )
                    scores = scores.masked_fill(
                        (k_pos[:, None, :] > q_pos[:, :, None])[:, None, :, :],
                        float("-inf"),
                    )
                if state.key_padding_mask is not None:
                    scores = scores.masked_fill(
                        ~state.key_padding_mask[:, None, None, k_start:k_end],
                        float("-inf"),
                    )
                probability = torch.exp(scores - lse_block.unsqueeze(-1))
                probability = torch.where(
                    torch.isfinite(lse_block).unsqueeze(-1),
                    probability,
                    torch.zeros_like(probability),
                )
                dv_expanded[:, :, k_start:k_end, :] += torch.matmul(
                    probability.transpose(-1, -2),
                    dout_block,
                )
                dp = torch.matmul(dout_block, v_block.transpose(-1, -2))
                # The global softmax dot term is dout dot the saved global output.
                ds = probability * (dp - (dout_block * out_block).sum(dim=-1, keepdim=True))
                dq_block += torch.matmul(ds, k_block) * state.scale
                dk_expanded[:, :, k_start:k_end, :] += (
                    torch.matmul(ds.transpose(-1, -2), q_block) * state.scale
                )
            dq[:, :, q_start:q_end, :] = dq_block

        dk = dk_expanded.reshape(
            batch,
            hkv,
            group_size,
            k.size(2),
            dim,
        ).sum(dim=2)
        dv = dv_expanded.reshape(
            batch,
            hkv,
            group_size,
            v.size(2),
            dim,
        ).sum(dim=2)
    return AttentionBackwardGradients(
        dq=dq.to(q.dtype),
        dk=dk.to(k.dtype),
        dv=dv.to(v.dtype),
    )


def _backward_strict_from_saved_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    state: AttentionSavedForwardState,
) -> AttentionBackwardGradients:
    """Run one batch row and the complete logical Q/KV domain at a time."""

    batch, hq, sq, dim = q.shape
    hkv = k.size(1)
    skv = k.size(2)
    group_size = hq // hkv
    if sq == 0:
        return AttentionBackwardGradients(
            dq=torch.zeros_like(q),
            dk=torch.zeros_like(k),
            dv=torch.zeros_like(v),
        )

    dq_rows: list[torch.Tensor] = []
    dk_rows: list[torch.Tensor] = []
    dv_rows: list[torch.Tensor] = []
    with NativeAttentionOp._strict_fp32_math(q.device.type):
        for batch_index in range(batch):
            qf = q[batch_index : batch_index + 1].float().contiguous()
            kf = k[batch_index : batch_index + 1].float().contiguous()
            vf = v[batch_index : batch_index + 1].float().contiguous()
            doutf = dout[batch_index : batch_index + 1].float().contiguous()
            k_expanded = kf.repeat_interleave(group_size, dim=1)
            v_expanded = vf.repeat_interleave(group_size, dim=1)
            scores = torch.matmul(qf, k_expanded.transpose(-1, -2)) * state.scale
            if state.causal:
                q_pos = state.query_position_offsets[batch_index : batch_index + 1, None]
                q_pos = q_pos + torch.arange(sq, device=q.device, dtype=torch.long)
                k_pos = state.key_position_offsets[batch_index : batch_index + 1, None]
                k_pos = k_pos + torch.arange(skv, device=q.device, dtype=torch.long)
                scores = scores.masked_fill(
                    (k_pos[:, None, :] > q_pos[:, :, None])[:, None, :, :],
                    float("-inf"),
                )
            if state.key_padding_mask is not None:
                scores = scores.masked_fill(
                    ~state.key_padding_mask[batch_index : batch_index + 1, None, None, :],
                    float("-inf"),
                )
            lse = state.lse[batch_index : batch_index + 1]
            probability = torch.exp(scores - lse.unsqueeze(-1))
            probability = torch.where(
                torch.isfinite(lse).unsqueeze(-1),
                probability,
                torch.zeros_like(probability),
            )
            dp = torch.matmul(doutf, v_expanded.transpose(-1, -2))
            out = state.out[batch_index : batch_index + 1]
            delta = (doutf * out).sum(dim=-1, keepdim=True)
            ds = probability * (dp - delta)
            dq_rows.append(torch.matmul(ds, k_expanded) * state.scale)
            dk_expanded = torch.matmul(ds.transpose(-1, -2), qf) * state.scale
            dv_expanded = torch.matmul(probability.transpose(-1, -2), doutf)
            dk_rows.append(dk_expanded.reshape(1, hkv, group_size, skv, dim).sum(dim=2))
            dv_rows.append(dv_expanded.reshape(1, hkv, group_size, skv, dim).sum(dim=2))
    return AttentionBackwardGradients(
        dq=torch.cat(dq_rows, dim=0).to(q.dtype),
        dk=torch.cat(dk_rows, dim=0).to(k.dtype),
        dv=torch.cat(dv_rows, dim=0).to(v.dtype),
    )


def _validate_saved_forward_state(
    state: AttentionSavedForwardState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: Optional[float],
    key_padding_mask: Optional[torch.Tensor],
    query_position_offsets: Optional[torch.Tensor],
    key_position_offsets: Optional[torch.Tensor],
    cp_world_size: int,
    kv_chunk_size: Optional[int],
    strict_bitwise: bool,
) -> None:
    if not isinstance(state, AttentionSavedForwardState):
        raise ValueError("saved_forward_state must be an AttentionSavedForwardState")
    expected_scale = float(scale if scale is not None else 1.0 / math.sqrt(q.size(-1)))
    expected_query_offsets = _normalize_position_offsets(
        query_position_offsets,
        q.size(0),
        q.device,
        default=k.size(2) - q.size(2),
        name="query_position_offsets",
    )
    expected_key_offsets = _normalize_position_offsets(
        key_position_offsets,
        q.size(0),
        q.device,
        default=0,
        name="key_position_offsets",
    )
    checks = {
        "out_shape": (tuple(state.out.shape), tuple(q.shape)),
        "lse_shape": (tuple(state.lse.shape), tuple(q.shape[:3])),
        "out_device": (state.out.device, q.device),
        "lse_device": (state.lse.device, q.device),
        "q_shape": (state.q_shape, tuple(q.shape)),
        "k_shape": (state.k_shape, tuple(k.shape)),
        "v_shape": (state.v_shape, tuple(v.shape)),
        "q_dtype": (state.q_dtype, q.dtype),
        "k_dtype": (state.k_dtype, k.dtype),
        "v_dtype": (state.v_dtype, v.dtype),
        "causal": (state.causal, causal),
        "scale": (state.scale, expected_scale),
        "cp_world_size": (state.cp_world_size, cp_world_size),
        "kv_chunk_size": (state.kv_chunk_size, kv_chunk_size),
        "strict_bitwise": (state.strict_bitwise, strict_bitwise),
        "strict_schedule": (
            state.strict_schedule,
            STRICT_ATTENTION_SCHEDULE_ID if strict_bitwise else None,
        ),
        "query_bounds": (state.query_bounds, tuple(_split_bounds(q.size(2), cp_world_size))),
        "kv_block_bounds": (
            state.kv_block_bounds,
            tuple(_kv_block_bounds(k.size(2), cp_world_size, kv_chunk_size)),
        ),
        "q_fingerprint": (state.q_fingerprint, _tensor_fingerprint(q)),
        "k_fingerprint": (state.k_fingerprint, _tensor_fingerprint(k)),
        "v_fingerprint": (state.v_fingerprint, _tensor_fingerprint(v)),
        "out_fingerprint": (state.out_fingerprint, _tensor_fingerprint(state.out)),
        "lse_fingerprint": (state.lse_fingerprint, _tensor_fingerprint(state.lse)),
        "query_position_offsets_fingerprint": (
            state.query_position_offsets_fingerprint,
            _tensor_fingerprint(state.query_position_offsets),
        ),
        "key_position_offsets_fingerprint": (
            state.key_position_offsets_fingerprint,
            _tensor_fingerprint(state.key_position_offsets),
        ),
        "query_position_offsets_device": (
            state.query_position_offsets.device,
            q.device,
        ),
        "key_position_offsets_device": (
            state.key_position_offsets.device,
            q.device,
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if not torch.equal(state.query_position_offsets, expected_query_offsets):
        mismatches.append("query_position_offsets")
    if not torch.equal(state.key_position_offsets, expected_key_offsets):
        mismatches.append("key_position_offsets")
    masks_match = (
        state.key_padding_mask is None
        and key_padding_mask is None
        or state.key_padding_mask is not None
        and key_padding_mask is not None
        and torch.equal(state.key_padding_mask, key_padding_mask)
    )
    if not masks_match:
        mismatches.append("key_padding_mask")
    actual_mask_fingerprint = (
        None if state.key_padding_mask is None else _tensor_fingerprint(state.key_padding_mask)
    )
    if state.key_padding_mask_fingerprint != actual_mask_fingerprint:
        mismatches.append("key_padding_mask_fingerprint")
    if mismatches:
        raise ValueError(
            "saved_forward_state does not match the backward invocation: " + ", ".join(mismatches)
        )


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tensor.device).encode())
    digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _compare_backward_path(
    candidate: AttentionBackwardPathResult,
    reference: AttentionBackwardPathResult,
) -> AttentionBackwardPathDrift:
    cp_world_size = _provenance_int(candidate.provenance, "cp_world_size")
    return AttentionBackwardPathDrift(
        candidate_name=candidate.name,
        dq=_drift_stats(candidate.gradients.dq, reference.gradients.dq),
        dk=_drift_stats(candidate.gradients.dk, reference.gradients.dk),
        dv=_drift_stats(candidate.gradients.dv, reference.gradients.dv),
        out=_drift_stats(candidate.out, reference.out),
        lse=_drift_stats(candidate.lse, reference.lse),
        per_rank=_per_rank_backward_drifts(candidate, reference, cp_world_size),
        provenance=candidate.provenance,
    )


def _per_rank_backward_drifts(
    candidate: AttentionBackwardPathResult,
    reference: AttentionBackwardPathResult,
    cp_world_size: int,
) -> tuple[AttentionBackwardRankDrift, ...]:
    q_bounds = _split_bounds(candidate.gradients.dq.size(2), cp_world_size)
    kv_bounds = _split_bounds(candidate.gradients.dk.size(2), cp_world_size)
    per_rank = []
    for rank, ((q_start, q_end), (kv_start, kv_end)) in enumerate(zip(q_bounds, kv_bounds)):
        per_rank.append(
            AttentionBackwardRankDrift(
                rank=rank,
                dq=_drift_stats(
                    candidate.gradients.dq[:, :, q_start:q_end, :],
                    reference.gradients.dq[:, :, q_start:q_end, :],
                ),
                dk=_drift_stats(
                    candidate.gradients.dk[:, :, kv_start:kv_end, :],
                    reference.gradients.dk[:, :, kv_start:kv_end, :],
                ),
                dv=_drift_stats(
                    candidate.gradients.dv[:, :, kv_start:kv_end, :],
                    reference.gradients.dv[:, :, kv_start:kv_end, :],
                ),
            )
        )
    return tuple(per_rank)


def _drift_stats(candidate: torch.Tensor, reference: torch.Tensor) -> GradientDriftStats:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match "
            f"reference shape {tuple(reference.shape)}"
        )
    diff = (candidate.float() - reference.float()).abs().reshape(-1)
    active_count = int(diff.numel())
    if active_count == 0:
        return GradientDriftStats(0.0, 0.0, 0.0, 0.0, 0)
    return GradientDriftStats(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        p95_abs=float(torch.quantile(diff, 0.95).item()),
        p99_abs=float(torch.quantile(diff, 0.99).item()),
        active_count=active_count,
    )


def _backward_path_name(*, cp_world_size: int, kv_chunk_size: Optional[int]) -> str:
    prefix = f"cp{cp_world_size}"
    if kv_chunk_size is None:
        return f"{prefix}_backward"
    return f"{prefix}_chunked_backward"


def _provenance_int(provenance: dict[str, object], key: str) -> int:
    value = provenance[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"provenance field {key!r} must be an int")
    return value


def _merge_two_states(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = torch.logaddexp(lse_a, lse_b)
    finite = torch.isfinite(merged_lse)
    weight_a = torch.where(finite, torch.exp(lse_a - merged_lse), torch.zeros_like(merged_lse))
    weight_b = torch.where(finite, torch.exp(lse_b - merged_lse), torch.zeros_like(merged_lse))
    merged_out = weight_a.unsqueeze(-1) * out_a + weight_b.unsqueeze(-1) * out_b
    return merged_out, merged_lse


def _validate_merge_shapes_and_ranges(states: Sequence[AttentionPartialState]) -> None:
    first = states[0]
    previous_end = first.block_end
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all partial states must have matching out/lse shapes")
        if state.block_start != previous_end:
            raise ValueError("partial state block ranges must be gap-free and non-overlapping")
        previous_end = state.block_end


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.size(0) != k.size(0) or q.size(3) != k.size(3):
        raise ValueError("q, k, and v must share batch size and head dim")
    if q.size(1) < 1 or k.size(1) < 1 or q.size(3) < 1:
        raise ValueError("q, k, and v must have positive head counts and head dim")
    if not all(torch.is_floating_point(tensor) for tensor in (q, k, v)) or any(
        torch.is_complex(tensor) for tensor in (q, k, v)
    ):
        raise ValueError("q, k, and v must be real floating-point tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.size(1) % k.size(1) != 0:
        raise ValueError(f"Hq={q.size(1)} not divisible by Hkv={k.size(1)} (GQA group)")


def _validate_scale(scale: Optional[float]) -> None:
    if scale is None:
        return
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale must be a positive finite number")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be a positive finite number")


def _validate_output_dtype(output_dtype: torch.dtype) -> None:
    if not isinstance(output_dtype, torch.dtype):
        raise ValueError("output_dtype must be a real floating-point torch dtype")
    probe = torch.empty((), dtype=output_dtype)
    if not torch.is_floating_point(probe) or torch.is_complex(probe):
        raise ValueError("output_dtype must be a real floating-point torch dtype")


def _validate_partition_args(
    cp_world_size: int,
    kv_chunk_size: Optional[int],
) -> None:
    if isinstance(cp_world_size, bool) or not isinstance(cp_world_size, int) or cp_world_size < 1:
        raise ValueError("cp_world_size must be >= 1")
    if kv_chunk_size is not None and (
        isinstance(kv_chunk_size, bool) or not isinstance(kv_chunk_size, int) or kv_chunk_size < 1
    ):
        raise ValueError("kv_chunk_size must be >= 1 when provided")


def _zero_dependency(*tensors: torch.Tensor) -> torch.Tensor:
    total = torch.tensor(0.0, device=tensors[0].device)
    for tensor in tensors:
        total = total + tensor.sum()
    return total * 0.0


def _normalize_position_offsets(
    offsets: Optional[torch.Tensor],
    batch: int,
    device: torch.device,
    *,
    default: int,
    name: str,
) -> torch.Tensor:
    if offsets is None:
        return torch.full((batch,), default, dtype=torch.long, device=device)
    if offsets.ndim != 1 or offsets.numel() != batch:
        raise ValueError(f"{name} must have shape [B]")
    if torch.is_floating_point(offsets) or torch.is_complex(offsets) or offsets.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer positions")
    return offsets.to(device=device, dtype=torch.long)


def _split_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(length, parts)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        width = base + (1 if index < extra else 0)
        end = start + width
        bounds.append((start, end))
        start = end
    return bounds


def _kv_block_bounds(
    length: int,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for start, end in _split_bounds(length, cp_world_size):
        if kv_chunk_size is None:
            bounds.append((start, end))
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + kv_chunk_size, end)
            bounds.append((cursor, chunk_end))
            cursor = chunk_end
    return bounds


def split_kv_execution_plan_provenance(
    length: int,
    *,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
    backend: str,
) -> list[dict[str, object]]:
    """Return the actual backend-local Split-KV plan for every CP owner."""

    if length < 1:
        raise ValueError("Split-KV sequence length must be >= 1")
    if cp_world_size < 1:
        raise ValueError("cp_world_size must be >= 1")
    if kv_chunk_size is not None and kv_chunk_size < 1:
        raise ValueError("kv_chunk_size must be >= 1 when provided")
    result: list[dict[str, object]] = []
    for owner_cp_rank, (rank_start, rank_end) in enumerate(_split_bounds(length, cp_world_size)):
        if rank_start == rank_end:
            continue
        boundaries: tuple[tuple[int, int], ...]
        if kv_chunk_size is None:
            boundaries = ((rank_start, rank_end),)
            mode = SplitKVMode.DISABLED
        else:
            boundaries = tuple(
                (start, min(start + kv_chunk_size, rank_end))
                for start in range(rank_start, rank_end, kv_chunk_size)
            )
            mode = SplitKVMode.FIXED
        plan = SplitKVExecutionPlan(
            requested_mode=mode,
            requested_split_size=kv_chunk_size,
            actual_mode=mode,
            actual_split_size=kv_chunk_size,
            boundaries=boundaries,
            backend=backend,
            source="reference_execution",
        )
        result.append({"owner_cp_rank": owner_cp_rank, **plan.to_dict()})
    return result


def _strict_no_split_plan_provenance(
    length: int,
    *,
    cp_world_size: int,
    backend: str,
) -> list[dict[str, object]]:
    """Describe the full logical KV row consumed by each strict CP executor."""

    if length < 1:
        raise ValueError("strict no-Split-KV sequence length must be >= 1")
    _validate_partition_args(cp_world_size, None)
    plan = SplitKVExecutionPlan(
        requested_mode=SplitKVMode.DISABLED,
        requested_split_size=None,
        actual_mode=SplitKVMode.DISABLED,
        actual_split_size=None,
        boundaries=((0, length),),
        backend=backend,
        source="canonical_strict_execution",
    ).to_dict()
    return [{"owner_cp_rank": cp_rank, **plan} for cp_rank in range(cp_world_size)]


def build_reference_split_kv_runtime_plan_set(
    total_kv_tokens: Sequence[int],
    *,
    tp_world_size: int,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
    backend: str = "deterministic_cp_reference",
) -> SplitKVRuntimePlanSet:
    """Build complete per-batch/TP/CP/owner plans for the reference path."""

    totals = tuple(total_kv_tokens)
    if not totals or any(total < cp_world_size for total in totals):
        raise ValueError("reference runtime plan sets require at least one KV token per CP owner")
    if tp_world_size < 1 or cp_world_size < 1:
        raise ValueError("TP and CP world sizes must be >= 1")
    if kv_chunk_size is not None and kv_chunk_size < 1:
        raise ValueError("kv_chunk_size must be >= 1 when provided")

    entries: list[SplitKVRuntimePlanEntry] = []
    boundaries: tuple[tuple[int, int], ...]
    for batch_index, total in enumerate(totals):
        owner_ranges = _split_bounds(total, cp_world_size)
        for tp_rank in range(tp_world_size):
            for cp_rank in range(cp_world_size):
                for owner_cp_rank, (owner_start, owner_end) in enumerate(owner_ranges):
                    if kv_chunk_size is None:
                        mode = SplitKVMode.DISABLED
                        boundaries = ((owner_start, owner_end),)
                    else:
                        mode = SplitKVMode.FIXED
                        boundaries = tuple(
                            (start, min(start + kv_chunk_size, owner_end))
                            for start in range(owner_start, owner_end, kv_chunk_size)
                        )
                    execution = SplitKVExecutionPlan(
                        requested_mode=mode,
                        requested_split_size=kv_chunk_size,
                        actual_mode=mode,
                        actual_split_size=kv_chunk_size,
                        boundaries=boundaries,
                        backend=backend,
                        source="reference_execution",
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


CPAttentionReferenceOp = DeterministicCPAttentionReferenceOp

__all__ = [
    "AttentionBackwardComparisonReport",
    "AttentionBackwardGradients",
    "AttentionBackwardPathDrift",
    "AttentionBackwardPathResult",
    "AttentionBackwardRankDrift",
    "AttentionPartialState",
    "AttentionRingBlock",
    "AttentionRingSchedule",
    "AttentionSavedForwardState",
    "build_reference_split_kv_runtime_plan_set",
    "CPAttentionReferenceOp",
    "DeterministicAttentionCore",
    "DeterministicAttentionCoreResult",
    "DeterministicCPAttentionReferenceOp",
    "GradientDriftStats",
    "STRICT_ATTENTION_CORE_ID",
    "STRICT_ATTENTION_SCHEDULE_ID",
    "compare_cp_attention_backward",
    "merge_attention_partial_states",
    "split_kv_execution_plan_provenance",
]
