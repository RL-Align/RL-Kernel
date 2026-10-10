# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-GPU WS2 attention cross-implementation comparison harness.

This module compares logically equivalent attention materializations before CP
communication is introduced.  The full path is the training-style reference.
The chunked-query and paged-KV paths emulate rollout-style prefill layouts on a
single device while preserving global causal positions and attention-domain LSE.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_metadata
import inspect
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import torch

from rl_engine.contracts.operators.attention import SplitKVExecutionPlan, SplitKVMode, SplitKVSpec
from rl_engine.reference.definitions import selected_logprobs_reference
from rl_engine.reference.rope.rope import NativeRoPEOp

MergeBackend = Literal["rl_kernel", "transformer_engine"]
RoPEState = Literal["pre_rope", "post_rope"]

_TE_CONTEXT_PARALLEL_MODULE = (
    "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"
)
_TE_CONTEXT_PARALLEL_HELPERS = {
    "flash_attn_fwd_softmax_lse_correction": ("softmax_lse", "softmax_lse_per_step"),
    "flash_attn_fwd_out_correction_init": (
        "out_init_step",
        "softmax_lse",
        "softmax_lse_init_step",
        "seq_dim",
    ),
    "flash_attn_fwd_out_correction": (
        "out",
        "out_per_step",
        "softmax_lse",
        "softmax_lse_per_step",
        "seq_dim",
    ),
}


class TransformerEngineUnavailable(RuntimeError):
    """Raised when the optional Transformer Engine oracle cannot be imported."""


@dataclass(frozen=True)
class AttentionComparisonInputs:
    """Inputs shared by every single-GPU attention comparison path."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool = True
    scale: float | None = None
    key_padding_mask: torch.Tensor | None = None
    lm_head_weight: torch.Tensor | None = None
    target_ids: torch.Tensor | None = None
    active_token_mask: torch.Tensor | None = None
    output_dtype: torch.dtype = torch.float32
    rope_positions: torch.Tensor | None = None
    rope_theta: float = 1_000_000.0
    rope_rotary_dim: int | None = None
    rope_cast_at: str = "after_rope"
    rope_output_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class DecodeKVCacheMetadata:
    """Logical identity and physical layout for decode-stage cached KV.

    ``block_table`` maps logical KV blocks to physical cache pages.  Positions
    are stored per physical cache slot; unused slots must contain ``-1``.
    Keeping both mappings explicit lets the harness distinguish layout changes
    from changes to the logical token sequence.
    """

    cache_position: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    global_token_positions: torch.Tensor
    query_position_ids: torch.Tensor
    key_position_ids: torch.Tensor
    page_size: int
    prefix_cache_key: str | None = None
    prefix_cache_enabled: bool = False
    prefix_length: int = 0
    prefix_cache_fingerprint: str | None = None
    q_rope_state: RoPEState = "post_rope"
    k_cache_rope_state: RoPEState = "post_rope"
    cp_block_owners: torch.Tensor | None = None
    cp_world_size: int = 1


@dataclass(frozen=True)
class DecodeAttentionInputs:
    """Decode queries and physically paged KV cache used by the PR6 harness."""

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    metadata: DecodeKVCacheMetadata
    scale: float | None = None
    output_dtype: torch.dtype = torch.float32
    rope_theta: float = 1_000_000.0
    rope_rotary_dim: int | None = None
    rope_cast_at: str = "after_rope"
    q_rope_output_dtype: torch.dtype | None = None
    k_cache_rope_output_dtype: torch.dtype | None = None
    lm_head_weight: torch.Tensor | None = None
    target_ids: torch.Tensor | None = None
    active_token_mask: torch.Tensor | None = None
    k_new: torch.Tensor | None = None
    v_new: torch.Tensor | None = None
    split_kv: SplitKVSpec | None = None


@dataclass(frozen=True)
class AttentionPathResult:
    """One materialized attention path result."""

    name: str
    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]
    post_rope_q: torch.Tensor | None = None
    post_rope_k: torch.Tensor | None = None


@dataclass(frozen=True)
class DriftStats:
    """Shape-aware absolute drift summary."""

    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    active_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p95_abs": self.p95_abs,
            "p99_abs": self.p99_abs,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class AttentionPathDrift:
    """Candidate-vs-reference drift for one attention path."""

    candidate_name: str
    out: DriftStats
    lse: DriftStats
    dlogp: DriftStats | None
    provenance: dict[str, Any]
    post_rope_q: DriftStats | None = None
    post_rope_k: DriftStats | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "out": self.out.to_dict(),
            "lse": self.lse.to_dict(),
            "dlogp": None if self.dlogp is None else self.dlogp.to_dict(),
            "post_rope_q": (None if self.post_rope_q is None else self.post_rope_q.to_dict()),
            "post_rope_k": (None if self.post_rope_k is None else self.post_rope_k.to_dict()),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AttentionComparisonReport:
    """Structured report for PR2 single-GPU attention attribution."""

    reference_name: str
    drifts: tuple[AttentionPathDrift, ...]
    unavailable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_name": self.reference_name,
            "drifts": [drift.to_dict() for drift in self.drifts],
            "unavailable": list(self.unavailable),
        }


@dataclass(frozen=True)
class _PartialAttentionState:
    out: torch.Tensor
    lse: torch.Tensor
    block_start: int
    block_end: int


def compare_single_gpu_attention(
    inputs: AttentionComparisonInputs,
    *,
    query_chunk_size: int | None = None,
    kv_page_size: int | None = None,
    include_transformer_engine: bool = False,
    strict_bitwise: bool = False,
) -> AttentionComparisonReport:
    """Compare full attention with chunked/paged single-GPU materializations.

    If ``lm_head_weight`` and ``target_ids`` are provided, the report also
    includes active-token selected-logprob drift using the #207 convention:
    candidate logp minus reference logp.
    """

    _validate_comparison_inputs(inputs)
    if strict_bitwise:
        reference = run_strict_shared_core_attention(inputs)
        candidates = [
            run_strict_shared_core_chunked_attention(inputs, query_chunk_size=query_chunk_size),
            run_strict_shared_core_paged_layout_attention(inputs, kv_page_size=kv_page_size),
        ]
    else:
        reference = run_full_attention(inputs)
        candidates = [
            run_chunked_query_attention(inputs, query_chunk_size=query_chunk_size),
            run_paged_kv_attention(inputs, kv_page_size=kv_page_size, merge_backend="rl_kernel"),
        ]
    unavailable: list[str] = []
    if include_transformer_engine:
        try:
            candidates.append(
                run_paged_kv_attention(
                    inputs,
                    kv_page_size=kv_page_size,
                    merge_backend="transformer_engine",
                )
            )
        except TransformerEngineUnavailable as exc:
            unavailable.append(f"transformer_engine_paged_kv: {exc}")

    drifts = tuple(_compare_path(candidate, reference, inputs) for candidate in candidates)
    return AttentionComparisonReport(
        reference_name=reference.name,
        drifts=drifts,
        unavailable=tuple(unavailable),
    )


def run_strict_shared_core_attention(inputs: AttentionComparisonInputs) -> AttentionPathResult:
    """Run the canonical single-row arithmetic schedule for the full path."""

    _validate_comparison_inputs(inputs)
    out, lse = _strict_attention_with_lse(
        inputs.q,
        inputs.k,
        inputs.v,
        causal=inputs.causal,
        scale=inputs.scale,
        key_padding_mask=inputs.key_padding_mask,
        q_start=0,
        k_start=0,
        total_query_len=inputs.q.size(2),
        total_kv_len=inputs.k.size(2),
        output_dtype=inputs.output_dtype,
    )
    return AttentionPathResult(
        name="strict_shared_core_full_prefill",
        out=out,
        lse=lse,
        provenance=_strict_shared_core_provenance(
            inputs, materialization="full_logical_kv_shared_core"
        ),
    )


def run_strict_shared_core_chunked_attention(
    inputs: AttentionComparisonInputs,
    *,
    query_chunk_size: int | None,
) -> AttentionPathResult:
    """Replay query chunks through the same one-row schedule as full prefill."""

    _validate_comparison_inputs(inputs)
    chunk_size = (
        inputs.q.size(2)
        if query_chunk_size is None
        else _positive_int(query_chunk_size, "query_chunk_size")
    )
    bounds = _chunk_bounds(inputs.q.size(2), chunk_size)
    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for q_start, q_end in bounds:
        out, lse = _strict_attention_with_lse(
            inputs.q[:, :, q_start:q_end, :],
            inputs.k,
            inputs.v,
            causal=inputs.causal,
            scale=inputs.scale,
            key_padding_mask=inputs.key_padding_mask,
            q_start=q_start,
            k_start=0,
            total_query_len=inputs.q.size(2),
            total_kv_len=inputs.k.size(2),
            output_dtype=inputs.output_dtype,
        )
        outs.append(out)
        lses.append(lse)
    return AttentionPathResult(
        name="strict_shared_core_chunked_prefill",
        out=torch.cat(outs, dim=2),
        lse=torch.cat(lses, dim=2),
        provenance={
            **_strict_shared_core_provenance(inputs, materialization="query_chunks_shared_core"),
            "query_chunk_size": chunk_size,
            "chunk_bounds": [list(bound) for bound in bounds],
        },
    )


def run_strict_shared_core_paged_layout_attention(
    inputs: AttentionComparisonInputs,
    *,
    kv_page_size: int | None,
) -> AttentionPathResult:
    """Restore paged KV to logical order, then call the shared core once.

    Strict mode deliberately does not create per-page partial states: page size
    is a storage detail, while Split-KV is disabled in the arithmetic contract.
    """

    _validate_comparison_inputs(inputs)
    page_size = (
        inputs.k.size(2) if kv_page_size is None else _positive_int(kv_page_size, "kv_page_size")
    )
    pages = [(start, end) for start, end in _chunk_bounds(inputs.k.size(2), page_size)]
    logical_k = torch.cat([inputs.k[:, :, start:end, :] for start, end in pages], dim=2)
    logical_v = torch.cat([inputs.v[:, :, start:end, :] for start, end in pages], dim=2)
    out, lse = _strict_attention_with_lse(
        inputs.q,
        logical_k,
        logical_v,
        causal=inputs.causal,
        scale=inputs.scale,
        key_padding_mask=inputs.key_padding_mask,
        q_start=0,
        k_start=0,
        total_query_len=inputs.q.size(2),
        total_kv_len=inputs.k.size(2),
        output_dtype=inputs.output_dtype,
    )
    return AttentionPathResult(
        name="strict_shared_core_paged_kv",
        out=out,
        lse=lse,
        provenance={
            **_strict_shared_core_provenance(inputs, materialization="paged_kv_layout_shared_core"),
            "kv_page_size": page_size,
            "kv_page_bounds": [list(bound) for bound in pages],
        },
    )


def compare_single_gpu_rope_attention(
    inputs: AttentionComparisonInputs,
) -> AttentionComparisonReport:
    """Compare canonical unfused RoPE+Attention with fused-like materialization.

    This attribution path keeps the computation on one device and checks the
    boundary that matters before CP communication: post-RoPE Q/K identity and
    the resulting attention ``out`` / attention-domain ``lse``.
    """

    _validate_comparison_inputs(inputs)
    _validate_rope_inputs(inputs)
    reference = run_unfused_rope_attention(inputs)
    candidates = [run_fused_like_rope_attention(inputs)]
    drifts = tuple(_compare_path(candidate, reference, inputs) for candidate in candidates)
    return AttentionComparisonReport(reference_name=reference.name, drifts=drifts)


def compare_decode_kv_replay(
    inputs: DecodeAttentionInputs,
    *,
    include_transformer_engine: bool = False,
    strict_bitwise: bool = False,
) -> AttentionComparisonReport:
    """Compare paged decode replay with a logical full-KV teacher-forcing view."""

    _validate_decode_inputs(inputs)
    if strict_bitwise:
        reference = _run_decode_strict_shared_core(inputs, materialization="logical_prefill")
        candidates = [_run_decode_strict_shared_core(inputs, materialization="paged_kv_layout")]
    else:
        reference = _run_decode_full_prefill_reference(inputs)
        candidates = [_run_decode_kv_replay(inputs, merge_backend="rl_kernel")]
    unavailable: list[str] = []
    if include_transformer_engine:
        try:
            candidates.append(_run_decode_kv_replay(inputs, merge_backend="transformer_engine"))
        except TransformerEngineUnavailable as exc:
            unavailable.append(f"transformer_engine_decode_kv_replay: {exc}")
    drifts = tuple(_compare_decode_path(candidate, reference, inputs) for candidate in candidates)
    return AttentionComparisonReport(
        reference_name=reference.name,
        drifts=drifts,
        unavailable=tuple(unavailable),
    )


def run_decode_strict_shared_core(
    inputs: DecodeAttentionInputs,
) -> AttentionPathResult:
    """Run decode through the same fixed one-query arithmetic schedule."""

    _validate_decode_inputs(inputs)
    return _run_decode_strict_shared_core(inputs, materialization="paged_kv_layout")


def _run_decode_strict_shared_core(
    inputs: DecodeAttentionInputs,
    *,
    materialization: str,
) -> AttentionPathResult:
    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_index in range(inputs.q.size(0)):
        q, k, v, logical_positions = _decode_logical_qkv(inputs, batch_index)
        batch_out: list[torch.Tensor] = []
        batch_lse: list[torch.Tensor] = []
        for query_index in range(q.size(2)):
            query_position = int(inputs.metadata.cache_position[batch_index, query_index].item())
            visible = logical_positions <= query_position
            if not bool(visible.any()):
                raise ValueError("each decode query must have at least one visible KV token")
            out, lse = _strict_decode_attention_with_lse(
                q[:, :, query_index : query_index + 1, :],
                k[:, :, visible, :],
                v[:, :, visible, :],
                output_dtype=inputs.output_dtype,
            )
            batch_out.append(out)
            batch_lse.append(lse)
        outs.append(torch.cat(batch_out, dim=2))
        lses.append(torch.cat(batch_lse, dim=2))
    return AttentionPathResult(
        name=f"strict_shared_core_{materialization}",
        out=torch.cat(outs, dim=0),
        lse=torch.cat(lses, dim=0),
        provenance={
            "attention_mode": "decode",
            "materialization": materialization,
            "execution_scope": "single_device_strict_shared_core",
            "runtime_verified": False,
            "actual_backend": "rlkernel.pytorch.strict_attention_reference",
            "communication_backend": "none",
            "production_ready": False,
            "supports_backward": False,
            "communication": "none",
            "strict_mode": True,
            "strict_core_id": "rlkernel.attention.deterministic_core.v1",
            "strict_schedule": "single_batch_single_query_global_kv_blocks",
            "native_attention_arithmetic": False,
            "fallback": False,
            "fallback_reason": None,
            "split_kv_policy": "disabled",
            "merge_order": "global_block_index",
            "lse_domain": "attention",
            "lse_exported": True,
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "cache_execution_fingerprint": _decode_cache_fingerprint(
                inputs,
                token_counts=tuple(int(length) for length in inputs.metadata.kv_seq_lens.tolist()),
                domain="strict_shared_core_decode",
            ),
            "kv_cache_identity_fingerprint": _decode_cache_fingerprint(
                inputs,
                token_counts=tuple(int(length) for length in inputs.metadata.kv_seq_lens.tolist()),
                domain="strict_shared_core_decode",
            ),
            "rope_identity_fingerprint": decode_rope_identity_fingerprint(inputs),
            "query_position_ids": inputs.metadata.query_position_ids.tolist(),
            "key_position_ids": inputs.metadata.key_position_ids.tolist(),
            "block_table": inputs.metadata.block_table.tolist(),
            "page_size": inputs.metadata.page_size,
        },
    )


def run_decode_full_prefill_reference(inputs: DecodeAttentionInputs) -> AttentionPathResult:
    """Materialize the full logical KV sequence for each decode query.

    This is the teacher-forcing side of the PR6 comparison.  It deliberately
    ignores physical page boundaries after restoring logical token order.
    """

    _validate_decode_inputs(inputs)
    return _run_decode_full_prefill_reference(inputs)


def _run_decode_full_prefill_reference(inputs: DecodeAttentionInputs) -> AttentionPathResult:
    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_index in range(inputs.q.size(0)):
        q, k, v, logical_positions = _decode_logical_qkv(inputs, batch_index)
        batch_out: list[torch.Tensor] = []
        batch_lse: list[torch.Tensor] = []
        for query_index in range(q.size(2)):
            query_position = int(inputs.metadata.cache_position[batch_index, query_index].item())
            visible = logical_positions <= query_position
            out, lse = _attention_with_lse(
                q[:, :, query_index : query_index + 1, :],
                k[:, :, visible, :],
                v[:, :, visible, :],
                causal=False,
                scale=inputs.scale,
                key_padding_mask=None,
                q_start=0,
                k_start=0,
                total_query_len=1,
                total_kv_len=int(visible.sum().item()),
                output_dtype=inputs.output_dtype,
            )
            _require_finite_decode_result(out, lse, path="full logical-KV reference")
            batch_out.append(out)
            batch_lse.append(lse)
        outs.append(torch.cat(batch_out, dim=2))
        lses.append(torch.cat(batch_lse, dim=2))
    return AttentionPathResult(
        name="full_prefill_decode_reference",
        out=torch.cat(outs, dim=0),
        lse=torch.cat(lses, dim=0),
        provenance={
            "attention_mode": "decode",
            "materialization": "full_logical_kv",
            "execution_scope": "single_device_logical_reference",
            "runtime_verified": False,
            "supports_backward": False,
            "communication": "none",
            "lse_domain": "attention",
            "accum_dtype": "fp32",
            "lse_dtype": "fp32",
            "downcast_at": "final_write",
            "output_dtype": str(inputs.output_dtype).replace("torch.", ""),
            "scale": _decode_attention_scale(inputs),
            "q_dtype": str(inputs.q.dtype).replace("torch.", ""),
            "k_cache_dtype": str(inputs.k_cache.dtype).replace("torch.", ""),
            "v_cache_dtype": str(inputs.v_cache.dtype).replace("torch.", ""),
        },
    )


def run_decode_kv_replay(
    inputs: DecodeAttentionInputs,
    *,
    merge_backend: MergeBackend = "rl_kernel",
) -> AttentionPathResult:
    """Replay decode over physical KV pages and merge by logical block index."""

    _validate_decode_inputs(inputs)
    return _run_decode_kv_replay(inputs, merge_backend=merge_backend)


def _run_decode_kv_replay(
    inputs: DecodeAttentionInputs,
    *,
    merge_backend: MergeBackend,
) -> AttentionPathResult:
    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    merge_orders: list[list[list[int]]] = []
    actual_split_plans: list[list[dict[str, Any]]] = []
    cp_block_owners: list[list[int]] = []
    split_kv = _resolved_decode_split_kv(inputs)
    for batch_index in range(inputs.q.size(0)):
        q, k, v, logical_positions = _decode_logical_qkv(inputs, batch_index)
        owners = _logical_block_owners(inputs, batch_index)
        cp_block_owners.append(owners)
        batch_out: list[torch.Tensor] = []
        batch_lse: list[torch.Tensor] = []
        batch_orders: list[list[int]] = []
        batch_split_plans: list[dict[str, Any]] = []
        for query_index in range(q.size(2)):
            query_position = int(inputs.metadata.cache_position[batch_index, query_index].item())
            states: list[_PartialAttentionState] = []
            order: list[int] = []
            visible_count = int((logical_positions <= query_position).sum().item())
            split_bounds = _decode_split_bounds(visible_count, split_kv)
            for block_index, (block_start, block_end) in enumerate(split_bounds):
                block_positions = logical_positions[block_start:block_end]
                visible = block_positions <= query_position
                if not bool(visible.any()):
                    continue
                visible_end = block_start + int(visible.sum().item())
                out, lse = _attention_with_lse(
                    q[:, :, query_index : query_index + 1, :],
                    k[:, :, block_start:visible_end, :],
                    v[:, :, block_start:visible_end, :],
                    causal=False,
                    scale=inputs.scale,
                    key_padding_mask=None,
                    q_start=0,
                    k_start=block_start,
                    total_query_len=1,
                    total_kv_len=visible_end,
                    output_dtype=torch.float32,
                )
                states.append(
                    _PartialAttentionState(
                        out=out,
                        lse=lse,
                        block_start=block_index,
                        block_end=block_index + 1,
                    )
                )
                order.append(block_index)
            if not states:
                raise ValueError("each decode query must have at least one visible cached KV token")
            out, lse = _merge_partial_states(states, backend=merge_backend)
            _require_finite_decode_result(out, lse, path=f"{merge_backend} paged-KV replay")
            batch_out.append(out.to(inputs.output_dtype))
            batch_lse.append(lse)
            batch_orders.append(order)
            plan = SplitKVExecutionPlan(
                requested_mode=split_kv.mode,
                requested_split_size=split_kv.fixed_split_size,
                actual_mode=split_kv.mode,
                actual_split_size=split_kv.fixed_split_size,
                boundaries=tuple(split_bounds),
                backend=f"{merge_backend}_decode_kv_replay",
                source="reference_execution",
            )
            batch_split_plans.append(plan.to_dict())
        merge_orders.append(batch_orders)
        actual_split_plans.append(batch_split_plans)
        outs.append(torch.cat(batch_out, dim=2))
        lses.append(torch.cat(batch_lse, dim=2))

    provenance: dict[str, Any] = {
        "attention_mode": "decode",
        "decode_semantics": (
            "past_kv_plus_new_kv_append" if inputs.k_new is not None else "cache_replay"
        ),
        "past_kv_lengths": inputs.metadata.kv_seq_lens.tolist(),
        "new_kv_length": (0 if inputs.k_new is None else inputs.k_new.size(2)),
        "materialization": "paged_kv_replay",
        "execution_scope": "single_device_logical_reference",
        "runtime_verified": False,
        "supports_backward": False,
        "communication": "none",
        "sq": inputs.q.size(2),
        "page_size": inputs.metadata.page_size,
        "cache_position": inputs.metadata.cache_position.tolist(),
        "kv_seq_lens": inputs.metadata.kv_seq_lens.tolist(),
        "block_table": inputs.metadata.block_table.tolist(),
        "global_token_positions": inputs.metadata.global_token_positions.tolist(),
        "query_position_ids": inputs.metadata.query_position_ids.tolist(),
        "key_position_ids": inputs.metadata.key_position_ids.tolist(),
        "prefix_cache_enabled": inputs.metadata.prefix_cache_enabled,
        "prefix_cache_key": inputs.metadata.prefix_cache_key,
        "prefix_length": inputs.metadata.prefix_length,
        "prefix_cache_fingerprint": inputs.metadata.prefix_cache_fingerprint,
        "q_rope_state": inputs.metadata.q_rope_state,
        "k_cache_rope_state": inputs.metadata.k_cache_rope_state,
        "rope_theta": float(inputs.rope_theta),
        "rotary_dim": _decode_rope_rotary_dim(inputs),
        "rope_cast_at": inputs.rope_cast_at,
        "q_rope_output_dtype": str(_decode_q_rope_output_dtype(inputs)).replace("torch.", ""),
        "k_cache_rope_output_dtype": str(_decode_k_rope_output_dtype(inputs)).replace("torch.", ""),
        "cp_block_owners": cp_block_owners,
        "cp_world_size": inputs.metadata.cp_world_size,
        "requested_split_kv_policy": split_kv.mode.value,
        "requested_split_kv_size": split_kv.fixed_split_size,
        "actual_split_kv_plans": actual_split_plans,
        "merge_order": "global_block_index",
        "logical_merge_orders": merge_orders,
        "merge_backend": merge_backend,
        "lse_domain": "attention",
        "lse_exported": True,
        "accum_dtype": "fp32",
        "lse_dtype": "fp32",
        "downcast_at": "final_write",
        "output_dtype": str(inputs.output_dtype).replace("torch.", ""),
        "scale": _decode_attention_scale(inputs),
        "q_dtype": str(inputs.q.dtype).replace("torch.", ""),
        "k_cache_dtype": str(inputs.k_cache.dtype).replace("torch.", ""),
        "v_cache_dtype": str(inputs.v_cache.dtype).replace("torch.", ""),
        "cache_execution_fingerprint": _decode_cache_fingerprint(
            inputs,
            token_counts=tuple(int(length) for length in inputs.metadata.kv_seq_lens.tolist()),
            domain="full_logical_kv_cache",
        ),
        "kv_cache_identity_fingerprint": _decode_cache_fingerprint(
            inputs,
            token_counts=tuple(int(length) for length in inputs.metadata.kv_seq_lens.tolist()),
            domain="full_logical_kv_cache",
        ),
        "rope_identity_fingerprint": decode_rope_identity_fingerprint(inputs),
        "preprocess_policy": "reference_only_not_production",
        "preprocess_backends": {
            "qk_rmsnorm": "not_executed_projected_qk_input",
            "rope": (
                "rlkernel.pytorch.rope_reference"
                if inputs.metadata.q_rope_state == "pre_rope"
                or inputs.metadata.k_cache_rope_state == "pre_rope"
                else "external_post_rope_input"
            ),
        },
        "preprocess_fallback": True,
        "preprocess_fallback_reason": (
            "decode logical replay is a single-device correctness reference, not native vLLM"
        ),
    }
    if merge_backend == "transformer_engine":
        provenance.update(_te_context_parallel_provenance())
        provenance.update(
            {
                "actual_backend": "te_context_parallel_merge_helpers",
                "communication_backend": "none",
                "production_ready": False,
            }
        )
    return AttentionPathResult(
        name=f"{merge_backend}_decode_kv_replay",
        out=torch.cat(outs, dim=0),
        lse=torch.cat(lses, dim=0),
        provenance=provenance,
    )


def run_full_attention(inputs: AttentionComparisonInputs) -> AttentionPathResult:
    """Training-style full-sequence attention with exported attention-domain LSE."""

    out, lse = _attention_with_lse(
        inputs.q,
        inputs.k,
        inputs.v,
        causal=inputs.causal,
        scale=inputs.scale,
        key_padding_mask=inputs.key_padding_mask,
        q_start=0,
        k_start=0,
        total_query_len=inputs.q.size(2),
        total_kv_len=inputs.k.size(2),
        output_dtype=inputs.output_dtype,
    )
    return AttentionPathResult(
        name="full_prefill",
        out=out,
        lse=lse,
        provenance={
            "attention_mode": "prefill",
            "materialization": "full_sequence",
            "lse_domain": "attention",
            **_single_gpu_reference_provenance(),
        },
    )


def run_unfused_rope_attention(inputs: AttentionComparisonInputs) -> AttentionPathResult:
    """Canonical ``RoPE -> Attention`` reference materialization."""

    post_rope_q, post_rope_k = _apply_rope_to_qk(inputs)
    out, lse = _attention_with_lse(
        post_rope_q,
        post_rope_k,
        inputs.v,
        causal=inputs.causal,
        scale=inputs.scale,
        key_padding_mask=inputs.key_padding_mask,
        q_start=0,
        k_start=0,
        total_query_len=post_rope_q.size(2),
        total_kv_len=post_rope_k.size(2),
        output_dtype=inputs.output_dtype,
    )
    return AttentionPathResult(
        name="unfused_rope_attention",
        out=out,
        lse=lse,
        provenance=_rope_attention_provenance(
            inputs,
            materialization="rope_then_attention",
            fusion_boundary="unfused_rope_attention",
        ),
        post_rope_q=post_rope_q,
        post_rope_k=post_rope_k,
    )


def run_fused_like_rope_attention(inputs: AttentionComparisonInputs) -> AttentionPathResult:
    """Semantic fused ``RoPE+Attention`` path using the same canonical RoPE rules."""

    post_rope_q, post_rope_k = _apply_rope_to_qk(inputs)
    out, lse = _attention_with_lse(
        post_rope_q,
        post_rope_k,
        inputs.v,
        causal=inputs.causal,
        scale=inputs.scale,
        key_padding_mask=inputs.key_padding_mask,
        q_start=0,
        k_start=0,
        total_query_len=post_rope_q.size(2),
        total_kv_len=post_rope_k.size(2),
        output_dtype=inputs.output_dtype,
    )
    return AttentionPathResult(
        name="fused_like_rope_attention",
        out=out,
        lse=lse,
        provenance=_rope_attention_provenance(
            inputs,
            materialization="fused_like_rope_attention",
            fusion_boundary="fused_rope_attention",
        ),
        post_rope_q=post_rope_q,
        post_rope_k=post_rope_k,
    )


def run_chunked_query_attention(
    inputs: AttentionComparisonInputs,
    *,
    query_chunk_size: int | None,
) -> AttentionPathResult:
    """Rollout-style chunked prefill replay over full KV on one device."""

    sq = inputs.q.size(2)
    chunk_size = (
        sq if query_chunk_size is None else _positive_int(query_chunk_size, "query_chunk_size")
    )
    out_chunks: list[torch.Tensor] = []
    lse_chunks: list[torch.Tensor] = []
    chunk_bounds = _chunk_bounds(sq, chunk_size)
    for q_start, q_end in chunk_bounds:
        out, lse = _attention_with_lse(
            inputs.q[:, :, q_start:q_end, :],
            inputs.k,
            inputs.v,
            causal=inputs.causal,
            scale=inputs.scale,
            key_padding_mask=inputs.key_padding_mask,
            q_start=q_start,
            k_start=0,
            total_query_len=sq,
            total_kv_len=inputs.k.size(2),
            output_dtype=inputs.output_dtype,
        )
        out_chunks.append(out)
        lse_chunks.append(lse)

    return AttentionPathResult(
        name="chunked_prefill",
        out=torch.cat(out_chunks, dim=2),
        lse=torch.cat(lse_chunks, dim=2),
        provenance={
            "attention_mode": "chunked_prefill",
            "materialization": "query_chunks",
            "query_chunk_size": chunk_size,
            "chunk_bounds": [list(bound) for bound in chunk_bounds],
            "lse_domain": "attention",
            **_single_gpu_reference_provenance(),
        },
    )


def run_paged_kv_attention(
    inputs: AttentionComparisonInputs,
    *,
    kv_page_size: int | None,
    merge_backend: MergeBackend = "rl_kernel",
) -> AttentionPathResult:
    """Rollout-style paged-KV prefill replay with explicit LSE merge."""

    skv = inputs.k.size(2)
    page_size = skv if kv_page_size is None else _positive_int(kv_page_size, "kv_page_size")
    states: list[_PartialAttentionState] = []
    page_bounds = _chunk_bounds(skv, page_size)
    for k_start, k_end in page_bounds:
        key_mask = (
            None if inputs.key_padding_mask is None else inputs.key_padding_mask[:, k_start:k_end]
        )
        out, lse = _attention_with_lse(
            inputs.q,
            inputs.k[:, :, k_start:k_end, :],
            inputs.v[:, :, k_start:k_end, :],
            causal=inputs.causal,
            scale=inputs.scale,
            key_padding_mask=key_mask,
            q_start=0,
            k_start=k_start,
            total_query_len=inputs.q.size(2),
            total_kv_len=skv,
            output_dtype=torch.float32,
        )
        states.append(
            _PartialAttentionState(
                out=out,
                lse=lse,
                block_start=k_start,
                block_end=k_end,
            )
        )

    out, lse = _merge_partial_states(states, backend=merge_backend)
    provenance = {
        "attention_mode": "prefill",
        "materialization": "paged_kv",
        "kv_page_size": page_size,
        "kv_page_bounds": [list(bound) for bound in page_bounds],
        "merge_backend": merge_backend,
        "requested_backend": merge_backend,
        "actual_backend": (
            "te_context_parallel_merge_helpers"
            if merge_backend == "transformer_engine"
            else "rl_kernel"
        ),
        "fallback": False,
        "fallback_reason": None,
        "merge_order": "global_block_index",
        "lse_domain": "attention",
        "lse_exported": True,
        "accum_dtype": "fp32",
        "downcast_at": "final_write",
        **_single_gpu_reference_provenance(),
    }
    if merge_backend == "transformer_engine":
        provenance.update(_te_context_parallel_provenance())
        provenance.update(
            {
                "actual_backend": "te_context_parallel_merge_helpers",
                "communication_backend": "none",
                "production_ready": False,
            }
        )
    return AttentionPathResult(
        name=f"{merge_backend}_paged_kv",
        out=out.to(inputs.output_dtype),
        lse=lse,
        provenance=provenance,
    )


def transformer_engine_context_parallel_available() -> bool:
    """Return whether the optional TE context-parallel helper module imports."""

    try:
        _load_te_context_parallel()
    except TransformerEngineUnavailable:
        return False
    return True


def decode_prefix_cache_fingerprint(
    inputs: DecodeAttentionInputs,
    *,
    prefix_length: int,
) -> str:
    """Fingerprint logical prefix positions and cached K/V content.

    The fingerprint is invariant to physical page placement because cache slots
    are first restored to logical token order. It includes every cache-side
    RoPE materialization fact that can change the replayed K values, so a
    prefix cannot be reused under a different rotary configuration.
    """

    prefix_length = _positive_int(prefix_length, "prefix_length")
    if bool((inputs.metadata.kv_seq_lens < prefix_length).any()):
        raise ValueError("prefix_length must not exceed any kv_seq_lens entry")
    return _decode_cache_fingerprint(
        inputs,
        token_counts=(prefix_length,) * inputs.q.size(0),
        domain="shared_prefix",
    )


def decode_kv_cache_fingerprint(inputs: DecodeAttentionInputs) -> str:
    """Fingerprint complete logical cache execution identity after validation."""

    _validate_decode_inputs(inputs)
    return _decode_cache_fingerprint(
        inputs,
        token_counts=tuple(int(length) for length in inputs.metadata.kv_seq_lens.tolist()),
        domain="full_logical_kv_cache",
    )


def decode_rope_identity_fingerprint(inputs: DecodeAttentionInputs) -> str:
    """Fingerprint every RoPE fact that changes decode Q/K interpretation."""

    _validate_decode_inputs(inputs)
    digest = hashlib.sha256()
    digest.update(
        (
            f"q_rope_state={inputs.metadata.q_rope_state};"
            f"k_cache_rope_state={inputs.metadata.k_cache_rope_state};"
            f"rope_theta={float(inputs.rope_theta):.17g};"
            f"rotary_dim={_decode_rope_rotary_dim(inputs)};"
            f"rope_cast_at={inputs.rope_cast_at};"
            f"q_rope_output_dtype={_decode_q_rope_output_dtype(inputs)};"
            f"k_rope_output_dtype={_decode_k_rope_output_dtype(inputs)}\n"
        ).encode()
    )
    for tensor in (inputs.metadata.query_position_ids, inputs.metadata.key_position_ids):
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _decode_cache_fingerprint(
    inputs: DecodeAttentionInputs,
    *,
    token_counts: tuple[int, ...],
    domain: str,
) -> str:
    if len(token_counts) != inputs.q.size(0):
        raise ValueError("cache fingerprint requires one token count per batch item")
    digest = hashlib.sha256()
    digest.update(
        (
            f"domain={domain};page_size={inputs.metadata.page_size};"
            f"prefix_cache_enabled={inputs.metadata.prefix_cache_enabled};"
            f"prefix_cache_key={inputs.metadata.prefix_cache_key!r};"
            f"prefix_length={inputs.metadata.prefix_length};"
            f"k_rope_state={inputs.metadata.k_cache_rope_state};"
            f"rope_theta={float(inputs.rope_theta):.17g};"
            f"rotary_dim={_decode_rope_rotary_dim(inputs)};"
            f"rope_cast_at={inputs.rope_cast_at};"
            f"k_rope_output_dtype={_decode_k_rope_output_dtype(inputs)};"
            f"k_dtype={inputs.k_cache.dtype};v_dtype={inputs.v_cache.dtype}\n"
        ).encode()
    )
    for batch_index, token_count in enumerate(token_counts):
        if token_count <= 0 or token_count > int(inputs.metadata.kv_seq_lens[batch_index].item()):
            raise ValueError("cache fingerprint token count is outside kv_seq_lens")
        slots = _decode_logical_slot_index(inputs, batch_index)[:token_count]
        owner_count = math.ceil(token_count / inputs.metadata.page_size)
        owners = _logical_block_owners(inputs, batch_index)[:owner_count]
        digest.update(f"batch={batch_index};tokens={token_count};owners={owners}\n".encode())
        for tensor in (
            inputs.metadata.global_token_positions[batch_index, slots],
            inputs.metadata.key_position_ids[batch_index, slots],
            inputs.k_cache[batch_index, :, slots, :],
            inputs.v_cache[batch_index, :, slots, :],
        ):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _compare_path(
    candidate: AttentionPathResult,
    reference: AttentionPathResult,
    inputs: AttentionComparisonInputs,
) -> AttentionPathDrift:
    dlogp = None
    if inputs.lm_head_weight is not None and inputs.target_ids is not None:
        candidate_logp = _selected_logps_from_attention(candidate.out, inputs)
        reference_logp = _selected_logps_from_attention(reference.out, inputs)
        dlogp = _drift_stats(candidate_logp, reference_logp, mask=inputs.active_token_mask)

    return AttentionPathDrift(
        candidate_name=candidate.name,
        out=_drift_stats(candidate.out, reference.out),
        lse=_drift_stats(candidate.lse, reference.lse),
        dlogp=dlogp,
        provenance=candidate.provenance,
        post_rope_q=(
            None
            if candidate.post_rope_q is None or reference.post_rope_q is None
            else _drift_stats(candidate.post_rope_q, reference.post_rope_q)
        ),
        post_rope_k=(
            None
            if candidate.post_rope_k is None or reference.post_rope_k is None
            else _drift_stats(candidate.post_rope_k, reference.post_rope_k)
        ),
    )


def _compare_decode_path(
    candidate: AttentionPathResult,
    reference: AttentionPathResult,
    inputs: DecodeAttentionInputs,
) -> AttentionPathDrift:
    dlogp = None
    if inputs.lm_head_weight is not None and inputs.target_ids is not None:
        candidate_logp = _selected_logps_from_decode_attention(candidate.out, inputs)
        reference_logp = _selected_logps_from_decode_attention(reference.out, inputs)
        dlogp = _drift_stats(candidate_logp, reference_logp, mask=inputs.active_token_mask)
    return AttentionPathDrift(
        candidate_name=candidate.name,
        out=_drift_stats(candidate.out, reference.out),
        lse=_drift_stats(candidate.lse, reference.lse),
        dlogp=dlogp,
        provenance=candidate.provenance,
    )


def _apply_rope_to_qk(inputs: AttentionComparisonInputs) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_rope_inputs(inputs)
    assert inputs.rope_positions is not None
    rope = NativeRoPEOp()
    output_dtype = _rope_output_dtype(inputs)
    q = rope.forward_fp32(inputs.q, inputs.rope_positions, theta=inputs.rope_theta).to(output_dtype)
    k = rope.forward_fp32(inputs.k, inputs.rope_positions, theta=inputs.rope_theta).to(output_dtype)
    return q, k


def _decode_logical_qkv(
    inputs: DecodeAttentionInputs,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore one batch's cache to logical order and materialize RoPE state."""

    metadata = inputs.metadata
    slot_index = _decode_logical_slot_index(inputs, batch_index)
    logical_position_tensor = metadata.global_token_positions[batch_index, slot_index].long()
    k = inputs.k_cache[batch_index : batch_index + 1, :, slot_index, :]
    v = inputs.v_cache[batch_index : batch_index + 1, :, slot_index, :]
    q = inputs.q[batch_index : batch_index + 1]

    rope = NativeRoPEOp()
    if metadata.q_rope_state == "pre_rope":
        q = rope.forward_fp32(
            q,
            metadata.query_position_ids[batch_index : batch_index + 1],
            theta=inputs.rope_theta,
        ).to(_decode_q_rope_output_dtype(inputs))
    if metadata.k_cache_rope_state == "pre_rope":
        key_positions = metadata.key_position_ids[batch_index, slot_index].unsqueeze(0)
        k = rope.forward_fp32(k, key_positions, theta=inputs.rope_theta).to(
            _decode_k_rope_output_dtype(inputs)
        )
    if inputs.k_new is not None:
        assert inputs.v_new is not None
        k_new = inputs.k_new[batch_index : batch_index + 1]
        if metadata.k_cache_rope_state == "pre_rope":
            k_new = rope.forward_fp32(
                k_new,
                metadata.query_position_ids[batch_index : batch_index + 1],
                theta=inputs.rope_theta,
            ).to(_decode_k_rope_output_dtype(inputs))
        k = torch.cat((k, k_new), dim=2)
        v = torch.cat((v, inputs.v_new[batch_index : batch_index + 1]), dim=2)
        logical_position_tensor = torch.cat(
            (
                logical_position_tensor,
                metadata.query_position_ids[batch_index].long(),
            )
        )
    return q, k, v, logical_position_tensor


def _decode_logical_slot_index(
    inputs: DecodeAttentionInputs,
    batch_index: int,
) -> torch.Tensor:
    metadata = inputs.metadata
    sequence_length = int(metadata.kv_seq_lens[batch_index].item())
    logical_block_count = math.ceil(sequence_length / metadata.page_size)
    logical_index = torch.arange(
        sequence_length,
        device=inputs.k_cache.device,
        dtype=torch.long,
    )
    pages = metadata.block_table[batch_index, :logical_block_count].long()
    return (
        pages[logical_index // metadata.page_size] * metadata.page_size
        + logical_index % metadata.page_size
    )


def _logical_block_owners(inputs: DecodeAttentionInputs, batch_index: int) -> list[int]:
    block_count = math.ceil(
        int(inputs.metadata.kv_seq_lens[batch_index].item()) / inputs.metadata.page_size
    )
    if inputs.metadata.cp_block_owners is None:
        return [0] * block_count
    return [
        int(owner) for owner in inputs.metadata.cp_block_owners[batch_index, :block_count].tolist()
    ]


def _decode_q_rope_output_dtype(inputs: DecodeAttentionInputs) -> torch.dtype:
    return inputs.q.dtype if inputs.q_rope_output_dtype is None else inputs.q_rope_output_dtype


def _decode_k_rope_output_dtype(inputs: DecodeAttentionInputs) -> torch.dtype:
    return (
        inputs.k_cache.dtype
        if inputs.k_cache_rope_output_dtype is None
        else inputs.k_cache_rope_output_dtype
    )


def _decode_rope_rotary_dim(inputs: DecodeAttentionInputs) -> int:
    return inputs.q.size(-1) if inputs.rope_rotary_dim is None else inputs.rope_rotary_dim


def _decode_attention_scale(inputs: DecodeAttentionInputs) -> float:
    return 1.0 / math.sqrt(inputs.q.size(-1)) if inputs.scale is None else float(inputs.scale)


def _rope_output_dtype(inputs: AttentionComparisonInputs) -> torch.dtype:
    return inputs.q.dtype if inputs.rope_output_dtype is None else inputs.rope_output_dtype


def _rope_rotary_dim(inputs: AttentionComparisonInputs) -> int:
    return inputs.q.size(-1) if inputs.rope_rotary_dim is None else inputs.rope_rotary_dim


def _rope_attention_provenance(
    inputs: AttentionComparisonInputs,
    *,
    materialization: str,
    fusion_boundary: str,
) -> dict[str, Any]:
    assert inputs.rope_positions is not None
    return {
        "attention_mode": "prefill",
        "materialization": materialization,
        "rope_state": "post_rope",
        "q_rope_state": "post_rope",
        "k_rope_state": "post_rope",
        "position_kind": "position_ids",
        "position_ids_shape": list(inputs.rope_positions.shape),
        "position_ids_min": int(inputs.rope_positions.min().item()),
        "position_ids_max": int(inputs.rope_positions.max().item()),
        "rope_theta": float(inputs.rope_theta),
        "rotary_dim": _rope_rotary_dim(inputs),
        "rope_cast_at": inputs.rope_cast_at,
        "rope_output_dtype": str(_rope_output_dtype(inputs)).replace("torch.", ""),
        "fusion_boundary": fusion_boundary,
        "lse_domain": "attention",
        "preprocess_backends": {
            "rope": "rlkernel.pytorch.rope_reference",
            "qk_rmsnorm": "not_executed_projected_qk_input",
        },
        **_single_gpu_reference_provenance(),
    }


def _single_gpu_reference_provenance() -> dict[str, Any]:
    return {
        "execution_scope": "single_device_correctness_reference",
        "runtime_verified": False,
        "actual_backend": "rlkernel.pytorch.attention_reference",
        "communication_backend": "none",
        "production_ready": False,
        "preprocess_policy": "reference_only_not_production",
        "native_backend_executed": False,
        "preprocess_fallback": True,
        "preprocess_fallback_reason": (
            "single-GPU attribution harness intentionally uses the PyTorch deterministic reference"
        ),
        "qkv_input_boundary": "projected_qkv",
        "qkv_projection_executed": False,
        "output_projection_executed": False,
        "communication": "none",
    }


def _strict_shared_core_provenance(
    inputs: AttentionComparisonInputs,
    *,
    materialization: str,
) -> dict[str, Any]:
    return {
        "execution_scope": "single_device_strict_shared_core",
        "runtime_verified": False,
        "actual_backend": "rlkernel.pytorch.strict_attention_reference",
        "communication_backend": "none",
        "production_ready": False,
        "strict_core_id": "rlkernel.attention.deterministic_core.v1",
        "strict_schedule": "single_batch_single_query_global_kv_blocks",
        "strict_mode": True,
        "native_backend_executed": False,
        "native_attention_arithmetic": False,
        "fallback": False,
        "fallback_reason": None,
        "materialization": materialization,
        "attention_mode": "prefill",
        "qkv_input_boundary": "projected_qkv",
        "qkv_projection_executed": False,
        "output_projection_executed": False,
        "communication": "none",
        "split_kv_policy": "disabled",
        "merge_order": "global_block_index",
        "accum_dtype": "fp32",
        "downcast_at": "final_write",
        "lse_domain": "attention",
        "lse_exported": True,
        "dtype": str(inputs.q.dtype).replace("torch.", ""),
    }


def _strict_decode_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical one-query score/softmax/value schedule for decode replay."""

    qf, kf, vf = q.float(), k.float(), v.float()
    hq, hkv = qf.size(1), kf.size(1)
    if hq % hkv:
        raise ValueError("strict decode Attention requires GQA-compatible head counts")
    if hq != hkv:
        kf = kf.repeat_interleave(hq // hkv, dim=1)
        vf = vf.repeat_interleave(hq // hkv, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(qf.size(-1))
    row_max = scores.amax(dim=-1, keepdim=True)
    finite = torch.isfinite(row_max)
    exp_scores = torch.where(finite, torch.exp(scores - row_max), torch.zeros_like(scores))
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    lse = torch.where(
        row_sum > 0,
        row_max + torch.log(row_sum),
        torch.full_like(row_sum, float("-inf")),
    )
    weights = torch.where(row_sum > 0, exp_scores / row_sum, torch.zeros_like(exp_scores))
    return torch.matmul(weights, vf).to(output_dtype), lse.squeeze(-1)


def _strict_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None,
    key_padding_mask: torch.Tensor | None,
    q_start: int,
    k_start: int,
    total_query_len: int,
    total_kv_len: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed one-batch/one-query arithmetic schedule locally.

    PR2 must remain independently runnable before PR3 is merged, so this small
    reference intentionally mirrors the PR3 schedule instead of importing its
    implementation.  The contract is the same: no Split-KV, FP32 intermediates,
    logical position offsets, and one reduction per query row.
    """

    if q.shape[0] != k.shape[0] or k.shape[:1] != v.shape[:1] or k.shape[2:] != v.shape[2:]:
        raise ValueError("strict Attention q/k/v batch and KV shapes must match")
    hq, hkv = q.size(1), k.size(1)
    if hq % hkv:
        raise ValueError("strict Attention requires GQA-compatible head counts")
    scale_value = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
    output_rows: list[torch.Tensor] = []
    lse_rows: list[torch.Tensor] = []
    for batch_index in range(q.size(0)):
        q_rows: list[torch.Tensor] = []
        lse_batch: list[torch.Tensor] = []
        k_batch = k[batch_index : batch_index + 1].float()
        v_batch = v[batch_index : batch_index + 1].float()
        if hq != hkv:
            k_batch = k_batch.repeat_interleave(hq // hkv, dim=1)
            v_batch = v_batch.repeat_interleave(hq // hkv, dim=1)
        for query_index in range(q.size(2)):
            q_row = q[batch_index : batch_index + 1, :, query_index : query_index + 1, :].float()
            scores = torch.matmul(q_row, k_batch.transpose(-1, -2)) * scale_value
            if causal:
                query_position = total_kv_len - total_query_len + q_start + query_index
                key_positions = torch.arange(k.size(2), device=q.device, dtype=torch.long) + k_start
                scores = scores.masked_fill(
                    key_positions.view(1, 1, 1, -1) > query_position,
                    float("-inf"),
                )
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    ~key_padding_mask[batch_index : batch_index + 1].view(1, 1, 1, -1),
                    float("-inf"),
                )
            row_max = scores.amax(dim=-1, keepdim=True)
            finite = torch.isfinite(row_max)
            exp_scores = torch.where(
                finite,
                torch.exp(scores - row_max),
                torch.zeros_like(scores),
            )
            row_sum = exp_scores.sum(dim=-1, keepdim=True)
            lse = torch.where(
                row_sum > 0,
                row_max + torch.log(row_sum),
                torch.full_like(row_sum, float("-inf")),
            )
            weights = torch.where(row_sum > 0, exp_scores / row_sum, torch.zeros_like(exp_scores))
            q_rows.append(torch.matmul(weights, v_batch).to(output_dtype))
            lse_batch.append(lse.squeeze(-1))
        output_rows.append(torch.cat(q_rows, dim=2))
        lse_rows.append(torch.cat(lse_batch, dim=2))
    return torch.cat(output_rows, dim=0), torch.cat(lse_rows, dim=0)


def _attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None,
    key_padding_mask: torch.Tensor | None,
    q_start: int,
    k_start: int,
    total_query_len: int,
    total_kv_len: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_qkv(q, k, v)
    if key_padding_mask is not None:
        if key_padding_mask.shape != (q.size(0), k.size(2)):
            raise ValueError("key_padding_mask must have shape [B, local_skv]")
        if key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool")

    qf, kf, vf = q.float(), k.float(), v.float()
    hq, sq, dim = qf.shape[1], qf.shape[2], qf.shape[3]
    hkv, skv = kf.shape[1], kf.shape[2]
    if hkv != hq:
        repeat = hq // hkv
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)

    scale_value = scale if scale is not None else 1.0 / math.sqrt(dim)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale_value
    if causal:
        query_offset = total_kv_len - total_query_len
        q_pos = torch.arange(sq, device=q.device) + q_start + query_offset
        k_pos = torch.arange(skv, device=q.device) + k_start
        scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    finite_lse = torch.isfinite(lse)
    weights = torch.exp(scores - lse.unsqueeze(-1))
    weights = torch.where(finite_lse.unsqueeze(-1), weights, torch.zeros_like(weights))
    out = torch.matmul(weights, vf)
    return out.to(output_dtype), lse


def _merge_partial_states(
    states: list[_PartialAttentionState],
    *,
    backend: MergeBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not states:
        raise ValueError("at least one partial state is required")
    ordered = sorted(states, key=lambda state: (state.block_start, state.block_end))
    _validate_partial_states(ordered)
    if backend == "rl_kernel":
        return _merge_partial_states_rl_kernel(ordered)
    if backend == "transformer_engine":
        return _merge_partial_states_transformer_engine(ordered)
    raise ValueError(f"unsupported merge backend: {backend}")


def _merge_partial_states_rl_kernel(
    states: list[_PartialAttentionState],
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_out = states[0].out.float()
    merged_lse = states[0].lse.float()
    for state in states[1:]:
        next_lse = torch.logaddexp(merged_lse, state.lse.float())
        finite = torch.isfinite(next_lse)
        weight_prev = torch.where(
            finite,
            torch.exp(merged_lse - next_lse),
            torch.zeros_like(next_lse),
        )
        weight_next = torch.where(
            finite,
            torch.exp(state.lse.float() - next_lse),
            torch.zeros_like(next_lse),
        )
        merged_out = (
            weight_prev.unsqueeze(-1) * merged_out + weight_next.unsqueeze(-1) * state.out.float()
        )
        merged_lse = next_lse
    return merged_out, merged_lse


def _merge_partial_states_transformer_engine(
    states: list[_PartialAttentionState],
) -> tuple[torch.Tensor, torch.Tensor]:
    te_cp = _load_te_context_parallel()
    merged_out = states[0].out.float()
    merged_lse = states[0].lse.float()
    for state in states[1:]:
        previous_lse = merged_lse
        state_out = state.out.float()
        state_lse = state.lse.float()
        both_masked = torch.isneginf(previous_lse) & torch.isneginf(state_lse)
        te_previous_lse = torch.where(both_masked, torch.zeros_like(previous_lse), previous_lse)
        te_state_lse = torch.where(both_masked, torch.zeros_like(state_lse), state_lse)
        merged_lse = te_previous_lse.clone()
        te_cp.flash_attn_fwd_softmax_lse_correction(merged_lse, te_state_lse)
        merged_out = te_cp.flash_attn_fwd_out_correction_init(
            merged_out,
            merged_lse,
            te_previous_lse,
            seq_dim=2,
        )
        te_cp.flash_attn_fwd_out_correction(
            merged_out,
            state_out,
            merged_lse,
            te_state_lse,
            seq_dim=2,
        )
        if both_masked.any():
            merged_lse = torch.where(both_masked, previous_lse, merged_lse)
            merged_out = torch.where(
                both_masked.unsqueeze(-1),
                torch.zeros_like(merged_out),
                merged_out,
            )
    return merged_out, merged_lse


def _load_te_context_parallel() -> Any:
    try:
        module = importlib.import_module(_TE_CONTEXT_PARALLEL_MODULE)
    except (ImportError, OSError, RuntimeError) as exc:
        raise TransformerEngineUnavailable(str(exc)) from exc
    _probe_te_context_parallel(module)
    return module


def _probe_te_context_parallel(module: Any) -> None:
    missing = [
        name for name in _TE_CONTEXT_PARALLEL_HELPERS if not callable(getattr(module, name, None))
    ]
    if missing:
        raise TransformerEngineUnavailable(
            f"{_TE_CONTEXT_PARALLEL_MODULE} missing required helpers: {', '.join(missing)}"
        )

    for name, expected in _TE_CONTEXT_PARALLEL_HELPERS.items():
        helper = getattr(module, name)
        try:
            parameters = tuple(inspect.signature(helper).parameters)
        except (TypeError, ValueError) as exc:
            raise TransformerEngineUnavailable(
                f"{_TE_CONTEXT_PARALLEL_MODULE}.{name} signature is not inspectable"
            ) from exc
        if parameters[: len(expected)] != expected:
            raise TransformerEngineUnavailable(
                f"{_TE_CONTEXT_PARALLEL_MODULE}.{name} has incompatible signature "
                f"{parameters}; expected prefix {expected}"
            )

    try:
        lse_a = torch.tensor([[[0.0, -1.0]]], dtype=torch.float32)
        lse_b = torch.tensor([[[1.0, -3.0]]], dtype=torch.float32)
        out_a = torch.tensor([[[[1.0, -2.0], [0.5, 2.0]]]], dtype=torch.float32)
        out_b = torch.tensor([[[[-1.0, 4.0], [3.0, -0.5]]]], dtype=torch.float32)
        expected_lse = torch.logaddexp(lse_a, lse_b)
        expected_out = (
            torch.exp(lse_a - expected_lse).unsqueeze(-1) * out_a
            + torch.exp(lse_b - expected_lse).unsqueeze(-1) * out_b
        )

        probed_lse = lse_a.clone()
        module.flash_attn_fwd_softmax_lse_correction(probed_lse, lse_b)
        probed_out = module.flash_attn_fwd_out_correction_init(
            out_a.clone(),
            probed_lse,
            lse_a,
            seq_dim=2,
        )
        module.flash_attn_fwd_out_correction(
            probed_out,
            out_b,
            probed_lse,
            lse_b,
            seq_dim=2,
        )
    except Exception as exc:
        raise TransformerEngineUnavailable(
            f"{_TE_CONTEXT_PARALLEL_MODULE} helper numeric self-test failed: {exc}"
        ) from exc

    if not torch.allclose(probed_lse, expected_lse, atol=1.0e-6, rtol=0.0):
        raise TransformerEngineUnavailable(
            f"{_TE_CONTEXT_PARALLEL_MODULE} LSE helper numeric self-test failed"
        )
    if not torch.allclose(probed_out, expected_out, atol=1.0e-6, rtol=0.0):
        raise TransformerEngineUnavailable(
            f"{_TE_CONTEXT_PARALLEL_MODULE} out helper numeric self-test failed"
        )


def _te_context_parallel_provenance() -> dict[str, Any]:
    return {
        "te_available": True,
        "te_version": _te_version(),
        "te_module": _TE_CONTEXT_PARALLEL_MODULE,
        "te_symbols": list(_TE_CONTEXT_PARALLEL_HELPERS),
        "te_capability_probe": "passed",
        "te_signature_checked": True,
        "te_numeric_selftest": "passed",
        "actual_backend_source": "rl_kernel_te_context_parallel_adapter",
        "deterministic_controls": "not_applicable_merge_only",
        "dropout_policy": "not_applicable_merge_only",
    }


def _te_version() -> str | None:
    for package_name in ("transformer-engine", "transformer_engine"):
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _selected_logps_from_attention(
    out: torch.Tensor,
    inputs: AttentionComparisonInputs,
) -> torch.Tensor:
    if inputs.lm_head_weight is None or inputs.target_ids is None:
        raise ValueError("lm_head_weight and target_ids are required for dlogp drift")
    batch, heads, seq, dim = out.shape
    hidden = out.transpose(1, 2).reshape(batch, seq, heads * dim)
    if inputs.lm_head_weight.shape[1] != hidden.size(-1):
        raise ValueError(
            "lm_head_weight hidden dimension must equal Hq * D; "
            f"got {inputs.lm_head_weight.shape[1]} and {hidden.size(-1)}"
        )
    logits = torch.matmul(hidden.float(), inputs.lm_head_weight.float().transpose(0, 1))
    return selected_logprobs_reference(
        logits,
        inputs.target_ids,
        mask=inputs.active_token_mask,
        output_dtype=torch.float32,
    )


def _selected_logps_from_decode_attention(
    out: torch.Tensor,
    inputs: DecodeAttentionInputs,
) -> torch.Tensor:
    if inputs.lm_head_weight is None or inputs.target_ids is None:
        raise ValueError("lm_head_weight and target_ids are required for decode dlogp drift")
    batch, heads, seq, dim = out.shape
    hidden = out.transpose(1, 2).reshape(batch, seq, heads * dim)
    if inputs.lm_head_weight.shape[1] != hidden.size(-1):
        raise ValueError(
            "lm_head_weight hidden dimension must equal Hq * D; "
            f"got {inputs.lm_head_weight.shape[1]} and {hidden.size(-1)}"
        )
    logits = torch.matmul(hidden.float(), inputs.lm_head_weight.float().transpose(0, 1))
    return selected_logprobs_reference(
        logits,
        inputs.target_ids,
        mask=inputs.active_token_mask,
        output_dtype=torch.float32,
    )


def _drift_stats(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> DriftStats:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match "
            f"reference shape {tuple(reference.shape)}"
        )
    candidate_fp32 = candidate.float()
    reference_fp32 = reference.float()
    raw_diff = (candidate_fp32 - reference_fp32).abs()
    diff = torch.where(
        candidate_fp32 == reference_fp32,
        torch.zeros_like(raw_diff),
        raw_diff,
    )
    values = _active_values(diff, mask)
    active_count = int(values.numel())
    if active_count == 0:
        return DriftStats(0.0, 0.0, 0.0, 0.0, 0)
    return DriftStats(
        max_abs=float(values.max().item()),
        mean_abs=float(values.mean().item()),
        p95_abs=float(torch.quantile(values, 0.95).item()),
        p99_abs=float(torch.quantile(values, 0.99).item()),
        active_count=active_count,
    )


def _active_values(diff: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return diff.reshape(-1)
    if mask.shape == diff.shape:
        return diff[mask.to(device=diff.device, dtype=torch.bool)]
    if mask.ndim == 2 and diff.ndim == 4 and mask.shape == (diff.size(0), diff.size(2)):
        expanded = mask[:, None, :, None].expand_as(diff)
        return diff[expanded.to(device=diff.device, dtype=torch.bool)]
    if mask.ndim == 2 and diff.ndim == 3 and mask.shape == (diff.size(0), diff.size(2)):
        expanded = mask[:, None, :].expand_as(diff)
        return diff[expanded.to(device=diff.device, dtype=torch.bool)]
    raise ValueError(f"mask shape {tuple(mask.shape)} cannot select diff shape {tuple(diff.shape)}")


def _validate_comparison_inputs(inputs: AttentionComparisonInputs) -> None:
    _validate_qkv(inputs.q, inputs.k, inputs.v)
    if inputs.key_padding_mask is not None:
        if inputs.key_padding_mask.shape != (inputs.q.size(0), inputs.k.size(2)):
            raise ValueError("key_padding_mask must have shape [B, Skv]")
        if inputs.key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool")
    if (inputs.lm_head_weight is None) != (inputs.target_ids is None):
        raise ValueError("lm_head_weight and target_ids must be provided together")
    if inputs.target_ids is not None and inputs.target_ids.shape != (
        inputs.q.size(0),
        inputs.q.size(2),
    ):
        raise ValueError("target_ids must have shape [B, Sq]")
    if inputs.active_token_mask is not None:
        if inputs.active_token_mask.shape != (inputs.q.size(0), inputs.q.size(2)):
            raise ValueError("active_token_mask must have shape [B, Sq]")
        if inputs.active_token_mask.dtype != torch.bool:
            raise ValueError("active_token_mask must be bool")
    if (
        not isinstance(inputs.rope_theta, (float, int))
        or isinstance(inputs.rope_theta, bool)
        or not math.isfinite(float(inputs.rope_theta))
        or float(inputs.rope_theta) <= 0
    ):
        raise ValueError("rope_theta must be a positive number")
    if inputs.rope_output_dtype is not None and not isinstance(
        inputs.rope_output_dtype, torch.dtype
    ):
        raise ValueError("rope_output_dtype must be a torch.dtype when provided")


def _validate_rope_inputs(inputs: AttentionComparisonInputs) -> None:
    if inputs.rope_positions is None:
        raise ValueError("rope_positions are required for RoPE+Attention comparison")
    if inputs.q.size(2) != inputs.k.size(2):
        raise ValueError("RoPE+Attention comparison currently requires Sq == Skv")
    if inputs.rope_rotary_dim is not None:
        if isinstance(inputs.rope_rotary_dim, bool) or inputs.rope_rotary_dim <= 0:
            raise ValueError("rope_rotary_dim must be a positive integer when provided")
        if inputs.rope_rotary_dim != inputs.q.size(-1):
            raise ValueError(
                "rope_rotary_dim must equal head_dim until partial-rotary RoPE is supported"
            )
    if inputs.rope_cast_at != "after_rope":
        raise ValueError("rope_cast_at must be 'after_rope' for the current fp32 RoPE reference")
    if (
        inputs.rope_positions.device != inputs.q.device
        or inputs.rope_positions.device != inputs.k.device
    ):
        raise ValueError("rope_positions must be on the same device as q/k")
    if inputs.rope_positions.dtype not in {torch.int32, torch.int64, torch.long}:
        raise ValueError("rope_positions must contain integer token positions")
    if inputs.rope_positions.ndim == 1:
        if inputs.rope_positions.numel() != inputs.q.size(2):
            raise ValueError("1D rope_positions must have length Sq")
    elif inputs.rope_positions.ndim == 2:
        if inputs.rope_positions.shape != (inputs.q.size(0), inputs.q.size(2)):
            raise ValueError("2D rope_positions must have shape [B, Sq]")
    else:
        raise ValueError("rope_positions must have shape [Sq] or [B, Sq]")


def _validate_decode_inputs(inputs: DecodeAttentionInputs) -> None:
    _validate_qkv(inputs.q, inputs.k_cache, inputs.v_cache)
    if inputs.q.device != inputs.k_cache.device or inputs.q.device != inputs.v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if (
        not inputs.q.is_floating_point()
        or not inputs.k_cache.is_floating_point()
        or not inputs.v_cache.is_floating_point()
    ):
        raise ValueError("q, k_cache, and v_cache must use floating-point dtypes")
    if (
        not isinstance(inputs.output_dtype, torch.dtype)
        or not inputs.output_dtype.is_floating_point
    ):
        raise ValueError("output_dtype must be a floating-point torch.dtype")
    if inputs.scale is not None:
        if (
            not isinstance(inputs.scale, (float, int))
            or isinstance(inputs.scale, bool)
            or not math.isfinite(float(inputs.scale))
            or float(inputs.scale) <= 0
        ):
            raise ValueError("scale must be a positive finite number")
    metadata = inputs.metadata
    batch, _, sq, head_dim = inputs.q.shape
    cache_capacity = inputs.k_cache.size(2)
    page_size = _positive_int(metadata.page_size, "page_size")
    if cache_capacity % page_size != 0:
        raise ValueError("physical KV cache capacity must be divisible by page_size")
    physical_page_count = cache_capacity // page_size
    if metadata.cache_position.shape != (batch, sq):
        raise ValueError("cache_position must have shape [B, Sq]")
    if metadata.query_position_ids.shape != (batch, sq):
        raise ValueError("query_position_ids must have shape [B, Sq]")
    if metadata.kv_seq_lens.shape != (batch,):
        raise ValueError("kv_seq_lens must have shape [B]")
    if metadata.block_table.ndim != 2 or metadata.block_table.size(0) != batch:
        raise ValueError("block_table must have shape [B, max_blocks]")
    expected_cache_shape = (batch, cache_capacity)
    if metadata.global_token_positions.shape != expected_cache_shape:
        raise ValueError("global_token_positions must have shape [B, cache_capacity]")
    if metadata.key_position_ids.shape != expected_cache_shape:
        raise ValueError("key_position_ids must have shape [B, cache_capacity]")
    integer_tensors = {
        "cache_position": metadata.cache_position,
        "query_position_ids": metadata.query_position_ids,
        "kv_seq_lens": metadata.kv_seq_lens,
        "block_table": metadata.block_table,
        "global_token_positions": metadata.global_token_positions,
        "key_position_ids": metadata.key_position_ids,
    }
    if metadata.cp_block_owners is not None:
        integer_tensors["cp_block_owners"] = metadata.cp_block_owners
    for name, tensor in integer_tensors.items():
        if tensor.device != inputs.q.device:
            raise ValueError(f"{name} must be on the same device as q/k/v")
        if tensor.dtype not in {torch.int32, torch.int64, torch.long}:
            raise ValueError(f"{name} must contain integers")
    if metadata.cp_block_owners is not None:
        if metadata.cp_block_owners.shape != metadata.block_table.shape:
            raise ValueError("cp_block_owners must have the same shape as block_table")
        cp_world_size = _positive_int(metadata.cp_world_size, "cp_world_size")
        if bool((metadata.cp_block_owners >= cp_world_size).any()):
            raise ValueError("cp_block_owners must be smaller than cp_world_size")
    if not torch.equal(metadata.cache_position, metadata.query_position_ids):
        raise ValueError("cache_position and query_position_ids must identify the same positions")
    if metadata.q_rope_state not in {"pre_rope", "post_rope"}:
        raise ValueError("q_rope_state must be 'pre_rope' or 'post_rope'")
    if metadata.k_cache_rope_state not in {"pre_rope", "post_rope"}:
        raise ValueError("k_cache_rope_state must be 'pre_rope' or 'post_rope'")
    if metadata.prefix_cache_enabled:
        if not metadata.prefix_cache_key:
            raise ValueError("prefix_cache_key is required when prefix cache is enabled")
        _positive_int(metadata.prefix_length, "prefix_length")
        if not metadata.prefix_cache_fingerprint:
            raise ValueError("prefix_cache_fingerprint is required when prefix cache is enabled")
    elif (
        metadata.prefix_cache_key is not None
        or metadata.prefix_length != 0
        or metadata.prefix_cache_fingerprint is not None
    ):
        raise ValueError(
            "prefix cache key/fingerprint must be None and prefix_length must be 0 "
            "when prefix cache is disabled"
        )
    if inputs.rope_cast_at != "after_rope":
        raise ValueError("rope_cast_at must be 'after_rope' for the current fp32 RoPE reference")
    if inputs.rope_rotary_dim is not None:
        if inputs.rope_rotary_dim != head_dim:
            raise ValueError("rope_rotary_dim must equal head_dim")
        _positive_int(inputs.rope_rotary_dim, "rope_rotary_dim")
    if (
        not isinstance(inputs.rope_theta, (float, int))
        or isinstance(inputs.rope_theta, bool)
        or not math.isfinite(float(inputs.rope_theta))
        or float(inputs.rope_theta) <= 0
    ):
        raise ValueError("rope_theta must be a positive number")
    if inputs.q_rope_output_dtype is not None and not isinstance(
        inputs.q_rope_output_dtype, torch.dtype
    ):
        raise ValueError("q_rope_output_dtype must be a torch.dtype when provided")
    if inputs.k_cache_rope_output_dtype is not None and not isinstance(
        inputs.k_cache_rope_output_dtype, torch.dtype
    ):
        raise ValueError("k_cache_rope_output_dtype must be a torch.dtype when provided")
    for name, dtype in (
        ("q_rope_output_dtype", inputs.q_rope_output_dtype),
        ("k_cache_rope_output_dtype", inputs.k_cache_rope_output_dtype),
    ):
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"{name} must be a floating-point torch.dtype")
    if (inputs.k_new is None) != (inputs.v_new is None):
        raise ValueError("k_new and v_new must be provided together")
    append_mode = inputs.k_new is not None
    if append_mode:
        assert inputs.k_new is not None and inputs.v_new is not None
        if inputs.k_new.shape != inputs.v_new.shape:
            raise ValueError("k_new and v_new must have matching shapes")
        expected_new_shape = (batch, inputs.k_cache.size(1), sq, head_dim)
        if inputs.k_new.shape != expected_new_shape:
            raise ValueError("k_new and v_new must have shape [B, Hkv, Sq, D]")
        if inputs.k_new.device != inputs.q.device or inputs.v_new.device != inputs.q.device:
            raise ValueError("k_new and v_new must be on the same device as q")
    if inputs.split_kv is not None and not isinstance(inputs.split_kv, SplitKVSpec):
        raise ValueError("split_kv must be a SplitKVSpec when provided")
    if inputs.split_kv is not None and inputs.split_kv.mode is SplitKVMode.AUTO:
        raise ValueError("decode replay requires disabled or fixed Split-KV, not auto")
    if (
        metadata.q_rope_state == "post_rope"
        and inputs.q_rope_output_dtype is not None
        and inputs.q.dtype != inputs.q_rope_output_dtype
    ):
        raise ValueError("post-RoPE q dtype must match q_rope_output_dtype")
    if (
        metadata.k_cache_rope_state == "post_rope"
        and inputs.k_cache_rope_output_dtype is not None
        and inputs.k_cache.dtype != inputs.k_cache_rope_output_dtype
    ):
        raise ValueError("post-RoPE k_cache dtype must match k_cache_rope_output_dtype")
    if (inputs.lm_head_weight is None) != (inputs.target_ids is None):
        raise ValueError("lm_head_weight and target_ids must be provided together")
    if inputs.target_ids is not None and inputs.target_ids.shape != (batch, sq):
        raise ValueError("target_ids must have shape [B, Sq]")
    if inputs.active_token_mask is not None:
        if inputs.active_token_mask.shape != (batch, sq):
            raise ValueError("active_token_mask must have shape [B, Sq]")
        if inputs.active_token_mask.dtype != torch.bool:
            raise ValueError("active_token_mask must be bool")

    for batch_index in range(batch):
        sequence_length = int(metadata.kv_seq_lens[batch_index].item())
        if sequence_length <= 0 or sequence_length > cache_capacity:
            raise ValueError("each kv_seq_lens entry must be in [1, cache_capacity]")
        block_count = math.ceil(sequence_length / page_size)
        if block_count > metadata.block_table.size(1):
            raise ValueError("block_table does not contain enough logical KV blocks")
        pages = metadata.block_table[batch_index, :block_count]
        if bool(((pages < 0) | (pages >= physical_page_count)).any()):
            raise ValueError("block_table contains an out-of-range physical page")
        if torch.unique(pages).numel() != block_count:
            raise ValueError("active block_table entries must not contain duplicate pages")
        if bool((metadata.block_table[batch_index, block_count:] != -1).any()):
            raise ValueError("unused block_table entries must be -1")
        if metadata.cp_block_owners is not None:
            active_owners = metadata.cp_block_owners[batch_index, :block_count]
            inactive_owners = metadata.cp_block_owners[batch_index, block_count:]
            if bool((active_owners < 0).any()):
                raise ValueError("active cp_block_owners must be non-negative")
            if bool((inactive_owners != -1).any()):
                raise ValueError("unused cp_block_owners entries must be -1")
        slot_index = _decode_logical_slot_index(inputs, batch_index)
        if not torch.isfinite(inputs.q[batch_index]).all():
            raise ValueError("active decode q values must be finite")
        if not torch.isfinite(inputs.k_cache[batch_index, :, slot_index, :]).all():
            raise ValueError("active logical k_cache values must be finite")
        if not torch.isfinite(inputs.v_cache[batch_index, :, slot_index, :]).all():
            raise ValueError("active logical v_cache values must be finite")
        active_slot_mask = torch.zeros(cache_capacity, device=inputs.q.device, dtype=torch.bool)
        active_slot_mask[slot_index] = True
        if bool((metadata.global_token_positions[batch_index, ~active_slot_mask] != -1).any()):
            raise ValueError("unused global_token_positions entries must be -1")
        if bool((metadata.key_position_ids[batch_index, ~active_slot_mask] != -1).any()):
            raise ValueError("unused key_position_ids entries must be -1")
        global_positions = metadata.global_token_positions[batch_index, slot_index]
        if bool((global_positions < 0).any()) or (
            sequence_length > 1 and bool((global_positions[1:] <= global_positions[:-1]).any())
        ):
            raise ValueError(
                "block_table/global_token_positions must reconstruct logical positions "
                "in strictly increasing global order"
            )
        position_offset = int(global_positions[0].item())
        key_positions = metadata.key_position_ids[batch_index, slot_index]
        if not torch.equal(key_positions, global_positions):
            raise ValueError("key_position_ids must match cached global token positions")
        cache_positions = metadata.cache_position[batch_index]
        if append_mode:
            expected_positions = torch.arange(
                position_offset,
                position_offset + sequence_length,
                device=inputs.q.device,
                dtype=global_positions.dtype,
            )
            if not torch.equal(global_positions, expected_positions):
                raise ValueError(
                    "append mode requires cached global positions to form one contiguous range"
                )
            expected_new_positions = torch.arange(
                position_offset + sequence_length,
                position_offset + sequence_length + sq,
                device=inputs.q.device,
                dtype=cache_positions.dtype,
            )
            if not torch.equal(cache_positions, expected_new_positions):
                raise ValueError(
                    "append cache_position must identify the contiguous new-token suffix"
                )
        else:
            query_is_cached = (cache_positions[:, None] == global_positions[None, :]).any(dim=1)
            if not bool(query_is_cached.all()):
                raise ValueError("cache_position must refer to a token present in the KV cache")
        if sq > 1 and bool((cache_positions[1:] <= cache_positions[:-1]).any()):
            raise ValueError("few-query cache_position values must be strictly increasing")

    if metadata.prefix_cache_enabled:
        actual_fingerprint = decode_prefix_cache_fingerprint(
            inputs,
            prefix_length=metadata.prefix_length,
        )
        if actual_fingerprint != metadata.prefix_cache_fingerprint:
            raise ValueError(
                "prefix_cache_fingerprint does not match the logical prefix positions/content"
            )


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have matching shape")
    if q.size(0) != k.size(0) or q.size(3) != k.size(3):
        raise ValueError("q, k, and v must share batch size and head dim")
    if q.size(1) % k.size(1) != 0:
        raise ValueError(f"Hq={q.size(1)} must be divisible by Hkv={k.size(1)}")


def _validate_partial_states(states: list[_PartialAttentionState]) -> None:
    if not states:
        raise ValueError("at least one partial attention state is required")
    first = states[0]
    if first.block_start != 0:
        raise ValueError("partial state coverage must start at logical KV token 0")
    if first.block_end <= first.block_start:
        raise ValueError("partial state ranges must satisfy 0 <= block_start < block_end")
    previous_end = first.block_end
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all partial states must have matching shapes")
        if state.block_end <= state.block_start:
            raise ValueError("partial state block ranges must have positive width")
        if state.block_start != previous_end:
            raise ValueError("partial state block ranges must be contiguous and non-overlapping")
        previous_end = state.block_end


def _require_finite_decode_result(
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    path: str,
) -> None:
    if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
        raise ValueError(f"{path} produced non-finite attention output or LSE")


def _chunk_bounds(length: int, chunk_size: int) -> list[tuple[int, int]]:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    bounds: list[tuple[int, int]] = []
    cursor = 0
    while cursor < length:
        end = min(cursor + chunk_size, length)
        bounds.append((cursor, end))
        cursor = end
    return bounds


def _resolved_decode_split_kv(inputs: DecodeAttentionInputs) -> SplitKVSpec:
    # Existing replay behavior used one partial state per logical cache page.
    # Keep that as the explicit default while allowing disabled/fixed sweeps on
    # the same physical page layout.
    return inputs.split_kv or SplitKVSpec.fixed(inputs.metadata.page_size)


def _decode_split_bounds(length: int, split_kv: SplitKVSpec) -> list[tuple[int, int]]:
    if split_kv.mode is SplitKVMode.AUTO:
        raise ValueError("decode replay cannot materialize an unknown auto Split-KV plan")
    chunk_size = length
    if split_kv.mode is SplitKVMode.FIXED:
        assert split_kv.fixed_split_size is not None
        chunk_size = split_kv.fixed_split_size
    return _chunk_bounds(length, chunk_size)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


__all__ = [
    "AttentionComparisonInputs",
    "AttentionComparisonReport",
    "AttentionPathDrift",
    "AttentionPathResult",
    "DecodeAttentionInputs",
    "DecodeKVCacheMetadata",
    "DriftStats",
    "TransformerEngineUnavailable",
    "compare_single_gpu_rope_attention",
    "compare_single_gpu_attention",
    "compare_decode_kv_replay",
    "decode_kv_cache_fingerprint",
    "decode_rope_identity_fingerprint",
    "decode_prefix_cache_fingerprint",
    "run_chunked_query_attention",
    "run_fused_like_rope_attention",
    "run_full_attention",
    "run_decode_full_prefill_reference",
    "run_decode_kv_replay",
    "run_decode_strict_shared_core",
    "run_paged_kv_attention",
    "run_strict_shared_core_attention",
    "run_strict_shared_core_chunked_attention",
    "run_strict_shared_core_paged_layout_attention",
    "run_unfused_rope_attention",
    "transformer_engine_context_parallel_available",
]
