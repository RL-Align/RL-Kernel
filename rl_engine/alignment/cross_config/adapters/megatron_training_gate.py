# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P1: fail-closed validation of one frozen training-call snapshot.

The gate reads only a :class:`TrainingCallSnapshot` plus caller-supplied
context that the VIME request does not carry (row layout, active mask,
checkpoint/weight identity, weight lock, LM-head LoRA state, sampling record).
It never touches live runtime objects and never recomputes numerics. Checks
run identity/discrete first and numerics last; the first violation raises
:class:`T05ContractError` with a contract section 7 status.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from rl_engine.alignment.cross_config.adapters.megatron_training import (
    PreUpdateWeightLock,
    ShardDigest,
    T05ContractError,
    canonical_sha256,
)
from rl_engine.integrations.megatron_training_readback import (
    EvidenceSource,
    TrainingCallSnapshot,
)
from rl_engine.integrations.runtime import _contains_fallback

TRAINING_M10_EVIDENCE_SCHEMA_VERSION = "rlkernel.p7.t05.training_m10_evidence.v1-draft"
STRICT_BACKEND_ID = "rlkernel.linear_logp.bitwise.v1"

RECOMPUTE_ENTRYPOINTS = frozenset(
    {"sm90_deterministic_linear_logp", "sm90_deterministic_linear_logp_tp"}
)
REUSE_ENTRYPOINTS = frozenset(
    {
        "sm90_deterministic_logp_from_local_logits_tp",
        "rocm_vocab_parallel_logp_from_local_logits_tp",
    }
)
ROCM_SHARED_ENTRYPOINT = "rocm_deterministic_linear_logp_tp"  # __call__ and training reuse
TOP_P_ENTRYPOINTS = frozenset(
    {
        "sm90_deterministic_top_p_logp_from_local_logits_tp",
        "sparse_nucleus_logp_from_local_logits_tp",
        "sparse_nucleus_logp_from_replicated_logits",
    }
)
IDENTITY_SAMPLING = {
    "temperature": 1.0,
    "top_k": "DISABLED",
    "top_p": "IDENTITY",
    "min_p": "DISABLED",
}
SAMPLING_LAYERS = ("requested", "normalized", "materialized", "actual")

_MISSING = object()


@dataclass(frozen=True)
class RowLayout:
    """Global (unsplit) packed samples behind the local rows of one call."""

    sample_ids: tuple[str, ...]
    sample_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sample_ids or len(self.sample_ids) != len(self.sample_lengths):
            raise ValueError("row layout needs one id per sample length")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("row layout sample ids must be unique")
        if any(int(length) <= 0 for length in self.sample_lengths):
            raise ValueError("row layout sample lengths must be positive")

    def local_row_keys(
        self, *, layout: str, cp_rank: int, cp_world: int
    ) -> tuple[Optional[tuple[str, int]], ...]:
        """(sample_id, position) for each local row; ``None`` marks a padding row.

        Reuses ``canonical_cp.packed_gather_indices`` (the mapping used by the
        canonical CP gradient path) instead of re-deriving the zigzag layout.
        """

        from rl_engine.integrations.canonical_cp import packed_gather_indices

        lengths = [int(length) for length in self.sample_lengths]
        if layout == "single":
            if cp_world != 1:
                raise T05ContractError(
                    "IDENTITY_DRIFT", "token_layout.layout", "single layout with CP > 1"
                )
            return tuple(
                (sample, position)
                for sample, length in zip(self.sample_ids, lengths, strict=True)
                for position in range(length)
            )
        if layout != "zigzag":
            raise T05ContractError(
                "UNSUPPORTED_CAPABILITY",
                "token_layout.layout",
                f"layout {layout!r} has no verified row mapping",
            )
        widths = [-(-length // (2 * cp_world)) for length in lengths]
        local_rows = sum(2 * width for width in widths)
        gather = packed_gather_indices(lengths, local_rows, cp_world)
        rank_major: list[Optional[tuple[str, int]]] = [None] * (local_rows * cp_world)
        canonical = [
            (sample, position)
            for sample, length in zip(self.sample_ids, lengths, strict=True)
            for position in range(length)
        ]
        for key, index in zip(canonical, gather, strict=True):
            if rank_major[index] is not None:
                raise T05ContractError(
                    "AMBIGUOUS_GLOBAL_TOKEN_MAPPING", "row_layout", f"row {index} mapped twice"
                )
            rank_major[index] = key
        return tuple(rank_major[cp_rank * local_rows : (cp_rank + 1) * local_rows])


@dataclass(frozen=True)
class TrainingCallContext:
    """Identity the VIME request does not carry; supplied by the recorded or live provider."""

    checkpoint_id: str
    weight_id: str
    weight_lock: PreUpdateWeightLock
    row_layout: RowLayout
    active_token_mask: torch.Tensor  # BOOL [N] aligned with the request rows
    lm_head_lora: Optional[bool]
    sampling_record: Optional[Mapping[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class MissingEvidence:
    """Evidence the contract requires but this boundary cannot observe; never a pass."""

    field: str
    status: str
    required_interface: str
    depends_on: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "status": self.status,
            "required_interface": self.required_interface,
            "depends_on": self.depends_on,
        }


_ALWAYS_MISSING = (
    MissingEvidence(
        "gemm.executed_spec",
        "UPSTREAM_EVIDENCE_MISSING",
        "readback of the executed LM-head GEMM spec (accumulator dtype, K order, tile, "
        "split-K, output cast point) bound to the fixed-K/batch-invariant harness",
        "T01 (harness/spec) + strict kernel owner (additive provenance)",
    ),
    MissingEvidence(
        "gemm.harness_fingerprint",
        "UPSTREAM_EVIDENCE_MISSING",
        "fingerprint of the T01 fixed-K/batch-invariant GEMM harness run on this build",
        "T01",
    ),
    MissingEvidence(
        "reduction.lse",
        "MISSING_PROVENANCE",
        "this call's LSE in the provider result: LinearLogpWrapper.__call__ discards it and the "
        "VIME provider requests it only for metric entropy without returning it",
        "rl_engine VIME provider (additive contract delta, T02 review)",
    ),
    MissingEvidence(
        "reduction.global_max_sum",
        "MISSING_PROVENANCE",
        "per-row global max and sum; the strict merge combines per-rank LSE values only",
        "strict kernel owner + T02 (M10 contract, additive delta)",
    ),
    MissingEvidence(
        "reduction.padded_vocab_mask_applied",
        "MISSING_PROVENANCE",
        "kernel readback of the real-vocab bound it masked (only the declared real_vocab_size "
        "is cross-checked today)",
        "strict kernel owner + T02",
    ),
)
_TP_MISSING = (
    MissingEvidence(
        "reduction.canonical_split_actual",
        "MISSING_PROVENANCE",
        "executed canonical summary intervals, summaries_per_rank and layout_matches "
        "(only environment configuration is visible)",
        "strict kernel owner (additive provenance, T01/T02 review)",
    ),
    MissingEvidence(
        "reduction.redistribution_collective",
        "MISSING_PROVENANCE",
        "whether logits were all-gathered for canonical redistribution, and by which collective",
        "strict kernel owner (additive provenance)",
    ),
    MissingEvidence(
        "reduction.merge_order_actual",
        "MISSING_PROVENANCE",
        "the rank order actually used to merge summaries (only 'rank_ordered_fixed' is declared)",
        "strict kernel owner (additive provenance)",
    ),
)


@dataclass(frozen=True)
class TrainingM10Evidence:
    """Validated training-side M10 output bound to this call's evidence."""

    selected_logprob: torch.Tensor  # FP32 [N]
    active_token_mask: torch.Tensor  # BOOL [N]
    row_keys: tuple[Optional[tuple[str, int]], ...]
    active_raw_bytes: bytes
    active_bytes_sha256: str
    path: str
    entrypoint: str
    runtime_platform: str
    tp_rank: int
    tp_world_size: int
    vocab_start: int
    vocab_end: int
    real_vocab_size: int
    padded_vocab_size: int
    weight_shard: ShardDigest
    snapshot_fingerprint: str
    certifying_sources: tuple[str, ...]
    target_owner_ranks: tuple[int, ...] = ()  # per active row, from observed targets
    canonical_configured: Optional[Mapping[str, Any]] = None  # configured only, never evidence
    runtime: Optional[Mapping[str, Any]] = None
    build: Optional[Mapping[str, Any]] = None
    missing_evidence: tuple[MissingEvidence, ...] = ()
    schema_version: str = TRAINING_M10_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_keys": [list(key) if key is not None else None for key in self.row_keys],
            "active_bytes_sha256": self.active_bytes_sha256,
            "active_count": int(self.active_token_mask.sum().item()),
            "path": self.path,
            "entrypoint": self.entrypoint,
            "runtime_platform": self.runtime_platform,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "vocab_ownership": [self.vocab_start, self.vocab_end],
            "real_vocab_size": self.real_vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "weight_shard": self.weight_shard.to_dict(),
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "certifying_sources": list(self.certifying_sources),
            "target_owner_ranks": list(self.target_owner_ranks),
            "canonical_configured": (
                None if self.canonical_configured is None else dict(self.canonical_configured)
            ),
            "runtime": None if self.runtime is None else dict(self.runtime),
            "build": None if self.build is None else dict(self.build),
            "missing_evidence": [item.to_dict() for item in self.missing_evidence],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def _fail(status: str, field_name: str, detail: str) -> T05ContractError:
    return T05ContractError(status, field_name, detail)


def _get(snapshot: TrainingCallSnapshot, name: str) -> Any:
    try:
        item = snapshot.field(name)
    except KeyError:
        return _MISSING
    return item.value if item.present else _MISSING


def _require(snapshot: TrainingCallSnapshot, name: str) -> Any:
    value = _get(snapshot, name)
    if value is _MISSING:
        raise _fail("MISSING_PROVENANCE", name, "not captured for this call")
    return value


def _tensor(snapshot: TrainingCallSnapshot, name: str, *, dtype: str, ndim: int) -> dict:
    value = _require(snapshot, name)
    if not isinstance(value, Mapping) or value.get("kind") != "tensor":
        raise _fail("SCHEMA_MISMATCH", name, f"expected a tensor, got {value!r}")
    if value["dtype"] != dtype or len(value["shape"]) != ndim:
        raise _fail(
            "SCHEMA_MISMATCH",
            name,
            f"expected {dtype} with {ndim} dims, got {value['dtype']} {value['shape']}",
        )
    return dict(value)


def _int(snapshot: TrainingCallSnapshot, name: str) -> int:
    value = _require(snapshot, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail("SCHEMA_MISMATCH", name, f"expected an integer, got {value!r}")
    return value


def _check_snapshot(snapshot: TrainingCallSnapshot) -> None:
    if snapshot.capture_error is not None:
        raise _fail("MISSING_PROVENANCE", "snapshot", f"capture failed: {snapshot.capture_error}")
    if snapshot.operator_error is not None:
        raise _fail("MISSING_PROVENANCE", "snapshot", f"operator raised {snapshot.operator_error}")
    if snapshot.selected_logprob is None:
        raise _fail("MISSING_PROVENANCE", "result.logp", "no selected logprob captured")


def _tp_coordinates(snapshot: TrainingCallSnapshot) -> tuple[int, int]:
    coords = _require(snapshot, "tensor_parallel_group.coordinates")
    if coords.get("group") is None:
        rank, world = 0, 1
    elif coords.get("rank") is None or coords.get("world_size") is None:
        raise _fail("MISSING_PROVENANCE", "tensor_parallel_group", "group coordinates unreadable")
    else:
        rank, world = int(coords["rank"]), int(coords["world_size"])
    metadata = _get(snapshot, "metadata")
    if isinstance(metadata, Mapping):
        for key, observed in (("tp_rank", rank), ("tp_world_size", world)):
            declared = metadata.get(key)
            if declared is not None and int(declared) != observed:
                raise _fail(
                    "IDENTITY_DRIFT",
                    f"metadata.{key}",
                    f"configured {declared} != observed {observed} from the TP group",
                )
    return rank, world


def _classify_path(linear: Mapping[str, Any], execution: Mapping[str, Any]) -> tuple[str, str]:
    entry = linear.get("strict_entrypoint")
    if not isinstance(entry, str) or not entry:
        raise _fail("MISSING_PROVENANCE", "linear_logp.strict_entrypoint", "absent")
    if entry in TOP_P_ENTRYPOINTS or linear.get("top_p_replay") or execution.get("top_p_replay"):
        raise _fail(
            "SAMPLING_PARAMS_MISMATCH", "linear_logp.strict_entrypoint", f"top-p path {entry}"
        )
    if entry in RECOMPUTE_ENTRYPOINTS:
        path = "recompute"
    elif entry in REUSE_ENTRYPOINTS:
        path = "reuse_local_logits"
    elif entry == ROCM_SHARED_ENTRYPOINT:
        path = "reuse_local_logits" if linear.get("lm_head_result_reused") else "recompute"
    else:
        raise _fail("MISSING_PROVENANCE", "linear_logp.strict_entrypoint", f"unknown {entry!r}")
    reused = execution.get("lm_head_result_reused")
    if reused is not None and bool(reused) != (path == "reuse_local_logits"):
        raise _fail(
            "MISSING_PROVENANCE",
            "provenance.execution.lm_head_result_reused",
            f"provider says reused={reused} but entrypoint {entry} implies {path}",
        )
    return path, entry


def _check_same_call(
    snapshot: TrainingCallSnapshot,
    linear: Mapping[str, Any],
    provider: Mapping[str, Any],
    *,
    path: str,
    hidden: Mapping[str, Any],
    weight: Mapping[str, Any],
    logits: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> None:
    """Declared provenance must describe this call's observed inputs, not a previous one."""

    expected = {"target_shape": targets["shape"]}
    if path == "recompute":
        expected.update(hidden_shape=hidden["shape"], lm_head_weight_shape=weight["shape"])
    else:
        expected.update(local_logits_shape=logits["shape"])
    for key, observed in expected.items():
        if list(linear.get(key) or []) != list(observed):
            raise _fail(
                "MISSING_PROVENANCE",
                f"linear_logp.{key}",
                f"declared {linear.get(key)} != observed {observed}: provenance not from this call",
            )
    request = provider.get("request")
    if isinstance(request, Mapping) and request.get("hidden_shape") is not None:
        if list(request["hidden_shape"]) != list(hidden["shape"]):
            raise _fail(
                "MISSING_PROVENANCE",
                "provenance.request.hidden_shape",
                "provider evidence is stale",
            )
    operator = _get(snapshot, "operator.provenance")
    if isinstance(operator, Mapping) and isinstance(operator.get("linear_logp"), Mapping):
        if dict(operator["linear_logp"]) != dict(linear):
            raise _fail(
                "MISSING_PROVENANCE", "operator.provenance.linear_logp", "differs from wrapper"
            )


def _check_sampling(
    snapshot: TrainingCallSnapshot, record: Optional[Mapping[str, Mapping[str, Any]]]
) -> None:
    keep_mask = _get(snapshot, "log_prob_keep_mask")
    if keep_mask is not _MISSING and keep_mask is not None:
        raise _fail("SAMPLING_PARAMS_MISMATCH", "log_prob_keep_mask", "top-p replay mask present")
    metadata = _get(snapshot, "metadata")
    if isinstance(metadata, Mapping) and metadata.get("top_p_sparse_token_ids") is not None:
        raise _fail("SAMPLING_PARAMS_MISMATCH", "metadata.top_p_sparse_token_ids", "sparse nucleus")
    temperature = _require(snapshot, "temperature.identity")
    if not temperature.get("identity"):
        raise _fail("SAMPLING_PARAMS_MISMATCH", "temperature", f"non-identity {temperature}")
    if record is None:
        raise _fail("MISSING_PROVENANCE", "sampling_record", "four-layer sampling record absent")
    for layer in SAMPLING_LAYERS:
        if not isinstance(record.get(layer), Mapping):
            raise _fail("MISSING_PROVENANCE", f"sampling_record.{layer}", "layer absent")
    for layer in SAMPLING_LAYERS[1:]:
        values = dict(record[layer])
        for key, identity in IDENTITY_SAMPLING.items():
            if key not in values:
                raise _fail("MISSING_PROVENANCE", f"sampling_record.{layer}.{key}", "absent")
            if values[key] != identity:
                raise _fail(
                    "SAMPLING_PARAMS_MISMATCH",
                    f"sampling_record.{layer}.{key}",
                    f"{values[key]!r} != {identity!r}",
                )


def _canonical_configured(
    snapshot: TrainingCallSnapshot,
    *,
    tensor_parallel: bool,
    tp_world: int,
    padded: int,
    local_size: int,
) -> Optional[dict[str, Any]]:
    """Mirror the kernel's canonical-TP parsing on configured values (legality only)."""

    raw_tp = _get(snapshot, "configured.canonical_tp")
    raw_vocab = _get(snapshot, "configured.canonical_vocab_size")
    if raw_tp is _MISSING or raw_vocab is _MISSING:
        raise _fail("MISSING_PROVENANCE", "configured.canonical_tp", "not captured")
    if not tensor_parallel:
        return {"applicable": False, "canonical_tp": raw_tp, "canonical_vocab_size": raw_vocab}
    try:
        canonical_tp = tp_world if raw_tp in (None, "") else int(str(raw_tp).strip())
        canonical_vocab = padded if raw_vocab in (None, "") else int(str(raw_vocab).strip())
    except ValueError as exc:
        raise _fail("IDENTITY_DRIFT", "configured.canonical_tp", f"unparsable: {exc}") from exc
    if canonical_tp < tp_world or canonical_tp % tp_world:
        raise _fail(
            "IDENTITY_DRIFT",
            "configured.canonical_tp",
            f"{canonical_tp} is not a positive multiple of physical TP {tp_world}",
        )
    if canonical_vocab < padded or canonical_vocab % canonical_tp:
        raise _fail(
            "IDENTITY_DRIFT",
            "configured.canonical_vocab_size",
            f"{canonical_vocab} must be >= padded {padded} and divisible by {canonical_tp}",
        )
    summaries = canonical_tp // tp_world
    canonical_local = summaries * (canonical_vocab // canonical_tp)
    return {
        "applicable": True,
        "source": "configured",
        "canonical_tp": canonical_tp,
        "canonical_vocab_size": canonical_vocab,
        "summaries_per_rank": summaries,
        "canonical_local_vocab": canonical_local,
        "layout_matches_if_configured": canonical_vocab == padded and canonical_local == local_size,
    }


def validate_training_call(
    snapshot: TrainingCallSnapshot, context: TrainingCallContext
) -> TrainingM10Evidence:
    """Validate one strict training logp call; raise on the first contract violation."""

    _check_snapshot(snapshot)

    hidden = _tensor(snapshot, "context.hidden", dtype="bfloat16", ndim=2)
    weight = _tensor(snapshot, "context.projection.weight", dtype="bfloat16", ndim=2)
    targets = _tensor(snapshot, "target_ids", dtype="int64", ndim=1)
    logits = _require(snapshot, "logits")
    if not isinstance(logits, Mapping) or len(logits.get("shape", ())) != 2:
        raise _fail("SCHEMA_MISMATCH", "logits", "expected [T, V_local]")
    rows, width = hidden["shape"]
    local_start = _int(snapshot, "context.vocab_partition.local_start")
    local_size = _int(snapshot, "context.vocab_partition.local_size")
    real = _int(snapshot, "context.vocab_partition.real_size")
    padded = _int(snapshot, "context.vocab_partition.padded_size")
    if weight["shape"] != [local_size, width] or targets["shape"] != [rows]:
        raise _fail(
            "SCHEMA_MISMATCH",
            "context",
            f"hidden {hidden['shape']}, weight {weight['shape']}, targets {targets['shape']}, "
            f"local vocab {local_size}",
        )
    if logits["shape"] != [rows, local_size]:
        raise _fail("SCHEMA_MISMATCH", "logits", f"{logits['shape']} != {[rows, local_size]}")

    layout = _require(snapshot, "token_layout.layout")
    cp_rank = _int(snapshot, "token_layout.rank")
    cp_world = _int(snapshot, "token_layout.world_size")
    row_keys = context.row_layout.local_row_keys(layout=layout, cp_rank=cp_rank, cp_world=cp_world)
    if len(row_keys) != rows:
        raise _fail(
            "IDENTITY_DRIFT", "row_layout", f"layout gives {len(row_keys)} rows, call has {rows}"
        )

    tp_rank, tp_world = _tp_coordinates(snapshot)
    if local_start != tp_rank * local_size or padded != local_size * tp_world or real > padded:
        raise _fail(
            "IDENTITY_DRIFT",
            "context.vocab_partition",
            f"[{local_start}, {local_start + local_size}) of padded {padded} (real {real}) "
            f"does not match TP rank {tp_rank}/{tp_world}",
        )

    if _get(snapshot, "context.projection.bias") not in (_MISSING, None):
        raise _fail(
            "UNSUPPORTED_CAPABILITY", "context.projection.bias", "not covered by the weight lock"
        )
    if "sha256" not in weight:
        raise _fail("MISSING_PROVENANCE", "context.projection.weight", "captured without digest")
    shard = ShardDigest(
        tp_rank=tp_rank,
        vocab_start=local_start,
        vocab_end=local_start + local_size,
        shape=tuple(weight["shape"]),
        dtype=weight["dtype"],
        sha256=weight["sha256"],
    )
    context.weight_lock.verify_shard(
        shard, checkpoint_id=context.checkpoint_id, weight_id=context.weight_id
    )

    linear = _require(snapshot, "linear_logp.provenance")
    provider = _require(snapshot, "result.provenance")
    operator = _get(snapshot, "operator.provenance")
    execution = provider.get("execution") if isinstance(provider, Mapping) else None
    if not isinstance(linear, Mapping) or not isinstance(execution, Mapping):
        raise _fail("MISSING_PROVENANCE", "provenance.execution", "provider execution block absent")
    path, entry = _classify_path(linear, execution)
    _check_same_call(
        snapshot,
        linear,
        provider,
        path=path,
        hidden=hidden,
        weight=weight,
        logits=logits,
        targets=targets,
    )
    if (
        linear.get("vocab_start_index") != local_start
        or linear.get("global_vocab_size") != padded
        or linear.get("real_vocab_size") != real
    ):
        raise _fail("IDENTITY_DRIFT", "linear_logp.vocab", "wrapper vocab differs from partition")
    backend = _require(snapshot, "result.backend_id")
    if backend != STRICT_BACKEND_ID or linear.get("actual_backend") != STRICT_BACKEND_ID:
        raise _fail("SILENT_FALLBACK", "backend_id", f"{backend} / {linear.get('actual_backend')}")
    if linear.get("deterministic_linear_logp") is not True or _contains_fallback(
        {"provider": provider, "operator": operator, "linear": linear}
    ):
        raise _fail("SILENT_FALLBACK", "provenance", "non-deterministic or fallback route")
    platform = linear.get("runtime_platform")
    if platform not in ("cuda", "rocm"):
        raise _fail("MISSING_PROVENANCE", "linear_logp.runtime_platform", repr(platform))

    _check_sampling(snapshot, context.sampling_record)

    if context.lm_head_lora is None:
        raise _fail("MISSING_PROVENANCE", "lm_head_lora", "LM-head LoRA state not inspected")
    if context.lm_head_lora:
        raise _fail("UNSUPPORTED_CAPABILITY", "lm_head_lora", "LM-head LoRA is out of contract")

    mask = context.active_token_mask
    if mask.dtype != torch.bool or tuple(mask.shape) != (rows,):
        raise _fail("SCHEMA_MISMATCH", "active_token_mask", f"{mask.dtype} {tuple(mask.shape)}")
    mask = mask.detach().cpu()
    padding_active = [index for index, key in enumerate(row_keys) if key is None and mask[index]]
    if padding_active:
        raise _fail("IDENTITY_DRIFT", "active_token_mask", f"padding rows {padding_active} active")
    if not bool(mask.any()):
        raise _fail("ZERO_ACTIVE_TOKENS", "active_token_mask", "no active token")

    selected = snapshot.selected_logprob
    assert selected is not None
    if selected.dtype != torch.float32 or selected.numel() != rows:
        raise _fail(
            "SCHEMA_MISMATCH",
            "result.logp",
            f"{selected.dtype} {tuple(selected.shape)} != fp32 [{rows}]",
        )
    selected = selected.reshape(rows)
    active_values = selected[mask]
    if not bool(torch.isfinite(active_values).all()):
        raise _fail("NON_FINITE", "result.logp", "non-finite active selected logprob")
    raw = active_values.contiguous().view(torch.int32).numpy().tobytes()

    target_values = snapshot.target_ids
    if target_values is None or target_values.numel() != rows:
        raise _fail("MISSING_PROVENANCE", "target_ids", "target values not captured for this call")
    target_values = target_values.reshape(rows).to(torch.int64)
    active_targets = target_values[mask]
    if bool(((active_targets < 0) | (active_targets >= real)).any()):
        raise _fail(
            "IDENTITY_DRIFT",
            "target_ids",
            f"active target outside the real vocabulary [0, {real}) (padded tail is masked)",
        )
    owners = tuple(int(target) // local_size for target in active_targets.tolist())
    tensor_parallel = tp_world > 1 or entry.endswith("_tp")
    canonical = _canonical_configured(
        snapshot,
        tensor_parallel=tensor_parallel,
        tp_world=tp_world,
        padded=padded,
        local_size=local_size,
    )
    missing = _ALWAYS_MISSING + (_TP_MISSING if tensor_parallel else ())
    build = {
        item.name: item.value
        for item in snapshot.fields
        if item.name.startswith("build.") and item.present
    }
    runtime = {
        name: _get(snapshot, name)
        for name in (
            "runtime.rng_before",
            "runtime.rng_after",
            "runtime.rng_consumed",
            "runtime.stream",
        )
        if _get(snapshot, name) is not _MISSING
    }

    certifying = tuple(
        sorted(
            {
                item.name
                for item in snapshot.fields
                if item.source in (EvidenceSource.OBSERVED, EvidenceSource.UPSTREAM)
            }
        )
    )
    return TrainingM10Evidence(
        selected_logprob=selected.clone(),
        active_token_mask=mask.clone(),
        row_keys=row_keys,
        active_raw_bytes=raw,
        active_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        path=path,
        entrypoint=entry,
        runtime_platform=str(platform),
        tp_rank=tp_rank,
        tp_world_size=tp_world,
        vocab_start=local_start,
        vocab_end=local_start + local_size,
        real_vocab_size=real,
        padded_vocab_size=padded,
        weight_shard=shard,
        snapshot_fingerprint=snapshot.fingerprint,
        certifying_sources=certifying,
        target_owner_ranks=owners,
        canonical_configured=canonical,
        runtime=runtime,
        build=build,
        missing_evidence=missing,
    )


__all__ = [
    "IDENTITY_SAMPLING",
    "MissingEvidence",
    "STRICT_BACKEND_ID",
    "TRAINING_M10_EVIDENCE_SCHEMA_VERSION",
    "RowLayout",
    "TrainingCallContext",
    "TrainingM10Evidence",
    "validate_training_call",
]
