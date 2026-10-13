# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 training-side identity: failure-batch filtering and the pre-update weight lock.

These types are T05's interface proposal for the shared T05/T06 failure-batch
contract (p7-task-contract.v3, T05/T06). They are versioned drafts until T01
publishes the public schema; nothing here defines the public P7 status enum or
Foundation ABI. Failures raise :class:`T05ContractError` carrying one of the
contract section 7 status names in :data:`T05_EMITTED_STATUSES`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import torch

FAILURE_BATCH_SCHEMA_VERSION = "rlkernel.p7.t05.failure_batch.v1-draft"
WEIGHT_LOCK_SCHEMA_VERSION = "rlkernel.p7.t05.pre_update_weight_lock.v1-draft"

# Literal section 7 names emitted by T05; the public enum itself belongs to T01.
T05_EMITTED_STATUSES = frozenset(
    {
        "IDENTITY_DRIFT",
        "PRE_UPDATE_WEIGHT_DRIFT",
        "MISSING_PROVENANCE",
        "MISSING_RANK",
        "SCHEMA_MISMATCH",
        "UNSUPPORTED_CAPABILITY",
        "SAMPLING_PARAMS_MISMATCH",
        "SILENT_FALLBACK",
        "UPSTREAM_EVIDENCE_MISSING",
        "UPSTREAM_CONTRACT_MISMATCH",
        "UPSTREAM_VERDICT_MISSING",
        "STALE_RUN_METADATA",
        "CORRUPT_ARTIFACT",
        "ZERO_ACTIVE_TOKENS",
        "NON_FINITE",
        "AMBIGUOUS_GLOBAL_TOKEN_MAPPING",
    }
)

_FILTER_DIMENSIONS = ("run_id", "checkpoint_id", "sample_id", "rank", "module")


class T05ContractError(RuntimeError):
    """Fail-closed T05 error with a contract status, the offending field and detail."""

    def __init__(self, status: str, field_name: str, detail: str):
        if status not in T05_EMITTED_STATUSES:
            raise ValueError(f"T05 does not emit status {status!r}")
        self.status = status
        self.field_name = field_name
        self.detail = detail
        super().__init__(f"{status}[{field_name}]: {detail}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's raw element bytes in logical (contiguous) order."""

    flat = tensor.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ShardDigest:
    """Identity of one LM-head local shard: ownership range plus raw-byte digest."""

    tp_rank: int
    vocab_start: int
    vocab_end: int
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def __post_init__(self) -> None:
        if self.tp_rank < 0 or self.vocab_start < 0 or self.vocab_end <= self.vocab_start:
            raise ValueError("shard digest requires tp_rank >= 0 and a non-empty [start, end)")
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))

    @classmethod
    def from_tensor(
        cls, weight: torch.Tensor, *, tp_rank: int, vocab_start: int, vocab_end: int
    ) -> "ShardDigest":
        if weight.ndim != 2 or weight.size(0) != vocab_end - vocab_start:
            raise ValueError("LM-head shard must be [vocab_end - vocab_start, hidden]")
        return cls(
            tp_rank=tp_rank,
            vocab_start=vocab_start,
            vocab_end=vocab_end,
            shape=tuple(weight.shape),
            dtype=str(weight.dtype).replace("torch.", ""),
            sha256=tensor_bytes_sha256(weight),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "vocab_start": self.vocab_start,
            "vocab_end": self.vocab_end,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PreUpdateWeightLock:
    """Immutable pre-update checkpoint/weight identity shared by baseline, ablation and rerun."""

    checkpoint_id: str
    weight_id: str
    lm_head_shards: tuple[ShardDigest, ...]
    schema_version: str = WEIGHT_LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WEIGHT_LOCK_SCHEMA_VERSION:
            raise T05ContractError(
                "SCHEMA_MISMATCH", "weight_lock.schema_version", repr(self.schema_version)
            )
        _nonempty(self.checkpoint_id, "checkpoint_id")
        _nonempty(self.weight_id, "weight_id")
        shards = tuple(sorted(self.lm_head_shards, key=lambda shard: shard.tp_rank))
        if not shards:
            raise ValueError("weight lock requires at least one LM-head shard digest")
        ranks = [shard.tp_rank for shard in shards]
        if len(set(ranks)) != len(ranks):
            raise ValueError("weight lock has duplicate TP ranks")
        object.__setattr__(self, "lm_head_shards", shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "weight_id": self.weight_id,
            "lm_head_shards": [shard.to_dict() for shard in self.lm_head_shards],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def verify_shard(self, observed: ShardDigest, *, checkpoint_id: str, weight_id: str) -> None:
        """Fail closed unless the observed shard is byte-identical to the locked one."""

        if checkpoint_id != self.checkpoint_id or weight_id != self.weight_id:
            raise T05ContractError(
                "PRE_UPDATE_WEIGHT_DRIFT",
                "checkpoint_id/weight_id",
                f"locked ({self.checkpoint_id}, {self.weight_id}) != "
                f"observed ({checkpoint_id}, {weight_id})",
            )
        locked = {shard.tp_rank: shard for shard in self.lm_head_shards}.get(observed.tp_rank)
        if locked is None:
            raise T05ContractError(
                "MISSING_PROVENANCE",
                f"lm_head_shards[{observed.tp_rank}]",
                "observed TP rank has no locked shard digest",
            )
        if locked != observed:
            raise T05ContractError(
                "PRE_UPDATE_WEIGHT_DRIFT",
                f"lm_head_shards[{observed.tp_rank}]",
                f"locked {locked.to_dict()} != observed {observed.to_dict()}",
            )


@dataclass(frozen=True)
class FailureBatchEntry:
    """One failing (run, checkpoint, sample, rank, module) slice with its stable case ID."""

    stable_case_id: str
    run_id: str
    checkpoint_id: str
    sample_id: str
    rank: str
    module: str
    global_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("stable_case_id", "run_id", "checkpoint_id", "sample_id", "rank", "module"):
            _nonempty(getattr(self, name), name)
        tokens = tuple(int(token) for token in self.global_token_ids)
        if len(set(tokens)) != len(tokens):
            raise T05ContractError(
                "AMBIGUOUS_GLOBAL_TOKEN_MAPPING",
                "global_token_ids",
                f"duplicate global token ids in {self.stable_case_id}",
            )
        object.__setattr__(self, "global_token_ids", tokens)

    @property
    def sort_key(self) -> tuple[str, ...]:
        return (self.run_id, self.checkpoint_id, self.sample_id, self.rank, self.module)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_case_id": self.stable_case_id,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "sample_id": self.sample_id,
            "rank": self.rank,
            "module": self.module,
            "global_token_ids": list(self.global_token_ids),
        }


@dataclass(frozen=True)
class FailureBatchManifest:
    """Post-training failure batch plus the pre-update weight lock every rerun must reuse."""

    manifest_id: str
    entries: tuple[FailureBatchEntry, ...]
    weight_lock: PreUpdateWeightLock
    schema_version: str = FAILURE_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FAILURE_BATCH_SCHEMA_VERSION:
            raise T05ContractError(
                "SCHEMA_MISMATCH", "failure_batch.schema_version", repr(self.schema_version)
            )
        _nonempty(self.manifest_id, "manifest_id")
        entries = tuple(sorted(self.entries, key=lambda entry: entry.sort_key))
        if not entries:
            raise ValueError("failure batch must contain at least one entry")
        keys = [entry.sort_key for entry in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("failure batch has duplicate (run, checkpoint, sample, rank, module)")
        foreign = sorted(
            {entry.checkpoint_id for entry in entries} - {self.weight_lock.checkpoint_id}
        )
        if foreign:
            raise T05ContractError(
                "PRE_UPDATE_WEIGHT_DRIFT",
                "entries.checkpoint_id",
                f"entries reference checkpoints {foreign} outside the weight lock",
            )
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "weight_lock": self.weight_lock.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class FailureBatchFilter:
    """Exact-match selection on run/checkpoint/sample/rank/module; ``None`` keeps a dimension."""

    run_id: Optional[frozenset[str]] = None
    checkpoint_id: Optional[frozenset[str]] = None
    sample_id: Optional[frozenset[str]] = None
    rank: Optional[frozenset[str]] = None
    module: Optional[frozenset[str]] = None

    def __post_init__(self) -> None:
        for name in _FILTER_DIMENSIONS:
            value = getattr(self, name)
            if value is None:
                continue
            values = frozenset(str(item) for item in value)
            if not values:
                raise ValueError(f"filter {name} must be None or non-empty")
            object.__setattr__(self, name, values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Iterable[str]]) -> "FailureBatchFilter":
        unknown = sorted(set(value) - set(_FILTER_DIMENSIONS))
        if unknown:
            raise ValueError(f"unknown failure-batch filter dimensions: {unknown}")
        return cls(**{name: frozenset(items) for name, items in value.items()})

    def to_dict(self) -> dict[str, Optional[list[str]]]:
        return {
            name: (None if getattr(self, name) is None else sorted(getattr(self, name)))
            for name in _FILTER_DIMENSIONS
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class FilteredFailureBatch:
    """Selected entries bound to the manifest, filter and weight lock that produced them."""

    manifest_fingerprint: str
    filter_fingerprint: str
    weight_lock: PreUpdateWeightLock
    entries: tuple[FailureBatchEntry, ...] = field(default_factory=tuple)

    @property
    def selection_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "manifest": self.manifest_fingerprint,
                "filter": self.filter_fingerprint,
                "weight_lock": self.weight_lock.fingerprint,
                "stable_case_ids": [entry.stable_case_id for entry in self.entries],
            }
        )


def select_failure_batch(
    manifest: FailureBatchManifest, selection: FailureBatchFilter
) -> FilteredFailureBatch:
    """Deterministically select entries; an empty selection is an identity error, not a pass."""

    chosen = tuple(
        entry
        for entry in manifest.entries
        if all(
            getattr(selection, name) is None or getattr(entry, name) in getattr(selection, name)
            for name in _FILTER_DIMENSIONS
        )
    )
    if not chosen:
        raise T05ContractError(
            "IDENTITY_DRIFT",
            "failure_batch.filter",
            f"filter {selection.to_dict()} selects no entry of manifest {manifest.manifest_id}",
        )
    return FilteredFailureBatch(
        manifest_fingerprint=manifest.fingerprint,
        filter_fingerprint=selection.fingerprint,
        weight_lock=manifest.weight_lock,
        entries=chosen,
    )


__all__ = [
    "FAILURE_BATCH_SCHEMA_VERSION",
    "T05_EMITTED_STATUSES",
    "WEIGHT_LOCK_SCHEMA_VERSION",
    "FailureBatchEntry",
    "FailureBatchFilter",
    "FailureBatchManifest",
    "FilteredFailureBatch",
    "PreUpdateWeightLock",
    "ShardDigest",
    "T05ContractError",
    "canonical_sha256",
    "select_failure_batch",
    "tensor_bytes_sha256",
]
