# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Typed WS2 contract for TP-aware selected-token log-probability.

The objects in this module describe a vocab-parallel logprob invocation:

``selected_logp[t] = logits[t, target[t]] - logsumexp_vocab(logits[t, :])``

Under vocab-parallel tensor parallelism the vocabulary-wide ``logsumexp``
requires cross-rank reduction.  This module only *describes* that invocation
(shard ownership, merge semantics, mask/ignore-index metadata); it does not
shard tensors, launch collectives, or implement the ``(max, sumexp)`` merge.
Keeping description and materialization separate lets dispatch reject an
incompatible backend before any numerically different path is launched.

Context parallelism is a declared non-merge axis: CP partitions tokens, never
the vocabulary, so the logprob reduction spans TP vocab shards only.  CP rank
metadata is carried for provenance and must never widen the merge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)

# Policy keywords accepted by KernelRegistry.get_logprob_op; a backend id must
# never shadow one of these, or it becomes unselectable by id.
RESERVED_DISPATCH_POLICIES = frozenset({"auto", "production", "reference", "deterministic"})
# Backend tiers; determinism is a separate axis (DeterminismScope).
IMPLEMENTATION_KINDS = frozenset({"production", "reference"})


class LogprobContractError(ValueError):
    """Raised when logprob metadata does not describe a valid invocation."""


class LogprobRole(str, Enum):
    TRAIN = "train"
    INFER = "infer"


class LogprobDType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class LogprobMerge(str, Enum):
    """Merge primitive for per-shard ``(local_max, local_sumexp)`` partials."""

    MAX_SUMEXP = "max_sumexp"


class MergeAxis(str, Enum):
    """The only reduction axis of this contract; CP is a non-merge axis."""

    TP_VOCAB = "tp_vocab"


class ReductionOrder(str, Enum):
    GLOBAL_VOCAB_SHARD_INDEX = "global_vocab_shard_index"


class ReductionTransport(str, Enum):
    """Collectives move partial states only; they never reduce numerically."""

    ALL_GATHER = "all_gather"


class DowncastPoint(str, Enum):
    FINAL_WRITE = "final_write"


class ReductionEngine(str, Enum):
    IN_OP_REFERENCE = "in_op_reference"


class DeterminismScope(str, Enum):
    """Strength of the reduction's determinism guarantee.

    ``fixed_topology``: bitwise-reproducible for one fixed TP degree; results
    at different TP degrees are compared against the #108 tolerance table.

    ``cross_tp_bitwise``: additionally bitwise-equal across TP degrees.  This
    requires the entire reduction to follow a global tile-level structure that
    is independent of TP partitioning: a fixed tile decomposition of the
    vocabulary plus a fixed merge order and rescaling tree over those tiles,
    identical at every TP degree, so the TP degree only selects which rank
    computes which tiles and never changes the floating-point grouping.
    Fixed shard-order merging alone is not sufficient, because shard
    boundaries would still group the combines differently across degrees.
    """

    FIXED_TOPOLOGY = "fixed_topology"
    CROSS_TP_BITWISE = "cross_tp_bitwise"


class MaskMode(str, Enum):
    """How a backend consumes inactive-token information.

    The contract permits inactive targets that do not hold ``ignore_index``,
    so an ``ignore_index``-only backend cannot serve a contract with inactive
    tokens.
    """

    EXPLICIT_ACTIVE_MASK = "explicit_active_mask"
    IGNORE_INDEX = "ignore_index"


class TPPlacement(str, Enum):
    REPLICATED = "replicated"


def _enum_value(enum_type: type[_EnumT], value: Any, field: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise LogprobContractError(f"{field} must be one of: {allowed}; got {value!r}") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LogprobContractError(f"{field} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogprobContractError(f"{field} must be a non-negative integer; got {value!r}")
    return value


def _plain_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogprobContractError(f"{field} must be an integer; got {value!r}")
    return value


@dataclass(frozen=True)
class ShardingSpec:
    """Logical vocab-parallel TP ownership for one logprob invocation.

    ``vocab_shard_bounds`` lists every TP rank's half-open ``[start, end)``
    vocab range, indexed by rank; the full table is required on every rank and
    must form a contiguous ``[0, padded_vocab_size)`` partition.
    ``padded_vocab_size`` is the shard-covered (weight) vocabulary,
    ``real_vocab_size`` the tokenizer vocabulary; padding columns occupy
    ``[real_vocab_size, padded_vocab_size)``.
    """

    tp_rank: int
    tp_world_size: int
    vocab_shard_bounds: tuple[tuple[int, int], ...]
    real_vocab_size: int
    padded_vocab_size: int
    cp_rank: int = 0
    cp_world_size: int = 1

    def __post_init__(self) -> None:
        tp_world_size = _positive_int(self.tp_world_size, "tp_world_size")
        tp_rank = _non_negative_int(self.tp_rank, "tp_rank")
        if tp_rank >= tp_world_size:
            raise LogprobContractError(
                f"tp_rank={tp_rank} must be smaller than tp_world_size={tp_world_size}"
            )
        cp_world_size = _positive_int(self.cp_world_size, "cp_world_size")
        cp_rank = _non_negative_int(self.cp_rank, "cp_rank")
        if cp_rank >= cp_world_size:
            raise LogprobContractError(
                f"cp_rank={cp_rank} must be smaller than cp_world_size={cp_world_size}"
            )

        real_vocab_size = _positive_int(self.real_vocab_size, "real_vocab_size")
        padded_vocab_size = _positive_int(self.padded_vocab_size, "padded_vocab_size")
        if padded_vocab_size < real_vocab_size:
            raise LogprobContractError(
                f"padded_vocab_size={padded_vocab_size} must not be smaller than "
                f"real_vocab_size={real_vocab_size}"
            )

        try:
            bounds = tuple((pair[0], pair[1]) for pair in self.vocab_shard_bounds)
        except (TypeError, IndexError) as exc:
            raise LogprobContractError(
                "vocab_shard_bounds must be an iterable of (start, end) integer pairs"
            ) from exc
        if len(bounds) != tp_world_size:
            raise LogprobContractError(
                "vocab_shard_bounds must declare exactly one (start, end) pair per TP rank; "
                f"got {len(bounds)} pairs for tp_world_size={tp_world_size}"
            )
        expected_start = 0
        for rank, (start, end) in enumerate(bounds):
            start = _plain_int(start, f"vocab_shard_bounds[{rank}][0]")
            end = _plain_int(end, f"vocab_shard_bounds[{rank}][1]")
            if end <= start:
                raise LogprobContractError(
                    f"vocab_shard_bounds[{rank}] must satisfy end > start; got [{start}, {end})"
                )
            if start != expected_start:
                raise LogprobContractError(
                    "vocab_shard_bounds must form a contiguous [0, padded_vocab_size) "
                    f"partition in TP-rank order; rank {rank} starts at {start}, "
                    f"expected {expected_start}"
                )
            expected_start = end
        if expected_start != padded_vocab_size:
            raise LogprobContractError(
                "vocab_shard_bounds must cover padded_vocab_size exactly; "
                f"covered {expected_start}, declared {padded_vocab_size}"
            )
        object.__setattr__(self, "vocab_shard_bounds", bounds)

    @property
    def local_vocab_start(self) -> int:
        return self.vocab_shard_bounds[self.tp_rank][0]

    @property
    def local_vocab_end(self) -> int:
        return self.vocab_shard_bounds[self.tp_rank][1]

    @property
    def local_vocab_size(self) -> int:
        start, end = self.vocab_shard_bounds[self.tp_rank]
        return end - start

    def owner_rank(self, token_id: int) -> int:
        """Return the unique TP rank owning ``token_id``; error outside real vocab."""

        token_id = _plain_int(token_id, "token_id")
        if token_id < 0 or token_id >= self.real_vocab_size:
            raise LogprobContractError(
                f"token_id={token_id} is outside the real vocabulary "
                f"[0, {self.real_vocab_size}); mask it as inactive instead"
            )
        for rank, (start, end) in enumerate(self.vocab_shard_bounds):
            if start <= token_id < end:
                return rank
        raise LogprobContractError(
            f"token_id={token_id} is not covered by any declared vocab shard"
        )


@dataclass(frozen=True)
class MaskSpec:
    """Active-token mask and ignore index for one logprob invocation.

    Inactive tokens are excluded from drift aggregates and from the
    single-owner target gather; their targets may legally hold ``ignore_index``.
    """

    num_tokens: int
    active_mask: tuple[bool, ...]
    ignore_index: int = -100
    _active_token_count: int = field(init=False, repr=False, compare=False)
    _active_mask_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        num_tokens = _positive_int(self.num_tokens, "num_tokens")
        _plain_int(self.ignore_index, "ignore_index")
        try:
            active_mask = tuple(self.active_mask)
        except TypeError as exc:
            raise LogprobContractError("active_mask must be an iterable of booleans") from exc
        for index, value in enumerate(active_mask):
            if not isinstance(value, bool):
                raise LogprobContractError(f"active_mask[{index}] must be a bool; got {value!r}")
        if len(active_mask) != num_tokens:
            raise LogprobContractError(
                "active_mask must contain exactly one entry per token; "
                f"got {len(active_mask)} entries for num_tokens={num_tokens}"
            )
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "_active_token_count", sum(active_mask))
        object.__setattr__(
            self, "_active_mask_sha256", hashlib.sha256(bytes(active_mask)).hexdigest()
        )

    @property
    def active_token_count(self) -> int:
        return self._active_token_count

    @property
    def active_mask_sha256(self) -> str:
        """Compact mask identity for provenance and cross-rank agreement."""
        return self._active_mask_sha256


@dataclass(frozen=True)
class ReductionSpec:
    """Deterministic TP-vocab ``(max, sumexp)`` merge semantics.

    Every rank first masks local columns whose global id lies in
    ``[real_vocab_size, padded_vocab_size)`` to ``-inf`` (padding never
    contributes to the logsumexp), then computes ``m_l = max(local_logits)``
    and ``s_l = sum(exp(local_logits - m_l))`` in fp32.  Partials travel by
    all-gather -- collectives are transport only, never a numerical
    reduction -- and every rank merges in fixed global vocab-shard index
    order::

        M   = max_l(m_l)
        S   = sum_l(s_l * exp(m_l - M))
        LSE = M + log(S)
        selected_logp = target_logit - LSE

    The selected target logit comes from a masked single-owner gather;
    downcast happens only at the final write.  The identity partial for a
    padding-only shard, or a row whose local columns are all ``-inf`` after
    masking, is ``(m_l, s_l) = (-inf, 0)``: a partial with ``s_l = 0``
    contributes nothing to the merge regardless of its ``m_l``, and
    implementations must use this identity directly rather than evaluate
    ``exp(-inf - (-inf))``, which would poison the merge with NaN.  Averaging
    per-rank logsumexp values, or letting a collective reduce numerically, is
    never conformant at either determinism scope.
    """

    merge: LogprobMerge = LogprobMerge.MAX_SUMEXP
    merge_axis: MergeAxis = MergeAxis.TP_VOCAB
    acc_dtype: LogprobDType = LogprobDType.FP32
    order: ReductionOrder = ReductionOrder.GLOBAL_VOCAB_SHARD_INDEX
    transport: ReductionTransport = ReductionTransport.ALL_GATHER
    downcast_at: DowncastPoint = DowncastPoint.FINAL_WRITE
    engine: ReductionEngine = ReductionEngine.IN_OP_REFERENCE
    determinism_scope: DeterminismScope = DeterminismScope.CROSS_TP_BITWISE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "determinism_scope",
            _enum_value(DeterminismScope, self.determinism_scope, "determinism_scope"),
        )
        object.__setattr__(self, "merge", _enum_value(LogprobMerge, self.merge, "merge"))
        object.__setattr__(
            self, "merge_axis", _enum_value(MergeAxis, self.merge_axis, "merge_axis")
        )
        object.__setattr__(
            self, "acc_dtype", _enum_value(LogprobDType, self.acc_dtype, "acc_dtype")
        )
        object.__setattr__(self, "order", _enum_value(ReductionOrder, self.order, "order"))
        object.__setattr__(
            self, "transport", _enum_value(ReductionTransport, self.transport, "transport")
        )
        object.__setattr__(
            self, "downcast_at", _enum_value(DowncastPoint, self.downcast_at, "downcast_at")
        )
        object.__setattr__(self, "engine", _enum_value(ReductionEngine, self.engine, "engine"))
        if self.acc_dtype is not LogprobDType.FP32:
            raise LogprobContractError(
                f"TP logprob accumulation must be fp32; got {self.acc_dtype.value}"
            )


@dataclass(frozen=True)
class LogprobOutputSpec:
    """Output surface every conforming backend must produce: fp32 selected
    logprob and fp32 vocab-domain LSE, replicated across the TP group."""

    selected_logp_dtype: LogprobDType = LogprobDType.FP32
    lse_dtype: LogprobDType = LogprobDType.FP32
    tp_placement: TPPlacement = TPPlacement.REPLICATED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_logp_dtype",
            _enum_value(LogprobDType, self.selected_logp_dtype, "selected_logp_dtype"),
        )
        object.__setattr__(
            self, "lse_dtype", _enum_value(LogprobDType, self.lse_dtype, "lse_dtype")
        )
        object.__setattr__(
            self, "tp_placement", _enum_value(TPPlacement, self.tp_placement, "tp_placement")
        )
        if self.selected_logp_dtype is not LogprobDType.FP32:
            raise LogprobContractError(
                f"selected logprob output must be fp32; got {self.selected_logp_dtype.value}"
            )
        if self.lse_dtype is not LogprobDType.FP32:
            raise LogprobContractError(f"vocab LSE output must be fp32; got {self.lse_dtype.value}")


@dataclass(frozen=True)
class LogprobContract:
    """Complete semantic request consumed by contract-aware dispatch."""

    role: LogprobRole
    dtype: LogprobDType
    mask: MaskSpec
    sharding: ShardingSpec
    reduction: ReductionSpec
    output: LogprobOutputSpec = field(default_factory=LogprobOutputSpec)
    export_lse: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum_value(LogprobRole, self.role, "role"))
        object.__setattr__(self, "dtype", _enum_value(LogprobDType, self.dtype, "dtype"))
        if not isinstance(self.mask, MaskSpec):
            raise LogprobContractError("mask must be a MaskSpec")
        if not isinstance(self.sharding, ShardingSpec):
            raise LogprobContractError("sharding must be a ShardingSpec")
        if not isinstance(self.reduction, ReductionSpec):
            raise LogprobContractError("reduction must be a ReductionSpec")
        if not isinstance(self.output, LogprobOutputSpec):
            raise LogprobContractError("output must be a LogprobOutputSpec")
        if not isinstance(self.export_lse, bool) or not self.export_lse:
            raise LogprobContractError(
                "export_lse must be True for the WS2 vocab-domain LSE drift contract"
            )
        if 0 <= self.mask.ignore_index < self.sharding.real_vocab_size:
            raise LogprobContractError(
                f"ignore_index={self.mask.ignore_index} must not collide with the real "
                f"vocabulary [0, {self.sharding.real_vocab_size})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-compatible requested-contract provenance."""

        sharding = {
            "tp_rank": self.sharding.tp_rank,
            "tp_world_size": self.sharding.tp_world_size,
            "cp_rank": self.sharding.cp_rank,
            "cp_world_size": self.sharding.cp_world_size,
            "vocab_shard_bounds": [list(pair) for pair in self.sharding.vocab_shard_bounds],
            "real_vocab_size": self.sharding.real_vocab_size,
            "padded_vocab_size": self.sharding.padded_vocab_size,
            "local_vocab_start": self.sharding.local_vocab_start,
            "local_vocab_end": self.sharding.local_vocab_end,
        }
        reduction = {
            "merge": self.reduction.merge.value,
            "merge_axis": self.reduction.merge_axis.value,
            "acc_dtype": self.reduction.acc_dtype.value,
            "order": self.reduction.order.value,
            "transport": self.reduction.transport.value,
            "downcast_at": self.reduction.downcast_at.value,
            "engine": self.reduction.engine.value,
            "determinism_scope": self.reduction.determinism_scope.value,
            "cp_is_merge_axis": False,
        }
        # The digest stands in for the raw per-token mask, which would
        # dominate the provenance size.
        mask = {
            "num_tokens": self.mask.num_tokens,
            "active_token_count": self.mask.active_token_count,
            "active_mask_sha256": self.mask.active_mask_sha256,
            "ignore_index": self.mask.ignore_index,
        }
        output = {
            "selected_logp_dtype": self.output.selected_logp_dtype.value,
            "lse_dtype": self.output.lse_dtype.value,
            "tp_placement": self.output.tp_placement.value,
        }
        return {
            "semantic_operator": "selected_token_logprob",
            "role": self.role.value,
            "dtype": self.dtype.value,
            "export_lse": self.export_lse,
            "lse_domain": "vocab",
            "mask": mask,
            "sharding": sharding,
            "reduction": reduction,
            "output": output,
        }

    def cross_rank_fingerprint(self) -> str:
        """Rank-independent identity for preflight agreement across ranks.

        Excludes ``tp_rank``/``cp_rank`` (and their derived local bounds) so
        every rank of one logical invocation computes the same value.
        All-gathering this fingerprint together with the resolved backend id
        and aborting on mismatch is the documented preflight for distributed
        dispatch; ``requested_backend="auto"`` is not distributed-safe
        without it.
        """

        payload = self.to_dict()
        payload["sharding"] = {
            key: value
            for key, value in payload["sharding"].items()
            if key not in {"tp_rank", "cp_rank", "local_vocab_start", "local_vocab_end"}
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LogprobBackendCapability:
    """Capabilities a concrete backend declares to contract-aware dispatch."""

    backend_id: str
    roles: frozenset[LogprobRole]
    dtypes: frozenset[LogprobDType]
    tp_world_sizes: tuple[int, ...] | None = None
    cp_world_sizes: tuple[int, ...] | None = None
    supports_vocab_padding: bool = False
    mask_modes: frozenset[MaskMode] = frozenset()
    exports_vocab_lse: bool = False
    determinism_scopes: frozenset[DeterminismScope] = frozenset()
    implementation_kind: str = "production"

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise LogprobContractError("backend_id must be a non-empty string")
        if self.backend_id.strip().lower() in RESERVED_DISPATCH_POLICIES:
            raise LogprobContractError(
                f"backend_id={self.backend_id!r} shadows a reserved dispatch policy keyword"
            )
        object.__setattr__(self, "backend_id", self.backend_id.strip())
        try:
            roles = frozenset(_enum_value(LogprobRole, value, "roles") for value in self.roles)
            dtypes = frozenset(_enum_value(LogprobDType, value, "dtypes") for value in self.dtypes)
        except TypeError as exc:
            raise LogprobContractError("roles and dtypes must be iterables of enum values") from exc
        if not roles or not dtypes:
            raise LogprobContractError("backend roles and dtypes must not be empty")
        tp_world_sizes = self._validated_world_sizes(self.tp_world_sizes, "tp_world_sizes")
        cp_world_sizes = self._validated_world_sizes(self.cp_world_sizes, "cp_world_sizes")
        try:
            mask_modes = frozenset(
                _enum_value(MaskMode, value, "mask_modes") for value in self.mask_modes
            )
            determinism_scopes = frozenset(
                _enum_value(DeterminismScope, value, "determinism_scopes")
                for value in self.determinism_scopes
            )
        except TypeError as exc:
            raise LogprobContractError(
                "mask_modes and determinism_scopes must be iterables of enum values"
            ) from exc
        for flag_name in ("supports_vocab_padding", "exports_vocab_lse"):
            if not isinstance(getattr(self, flag_name), bool):
                raise LogprobContractError(f"{flag_name} must be a bool")
        if self.implementation_kind not in IMPLEMENTATION_KINDS:
            raise LogprobContractError(
                f"implementation_kind must be one of: {', '.join(sorted(IMPLEMENTATION_KINDS))}"
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(self, "tp_world_sizes", tp_world_sizes)
        object.__setattr__(self, "cp_world_sizes", cp_world_sizes)
        object.__setattr__(self, "mask_modes", mask_modes)
        object.__setattr__(self, "determinism_scopes", determinism_scopes)

    @staticmethod
    def _validated_world_sizes(
        values: tuple[int, ...] | None, field: str
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        try:
            sizes = tuple(values)
        except TypeError as exc:
            raise LogprobContractError(f"{field} must be an iterable of integers") from exc
        if not sizes:
            raise LogprobContractError(f"{field} must not be empty; use None for unrestricted")
        for value in sizes:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LogprobContractError(f"{field} must contain positive values; got {value!r}")
        if len(set(sizes)) != len(sizes):
            raise LogprobContractError(f"{field} must not contain duplicates")
        return sizes

    def incompatibilities(self, contract: LogprobContract) -> tuple[str, ...]:
        """Explain every reason this backend cannot materialize ``contract``."""

        reasons: list[str] = []
        if contract.role not in self.roles:
            reasons.append(f"role={contract.role.value} is unsupported")
        if contract.dtype not in self.dtypes:
            reasons.append(f"dtype={contract.dtype.value} is unsupported")
        tp_size = contract.sharding.tp_world_size
        cp_size = contract.sharding.cp_world_size
        if self.tp_world_sizes is not None and tp_size not in self.tp_world_sizes:
            reasons.append(f"TP={tp_size} is unsupported")
        if self.cp_world_sizes is not None and cp_size not in self.cp_world_sizes:
            reasons.append(f"CP={cp_size} is unsupported")
        if (
            contract.sharding.padded_vocab_size != contract.sharding.real_vocab_size
            and not self.supports_vocab_padding
        ):
            reasons.append("padded-vs-real vocab masking is unsupported")
        if (
            contract.mask.active_token_count != contract.mask.num_tokens
            and MaskMode.EXPLICIT_ACTIVE_MASK not in self.mask_modes
        ):
            # Inactive targets need not hold ignore_index (see MaskMode).
            reasons.append("explicit active-token masking is unsupported")
        if contract.export_lse and not self.exports_vocab_lse:
            reasons.append("vocab-domain LSE export is unsupported")
        if contract.reduction.determinism_scope not in self.determinism_scopes:
            reasons.append(
                f"determinism_scope={contract.reduction.determinism_scope.value} is unsupported"
            )
        return tuple(reasons)

    def supports(self, contract: LogprobContract) -> bool:
        return not self.incompatibilities(contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "roles": sorted(role.value for role in self.roles),
            "dtypes": sorted(dtype.value for dtype in self.dtypes),
            "tp_world_sizes": list(self.tp_world_sizes) if self.tp_world_sizes else None,
            "cp_world_sizes": list(self.cp_world_sizes) if self.cp_world_sizes else None,
            "supports_vocab_padding": self.supports_vocab_padding,
            "mask_modes": sorted(mode.value for mode in self.mask_modes),
            "exports_vocab_lse": self.exports_vocab_lse,
            "determinism_scopes": sorted(scope.value for scope in self.determinism_scopes),
            "implementation_kind": self.implementation_kind,
        }


@dataclass(frozen=True)
class LogprobDispatchResult:
    """A concrete backend plus the actual provenance bound to the request."""

    op: Any
    capability: LogprobBackendCapability
    provenance: dict[str, Any]


__all__ = [
    "IMPLEMENTATION_KINDS",
    "RESERVED_DISPATCH_POLICIES",
    "DeterminismScope",
    "DowncastPoint",
    "LogprobBackendCapability",
    "LogprobContract",
    "LogprobContractError",
    "LogprobDType",
    "LogprobDispatchResult",
    "LogprobMerge",
    "LogprobOutputSpec",
    "LogprobRole",
    "MaskMode",
    "MaskSpec",
    "MergeAxis",
    "ReductionEngine",
    "ReductionOrder",
    "ReductionSpec",
    "ReductionTransport",
    "ShardingSpec",
    "TPPlacement",
]
