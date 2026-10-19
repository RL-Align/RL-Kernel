# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Typed WS2 contract for deterministic GRPO loss on the TP-aware logprob path.

The GRPO objective consumes selected-token log-probabilities and reduces them
to a scalar::

    ratio_t   = exp(logp_policy_t - old_logp_t)
    surrogate = -min(ratio_t * adv_t, clip(ratio_t) * adv_t)
    loss      = normalize(sum_t surrogate_t) + beta * normalize(sum_t kl_t)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from rl_engine.contracts.operators.logprob import (
    IMPLEMENTATION_KINDS,
    RESERVED_DISPATCH_POLICIES,
    DeterminismScope,
    DowncastPoint,
    LogprobContract,
    LogprobDType,
)

_EnumT = TypeVar("_EnumT", bound=Enum)


class LossContractError(ValueError):
    """Raised when loss metadata does not describe a valid GRPO invocation."""


class TokenNormalizer(str, Enum):
    """Denominator applied to the summed per-token loss terms.

    ``global_active_tokens``: divide by the number of active tokens in the
    *global* batch, gathered across every DP rank.  Long sequences therefore
    contribute proportionally more.  This matches the existing single-GPU
    ``NativeGRPOLossOp`` masked mean at DP=1.

    ``per_sequence_then_mean``: divide each sequence's sum by that sequence's
    own active-token count, then average over sequences that hold at least one
    active token.  Sequences are weighted equally regardless of length.

    ``fixed_constant``: divide by a declared constant, independent of the mask.
    Requires ``LossReductionSpec.fixed_normalizer_constant``.

    The three differ by more than a scale factor once sequence lengths vary, so
    the choice is part of the numerical identity and travels in the fingerprint.
    """

    GLOBAL_ACTIVE_TOKENS = "global_active_tokens"
    PER_SEQUENCE_THEN_MEAN = "per_sequence_then_mean"
    FIXED_CONSTANT = "fixed_constant"


class SummationOrder(str, Enum):
    """Fixed combine order for per-token partials.

    ``sequence_major_fixed``: within one sequence, tokens combine over the full
    ``padded_seq_len`` extent on the single rank that owns the sequence;
    sequences then combine in ascending global sequence index.  Both extents are
    contract-fixed, so the floating-point grouping is identical at every DP
    degree.  Only *which rank* computes a sequence changes.
    """

    SEQUENCE_MAJOR_FIXED = "sequence_major_fixed"


class LossTransport(str, Enum):
    """Collectives move partial sums only; they never reduce numerically.

    ``all_reduce`` is excluded on purpose: NCCL's reduction order depends on
    world size and topology, so it would silently regroup the per-sequence
    combines and break the cross-DP guarantee ``SummationOrder`` provides.
    """

    ALL_GATHER = "all_gather"


class KLEstimator(str, Enum):
    """Per-token reference-KL estimator.

    ``k3_unbiased``: ``exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1``,
    the non-negative low-variance estimator used by the existing ratio/KL op.

    ``k1_log_ratio``: ``logp_policy - logp_ref``, the plain log-ratio.
    """

    K3_UNBIASED = "k3_unbiased"
    K1_LOG_RATIO = "k1_log_ratio"


class ClipMode(str, Enum):
    MIN_OF_UNCLIPPED_AND_CLIPPED = "min_of_unclipped_and_clipped"


class AdvantageNormalizer(str, Enum):
    """Group-relative reward normalization.

    ``mean_std_population``: ``(r - mean) / std`` with the population (biased)
    standard deviation, the original GRPO form.

    ``mean_only``: ``r - mean``, the Dr.GRPO form that drops the std divisor to
    avoid its length/difficulty bias.
    """

    MEAN_STD_POPULATION = "mean_std_population"
    MEAN_ONLY = "mean_only"


class VarianceFormula(str, Enum):
    """Only the two-pass form is conformant.

    ``E[x^2] - E[x]^2`` cancels catastrophically once rewards share a large
    offset, and its error depends on group size, so it cannot support a bitwise
    claim.  The two-pass form subtracts the mean before squaring.
    """

    TWO_PASS = "two_pass"


class GroupReplication(str, Enum):
    """How advantage groups are evaluated when they span DP ranks.

    ``replicated_all_gather``: per-sequence rewards are all-gathered and *every*
    rank normalizes *every* group identically, then keeps its own slice.
    Rewards are one scalar per sequence, so replicating the whole computation is
    cheaper than making a partial-statistic merge bitwise-reproducible.
    """

    REPLICATED_ALL_GATHER = "replicated_all_gather"


class LossPlacement(str, Enum):
    REPLICATED = "replicated"


def _enum_value(enum_type: type[_EnumT], value: Any, field: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise LossContractError(f"{field} must be one of: {allowed}; got {value!r}") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LossContractError(f"{field} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LossContractError(f"{field} must be a non-negative integer; got {value!r}")
    return value


def _non_negative_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LossContractError(f"{field} must be a real number; got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise LossContractError(f"{field} must be finite and non-negative; got {value!r}")
    return value


@dataclass(frozen=True)
class ClipSpec:
    """Asymmetric PPO-style ratio clipping bounds.

    Separate low/high epsilons cover the "clip-higher" variants; passing the
    same value twice recovers the symmetric ``[1-eps, 1+eps]`` form.
    """

    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    mode: ClipMode = ClipMode.MIN_OF_UNCLIPPED_AND_CLIPPED

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _enum_value(ClipMode, self.mode, "clip.mode"))
        low = _non_negative_float(self.clip_eps_low, "clip_eps_low")
        high = _non_negative_float(self.clip_eps_high, "clip_eps_high")
        if low >= 1.0:
            raise LossContractError(
                f"clip_eps_low={low} must be smaller than 1.0; the lower clip bound "
                "1 - clip_eps_low must stay positive"
            )
        object.__setattr__(self, "clip_eps_low", low)
        object.__setattr__(self, "clip_eps_high", high)

    @property
    def lower_bound(self) -> float:
        return 1.0 - self.clip_eps_low

    @property
    def upper_bound(self) -> float:
        return 1.0 + self.clip_eps_high


@dataclass(frozen=True)
class AdvantageSpec:
    """Group-relative advantage normalization semantics."""

    normalizer: AdvantageNormalizer = AdvantageNormalizer.MEAN_STD_POPULATION
    variance: VarianceFormula = VarianceFormula.TWO_PASS
    std_eps: float = 1e-6
    replication: GroupReplication = GroupReplication.REPLICATED_ALL_GATHER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalizer",
            _enum_value(AdvantageNormalizer, self.normalizer, "advantage.normalizer"),
        )
        object.__setattr__(
            self, "variance", _enum_value(VarianceFormula, self.variance, "advantage.variance")
        )
        object.__setattr__(
            self,
            "replication",
            _enum_value(GroupReplication, self.replication, "advantage.replication"),
        )
        std_eps = _non_negative_float(self.std_eps, "advantage.std_eps")
        if std_eps <= 0.0:
            raise LossContractError(
                f"advantage.std_eps={std_eps} must be strictly positive; it is the floor "
                "that keeps a zero-variance group from dividing by zero"
            )
        object.__setattr__(self, "std_eps", std_eps)


@dataclass(frozen=True)
class ObjectiveSpec:
    """The GRPO objective itself: clipping, reference KL, advantage shaping."""

    clip: ClipSpec = field(default_factory=ClipSpec)
    advantage: AdvantageSpec = field(default_factory=AdvantageSpec)
    kl_estimator: KLEstimator = KLEstimator.K3_UNBIASED
    beta: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.clip, ClipSpec):
            raise LossContractError("objective.clip must be a ClipSpec")
        if not isinstance(self.advantage, AdvantageSpec):
            raise LossContractError("objective.advantage must be an AdvantageSpec")
        object.__setattr__(
            self,
            "kl_estimator",
            _enum_value(KLEstimator, self.kl_estimator, "objective.kl_estimator"),
        )
        object.__setattr__(self, "beta", _non_negative_float(self.beta, "objective.beta"))

    @property
    def uses_reference_model(self) -> bool:
        """Whether reference logits are required at all.

        ``beta == 0`` drops the KL term from the loss, so a backend may skip the
        reference forward entirely.  The KL is still *reported*, so a caller
        that wants the diagnostic must supply reference logits regardless.
        """

        return self.beta > 0.0


@dataclass(frozen=True)
class LossReductionSpec:
    """Deterministic token/sequence summation and normalizer semantics.

    Per-token loss terms are accumulated in fp32, combined in the order given by
    ``summation_order``, moved between ranks by ``transport`` (never reduced by
    it), and divided by the denominator selected by ``token_normalizer``.  The
    scalar is downcast, if at all, only at ``downcast_at``.

    ``determinism_scope`` reuses the logprob scale.  ``cross_tp_bitwise`` here
    means the scalar loss and its gradient are bitwise-identical across TP and
    DP degrees, given a fixed vocab tile count.
    """

    token_normalizer: TokenNormalizer = TokenNormalizer.GLOBAL_ACTIVE_TOKENS
    summation_order: SummationOrder = SummationOrder.SEQUENCE_MAJOR_FIXED
    acc_dtype: LogprobDType = LogprobDType.FP32
    transport: LossTransport = LossTransport.ALL_GATHER
    downcast_at: DowncastPoint = DowncastPoint.FINAL_WRITE
    determinism_scope: DeterminismScope = DeterminismScope.CROSS_TP_BITWISE
    fixed_normalizer_constant: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_normalizer",
            _enum_value(TokenNormalizer, self.token_normalizer, "token_normalizer"),
        )
        object.__setattr__(
            self,
            "summation_order",
            _enum_value(SummationOrder, self.summation_order, "summation_order"),
        )
        object.__setattr__(
            self, "acc_dtype", _enum_value(LogprobDType, self.acc_dtype, "acc_dtype")
        )
        object.__setattr__(
            self, "transport", _enum_value(LossTransport, self.transport, "transport")
        )
        object.__setattr__(
            self, "downcast_at", _enum_value(DowncastPoint, self.downcast_at, "downcast_at")
        )
        object.__setattr__(
            self,
            "determinism_scope",
            _enum_value(DeterminismScope, self.determinism_scope, "determinism_scope"),
        )
        if self.acc_dtype is not LogprobDType.FP32:
            raise LossContractError(f"loss accumulation must be fp32; got {self.acc_dtype.value}")

        needs_constant = self.token_normalizer is TokenNormalizer.FIXED_CONSTANT
        if needs_constant:
            object.__setattr__(
                self,
                "fixed_normalizer_constant",
                _positive_int(self.fixed_normalizer_constant, "fixed_normalizer_constant"),
            )
        elif self.fixed_normalizer_constant is not None:
            raise LossContractError(
                "fixed_normalizer_constant is only meaningful for "
                f"token_normalizer={TokenNormalizer.FIXED_CONSTANT.value}; got "
                f"{self.token_normalizer.value}"
            )


@dataclass(frozen=True)
class LossOutputSpec:
    """Output surface: fp32 scalars replicated across every DP and TP rank."""

    loss_dtype: LogprobDType = LogprobDType.FP32
    placement: LossPlacement = LossPlacement.REPLICATED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "loss_dtype", _enum_value(LogprobDType, self.loss_dtype, "loss_dtype")
        )
        object.__setattr__(
            self, "placement", _enum_value(LossPlacement, self.placement, "placement")
        )
        if self.loss_dtype is not LogprobDType.FP32:
            raise LossContractError(f"loss output must be fp32; got {self.loss_dtype.value}")


@dataclass(frozen=True)
class LossShardingSpec:
    """Which sequences of the global batch this DP rank owns.

    The global batch is ``num_sequences`` sequences of ``padded_seq_len`` token
    slots each.  ``sequence_shard_bounds`` lists every DP rank's half-open
    ``[start, end)`` sequence range, indexed by rank; the full table is required
    on every rank and must form a contiguous ``[0, num_sequences)`` partition,
    exactly as ``ShardingSpec.vocab_shard_bounds`` does for the vocabulary.
    """

    dp_rank: int
    dp_world_size: int
    num_sequences: int
    padded_seq_len: int
    sequence_shard_bounds: tuple[tuple[int, int], ...]
    group_boundaries: tuple[int, ...]
    cp_rank: int = 0
    cp_world_size: int = 1

    def __post_init__(self) -> None:
        dp_world_size = _positive_int(self.dp_world_size, "dp_world_size")
        dp_rank = _non_negative_int(self.dp_rank, "dp_rank")
        if dp_rank >= dp_world_size:
            raise LossContractError(
                f"dp_rank={dp_rank} must be smaller than dp_world_size={dp_world_size}"
            )
        cp_world_size = _positive_int(self.cp_world_size, "cp_world_size")
        _non_negative_int(self.cp_rank, "cp_rank")
        if cp_world_size != 1:
            raise LossContractError(
                f"cp_world_size={cp_world_size} is unsupported: context parallelism splits a "
                "sequence's tokens across ranks, so the loss reduction would span an axis this "
                "contract does not model. Reduce across CP outside this operator."
            )
        if self.cp_rank != 0:
            raise LossContractError(f"cp_rank must be 0 when cp_world_size=1; got {self.cp_rank}")

        _positive_int(self.num_sequences, "num_sequences")
        _positive_int(self.padded_seq_len, "padded_seq_len")
        object.__setattr__(
            self,
            "sequence_shard_bounds",
            self._validated_bounds(self.sequence_shard_bounds, dp_world_size, self.num_sequences),
        )
        object.__setattr__(
            self,
            "group_boundaries",
            self._validated_groups(self.group_boundaries, self.num_sequences),
        )

    @staticmethod
    def _validated_bounds(
        raw: Any, dp_world_size: int, num_sequences: int
    ) -> tuple[tuple[int, int], ...]:
        try:
            bounds = tuple((pair[0], pair[1]) for pair in raw)
        except (TypeError, IndexError, KeyError) as exc:
            raise LossContractError(
                "sequence_shard_bounds must be an iterable of (start, end) integer pairs"
            ) from exc
        if len(bounds) != dp_world_size:
            raise LossContractError(
                "sequence_shard_bounds must declare exactly one (start, end) pair per DP rank; "
                f"got {len(bounds)} pairs for dp_world_size={dp_world_size}"
            )
        expected_start = 0
        for rank, (start, end) in enumerate(bounds):
            for name, value in (
                (f"sequence_shard_bounds[{rank}][0]", start),
                (f"sequence_shard_bounds[{rank}][1]", end),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise LossContractError(f"{name} must be an integer; got {value!r}")
            if end <= start:
                raise LossContractError(
                    f"sequence_shard_bounds[{rank}] must satisfy end > start; got [{start}, {end})"
                )
            if start != expected_start:
                raise LossContractError(
                    "sequence_shard_bounds must form a contiguous [0, num_sequences) partition "
                    f"in DP-rank order; rank {rank} starts at {start}, expected {expected_start}"
                )
            expected_start = end
        if expected_start != num_sequences:
            raise LossContractError(
                "sequence_shard_bounds must cover num_sequences exactly; covered "
                f"{expected_start}, declared {num_sequences}"
            )
        return bounds

    @staticmethod
    def _validated_groups(raw: Any, num_sequences: int) -> tuple[int, ...]:
        try:
            offsets = tuple(raw)
        except TypeError as exc:
            raise LossContractError("group_boundaries must be an iterable of integers") from exc
        if len(offsets) < 2:
            raise LossContractError(
                "group_boundaries must hold num_groups + 1 offsets, so at least 2 entries"
            )
        for index, value in enumerate(offsets):
            if isinstance(value, bool) or not isinstance(value, int):
                raise LossContractError(f"group_boundaries[{index}] must be an integer")
        if offsets[0] != 0 or offsets[-1] != num_sequences:
            raise LossContractError(
                "group_boundaries must start at 0 and end at "
                f"num_sequences={num_sequences}; got [{offsets[0]}, ..., {offsets[-1]}]"
            )
        for index in range(1, len(offsets)):
            if offsets[index] <= offsets[index - 1]:
                raise LossContractError(
                    "group_boundaries must be strictly increasing; "
                    f"offset {index} is {offsets[index]} after {offsets[index - 1]}"
                )
        return offsets

    @property
    def num_groups(self) -> int:
        return len(self.group_boundaries) - 1

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(
            self.group_boundaries[i + 1] - self.group_boundaries[i] for i in range(self.num_groups)
        )

    @property
    def local_sequence_start(self) -> int:
        return self.sequence_shard_bounds[self.dp_rank][0]

    @property
    def local_sequence_end(self) -> int:
        return self.sequence_shard_bounds[self.dp_rank][1]

    @property
    def local_num_sequences(self) -> int:
        start, end = self.sequence_shard_bounds[self.dp_rank]
        return end - start

    @property
    def local_num_token_slots(self) -> int:
        """Token slots this rank holds -- the row count of its logprob call."""

        return self.local_num_sequences * self.padded_seq_len


@dataclass(frozen=True)
class GRPOLossContract:
    """Complete semantic request for one deterministic GRPO loss invocation."""

    logprob: LogprobContract
    sharding: LossShardingSpec
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    reduction: LossReductionSpec = field(default_factory=LossReductionSpec)
    output: LossOutputSpec = field(default_factory=LossOutputSpec)

    def __post_init__(self) -> None:
        if not isinstance(self.logprob, LogprobContract):
            raise LossContractError("logprob must be a LogprobContract")
        if not isinstance(self.sharding, LossShardingSpec):
            raise LossContractError("sharding must be a LossShardingSpec")
        if not isinstance(self.objective, ObjectiveSpec):
            raise LossContractError("objective must be an ObjectiveSpec")
        if not isinstance(self.reduction, LossReductionSpec):
            raise LossContractError("reduction must be a LossReductionSpec")
        if not isinstance(self.output, LossOutputSpec):
            raise LossContractError("output must be a LossOutputSpec")

        # The nested logprob contract describes this rank's own rows, so its
        # token count must match the cells this rank owns.  Catching the
        # mismatch here turns a silent shape error deep inside the reduction
        # into a contract failure at construction.
        expected_tokens = self.sharding.local_num_token_slots
        if self.logprob.mask.num_tokens != expected_tokens:
            raise LossContractError(
                f"logprob.mask.num_tokens={self.logprob.mask.num_tokens} must equal the "
                f"{expected_tokens} token slots this rank owns "
                f"({self.sharding.local_num_sequences} sequences x "
                f"{self.sharding.padded_seq_len} slots per sequence)"
            )
        if self.logprob.reduction.determinism_scope is not self.reduction.determinism_scope:
            raise LossContractError(
                "the loss cannot claim a stronger or weaker determinism scope than the "
                f"logprob path it consumes; loss={self.reduction.determinism_scope.value}, "
                f"logprob={self.logprob.reduction.determinism_scope.value}"
            )
        if self.logprob.sharding.cp_world_size != self.sharding.cp_world_size:
            raise LossContractError(
                f"cp_world_size disagrees between the logprob contract "
                f"({self.logprob.sharding.cp_world_size}) and the loss sharding "
                f"({self.sharding.cp_world_size})"
            )
        if self.objective.advantage.normalizer is AdvantageNormalizer.MEAN_STD_POPULATION:
            # A singleton group has zero population variance, so its advantage
            # would collapse to 0 and the sequence would contribute nothing.
            # Reject it rather than silently training on a dead group.
            small = [index for index, size in enumerate(self.sharding.group_sizes) if size < 2]
            if small:
                raise LossContractError(
                    f"advantage normalizer {AdvantageNormalizer.MEAN_STD_POPULATION.value} "
                    f"needs at least 2 sequences per group; groups {small} are smaller"
                )

    @property
    def global_token_slots(self) -> int:
        return self.sharding.num_sequences * self.sharding.padded_seq_len

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-compatible requested-contract provenance."""

        sharding = {
            "dp_rank": self.sharding.dp_rank,
            "dp_world_size": self.sharding.dp_world_size,
            "cp_rank": self.sharding.cp_rank,
            "cp_world_size": self.sharding.cp_world_size,
            "num_sequences": self.sharding.num_sequences,
            "padded_seq_len": self.sharding.padded_seq_len,
            "sequence_shard_bounds": [list(pair) for pair in self.sharding.sequence_shard_bounds],
            "group_boundaries": list(self.sharding.group_boundaries),
            "local_sequence_start": self.sharding.local_sequence_start,
            "local_sequence_end": self.sharding.local_sequence_end,
        }
        objective = {
            "clip_eps_low": self.objective.clip.clip_eps_low,
            "clip_eps_high": self.objective.clip.clip_eps_high,
            "clip_mode": self.objective.clip.mode.value,
            "kl_estimator": self.objective.kl_estimator.value,
            "beta": self.objective.beta,
            "advantage_normalizer": self.objective.advantage.normalizer.value,
            "advantage_variance": self.objective.advantage.variance.value,
            "advantage_std_eps": self.objective.advantage.std_eps,
            "advantage_replication": self.objective.advantage.replication.value,
        }
        reduction = {
            "token_normalizer": self.reduction.token_normalizer.value,
            "summation_order": self.reduction.summation_order.value,
            "acc_dtype": self.reduction.acc_dtype.value,
            "transport": self.reduction.transport.value,
            "downcast_at": self.reduction.downcast_at.value,
            "determinism_scope": self.reduction.determinism_scope.value,
            "fixed_normalizer_constant": self.reduction.fixed_normalizer_constant,
            "cp_is_merge_axis": False,
            "dp_is_merge_axis": True,
        }
        return {
            "semantic_operator": "grpo_loss",
            "logprob": self.logprob.to_dict(),
            "sharding": sharding,
            "objective": objective,
            "reduction": reduction,
            "output": {
                "loss_dtype": self.output.loss_dtype.value,
                "placement": self.output.placement.value,
            },
        }

    def cross_rank_fingerprint(self) -> str:
        """Rank-independent identity for preflight agreement across all ranks.

        Drops every rank-local field so all ``dp_world_size x tp_world_size``
        ranks of one logical invocation agree.  That includes the nested
        logprob contract's mask: each DP rank holds a different slice of
        sequences, so its ``num_tokens`` and mask digest legitimately differ
        even though the invocation is the same one.  The global token geometry
        is still pinned, by ``num_sequences``/``padded_seq_len`` and by the full
        ``sequence_shard_bounds`` table, which every rank declares identically.
        """

        payload = self.to_dict()
        logprob = payload["logprob"]
        logprob.pop("mask", None)
        logprob["sharding"] = {
            key: value
            for key, value in logprob["sharding"].items()
            if key not in {"tp_rank", "cp_rank", "local_vocab_start", "local_vocab_end"}
        }
        payload["sharding"] = {
            key: value
            for key, value in payload["sharding"].items()
            if key not in {"dp_rank", "cp_rank", "local_sequence_start", "local_sequence_end"}
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LossBackendCapability:
    """Capabilities a concrete loss backend declares to contract-aware dispatch."""

    backend_id: str
    token_normalizers: frozenset[TokenNormalizer]
    kl_estimators: frozenset[KLEstimator]
    advantage_normalizers: frozenset[AdvantageNormalizer]
    determinism_scopes: frozenset[DeterminismScope]
    dp_world_sizes: tuple[int, ...] | None = None
    supports_variable_group_sizes: bool = False
    supports_asymmetric_clip: bool = False
    implementation_kind: str = "production"

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise LossContractError("backend_id must be a non-empty string")
        if self.backend_id.strip().lower() in RESERVED_DISPATCH_POLICIES:
            raise LossContractError(
                f"backend_id={self.backend_id!r} shadows a reserved dispatch policy keyword"
            )
        object.__setattr__(self, "backend_id", self.backend_id.strip())
        for name, enum_type in (
            ("token_normalizers", TokenNormalizer),
            ("kl_estimators", KLEstimator),
            ("advantage_normalizers", AdvantageNormalizer),
            ("determinism_scopes", DeterminismScope),
        ):
            try:
                values = frozenset(
                    _enum_value(enum_type, value, name) for value in getattr(self, name)
                )
            except TypeError as exc:
                raise LossContractError(f"{name} must be an iterable of enum values") from exc
            if not values:
                raise LossContractError(f"{name} must not be empty")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "dp_world_sizes",
            self._validated_world_sizes(self.dp_world_sizes, "dp_world_sizes"),
        )
        for flag_name in ("supports_variable_group_sizes", "supports_asymmetric_clip"):
            if not isinstance(getattr(self, flag_name), bool):
                raise LossContractError(f"{flag_name} must be a bool")
        if self.implementation_kind not in IMPLEMENTATION_KINDS:
            raise LossContractError(
                f"implementation_kind must be one of: {', '.join(sorted(IMPLEMENTATION_KINDS))}"
            )

    @staticmethod
    def _validated_world_sizes(
        values: tuple[int, ...] | None, field: str
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        try:
            sizes = tuple(values)
        except TypeError as exc:
            raise LossContractError(f"{field} must be an iterable of integers") from exc
        if not sizes:
            raise LossContractError(f"{field} must not be empty; use None for unrestricted")
        for value in sizes:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LossContractError(f"{field} must contain positive values; got {value!r}")
        if len(set(sizes)) != len(sizes):
            raise LossContractError(f"{field} must not contain duplicates")
        return sizes

    def incompatibilities(self, contract: GRPOLossContract) -> tuple[str, ...]:
        """Explain every reason this backend cannot materialize ``contract``."""

        reasons: list[str] = []
        reduction = contract.reduction
        objective = contract.objective
        sharding = contract.sharding
        if reduction.token_normalizer not in self.token_normalizers:
            reasons.append(f"token_normalizer={reduction.token_normalizer.value} is unsupported")
        if objective.kl_estimator not in self.kl_estimators:
            reasons.append(f"kl_estimator={objective.kl_estimator.value} is unsupported")
        if objective.advantage.normalizer not in self.advantage_normalizers:
            reasons.append(
                f"advantage normalizer={objective.advantage.normalizer.value} is unsupported"
            )
        if reduction.determinism_scope not in self.determinism_scopes:
            reasons.append(f"determinism_scope={reduction.determinism_scope.value} is unsupported")
        if self.dp_world_sizes is not None and sharding.dp_world_size not in self.dp_world_sizes:
            reasons.append(f"DP={sharding.dp_world_size} is unsupported")
        if len(set(sharding.group_sizes)) > 1 and not self.supports_variable_group_sizes:
            reasons.append("variable advantage group sizes are unsupported")
        if (
            objective.clip.clip_eps_low != objective.clip.clip_eps_high
            and not self.supports_asymmetric_clip
        ):
            reasons.append("asymmetric ratio clipping is unsupported")
        return tuple(reasons)

    def supports(self, contract: GRPOLossContract) -> bool:
        return not self.incompatibilities(contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "token_normalizers": sorted(item.value for item in self.token_normalizers),
            "kl_estimators": sorted(item.value for item in self.kl_estimators),
            "advantage_normalizers": sorted(item.value for item in self.advantage_normalizers),
            "determinism_scopes": sorted(item.value for item in self.determinism_scopes),
            "dp_world_sizes": list(self.dp_world_sizes) if self.dp_world_sizes else None,
            "supports_variable_group_sizes": self.supports_variable_group_sizes,
            "supports_asymmetric_clip": self.supports_asymmetric_clip,
            "implementation_kind": self.implementation_kind,
        }


@dataclass(frozen=True)
class LossDispatchResult:
    """A concrete backend plus the actual provenance bound to the request."""

    op: Any
    capability: LossBackendCapability
    provenance: dict[str, Any]


__all__ = [
    "AdvantageNormalizer",
    "AdvantageSpec",
    "ClipMode",
    "ClipSpec",
    "GRPOLossContract",
    "GroupReplication",
    "KLEstimator",
    "LossBackendCapability",
    "LossContractError",
    "LossDispatchResult",
    "LossOutputSpec",
    "LossPlacement",
    "LossReductionSpec",
    "LossShardingSpec",
    "LossTransport",
    "ObjectiveSpec",
    "SummationOrder",
    "TokenNormalizer",
    "VarianceFormula",
]
