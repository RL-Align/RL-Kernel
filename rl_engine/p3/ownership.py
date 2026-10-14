"""Token ownership under combined parallelism (T08).

`mocks.py` records *what* each parallel dimension does to token ownership:

    tp: replica            sp: partition_by_sequence    cp: partition
    dp: partition          pp: replica_by_layer         ep: not_a_token_dimension

That table is the contract. This module turns it into behaviour, because the
six modes only become checkable once they are composed: a rank under
``tp=2, cp=2, dp=2, pp=2, ep=2`` is simultaneously a sequence-replica, a
sequence-partition, a batch-partition, a layer-replica and token-irrelevant,
and no single rule covers it.

The unit of ownership is a **routing decision** ``(absolute_layer,
global_token_id)``, not a token. The same token routed at two layers is two
decisions and they live on different pipeline stages, so a token-keyed model
cannot express ``pp: replica_by_layer`` at all.

Everything is expressed through one function: ``carriers`` — the set of global
ranks that must hold a given decision. Each dimension either shrinks that set
(partitioning) or leaves it alone (replication), so composition is an
intersection and the six modes need no special cases downstream:

- a rank holding a decision it does not carry is ``AMBIGUOUS_GLOBAL_TOKEN_MAPPING``
- a carrier missing a decision it owes is ``INCOMPLETE_ARTIFACT``
- two carriers of one decision disagreeing is ``ROUTE_SEMANTIC_FINGERPRINT_MISMATCH``

Two things are declared rather than inferred, because guessing either one
produces a validator that is stable in testing and wrong in production:
``rank_order`` (how a global rank decomposes into coordinates) and the
sequence split order (CP before SP within a TP group).
"""

from dataclasses import dataclass, field

from .contract import P3Verdict
from .mocks import OWNERSHIP
from .oracle import require

OWNERSHIP_SCHEMA = "p3-ownership.v1"

#: Dimensions that subdivide the decision space, and what they cut.
PARTITION_BY_BATCH = ("dp",)
PARTITION_BY_SEQUENCE = ("cp", "sp")
PARTITION_BY_LAYER = ("pp",)
#: Dimensions that replicate it. ``ep`` is listed separately from ``tp``
#: because an EP-induced token difference is a different defect, worth its own
#: diagnostic, even though the carrier arithmetic is the same.
REPLICATE = ("tp", "ep")

#: Default decomposition of a global rank, fastest-varying dimension first.
#: Declared, never inferred: a validator that guesses this agrees with the
#: truth on symmetric configurations and silently disagrees on the rest.
DEFAULT_RANK_ORDER = ("tp", "cp", "ep", "dp", "pp")


def partition_slices(count, parts):
    """Contiguous ranges over ``count`` items split across ``parts`` holders.

    The first ``count % parts`` holders take one extra item. This is the same
    rule the placement map uses for ``block``; keeping one convention means a
    reader never has to ask which split a given remainder followed.
    """
    require(parts > 0, P3Verdict.SCHEMA_MISMATCH, f"parts must be positive, got {parts}")
    base, extra = divmod(count, parts)
    slices, start = [], 0
    for index in range(parts):
        width = base + (1 if index < extra else 0)
        slices.append(range(start, start + width))
        start += width
    return slices


@dataclass(frozen=True)
class ParallelConfig:
    """One parallel configuration and how its global ranks decompose.

    ``sp`` is a flag rather than a size: sequence parallelism subdivides along
    the existing TP group, so it has no axis of its own. With ``sp`` set, TP
    stops being a pure replica dimension and starts partitioning the sequence
    (contract: ``tp`` is ``replica`` or ``partition_by_sequence``).
    """

    tp: int = 1
    sp: bool = False
    cp: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1
    rank_order: tuple = DEFAULT_RANK_ORDER
    schema_version: str = OWNERSHIP_SCHEMA

    def __post_init__(self):
        for name in ("tp", "cp", "dp", "pp", "ep"):
            size = getattr(self, name)
            require(
                isinstance(size, int) and size > 0,
                P3Verdict.SCHEMA_MISMATCH,
                f"{name} must be a positive int, got {size!r}",
            )
        require(
            tuple(sorted(self.rank_order)) == tuple(sorted(DEFAULT_RANK_ORDER)),
            P3Verdict.SCHEMA_MISMATCH,
            f"rank_order must permute {DEFAULT_RANK_ORDER}, got {self.rank_order}",
        )

    @property
    def world_size(self):
        return self.tp * self.cp * self.ep * self.dp * self.pp

    def size_of(self, dimension):
        """Axis length; ``sp`` has none of its own and reports the TP size."""
        return self.tp if dimension == "sp" else getattr(self, dimension)

    def ownership_of(self, dimension):
        """The contract's ownership mode for one dimension.

        ``tp`` is the only dimension whose mode depends on configuration, and
        it is read through the shared ``OWNERSHIP`` table rather than restated
        here so that the table stays the single source.
        """
        if dimension == "tp":
            return OWNERSHIP["sp"] if self.sp else OWNERSHIP["tp"]
        return OWNERSHIP[dimension]

    def coordinates(self, global_rank):
        """Decompose a global rank into per-dimension coordinates."""
        require(
            0 <= global_rank < self.world_size,
            P3Verdict.MISSING_RANK,
            f"rank {global_rank} outside [0, {self.world_size})",
        )
        coords, remainder = {}, global_rank
        for dimension in self.rank_order:
            size = getattr(self, dimension)
            coords[dimension] = remainder % size
            remainder //= size
        return coords

    def global_rank(self, **coords):
        """Compose per-dimension coordinates back into a global rank."""
        rank, stride = 0, 1
        for dimension in self.rank_order:
            size = getattr(self, dimension)
            value = int(coords.get(dimension, 0))
            require(
                0 <= value < size,
                P3Verdict.MISSING_RANK,
                f"{dimension} coordinate {value} outside [0, {size})",
            )
            rank += value * stride
            stride *= size
        return rank

    def describe(self):
        """Human-readable configuration label used in reports and groups."""
        parts = [f"tp{self.tp}" + ("sp" if self.sp else "")]
        parts += [f"{d}{getattr(self, d)}" for d in ("cp", "dp", "pp", "ep")]
        return "-".join(parts)


@dataclass(frozen=True)
class TokenLayout:
    """The global decision space a configuration is sliced out of.

    ``sequences`` holds one tuple of ``global_token_id`` per batch item, in
    order; ``layers`` holds the model's absolute layer ids. Together they
    enumerate every ``(layer, token)`` decision the whole job must produce.
    """

    sequences: tuple
    layers: tuple

    def __post_init__(self):
        require(bool(self.sequences), P3Verdict.SCHEMA_MISMATCH, "no sequences in layout")
        require(bool(self.layers), P3Verdict.SCHEMA_MISMATCH, "no layers in layout")
        seen = [token for sequence in self.sequences for token in sequence]
        require(
            len(set(seen)) == len(seen),
            P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
            "global_token_id repeated across the layout; it must be unique job-wide",
        )

    @classmethod
    def flat(cls, tokens, layers):
        """A single-sequence layout; convenient for single-layer fixtures."""
        return cls(sequences=(tuple(tokens),), layers=tuple(layers))

    def decisions(self):
        """Every ``(layer, token)`` decision in the job, in canonical order."""
        return tuple(
            (layer, token)
            for layer in self.layers
            for sequence in self.sequences
            for token in sequence
        )


# --- carriers ----------------------------------------------------------------


def _sequence_owner(config, sequence, cp_rank, tp_rank):
    """Tokens of one sequence held by a (cp, tp) pair.

    CP splits the sequence first, then SP subdivides each CP chunk inside the
    TP group. That order is the declared composition; reversing it produces a
    different, equally self-consistent assignment, which is exactly why it has
    to be written down rather than discovered.
    """
    chunk = partition_slices(len(sequence), config.cp)[cp_rank]
    if not config.sp:
        return tuple(sequence[i] for i in chunk)
    sub = partition_slices(len(chunk), config.tp)[tp_rank]
    return tuple(sequence[chunk.start + i] for i in sub)


def expected_decisions(config, layout, global_rank):
    """Decisions one global rank must hold, and only those."""
    coords = config.coordinates(global_rank)
    layer_slice = partition_slices(len(layout.layers), config.pp)[coords["pp"]]
    layers = [layout.layers[i] for i in layer_slice]
    sequence_slice = partition_slices(len(layout.sequences), config.dp)[coords["dp"]]

    tokens = []
    for index in sequence_slice:
        tokens.extend(_sequence_owner(config, layout.sequences[index], coords["cp"], coords["tp"]))
    return frozenset((layer, token) for layer in layers for token in tokens)


def carriers(config, layout):
    """Map every decision to the set of global ranks that must hold it.

    Built by inverting ``expected_decisions`` over the world rather than by
    composing per-dimension rules a second time: one derivation, so the two
    directions cannot drift apart.
    """
    out = {}
    for rank in range(config.world_size):
        for decision in expected_decisions(config, layout, rank):
            out.setdefault(decision, set()).add(rank)
    return {decision: frozenset(ranks) for decision, ranks in out.items()}


def replica_dimensions(config):
    """Dimensions that replicate decisions under this configuration."""
    dims = [d for d in REPLICATE if config.size_of(d) > 1]
    if config.sp:
        dims = [d for d in dims if d != "tp"]
    return tuple(dims)


# --- evidence ----------------------------------------------------------------


@dataclass(frozen=True)
class RankEvidence:
    """What one rank reported for one layer."""

    rank: int
    group: str
    layer: int
    fingerprints: dict = field(default_factory=dict)  # global_token_id -> semantic hash

    @classmethod
    def from_plan(cls, rank, group, layer, plan):
        """Read a RoutePlan's per-token semantic fingerprints."""
        return cls(
            rank=rank,
            group=group,
            layer=layer,
            fingerprints={int(k): v for k, v in plan["per_token_fingerprints"].items()},
        )

    def decisions(self):
        return frozenset((self.layer, token) for token in self.fingerprints)


# --- gates -------------------------------------------------------------------


def check_rank_set(config, evidence):
    """The reported rank set must be exactly one configuration's world.

    Duplicate ranks, unexpected ranks and mixed group labels are stale or
    mixed evidence and take precedence over a simultaneous missing-rank
    symptom: a contaminated rank set makes the missing-rank finding an
    artefact of the contamination rather than a fact about the run.
    """
    seen, groups = {}, {item.group for item in evidence}
    for item in evidence:
        seen.setdefault((item.rank, item.layer), []).append(item)

    duplicated = sorted(key for key, items in seen.items() if len(items) > 1)
    ranks = {rank for rank, _ in seen}
    unexpected = sorted(r for r in ranks if not 0 <= r < config.world_size)
    if duplicated or unexpected or len(groups) > 1:
        problems = []
        if duplicated:
            problems.append(f"duplicated (rank, layer) evidence: {duplicated[:8]}")
        if unexpected:
            problems.append(f"ranks outside [0, {config.world_size}): {unexpected[:8]}")
        if len(groups) > 1:
            problems.append(f"mixed groups: {sorted(groups)}")
        require(False, P3Verdict.STALE_RUN_METADATA, "; ".join(problems))

    missing = sorted(set(range(config.world_size)) - ranks)
    require(
        not missing,
        P3Verdict.MISSING_RANK,
        f"missing ranks for {config.describe()}: {missing[:16]}",
    )


def check_rank_ownership(config, layout, rank, items):
    """One rank must hold exactly the decisions it carries.

    ``items`` is every evidence record for this rank. A rank reports one
    record per layer it owns, so ownership can only be judged over the union:
    checking a single record would read every other layer of the same rank as
    a gap.

    Unauthorized decisions are reported before missing ones: holding a
    decision that belongs to another rank means the partition itself is wrong,
    and the gaps elsewhere are its consequence.
    """
    expected = expected_decisions(config, layout, rank)
    actual = frozenset().union(*(item.decisions() for item in items)) if items else frozenset()

    unauthorized = sorted(actual - expected)
    if unauthorized:
        owners = carriers(config, layout)
        detail = ", ".join(
            f"(layer={layer}, token={token})->ranks{sorted(owners.get((layer, token), ()))[:4]}"
            for layer, token in unauthorized[:6]
        )
        require(
            False,
            P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
            f"rank {rank} reports decisions it does not carry: {detail}",
        )

    missing = sorted(expected - actual)
    require(
        not missing,
        P3Verdict.INCOMPLETE_ARTIFACT,
        f"rank {rank} owes {len(missing)} decisions, first missing {missing[:6]}",
    )


def check_replica_agreement(config, layout, evidence):
    """Every carrier of a decision must report the same semantic fingerprint.

    Comparing a deduplicated single copy would miss a single-rank drift, which
    is the whole failure mode replication is supposed to rule out. Under
    ``ep`` the message names the dimension explicitly: a token-set or
    fingerprint difference that tracks the EP coordinate means expert
    placement leaked into the routing decision, which is a different bug from
    a TP replica drifting.
    """
    by_rank = {(item.rank, item.layer): item for item in evidence}
    for decision, ranks in carriers(config, layout).items():
        if len(ranks) < 2:
            continue
        layer, token = decision
        reported = {}
        for rank in sorted(ranks):
            item = by_rank.get((rank, layer))
            if item is not None and token in item.fingerprints:
                reported.setdefault(item.fingerprints[token], []).append(rank)
        if len(reported) > 1:
            dims = replica_dimensions(config) or ("replica",)
            spread = "; ".join(f"{h[:12]}<-ranks{r}" for h, r in sorted(reported.items()))
            require(
                False,
                P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
                f"carriers of (layer={layer}, token={token}) disagree across "
                f"{'/'.join(dims)}: {spread}",
            )


def validate_ownership(config, layout, evidence):
    """Full ownership gate for one configuration's evidence.

    Order is rank set -> per-rank ownership -> replica agreement, a dependency
    order: ownership of a contaminated rank set is meaningless, and comparing
    carriers before knowing each rank holds the right decisions would attribute
    a partition error to a numeric disagreement.
    """
    check_rank_set(config, evidence)
    by_rank = {}
    for item in evidence:
        by_rank.setdefault(item.rank, []).append(item)
    for rank in sorted(by_rank):
        check_rank_ownership(config, layout, rank, by_rank[rank])
    check_replica_agreement(config, layout, evidence)


def check_cross_config(layout, base, other):
    """Two configurations must produce identical decisions for every token.

    Token-centric, never rank-against-rank: the whole point of changing the
    parallel layout is that rank N holds different tokens afterwards. Each
    side is folded to ``(layer, token) -> fingerprint`` first, and only then
    compared.
    """
    base_config, base_evidence = base
    other_config, other_evidence = other
    validate_ownership(base_config, layout, base_evidence)
    validate_ownership(other_config, layout, other_evidence)

    folded = []
    for evidence in (base_evidence, other_evidence):
        merged = {}
        for item in evidence:
            for token, value in item.fingerprints.items():
                merged[(item.layer, token)] = value
        folded.append(merged)
    base_map, other_map = folded

    missing = sorted(set(base_map) - set(other_map))
    require(
        not missing,
        P3Verdict.INCOMPLETE_ARTIFACT,
        f"{other_config.describe()} is missing {len(missing)} decisions, first {missing[:6]}",
    )
    ghost = sorted(set(other_map) - set(base_map))
    require(
        not ghost,
        P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
        f"{other_config.describe()} reports decisions no base rank owns: {ghost[:6]}",
    )
    for decision in sorted(base_map):
        require(
            base_map[decision] == other_map[decision],
            P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
            f"(layer={decision[0]}, token={decision[1]}) differs between "
            f"{base_config.describe()} and {other_config.describe()}",
        )
