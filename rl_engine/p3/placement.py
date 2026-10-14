"""Versioned logical/physical expert placement map (T08).

``logical_expert_id`` is the canonical route key: the router's semantic output,
which must not change when expert parallelism moves an expert to another
device. ``physical_expert_id`` is Envelope metadata describing where that
expert currently lives. This module owns the versioned bijection between them.

Two readings of ``physical_expert_id`` are in circulation and this module
supports both explicitly, because they are not interchangeable and the choice
has not yet been settled:

``global_permutation``
    ``physical_expert_id`` is unique across all ``E`` experts, so the map is a
    permutation of ``range(E)``. This is what ``assembler.assemble`` builds by
    default and what ``checker.validate_plan`` enforces today
    (``sorted(mapping) == list(range(E))``).

``rank_local``
    ``physical_expert_id`` indexes the hosting rank's own expert table, so ids
    repeat across EP ranks and the bijection is
    ``logical <-> (ep_rank, physical)``. This is the reading in the task
    contract, which defines ``physical_expert_id`` as "the rank-local id under
    the current EP placement".

Under ``EP=8`` with ``E=256`` the two disagree: rank-local ids are ``0..31`` on
each of the eight ranks, so ``sorted(mapping)`` is ``[0]*8 + [1]*8 + ...`` and
the current bijection check would reject a correct placement. A map carries its
``semantics`` so validation can apply the matching rule instead of guessing,
and so a map recorded under one reading can never be silently judged under the
other.

``plan_fragment`` extends the existing ``placement_map`` object additively:
``version`` and ``logical_to_physical`` keep their meaning and position, so
consumers that read only those two keep working unchanged.
"""

from dataclasses import dataclass

from .contract import E, P3Verdict
from .oracle import require
from .serialization import fingerprint

PLACEMENT_SCHEMA = "p3-placement.v1"

#: How ``physical_expert_id`` is numbered. Recorded, never inferred.
SEMANTICS_GLOBAL = "global_permutation"
SEMANTICS_RANK_LOCAL = "rank_local"
SEMANTICS = (SEMANTICS_GLOBAL, SEMANTICS_RANK_LOCAL)

#: How the table was produced (provenance only; validation never branches here).
STRATEGY_EXPLICIT = "explicit"
STRATEGY_IDENTITY = "identity"
STRATEGY_BLOCK = "block"  # contiguous chunk per rank
STRATEGY_INTERLEAVE = "interleave"  # round robin: rank = logical % ep_size
STRATEGIES = (STRATEGY_EXPLICIT, STRATEGY_IDENTITY, STRATEGY_BLOCK, STRATEGY_INTERLEAVE)

#: Padding rows carry expert -1 on both sides and are never looked up.
PADDING_EXPERT_ID = -1


@dataclass(frozen=True)
class PlacementMap:
    """A versioned expert placement under one declared numbering.

    ``ep_rank_of`` and ``physical_of`` are both indexed by
    ``logical_expert_id``. Tuples keep the instance hashable and the
    fingerprint order fixed.

    Construction never validates. A map is a claim; ``placement_problems`` and
    the gates below are where the claim is tested. A map that cannot be built
    also cannot be rejected with a verdict, and the negative fixtures need to
    build broken maps in order to prove the gates reject them.
    """

    version: str
    semantics: str
    ep_size: int
    ep_rank_of: tuple[int, ...]
    physical_of: tuple[int, ...]
    strategy: str = STRATEGY_EXPLICIT
    num_experts: int = E
    schema_version: str = PLACEMENT_SCHEMA

    # --- lookups ---------------------------------------------------------

    def locate(self, logical_expert_id):
        """Return ``(ep_rank, physical_expert_id)`` for a logical expert."""
        require(
            0 <= logical_expert_id < len(self.physical_of),
            P3Verdict.INVALID_PLACEMENT_MAP,
            f"logical expert {logical_expert_id} outside [0, {len(self.physical_of)})",
        )
        return self.ep_rank_of[logical_expert_id], self.physical_of[logical_expert_id]

    def ep_rank(self, logical_expert_id):
        """Return the EP rank hosting a logical expert."""
        return self.locate(logical_expert_id)[0]

    def physical_expert_id(self, logical_expert_id):
        """Return the physical id of a logical expert under this numbering."""
        return self.locate(logical_expert_id)[1]

    def experts_on_rank(self, ep_rank):
        """Logical experts hosted by one EP rank, ordered by physical id."""
        owned = sorted(
            (self.physical_of[logical], logical)
            for logical, rank in enumerate(self.ep_rank_of)
            if rank == ep_rank
        )
        return tuple(logical for _, logical in owned)

    # --- interop ---------------------------------------------------------

    def logical_to_physical(self):
        """The flat ``logical -> physical`` list used by the existing plan.

        Under ``rank_local`` this list contains repeats by construction, which
        the current ``assembler``/``checker`` bijection check rejects. That is
        the open question this module makes explicit rather than papering over.
        """
        return list(self.physical_of)

    def plan_fragment(self):
        """The ``placement_map`` object for a RoutePlan.

        Additive: ``version`` and ``logical_to_physical`` are unchanged, so a
        consumer reading only those two is unaffected by the extra keys.
        """
        return {
            "version": self.version,
            "logical_to_physical": self.logical_to_physical(),
            "placement_schema_version": self.schema_version,
            "semantics": self.semantics,
            "ep_size": self.ep_size,
            "logical_to_ep_rank": list(self.ep_rank_of),
            "strategy": self.strategy,
            "placement_fingerprint": self.fingerprint(),
        }

    def fingerprint(self):
        """Canonical hash over schema, identity, numbering, shape and table.

        ``semantics`` and ``strategy`` are included: two tables that happen to
        coincide but were declared under different readings are different
        provenance, and must not compare equal.
        """
        return fingerprint(
            {
                "schema_version": self.schema_version,
                "version": self.version,
                "semantics": self.semantics,
                "strategy": self.strategy,
                "num_experts": self.num_experts,
                "ep_size": self.ep_size,
                "ep_rank_of": list(self.ep_rank_of),
                "physical_of": list(self.physical_of),
            }
        )

    # --- construction ----------------------------------------------------

    @classmethod
    def identity(cls, version="identity.v1", num_experts=E):
        """The single-rank identity placement the assembler defaults to."""
        return cls(
            version=version,
            semantics=SEMANTICS_GLOBAL,
            ep_size=1,
            ep_rank_of=(0,) * num_experts,
            physical_of=tuple(range(num_experts)),
            strategy=STRATEGY_IDENTITY,
            num_experts=num_experts,
        )

    @classmethod
    def from_logical_to_physical(
        cls,
        version,
        mapping,
        ep_size=1,
        ep_rank_of=None,
        semantics=SEMANTICS_GLOBAL,
        strategy=STRATEGY_EXPLICIT,
    ):
        """Adopt an existing flat ``logical -> physical`` list.

        This is the bridge from plans already recorded under the global
        reading; ``ep_rank_of`` defaults to rank 0 for every expert, which is
        accurate for the single-rank case and informational otherwise.
        """
        physical = tuple(int(p) for p in mapping)
        ranks = (
            tuple(int(r) for r in ep_rank_of) if ep_rank_of is not None else (0,) * len(physical)
        )
        return cls(
            version=version,
            semantics=semantics,
            ep_size=ep_size,
            ep_rank_of=ranks,
            physical_of=physical,
            strategy=strategy,
            num_experts=len(physical),
        )

    @classmethod
    def from_pairs(
        cls,
        version,
        pairs,
        ep_size,
        semantics=SEMANTICS_RANK_LOCAL,
        strategy=STRATEGY_EXPLICIT,
        num_experts=E,
    ):
        """Build from an explicit ``logical -> (ep_rank, physical)`` table.

        Logical ids missing from ``pairs`` become ``(-1, -1)`` holes so
        ``placement_problems`` reports them as gaps.
        """
        table = {int(k): (int(v[0]), int(v[1])) for k, v in dict(pairs).items()}
        entries = [table.get(i, (-1, -1)) for i in range(num_experts)]
        return cls(
            version=version,
            semantics=semantics,
            ep_size=ep_size,
            ep_rank_of=tuple(r for r, _ in entries),
            physical_of=tuple(p for _, p in entries),
            strategy=strategy,
            num_experts=num_experts,
        )

    @classmethod
    def from_ep_config(
        cls,
        version,
        ep_size,
        strategy=STRATEGY_BLOCK,
        semantics=SEMANTICS_RANK_LOCAL,
        num_experts=E,
    ):
        """Generate a regular placement; bijective under either semantics.

        ``block`` gives each rank a contiguous chunk, with the first
        ``num_experts % ep_size`` ranks taking one extra expert; ``interleave``
        assigns round robin.

        A global physical id is the expert's position in the rank-major
        concatenation of all per-rank tables, which is the only numbering that
        agrees with the rank-local one on where an expert sits inside its own
        rank. For ``block`` that collapses to the identity permutation.
        """
        require(
            num_experts > 0 and ep_size > 0,
            P3Verdict.INVALID_PLACEMENT_MAP,
            f"num_experts and ep_size must be positive, got {num_experts}/{ep_size}",
        )
        require(
            strategy in (STRATEGY_BLOCK, STRATEGY_INTERLEAVE),
            P3Verdict.INVALID_PLACEMENT_MAP,
            f"cannot generate strategy {strategy!r}",
        )
        require(
            semantics in SEMANTICS,
            P3Verdict.INVALID_PLACEMENT_MAP,
            f"unknown placement semantics {semantics!r}",
        )

        base, extra = divmod(num_experts, ep_size)
        widths = [base + (1 if r < extra else 0) for r in range(ep_size)]
        starts, offset = [], 0
        for width in widths:
            starts.append(offset)
            offset += width

        ranks, locals_ = [0] * num_experts, [0] * num_experts
        for rank in range(ep_size):
            if strategy == STRATEGY_BLOCK:
                owned = range(starts[rank], starts[rank] + widths[rank])
            else:
                owned = range(rank, num_experts, ep_size)
            for local, logical in enumerate(owned):
                ranks[logical] = rank
                locals_[logical] = local

        if semantics == SEMANTICS_RANK_LOCAL:
            physical = locals_
        else:
            physical = [starts[ranks[i]] + locals_[i] for i in range(num_experts)]

        return cls(
            version=version,
            semantics=semantics,
            ep_size=ep_size,
            ep_rank_of=tuple(ranks),
            physical_of=tuple(physical),
            strategy=strategy,
            num_experts=num_experts,
        )

    @classmethod
    def from_plan_fragment(cls, fragment):
        """Rebuild a map from a plan's ``placement_map`` object.

        A fragment without the extended keys is read as the global reading at
        ``ep_size=1``: that is what such a plan was recorded under, and
        assuming anything else would reinterpret old evidence.
        """
        mapping = fragment["logical_to_physical"]
        ranks = fragment.get("logical_to_ep_rank")
        return cls.from_logical_to_physical(
            fragment["version"],
            mapping,
            ep_size=int(fragment.get("ep_size", 1)),
            ep_rank_of=ranks,
            semantics=fragment.get("semantics", SEMANTICS_GLOBAL),
            strategy=fragment.get("strategy", STRATEGY_EXPLICIT),
        )


# --- structural validation ---------------------------------------------------


def placement_problems(placement):
    """Return every structural defect in a map (empty list == valid).

    Defects are collected rather than short-circuited: a caller repairing a
    table wants the whole list, while the gate only needs to know whether the
    list is empty. Ordered by how early a defect corrupts the table, so the
    first entry is usually the root cause.
    """
    problems = []

    if not placement.version:
        problems.append("placement version is empty")
    if placement.semantics not in SEMANTICS:
        problems.append(f"unknown semantics {placement.semantics!r}")
    if placement.strategy not in STRATEGIES:
        problems.append(f"unknown strategy {placement.strategy!r}")
    if placement.num_experts <= 0:
        problems.append(f"num_experts must be positive, got {placement.num_experts}")
    if placement.ep_size <= 0:
        problems.append(f"ep_size must be positive, got {placement.ep_size}")
    if len(placement.physical_of) != placement.num_experts:
        problems.append(
            f"table holds {len(placement.physical_of)} physical entries but "
            f"num_experts is {placement.num_experts}"
        )
    if len(placement.ep_rank_of) != len(placement.physical_of):
        problems.append(
            f"{len(placement.ep_rank_of)} rank entries against "
            f"{len(placement.physical_of)} physical entries"
        )
    if problems:
        return problems  # shape is broken; per-entry findings would be noise

    gaps, out_of_range = [], []
    for logical, (rank, physical) in enumerate(
        zip(placement.ep_rank_of, placement.physical_of, strict=True)
    ):
        if rank < 0 or physical < 0:
            gaps.append(logical)
        elif rank >= placement.ep_size:
            out_of_range.append(logical)
    if gaps:
        problems.append(f"unmapped logical experts (gaps): {gaps[:16]}")
    if out_of_range:
        problems.append(
            f"logical experts on an ep_rank outside [0, {placement.ep_size}): {out_of_range[:16]}"
        )
    if problems:
        return problems

    if placement.semantics == SEMANTICS_GLOBAL:
        problems.extend(_global_problems(placement))
    else:
        problems.extend(_rank_local_problems(placement))
    return problems


def _global_problems(placement):
    """Under the global reading the physical ids are a permutation of range(E)."""
    physical = sorted(placement.physical_of)
    if physical == list(range(placement.num_experts)):
        return []
    seen, duplicates = set(), []
    for value in placement.physical_of:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    missing = [v for v in range(placement.num_experts) if v not in seen]
    detail = []
    if duplicates:
        detail.append(f"repeated physical ids {sorted(set(duplicates))[:16]}")
    if missing:
        detail.append(f"physical ids never used {missing[:16]}")
    outside = [v for v in placement.physical_of if v >= placement.num_experts]
    if outside:
        detail.append(f"physical ids >= num_experts {sorted(set(outside))[:16]}")
    return [
        "global_permutation requires physical ids to be a permutation of "
        f"[0, {placement.num_experts}): " + "; ".join(detail)
    ]


def _rank_local_problems(placement):
    """Under the rank-local reading each rank's ids are exactly ``0..n-1``."""
    problems = []
    claimed, slots_by_rank = {}, {}
    for logical, (rank, physical) in enumerate(
        zip(placement.ep_rank_of, placement.physical_of, strict=True)
    ):
        claimed.setdefault((rank, physical), []).append(logical)
        slots_by_rank.setdefault(rank, []).append(physical)

    overlaps = {slot: ids for slot, ids in claimed.items() if len(ids) > 1}
    if overlaps:
        shown = sorted(overlaps.items())[:8]
        problems.append(
            "physical slots claimed by more than one logical expert: "
            + ", ".join(f"(ep_rank={r}, physical={p})<-{ids}" for (r, p), ids in shown)
        )

    for rank in sorted(slots_by_rank):
        slots = sorted(slots_by_rank[rank])
        if slots != list(range(len(slots))):
            problems.append(
                f"ep_rank {rank} physical slots are not 0..{len(slots) - 1} (got {slots[:16]}): "
                "a rank-local id must index that rank's own expert table"
            )
    return problems


# --- gates -------------------------------------------------------------------


def validate_placement_map(placement):
    """Fail closed on a structurally invalid map -> ``INVALID_PLACEMENT_MAP``."""
    problems = placement_problems(placement)
    require(not problems, P3Verdict.INVALID_PLACEMENT_MAP, "; ".join(problems))


def check_placement_version(declared_version, placement):
    """Fail closed when evidence was produced under another placement.

    Separate from the structural gate because the repair paths differ: a
    version mismatch means validating the table in hand proves nothing about
    the evidence, and the fix is to fetch the right map.
    """
    require(
        declared_version == placement.version,
        P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH,
        f"evidence declares placement {declared_version!r} but the map is {placement.version!r}",
    )


def validate_plan_placement(plan, placement=None):
    """Check every Envelope physical id in a RoutePlan against the map.

    Gate order is version -> structure -> rows, a dependency order: rows cannot
    be judged against a map from another placement, nor against a table that is
    not a bijection under its own declared semantics.

    The Envelope ``rank`` is deliberately not compared with ``ep_rank``. Expert
    parallelism is not a token dimension: the rank that recorded a token is a
    token-dimension coordinate (DP/CP/TP/PP) while ``ep_rank`` is an
    expert-dimension one, so under any combined configuration a token recorded
    on one rank routinely routes to experts hosted on another. Token ownership
    is a separate concern with its own hooks.
    """
    fragment = plan["placement_map"]
    placement = PlacementMap.from_plan_fragment(fragment) if placement is None else placement
    check_placement_version(fragment["version"], placement)
    validate_placement_map(placement)

    rows, envelopes = plan["core"], plan["envelope"]
    require(
        len(rows) == len(envelopes),
        P3Verdict.INVALID_PLACEMENT_MAP,
        f"{len(rows)} core rows against {len(envelopes)} envelopes",
    )
    for row, (core, env) in enumerate(zip(rows, envelopes, strict=True)):
        require(
            env["placement_map_version"] == placement.version,
            P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH,
            f"row {row}: envelope placement version mismatch",
        )
        expert = core["logical_expert_id"]
        if expert == PADDING_EXPERT_ID:
            require(
                env["physical_expert_id"] == PADDING_EXPERT_ID,
                P3Verdict.INVALID_PLACEMENT_MAP,
                f"row {row}: padding carries physical {env['physical_expert_id']}, expected -1",
            )
            continue
        expected = placement.physical_expert_id(expert)
        require(
            env["physical_expert_id"] == expected,
            P3Verdict.INVALID_PLACEMENT_MAP,
            f"row {row}: logical expert {expert} maps to physical {expected} under "
            f"{placement.version!r} ({placement.semantics}), "
            f"envelope records {env['physical_expert_id']}",
        )
