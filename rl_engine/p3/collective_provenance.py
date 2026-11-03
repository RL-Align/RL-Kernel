"""Gate parameter-gradient collective provenance under DP (T08).

P3 produces ``dz`` and stops. P1 owns the gate GEMM backward, and the
all-reduce that sums ``dW_gate`` across data-parallel ranks is the training
framework's. **Nothing here implements or wraps a collective.**

It is recorded because it is a reduction, and a reduction has an order. Bucket
composition, bucket order, the algorithm and the accumulation dtype together
decide the floating-point summation order of the gate weight gradient. Change
any of them and ``dW_gate`` changes in the last bits, which moves the next
step's gate logits, which moves router scores, which -- the router being a
discrete decision layer -- can flip a Top-K selection. By then the drift looks
like a router defect.

So the record exists to make that attributable: an unexplained routing change
between two runs can be checked against their collective provenance before
anyone goes looking inside the router.

Two things the contract insists on and this module enforces:

- **A configured value may not pose as an actual readback.** The record
  carries both and a silent substitution is ``SILENT_FALLBACK``, not a
  rounding of the truth.
- **Absent evidence is not blank evidence.** A missing field fails closed with
  ``MISSING_PROVENANCE`` rather than defaulting, because a default would make
  two unrelated runs compare equal.

At ``dp_size == 1`` there is no all-reduce, so there is nothing to record.
``CollectiveProvenance.none()`` says exactly that and is valid; demanding
evidence of a collective that does not run would be fail-closed theatre.
"""

from dataclasses import dataclass, field

from .contract import P3Verdict
from .oracle import require
from .serialization import fingerprint

COLLECTIVE_SCHEMA = "p3-gate-grad-collective.v1"

#: Reduction algorithms whose summation order is determined by the algorithm
#: plus the rank count. An implementation outside this set may still be used,
#: but it has to be named in a contract delta first: an unrecognised algorithm
#: is unattributable, which is the one thing this record exists to prevent.
KNOWN_ALGORITHMS = (
    "ring",
    "tree",
    "reduce_scatter_all_gather",
    "double_binary_tree",
)

KNOWN_REDUCE_OPS = ("sum", "avg", "premul_sum")


@dataclass(frozen=True)
class GradientBucket:
    """One all-reduce bucket, in the order its parameters are reduced.

    ``parameters`` is ordered, not a set: the order inside a bucket is part of
    the summation order, and two frameworks that bucket the same parameters
    differently produce different last bits.
    """

    index: int
    parameters: tuple
    numel: int
    dtype: str

    def to_record(self):
        return {
            "index": self.index,
            "parameters": list(self.parameters),
            "numel": self.numel,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class CollectiveProvenance:
    """What the gate-gradient all-reduce actually did, as evidence.

    ``configured_algorithm`` is what was asked for and ``algorithm`` is what
    was read back. They are separate fields on purpose; collapsing them would
    let a configured value stand in for an observation.

    ``source`` names how the readback was obtained. An unsourced record is not
    an observation, so it is treated as missing evidence rather than as a
    weaker kind of evidence.
    """

    dp_size: int
    dp_ranks: tuple = ()
    algorithm: str = ""
    configured_algorithm: str = ""
    reduce_op: str = ""
    accumulate_dtype: str = ""
    buckets: tuple = ()
    overlap_with_backward: bool = False
    gradient_accumulation_steps: int = 1
    bucket_cap_mb: float = 0.0
    process_group: str = ""
    source: str = ""
    fallback_reason: str = ""
    schema_version: str = COLLECTIVE_SCHEMA
    notes: dict = field(default_factory=dict)

    @classmethod
    def none(cls, dp_size=1):
        """The record for a configuration that runs no gate-gradient collective."""
        return cls(dp_size=dp_size, source="no_collective")

    @property
    def required(self):
        """Whether this configuration runs a collective at all."""
        return self.dp_size > 1

    def reduction_order(self):
        """The parameter-level summation order this record pins down.

        Returned as data rather than a string so a caller can diff two orders
        and name the first parameter that moved.
        """
        return tuple(
            (bucket.index, parameter) for bucket in self.buckets for parameter in bucket.parameters
        )

    def to_record(self):
        return {
            "schema_version": self.schema_version,
            "dp_size": self.dp_size,
            "dp_ranks": list(self.dp_ranks),
            "algorithm": self.algorithm,
            "configured_algorithm": self.configured_algorithm,
            "reduce_op": self.reduce_op,
            "accumulate_dtype": self.accumulate_dtype,
            "buckets": [bucket.to_record() for bucket in self.buckets],
            "overlap_with_backward": self.overlap_with_backward,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "bucket_cap_mb": self.bucket_cap_mb,
            "process_group": self.process_group,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
            "notes": dict(sorted(self.notes.items())),
        }

    def fingerprint(self):
        """Hash over everything that can change the summation order.

        ``source`` and ``notes`` are excluded: how the same behaviour was
        observed is not itself a numeric difference, and including it would
        make two identical collectives read back by different mechanisms look
        like a drift.
        """
        record = self.to_record()
        for excluded in ("source", "notes"):
            record.pop(excluded)
        return fingerprint(record)


# --- validation --------------------------------------------------------------


def collective_problems(record):
    """Structural defects in a record (empty list == usable evidence)."""
    problems = []
    if record.schema_version != COLLECTIVE_SCHEMA:
        problems.append(f"unsupported schema {record.schema_version!r}")
    if record.dp_size < 1:
        problems.append(f"dp_size must be at least 1, got {record.dp_size}")
    if record.gradient_accumulation_steps < 1:
        problems.append(
            f"gradient_accumulation_steps must be at least 1, "
            f"got {record.gradient_accumulation_steps}"
        )
    if not record.required:
        return problems

    ranks = tuple(record.dp_ranks)
    if len(ranks) != record.dp_size or len(set(ranks)) != len(ranks):
        problems.append(
            f"dp_ranks must list {record.dp_size} distinct ranks, got {list(ranks)[:8]}"
        )
    if record.algorithm and record.algorithm not in KNOWN_ALGORITHMS:
        problems.append(
            f"algorithm {record.algorithm!r} is not one of {KNOWN_ALGORITHMS}; "
            "an unrecognised reduction is unattributable"
        )
    if record.reduce_op and record.reduce_op not in KNOWN_REDUCE_OPS:
        problems.append(f"reduce_op {record.reduce_op!r} is not one of {KNOWN_REDUCE_OPS}")

    seen, indices = {}, []
    for bucket in record.buckets:
        indices.append(bucket.index)
        if not bucket.parameters:
            problems.append(f"bucket {bucket.index} lists no parameters")
        if bucket.numel < 0:
            problems.append(f"bucket {bucket.index} has negative numel {bucket.numel}")
        if not bucket.dtype:
            problems.append(f"bucket {bucket.index} declares no dtype")
        for parameter in bucket.parameters:
            if parameter in seen:
                problems.append(
                    f"parameter {parameter!r} appears in buckets "
                    f"{seen[parameter]} and {bucket.index}; a gradient is reduced once"
                )
            seen[parameter] = bucket.index
    if record.buckets and indices != sorted(set(indices)):
        problems.append(f"bucket indices must be distinct and ascending, got {indices[:8]}")
    return problems


def validate_collective_provenance(record):
    """Fail closed on unusable gate-gradient collective evidence.

    Order is structure -> completeness -> fallback. Structure first because a
    malformed record cannot be judged complete; completeness before fallback
    because a fallback reported by a record that is missing its readback is
    not an observation of a fallback.
    """
    problems = collective_problems(record)
    require(not problems, P3Verdict.SCHEMA_MISMATCH, "; ".join(problems))

    if not record.required:
        return

    missing = [
        name
        for name in (
            "algorithm",
            "configured_algorithm",
            "reduce_op",
            "accumulate_dtype",
            "process_group",
            "source",
        )
        if not getattr(record, name)
    ]
    if not record.buckets:
        missing.append("buckets")
    require(
        not missing,
        P3Verdict.MISSING_PROVENANCE,
        f"gate-gradient collective evidence is incomplete at dp={record.dp_size}: "
        f"missing {sorted(missing)}",
    )

    require(
        not record.fallback_reason,
        P3Verdict.SILENT_FALLBACK,
        f"gate-gradient collective fell back: {record.fallback_reason}",
    )
    require(
        record.algorithm == record.configured_algorithm,
        P3Verdict.SILENT_FALLBACK,
        f"configured collective {record.configured_algorithm!r} but read back "
        f"{record.algorithm!r} with no fallback_reason; a configured value may not "
        "pose as an observation",
    )


# --- attribution --------------------------------------------------------------


def collective_delta(base, other):
    """Everything that differs between two records, as attribution evidence.

    Returns descriptions rather than raising. Two configurations with
    different DP sizes *must* have different collectives, so a difference is
    not by itself a defect -- it is the thing a downstream drift should be
    checked against first.
    """
    deltas = []
    for name in (
        "dp_size",
        "algorithm",
        "reduce_op",
        "accumulate_dtype",
        "gradient_accumulation_steps",
        "overlap_with_backward",
        "bucket_cap_mb",
        "process_group",
    ):
        left, right = getattr(base, name), getattr(other, name)
        if left != right:
            deltas.append(f"{name}: {left!r} -> {right!r}")

    left_order, right_order = base.reduction_order(), other.reduction_order()
    if left_order != right_order:
        first = next(
            (
                index
                for index, (a, b) in enumerate(zip(left_order, right_order, strict=False))
                if a != b
            ),
            min(len(left_order), len(right_order)),
        )
        detail = f"reduction order differs from position {first}"
        if first < min(len(left_order), len(right_order)):
            detail += f": {left_order[first]} -> {right_order[first]}"
        elif len(left_order) != len(right_order):
            detail += f": {len(left_order)} vs {len(right_order)} reduced parameters"
        deltas.append(detail)
    return tuple(deltas)


def require_same_reduction(base, other):
    """Fail closed when a comparison claims DP is not the variable under test.

    In a one-at-a-time experiment where the gate-gradient collective is not
    the declared axis, a difference here is an uncontrolled variable: the
    comparison would attribute its effect to whatever *was* declared.
    """
    validate_collective_provenance(base)
    validate_collective_provenance(other)
    deltas = collective_delta(base, other)
    require(
        not deltas,
        P3Verdict.UPSTREAM_CONTRACT_MISMATCH,
        "gate-gradient collective is not held constant across the comparison: " + "; ".join(deltas),
    )


def collective_evidence(record):
    """The provenance fragment a RoutePlan or binding carries.

    Additive and self-describing: a consumer that does not know about the
    gate-gradient collective sees one more key, and one that does gets the
    fingerprint without having to re-derive it.
    """
    return {
        "gate_grad_collective": record.to_record(),
        "gate_grad_collective_fingerprint": record.fingerprint(),
    }
