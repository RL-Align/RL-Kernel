"""Gate parameter-gradient collective provenance tests (T08).

P3 does not implement the all-reduce; it records what the all-reduce did, so
that a routing change can be checked against the reduction order before
anyone goes looking inside the router. These tests pin that the record is
usable as evidence: complete or fail-closed, fingerprinted over exactly what
changes the summation order, and able to name what moved between two runs.

Every negative case injects exactly one defect.
"""

import dataclasses

import pytest

from rl_engine.p3.collective_provenance import (
    COLLECTIVE_SCHEMA,
    CollectiveProvenance,
    GradientBucket,
    collective_delta,
    collective_evidence,
    collective_problems,
    require_same_reduction,
    validate_collective_provenance,
)
from rl_engine.p3.contract import P3Error, P3Verdict

WEIGHT = GradientBucket(0, ("gate.weight",), 4096 * 256, "fp32")
BIAS = GradientBucket(1, ("gate.bias",), 256, "fp32")


def _record(**overrides):
    fields = {
        "dp_size": 2,
        "dp_ranks": (0, 1),
        "algorithm": "ring",
        "configured_algorithm": "ring",
        "reduce_op": "avg",
        "accumulate_dtype": "fp32",
        "process_group": "dp:0-1",
        "source": "torch.distributed readback",
        "buckets": (WEIGHT, BIAS),
        **overrides,
    }
    return CollectiveProvenance(**fields)


def _verdict(excinfo):
    return excinfo.value.verdict


# --- usable evidence ---------------------------------------------------------


def test_complete_record_validates():
    record = _record()
    assert collective_problems(record) == []
    validate_collective_provenance(record)
    assert record.required


def test_dp1_runs_no_collective_and_needs_no_evidence():
    record = CollectiveProvenance.none()
    assert not record.required
    validate_collective_provenance(record)
    assert record.to_record()["source"] == "no_collective"


def test_reduction_order_is_bucket_then_parameter():
    assert _record().reduction_order() == ((0, "gate.weight"), (1, "gate.bias"))


def test_evidence_fragment_carries_record_and_fingerprint():
    record = _record()
    fragment = collective_evidence(record)
    assert fragment["gate_grad_collective"]["schema_version"] == COLLECTIVE_SCHEMA
    assert fragment["gate_grad_collective_fingerprint"] == record.fingerprint()


# --- fingerprint -------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_size": 4, "dp_ranks": (0, 1, 2, 3)},
        {"algorithm": "tree", "configured_algorithm": "tree"},
        {"reduce_op": "sum"},
        {"accumulate_dtype": "bf16"},
        {"buckets": (BIAS, WEIGHT)},
        {"gradient_accumulation_steps": 2},
        {"overlap_with_backward": True},
        {"bucket_cap_mb": 25.0},
        {"process_group": "dp:2-3"},
    ],
)
def test_anything_that_moves_the_summation_order_moves_the_fingerprint(overrides):
    assert _record().fingerprint() != _record(**overrides).fingerprint()


@pytest.mark.parametrize(
    "overrides",
    [{"source": "nccl env dump"}, {"notes": {"host": "node-7"}}],
)
def test_how_it_was_observed_is_not_a_numeric_difference(overrides):
    """Control: the same collective seen two ways must not read as drift."""
    assert _record().fingerprint() == _record(**overrides).fingerprint()


# --- structural defects ------------------------------------------------------


def _assert_schema_mismatch(record, needle):
    problems = collective_problems(record)
    assert any(needle in problem for problem in problems), problems
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(record)
    assert _verdict(excinfo) == P3Verdict.SCHEMA_MISMATCH


def test_unsupported_schema_version():
    _assert_schema_mismatch(_record(schema_version="p3-gate-grad-collective.v0"), "unsupported")


def test_non_positive_dp_size():
    _assert_schema_mismatch(_record(dp_size=0), "dp_size")


def test_non_positive_accumulation_steps():
    _assert_schema_mismatch(_record(gradient_accumulation_steps=0), "gradient_accumulation_steps")


def test_rank_list_shorter_than_dp_size():
    _assert_schema_mismatch(_record(dp_ranks=(0,)), "distinct ranks")


def test_duplicate_rank():
    _assert_schema_mismatch(_record(dp_ranks=(0, 0)), "distinct ranks")


def test_unrecognised_algorithm_is_unattributable():
    _assert_schema_mismatch(_record(algorithm="homegrown"), "unattributable")


def test_unrecognised_reduce_op():
    _assert_schema_mismatch(_record(reduce_op="max"), "reduce_op")


def test_empty_bucket():
    _assert_schema_mismatch(
        _record(buckets=(GradientBucket(0, (), 0, "fp32"),)), "lists no parameters"
    )


def test_bucket_without_a_dtype():
    _assert_schema_mismatch(
        _record(buckets=(GradientBucket(0, ("gate.weight",), 1, ""),)), "declares no dtype"
    )


def test_one_parameter_reduced_in_two_buckets():
    duplicate = GradientBucket(1, ("gate.weight",), 1, "fp32")
    _assert_schema_mismatch(_record(buckets=(WEIGHT, duplicate)), "reduced once")


def test_bucket_indices_must_ascend():
    _assert_schema_mismatch(
        _record(buckets=(BIAS, dataclasses.replace(WEIGHT, index=0))), "ascending"
    )


# --- missing evidence --------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "algorithm",
        "configured_algorithm",
        "reduce_op",
        "accumulate_dtype",
        "process_group",
        "source",
    ],
)
def test_a_missing_field_is_missing_evidence_not_a_default(field):
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(**{field: ""}))
    assert _verdict(excinfo) == P3Verdict.MISSING_PROVENANCE
    assert field in str(excinfo.value)


def test_no_buckets_is_missing_evidence():
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(buckets=()))
    assert _verdict(excinfo) == P3Verdict.MISSING_PROVENANCE


def test_an_unsourced_record_is_not_an_observation():
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(source=""))
    assert _verdict(excinfo) == P3Verdict.MISSING_PROVENANCE


# --- fallback ----------------------------------------------------------------


def test_declared_fallback_fails_closed():
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(fallback_reason="nccl unavailable, used gloo"))
    assert _verdict(excinfo) == P3Verdict.SILENT_FALLBACK


def test_configured_value_may_not_pose_as_an_observation():
    """Read back something other than what was configured, with no reason given."""
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(configured_algorithm="tree", algorithm="ring"))
    assert _verdict(excinfo) == P3Verdict.SILENT_FALLBACK
    assert "pose as an observation" in str(excinfo.value)


def test_structure_precedes_completeness():
    """A malformed record cannot be judged complete."""
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(dp_ranks=(0, 0), source=""))
    assert _verdict(excinfo) == P3Verdict.SCHEMA_MISMATCH


def test_completeness_precedes_fallback():
    """A fallback reported by a record with no readback is not an observation."""
    with pytest.raises(P3Error) as excinfo:
        validate_collective_provenance(_record(source="", fallback_reason="gloo"))
    assert _verdict(excinfo) == P3Verdict.MISSING_PROVENANCE


# --- attribution -------------------------------------------------------------


def test_identical_records_have_no_delta():
    assert collective_delta(_record(), _record()) == ()


def test_delta_names_the_first_position_the_order_moved():
    deltas = collective_delta(
        _record(), _record(buckets=(BIAS, dataclasses.replace(WEIGHT, index=1)))
    )
    assert any("position 0" in d and "gate.weight" in d and "gate.bias" in d for d in deltas)


def test_delta_reports_a_shorter_or_longer_order():
    deltas = collective_delta(_record(), _record(buckets=(WEIGHT,)))
    assert any("2 vs 1 reduced parameters" in d for d in deltas)


def test_a_different_dp_size_is_a_delta_not_a_defect():
    """Two DP sizes must have different collectives; that is attribution, not failure."""
    other = _record(dp_size=4, dp_ranks=(0, 1, 2, 3))
    deltas = collective_delta(_record(), other)
    assert any(d.startswith("dp_size:") for d in deltas)
    validate_collective_provenance(other)  # still perfectly valid evidence


def test_require_same_reduction_passes_on_identical_records():
    require_same_reduction(_record(), _record())


def test_require_same_reduction_rejects_an_uncontrolled_variable():
    with pytest.raises(P3Error) as excinfo:
        require_same_reduction(_record(), _record(accumulate_dtype="bf16"))
    assert _verdict(excinfo) == P3Verdict.UPSTREAM_CONTRACT_MISMATCH
    assert "held constant" in str(excinfo.value)


def test_require_same_reduction_validates_both_sides_first():
    """A broken side is reported as broken, not as a difference."""
    with pytest.raises(P3Error) as excinfo:
        require_same_reduction(_record(), _record(source=""))
    assert _verdict(excinfo) == P3Verdict.MISSING_PROVENANCE
