# SPDX-License-Identifier: Apache-2.0
import struct

import pytest

from rl_engine.p2.contract import (
    ContractError,
    Identity,
    Status,
    runtime_policy,
    validate_runtime_policy,
)
from rl_engine.p2.negative import evaluate_negative, negative_cases
from rl_engine.p2.planner import (
    PlannerMock,
    validate_candidates,
    validate_snapshot,
    validate_topology,
)
from rl_engine.p2.provider import validate_envelope


@pytest.mark.parametrize("case", negative_cases(), ids=lambda c: c["case_id"])
def test_executable_negative_fixture(case):
    assert evaluate_negative(case) == case["expected_status"]
    assert case["expected_status"] != Status.PASS.value


@pytest.mark.parametrize("field", ["split_k", "split_kv", "stream_k", "dynamic_partition"])
def test_all_forbidden_partitions(field):
    policy = runtime_policy()
    policy[field] = True
    with pytest.raises(ContractError, match=Status.FORBIDDEN_SPLIT_REDUCTION.value):
        validate_runtime_policy(policy)


@pytest.mark.parametrize("value", [True, 0, -1, 2, "auto", None])
def test_dynamic_or_invalid_num_splits(value):
    policy = runtime_policy()
    policy["num_splits"] = value
    with pytest.raises(ContractError, match=Status.FORBIDDEN_SPLIT_REDUCTION.value):
        validate_runtime_policy(policy)


@pytest.mark.parametrize("value", [{}, None, [], {"schema": "unknown"}])
@pytest.mark.parametrize(
    "validator",
    [
        validate_snapshot,
        validate_candidates,
        validate_envelope,
        validate_runtime_policy,
        lambda x: validate_topology(x, 129),
    ],
)
def test_malformed_public_metadata_has_versioned_status(validator, value):
    with pytest.raises(ContractError) as error:
        validator(value)
    assert error.value.status == Status.SCHEMA_MISMATCH


def test_unknown_kernel_readback_cannot_be_faked():
    policy = runtime_policy()
    policy["kernel_fingerprint"] = "configured-but-not-read-back"
    with pytest.raises(ContractError, match=Status.MISSING_PROVENANCE.value):
        validate_runtime_policy(policy)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_partial_state_is_rejected_without_mutation(value):
    planner = PlannerMock(Identity("C4"))
    before = planner.snapshot()
    with pytest.raises(ContractError, match=Status.NON_FINITE.value):
        planner.step(0, b"recent", struct.pack("<f", value), struct.pack("<f", 1.0))
    assert planner.snapshot() == before
