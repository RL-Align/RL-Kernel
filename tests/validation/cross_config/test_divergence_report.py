# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json

import pytest
import torch

from rl_engine.validation.cross_config.divergence import (
    compute_mismatch_mask,
    first_divergence,
    runtime_identity,
)


def test_first_divergence_records_token_and_runtime_identity_deterministically():
    expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = expected.clone()
    actual[0, 2] = 3.25
    actual[1, 0] = 9.0

    report = first_divergence(
        expected,
        actual,
        tensor_name="moe.dX_norm",
        axis_names=("token", "hidden"),
        provenance={
            "rank": 1,
            "world_size": 8,
            "plan_version": "p6.v2",
            "actual_backend": "rlkernel.reference",
        },
    )

    assert report.passed is False
    assert report.mismatch_count == 2
    assert report.first is not None
    assert report.first.coordinates == (0, 2)
    assert report.first.to_dict()["coordinate_map"] == {"token": 0, "hidden": 2}
    assert report.first.provenance["actual_backend"] == "rlkernel.reference"
    assert json.dumps(report.to_dict(), sort_keys=True, allow_nan=False)
    assert (
        report.to_dict()
        == first_divergence(
            expected,
            actual,
            tensor_name="moe.dX_norm",
            axis_names=("token", "hidden"),
            provenance={
                "rank": 1,
                "world_size": 8,
                "plan_version": "p6.v2",
                "actual_backend": "rlkernel.reference",
            },
        ).to_dict()
    )


def test_first_divergence_supports_active_tokens_and_metadata_shape_failures():
    expected = torch.zeros((1, 3))
    actual = torch.tensor([[0.0, 1.0, 2.0]])
    report = first_divergence(
        expected,
        actual,
        active_mask=torch.tensor([[False, True, False]]),
        axis_names=("sample", "token"),
    )
    assert report.mismatch_count == 1
    assert report.first is not None
    assert report.first.coordinates == (0, 1)

    shape = first_divergence(torch.zeros((2, 1)), torch.zeros((2, 2)))
    assert shape.first is not None
    assert shape.first.reason == "shape_mismatch"
    assert shape.compared_count == 0
    assert shape.metadata_errors == ("shape_mismatch",)


def test_first_divergence_handles_tolerance_infinities_and_signed_zero():
    within = first_divergence(
        torch.tensor([1.0]),
        torch.tensor([1.0 + 1e-6]),
        atol=1e-5,
    )
    assert within.passed is True
    relative = first_divergence(
        torch.tensor([100.0]),
        torch.tensor([101.0]),
        rtol=0.005,
    )
    assert relative.passed is False
    relative_within = first_divergence(
        torch.tensor([100.0]),
        torch.tensor([101.0]),
        rtol=0.02,
    )
    assert relative_within.passed is True

    infinities = first_divergence(
        torch.tensor([float("-inf"), float("nan")]),
        torch.tensor([float("-inf"), 1.0]),
    )
    assert infinities.mismatch_count == 1
    assert infinities.first is not None
    assert infinities.first.coordinates == (1,)
    assert infinities.first.reason == "non_finite"

    signed_zero = first_divergence(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert signed_zero.passed is False
    assert signed_zero.first is not None
    assert signed_zero.first.reason == "signed_zero"
    with pytest.raises(ValueError, match="does not match tensor comparison"):
        first_divergence(
            torch.zeros(1),
            torch.ones(1),
            atol=1e-5,
            mismatch_mask=torch.zeros(1, dtype=torch.bool),
        )


def test_first_divergence_compares_booleans_exactly():
    expected = torch.tensor([True, False, True])

    assert first_divergence(expected, expected.clone()).passed is True
    report = first_divergence(expected, torch.tensor([True, True, True]))

    assert report.passed is False
    assert report.mismatch_count == 1
    assert report.first is not None
    assert report.first.flat_index == 1
    assert (report.first.expected_value, report.first.actual_value) == (False, True)
    assert report.first.absolute_diff is None
    assert report.first.relative_diff is None
    assert json.dumps(report.to_dict(), sort_keys=True, allow_nan=False)
    assert compute_mismatch_mask(
        expected,
        torch.tensor([False, False, True]),
        torch.ones(3, dtype=torch.bool),
        0.0,
    ).tolist() == [True, False, False]


def test_first_divergence_compares_int64_without_subtraction_overflow():
    int64 = torch.iinfo(torch.int64)
    expected = torch.tensor([-1, int64.min, 2**62], dtype=torch.int64)
    actual = torch.tensor([int64.max, int64.max, 2**62 + 1], dtype=torch.int64)

    # Wrapped int64 subtraction makes the first pair look equal.
    assert torch.abs(actual - expected)[0] <= 0
    report = first_divergence(expected, actual)

    assert report.passed is False
    assert report.mismatch_count == 3
    assert report.first is not None
    assert report.first.flat_index == 0
    assert report.first.absolute_diff == 2**63
    assert first_divergence(expected[2:], actual[2:]).first.absolute_diff == 1
    assert first_divergence(expected, expected.clone()).passed is True


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64])
def test_exact_comparison_rejects_tolerances(dtype):
    value = torch.ones(2, dtype=dtype)

    with pytest.raises(ValueError, match="zero tolerances"):
        first_divergence(value, value.clone(), atol=1.0)
    with pytest.raises(ValueError, match="zero tolerances"):
        first_divergence(value, value.clone(), rtol=0.5)


def test_first_divergence_rejects_vacuous_samples_and_invalid_axis_metadata():
    with pytest.raises(ValueError, match="non-empty"):
        first_divergence(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError, match="at least one active"):
        first_divergence(
            torch.zeros(2),
            torch.zeros(2),
            active_mask=torch.zeros(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="axis_names"):
        first_divergence(torch.zeros(2), torch.zeros(2), axis_names=("token", "slot"))
    with pytest.raises(ValueError, match="missing required fields"):
        runtime_identity({}, model_version="v1", checkpoint_id="c1")
