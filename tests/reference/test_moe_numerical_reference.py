# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import json
from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.moe_numerical import (
    MOE_PHASES,
    MoECombineInputs,
    MoEContractError,
    MoENumericalContract,
)
from rl_engine.reference.moe_numerical import (
    combine_accuracy_reference,
    combine_fixed_tree,
    route_accuracy_reference,
    route_fixed_tree,
)
from rl_engine.validation.moe_numerical.phase_contract import MoEPhaseOutput, compare_moe_phases


def _case():
    contract = MoENumericalContract(3, 2, 2, input_dtype="float32", output_dtype="float32")
    # Physical order deliberately differs from token/slot order; expert 2 is empty.
    inputs = MoECombineInputs(
        num_tokens=2,
        expert_outputs=torch.tensor([[10.0, 20.0], [2.0, 4.0], [30.0, 40.0], [6.0, 8.0]]),
        token_indices=torch.tensor([1, 0, 1, 0]),
        slot_indices=torch.tensor([0, 0, 1, 1]),
        expert_indices=torch.tensor([0, 1, 1, 0]),
        routing_expert_ids=torch.tensor([[1, 0], [0, 1]]),
        routing_weights=torch.tensor([[0.25, 0.75], [0.5, 1.5]]),
        mapping_run_id="run-1",
    )
    return contract, inputs


def _combine(contract, inputs):
    return combine_fixed_tree(contract, inputs, expected_mapping_run_id="run-1")


def _accuracy(contract, inputs):
    return combine_accuracy_reference(contract, inputs, expected_mapping_run_id="run-1")


def _observations(contract, output):
    return [
        MoEPhaseOutput(
            phase, output.clone(), contract.fingerprint, "run-1", "weights-1", "cpu.fixture", "abc"
        )
        for phase in MOE_PHASES
    ]


def _compare(contract, inputs, observations):
    return compare_moe_phases(
        contract,
        inputs,
        observations,
        expected_mapping_run_id="run-1",
        expected_parameter_version="weights-1",
    )


def test_routing_ties_scaling_normalization_and_independent_oracle():
    contract = MoENumericalContract(4, 2, 2, input_dtype="float32", routing_scale=2)
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 3.0, 3.0, -5.0]])
    result = route_fixed_tree(contract, logits)
    assert result.expert_ids.tolist() == [[0, 1], [1, 2]]
    assert torch.equal(result.weights, torch.ones(2, 2))
    accuracy = route_accuracy_reference(contract, logits)
    assert torch.equal(accuracy.expert_ids, result.expert_ids)
    torch.testing.assert_close(result.weights.double(), accuracy.weights, atol=1e-6, rtol=1e-6)
    unnormalized = route_fixed_tree(replace(contract, normalize_selected_weights=False), logits)
    assert torch.equal(unnormalized.weights[0], torch.tensor([0.5, 0.5]))
    assert bool((unnormalized.weights[1] < 1).all())
    # No gradient path is tested by comparing a result with itself.
    variable = torch.tensor([[1.0, 0.0, -1.0, -2.0]], requires_grad=True)
    weighted = route_fixed_tree(contract, variable).weights
    weighted[0, 0].backward()
    assert variable.grad is not None and bool(torch.isfinite(variable.grad).all())
    assert variable.grad[0, 0] > 0 and variable.grad[0, 1] < 0


def test_routing_token_chunking_and_recompute_preserve_bytes():
    contract = MoENumericalContract(5, 3, 2, input_dtype="float32")
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0], [0.0, -0.0, 0.0, 1.0, 2.0]])
    batched = route_fixed_tree(contract, logits)
    chunks = [route_fixed_tree(contract, logits[i : i + 1]) for i in range(2)]
    assert torch.equal(batched.expert_ids, torch.cat([part.expert_ids for part in chunks]))
    assert torch.equal(
        batched.weights.view(torch.uint8),
        torch.cat([part.weights for part in chunks]).view(torch.uint8),
    )
    assert torch.equal(
        batched.weights.view(torch.uint8),
        route_fixed_tree(contract, logits).weights.view(torch.uint8),
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_router_rejects_non_finite_values(bad):
    contract = MoENumericalContract(2, 1, 1, input_dtype="float32")
    with pytest.raises(MoEContractError, match="finite"):
        route_fixed_tree(contract, torch.tensor([[bad, 0.0]]))


def test_router_handles_extreme_finite_logits_and_empty_accuracy_domain():
    contract = MoENumericalContract(3, 2, 1, input_dtype="float32")
    maximum = torch.finfo(torch.float32).max
    logits = torch.tensor([[maximum, -maximum, 0.0]])
    fixed = route_fixed_tree(contract, logits)
    accurate = route_accuracy_reference(contract, logits)
    assert fixed.expert_ids.tolist() == [[0, 2]]
    assert torch.equal(fixed.expert_ids, accurate.expert_ids)
    assert fixed.weights.tolist() == [[1.0, 0.0]]
    assert torch.equal(fixed.weights.double(), accurate.weights)
    empty = route_accuracy_reference(contract, torch.empty(0, 3))
    assert empty.expert_ids.shape == empty.weights.shape == (0, 2)


def test_combine_nonuniform_weights_mapping_and_arrival_order():
    contract, inputs = _case()
    expected = torch.tensor([[5.0, 7.0], [50.0, 70.0]])
    assert torch.equal(_combine(contract, inputs), expected)
    assert torch.equal(_accuracy(contract, inputs), expected.double())
    permutation = torch.tensor([3, 2, 0, 1])
    permuted = replace(
        inputs,
        **{
            name: getattr(inputs, name)[permutation]
            for name in ("expert_outputs", "token_indices", "slot_indices", "expert_indices")
        },
    )
    assert torch.equal(_combine(contract, permuted).view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(_combine(contract, inputs).view(torch.uint8), expected.view(torch.uint8))
    # Non-contiguous input storage must preserve logical mapping and leave sources unchanged.
    storage = torch.zeros(4, 4)
    storage[:, ::2] = inputs.expert_outputs
    before = storage.clone()
    assert torch.equal(
        _combine(contract, replace(inputs, expert_outputs=storage[:, ::2])), expected
    )
    assert torch.equal(storage, before)


def test_preweighted_rows_are_not_weighted_twice():
    contract, inputs = _case()
    weighted = inputs.expert_outputs * inputs.routing_weights[
        inputs.token_indices, inputs.slot_indices
    ].unsqueeze(1)
    changed = replace(inputs, expert_outputs=weighted)
    assert torch.equal(
        _combine(replace(contract, expert_weight_applied=True), changed),
        torch.tensor([[5.0, 7.0], [50.0, 70.0]]),
    )
    assert not torch.equal(_combine(contract, changed), torch.tensor([[5.0, 7.0], [50.0, 70.0]]))


def test_combine_token_chunking_preserves_bytes():
    contract, inputs = _case()
    outputs = []
    for token in range(inputs.num_tokens):
        selected = inputs.token_indices == token
        single = replace(
            inputs,
            num_tokens=1,
            expert_outputs=inputs.expert_outputs[selected],
            token_indices=inputs.token_indices[selected] - token,
            slot_indices=inputs.slot_indices[selected],
            expert_indices=inputs.expert_indices[selected],
            routing_expert_ids=inputs.routing_expert_ids[token : token + 1],
            routing_weights=inputs.routing_weights[token : token + 1],
        )
        outputs.append(_combine(contract, single))
    assert torch.equal(
        torch.cat(outputs).view(torch.uint8), _combine(contract, inputs).view(torch.uint8)
    )


@pytest.mark.parametrize("stage", ["product", "output_cast"])
def test_combine_rejects_arithmetic_overflow(stage):
    contract, inputs = _case()
    if stage == "product":
        inputs = replace(
            inputs,
            expert_outputs=torch.full((4, 2), torch.finfo(torch.float32).max),
            routing_weights=torch.full((2, 2), 2.0),
        )
    else:
        contract = replace(contract, output_dtype="float16")
        inputs = replace(inputs, expert_outputs=torch.full((4, 2), 65504.0))
    with pytest.raises(MoEContractError, match="non-finite output"):
        _combine(contract, inputs)
    # The independent FP64 oracle stays finite and exposes the lost range.
    assert bool(torch.isfinite(_accuracy(contract, inputs)).all())


def test_tree_grouping_is_not_sequential_and_accuracy_is_independent():
    contract = MoENumericalContract(4, 4, 1, input_dtype="float32", output_dtype="float32")
    values = torch.tensor([[2.0**24], [1.0], [-(2.0**24)], [1.0]])
    inputs = MoECombineInputs(
        1,
        values,
        torch.zeros(4, dtype=torch.int64),
        torch.arange(4),
        torch.arange(4),
        torch.arange(4).reshape(1, 4),
        torch.ones(1, 4),
        "run-1",
    )
    result = _combine(contract, inputs)
    sequential = ((values[0] + values[1]) + values[2]) + values[3]
    assert result.item() == 1.0  # (2**24 + 1) + (-2**24 + 1)
    # Use a different tree that pairs the large cancelling leaves first.
    reordered = (values[0] + values[2]) + (values[1] + values[3])
    assert reordered.item() == 2.0
    assert _accuracy(contract, inputs).item() == 2.0
    # A separate fixture discriminates the declared tree from left-fold addition.
    values2 = torch.tensor([[2.0**24], [1.0], [1.0], [-(2.0**24)]])
    result2 = _combine(contract, replace(inputs, expert_outputs=values2))
    sequential2 = ((values2[0] + values2[1]) + values2[2]) + values2[3]
    assert result2.item() == 1.0 and sequential2.item() == 0.0
    assert sequential.item() == 1.0


def test_odd_tree_and_shared_residual_order():
    contract = MoENumericalContract(
        3,
        3,
        1,
        input_dtype="float32",
        output_dtype="float32",
        shared_expert_mode="separate",
        residual_mode="external",
    )
    inputs = MoECombineInputs(
        1,
        torch.tensor([[2.0**24], [0.0], [0.0]]),
        torch.zeros(3, dtype=torch.int64),
        torch.arange(3),
        torch.arange(3),
        torch.arange(3).reshape(1, 3),
        torch.ones(1, 3),
        "run-1",
        shared_output=torch.ones(1, 1),
        residual=torch.tensor([[-(2.0**24)]]),
    )
    assert _combine(contract, inputs).item() == 0
    assert _accuracy(contract, inputs).item() == 1
    assert (torch.tensor(2.0**24) + inputs.residual + inputs.shared_output).item() == 1


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_multiply_in_fp32_and_cast_only_at_final_write(dtype):
    # Two individually rounded-away half-ULP contributions cross the midpoint together.
    contract = MoENumericalContract(2, 2, 1, input_dtype=dtype, output_dtype=dtype)
    unit = torch.tensor(1.0, dtype=getattr(torch, dtype))
    eps = torch.finfo(unit.dtype).eps
    inputs = MoECombineInputs(
        1,
        torch.tensor([[1.0], [eps * 0.375]], dtype=unit.dtype),
        torch.zeros(2, dtype=torch.int64),
        torch.arange(2),
        torch.arange(2),
        torch.arange(2).reshape(1, 2),
        torch.tensor([[1.0 + eps * 0.375, 1.0]]),
        "run-1",
    )
    output = _combine(contract, inputs)
    expected = (inputs.expert_outputs.float().flatten() * inputs.routing_weights.flatten()).sum()
    assert torch.equal(output, expected.to(unit.dtype).reshape(1, 1))
    if dtype != "float32":
        early = (inputs.expert_outputs.float().flatten() * inputs.routing_weights.flatten()).to(
            unit.dtype
        )
        assert output.item() == 1.0 + eps and early.sum().item() == 1.0


@pytest.mark.parametrize(
    "change, message",
    [
        ({"mapping_run_id": "old"}, "mapping_run_id"),
        ({"num_tokens": True}, "num_tokens"),
        ({"expert_outputs": torch.zeros(3, 2)}, "shape"),
        ({"token_indices": torch.tensor([1, 0, 2, 0])}, "out of range"),
        ({"slot_indices": torch.tensor([0, 0, -1, 1])}, "out of range"),
        ({"slot_indices": torch.tensor([0, 0, 0, 1])}, "exactly once"),
        ({"expert_indices": torch.tensor([1, 1, 1, 0])}, "canonical routing"),
        ({"routing_expert_ids": torch.tensor([[1, 1], [0, 1]])}, "distinct"),
        ({"routing_weights": torch.ones(2, 2, dtype=torch.float64)}, "float32"),
        ({"routing_weights": torch.full((2, 2), -1.0)}, "non-negative"),
        ({"routing_weights": torch.full((2, 2), float("nan"))}, "finite"),
        ({"token_indices": torch.zeros(4, dtype=torch.int32)}, "int64"),
        ({"expert_outputs": torch.empty(4, 2, device="meta")}, "CPU"),
        ({"slot_indices": torch.empty(4, dtype=torch.int64, device="meta")}, "materialized"),
        ({"residual": torch.ones(2, 2)}, "must be omitted"),
        ({"shared_output": torch.ones(2, 2)}, "must be omitted"),
    ],
)
def test_mapping_validation_rejects_malformed_inputs(change, message):
    contract, inputs = _case()
    with pytest.raises(MoEContractError, match=message):
        _combine(contract, replace(inputs, **change))


@pytest.mark.parametrize(
    "branch, mode", [("shared_output", "shared_expert_mode"), ("residual", "residual_mode")]
)
def test_branch_modes_require_individually_valid_token_shapes(branch, mode):
    contract, inputs = _case()
    contract = replace(contract, **{mode: "separate" if branch == "shared_output" else "external"})
    with pytest.raises(MoEContractError, match="required"):
        _combine(contract, inputs)
    for shape in ((), (2, 3), (3, 2)):
        with pytest.raises(MoEContractError, match="shape"):
            _combine(contract, replace(inputs, **{branch: torch.ones(shape)}))


def test_empty_execution_is_valid_but_not_acceptance_evidence():
    contract, inputs = _case()
    empty = replace(
        inputs,
        num_tokens=0,
        expert_outputs=inputs.expert_outputs[:0],
        token_indices=inputs.token_indices[:0],
        slot_indices=inputs.slot_indices[:0],
        expert_indices=inputs.expert_indices[:0],
        routing_weights=inputs.routing_weights[:0],
        routing_expert_ids=inputs.routing_expert_ids[:0],
    )
    assert _combine(contract, empty).shape == (0, 2)
    assert _accuracy(contract, empty).shape == (0, 2)
    assert route_fixed_tree(contract, torch.empty(0, 3)).weights.shape == (0, 2)
    with pytest.raises(MoEContractError, match="empty samples"):
        _compare(contract, empty, _observations(contract, torch.empty(0, 2)))


def test_phase_evidence_checks_tensors_not_only_matching_contract_ids():
    contract, inputs = _case()
    observations = _observations(contract, torch.tensor([[5.0, 7.0], [50.0, 70.0]]))
    report = _compare(contract, inputs, observations)
    assert report["passed"] and report["mapping_run_id_verified"]
    assert report["forward_invariance_passed"]
    assert report["forward_accuracy_passed"]
    assert not report["device_execution_verified"]
    assert not report["parameter_content_verified"]
    observations[-1].output[1, 0] += 1
    changed = _compare(contract, inputs, observations)
    assert not changed["passed"]
    assert changed["phases"]["verify"]["first_bit_mismatch"] == [1, 0]
    assert changed["phases"]["verify"]["bit_mismatch_count"] == 1
    assert changed["phases"]["verify"]["accuracy_mismatch_count"] == 1
    assert json.dumps(changed, allow_nan=False)


def test_phase_report_separates_fixed_tree_and_accuracy_verdicts():
    contract = MoENumericalContract(4, 4, 1, input_dtype="float32", output_dtype="float32")
    inputs = MoECombineInputs(
        num_tokens=1,
        expert_outputs=torch.tensor([[2**24], [1.0], [-(2**24)], [1.0]]),
        token_indices=torch.zeros(4, dtype=torch.int64),
        slot_indices=torch.arange(4, dtype=torch.int64),
        expert_indices=torch.arange(4, dtype=torch.int64),
        routing_expert_ids=torch.arange(4, dtype=torch.int64).reshape(1, 4),
        routing_weights=torch.ones(1, 4),
        mapping_run_id="run-1",
    )
    fixed = _combine(contract, inputs)
    report = _compare(contract, inputs, _observations(contract, fixed))

    assert fixed.item() == 1.0
    assert report["forward_invariance_passed"]
    assert not report["forward_accuracy_passed"]
    assert report["passed_policy"] == "forward_invariance_and_accuracy"
    assert not report["passed"]


@pytest.mark.parametrize("field", ["contract_fingerprint", "mapping_run_id", "parameter_version"])
def test_phase_evidence_rejects_stale_rules_runs_and_parameters(field):
    contract, inputs = _case()
    observations = _observations(contract, torch.zeros(2, 2))
    observations[1] = replace(observations[1], **{field: "stale"})
    with pytest.raises(MoEContractError, match=field):
        _compare(contract, inputs, observations)


def test_phase_evidence_rejects_missing_duplicate_phases_and_uses_honest_counts():
    contract, inputs = _case()
    observations = _observations(contract, torch.tensor([[5.0, 7.0], [50.0, 70.0]]))
    for wrong in (observations[:3], [observations[0]] * 4):
        with pytest.raises(MoEContractError, match="exactly one"):
            _compare(contract, inputs, wrong)
    observations[1] = replace(observations[1], output=torch.zeros(4))
    result = _compare(contract, inputs, observations)["phases"]["prefill"]
    assert result["metadata_errors"] == ["shape_mismatch"]
    assert result["compared_count"] == result["bit_mismatch_count"] == 0
    assert not result["passed"]


def test_phase_dtype_mismatch_is_metadata_failure_without_element_counts():
    contract, inputs = _case()
    observations = _observations(contract, torch.tensor([[5.0, 7.0], [50.0, 70.0]]))
    observations[0] = replace(observations[0], output=observations[0].output.double())
    result = _compare(contract, inputs, observations)["phases"]["train"]
    assert result["metadata_errors"] == ["dtype_mismatch"]
    assert result["compared_count"] == result["accuracy_mismatch_count"] == 0
    assert not result["passed"]


def test_phase_non_tensor_output_is_structured_metadata_failure():
    contract, inputs = _case()
    observations = _observations(contract, torch.tensor([[5.0, 7.0], [50.0, 70.0]]))
    observations[0] = replace(observations[0], output="not-a-tensor")

    result = _compare(contract, inputs, observations)

    phase = result["phases"]["train"]
    assert phase["metadata_errors"] == ["non_tensor_output"]
    assert phase["compared_count"] == 0
    assert not result["forward_invariance_passed"]
    assert not result["forward_accuracy_passed"]
    assert not result["passed"]


def test_phase_evidence_distinguishes_one_ulp_and_nonfinite_values():
    contract, inputs = _case()
    observations = _observations(contract, torch.tensor([[5.0, 7.0], [50.0, 70.0]]))
    observations[0].output[0, 0] = torch.nextafter(torch.tensor(5.0), torch.tensor(6.0))
    report = _compare(contract, inputs, observations)
    assert not report["passed"]
    assert report["phases"]["train"]["accuracy_passed"]
    assert not report["phases"]["train"]["fixed_tree_passed"]
    observations[0].output[0, 0] = float("nan")
    report = _compare(contract, inputs, observations)
    assert not report["phases"]["train"]["accuracy_passed"]
    assert json.dumps(report, allow_nan=False)


def test_signed_zero_leaf_is_preserved_and_checked_as_bytes():
    contract = MoENumericalContract(1, 1, 1, input_dtype="float32", output_dtype="float32")
    inputs = MoECombineInputs(
        1,
        torch.tensor([[-0.0]]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([[0]]),
        torch.ones(1, 1),
        "run-1",
    )
    assert torch.signbit(_combine(contract, inputs)).item()
    report = _compare(contract, inputs, _observations(contract, torch.zeros(1, 1)))
    assert not report["passed"]
    assert report["phases"]["train"]["accuracy_passed"]
    assert report["phases"]["train"]["bit_mismatch_count"] == 1


def test_signed_zero_odd_tree_carries_leaf_without_padding():
    contract = MoENumericalContract(3, 3, 1, input_dtype="float32", output_dtype="float32")
    inputs = MoECombineInputs(
        1,
        torch.tensor([[-0.0], [-0.0], [-0.0]]),
        torch.zeros(3, dtype=torch.int64),
        torch.arange(3, dtype=torch.int64),
        torch.arange(3, dtype=torch.int64),
        torch.arange(3, dtype=torch.int64).reshape(1, 3),
        torch.ones(1, 3),
        "run-1",
    )

    fixed = _combine(contract, inputs)
    report = _compare(contract, inputs, _observations(contract, torch.zeros(1, 1)))

    assert torch.signbit(fixed).item()
    assert not report["passed"]
    assert report["phases"]["train"]["accuracy_passed"]
    assert not report["phases"]["train"]["fixed_tree_passed"]
    assert report["phases"]["train"]["bit_mismatch_count"] == 1
