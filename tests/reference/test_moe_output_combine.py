# SPDX-License-Identifier: Apache-2.0
"""P6 contracts, request preparation, inverse mapping and explicit dispatch."""

from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.reference.moe import (
    combine_local_input_gradients,
    inverse_map_expert_output,
    inverse_mapping_provenance,
    local_input_gradient_provenance,
    merge_residual_gradient_after_rms,
    prepare_output_combine,
)


def _plan(**kwargs):
    values = {"top_k": 2, "hidden_size": 4}
    values.update(kwargs)
    return MoEOutputCombinePlan(**values)


def _inputs(plan, **kwargs):
    values = {
        "expert_output": torch.zeros(4, plan.hidden_size, dtype=torch.bfloat16),
        "token_indices": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        "expert_indices": torch.tensor([2, 3, 2, 3], dtype=torch.int64),
        "slot_indices": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        "routing_weights": torch.ones(4, dtype=torch.float32),
    }
    values.update(kwargs)
    return MoEOutputCombineInputs(**values)


def test_plan_has_stable_fingerprint_and_fixed_slot_order():
    first = _plan()
    second = _plan()

    assert first.merge_order == (0, 1)
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["accumulation_dtype"] == "float32"


def test_plan_freezes_nested_metadata_without_reordering_sequences():
    plan = _plan(metadata={"ordered": ["second", "first"], "nested": (({"value": 1},),)})

    assert plan.to_dict()["metadata"]["ordered"] == ["second", "first"]
    with pytest.raises(TypeError):
        plan.metadata["nested"][0][0]["value"] = 2


def test_plan_rejects_bool_dimensions_and_mutable_metadata():
    with pytest.raises(MoEOutputCombineContractError):
        _plan(top_k=True)
    plan = _plan(metadata={"aux": [torch.bfloat16]})
    fingerprint = plan.fingerprint
    assert plan.fingerprint == fingerprint
    with pytest.raises(TypeError):
        plan.metadata["new"] = "value"
    with pytest.raises(MoEOutputCombineContractError):
        _plan(expert_weight_applied=1)
    with pytest.raises(MoEOutputCombineContractError):
        _plan(merge_order=(1.0, 0.0))


def test_plan_rejects_non_fixed_merge_order():
    with pytest.raises(MoEOutputCombineContractError, match="merge_order"):
        _plan(merge_order=(1, 0))
    with pytest.raises(MoEOutputCombineContractError, match="merge_order must contain integer"):
        _plan(merge_order=(1.0, 0.0))


def test_plan_rejects_unsupported_output_dtype():
    with pytest.raises(MoEOutputCombineContractError, match="output_dtype"):
        _plan(output_dtype="float8_e4m3fn")


def test_inputs_validate_against_plan():
    plan = _plan()
    inputs = _inputs(plan)

    inputs.validate(plan)
    provenance = inputs.provenance(plan)
    assert provenance["semantic_op"] == "moe_output_combine"
    assert provenance["plan_fingerprint"] == plan.fingerprint
    assert provenance["mapping_run_id_verified"] is False


def test_weights_must_be_explicitly_owned_by_one_boundary():
    plan = _plan(expert_weight_applied=True)
    inputs = _inputs(plan)

    with pytest.raises(MoEOutputCombineContractError, match="already applied"):
        inputs.validate(plan)


def test_schema_and_shape_mismatches_fail_closed():
    plan = _plan()
    inputs = _inputs(plan, mapping_schema_version="p4-token-slot.v0")

    with pytest.raises(MoEOutputCombineContractError, match="mapping schema"):
        inputs.validate(plan)
    with pytest.raises(
        MoEOutputCombineContractError, match="one float32 value per expert output slot"
    ):
        _inputs(plan, routing_weights=torch.ones(3)).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="slot_indices must be within top_k"):
        _inputs(plan, slot_indices=torch.tensor([0, 1, 0, 2])).validate(plan)


def test_auxiliary_outputs_require_explicit_token_count():
    plan = _plan(shared_expert_mode="separate")
    base = {
        "expert_output": torch.zeros(6, plan.hidden_size, dtype=torch.bfloat16),
        "token_indices": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        "expert_indices": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
        "slot_indices": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
        "routing_weights": torch.ones(6, dtype=torch.float32),
    }
    valid = _inputs(
        plan,
        **base,
        num_tokens=3,
        shared_output=torch.zeros(3, plan.hidden_size, dtype=torch.bfloat16),
    )
    valid.validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="shared_output must have shape"):
        _inputs(
            plan,
            **base,
            num_tokens=3,
            shared_output=torch.zeros(4, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="num_tokens is required"):
        _inputs(
            plan,
            **base,
            shared_output=torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)


def test_declared_token_count_requires_complete_slot_domain():
    plan = _plan()
    with pytest.raises(
        MoEOutputCombineContractError,
        match=r"slot count must equal num_tokens \* top_k",
    ):
        _inputs(plan, num_tokens=3).validate(plan)


def test_external_residual_uses_declared_token_shape():
    plan = _plan(residual_mode="external")
    base = {
        "expert_output": torch.zeros(6, plan.hidden_size, dtype=torch.bfloat16),
        "token_indices": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        "expert_indices": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
        "slot_indices": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
        "routing_weights": torch.ones(6, dtype=torch.float32),
    }
    valid = _inputs(
        plan,
        **base,
        num_tokens=3,
        residual=torch.zeros(3, plan.hidden_size, dtype=torch.bfloat16),
    )
    valid.validate(plan)

    with pytest.raises(MoEOutputCombineContractError, match="residual must have shape"):
        _inputs(
            plan,
            **base,
            num_tokens=3,
            residual=torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="residual is required"):
        _inputs(plan, **base, num_tokens=3).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="residual must be omitted"):
        excluded = _plan()
        _inputs(
            excluded,
            residual=torch.zeros(2, excluded.hidden_size, dtype=torch.bfloat16),
        ).validate(excluded)


def test_prepare_output_combine_returns_versioned_provenance():
    plan = MoEOutputCombinePlan(top_k=1, hidden_size=3)
    inputs = MoEOutputCombineInputs(
        expert_output=torch.zeros(2, 3, dtype=torch.bfloat16),
        token_indices=torch.tensor([0, 1], dtype=torch.int64),
        expert_indices=torch.tensor([4, 4], dtype=torch.int64),
        slot_indices=torch.tensor([0, 0], dtype=torch.int64),
        routing_weights=torch.ones(2, dtype=torch.float32),
        producer_backend="test.reference",
        producer_build_fingerprint="test-build-v1",
        num_tokens=2,
    )

    prepared = prepare_output_combine(plan, inputs)

    assert prepared.provenance["semantic_op"] == "moe_output_combine"
    assert prepared.provenance["producer_backend"] == "test.reference"
    assert prepared.provenance["producer_build_fingerprint"] == "test-build-v1"
    assert prepared.provenance["expert_output_stride"] == [3, 1]
    assert prepared.provenance["num_tokens"] == 2


def test_reference_preparation_requires_explicit_token_count():
    plan = MoEOutputCombinePlan(top_k=1, hidden_size=3)
    inputs = MoEOutputCombineInputs(
        expert_output=torch.zeros(2, 3, dtype=torch.bfloat16),
        token_indices=torch.tensor([0, 1], dtype=torch.int64),
        expert_indices=torch.tensor([4, 4], dtype=torch.int64),
        slot_indices=torch.tensor([0, 0], dtype=torch.int64),
        routing_weights=torch.ones(2, dtype=torch.float32),
    )

    with pytest.raises(MoEOutputCombineContractError, match="explicit num_tokens"):
        prepare_output_combine(plan, inputs)


def test_reference_entry_points_reject_non_cpu_inputs():
    plan = _plan()
    meta = torch.empty(4, plan.hidden_size, dtype=torch.bfloat16, device="meta")
    inputs = _inputs(
        plan,
        expert_output=meta,
        token_indices=torch.zeros(4, dtype=torch.int64, device="meta"),
        expert_indices=torch.zeros(4, dtype=torch.int64, device="meta"),
        slot_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="meta"),
        routing_weights=torch.ones(4, dtype=torch.float32, device="meta"),
    )

    with pytest.raises(MoEOutputCombineContractError, match="on CPU"):
        prepare_output_combine(plan, inputs)
    with pytest.raises(MoEOutputCombineContractError, match="meta tensors"):
        inputs.validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="on CPU"):
        inverse_map_expert_output(plan, inputs, num_tokens=2)


def _request(
    *, index_dtype=torch.int64, expert_output=None, tokens=None, slots=None, run_id="run-1"
):
    plan = MoEOutputCombinePlan(
        top_k=2,
        hidden_size=3,
        token_index_dtype=str(index_dtype).removeprefix("torch."),
    )
    if expert_output is None:
        expert_output = torch.tensor(
            [[20, 21, 22], [0, 1, 2], [30, 31, 32], [10, 11, 12]],
            dtype=torch.bfloat16,
        )
    if tokens is None:
        tokens = [1, 0, 1, 0]
    if slots is None:
        slots = [0, 0, 1, 1]
    inputs = MoEOutputCombineInputs(
        expert_output=expert_output,
        token_indices=torch.tensor(tokens, dtype=index_dtype),
        expert_indices=torch.tensor([4, 4, 5, 5], dtype=index_dtype),
        slot_indices=torch.tensor(slots, dtype=index_dtype),
        routing_weights=torch.ones(4, dtype=torch.float32),
        mapping_run_id=run_id,
    )
    return plan, inputs


def test_inverse_mapping_restores_token_major_slot_order():
    plan, inputs = _request()

    restored = inverse_map_expert_output(plan, inputs, num_tokens=2, num_experts=8)

    expected = torch.tensor(
        [[[0, 1, 2], [10, 11, 12]], [[20, 21, 22], [30, 31, 32]]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(restored, expected)


def test_inverse_mapping_supports_non_contiguous_expert_output_and_int32_indices():
    base = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    expert_output = base.t()
    plan, inputs = _request(
        index_dtype=torch.int32,
        expert_output=expert_output,
        tokens=[0, 1, 0, 1],
        slots=[0, 0, 1, 1],
    )

    restored = inverse_map_expert_output(plan, inputs, num_tokens=2, num_experts=8)

    assert restored.shape == (2, 2, 3)
    assert restored.is_contiguous()
    expected = torch.tensor(
        [
            [[0, 4, 8], [2, 6, 10]],
            [[1, 5, 9], [3, 7, 11]],
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(restored, expected)


@pytest.mark.parametrize(
    ("tokens", "slots", "num_tokens", "message"),
    [
        ([0, 0, 0, 1], [0, 0, 1, 1], 2, "pairs must be unique"),
        ([0, 0, 1, 1], [0, 1, 0, 1], 3, "slot count"),
        ([0, 0, 1, 2], [0, 1, 0, 1], 2, "outside"),
    ],
)
def test_inverse_mapping_rejects_invalid_mapping(tokens, slots, num_tokens, message):
    plan, inputs = _request(tokens=tokens, slots=slots)

    with pytest.raises(MoEOutputCombineContractError, match=message):
        inverse_map_expert_output(plan, inputs, num_tokens=num_tokens)


def test_inverse_mapping_provenance_records_canonical_layout():
    plan, inputs = _request()

    provenance = inverse_mapping_provenance(
        plan, inputs, num_tokens=2, num_experts=8, expected_mapping_run_id="run-1"
    )

    assert provenance["inverse_mapping"] == "token_major_slot_minor"
    assert provenance["restored_shape"] == [2, 2, 3]
    assert provenance["mapping_run_id_verified"] is True


def test_inverse_mapping_rejects_wrong_run_id_and_expert_overflow():
    plan, inputs = _request()

    with pytest.raises(MoEOutputCombineContractError, match="run id"):
        inverse_map_expert_output(plan, inputs, num_tokens=2, expected_mapping_run_id="other-run")

    bad_expert = inputs.expert_indices.clone()
    bad_expert[0] = 8
    bad_inputs = replace(inputs, expert_indices=bad_expert)
    with pytest.raises(MoEOutputCombineContractError, match="expert index"):
        inverse_map_expert_output(plan, bad_inputs, num_tokens=2, num_experts=8)


def test_inverse_mapping_rejects_boolean_counts():
    plan, inputs = _request()
    with pytest.raises(MoEOutputCombineContractError, match="num_tokens"):
        inverse_map_expert_output(plan, inputs, num_tokens=True)
    with pytest.raises(MoEOutputCombineContractError, match="num_experts"):
        inverse_map_expert_output(plan, inputs, num_tokens=2, num_experts=True)


def test_inverse_mapping_matches_declared_input_token_count():
    plan, inputs = _request()
    inputs = inputs.__class__(
        expert_output=inputs.expert_output,
        token_indices=inputs.token_indices,
        expert_indices=inputs.expert_indices,
        slot_indices=inputs.slot_indices,
        routing_weights=inputs.routing_weights,
        mapping_run_id=inputs.mapping_run_id,
        num_tokens=2,
    )
    with pytest.raises(MoEOutputCombineContractError, match="disagrees"):
        inverse_map_expert_output(plan, inputs, num_tokens=3)


def _gradient_request(*, shared=False, residual=False, run_id="run-1"):
    plan = _plan(
        shared_expert_mode="separate" if shared else "excluded",
        residual_mode="external" if residual else "excluded",
    )
    inputs = _inputs(
        plan,
        mapping_run_id=run_id,
        num_tokens=2,
        shared_output=(torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16) if shared else None),
        residual=(torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16) if residual else None),
    )
    return plan, inputs


def test_local_input_gradients_stop_before_rms_and_preserve_fixed_order():
    plan, inputs = _gradient_request(shared=True, residual=True)
    expert_gradient = torch.tensor(
        [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]],
        dtype=torch.bfloat16,
    )
    shared_gradient = torch.full((2, 4), 0.5, dtype=torch.bfloat16)
    gate_gradient = torch.ones(2, 4, dtype=torch.float32)

    d_x_norm = combine_local_input_gradients(
        plan,
        inputs,
        expert_gradient,
        shared_expert_gradient=shared_gradient,
        gate_gradient=gate_gradient,
        num_tokens=2,
    )

    expected = torch.tensor([[5.5, 7.5, 9.5, 11.5], [13.5, 15.5, 17.5, 19.5]])
    assert torch.equal(d_x_norm, expected.to(torch.bfloat16))
    after_rms = merge_residual_gradient_after_rms(
        plan,
        d_x_norm,
        torch.full((2, 4), 2.0, dtype=torch.bfloat16),
    )
    assert torch.equal(after_rms, (expected + 2).to(torch.bfloat16))


def test_local_input_gradients_use_order_sensitive_values_and_recompute():
    plan, inputs = _gradient_request(shared=True)
    expert_gradient = torch.tensor(
        [[2**24, 0, 0, 0], [0, 0, 0, 0], [2**24, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.bfloat16,
    )
    shared_gradient = torch.ones(2, 4, dtype=torch.bfloat16)
    gate_gradient = torch.full((2, 4), -(2**24), dtype=torch.bfloat16)

    first = combine_local_input_gradients(
        plan,
        inputs,
        expert_gradient,
        shared_expert_gradient=shared_gradient,
        gate_gradient=gate_gradient,
        num_tokens=2,
    )
    second = combine_local_input_gradients(
        plan,
        inputs,
        expert_gradient,
        shared_expert_gradient=shared_gradient,
        gate_gradient=gate_gradient,
        num_tokens=2,
    )

    assert torch.equal(first, second)
    assert torch.equal(first[:, 0], torch.zeros(2, dtype=torch.bfloat16))


def test_local_input_gradients_apply_nonuniform_routing_weights():
    plan, inputs = _gradient_request()
    inputs = replace(
        inputs,
        routing_weights=torch.tensor([0.5, 2.0, 1.5, 0.25], dtype=torch.float32),
    )
    result = combine_local_input_gradients(
        plan,
        inputs,
        torch.ones(4, plan.hidden_size, dtype=torch.bfloat16),
        num_tokens=2,
    )
    expected = torch.tensor(
        [[2.5] * plan.hidden_size, [1.75] * plan.hidden_size], dtype=torch.bfloat16
    )
    assert torch.equal(result, expected)


def test_local_input_gradients_enforce_modes_and_mapping_run_id():
    plan, inputs = _gradient_request()
    expert_gradient = torch.ones(4, plan.hidden_size, dtype=torch.bfloat16)
    with pytest.raises(
        MoEOutputCombineContractError, match="shared_expert_gradient must be omitted"
    ):
        combine_local_input_gradients(
            plan,
            inputs,
            expert_gradient,
            shared_expert_gradient=torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16),
            num_tokens=2,
        )

    shared_plan, shared_inputs = _gradient_request(shared=True)
    with pytest.raises(MoEOutputCombineContractError, match="shared_expert_gradient is required"):
        combine_local_input_gradients(
            shared_plan,
            shared_inputs,
            expert_gradient,
            num_tokens=2,
        )
    with pytest.raises(MoEOutputCombineContractError, match="run id"):
        combine_local_input_gradients(
            shared_plan,
            shared_inputs,
            expert_gradient,
            shared_expert_gradient=torch.zeros(2, shared_plan.hidden_size, dtype=torch.bfloat16),
            expected_mapping_run_id="stale-run",
            num_tokens=2,
        )
    with pytest.raises(MoEOutputCombineContractError, match="residual_mode"):
        merge_residual_gradient_after_rms(
            plan,
            torch.ones(2, plan.hidden_size, dtype=torch.bfloat16),
            torch.ones(2, plan.hidden_size, dtype=torch.bfloat16),
        )

    with pytest.raises(MoEOutputCombineContractError, match="separate tensors"):
        shared_gradient = torch.zeros(2, shared_plan.hidden_size, dtype=torch.bfloat16)
        combine_local_input_gradients(
            shared_plan,
            shared_inputs,
            expert_gradient,
            shared_expert_gradient=shared_gradient,
            gate_gradient=shared_gradient,
            num_tokens=2,
        )


def test_gradient_combine_allows_missing_forward_shared_output():
    plan, inputs = _gradient_request(shared=True)
    result = combine_local_input_gradients(
        plan,
        replace(inputs, shared_output=None),
        torch.ones(4, plan.hidden_size, dtype=torch.bfloat16),
        shared_expert_gradient=torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16),
        num_tokens=2,
    )
    assert result.shape == (2, plan.hidden_size)


def test_local_input_gradient_provenance_records_checks():
    plan, inputs = _gradient_request(shared=True, run_id="run-1")
    provenance = local_input_gradient_provenance(
        plan,
        inputs,
        torch.ones(4, plan.hidden_size, dtype=torch.bfloat16),
        shared_expert_gradient=torch.ones(2, plan.hidden_size, dtype=torch.bfloat16),
        expected_mapping_run_id="run-1",
        num_tokens=2,
    )
    assert provenance["semantic_op"] == "moe_input_gradient_combine"
    assert provenance["producer_backend"] == "unknown"
    assert provenance["num_slots"] == 4
    assert provenance["num_tokens"] == 2
    assert provenance["expert_gradient_dtype"] == "bfloat16"
    assert "expert_output_dtype" not in provenance
    assert provenance["branches"]["gate"]["dtype"] is None
    assert provenance["branches"]["shared_expert"]["dtype"] == "bfloat16"
    assert provenance["routing_weights_applied"] is True
    assert provenance["mapping_run_id_verified"] is True
    assert provenance["merge_order"] == [0, 1]
