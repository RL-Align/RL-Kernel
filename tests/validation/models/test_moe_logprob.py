# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from dataclasses import replace

import pytest
import torch

from rl_engine.validation.cross_config.moe_logprob import compare_cross_config_moe_logprobs
from rl_engine.validation.models.moe_logprob import (
    MoELogprobCase,
    MoELogprobError,
    MoELogprobObservation,
    compare_moe_logprobs,
    independently_recompute_logprobs,
)


def _case():
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.5, -2.0]], dtype=torch.float32
    )
    return MoELogprobCase(
        logits,
        torch.tensor([0, 1, 0]),
        torch.tensor([2, 2, 3]),
        torch.tensor([10, 11, 12]),
        torch.tensor([0, 1, 0]),
        torch.tensor([4, 7, 4]),
        torch.tensor([0, 1, 1]),
        "checkpoint-1",
        "weights-1",
        "mapping-1",
        routing_weights=torch.tensor([0.25, 0.75, 1.0]),
    )


def _fixed(case):
    return torch.log_softmax(case.logits, dim=-1).gather(1, case.target_ids[:, None]).flatten()


def _candidate(values):
    return MoELogprobObservation(values, "ascend", "ascend.npu", "impl-1", rank=1, world_size=2)


def _compare(case, candidate_case=None, values=None):
    return compare_moe_logprobs(
        case,
        _fixed(case),
        candidate_case or case,
        _candidate(values if values is not None else _fixed(case)),
        expected_mapping_run_id="mapping-1",
        expected_model_version="weights-1",
    )


def test_independent_recompute_is_repeatable_and_preserves_provenance():
    case = _case()
    first = independently_recompute_logprobs(case)
    second = independently_recompute_logprobs(case)
    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
    provenance = case.provenance()
    assert provenance["routing_weights_present"]
    assert provenance["token_indices_sha256"]
    assert provenance["probability_rule"] == "log_softmax_fp32_selected"


def test_valid_candidate_passes_both_fixed_and_independent_verdicts():
    result = _compare(_case())
    assert result["passed"]
    assert result["fixed_tree_passed"]
    assert result["accuracy_passed"]
    assert result["inputs_preserved"]
    assert result["routing_weights_preserved"]
    assert not result["device_execution_verified"]
    assert result["first_divergence"] is None


def test_accuracy_mismatch_is_reported_when_fixed_reference_matches_bytes():
    case = _case()
    fixed_reference = _fixed(case).clone()
    fixed_reference[0] += 0.1
    result = compare_moe_logprobs(
        case,
        fixed_reference,
        case,
        _candidate(fixed_reference.clone()),
        expected_mapping_run_id="mapping-1",
        expected_model_version="weights-1",
    )
    assert not result["passed"]
    assert result["fixed_tree_passed"]
    assert not result["accuracy_passed"]
    assert result["fixed_mismatch_count"] == 0
    assert result["accuracy_mismatch_count"] == 1
    assert result["first_divergence"]["reason"] == "accuracy_mismatch"
    assert result["first_divergence"]["flat_index"] == 0


def test_cross_config_wrapper_preserves_case_and_attempt_identity():
    case = _case()
    result = compare_cross_config_moe_logprobs(
        case_id="case-1",
        attempt_id="attempt-1",
        reference_case=case,
        reference_values=_fixed(case),
        candidate_case=case,
        candidate=_candidate(_fixed(case)),
        expected_mapping_run_id="mapping-1",
        expected_model_version="weights-1",
    )
    assert result["passed"]
    assert result["cross_config_identity"] == {"case_id": "case-1", "attempt_id": "attempt-1"}


def test_numeric_injection_reports_layer_token_slot_expert_and_rank():
    case = _case()
    changed = _fixed(case).clone()
    changed[1] += 0.01
    result = _compare(case, values=changed)
    assert not result["passed"]
    record = result["first_divergence"]
    assert record["layer"] == 2
    assert record["token"] == 11
    assert record["slot"] == 1
    assert record["expert"] == 7
    assert record["rank"] == 1
    assert record["flat_index"] == 1
    assert record["reason"] == "bit_mismatch"
    assert record["expected"] == float(_fixed(case)[1])
    assert record["actual"] == float(changed[1])
    assert record["independent"] == float(independently_recompute_logprobs(case)[1])
    assert record["absolute_error"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "field, reason",
    [("token_indices", "token_indices_mismatch"), ("expert_indices", "expert_indices_mismatch")],
)
def test_mapping_injection_is_separate_from_numeric_error(field, reason):
    case = _case()
    changed = getattr(case, field).clone()
    changed[1] += 1
    result = _compare(case, candidate_case=replace(case, **{field: changed}))
    assert result["status"] == "invalid_identity"
    assert result["first_divergence"]["reason"] == reason
    assert result["first_divergence"]["flat_index"] == 1
    assert result["fixed_mismatch_count"] == result["accuracy_mismatch_count"] == 0


def test_rank_mapping_injection_is_localized():
    case = _case()
    changed = case.rank_indices.clone()
    changed[2] = 0
    result = _compare(case, candidate_case=replace(case, rank_indices=changed))
    assert result["status"] == "invalid_identity"
    assert result["first_divergence"]["reason"] == "rank_indices_mismatch"
    assert result["first_divergence"]["flat_index"] == 2


def test_stale_run_and_parameter_reference_fail_closed():
    case = _case()
    with pytest.raises(MoELogprobError, match="stale mapping"):
        compare_moe_logprobs(
            replace(case, mapping_run_id="old"),
            _fixed(case),
            case,
            _candidate(_fixed(case)),
            expected_mapping_run_id="mapping-1",
            expected_model_version="weights-1",
        )
    with pytest.raises(MoELogprobError, match="stale model"):
        compare_moe_logprobs(
            replace(case, model_version="old"),
            _fixed(case),
            case,
            _candidate(_fixed(case)),
            expected_mapping_run_id="mapping-1",
            expected_model_version="weights-1",
        )


@pytest.mark.parametrize("bad", [torch.tensor(1.0), torch.ones(3, 2)])
def test_observation_shape_is_rejected(bad):
    with pytest.raises(MoELogprobError, match="one-dimensional"):
        _candidate(bad)


def test_wrong_one_dimensional_observation_length_is_reported():
    result = _compare(_case(), values=torch.ones(2))
    assert result["status"] == "invalid_artifact"
    assert result["first_divergence"]["reason"] == "shape_mismatch"


def test_nonfinite_logits_and_unsupported_probability_rule_are_rejected():
    case = _case()
    with pytest.raises(MoELogprobError, match="finite"):
        replace(case, logits=case.logits.clone().fill_(float("nan")))
    with pytest.raises(MoELogprobError, match="unsupported probability"):
        replace(case, probability_rule="log_softmax_npu_fast_math")


@pytest.mark.parametrize("field", ["target_ids", "token_indices", "expert_indices"])
def test_float_indices_are_rejected_instead_of_truncated(field):
    case = _case()
    with pytest.raises(MoELogprobError, match="integer dtype"):
        replace(case, **{field: getattr(case, field).float()})
