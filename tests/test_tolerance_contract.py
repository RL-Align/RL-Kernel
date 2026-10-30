# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Schema and resolver tests for WS1 C1 four-judgment contract (#267)."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from rl_engine.kernels.gtest.tolerance import (
    CHAIN_AGGREGATE_METRICS,
    JUDGMENTS,
    OP_CLASSES,
    BackendProvenance,
    ContractResolveError,
    ContractSchemaError,
    assert_comparison_roles,
    compute_logprob_aggregates,
    default_clip_interval,
    judge_logprob_aggregates,
    load_contract,
    resolve_chain_aggregate_thresholds,
    resolve_comparison_roles,
    resolve_dtype_policy,
    resolve_tolerance,
    resolve_tolerance_support,
    validate_backend_provenance,
    validate_contract_schema,
)


def test_load_contract_contains_expected_operator_classes():
    contract = load_contract()
    accuracy = contract["accuracy"]["default"]
    assert set(accuracy) == {
        "elementwise",
        "reduction",
        "logprob",
        "attention",
        "mhc_controller",
    }


def test_load_contract_contains_expected_dtypes():
    contract = load_contract()
    for op_class in ("elementwise", "reduction", "logprob", "attention"):
        assert set(contract["accuracy"]["default"][op_class]) == {
            "float32",
            "bfloat16",
            "float16",
        }


def test_logprob_bfloat16_tolerance_covers_observed_reference_drift():
    contract = load_contract()
    tolerance = contract["accuracy"]["default"]["logprob"]["bfloat16"]
    assert tolerance["atol"] >= 5.0e-2
    assert tolerance["rtol"] == 0.0


def test_attention_bfloat16_tolerance_matches_contract():
    contract = load_contract()
    tolerance = contract["accuracy"]["default"]["attention"]["bfloat16"]
    assert tolerance["atol"] >= 5.0e-2
    assert tolerance["rtol"] >= 2.0e-2


def test_contract_schema_validates_on_load():
    contract = load_contract(validate=True)
    validate_contract_schema(contract)


def test_dtype_policy_locks_bf16_fp32_fp8_tf32():
    policy = resolve_dtype_policy(load_contract())
    assert policy.execution_dtype == "bfloat16"
    assert policy.accumulation_dtype == "float32"
    assert policy.reference_dtype == "float32"
    assert policy.output_dtype_default == "bfloat16"
    assert policy.logprob_aggregates_dtype == "float32"
    assert policy.fp8 == "out_of_scope"
    assert policy.fp16_status == "optional"
    assert policy.tf32_reference == "disabled"
    assert policy.tf32_candidate_execution == "disabled"
    assert "cuda_bf16" in policy.backend_profiles
    assert "triton_cuda_bf16" in policy.backend_profiles
    assert policy.backend_private_tolerance_relaxation is False


def test_four_judgments_present_and_complete():
    contract = load_contract()
    assert set(contract["judgments"]) == set(JUDGMENTS)
    for judgment in JUDGMENTS:
        by_op = contract["judgments"][judgment]["by_op_class"]
        assert set(by_op) == set(OP_CLASSES)
        for op_class in OP_CLASSES:
            for dtype_name in ("float32", "bfloat16", "float16", "float8"):
                assert dtype_name in by_op[op_class]


def test_invariance_rows_are_bitwise_zero():
    contract = load_contract()
    for judgment in ("forward_invariance", "gradient_invariance"):
        for op_class in OP_CLASSES:
            for dtype_name in ("float32", "bfloat16"):
                spec = resolve_tolerance(
                    contract,
                    judgment=judgment,
                    op_class=op_class,
                    dtype=dtype_name,
                )
                assert spec.mode == "bitwise"
                assert spec.atol == 0.0
                assert spec.rtol == 0.0


def test_cuda_and_triton_profiles_share_thresholds():
    contract = load_contract()
    for judgment in JUDGMENTS:
        for op_class in OP_CLASSES:
            for dtype_name in ("float32", "bfloat16"):
                cuda = resolve_tolerance(
                    contract,
                    judgment=judgment,
                    op_class=op_class,
                    dtype=dtype_name,
                    backend_profile="cuda_bf16",
                )
                triton = resolve_tolerance(
                    contract,
                    judgment=judgment,
                    op_class=op_class,
                    dtype=dtype_name,
                    backend_profile="triton_cuda_bf16",
                )
                assert (cuda.atol, cuda.rtol, cuda.mode) == (
                    triton.atol,
                    triton.rtol,
                    triton.mode,
                )


def test_unknown_backend_profile_hard_fails():
    contract = load_contract()
    with pytest.raises(ContractResolveError, match="backend_profile"):
        resolve_tolerance(
            contract,
            judgment="forward_accuracy",
            op_class="reduction",
            dtype="bfloat16",
            backend_profile="private_backend",
        )


def test_backend_provenance_checks_profile_backend_and_all_dtypes():
    contract = load_contract()
    provenance = BackendProvenance(
        backend_profile="cuda_bf16",
        requested_backend="cuda",
        actual_backend="cuda",
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    assert validate_backend_provenance(contract, provenance) == provenance
    with pytest.raises(ContractResolveError, match="actual_backend"):
        validate_backend_provenance(
            contract,
            BackendProvenance(**{**provenance.to_dict(), "actual_backend": "triton"}),
        )
    with pytest.raises(ContractResolveError, match="output_dtype"):
        validate_backend_provenance(
            contract,
            BackendProvenance(**{**provenance.to_dict(), "output_dtype": "float32"}),
        )
    with pytest.raises(ContractResolveError, match="candidate_tf32_enabled"):
        validate_backend_provenance(
            contract,
            BackendProvenance(**{**provenance.to_dict(), "candidate_tf32_enabled": True}),
        )


def test_not_applicable_has_explicit_support_result_but_no_threshold():
    contract = copy.deepcopy(load_contract())
    cell = contract["judgments"]["forward_accuracy"]["by_op_class"]["elementwise"]["float16"]
    cell["status"] = "not_applicable"
    cell["reason"] = "profile does not declare FP16"
    validate_contract_schema(contract)
    support = resolve_tolerance_support(
        contract, judgment="forward_accuracy", op_class="elementwise", dtype="float16"
    )
    assert support.status == "not_applicable"
    with pytest.raises(ContractResolveError, match="not_applicable"):
        resolve_tolerance(
            contract, judgment="forward_accuracy", op_class="elementwise", dtype="float16"
        )


def test_fp8_request_hard_fails():
    contract = load_contract()
    with pytest.raises(ContractResolveError, match="out of scope"):
        resolve_tolerance(
            contract,
            judgment="forward_accuracy",
            op_class="reduction",
            dtype="float8",
        )


def test_missing_applicable_cell_hard_fails():
    contract = copy.deepcopy(load_contract())
    del contract["judgments"]["forward_accuracy"]["by_op_class"]["attention"]["bfloat16"]
    with pytest.raises(ContractSchemaError):
        validate_contract_schema(contract)
    # Resolver path: re-insert schema-invalid by skipping validate, then resolve.
    with pytest.raises(ContractResolveError, match="missing declared cell"):
        resolve_tolerance(
            contract,
            judgment="forward_accuracy",
            op_class="attention",
            dtype="bfloat16",
        )


def test_gradient_thresholds_do_not_inherit_forward():
    contract = copy.deepcopy(load_contract())
    # Mutate only forward_accuracy BF16 reduction.
    contract["judgments"]["forward_accuracy"]["by_op_class"]["reduction"]["bfloat16"]["atol"] = 9.9
    # Keep compat mirror in sync is not required for this unit test of independence.
    fwd = resolve_tolerance(
        contract,
        judgment="forward_accuracy",
        op_class="reduction",
        dtype="bfloat16",
    )
    grad = resolve_tolerance(
        contract,
        judgment="gradient_accuracy",
        op_class="reduction",
        dtype="bfloat16",
    )
    assert fwd.atol == 9.9
    assert grad.atol == 1.0e-1
    assert grad.atol != fwd.atol


def test_comparison_roles_by_report_kind():
    contract = load_contract()
    expected = {
        "forward_accuracy": ("bf16_candidate", "fp32_reference"),
        "forward_invariance": ("transformed_config", "canonical_config"),
        "train_infer_logprob_parity": (
            "training_style_teacher_forcing",
            "inference_style_rollout_decode",
        ),
        "gradient_accuracy": ("bf16_candidate", "fp32_reference"),
        "gradient_invariance": ("transformed_config", "canonical_config"),
    }
    for kind, (lhs, rhs) in expected.items():
        roles = resolve_comparison_roles(contract, kind)
        assert roles.comparison_lhs_role == lhs
        assert roles.comparison_rhs_role == rhs
        assert_comparison_roles(contract, kind, lhs, rhs)


def test_forbidden_and_reversed_roles_hard_fail():
    contract = load_contract()
    with pytest.raises(ContractResolveError, match="role mismatch"):
        assert_comparison_roles(
            contract,
            "train_infer_logprob_parity",
            "inference_style_rollout_decode",
            "training_style_teacher_forcing",
        )


def test_aggregate_requires_declared_roles_and_direction():
    contract = load_contract()
    values = torch.zeros(2)
    with pytest.raises(ContractResolveError, match="role mismatch"):
        compute_logprob_aggregates(
            values,
            values,
            torch.ones(2, dtype=torch.bool),
            contract=contract,
            report_kind="train_infer_logprob_parity",
            clip_interval=(0.8, 1.2),
            comparison_lhs_role="inference_style_rollout_decode",
            comparison_rhs_role="training_style_teacher_forcing",
        )
    with pytest.raises(ContractResolveError, match="forbidden"):
        assert_comparison_roles(
            contract,
            "forward_accuracy",
            "baseline",
            "fp32_reference",
        )
    with pytest.raises(ContractResolveError, match="forbidden"):
        assert_comparison_roles(
            contract,
            "forward_invariance",
            "singleton_aggregate",
            "canonical_config",
        )


def test_resolve_tolerance_attaches_roles():
    contract = load_contract()
    spec = resolve_tolerance(
        contract,
        judgment="forward_accuracy",
        op_class="logprob",
        dtype=torch.bfloat16,
    )
    assert spec.comparison_lhs_role == "bf16_candidate"
    assert spec.comparison_rhs_role == "fp32_reference"
    assert "baseline" not in (spec.comparison_lhs_role, spec.comparison_rhs_role)


def test_chain_aggregate_named_resolve():
    contract = load_contract()
    expected = {
        "max_abs_dlogp": {"bfloat16": 6.0e-2, "float32": 1.0e-5},
        "approx_kl0": {"bfloat16": 5.0e-2, "float32": 1.0e-5},
        "clipfrac0": {"bfloat16": 0.0, "float32": 0.0},
    }
    assert set(expected) == set(CHAIN_AGGREGATE_METRICS)
    for metric in CHAIN_AGGREGATE_METRICS:
        for dtype, value in expected[metric].items():
            assert resolve_chain_aggregate_thresholds(contract, metric, dtype) == value
    with pytest.raises(ContractResolveError, match="unknown chain aggregate"):
        resolve_chain_aggregate_thresholds(contract, "mean_abs_dlogp", "bfloat16")


def test_calibrated_thresholds_record_calibration_rationale():
    contract = load_contract()
    gradient = contract["judgments"]["gradient_accuracy"]
    assert gradient["calibration_status"] == "calibrated_from_h20_full_model_backward_evidence"
    assert "0.0978" in gradient["calibration_note"]
    assert "0.1034" in gradient["calibration_note"]
    assert "both required profiles" in gradient["calibration_note"]

    approx_kl0 = contract["chain_logprob_aggregates"]["metrics"]["approx_kl0"]
    assert "max_abs_dlogp is therefore the stricter guard" in approx_kl0["threshold_rationale"]


def test_compute_logprob_aggregates_formulas():
    # lhs - rhs = [0.0, 0.1, -0.2]
    lhs = torch.tensor([1.0, 2.1, 0.8], dtype=torch.float32)
    rhs = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([True, True, True])
    clip = (0.8, 1.2)
    agg = compute_logprob_aggregates(
        lhs,
        rhs,
        mask,
        contract=load_contract(),
        report_kind="train_infer_logprob_parity",
        clip_interval=clip,
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    dlogp = torch.tensor([0.0, 0.1, -0.2])
    expected_max = float(dlogp.abs().max())
    expected_kl = float((torch.exp(dlogp) - 1.0 - dlogp).mean())
    ratio = torch.exp(dlogp)
    expected_clip = float(((ratio < clip[0]) | (ratio > clip[1])).float().mean())
    assert math.isclose(agg.max_abs_dlogp, expected_max, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(agg.approx_kl0, expected_kl, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(agg.clipfrac0, expected_clip, rel_tol=0.0, abs_tol=1e-6)
    assert agg.active_token_count == 3


def test_active_mask_filters_tokens():
    lhs = torch.tensor([0.0, 10.0], dtype=torch.float32)
    rhs = torch.tensor([0.0, 0.0], dtype=torch.float32)
    mask = torch.tensor([True, False])
    agg = compute_logprob_aggregates(
        lhs,
        rhs,
        mask,
        contract=load_contract(),
        report_kind="train_infer_logprob_parity",
        clip_interval=default_clip_interval(load_contract()),
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    assert agg.max_abs_dlogp == 0.0
    assert agg.active_token_count == 1


def test_empty_active_set_hard_fails():
    lhs = torch.zeros(2)
    rhs = torch.zeros(2)
    mask = torch.tensor([False, False])
    with pytest.raises(ContractResolveError, match="empty active-token"):
        compute_logprob_aggregates(
            lhs,
            rhs,
            mask,
            contract=load_contract(),
            report_kind="train_infer_logprob_parity",
            clip_interval=(0.8, 1.2),
            comparison_lhs_role="training_style_teacher_forcing",
            comparison_rhs_role="inference_style_rollout_decode",
        )


def test_nan_inf_hard_fail():
    lhs = torch.tensor([float("nan"), 0.0])
    rhs = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, True])
    with pytest.raises(ContractResolveError, match="NaN/Inf"):
        compute_logprob_aggregates(
            lhs,
            rhs,
            mask,
            contract=load_contract(),
            report_kind="train_infer_logprob_parity",
            clip_interval=(0.8, 1.2),
            comparison_lhs_role="training_style_teacher_forcing",
            comparison_rhs_role="inference_style_rollout_decode",
        )

    # Finite dlogp can still overflow exp(dlogp), which is a separate hard-fail.
    lhs = torch.tensor([200.0, 0.0], dtype=torch.float32)
    rhs = torch.zeros(2)
    with pytest.raises(ContractResolveError, match="ratio0"):
        compute_logprob_aggregates(
            lhs,
            rhs,
            torch.ones(2, dtype=torch.bool),
            contract=load_contract(),
            report_kind="train_infer_logprob_parity",
            clip_interval=(0.8, 1.2),
            comparison_lhs_role="training_style_teacher_forcing",
            comparison_rhs_role="inference_style_rollout_decode",
        )

    lhs = torch.tensor([float("inf"), 0.0])
    with pytest.raises(ContractResolveError, match="NaN/Inf"):
        compute_logprob_aggregates(
            lhs,
            rhs,
            mask,
            contract=load_contract(),
            report_kind="train_infer_logprob_parity",
            clip_interval=(0.8, 1.2),
            comparison_lhs_role="training_style_teacher_forcing",
            comparison_rhs_role="inference_style_rollout_decode",
        )


def test_inactive_nan_is_ignored():
    agg = compute_logprob_aggregates(
        torch.tensor([0.0, float("nan")]),
        torch.zeros(2),
        torch.tensor([True, False]),
        contract=load_contract(),
        report_kind="train_infer_logprob_parity",
        clip_interval=(0.8, 1.2),
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    assert agg.active_token_count == 1
    assert agg.max_abs_dlogp == 0.0


def test_clipfrac0_counts_ratios_outside_the_interval():
    agg = compute_logprob_aggregates(
        torch.tensor([0.0, 1.0, -1.0]),
        torch.zeros(3),
        torch.ones(3, dtype=torch.bool),
        contract=load_contract(),
        report_kind="train_infer_logprob_parity",
        clip_interval=(0.8, 1.2),
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    assert math.isclose(agg.clipfrac0, 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-6)


def test_clip_interval_endpoints_count_as_inside():
    # Drive endpoints through the same float32 exp path the implementation uses so
    # ratio0 lands exactly on the clip interval bounds (no log/exp float round-trip).
    dlogp = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    ratio0 = torch.exp(dlogp)
    lo = float(ratio0[0].item())
    hi = float(ratio0[1].item())
    agg = compute_logprob_aggregates(
        dlogp,
        torch.zeros(2, dtype=torch.float32),
        torch.ones(2, dtype=torch.bool),
        contract=load_contract(),
        report_kind="train_infer_logprob_parity",
        clip_interval=(lo, hi),
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    assert agg.clipfrac0 == 0.0


def test_judge_requires_all_three_aggregates():
    contract = load_contract()
    clip = default_clip_interval(contract)
    # Perfect match → all pass.
    lhs = torch.zeros(4)
    rhs = torch.zeros(4)
    mask = torch.ones(4, dtype=torch.bool)
    agg = compute_logprob_aggregates(
        lhs,
        rhs,
        mask,
        contract=contract,
        report_kind="train_infer_logprob_parity",
        clip_interval=clip,
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    verdict = judge_logprob_aggregates(agg, contract, execution_dtype="bfloat16")
    assert verdict.passed
    assert {m.metric for m in verdict.metrics} == set(CHAIN_AGGREGATE_METRICS)
    assert all(m.passed for m in verdict.metrics)

    # A small in-interval drift fails only max_abs_dlogp. This proves the
    # overall verdict requires all three metrics, rather than any one metric.
    lhs = torch.tensor([0.1])
    rhs = torch.zeros(1)
    mask = torch.ones(1, dtype=torch.bool)
    agg = compute_logprob_aggregates(
        lhs,
        rhs,
        mask,
        contract=contract,
        report_kind="train_infer_logprob_parity",
        clip_interval=clip,
        comparison_lhs_role="training_style_teacher_forcing",
        comparison_rhs_role="inference_style_rollout_decode",
    )
    verdict = judge_logprob_aggregates(agg, contract, execution_dtype="bfloat16")
    assert not verdict.passed
    by_metric = {metric.metric: metric.passed for metric in verdict.metrics}
    assert by_metric == {
        "max_abs_dlogp": False,
        "approx_kl0": True,
        "clipfrac0": True,
    }


def test_compat_accuracy_mirrors_forward_accuracy():
    contract = load_contract()
    for op_class in OP_CLASSES:
        for dtype_name in ("float32", "bfloat16", "float16"):
            acc = contract["accuracy"]["default"][op_class][dtype_name]
            cell = contract["judgments"]["forward_accuracy"]["by_op_class"][op_class][dtype_name]
            assert acc["atol"] == cell["atol"]
            assert acc["rtol"] == cell["rtol"]
    assert contract["batch_invariance"] == {"atol": 0.0, "rtol": 0.0}


def test_schema_rejects_nonzero_invariance_tolerance():
    contract = copy.deepcopy(load_contract())
    contract["judgments"]["forward_invariance"]["by_op_class"]["logprob"]["bfloat16"]["atol"] = 1e-3
    with pytest.raises(ContractSchemaError, match="bitwise"):
        validate_contract_schema(contract)


def test_schema_rejects_baseline_role():
    contract = copy.deepcopy(load_contract())
    contract["comparison_roles"]["by_report_kind"]["forward_accuracy"][
        "comparison_lhs_role"
    ] = "baseline"
    with pytest.raises(ContractSchemaError, match="forbidden role"):
        validate_contract_schema(contract)
