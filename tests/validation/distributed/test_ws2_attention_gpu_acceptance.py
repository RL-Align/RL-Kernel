# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU-safe validation for the strict WS2 Attention GPU acceptance runner."""

from __future__ import annotations

import json
import subprocess

from tools.validation.distributed.ws2_attention_gpu_acceptance import (
    AcceptanceCase,
    _run_case,
    build_acceptance_cases,
    parse_args,
    run_acceptance,
    validate_native_te_report,
    validate_p2p_report,
    validate_pr5_report,
    validate_pr7_report,
)


def test_manifest_fails_closed_for_every_unexecuted_required_case(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = run_acceptance(args)

    assert report["status"] == "failed"
    assert report["passed"] is False
    assert "custom_cuda_ag_rs" in report["failed_required_cases"]
    assert all(not case["passed"] for case in report["cases"])


def test_matrix_contains_required_modes_splitk_and_communication(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    cases = build_acceptance_cases(args)
    names = {case.name for case in cases}

    assert "pr5_cp_forward_backward_dlogp" in names
    assert "p2p_nccl_reference" in names
    assert "native_te_kv_ring_cp_compare" in names
    assert "pr7_flashinfer_decode_disabled" in names
    assert "pr7_flashinfer_decode_fixed" in names
    assert "pr7_flashinfer_prefill_disabled" in names
    assert "pr7_flashinfer_prefill_fixed" in names
    assert "custom_cuda_ag_rs" in names
    assert "p2p_nccl_reference_tp2_cp2" in names
    assert "p2p_nccl_reference_tp2_cp2_replica2" in names
    assert "custom_cuda_ag_rs_tp2_cp2" in names
    assert "custom_cuda_ag_rs_tp2_cp2_replica2" in names
    communication_cases = [
        case for case in cases if case.name.startswith(("p2p_nccl_reference", "custom_cuda_ag_rs"))
    ]
    assert len(communication_cases) == 6
    assert all(
        case.command is not None and "--transport" in case.command for case in communication_cases
    )
    by_name = {case.name: case for case in cases}
    assert by_name["pr7_flashinfer_decode_disabled"].required is True
    strict_command = by_name["pr7_flashinfer_decode_disabled"].command
    if strict_command is not None:
        assert "--strict" in strict_command
    assert by_name["pr7_flashinfer_decode_fixed"].required is False
    diagnostic_command = by_name["pr7_flashinfer_decode_fixed"].command
    if diagnostic_command is not None:
        assert "--strict" not in diagnostic_command


def test_formal_communication_cases_cover_four_and_eight_rank_entrypoints(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    cases = {case.name: case for case in build_acceptance_cases(args)}

    p2p = cases["p2p_nccl_reference_tp2_cp2"]
    assert p2p.command is not None
    assert "--nproc-per-node=4" in p2p.command
    assert "--transport" in p2p.command
    assert "p2p_nccl_reference" in p2p.command
    assert "--repeats" in p2p.command
    assert p2p.report_path is not None

    custom = cases["custom_cuda_ag_rs_tp2_cp2"]
    assert custom.command is not None
    assert "--nproc-per-node=4" in custom.command
    assert "cuda_ag_rs" in custom.command
    assert "--strict-shared-core" in custom.command
    assert custom.report_path is not None

    replicated = cases["custom_cuda_ag_rs_tp2_cp2_replica2"]
    assert replicated.command is not None
    assert "--nproc-per-node=8" in replicated.command
    assert "--strict-shared-core" in replicated.command

    p2p_replica = cases["p2p_nccl_reference_tp2_cp2_replica2"]
    assert p2p_replica.command is not None
    assert "--strict-shared-core" not in p2p_replica.command


def test_native_te_kv_ring_is_optional_diagnostic(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    case = next(
        case for case in build_acceptance_cases(args) if case.name == "native_te_kv_ring_cp_compare"
    )

    assert case.required is False


def test_allreduce_is_optional_for_attention_gate(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    case = next(
        case for case in build_acceptance_cases(args) if case.name == "custom_cuda_allreduce"
    )

    assert case.required is False


def test_dlogp_default_uses_shared_bf16_logprob_tolerance(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])

    assert args.dlogp_atol == 5.0e-2
    assert args.pr7_out_atol == 1.0e-2
    assert args.pr7_lse_atol == 2.0e-3
    assert args.pr7_dlogp_atol == 2.0e-3


def test_native_te_validator_requires_native_kv_ring_and_cp_compare(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = {
        "schema_version": "ws2_megatron_te_cp_compare/v1",
        "status": "passed",
        "passed": True,
        "transport": "native_te_kv_ring",
        "requested": {"cp_comm_type": "p2p", "context_parallel_sizes": [1, 2]},
        "comparison": {
            "pass": True,
            "left_cp_size": 1,
            "right_cp_size": 2,
            "max_abs": 0.0,
            "token_ids_sha256": "a" * 64,
        },
        "runs": [
            {
                "cp_size": cp_size,
                "status": "passed",
                "active_token_count": 2,
                "token_ids_sha256": "a" * 64,
                "actual": {
                    "provider": {
                        "transformer_impl": "transformer_engine",
                        "cp_comm_type": "p2p",
                    }
                },
            }
            for cp_size in (1, 2)
        ],
        "errors": [],
    }

    assert validate_native_te_report(report, args) == []
    report["comparison"]["max_abs"] = "not-a-number"
    assert any("not numeric" in error for error in validate_native_te_report(report, args))
    report["comparison"]["max_abs"] = 0.0
    report["transport"] = "native_te_kv_all_gather"
    assert validate_native_te_report(report, args)
    report["transport"] = "native_te_kv_ring"
    report["runs"] = report["runs"][:1]
    assert "native TE report must contain exactly two runs" in validate_native_te_report(
        report, args
    )


def test_run_mode_preserves_structured_not_available_reports(tmp_path):
    report_path = tmp_path / "not-available.json"
    args = parse_args(["--mode", "run", "--output", str(tmp_path / "acceptance.json")])
    case = AcceptanceCase(
        name="optional",
        command=("fake",),
        report_path=report_path,
    )

    def fake_runner(command, **kwargs):
        report_path.write_text(
            json.dumps(
                {
                    "status": "not_available",
                    "errors": ["FlashInfer unavailable: missing wheel"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    row = _run_case(case, args, runner=fake_runner)
    assert row["status"] == "not_available"
    assert row["passed"] is False
    assert row["errors"] == ["FlashInfer unavailable: missing wheel"]


def test_run_mode_does_not_pass_when_reports_are_missing(tmp_path):
    args = parse_args(
        [
            "--mode",
            "run",
            "--output",
            str(tmp_path / "acceptance.json"),
        ]
    )

    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    report = run_acceptance(args, runner=fake_runner)

    assert report["passed"] is False
    assert "custom_cuda_ag_rs" in report["failed_required_cases"]
    assert any(case["status"] == "invalid_report" for case in report["cases"])


def test_pr7_strict_validation_rejects_requested_only_split_plan(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = {
        "status": "passed",
        "passed": True,
        "candidate_provenance": {
            "arithmetic_semantics_verified": True,
            "actual_split_kv_plans": [
                {
                    "actual_split_kv_policy": None,
                    "actual_split_boundaries": [],
                }
            ],
            "actual_split_kv_plan_set": None,
        },
        "drift": {
            "out": {"max_abs": 5.0e-3},
            "lse": {"max_abs": 1.0e-3},
            "dlogp": {"max_abs": 1.0e-3},
        },
        "batch_invariant_sweep": {"passed": True},
        "page_layout_invariant_sweep": {"passed": True},
    }

    errors = validate_pr7_report(report, args, expected_policy="fixed")

    assert any("actual Split-K policy" in error for error in errors)
    assert any("boundaries" in error for error in errors)
    assert any("plan set" in error for error in errors)
    assert not any(error.startswith("PR7.") for error in errors)

    report["drift"]["out"]["max_abs"] = 2.0e-2
    assert any(
        error.startswith("PR7.out")
        for error in validate_pr7_report(report, args, expected_policy="fixed")
    )


def test_pr7_strict_validation_accepts_shared_no_split_core(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = {
        "status": "passed",
        "passed": True,
        "candidate_provenance": {
            "arithmetic_semantics_verified": True,
            "strict_mode": True,
            "strict_core_id": "rlkernel.attention.deterministic_core.v1",
            "strict_schedule": "single_batch_single_query_global_kv_blocks",
            "native_attention_arithmetic": False,
            "fallback": False,
            "strict_core_row_plans": [
                {
                    "actual_split_kv_policy": "disabled",
                    "actual_split_boundaries": [[0, 8]],
                }
            ],
        },
        "drift": {
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {
            "passed": True,
            "out_max_abs": 0.0,
            "lse_max_abs": 0.0,
        },
        "page_layout_invariant_sweep": {
            "passed": True,
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
        },
    }

    assert (
        validate_pr7_report(
            report,
            args,
            expected_policy="disabled",
            strict_expected=True,
        )
        == []
    )

    report["candidate_provenance"]["strict_schedule"] = "different_schedule"
    assert any(
        "strict arithmetic schedule" in error
        for error in validate_pr7_report(
            report,
            args,
            expected_policy="disabled",
            strict_expected=True,
        )
    )


def test_acceptance_report_is_json_serializable(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    json.dumps(run_acceptance(args))


def _valid_pr5_report():
    def case(mode, policy, split_size):
        stats = {"max_abs": 0.0}
        entries = []
        for tp_rank in range(2):
            for cp_rank in range(2):
                for owner_cp_rank, owner_range in enumerate(([0, 2], [2, 4])):
                    entries.append(
                        {
                            "batch_index": 0,
                            "tp_rank": tp_rank,
                            "cp_rank": cp_rank,
                            "owner_cp_rank": owner_cp_rank,
                            "expected_kv_range": owner_range,
                            "requested_split_kv_policy": policy,
                            "actual_split_kv_policy": policy,
                            "actual_split_kv_size": split_size,
                            "actual_split_kv_count": 1 if split_size is None else 2,
                            "actual_split_boundaries": (
                                [owner_range]
                                if split_size is None
                                else [
                                    [owner_range[0], owner_range[0] + 1],
                                    [owner_range[0] + 1, owner_range[1]],
                                ]
                            ),
                            "split_kv_merge_order": "global_block_index",
                            "split_kv_accum_dtype": "fp32",
                            "split_kv_downcast_at": "final_write",
                            "split_kv_plan_source": "test_runtime",
                            "split_kv_fallback": False,
                            "split_kv_fallback_reason": None,
                        }
                    )
        return {
            "case_name": f"{mode}-{policy}",
            "attention_mode": mode,
            "topology": {"tp_world_size": 2, "cp_world_size": 2},
            "provenance": {
                "requested_split_kv_policy": policy,
                "requested_split_kv_size": split_size,
                "rope": {"rope_state": "post_rope"},
                "actual_split_kv_plan_set": {
                    "batch_size": 1,
                    "tp_world_size": 2,
                    "cp_world_size": 2,
                    "total_kv_tokens": [4],
                    "entries": entries,
                    "coverage": "complete_batch_tp_cp_owner_cartesian_product",
                },
            },
            "drift": {"cp_merge_fp32": {"out": stats, "lse": stats}},
            "dlogp": {"status": "available", "drift": stats},
            "backward": {
                "status": "available",
                "report": {"drifts": [{"dq": stats, "dk": stats, "dv": stats}]},
            },
        }

    return {
        "schema_version": "ws2_cp_attention_drift/v2",
        "issue": 235,
        "pr": 5,
        "runtime": {"device": "cuda:0"},
        "target": {
            "model": "qwen3-8b",
            "dtype": "bf16",
            "global_num_query_heads": 32,
            "global_num_kv_heads": 8,
            "head_dim": 128,
            "batch": 1,
        },
        "cases": [case("prefill", "disabled", None), case("chunked_prefill", "fixed", 4)],
    }


def test_pr5_validation_binds_gpu_identity_and_nonempty_backward(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    assert validate_pr5_report(report, args) == []

    report["runtime"]["device"] = "cpu"
    report["cases"][0]["backward"]["report"]["drifts"] = []
    errors = validate_pr5_report(report, args)
    assert any("not produced on CUDA" in error for error in errors)
    assert any("backward drift rows" in error for error in errors)


def test_pr5_validation_rejects_nonfinite_or_negative_drift(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    report["cases"][0]["drift"]["cp_merge_fp32"]["out"] = {"max_abs": float("nan")}
    report["cases"][1]["dlogp"]["drift"] = {"max_abs": -1.0}

    errors = validate_pr5_report(report, args)
    assert sum("finite and non-negative" in error for error in errors) == 2


def _valid_p2p_report(world_size=4, transport="p2p_nccl_reference"):
    tp_world_size = 1 if world_size == 2 else 2
    replica_count = 2 if world_size == 8 else 1
    manifest_by_tp = {}
    rows = []
    for rank in range(world_size):
        replica_rank = rank % 4 if world_size == 8 else rank
        replica_index = rank // 4 if world_size == 8 else 0
        tp_rank = 0 if world_size == 2 else replica_rank // 2
        cp_rank = replica_rank % 2
        manifest = manifest_by_tp.setdefault(
            tp_rank,
            [
                {
                    "global_block_index": block,
                    "kv_block_start": block * 4,
                    "kv_block_end": block * 4 + 4,
                    "owner_cp_rank": 0 if block < 2 else 1,
                    "owner_tp_rank": tp_rank,
                }
                for block in range(4)
            ],
        )
        rows.append(
            {
                "rank": rank,
                "global_world_size": world_size,
                "tp_rank": tp_rank,
                "tp_world_size": 2,
                "cp_rank": cp_rank,
                "cp_world_size": 2,
                "replica_index": replica_index,
                "replica_count": replica_count,
                "passed": True,
                "global_failure_count": 0,
                "transport": transport,
                "query_ag": transport,
                "protocol": "ag_query_local_kv_rs_out_lse",
                "strict_protocol": "ag_qkv_positions_shared_core_rs_out_lse",
                "query_ag_max_abs": 0.0,
                "device": f"cuda:{rank}",
                "dtype": "bf16",
                "accum_dtype": "fp32",
                "downcast_at": "final_write",
                "final_output_dtype": "bfloat16",
                "query_range": [0, 8] if cp_rank == 0 else [8, 16],
                "expected_block_manifest": manifest,
                "local_block_indices": [0, 1] if cp_rank == 0 else [2, 3],
                "gathered_block_indices": [0, 1, 2, 3],
                "repeat_count": 3,
                "repeat_query_bitwise": True,
                "repeat_out_bitwise": True,
                "repeat_lse_bitwise": True,
                "repeat_manifest_bitwise": True,
                "out_max_abs": 0.0,
                "lse_max_abs": 0.0,
                "final_out_max_abs": 0.0,
                "atol": 2.0e-4,
                "final_write_atol": 2.0e-2,
                "strict_shared_core": (
                    {
                        "executed": True,
                        "passed": True,
                        "strict_core_id": "rlkernel.attention.deterministic_core.v1",
                        "strict_schedule": "single_batch_single_query_global_kv_blocks",
                        "actual_backend": "rlkernel.cuda.deterministic_attention",
                        "communication_backend": "self_owned_cuda_ag_rs",
                        "production_ready": True,
                        "strict_mode": True,
                        "native_attention_arithmetic": False,
                        "fallback": False,
                        "split_kv_policy": "disabled",
                        "communication_autograd": True,
                        "bitwise": {
                            "out": True,
                            "lse": True,
                            "dq": True,
                            "dk": True,
                            "dv": True,
                        },
                        "max_abs": {
                            "out": 0.0,
                            "lse": 0.0,
                            "dq": 0.0,
                            "dk": 0.0,
                            "dv": 0.0,
                        },
                        "repeat_out_bitwise": True,
                        "repeat_lse_bitwise": True,
                    }
                    if transport == "cuda_ag_rs"
                    else {"executed": False, "passed": False}
                ),
            }
        )
    return {
        "schema_version": (
            "ws2_p2p_nccl_attention_reference/v1"
            if transport == "p2p_nccl_reference"
            else "ws2_cuda_ag_rs_attention/v1"
        ),
        "backend": "nccl",
        "transport": transport,
        "world_size": world_size,
        "tp_world_size": tp_world_size,
        "cp_world_size": 2,
        "replica_count": replica_count,
        "global_failure_count": 0,
        "ranks": rows,
    }


def test_p2p_validation_binds_nccl_rank_and_arithmetic_provenance():
    report = _valid_p2p_report()
    assert validate_p2p_report(report, expected_world_size=4) == []

    report["ranks"][1]["transport"] = "gloo"
    report["ranks"][1]["rank"] = 0
    errors = validate_p2p_report(report, expected_world_size=4)
    assert any("p2p_nccl_reference" in error for error in errors)
    assert any("ranks 0 through 3" in error for error in errors)


def test_p2p_validation_accepts_legacy_two_rank_artifact():
    report = _valid_p2p_report(world_size=2)
    assert validate_p2p_report(report) == []
    assert validate_p2p_report(report, expected_world_size=4)


def test_cuda_ag_rs_validation_accepts_two_tp2_cp2_replicas():
    report = _valid_p2p_report(world_size=8, transport="cuda_ag_rs")

    assert (
        validate_p2p_report(
            report,
            expected_transport="cuda_ag_rs",
            expected_world_size=8,
            expected_strict_core=True,
        )
        == []
    )


def test_cuda_ag_rs_validation_rejects_missing_strict_gradient_bitwise_evidence():
    report = _valid_p2p_report(world_size=4, transport="cuda_ag_rs")
    report["ranks"][0]["strict_shared_core"]["bitwise"]["dk"] = False

    errors = validate_p2p_report(
        report,
        expected_transport="cuda_ag_rs",
        expected_world_size=4,
        expected_strict_core=True,
    )
    assert any("gradient bitwise evidence" in error for error in errors)


def test_p2p_validation_rejects_claimed_downcast_without_final_output_evidence():
    report = _valid_p2p_report()
    for row in report["ranks"]:
        row.pop("final_output_dtype")
        row.pop("expected_block_manifest")
        row.pop("final_out_max_abs")
        row.pop("final_write_atol")

    errors = validate_p2p_report(report, expected_world_size=4)
    assert any("final output dtype" in error for error in errors)
    assert any("gathered block order/coverage" in error for error in errors)
    assert any("final_out_max_abs" in error for error in errors)


def test_p2p_validation_rejects_forged_manifest_and_rank_query_mapping():
    report = _valid_p2p_report()
    report["ranks"][0]["expected_block_manifest"][1]["kv_block_start"] = 5
    report["ranks"][0]["expected_block_manifest"][1]["owner_cp_rank"] = 3
    report["ranks"][0]["query_range"] = [8, 16]
    report["ranks"][1]["query_range"] = [0, 8]

    errors = validate_p2p_report(report, expected_world_size=4)
    assert any("gap-free KV coverage" in error for error in errors)
    assert any("outside the TP-local CP=2 group" in error for error in errors)
    assert any("both CP ranks" in error for error in errors)
    assert any("query ownership ranges" in error for error in errors)


def test_pr5_validation_rejects_forged_plan_set_coverage(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    plan_set = report["cases"][0]["provenance"]["actual_split_kv_plan_set"]
    plan_set["entries"] = plan_set["entries"][:-1]
    plan_set["entries"][0]["split_kv_accum_dtype"] = "bf16"

    errors = validate_pr5_report(report, args)
    assert any("coordinate coverage is incomplete" in error for error in errors)
    assert any("accumulation dtype is not fp32" in error for error in errors)


def test_pr5_validation_reports_malformed_coordinates_without_crashing(tmp_path):
    args = parse_args(["--output", str(tmp_path / "acceptance.json")])
    report = _valid_pr5_report()
    plan_set = report["cases"][0]["provenance"]["actual_split_kv_plan_set"]
    plan_set["entries"][0]["batch_index"] = []

    errors = validate_pr5_report(report, args)
    assert any("coordinate must contain integers" in error for error in errors)
    assert any("coordinate coverage is incomplete" in error for error in errors)
