# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the WS2 CP attention PR5 drift benchmark artifact."""

from __future__ import annotations

import json

import pytest

from benchmarks.distributed.benchmark_ws2_cp_attention_drift import (
    SCHEMA_VERSION,
    parse_args,
    run_benchmark,
    write_report,
)


def test_smoke_report_has_pr5_schema_and_qwen3_tp2_cp2_case():
    report = run_benchmark(
        parse_args(
            [
                "--smoke",
                "--tp-world-sizes",
                "2",
                "--cp-world-sizes",
                "2",
                "--kv-chunk-sizes",
                "none,1",
            ]
        )
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["report_family"] == "ws2_cross_config_drift_report"
    assert report["tolerance_source"] == "#108"
    assert report["issue"] == 235
    assert report["pr"] == 5
    assert report["target"]["model"] == "qwen3-8b"
    assert report["target"]["global_num_query_heads"] == 32
    assert report["target"]["global_num_kv_heads"] == 8
    assert report["te_context_parallel_merge"]["te_module"].endswith("context_parallel")
    assert report["dlogp"]["status"] == "not_requested"
    assert len(report["cases"]) == 2

    names = {case["case_name"] for case in report["cases"]}
    assert "qwen3_8b_tp2_cp2_prefill_bf16" in names
    assert "qwen3_8b_tp2_cp2_chunk1_bf16" in names

    chunked = next(case for case in report["cases"] if case["attention_mode"] == "chunked_prefill")
    assert chunked["topology"]["local_num_query_heads"] == 16
    assert chunked["topology"]["local_num_kv_heads"] == 4
    assert chunked["topology"]["local_query_head_range"] == [0, 16]
    assert chunked["topology"]["local_kv_head_range"] == [0, 4]
    assert chunked["provenance"]["merge_order"] == "global_block_index"
    assert chunked["provenance"]["split_kv_policy"] == "fixed"
    assert chunked["provenance"]["requested_split_kv_size"] == 1
    assert chunked["provenance"]["actual_split_kv_plans"][0]["actual_split_boundaries"]
    assert chunked["provenance"]["actual_split_kv_plans"][0]["split_kv_accum_dtype"] == "fp32"
    assert (
        chunked["provenance"]["actual_split_kv_plans"][0]["split_kv_downcast_at"] == "final_write"
    )
    plan_set = chunked["provenance"]["actual_split_kv_plan_set"]
    assert plan_set["coverage"] == "complete_batch_tp_cp_owner_cartesian_product"
    assert len(plan_set["entries"]) == 8
    assert chunked["distributed_p2p_reference"]["status"] == "not_requested"
    assert chunked["provenance"]["block_metadata_hash"]
    assert chunked["provenance"]["rope"]["rope_state"] == "post_rope"
    assert chunked["drift"]["rope"]["status"] == "available"
    assert chunked["drift"]["cp_merge_fp32"]["out"]["max_abs"] <= 1.0e-5
    assert chunked["drift"]["cp_merge_fp32"]["lse"]["max_abs"] <= 1.0e-5
    assert chunked["merge_order_probe"]["out"]["max_abs"] == 0.0
    assert len(chunked["per_rank"]) == 2
    assert chunked["per_rank"][0]["out"]["active_count"] > 0
    assert chunked["dlogp"]["status"] == "not_requested"


def test_include_dlogp_reports_active_selected_token_drift():
    report = run_benchmark(
        parse_args(
            [
                "--smoke",
                "--include-dlogp",
                "--tp-world-sizes",
                "2",
                "--cp-world-sizes",
                "2",
                "--kv-chunk-sizes",
                "none,1",
            ]
        )
    )

    assert report["dlogp"]["status"] == "requested"
    for case in report["cases"]:
        dlogp = case["dlogp"]
        assert dlogp["status"] == "available"
        assert dlogp["projection"] == "synthetic_fp32_lm_head_projection"
        assert dlogp["active_token_count"] == 3
        assert dlogp["drift"]["active_count"] == 3
        assert dlogp["drift"]["max_abs"] >= 0.0


def test_report_writes_reproducible_json_artifact(tmp_path):
    output = tmp_path / "ws2-cp-attention-drift.json"
    report = run_benchmark(
        parse_args(
            [
                "--smoke",
                "--no-rope",
                "--tp-world-sizes",
                "1",
                "--cp-world-sizes",
                "1",
                "--kv-chunk-sizes",
                "none",
            ]
        )
    )

    write_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["cases"][0]["provenance"]["rope"]["rope_state"] == "not_composed"
    assert loaded["cases"][0]["attention_mode"] == "prefill"


def test_include_backward_adds_pr8_gradient_drift_report():
    report = run_benchmark(
        parse_args(
            [
                "--smoke",
                "--include-backward",
                "--tp-world-sizes",
                "2",
                "--cp-world-sizes",
                "2",
                "--kv-chunk-sizes",
                "1",
            ]
        )
    )

    backward = report["cases"][0]["backward"]
    assert backward["status"] == "available"
    drift = backward["report"]["drifts"][0]
    assert drift["candidate_name"] == "cp2_chunked_backward"
    assert drift["provenance"]["attention_mode"] == "chunked_prefill"
    assert drift["provenance"]["downcast_at"] == "final_write"
    assert drift["dq"]["max_abs"] <= 5.0e-2
    assert drift["dk"]["max_abs"] <= 5.0e-2
    assert drift["dv"]["max_abs"] <= 5.0e-2
    assert len(drift["per_rank"]) == 2


def test_invalid_qwen3_tp_topology_is_rejected():
    with pytest.raises(ValueError, match="query heads"):
        run_benchmark(
            parse_args(
                [
                    "--tp-world-sizes",
                    "3",
                    "--cp-world-sizes",
                    "1",
                    "--kv-chunk-sizes",
                    "none",
                ]
            )
        )
