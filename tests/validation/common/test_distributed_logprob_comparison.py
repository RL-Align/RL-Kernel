# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import rl_engine.validation.common.distributed_logprob_comparison as distributed_comparison
from rl_engine.contracts.operators.logprob import ShardingSpec
from rl_engine.reference.logprob.vocab_parallel_logp import BACKEND_ID
from rl_engine.validation.common.distributed_logprob_comparison import (
    DistributedLogprobCase,
    _drift_detail,
    _strict_report_json,
    format_launch_command,
    plan_distributed_logprob_cases,
    rank_topology,
    run_distributed_logprob_case,
    token_shard_bounds,
    vocab_shard_bounds,
)
from rl_engine.validation.common.logprob_drift import summarize_logprob_drift


def _small_case(*, tp: int = 1, cp: int = 1) -> DistributedLogprobCase:
    return DistributedLogprobCase(
        tp_world_size=tp,
        cp_world_size=cp,
        real_vocab_size=13,
        padded_vocab_size=16,
        num_vocab_tiles=8,
        batch_size=1,
        sequence_length=4,
        prompt_tokens=1,
        seed=7,
    )


def test_planner_builds_the_scoped_topology_product():
    cases = plan_distributed_logprob_cases(
        real_vocab_size=13,
        padded_vocab_size=16,
        num_vocab_tiles=8,
    )

    assert [(case.tp_world_size, case.cp_world_size) for case in cases] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
    ]
    assert [case.world_size for case in cases] == [1, 2, 2, 4, 4, 8]


def test_rank_mapping_keeps_cp_out_of_the_tp_merge_axis():
    case = _small_case(tp=2, cp=2)

    assert rank_topology(case, 0).tp_group_ranks == (0, 1)
    assert rank_topology(case, 1).tp_group_ranks == (0, 1)
    assert rank_topology(case, 2).tp_group_ranks == (2, 3)
    assert rank_topology(case, 3).tp_group_ranks == (2, 3)
    assert [
        (rank_topology(case, rank).cp_rank, rank_topology(case, rank).tp_rank) for rank in range(4)
    ] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_token_and_vocab_bounds_cover_each_axis_once():
    case = _small_case(tp=4, cp=2)

    assert token_shard_bounds(case.num_tokens, case.cp_world_size) == ((0, 2), (2, 4))
    assert vocab_shard_bounds(case) == ((0, 4), (4, 8), (8, 12), (12, 16))


def test_case_rejects_implicit_backend_and_non_tileable_vocab():
    with pytest.raises(ValueError, match="explicit non-auto backend"):
        DistributedLogprobCase(tp_world_size=2, cp_world_size=1, requested_backend="auto")
    with pytest.raises(ValueError, match="must divide"):
        DistributedLogprobCase(
            tp_world_size=2,
            cp_world_size=1,
            padded_vocab_size=15,
            real_vocab_size=13,
            num_vocab_tiles=8,
        )


def test_launch_command_records_the_materialized_case(tmp_path):
    case = _small_case(tp=2, cp=2)
    command = format_launch_command(case, output=tmp_path / "report.json")

    assert "--nproc-per-node=4" in command
    assert "--tp 2 --cp 2" in command
    assert f"--backend {BACKEND_ID}" in command
    assert "--real-vocab 13 --padded-vocab 16" in command


def test_shared_pr2_drift_summary_preserves_active_mask_semantics():
    candidate = torch.tensor([100.0, 1.0, 3.0])
    reference = torch.tensor([0.0, 2.0, 1.0])
    mask = torch.tensor([False, True, True])

    stats = summarize_logprob_drift(candidate, reference, mask=mask)

    assert stats.active_count == 2
    assert stats.max_abs == 2.0
    assert stats.mean_abs == 1.5


def test_relative_drift_near_zero_stays_finite():
    sharding = ShardingSpec(
        tp_rank=0,
        tp_world_size=1,
        vocab_shard_bounds=((0, 16),),
        real_vocab_size=13,
        padded_vocab_size=16,
    )
    detail = _drift_detail(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        target_ids=torch.tensor([1]),
        global_positions=torch.tensor([3]),
        sharding=sharding,
        atol=0.0,
        rtol=0.0,
    )

    assert math.isfinite(detail.max_rel)
    assert detail.max_rel == pytest.approx(1.0e12)
    assert json.loads(_strict_report_json({"max_rel": detail.max_rel}))["max_rel"] == pytest.approx(
        1.0e12
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        _strict_report_json({"max_rel": float("nan")})


def test_tp1_cpu_case_writes_116_compatible_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    output = tmp_path / "tp1-cp1.json"

    report = run_distributed_logprob_case(
        _small_case(),
        device_name="cpu",
        dist_backend="gloo",
        output=output,
    )

    assert report is not None and report.passed
    assert report.aggregate["lse"].stats.active_count == 4
    assert report.aggregate["dlogp"].stats.active_count == 3
    assert report.ranks[0].actual_backend == BACKEND_ID
    assert report.ranks[0].fallback is False
    assert report.ranks[0].tp_outputs_bitwise_replicated
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["ranks"][0]["contract"]["reduction"]["cp_is_merge_axis"] is False
    assert payload["ranks"][0]["sp_world_size"] == 1
    assert payload["ranks"][0]["dp_world_size"] == 1
    assert payload["environment"]["materialization"]["consistent"] is True
    assert payload["aggregate"]["dlogp"]["worst_target_id"] is not None
    fingerprints = payload["bitwise_fingerprints"]
    assert len(fingerprints["candidate_logp_sha256"]) == 64
    assert len(fingerprints["candidate_lse_sha256"]) == 64
    assert fingerprints["dtype"] == "float32"
    assert fingerprints["shape"] == [4]
    assert payload["launch_command"].startswith("torchrun --standalone")


def test_world_size_mismatch_fails_before_process_group_init(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(RuntimeError, match=r"does not match TP\*CP"):
        run_distributed_logprob_case(
            _small_case(),
            device_name="cpu",
            dist_backend="gloo",
            output=tmp_path / "unused.json",
        )


def test_setup_failure_destroys_owned_process_group(tmp_path, monkeypatch):
    state = {"initialized": False, "destroyed": False}

    def init_process_group(*, backend, timeout):
        state["initialized"] = True

    def destroy_process_group():
        state["destroyed"] = True
        state["initialized"] = False

    def fail_group_setup(case, topology):
        raise RuntimeError("group setup failed")

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "destroy_process_group", destroy_process_group)
    monkeypatch.setattr(distributed_comparison, "_create_tp_group", fail_group_setup)

    with pytest.raises(RuntimeError, match="group setup failed"):
        run_distributed_logprob_case(
            _small_case(tp=2),
            device_name="cpu",
            dist_backend="gloo",
            output=tmp_path / "unused.json",
        )

    assert state == {"initialized": False, "destroyed": True}


def test_partial_initialization_failure_destroys_owned_process_group(tmp_path, monkeypatch):
    state = {"initialized": False, "destroyed": False}

    def fail_initialization(*, backend, timeout):
        state["initialized"] = True
        raise RuntimeError("initialization failed")

    def destroy_process_group():
        state["destroyed"] = True
        state["initialized"] = False

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(dist, "init_process_group", fail_initialization)
    monkeypatch.setattr(dist, "destroy_process_group", destroy_process_group)

    with pytest.raises(RuntimeError, match="initialization failed"):
        run_distributed_logprob_case(
            _small_case(tp=2),
            device_name="cpu",
            dist_backend="gloo",
            output=tmp_path / "unused.json",
        )

    assert state == {"initialized": False, "destroyed": True}


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed required")
def test_tp2_cp2_gloo_cli_emits_per_rank_report(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "rl_engine/validation/common/distributed_logprob_comparison.py"
    )
    output = tmp_path / "tp2-cp2.json"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(script),
            "--tp",
            "2",
            "--cp",
            "2",
            "--device",
            "cpu",
            "--dist-backend",
            "gloo",
            "--real-vocab",
            "13",
            "--padded-vocab",
            "16",
            "--num-vocab-tiles",
            "8",
            "--batch",
            "1",
            "--seq",
            "4",
            "--prompt-tokens",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert len(payload["ranks"]) == 4
    assert all(rank["tp_outputs_bitwise_replicated"] for rank in payload["ranks"])
    assert {rank["actual_backend"] for rank in payload["ranks"]} == {BACKEND_ID}
    assert {rank["cp_rank"] for rank in payload["ranks"]} == {0, 1}
    assert payload["aggregate"]["dlogp"]["stats"]["active_count"] == 3
    assert json.loads(result.stdout)["case"]["tp_world_size"] == 2


@pytest.mark.skipif(torch.version.hip is None, reason="requires a ROCm PyTorch build")
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four ROCm GPUs")
def test_tp2_cp2_rocm_native_cli_emits_strict_report(tmp_path):
    """Run the production ROCm backend across TP2 x CP2 and require provenance."""

    from rl_engine.runtime.registry import _rocm_vocab_logprob_native_available

    if not _rocm_vocab_logprob_native_available():
        pytest.skip("requires the compiled ROCm logprob extension")

    script = (
        Path(__file__).resolve().parents[3]
        / "rl_engine/validation/common/distributed_logprob_comparison.py"
    )
    output = tmp_path / "tp2-cp2-rocm-native.json"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(script),
            "--tp",
            "2",
            "--cp",
            "2",
            "--device",
            "cuda",
            "--dist-backend",
            "nccl",
            "--backend",
            "rocm-vocab-parallel-logp-ws2",
            "--real-vocab",
            "13",
            "--padded-vocab",
            "16",
            "--num-vocab-tiles",
            "8",
            "--batch",
            "1",
            "--seq",
            "4",
            "--prompt-tokens",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert {rank["actual_backend"] for rank in payload["ranks"]} == {"rocm-vocab-parallel-logp-ws2"}
    assert {rank["fallback"] for rank in payload["ranks"]} == {False}
    assert {rank["contract"]["sharding"]["cp_world_size"] for rank in payload["ranks"]} == {2}
    assert json.loads(result.stdout)["case"]["cp_world_size"] == 2
