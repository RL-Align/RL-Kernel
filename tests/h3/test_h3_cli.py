# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The H3 CLIs preserve chain dependencies and require pinned report weights."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_engine.testing import h3_chain
from rl_engine.testing.h3_weights import WEIGHTS_ENV
from scripts import h3_chain_replay, h3_evidence


@pytest.fixture
def cpu_chain_cli(monkeypatch):
    executed = []
    state = {"provider_shift": 0}

    class CpuOp:
        def __call__(self, value):
            return value + 1

        forward_fp32 = __call__

    stages = []
    for index, stage in enumerate(h3_chain.STAGES):

        def stage_input(ctx, upstream, index=index):
            if index == 0:
                return ctx["timestep"]
            assert isinstance(upstream, torch.Tensor), "a preceding stage must run first"
            return upstream

        stages.append(
            replace(
                stage,
                candidate=lambda op, ctx, up, stage_input=stage_input: op(stage_input(ctx, up)),
                golden=lambda op, ctx, up, stage_input=stage_input: op(stage_input(ctx, up)),
                provider=lambda ctx, up, stage_input=stage_input, index=index: (
                    stage_input(ctx, up) + 1 + (state["provider_shift"] if index == 0 else 0)
                ),
            )
        )

    def get_op(op_type, *, device):
        executed.append(op_type)
        return CpuOp()

    monkeypatch.setattr(h3_chain, "STAGES", stages)
    monkeypatch.setattr(h3_chain, "make_context", lambda *a, **kw: {"timestep": torch.zeros(1)})
    monkeypatch.setattr(h3_chain, "golden_op", lambda *a: CpuOp())
    monkeypatch.setattr(h3_chain_replay, "KernelRegistry", lambda: SimpleNamespace(get_op=get_op))
    monkeypatch.setattr(h3_chain_replay, "load_h3_conditioning_weights", lambda *a: {})
    monkeypatch.setattr(h3_chain, "environment", lambda: {})
    monkeypatch.setattr(h3_chain, "git_state", lambda: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    return executed, state


@pytest.mark.parametrize("provider_shift", [0, 1])
@pytest.mark.parametrize(
    "requested, execution_count",
    [
        ("timestep_mlp_fp32", 2),
        ("adaln_projection_3mod", 3),
        ("timestep_sinusoid_h3,adaln_projection_3mod", 3),
        ("timestep_mlp_fp32,timestep_sinusoid_h3", 2),
    ],
)
def test_chain_selection_runs_dependencies(
    cpu_chain_cli, monkeypatch, tmp_path, capsys, requested, execution_count, provider_shift
):
    executed, state = cpu_chain_cli
    state["provider_shift"] = provider_shift
    out = tmp_path / "chain.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "h3_chain_replay.py",
            "--stages",
            requested,
            "--timesteps",
            "1",
            "--seq-lens",
            "3",
            "--out",
            str(out),
        ],
    )
    h3_chain_replay.main()

    report = json.loads(out.read_text())
    stage_names = [stage.name for stage in h3_chain.STAGES]
    expected_report = [name for name in stage_names if name in requested.split(",")]
    assert executed == stage_names[:execution_count]
    assert report["executed_stages"] == executed
    assert report["stages"] == expected_report
    case = report["cases"][0]
    assert [entry["stage"] for entry in case["stages"]] == expected_report
    assert all(entry["chained_vs_golden"]["bitwise_equal"] for entry in case["stages"])
    assert case["first_drift"] == (stage_names[0] if provider_shift else None)
    assert case["first_isolated_drift"] == (stage_names[0] if provider_shift else None)
    summary = capsys.readouterr().out.split("; first_drift=", 1)[0]
    for name in stage_names:
        assert (f"{name}=" in summary) == (name in expected_report)


@pytest.mark.parametrize("op", ["timestep_mlp_fp32", "adaln_projection_3mod"])
@pytest.mark.parametrize("weights_env", [None, "  "])
def test_weighted_evidence_requires_pinned_weights(monkeypatch, tmp_path, op, weights_env):
    if weights_env is None:
        monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    else:
        monkeypatch.setenv(WEIGHTS_ENV, weights_env)
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["h3_evidence.py", "--op", op, "--out", str(out)])
    with pytest.raises(SystemExit, match=WEIGHTS_ENV):
        h3_evidence.main()
    assert not out.exists()


@pytest.mark.parametrize(
    "op, weight_source",
    [
        ("timestep_sinusoid_h3", "not_applicable"),
        ("timestep_mlp_fp32", "pinned_checkpoint"),
        ("adaln_projection_3mod", "pinned_checkpoint"),
    ],
)
def test_evidence_records_weight_source(monkeypatch, tmp_path, op, weight_source):
    if weight_source == "not_applicable":
        monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    else:
        monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(h3_evidence, "KernelRegistry", lambda: object())
    monkeypatch.setattr(
        h3_evidence,
        "git_state",
        lambda: {"rl_kernel_commit": "test_commit", "tracked_tree_dirty": False},
    )
    monkeypatch.setattr(h3_evidence, "environment", lambda: {})
    monkeypatch.setitem(h3_evidence.ACCURACY, op, lambda registry: {})
    monkeypatch.setitem(h3_evidence.PERF_CASES, op, lambda registry: [])
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["h3_evidence.py", "--op", op, "--out", str(out)])
    h3_evidence.main()
    assert json.loads(out.read_text())["weight_source"] == weight_source
