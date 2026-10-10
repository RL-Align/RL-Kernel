# SPDX-License-Identifier: Apache-2.0
"""Check benchmark admission and cost reporting without treating mocks as hardware evidence."""

import os

import pytest
import torch

from benchmarks.tuning import autotuning_experiment as experiment


def test_cuda_admission_rejects_external_process(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    calls = []

    def processes(command, **kwargs):
        calls.append(command)
        return f"{os.getpid()}\n123456789\n"

    monkeypatch.setattr(experiment.subprocess, "check_output", processes)
    with pytest.raises(RuntimeError, match="occupied"):
        experiment.ensure_idle()
    assert calls[0][2] == "1"
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **kw: str(os.getpid()))
    experiment.ensure_idle()


@pytest.mark.parametrize(
    "listing", ["amdsmi module not found", "GPU:0\nprocess 123456789 uses 10 MB GPU memory"]
)
def test_rocm_admission_fails_closed(monkeypatch, listing):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "list_gpu_processes", lambda _: listing)
    with pytest.raises(RuntimeError):
        experiment.ensure_idle()


def test_rocm_admission_ignores_own_process(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        torch.cuda,
        "list_gpu_processes",
        lambda _: f"GPU:0\nprocess {os.getpid()} uses 10 MB GPU memory",
    )
    experiment.ensure_idle()


@pytest.mark.parametrize(
    "tuned,benefit,break_even",
    [(8, "confirmed", 5000), (12, "unconfirmed", None), (10, "unconfirmed", None)],
)
def test_amortization_requires_confirmed_positive_savings(tuned, benefit, break_even):
    trials = []
    for _ in range(5):
        timing = {"fixed": {"median": 10}, "tuned": {"median": tuned}}
        trials.append(
            {
                "precompiled": {"cases": [{"first_call_ms": 10}]},
                "restart": {"cases": [{"timings_us": {"eager": timing, "graph": timing}}]},
            }
        )
    result = experiment.aggregate_trials([{"id": "test"}], trials)[0]
    assert result["eager"]["benefit"] == benefit
    assert result["eager"]["break_even_calls"] == break_even


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm required")
@pytest.mark.parametrize(
    "name",
    [
        "silu",
        "swiglu",
        "rope",
        "embedding",
        "rmsnorm",
        "ratio_kl",
        "grpo",
        "gemm",
        "attention",
        "logp",
    ],
)
def test_benchmark_paths_without_performance_claims(monkeypatch, tmp_path, name):
    """Exercise the actual benchmark runners; collect no steady-state timings."""
    from rl_engine.backends.shared.triton import autotuning

    monkeypatch.setenv("RL_KERNEL_MODE", "auto")
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
    monkeypatch.setattr(autotuning, "_TUNER", autotuning._TUNER)
    monkeypatch.setattr(autotuning, "_profile_candidates", autotuning._profile_candidates)
    phases = (
        ("forward",)
        if name in ("embedding", "grpo")
        else ("forward", "backward", "forward_backward")
    )
    for phase in phases:
        spec = {
            "operator": name,
            "preset": "small",
            "phase": phase,
            "dtype": "bfloat16",
            "id": phase,
        }
        cold = experiment.experiment_case(spec, str(tmp_path / "reuse"), "cold", validate_only=True)
        restart = experiment.experiment_case(
            spec, str(tmp_path / "reuse"), "restart", validate_only=True
        )
        precompiled = experiment.experiment_case(
            spec, str(tmp_path / "compiled"), "precompiled", validate_only=True
        )
        assert all(x["source"] == "tuned" for x in cold["launches"])
        assert all(x["source"] == "disk" for x in restart["launches"])
        assert all(x["source"] == "tuned" for x in precompiled["launches"])
        assert cold["hot_selection_calls"] == restart["hot_selection_calls"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm required")
def test_benchmark_cache_reused_by_new_process(tmp_path):
    import json
    import subprocess
    import sys

    # Validate actual process restart without producing performance observations.
    specs = [
        dict(operator=name, preset="small", phase="forward_backward", dtype="bfloat16", id=name)
        for name in ("silu", "gemm", "attention", "logp")
    ]
    command = [
        sys.executable,
        "-m",
        "benchmarks.tuning.benchmark_autotuning",
        "--experiment-worker",
        json.dumps(specs),
        "--cache-dir",
        str(tmp_path),
        "--experiment-validate-only",
        "--experiment-mode",
    ]
    cold = json.loads(subprocess.check_output(command + ["cold"], text=True))
    restarted = json.loads(subprocess.check_output(command + ["restart"], text=True))
    assert cold["status"] == restarted["status"] == "validation_only"
    for result in cold["cases"]:
        assert all(record["source"] == "tuned" for record in result["launches"])
    for result in restarted["cases"]:
        assert all(record["source"] == "disk" for record in result["launches"])
        assert result["hot_selection_calls"] == 0
