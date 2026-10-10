# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json

import pytest
import torch

from benchmarks.common.profiler import (
    GPUProfiler,
    PerformanceProfiler,
    available_workloads,
    build_arg_parser,
    run_automated_suite,
)


def test_gpu_profiler_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    info = GPUProfiler.get_target_info()

    assert info.name == "CPU"
    assert info.backend == "cpu"
    assert info.device_index == -1


def test_logp_tflops_estimate_scales_with_shape():
    small = PerformanceProfiler.estimate_logp_tflops(2, 4, 8)
    large = PerformanceProfiler.estimate_logp_tflops(4, 4, 8)

    assert small == pytest.approx(2 * 4 * 8 * 5 / 1e12)
    assert large == pytest.approx(small * 2)


def test_time_kernel_resets_peak_memory_once(monkeypatch):
    profiler = PerformanceProfiler(device=torch.device("cpu"), warmup=1, repeat=3)
    reset_calls = 0
    run_calls = 0

    def reset_memory():
        nonlocal reset_calls
        reset_calls += 1

    def workload():
        nonlocal run_calls
        run_calls += 1
        return run_calls

    monkeypatch.setattr(profiler, "_reset_memory", reset_memory)

    result, latency_ms, latency_std = profiler._time_kernel(workload)

    assert result == 4
    assert latency_ms >= 0.0
    assert latency_std is not None
    assert reset_calls == 1


def test_cpu_smoke_suite_collects_metrics():
    args = argparse.Namespace(
        device="cpu",
        dtype="float32",
        batch_sizes="2",
        seq_lens="4",
        vocab_sizes="32",
        workloads="logp-native,logp-fused",
        top_k=4,
        top_p=0.9,
        seed=0,
        warmup=0,
        repeat=1,
        smoke=True,
    )

    profiler = run_automated_suite(args)
    metrics = profiler.get_metrics()

    assert len(metrics) == 2
    assert metrics[0].benchmark_name == "logp_native"
    assert metrics[0].status == "pass"
    assert metrics[0].tokens_per_sec > 0
    assert metrics[1].benchmark_name == "logp_fused"
    assert metrics[1].status == "blocked"


def test_profiler_report_writers(tmp_path):
    args = argparse.Namespace(
        device="cpu",
        dtype="float32",
        batch_sizes="2",
        seq_lens="4",
        vocab_sizes="32",
        workloads="logp-native",
        top_k=4,
        top_p=0.9,
        seed=0,
        warmup=0,
        repeat=1,
        smoke=True,
    )
    profiler = run_automated_suite(args)

    csv_path = tmp_path / "profile.csv"
    json_path = tmp_path / "profile.json"
    profiler.save_csv(csv_path)
    profiler.save_json(json_path)

    assert "tokens_per_sec" in csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_version"] == "1.0"
    assert payload["metrics"][0]["benchmark_name"] == "logp_native"


def test_sampling_native_cpu_smoke():
    args = argparse.Namespace(
        device="cpu",
        dtype="float32",
        batch_sizes="2",
        seq_lens="4",
        vocab_sizes="32",
        workloads="sampling-native",
        top_k=4,
        top_p=0.9,
        seed=0,
        warmup=0,
        repeat=1,
        smoke=True,
    )

    profiler = run_automated_suite(args)
    metrics = profiler.get_metrics()

    assert len(metrics) == 1
    assert metrics[0].benchmark_name == "sampling_native"
    assert metrics[0].status == "pass"
    assert metrics[0].tokens_per_sec > 0


def test_workload_registry_drives_cli_defaults():
    assert available_workloads() == ("logp-native", "logp-fused", "sampling-native")

    args = build_arg_parser().parse_args([])

    assert args.workloads == ",".join(available_workloads())


def test_unknown_workload_is_rejected():
    args = argparse.Namespace(
        device="cpu",
        dtype="float32",
        batch_sizes="2",
        seq_lens="4",
        vocab_sizes="32",
        workloads="logp-native,missing-op",
        top_k=4,
        top_p=0.9,
        seed=0,
        warmup=0,
        repeat=1,
        smoke=True,
    )

    with pytest.raises(ValueError, match="unsupported workloads: missing-op"):
        run_automated_suite(args)


def test_measure_peak_memory_once(monkeypatch):
    profiler = PerformanceProfiler(device=torch.device("cpu"), warmup=2, repeat=3)

    sync_calls = 0
    reset_calls = 0
    peak_calls = 0
    fn_calls = 0

    def fake_sync():
        nonlocal sync_calls
        sync_calls += 1

    def fake_reset():
        nonlocal reset_calls
        reset_calls += 1

    def fake_peak():
        nonlocal peak_calls
        peak_calls += 1
        return 1.23

    def fake_fn():
        nonlocal fn_calls
        fn_calls += 1

    monkeypatch.setattr(profiler, "_sync", fake_sync)
    monkeypatch.setattr(profiler, "_reset_memory", fake_reset)
    monkeypatch.setattr(profiler, "_peak_memory_gb", fake_peak)

    result = profiler._measure_peak_memory_once(fake_fn)

    assert result == 1.23
    assert fn_calls == 3  # 2 warmup + 1 measured
    assert sync_calls == 2  # after warmup + after measured
    assert reset_calls == 1  # before measured run
    assert peak_calls == 1
