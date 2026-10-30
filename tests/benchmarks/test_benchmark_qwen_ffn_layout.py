# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

import pytest
import torch

from benchmarks.layers import benchmark_qwen_ffn_layout as benchmark
from benchmarks.layers import vllm_batch_invariant_configs as vllm_configs
from benchmarks.layers import vllm_batch_invariant_matmul as vllm_matmul


def _timing(median_ms: float) -> dict[str, object]:
    return {
        "median_ms": median_ms,
        "p95_ms": median_ms * 1.1,
        "min_ms": median_ms * 0.9,
        "max_ms": median_ms * 1.2,
        "samples_ms": [median_ms],
    }


def _vllm_default_config(*, tokens: int, n: int, k: int) -> dict[str, object]:
    return {
        "shape": {"M": tokens, "N": n, "K": k},
        "dtype": "bfloat16",
        "config": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        "selection": "default",
        "is_tuned": False,
        "device_capability": [9, 0],
        "arch_family": "hopper",
    }


def _payload(*, include_profile: bool = True) -> dict[str, object]:
    results = []
    for tokens in (1, 128):
        for direction, baseline_ms, optimized_ms, cublas_ms, vllm_ms in (
            ("forward", 4.0, 2.0, 0.5, 0.75),
            ("forward_backward", 10.0, 6.0, 1.5, None),
        ):
            row = {
                "tokens": tokens,
                "hidden": 4096,
                "intermediate": 12288,
                "dtype": "bfloat16",
                "direction": direction,
                "baseline": _timing(baseline_ms),
                "optimized": _timing(optimized_ms),
                "cublas": _timing(cublas_ms),
                "speedup": baseline_ms / optimized_ms,
                "optimized_speedup_vs_legacy": baseline_ms / optimized_ms,
                "latency_reduction_percent": 100.0 * (1.0 - optimized_ms / baseline_ms),
                "cublas_speedup_vs_optimized": optimized_ms / cublas_ms,
                "optimized_overhead_vs_cublas_percent": 100.0 * (optimized_ms / cublas_ms - 1.0),
            }
            if vllm_ms is not None:
                row.update(
                    {
                        "vllm": _timing(vllm_ms),
                        "vllm_matmul_configs": {
                            "gate": _vllm_default_config(
                                tokens=tokens,
                                n=12288,
                                k=4096,
                            ),
                            "up": _vllm_default_config(
                                tokens=tokens,
                                n=12288,
                                k=4096,
                            ),
                            "down": _vllm_default_config(
                                tokens=tokens,
                                n=4096,
                                k=12288,
                            ),
                        },
                        "vllm_speedup_vs_optimized": optimized_ms / vllm_ms,
                        "optimized_overhead_vs_vllm_percent": 100.0
                        * (optimized_ms / vllm_ms - 1.0),
                        "cublas_speedup_vs_vllm": vllm_ms / cublas_ms,
                        "vllm_overhead_vs_cublas_percent": 100.0 * (vllm_ms / cublas_ms - 1.0),
                    }
                )
            results.append(row)
    zero_mismatch = {
        "inference_output": 0,
        "training_output": 0,
        "baseline_train_infer": 0,
        "optimized_train_infer": 0,
        "dHidden": 0,
        "dGateWeight": 0,
        "dUpWeight": 0,
        "dDownWeight": 0,
    }
    profile = None
    if include_profile:
        profile = {
            "tokens": 128,
            "baseline": {
                "forward": {
                    "direct_copy_kernel": 6,
                    "det_gemm_sm90_kernel": 3,
                    "aten_copy": 6,
                },
                "forward_backward": {
                    "direct_copy_kernel": 18,
                    "det_gemm_sm90_kernel": 9,
                    "aten_copy": 18,
                },
            },
            "optimized": {
                "forward": {
                    "direct_copy_kernel": 0,
                    "det_gemm_sm90_kernel": 3,
                    "aten_copy": 0,
                },
                "forward_backward": {
                    "direct_copy_kernel": 9,
                    "det_gemm_sm90_kernel": 9,
                    "aten_copy": 9,
                },
            },
            "production_matmul": {
                "forward": {
                    "direct_copy_kernel": 0,
                    "det_gemm_sm90_kernel": 0,
                    "aten_copy": 0,
                    "aten_mm": 3,
                    "cuda_gemm_kernels": ["nvjet_sm90_forward"],
                },
                "forward_backward": {
                    "direct_copy_kernel": 0,
                    "det_gemm_sm90_kernel": 0,
                    "aten_copy": 0,
                    "aten_mm": 9,
                    "cuda_gemm_kernels": ["nvjet_sm90_backward"],
                },
            },
        }
    numerical_error = {
        name: {
            "bitwise_mismatch_count": 1,
            "bitwise_mismatch_fraction": 0.01,
            "max_abs": 1e-5,
            "mean_abs": 1e-6,
            "relative_l2": 0.01,
            "max_abs_over_reference_max": 0.02,
            "candidate_finite": 1,
        }
        for name in (
            "inference_output",
            "training_output",
            "dHidden",
            "dGateWeight",
            "dUpWeight",
            "dDownWeight",
        )
    }
    vllm_numerical_error = {
        "bitwise_mismatch_count": 1,
        "bitwise_mismatch_fraction": 0.01,
        "max_abs": 1e-5,
        "mean_abs": 1e-6,
        "relative_l2": 0.01,
        "max_abs_over_reference_max": 0.02,
        "candidate_finite": 1,
    }
    return {
        "environment": {"gpu": "NVIDIA H100", "git_commit": "deadbeef"},
        "methodology": {
            "tokens": [1, 128],
            "hidden": 4096,
            "intermediate": 12288,
            "warmup": 3,
            "samples": 20,
            "training_samples": 10,
            "forward_implementations": ["baseline", "optimized", "cublas", "vllm"],
            "forward_permutation_cycle_length": 24,
            "forward_backward_implementations": ["baseline", "optimized", "cublas"],
            "forward_backward_permutation_cycle_length": 6,
            "vllm_forward_only": True,
            "vllm_matmul_config_by_tokens": {
                str(tokens): {
                    "gate": _vllm_default_config(tokens=tokens, n=12288, k=4096),
                    "up": _vllm_default_config(tokens=tokens, n=12288, k=4096),
                    "down": _vllm_default_config(tokens=tokens, n=4096, k=12288),
                }
                for tokens in (1, 128)
            },
        },
        "command": "CUDA_VISIBLE_DEVICES=0 python benchmark_qwen_ffn_layout.py",
        "results": results,
        "correctness": {
            "1": {
                "mismatch_count": zero_mismatch,
                "production_matmul_numerical_error": numerical_error,
                "vllm_batch_invariant_numerical_error": vllm_numerical_error,
            },
            "128": {
                "mismatch_count": zero_mismatch,
                "production_matmul_numerical_error": numerical_error,
                "vllm_batch_invariant_numerical_error": vllm_numerical_error,
            },
        },
        "vllm_batch_invariance": {
            "status": "pass",
            "contract": "raw BF16 first-row equality against M=1 with identical input rows",
            "reference_tokens": 1,
            "tested_tokens": [1, 128],
            "first_row_bitwise_mismatch_count": {"1": 0, "128": 0},
        },
        "kernel_profile": profile,
    }


def test_parse_tokens_and_default():
    assert benchmark._parse_tokens("1, 8,128") == (1, 8, 128)
    assert benchmark.build_arg_parser().parse_args([]).tokens == (1, 8, 32, 128)
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_tokens("")
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_tokens("1,1")
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_tokens("0,8")


def test_summary_uses_interpolated_p95():
    summary = benchmark._summary_ms([1.0, 2.0, 3.0, 4.0])
    assert summary["median_ms"] == 2.5
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["samples_ms"] == [1.0, 2.0, 3.0, 4.0]


def test_raw_bf16_mismatch_compares_storage_bits():
    left = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert benchmark._raw_bf16_mismatch(left, left.clone()) == 0
    right = torch.tensor([1.0, 3.0], dtype=torch.bfloat16)
    assert benchmark._raw_bf16_mismatch(left, right) == 1


def test_four_way_permutation_cycle_is_unique_and_position_balanced():
    names = ("baseline", "optimized", "cublas", "vllm")
    cycle = benchmark._balanced_permutation_cycle(names)

    assert len(cycle) == 24
    assert len(set(cycle)) == 24
    assert all(set(order) == set(names) for order in cycle)
    for prefix_length, expected_position_count in ((20, 5), (24, 6)):
        prefix = cycle[:prefix_length]
        for name in names:
            assert [
                sum(order[position] == name for order in prefix) for position in range(len(names))
            ] == [expected_position_count] * len(names)


def test_qwen3_8b_vllm_shapes_use_cpu_safe_default_config(monkeypatch):
    monkeypatch.setattr(
        vllm_configs,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        vllm_configs._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["hopper"],
    )
    monkeypatch.setattr(vllm_configs, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    for n, k in ((12288, 4096), (4096, 12288)):
        metadata = vllm_matmul.matmul_config_metadata(128, n, k, torch.bfloat16)
        assert metadata["shape"] == {"M": 128, "N": n, "K": k}
        assert metadata["selection"] == "default"
        assert metadata["is_tuned"] is False
        assert metadata["device_capability"] is None
        assert metadata["config"] == {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        }

    tuned_metadata = vllm_matmul.matmul_config_metadata(
        128,
        12288,
        2048,
        torch.bfloat16,
    )
    assert tuned_metadata["selection"] == "tuned"
    assert tuned_metadata["is_tuned"] is True


def test_vllm_batch_invariance_helper_accepts_and_rejects_cpu_outputs(monkeypatch):
    weights = tuple(torch.empty(0, dtype=torch.bfloat16) for _ in range(3))

    monkeypatch.setattr(
        benchmark,
        "_vllm_batch_invariant_qwen3_ffn",
        lambda hidden, *_weights: hidden.clone(),
    )
    result = benchmark._verify_vllm_batch_invariance(
        token_counts=(8, 2),
        hidden_size=4,
        weights=weights,
        device=torch.device("cpu"),
        seed=123,
    )
    assert result["status"] == "pass"
    assert result["tested_tokens"] == [1, 2, 8]
    assert result["first_row_bitwise_mismatch_count"] == {"1": 0, "2": 0, "8": 0}

    def batch_dependent_output(hidden, *_weights):
        output = hidden.clone()
        output[0, 0] = hidden.shape[0]
        return output

    monkeypatch.setattr(
        benchmark,
        "_vllm_batch_invariant_qwen3_ffn",
        batch_dependent_output,
    )
    with pytest.raises(RuntimeError, match="not batch invariant"):
        benchmark._verify_vllm_batch_invariance(
            token_counts=(2,),
            hidden_size=4,
            weights=weights,
            device=torch.device("cpu"),
            seed=123,
        )


def test_report_csv_and_comparison_svg_are_self_contained(tmp_path):
    payload = _payload()
    report = tmp_path / "report.md"
    results_csv = tmp_path / "results.csv"
    comparison = tmp_path / "qwen_ffn_cublas_comparison.svg"
    forward_backward_comparison = tmp_path / "qwen_ffn_forward_backward_comparison.svg"

    benchmark._write_report(payload, report)
    benchmark._write_csv(payload, results_csv)
    benchmark._write_production_context_figure(payload, comparison)
    benchmark._write_forward_backward_context_figure(
        payload,
        forward_backward_comparison,
    )

    report_text = report.read_text(encoding="utf-8")
    assert "Replayed legacy (ms)" in report_text
    assert "2.00x" in report_text
    assert "Deterministic bitwise consistency" in report_text
    assert "Production matmul numerical agreement" in report_text
    assert "CUDA BLAS (ms)" in report_text
    assert "vLLM batch-invariant (ms)" in report_text
    assert "vLLM batch-invariant forward checks" in report_text
    assert "M=1: 0, M=128: 0" in report_text
    assert "upstream default configuration" in report_text
    assert "18" in report_text
    assert "qwen_ffn_forward_backward_comparison.svg" in report_text
    csv_text = results_csv.read_text(encoding="utf-8")
    assert "baseline_median_ms" in csv_text
    assert "cublas_median_ms" in csv_text
    assert "vllm_median_ms" in csv_text
    assert ET.parse(comparison).getroot().tag.endswith("svg")
    comparison_text = comparison.read_text(encoding="utf-8")
    assert "font-family" in comparison_text
    assert "RL-Kernel optimized deterministic GEMM" in comparison_text
    assert benchmark.VLLM_LABEL in comparison_text
    assert "PR #53247" in comparison_text
    assert benchmark.CUBLAS_LABEL in comparison_text
    assert "RL / vLLM: 2.67x" in comparison_text
    assert "legacy" not in comparison_text.lower()
    assert "backward" not in comparison_text.lower()
    assert "upstream default BF16 fallback configuration" in comparison_text

    assert ET.parse(forward_backward_comparison).getroot().tag.endswith("svg")
    forward_backward_text = forward_backward_comparison.read_text(encoding="utf-8")
    assert "Qwen3 FFN forward plus backward latency comparison" in forward_backward_text
    assert "H=4096, I=12288" in forward_backward_text
    assert forward_backward_text.count("RL-Kernel optimized deterministic GEMM") == 1
    assert forward_backward_text.count(benchmark.CUBLAS_LABEL) == 1
    assert benchmark.VLLM_LABEL not in forward_backward_text
    assert "persistent matmul" not in forward_backward_text
    assert "legacy" not in forward_backward_text.lower()
    assert forward_backward_text.count("RL / cuBLAS: 4.00x") == 2
    assert "vLLM PR #53247 is forward-only" in forward_backward_text
    assert "no vLLM backward path" in forward_backward_text


def test_comparison_figure_uses_payload_shape_metadata(tmp_path):
    payload = _payload()
    payload["methodology"]["hidden"] = 8192
    payload["methodology"]["intermediate"] = 28672
    comparison = tmp_path / "qwen_ffn_cublas_comparison.svg"

    benchmark._write_production_context_figure(payload, comparison)

    assert "H=8192, I=28672" in comparison.read_text(encoding="utf-8")


def test_portable_and_exact_commands(monkeypatch, tmp_path):
    args = benchmark.build_arg_parser().parse_args(
        [
            "--device-index",
            "3",
            "--output-dir",
            str(tmp_path),
            "--skip-profiler",
        ]
    )
    portable = benchmark._portable_command(args)
    assert portable.startswith("python benchmarks/layers/benchmark_qwen_ffn_layout.py")
    assert "CUDA_VISIBLE_DEVICES" not in portable
    assert "--device-index 3" in portable
    assert portable.endswith("--skip-profiler")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--device-index", "3"])
    assert benchmark._exact_invocation().endswith("benchmark.py --device-index 3")
    assert "CUDA_VISIBLE_DEVICES" not in benchmark._exact_invocation()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    assert benchmark._exact_invocation().startswith("CUDA_VISIBLE_DEVICES=5,7 ")


def test_verify_bitwise_checks_public_operator_parity(monkeypatch):
    values = tuple(torch.zeros(4, dtype=torch.bfloat16) for _ in range(4))
    grad_output = torch.zeros(4, dtype=torch.bfloat16)

    def identity(*operator_values):
        return operator_values[0].clone()

    def fake_training_result(operator, operator_values, _grad_output):
        return operator(*operator_values), [value.clone() for value in operator_values]

    monkeypatch.setattr(benchmark, "_legacy_qwen3_ffn", identity)
    monkeypatch.setattr(benchmark, "_optimized_qwen3_ffn", identity)
    monkeypatch.setattr(benchmark, "_production_matmul_qwen3_ffn", identity)
    monkeypatch.setattr(benchmark, "_public_production_matmul_qwen3_ffn", identity)
    monkeypatch.setattr(benchmark, "qwen3_ffn", identity)
    monkeypatch.setattr(benchmark, "_training_result", fake_training_result)

    result = benchmark._verify_bitwise(values, grad_output)
    assert not any(result["mismatch_count"].values())
    assert not any(result["public_parity_mismatch_count"].values())
    assert all(
        metrics["relative_l2"] == 0
        for metrics in result["production_matmul_numerical_error"].values()
    )

    monkeypatch.setattr(
        benchmark,
        "qwen3_ffn",
        lambda *operator_values: operator_values[0].clone().fill_(1),
    )
    with pytest.raises(RuntimeError, match="public qwen3_ffn"):
        benchmark._verify_bitwise(values, grad_output)


def test_numerical_error_reports_non_bitwise_close_values():
    reference = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    candidate = torch.tensor([1.0, 2.015625], dtype=torch.bfloat16)
    metrics = benchmark._require_numerical_agreement(
        reference,
        candidate,
        name="test",
    )
    assert metrics["bitwise_mismatch_count"] == 1
    assert metrics["candidate_finite"] == 1
    assert metrics["relative_l2"] < benchmark.CUBLAS_RELATIVE_L2_LIMIT
