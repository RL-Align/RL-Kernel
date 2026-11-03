# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the joint-attention softmax benchmark harness."""

from argparse import Namespace

import pytest
import torch

from benchmarks.operators.attention import benchmark_joint_attn_softmax as benchmark


def test_benchmark_reports_unavailable_when_git_cannot_start(monkeypatch):
    def raise_missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git executable is unavailable")

    monkeypatch.setattr(benchmark.subprocess, "run", raise_missing_git)

    assert benchmark._git_commit() == "unavailable"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)
def test_masked_benchmark_reports_prompt_layout(monkeypatch):
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "test-commit")

    report = benchmark.run_benchmark(
        Namespace(
            rows=2,
            dtype="bf16",
            backward=False,
            warmup=0,
            iterations=1,
            backends=["cuda", "triton", "pytorch"],
            cases={"synthetic": 513},
            valid_text_keys=73,
        )
    )

    assert report["schema_version"] == "rlkernel.joint_attn_softmax_benchmark.v3"
    assert report["configuration"]["valid_text_keys"] == 73
    assert report["configuration"]["warmup"] == 0
    assert report["configuration"]["iterations"] == 1
    assert report["environment"]["git_commit"] == "test-commit"
    assert {record["backend"] for record in report["results"]} == {
        "cuda",
        "triton",
        "pytorch",
    }
    assert all(record["masked_prompt_keys"] == 439 for record in report["results"])
    assert all(record["byte_equal_to_cuda"] for record in report["results"])
