# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""GPU smoke: every WS1 single op is in gtest, and C3/C4 run on real candidates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.kernels.gtest.operator_specs import operator_names

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="WS1 gtest GPU smoke needs CUDA"
)


def _run(script: str, *args: str, timeout: int = 300) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_all_ws1_single_ops_are_registered():
    names = set(operator_names())
    assert {
        "rms_norm",
        "qk_norm",
        "det_gemm",
        "attention",
        "logp",
        "batch_invariant_logp",
        "embedding",
        "lm_head",
        "rope",
        "silu",
        "swiglu",
        "pack",
        "linear_logp",
        "prefix_shared_attention",
    } <= names


@pytest.mark.parametrize(
    ("op", "candidate"),
    [
        ("rms_norm", "cuda"),
        ("qk_norm", "cuda"),
        ("silu", "triton"),
        ("swiglu", "triton"),
        ("rope", "triton"),
        ("pack", "pytorch"),
    ],
)
def test_check_operator_runs_ported_ops(op, candidate):
    _run(
        "scripts/check_operator.py",
        "--op",
        op,
        "--candidate",
        candidate,
        "--device",
        "cuda",
        "--dtype",
        "bf16",
        "--batch",
        "1",
        "--seq",
        "2",
        "--check-grad",
    )


def test_c3_triton_silu_is_bitwise_invariant():
    _run(
        "scripts/check_forward_invariance.py",
        "--op",
        "silu",
        "--candidate",
        "triton",
        "--backend-profile",
        "triton_cuda_bf16",
    )


def test_c4_cuda_rms_norm_is_bitwise_invariant():
    _run(
        "scripts/check_gradient_invariance.py",
        "--op",
        "rms_norm",
        "--candidate",
        "cuda",
        "--backend-profile",
        "cuda_bf16",
    )
