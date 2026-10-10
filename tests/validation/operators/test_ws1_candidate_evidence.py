# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""GPU acceptance coverage for WS1 C2 representative candidate provenance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_SCRIPT = REPO_ROOT / "tools/validation/operators/ws1_candidate_evidence.py"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="WS1 candidate evidence requires CUDA")
def test_ws1_cuda_and_triton_candidate_runtime_provenance():
    proc = subprocess.run(
        [sys.executable, str(EVIDENCE_SCRIPT), "--emit-json", "-"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["passed"] is True
    assert payload["profiles"] == ["cuda_bf16", "triton_cuda_bf16"]
    assert payload["device"]["index"] == 0
    assert payload["device"]["execution_world_size"] == 1
    # gemm 4 + attention 6 (primary/long/short × 2 profiles) + logprob 4
    assert len(payload["cases"]) == 14
    assert {case["actual_backend_id"] for case in payload["cases"]} == {"cuda", "triton"}
    for case in payload["cases"]:
        assert case["runtime_status"] == "passed"
        assert case["actual_backend_id"] == case["expected_backend_id"]
        assert case["actual_kernel_config_id"] == case["expected_kernel_config_id"]
        assert case["outputs"]
        assert all(output["passed"] for output in case["outputs"])
