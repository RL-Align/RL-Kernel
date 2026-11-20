# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Exercise pytest environment behavior with controlled external runtime identity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_pytest(script, *nodes, require_ext="0"):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script + "\nimport pytest, sys\nsys.exit(pytest.main(sys.argv[1:]))",
            "-q",
            "-r",
            "s",
            *nodes,
        ],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            CUDA_VISIBLE_DEVICES="",
            RL_KERNEL_REQUIRE_EXT=require_ext,
            RLK_ADALN_REAL_SHAPES="1",
        ),
        capture_output=True,
        text=True,
        timeout=45,
    )
    return result


CPU_WHEEL = """
import torch
torch.version.cuda = None
torch.version.hip = None
torch.cuda.is_available = lambda: False
def no_cuda_init():
    raise AssertionError("Torch not compiled with CUDA enabled")
torch.cuda.init = no_cuda_init
"""


def test_cuda_execution_tests_skip_with_cpu_wheel_identity():
    result = run_pytest(
        CPU_WHEEL,
        "tests/ops/norm/test_adaln_modulation.py::test_triton_adaln_modulation_forward_backward_and_batch_position",
        "tests/ops/norm/test_adaln_modulation.py::test_triton_irregular_hidden_and_bf16_tie_rounding",
        "tests/ops/norm/test_adaln_modulation.py::test_adaln_gpu_padding_and_batch_size_do_not_change_valid_bytes",
        "tests/ops/norm/test_adaln_modulation.py::test_cuda_adaln_bit_equality_harness",
        "tests/ops/norm/test_adaln_modulation.py::test_adaln_real_image_shapes_forward_backward",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "17 skipped" in result.stdout


def test_resolution_with_cpu_wheel_identity_needs_no_physical_gpu():
    pytest.importorskip("triton")
    result = run_pytest(
        CPU_WHEEL,
        "tests/runtime/test_adaln_dispatch.py::test_adaln_cuda_registry_reports_real_fallback",
        "tests/runtime/test_adaln_dispatch.py::test_adaln_registry_skips_cuda_for_large_hidden",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_missing_extension_still_fails_when_required():
    result = run_pytest(
        CPU_WHEEL + '\nimport sys\nsys.modules["rl_engine._C"] = None\n',
        "tests/ops/norm/test_adaln_modulation.py::test_cuda_adaln_bit_equality_harness",
        "tests/runtime/test_adaln_dispatch.py::test_cuda_shared_registry_respects_hidden_limit",
        require_ext="1",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CUDA AdaLN extension symbols are unavailable" in result.stdout
    assert "6 failed, 3 errors" in result.stdout


@pytest.mark.parametrize("hip", [None, "test-HIP-runtime"])
def test_indexed_qualification_tests_with_cpu_wheel_identity(hip):
    result = run_pytest(
        CPU_WHEEL + f"\ntorch.version.hip = {hip!r}\n",
        "tests/runtime/test_adaln_strict_policy.py::test_explicit_indexed_activation_resolves_without_enabling_cpu_select01",
        "tests/runtime/test_adaln_strict_policy.py::test_activated_strict_rechecks_actual_domain_before_execution",
        "tests/runtime/test_adaln_strict_policy.py::test_activation_scope_does_not_qualify_another_backend_or_runtime",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 passed, 4 skipped" in result.stdout


@pytest.mark.parametrize("require_ext", ["0", "1"])
def test_cli_tests_without_cuda_extension_only_skip_optional_cuda_candidate(require_ext):
    result = run_pytest(
        'import sys\nsys.modules["rl_engine._C"] = None\n',
        "tests/validation/operators/test_check_operator_cli.py",
        require_ext=require_ext,
    )
    if require_ext == "1":
        assert result.returncode == 1, result.stdout + result.stderr
        assert "1 failed, 2 passed" in result.stdout
        assert "CUDA AdaLN extension symbols are unavailable" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 passed, 1 skipped" in result.stdout
