# SPDX-License-Identifier: Apache-2.0
"""The directory migration must preserve state identity and pinned numerical inputs."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import torch

from rl_engine.runtime.provenance.identity import stable_operator_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("legacy", "canonical", "shared_object"),
    [
        ("rl_engine.kernels.registry", "rl_engine.runtime.registry", "kernel_registry"),
        ("rl_engine.runtime_mode", "rl_engine.runtime.policy", "strict_contract_enabled"),
        ("rl_engine.integrations.state", "rl_engine.integrations.common.state", "_ACTIVE"),
        ("rl_engine.kernels.ops.vjp_fp32", "rl_engine.ops.autograd.vjp_fp32", "reduce_rows_fp32"),
        ("rl_engine.kernels.gtest.tolerance", "rl_engine.contracts.numerical", "load_contract"),
    ],
)
def test_legacy_imports_share_module_and_state(legacy, canonical, shared_object):
    old = importlib.import_module(legacy)
    new = importlib.import_module(canonical)
    assert old is new
    assert getattr(old, shared_object) is getattr(new, shared_object)


@pytest.mark.parametrize(
    ("legacy", "canonical", "shared_object"),
    [
        (
            "rl_engine.reference.ffn.ffn",
            (
                "rl_engine.backends.rocm.ffn.ffn"
                if torch.version.hip is not None
                else "rl_engine.backends.cuda.ffn.ffn"
            ),
            "Qwen3FFNOp",
        ),
        (
            "rl_engine.backends.shared.triton.ffn.ffn",
            "rl_engine.backends.rocm.ffn.triton",
            "qwen3_ffn",
        ),
    ],
)
def test_ffn_compatibility_imports_share_platform_modules(legacy, canonical, shared_object):
    old = importlib.import_module(legacy)
    new = importlib.import_module(canonical)
    assert old is new
    assert getattr(old, shared_object) is getattr(new, shared_object)


def test_stable_operator_id_does_not_hide_an_unknown_implementation():
    from rl_engine.reference.gemm.det_gemm import NativeGemmOp

    assert stable_operator_path(NativeGemmOp()) == (
        "rl_engine.kernels.ops.pytorch.matmul.det_gemm.NativeGemmOp"
    )

    class Unregistered:
        pass

    op = Unregistered()
    assert stable_operator_path(op) == f"{type(op).__module__}.{type(op).__qualname__}"


def test_recorded_algorithm_sources_resolve_to_real_implementations():
    from rl_engine.config.workload import load_manifest

    moves = json.loads((ROOT / "tools/migration/layout.json").read_text())["moves"]
    for case in load_manifest().representative_cases:
        source, symbol = case["provenance_evidence"]["algorithm_source"].rsplit(":", 1)
        implementation = ROOT / moves.get(source, source)
        text = implementation.read_text()
        assert "Compatibility import" not in text
        assert symbol in text


def test_unimplemented_models_do_not_register_backends():
    from rl_engine.runtime.registry import kernel_registry

    before = dict(vars(kernel_registry))
    for name in ("deepseek_v4", "gemma", "minimax_h3"):
        module = importlib.import_module(f"rl_engine.models.{name}")
        assert not any(not key.startswith("_") for key in vars(module))
    assert vars(kernel_registry) == before


@pytest.mark.parametrize(
    ("package", "resource", "digest"),
    [
        (
            "rl_engine.config",
            "workloads/qwen3_8b.json",
            "80395644b5e84ed1a76729ffc6f78eb11f41b9ab42fb36e2d00f19635d3df0f6",
        ),
        (
            "rl_engine.contracts",
            "profiles/precision/ws1.json",
            "044167ec21e372017a2758dc09f805b368283c06b83f64061877628cbd101532",
        ),
    ],
)
def test_packaged_contract_bytes_match_main(package, resource, digest):
    assert hashlib.sha256(files(package).joinpath(resource).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "entry",
    [
        "rl_engine/repro.py",
        "scripts/ws1_reference.py",
        "rl_engine/testing/logprob_comparison.py",
        "examples/vime_rocm_attention_ablation/run.py",
    ],
)
def test_legacy_cli_runs_outside_checkout(entry, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / entry), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
