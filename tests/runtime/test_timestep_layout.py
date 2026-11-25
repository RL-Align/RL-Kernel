# SPDX-License-Identifier: Apache-2.0
"""Timestep migration preserves imports, provenance, and source discovery."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from rl_engine.runtime.provenance.identity import stable_operator_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("legacy", "canonical", "symbol"),
    [
        (
            "rl_engine.kernels.ops.timestep_embed_mlp",
            "rl_engine.ops.embedding.timestep_embed_mlp",
            "TimestepEmbedMLPOp",
        ),
        (
            "rl_engine.kernels.ops.cuda.timestep_embed_mlp",
            "rl_engine.backends.cuda.embedding.timestep_embed_mlp",
            "CudaTimestepEmbedMLPOp",
        ),
        (
            "rl_engine.kernels.ops.triton.timestep_embed_mlp",
            "rl_engine.backends.cuda.embedding.timestep_embed_mlp_triton",
            "TritonTimestepEmbedMLPOp",
        ),
        (
            "rl_engine.kernels.ops.pytorch.timestep_embed_mlp",
            "rl_engine.reference.embedding.timestep_embed_mlp",
            "NativeTimestepEmbedMLPOp",
        ),
    ],
)
def test_timestep_legacy_module_and_provenance_identity(legacy, canonical, symbol):
    if "triton" in canonical:
        pytest.importorskip("triton")
    old = importlib.import_module(legacy)
    new = importlib.import_module(canonical)
    assert old is new
    assert getattr(old, symbol) is getattr(new, symbol)
    assert stable_operator_path(getattr(new, symbol)()) == f"{legacy}.{symbol}"


def test_timestep_jit_discovers_relocated_native_source(monkeypatch):
    import torch.utils.cpp_extension

    from rl_engine.backends.cuda.embedding import timestep_embed_mlp as backend

    calls = []
    extension = object()

    def load(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setenv("TIMESTEP_CUDA_JIT_ONLY", "1")
    monkeypatch.setattr(torch.utils.cpp_extension, "load", load)
    backend._extension.cache_clear()
    try:
        assert backend._extension() is extension
        legacy = importlib.import_module("rl_engine.kernels.ops.cuda.timestep_embed_mlp")
        assert legacy._extension() is extension
        assert len(calls) == 1
        source = ROOT / "csrc/cuda/embedding/timestep_embed_mlp.cu"
        assert source.is_file()
        assert calls[0]["sources"] == [str(source)]
        assert calls[0]["extra_cuda_cflags"] == ["-O3", "--fmad=false"]
    finally:
        backend._extension.cache_clear()


@pytest.mark.parametrize("script", ["validate_timestep_official.py", "audit_timestep_reference.py"])
def test_timestep_commands_import_outside_checkout(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/validation/operators" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
