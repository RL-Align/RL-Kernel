# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fail-closed DeepGEMM capability check for the DSv4 SM90 FP8 GEMMs."""

from __future__ import annotations

import importlib.metadata
import types

import pytest
import torch

from rl_engine.kernels.ops.cuda import deepgemm as dg


@pytest.fixture(autouse=True)
def _fresh_cache():
    dg._require.cache_clear()
    yield
    dg._require.cache_clear()


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """An SM90 device, the pinned version, a CUDA 12.8 nvcc and a stub module."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: dg.DEEPGEMM_VERSION)
    monkeypatch.setattr(dg, "nvcc_version", lambda path: (12, 8))
    monkeypatch.delenv("DG_JIT_NVCC_COMPILER", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv(dg.ALLOW_UNPINNED_ENV, raising=False)
    stub = types.SimpleNamespace(**{name: object() for name in dg.REQUIRED_SYMBOLS})
    stub.get_num_sms = lambda: 132
    monkeypatch.setattr(dg.importlib, "import_module", lambda name: stub)
    return tmp_path


def test_accepts_the_pinned_build(fake_env):
    module, capability = dg.require_deepgemm(0)
    assert capability.pinned and capability.version == dg.DEEPGEMM_VERSION
    assert capability.nvcc_version == (12, 8) and capability.num_sms == 132
    assert dg.deepgemm_available(0)


def test_rejects_non_sm90(fake_env, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    with pytest.raises(dg.DeepGemmUnavailable, match="SM90 only"):
        dg.require_deepgemm(0)
    assert not dg.deepgemm_available(0)


def test_rejects_a_missing_install(fake_env, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(dg.DeepGemmUnavailable, match="not installed"):
        dg.require_deepgemm(0)


def test_rejects_another_build_unless_allowed(fake_env, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.6.1+local")
    with pytest.raises(dg.DeepGemmUnavailable, match="validated against"):
        dg.require_deepgemm(0)
    monkeypatch.setenv(dg.ALLOW_UNPINNED_ENV, "1")
    _, capability = dg.require_deepgemm(0)
    assert not capability.pinned and capability.version == "2.6.1+local"


def test_rejects_an_old_jit_nvcc(fake_env, monkeypatch):
    monkeypatch.setattr(dg, "nvcc_version", lambda path: (12, 4))
    with pytest.raises(dg.DeepGemmUnavailable, match="12.8\\+ is required"):
        dg.require_deepgemm(0)


def test_rejects_a_missing_nvcc(fake_env, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "nowhere"))
    with pytest.raises(dg.DeepGemmUnavailable, match="no nvcc"):
        dg.require_deepgemm(0)


def test_rejects_missing_symbols(fake_env, monkeypatch):
    monkeypatch.setattr(dg.importlib, "import_module", lambda name: types.SimpleNamespace())
    with pytest.raises(dg.DeepGemmUnavailable, match="lacks"):
        dg.require_deepgemm(0)


def test_nvcc_resolution_follows_deepgemm(monkeypatch, tmp_path):
    monkeypatch.delenv("DG_JIT_NVCC_COMPILER", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("CUDA_PATH", "/opt/cuda-path")
    assert dg.jit_nvcc_path() == "/opt/cuda-path/bin/nvcc"
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-home")
    assert dg.jit_nvcc_path() == "/opt/cuda-home/bin/nvcc"
    monkeypatch.setenv("DG_JIT_NVCC_COMPILER", "/opt/explicit/nvcc")
    assert dg.jit_nvcc_path() == "/opt/explicit/nvcc"


def test_real_installation_when_present():
    """On a provisioned SM90 host the real check passes; elsewhere it fails closed."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 required")
    try:
        importlib.metadata.version("deep_gemm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("deep_gemm not installed")
    try:
        module, capability = dg.require_deepgemm()
    except dg.DeepGemmUnavailable as exc:
        pytest.skip(f"DeepGEMM present but not usable here: {exc}")
    assert capability.nvcc_version >= dg.MIN_NVCC
    assert all(hasattr(module, name) for name in dg.REQUIRED_SYMBOLS)
