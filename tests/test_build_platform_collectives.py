# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import runpy
from typing import Any

import setuptools
import torch
from torch.utils import cpp_extension


def _load_extension_config(
    monkeypatch, *, hip: str | None, fast_math: bool = False
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_setup(**kwargs: Any) -> None:
        captured.update(kwargs)

    def fake_extension(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setattr(cpp_extension, "CUDAExtension", fake_extension)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.delenv("KERNEL_ALIGN_FORCE_SM90", raising=False)
    monkeypatch.delenv("KERNEL_ALIGN_DET_GEMM_SM90", raising=False)
    if fast_math:
        monkeypatch.setenv("KERNEL_ALIGN_USE_FAST_MATH", "1")
    else:
        monkeypatch.delenv("KERNEL_ALIGN_USE_FAST_MATH", raising=False)
    if hip is None:
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    else:
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx942")

    runpy.run_path("setup.py", run_name=f"rl_kernel_setup_probe_{hip or 'cuda'}")
    return captured["ext_modules"][0]


def test_rocm_build_excludes_cuda_ipc_collective_and_driver(monkeypatch) -> None:
    extension = _load_extension_config(monkeypatch, hip="test")

    assert "csrc/cuda/distributed/deterministic_collective.cu" not in extension["sources"]
    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "csrc/rocm/distributed/deterministic_collective.hip" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" not in extension["extra_compile_args"]["cxx"]
    assert "-lcuda" not in extension["extra_link_args"]


def test_cuda_build_keeps_existing_ipc_collective(monkeypatch) -> None:
    extension = _load_extension_config(monkeypatch, hip=None)

    assert "csrc/cuda/distributed/deterministic_collective.cu" in extension["sources"]
    assert "csrc/cuda/attention/joint_attn_softmax.cu" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" not in extension["extra_compile_args"]["cxx"]
    assert ("-lcuda" in extension["extra_link_args"]) == (os.name != "nt")


def test_fast_math_build_excludes_bit_reference_softmax(monkeypatch) -> None:
    extension = _load_extension_config(monkeypatch, hip=None, fast_math=True)

    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" not in extension["extra_compile_args"]["cxx"]
