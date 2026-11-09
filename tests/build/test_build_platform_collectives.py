# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import setuptools
import torch
from torch.utils import cpp_extension

from build_tools import extensions
from build_tools.extensions import _resolve_package_data_source, get_cmdclass


def _load_extension_config(
    monkeypatch,
    *,
    hip: str | None,
    env: dict[str, str] | None = None,
    capability: tuple[int, int] = (8, 0),
) -> list[dict[str, Any]]:
    captured: dict[str, Any] = {}

    def fake_setup(**kwargs: Any) -> None:
        captured.update(kwargs)

    def fake_extension(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setattr(cpp_extension, "CUDAExtension", fake_extension)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.delenv("TORCH_MUSA_ARCH_LIST", raising=False)
    monkeypatch.delenv("FORCE_MUSA", raising=False)
    if hasattr(torch, "musa"):
        monkeypatch.setattr(torch.musa, "is_available", lambda: False)
    monkeypatch.delenv("KERNEL_ALIGN_FORCE_SM90", raising=False)
    monkeypatch.delenv("KERNEL_ALIGN_DET_GEMM_SM90", raising=False)
    monkeypatch.delenv("KERNEL_ALIGN_MOE_SM90", raising=False)
    monkeypatch.delenv("KERNEL_ALIGN_USE_FAST_MATH", raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    if hip is None:
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    else:
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx942")

    runpy.run_path("setup.py", run_name=f"rl_kernel_setup_probe_{hip or 'cuda'}")
    return captured["ext_modules"]


def test_rocm_build_excludes_cuda_ipc_collective_and_driver(monkeypatch) -> None:
    extension, mhc_extension = _load_extension_config(monkeypatch, hip="test")

    assert "csrc/cuda/collectives/deterministic_collective.cu" not in extension["sources"]
    assert "csrc/rocm/collectives/deterministic_collective.hip" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" not in extension["extra_compile_args"]["cxx"]
    assert "-lcuda" not in extension["extra_link_args"]
    assert "csrc/cuda/mhc/mhc_pre_h_aggregate.cu" not in extension["sources"]
    assert mhc_extension["name"] == "rl_engine._C_mhc"
    assert "-ffp-contract=off" in mhc_extension["extra_compile_args"]["nvcc"]
    assert "--use_fast_math" not in mhc_extension["extra_compile_args"]["nvcc"]


def test_cuda_build_keeps_existing_ipc_collective(monkeypatch) -> None:
    extension, mhc_extension = _load_extension_config(monkeypatch, hip=None)

    assert "csrc/cuda/collectives/deterministic_collective.cu" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" not in extension["extra_compile_args"]["cxx"]
    assert "-lcuda" in extension["extra_link_args"]
    assert "csrc/cuda/mhc/mhc_pre_h_aggregate.cu" in extension["sources"]
    assert mhc_extension["name"] == "rl_engine._C_mhc"
    assert mhc_extension["sources"] == ["csrc/cuda/mhc/fp32_gemm_rms.cu"]
    assert "-fmad=false" in mhc_extension["extra_compile_args"]["nvcc"]
    assert "--use_fast_math" not in mhc_extension["extra_compile_args"]["nvcc"]


SM90_MOE_SOURCE = "csrc/cuda/moe/sm90_fused_moe_mlp.cu"


def test_default_cuda_build_excludes_sm90_moe(monkeypatch) -> None:
    extension, _ = _load_extension_config(monkeypatch, hip=None, capability=(9, 0))

    assert SM90_MOE_SOURCE not in extension["sources"]
    assert "-DRL_KERNEL_ENABLE_MOE_SM90" not in extension["extra_compile_args"]["cxx"]


def test_det_gemm_sm90_does_not_bind_sm90_moe(monkeypatch) -> None:
    # det_gemm also defines RL_KERNEL_ENABLE_SM90; the sm90_moe_* bindings must
    # not follow it, or ops.cpp would reference symbols that are not compiled.
    extension, _ = _load_extension_config(
        monkeypatch, hip=None, env={"KERNEL_ALIGN_DET_GEMM_SM90": "1"}, capability=(9, 0)
    )

    assert "-DRL_KERNEL_ENABLE_SM90" in extension["extra_compile_args"]["cxx"]
    assert SM90_MOE_SOURCE not in extension["sources"]
    assert "-DRL_KERNEL_ENABLE_MOE_SM90" not in extension["extra_compile_args"]["cxx"]


def test_moe_sm90_build_compiles_only_sm90a(monkeypatch) -> None:
    extension, _ = _load_extension_config(
        monkeypatch, hip=None, env={"KERNEL_ALIGN_MOE_SM90": "1"}, capability=(9, 0)
    )
    nvcc = extension["extra_compile_args"]["nvcc"]

    assert SM90_MOE_SOURCE in extension["sources"]
    assert "-DRL_KERNEL_ENABLE_MOE_SM90" in nvcc
    assert "-DRL_KERNEL_ENABLE_MOE_SM90" in extension["extra_compile_args"]["cxx"]
    assert "-gencode=arch=compute_90a,code=sm_90a" in nvcc
    # A plain compute_90 pass would also compile the WGMMA/TMA PTX and fail ptxas.
    assert "-gencode=arch=compute_90,code=sm_90" not in nvcc


def test_moe_sm90_build_rejects_non_hopper(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="KERNEL_ALIGN_MOE_SM90=1 requires an SM90"):
        _load_extension_config(
            monkeypatch, hip=None, env={"KERNEL_ALIGN_MOE_SM90": "1"}, capability=(8, 0)
        )


def test_rocm_build_excludes_sm90_moe(monkeypatch) -> None:
    extension, _ = _load_extension_config(
        monkeypatch, hip="test", env={"KERNEL_ALIGN_MOE_SM90": "1"}
    )

    assert SM90_MOE_SOURCE not in extension["sources"]
    assert "-DRL_KERNEL_ENABLE_MOE_SM90" not in extension["extra_compile_args"]["cxx"]


@pytest.mark.parametrize("hip", [None, "test"])
def test_fast_math_is_rejected_on_cuda_and_rocm(monkeypatch, hip) -> None:
    with pytest.raises(RuntimeError, match="KERNEL_ALIGN_USE_FAST_MATH"):
        _load_extension_config(monkeypatch, hip=hip, env={"KERNEL_ALIGN_USE_FAST_MATH": "1"})


def test_force_musa_enables_cross_compile_without_visible_device(monkeypatch) -> None:
    fake_torch = SimpleNamespace(musa=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch_musa", SimpleNamespace())
    monkeypatch.setenv("FORCE_MUSA", "1")
    monkeypatch.delenv("TORCH_MUSA_ARCH_LIST", raising=False)

    assert extensions._musa_build_available(fake_torch)


def test_package_data_json_links_resolve_to_real_content(tmp_path) -> None:
    target = tmp_path / "configs" / "profile.json"
    target.parent.mkdir()
    target.write_text('{"name": "profile"}\n', encoding="utf-8")
    source = tmp_path / "rl_engine" / "profile.json"
    source.parent.mkdir()
    source.symlink_to("../configs/profile.json")

    assert _resolve_package_data_source(source, project_root=tmp_path) == target

    source.unlink()
    source.write_text("../configs/profile.json", encoding="utf-8")
    assert _resolve_package_data_source(source, project_root=tmp_path) == target


def test_package_data_real_json_is_not_reinterpreted(tmp_path) -> None:
    source = tmp_path / "rl_engine" / "profile.json"
    source.parent.mkdir()
    source.write_text('{"name": "profile"}\n', encoding="utf-8")

    assert _resolve_package_data_source(source, project_root=tmp_path) == source
    assert "build_py" in get_cmdclass()
