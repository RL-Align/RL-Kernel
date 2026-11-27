# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import setuptools
import torch
from torch.utils import cpp_extension

from build_tools.extensions import _resolve_package_data_source, get_cmdclass


def _load_extension_config(
    monkeypatch,
    *,
    hip: str | None,
    fast_math: bool = False,
    nvcc_env: dict[str, str] | None = None,
) -> dict[str, Any]:
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
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS", "PYTORCH_NVCC"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (nvcc_env or {}).items():
        monkeypatch.setenv(name, value)
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

    assert "csrc/cuda/collectives/deterministic_collective.cu" not in extension["sources"]
    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "csrc/cuda/gemm/attn_out_bias_gemm.cu" not in extension["sources"]
    assert "csrc/rocm/collectives/deterministic_collective.hip" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" not in extension["extra_compile_args"]["cxx"]
    assert "-lcuda" not in extension["extra_link_args"]


def test_cuda_build_keeps_existing_ipc_collective(monkeypatch) -> None:
    extension = _load_extension_config(monkeypatch, hip=None)

    assert "csrc/cuda/collectives/deterministic_collective.cu" in extension["sources"]
    assert "csrc/cuda/attention/joint_attn_softmax.cu" in extension["sources"]
    assert "csrc/cuda/gemm/attn_out_bias_gemm.cu" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" not in extension["extra_compile_args"]["cxx"]
    assert ("-lcuda" in extension["extra_link_args"]) == (os.name != "nt")


def test_fast_math_build_excludes_bit_reference_softmax(monkeypatch) -> None:
    extension = _load_extension_config(monkeypatch, hip=None, fast_math=True)

    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" not in extension["extra_compile_args"]["cxx"]


@pytest.mark.parametrize(
    ("hip", "fast_math", "nvcc_env", "expected"),
    [
        (None, False, {}, True),
        (None, True, {}, False),
        (None, False, {"NVCC_APPEND_FLAGS": "--ftz=true"}, False),
        (None, False, {"PYTORCH_NVCC": 'ccache "/opt/CUDA Toolkit/bin/nvcc" --ftz=true'}, False),
        (None, False, {"PYTORCH_NVCC": "/opt/cuda/bin/nvcc --use_fast_math"}, False),
        (None, False, {"PYTORCH_NVCC": 'ccache "/opt/CUDA Toolkit/bin/nvcc" --ftz=false'}, True),
        (None, False, {"PYTORCH_NVCC": "env NVCC_APPEND_FLAGS=--ftz=true nvcc"}, False),
        (None, False, {"PYTORCH_NVCC": 'NVCC_PREPEND_FLAGS="-O2 --ftz true" nvcc'}, False),
        (None, False, {"PYTORCH_NVCC": 'env NVCC_APPEND_FLAGS="$EXTRA_NVCC_FLAGS" nvcc'}, False),
        (
            None,
            False,
            {"PYTORCH_NVCC": 'env NVCC_APPEND_FLAGS="-O2 --ftz=false" ccache "/opt/CUDA Kit/nvcc"'},
            True,
        ),
        ("test", False, {}, False),
    ],
)
def test_attn_out_bindings_follow_build_config(monkeypatch, hip, fast_math, nvcc_env, expected):
    warning = (
        pytest.warns(RuntimeWarning, match="attn-out bias GEMM")
        if nvcc_env and not expected
        else nullcontext()
    )
    with warning:
        extension = _load_extension_config(
            monkeypatch, hip=hip, fast_math=fast_math, nvcc_env=nvcc_env
        )
    assert ("csrc/cuda/gemm/attn_out_bias_gemm.cu" in extension["sources"]) == expected
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ preprocessor unavailable")
    source = (Path(__file__).resolve().parents[2] / "csrc/bindings/ops.cpp").read_text()
    # Exercise the actual platform guards without needing CUDA/ROCm headers.
    source = "\n".join(line for line in source.splitlines() if not line.startswith("#include"))
    result = subprocess.run(
        [compiler, "-E", "-P", "-x", "c++", *extension["extra_compile_args"]["cxx"], "-"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    for symbol in (
        "attn_out_bias_gemm_cuda_forward",
        "attn_out_tree_gemm_cuda",
        "attn_out_dw_left_fold_cuda",
    ):
        assert (symbol in result.stdout) == expected


@pytest.mark.parametrize("variable", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
def test_external_fast_math_flags_exclude_bit_reference_softmax(monkeypatch, variable) -> None:
    with pytest.warns(RuntimeWarning, match="joint-attention softmax CUDA bit reference"):
        extension = _load_extension_config(
            monkeypatch,
            hip=None,
            nvcc_env={variable: "--use_fast_math"},
        )

    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" not in extension["extra_compile_args"]["cxx"]


@pytest.mark.parametrize(
    ("variable", "flags"),
    [
        ("NVCC_PREPEND_FLAGS", "--ftz=true"),
        ("NVCC_APPEND_FLAGS", "--ftz true"),
        ("TORCH_NVCC_FLAGS", "--prec-div=false"),
        ("TORCH_NVCC_FLAGS", "--prec-sqrt false"),
        ("NVCC_APPEND_FLAGS", '"-O2 --use_fast_math"'),
    ],
)
def test_external_relaxed_math_spellings_exclude_bit_reference_softmax(
    monkeypatch, variable, flags
) -> None:
    with pytest.warns(RuntimeWarning, match="joint-attention softmax CUDA bit reference"):
        extension = _load_extension_config(
            monkeypatch,
            hip=None,
            nvcc_env={variable: flags},
        )

    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" not in extension["extra_compile_args"]["cxx"]


@pytest.mark.parametrize(
    "flags",
    ["@external.flags", "--options-file=external.flags", "-optf external.flags"],
)
def test_uncheckable_external_options_file_excludes_bit_reference_softmax(
    monkeypatch, flags
) -> None:
    with pytest.warns(RuntimeWarning, match="joint-attention softmax CUDA bit reference"):
        extension = _load_extension_config(
            monkeypatch,
            hip=None,
            nvcc_env={"NVCC_APPEND_FLAGS": flags},
        )

    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" not in extension["extra_compile_args"]["cxx"]


def test_unparseable_external_nvcc_flags_exclude_bit_reference_softmax(monkeypatch) -> None:
    with pytest.warns(RuntimeWarning, match="NVCC_APPEND_FLAGS cannot be parsed"):
        extension = _load_extension_config(
            monkeypatch,
            hip=None,
            nvcc_env={"NVCC_APPEND_FLAGS": "'unterminated"},
        )

    assert "csrc/cuda/attention/joint_attn_softmax.cu" not in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" not in extension["extra_compile_args"]["cxx"]


@pytest.mark.parametrize(
    "flags",
    ["-lineinfo", "--ftz=false --prec-div=true --prec-sqrt true"],
)
def test_safe_external_nvcc_flags_keep_bit_reference_softmax(monkeypatch, flags) -> None:
    extension = _load_extension_config(
        monkeypatch,
        hip=None,
        nvcc_env={"NVCC_APPEND_FLAGS": flags},
    )

    assert "csrc/cuda/attention/joint_attn_softmax.cu" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_JOINT_ATTN_SOFTMAX" in extension["extra_compile_args"]["cxx"]


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


@pytest.mark.parametrize(
    "hip,fast_math,enabled", [(None, "0", True), (None, "1", False), ("test", "0", False)]
)
def test_adaln_build_registers_native_source_only_for_precise_cuda(
    monkeypatch, hip, fast_math, enabled
):
    extension = _load_extension_config(monkeypatch, hip=hip, fast_math=fast_math == "1")
    assert ("csrc/cuda/norm/adaln_modulation.cu" in extension["sources"]) is enabled
    assert ("-DRLK_ADALN_CUDA_ENABLED" in extension["extra_compile_args"]["cxx"]) is enabled
