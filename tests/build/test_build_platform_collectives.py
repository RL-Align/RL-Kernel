# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import re
import runpy
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import setuptools
import torch
from torch.utils import cpp_extension

from build_tools.extensions import (
    H3_BITWISE_SOURCES,
    H3_KERNELS_DEFINE,
    _external_relaxed_math,
    _relaxed_math_reason,
    _resolve_package_data_source,
    _tokens_relax_math,
    get_cmdclass,
)

_NVCC_ENV = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS")


def _clear_nvcc_env(monkeypatch) -> None:
    # The build reads these variables; a runner that sets one must not change what a test sees.
    monkeypatch.delenv("KERNEL_ALIGN_USE_FAST_MATH", raising=False)
    for name in _NVCC_ENV:
        monkeypatch.delenv(name, raising=False)


def _load_extension_config(monkeypatch, *, hip: str | None) -> dict[str, Any]:
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
    assert "csrc/rocm/collectives/deterministic_collective.hip" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" not in extension["extra_compile_args"]["cxx"]
    assert "-lcuda" not in extension["extra_link_args"]


def test_cuda_build_keeps_existing_ipc_collective(monkeypatch) -> None:
    extension = _load_extension_config(monkeypatch, hip=None)

    assert "csrc/cuda/collectives/deterministic_collective.cu" in extension["sources"]
    assert "-DKERNEL_ALIGN_WITH_CUDA" in extension["extra_compile_args"]["cxx"]
    assert "-DKERNEL_ALIGN_WITH_ROCM" not in extension["extra_compile_args"]["cxx"]
    assert "-lcuda" in extension["extra_link_args"]


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


def test_cuda_build_compiles_h3_kernels_without_fast_math(monkeypatch) -> None:
    _clear_nvcc_env(monkeypatch)
    extension = _load_extension_config(monkeypatch, hip=None)

    assert set(H3_BITWISE_SOURCES) <= set(extension["sources"])
    assert H3_KERNELS_DEFINE in extension["extra_compile_args"]["cxx"]
    assert H3_KERNELS_DEFINE in extension["extra_compile_args"]["nvcc"]
    assert "--use_fast_math" not in extension["extra_compile_args"]["nvcc"]


def test_fast_math_build_excludes_bitwise_h3_kernels(monkeypatch) -> None:
    # --use_fast_math swaps expf/sinf/cosf/rsqrtf and division for approximations
    # and enables FTZ, which breaks the H3 kernels' bitwise parity promise; nvcc
    # exposes no macro to reject it in-source, so the build drops the kernels
    # and their bindings.
    _clear_nvcc_env(monkeypatch)
    monkeypatch.setenv("KERNEL_ALIGN_USE_FAST_MATH", "1")
    extension = _load_extension_config(monkeypatch, hip=None)

    assert "--use_fast_math" in extension["extra_compile_args"]["nvcc"]
    assert not set(H3_BITWISE_SOURCES) & set(extension["sources"])
    assert not [s for s in extension["sources"] if s.startswith("csrc/cuda/h3/")]
    assert H3_KERNELS_DEFINE not in extension["extra_compile_args"]["cxx"]
    assert H3_KERNELS_DEFINE not in extension["extra_compile_args"]["nvcc"]


@pytest.mark.parametrize("variable", _NVCC_ENV)
@pytest.mark.parametrize(
    "flags",
    [
        "-O2 --use_fast_math --ftz=true",
        "--ftz true",
        "-ftz=1",
        "--prec-div false",
        "-prec-sqrt=false",
        "--use_fast_math;--prec-div=false",
        "'-O2 --use_fast_math'",
    ],
)
def test_relaxed_math_nvcc_flags_from_the_environment_exclude_bitwise_h3_kernels(
    monkeypatch, variable, flags
) -> None:
    # The flags are appended after the ones built here, so a build with
    # NVCC_APPEND_FLAGS='--use_fast_math' would otherwise ship H3 kernels with approximated math.
    _clear_nvcc_env(monkeypatch)
    control = _load_extension_config(monkeypatch, hip=None)
    assert set(H3_BITWISE_SOURCES) <= set(control["sources"])  # the flags are the only difference

    monkeypatch.setenv(variable, flags)
    with pytest.warns(RuntimeWarning, match="bitwise MiniMax-H3 kernels"):
        extension = _load_extension_config(monkeypatch, hip=None)

    assert not [s for s in extension["sources"] if s.startswith("csrc/cuda/h3/")]
    assert H3_KERNELS_DEFINE not in extension["extra_compile_args"]["cxx"]
    assert H3_KERNELS_DEFINE not in extension["extra_compile_args"]["nvcc"]


@pytest.mark.parametrize(
    "flags",
    [
        "--use_fast_math",
        "-use_fast_math",
        "--use_fast_math=true",
        "--ftz=true",
        "-ftz=true",
        "--ftz true",
        "-ftz 1",
        "--ftz",
        "--prec-div=false",
        "--prec-div false",
        "-prec-div 0",
        "--prec-sqrt=false",
        "-prec-sqrt false",
        "-O2;--use_fast_math",
        "-O2 ; --prec-div=false",
        "'-O2 --use_fast_math'",
        "-lineinfo --ftz=true",
    ],
)
def test_predicate_sees_every_spelling_of_relaxed_math(monkeypatch, flags) -> None:
    _clear_nvcc_env(monkeypatch)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", flags)
    assert _external_relaxed_math()


@pytest.mark.parametrize(
    "flags",
    [
        "",
        "-lineinfo --fmad=true",
        "--fmad=false",
        "--ftz=false",
        "--ftz false",
        "-ftz=0",
        "--prec-div=true",
        "--prec-div true -O2",
        "-prec-sqrt 1",
        "-O2 -gencode arch=compute_90,code=sm_90",
        "--use_fast_mathematically",
    ],
)
def test_predicate_keeps_the_h3_kernels_for_unrelated_flags(monkeypatch, flags) -> None:
    _clear_nvcc_env(monkeypatch)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", flags)
    assert not _external_relaxed_math()


def test_predicate_reads_response_files_and_fails_closed(monkeypatch, tmp_path) -> None:
    _clear_nvcc_env(monkeypatch)
    relaxed = tmp_path / "relaxed.opts"
    relaxed.write_text("-O2\n--use_fast_math\n")
    plain = tmp_path / "plain.opts"
    plain.write_text("-lineinfo\n")

    monkeypatch.setenv("NVCC_APPEND_FLAGS", f"@{relaxed}")
    assert _external_relaxed_math()
    monkeypatch.setenv("NVCC_APPEND_FLAGS", f"@{plain}")
    assert not _external_relaxed_math()
    # Nothing is known about a file that cannot be read or a value that cannot be split.
    monkeypatch.setenv("NVCC_APPEND_FLAGS", f"@{tmp_path / 'missing.opts'}")
    assert _external_relaxed_math()
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "'unterminated")
    assert _external_relaxed_math()


_OPTION_FILES = {
    "fast.opts": "--use_fast_math\n",
    "ftz.opts": "-O2\n--ftz=true\n--prec-div=false\n",
    "sqrt.opts": "--prec-sqrt=false",
    "plain.opts": "-lineinfo\n--fmad=false\n",
    "nested.opts": "-optf fast.opts\n",
}


@pytest.mark.parametrize(
    ("flags", "relaxed"),
    [
        ("-optf fast.opts", True),
        ("--options-file fast.opts", True),
        ("--options-file=fast.opts", True),
        ("-options-file=ftz.opts", True),
        ("--options-file plain.opts,sqrt.opts", True),
        ("-optf=plain.opts,fast.opts", True),
        ("-O2 -optf nested.opts", True),
        ("-optf plain.opts", False),
        ("--options-file=plain.opts,plain.opts", False),
        ("@fast.opts", True),
        ("@plain.opts", False),
    ],
)
def test_predicate_reads_the_options_files_nvcc_reads(
    monkeypatch, tmp_path, flags, relaxed
) -> None:
    # nvcc expands ``--options-file`` / ``-optf`` (a comma-separated list) wherever the option
    # comes from, so ``NVCC_APPEND_FLAGS='--options-file flags.txt'`` can switch fast math on.
    _clear_nvcc_env(monkeypatch)
    for name, content in _OPTION_FILES.items():
        (tmp_path / name).write_text(content)
    monkeypatch.chdir(tmp_path)
    for variable in _NVCC_ENV:
        monkeypatch.setenv(variable, flags)
        assert _external_relaxed_math() is relaxed, (variable, flags)
        monkeypatch.delenv(variable)


@pytest.mark.parametrize("flags", ["-optf", "--options-file -O2", "-optf missing.opts"])
def test_options_files_nvcc_cannot_open_count_as_relaxed(monkeypatch, tmp_path, flags) -> None:
    _clear_nvcc_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", flags)
    assert _external_relaxed_math()


def test_the_build_warning_names_the_variable_and_the_reason(monkeypatch, tmp_path) -> None:
    _clear_nvcc_env(monkeypatch)
    missing = str(tmp_path / "missing.opts")
    for flags, reason in [
        ("-O2 --ftz true", "NVCC_APPEND_FLAGS contains --ftz true"),
        ("--prec-div", "NVCC_APPEND_FLAGS contains --prec-div without a value"),
        ("--use_fast_math", "NVCC_APPEND_FLAGS contains --use_fast_math"),
        (f"-optf {missing}", f"NVCC_APPEND_FLAGS: cannot read options file '{missing}'"),
        ("'unterminated", "NVCC_APPEND_FLAGS cannot be parsed"),
    ]:
        monkeypatch.setenv("NVCC_APPEND_FLAGS", flags)
        assert _relaxed_math_reason().startswith(reason), (flags, _relaxed_math_reason())
        with pytest.warns(RuntimeWarning, match=re.escape(reason)):
            _load_extension_config(monkeypatch, hip=None)
    monkeypatch.delenv("NVCC_APPEND_FLAGS")
    assert _relaxed_math_reason() is None


def _nvcc() -> str | None:
    for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if root and (Path(root) / "bin" / "nvcc").exists():
            return str(Path(root) / "bin" / "nvcc")
    return shutil.which("nvcc")


@pytest.mark.skipif(_nvcc() is None, reason="needs nvcc")
@pytest.mark.parametrize(
    "flags",
    [
        "",
        "-lineinfo --fmad=false",
        "--use_fast_math",
        "-use_fast_math",
        "--ftz=true",
        "--ftz=false",
        "-ftz true",
        "--ftz false",
        "--prec-div=false",
        "--prec-div true",
        "-prec-div false",
        "--prec-sqrt=false",
        "-prec-sqrt=true",
        "-optf fast.opts",
        "--options-file=ftz.opts",
        "--options-file plain.opts,sqrt.opts",
        "-optf plain.opts",
        "-optf nested.opts",
    ],
)
@pytest.mark.parametrize("variable", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
def test_predicate_agrees_with_what_nvcc_compiles_with(
    monkeypatch, tmp_path, flags, variable
) -> None:
    # The oracle is nvcc itself: ``--dryrun`` prints the cicc command it would run, whose
    # -ftz / -prec_div / -prec_sqrt say whether the flags turned FP32 math into the fast variants.
    # (nvcc takes only true / false for these options and stops on ``--ftz 1``; the predicate reads
    # 1 / 0 as relaxed anyway, which is the safe side of a build that fails.)
    _clear_nvcc_env(monkeypatch)
    for name, content in _OPTION_FILES.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "k.cu").write_text("__global__ void k(float* a) { a[0] = 1.0f / a[1]; }\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(variable, flags)

    done = subprocess.run(
        [_nvcc(), "-arch=compute_80", "-ptx", "k.cu", "-o", "k.ptx", "--dryrun"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert done.returncode == 0, done.stderr
    cicc = [line for line in done.stderr.splitlines() if "cicc" in line]
    assert cicc, done.stderr
    nvcc_relaxes = any(
        f" {opt} " in f" {cicc[0]} " for opt in ("-ftz=1", "-prec_div=0", "-prec_sqrt=0")
    )
    assert _external_relaxed_math() is nvcc_relaxes, (variable, flags, cicc[0])


def test_no_flag_built_here_relaxes_math_while_the_h3_define_is_set(monkeypatch) -> None:
    # The predicate looks at the environment; this looks at the nvcc list that is handed to the
    # compiler, so a relaxed flag added to the list later cannot ship together with the H3 define.
    _clear_nvcc_env(monkeypatch)
    default = _load_extension_config(monkeypatch, hip=None)["extra_compile_args"]["nvcc"]
    assert H3_KERNELS_DEFINE in default and _tokens_relax_math(default) is None

    monkeypatch.setenv("KERNEL_ALIGN_USE_FAST_MATH", "1")
    fast = _load_extension_config(monkeypatch, hip=None)["extra_compile_args"]["nvcc"]
    assert H3_KERNELS_DEFINE not in fast and _tokens_relax_math(fast) == "--use_fast_math"


def test_unrelated_nvcc_flags_keep_the_h3_kernels(monkeypatch) -> None:
    _clear_nvcc_env(monkeypatch)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-lineinfo --fmad=true")
    extension = _load_extension_config(monkeypatch, hip=None)

    assert set(H3_BITWISE_SOURCES) <= set(extension["sources"])
    assert H3_KERNELS_DEFINE in extension["extra_compile_args"]["nvcc"]
