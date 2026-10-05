# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for native-version selection and attention-only JIT refresh."""

from pathlib import Path
from types import ModuleType

import pytest
import torch

import rl_engine
from rl_engine.kernels.ops import base
from rl_engine.kernels.p2 import cuda_runtime
from rl_engine.kernels.p2.attention import mqa_joint_attention_sink as mqa

_VERSION = "mqa_joint_attention_sink_workspace_validation_version"
_ATTENTION = (
    "mqa_joint_attention_sink_forward",
    "mqa_joint_attention_sink_forward_into",
    "mqa_joint_attention_sink_backward",
)
_GEMM = ("det_gemm_fwd_rhs_transposed", "det_gemm_fwd", "det_gemm_db_transposed")


def _module(name, symbols, version=None):
    module = ModuleType(name)
    for symbol in symbols:
        setattr(module, symbol, object())
    if version is not None:
        setattr(module, _VERSION, version)
    return module


@pytest.fixture
def isolated_runtime(monkeypatch):
    monkeypatch.setattr(cuda_runtime, "_MERGED", None)
    monkeypatch.setattr(cuda_runtime, "_prepare_env", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base, "_C", None)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(mqa, "_C", None)
    monkeypatch.setattr(mqa, "_EXT_AVAILABLE", False)


@pytest.mark.parametrize(
    "version, missing",
    [(None, None), (0, None), (1, None), (2, _ATTENTION[1]), (2, _ATTENTION[2])],
)
def test_stale_native_refreshes_only_attention_preserving_module_and_exports(
    isolated_runtime, monkeypatch, version, missing
):
    native = _module("_C", (*_ATTENTION, *_GEMM, "unrelated_operator"), version)
    if missing is not None:
        delattr(native, missing)
    old_exports = vars(native).copy()
    monkeypatch.setattr(rl_engine, "_C", native, raising=False)
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    attention = _module("attention_jit", _ATTENTION, 2)
    calls = []

    def load(name, sources, extra_include_paths=None):
        assert name == "mqa_t06_verify"
        assert sources == [
            str(cuda_runtime._ROOT / "csrc/cuda/attention/mqa_joint_attention_sink.cu"),
            str(cuda_runtime._ROOT / "csrc/cuda/attention/mqa_joint_attention_sink_jitbind.cpp"),
        ]
        calls.append(name)
        return attention

    monkeypatch.setattr(cuda_runtime, "_load", load)
    assert cuda_runtime.ensure_native_kernels() == "jit"
    assert cuda_runtime.ensure_native_kernels() == "jit"
    assert calls == ["mqa_t06_verify"]
    assert base._C is mqa._C is rl_engine._C is native
    assert base._EXT_AVAILABLE and mqa._EXT_AVAILABLE
    for symbol in (*_ATTENTION, _VERSION):
        assert getattr(native, symbol) is getattr(attention, symbol)
    for symbol, value in old_exports.items():
        if symbol not in (*_ATTENTION, _VERSION):
            assert getattr(native, symbol) is value


def test_validated_native_is_reused_without_cuda_or_jit(isolated_runtime, monkeypatch):
    native = _module("_C", (*_ATTENTION, *_GEMM), 2)
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    monkeypatch.setattr(cuda_runtime, "_load", lambda *_args: pytest.fail("JIT requested"))
    assert cuda_runtime.ensure_native_kernels() == "native"
    assert base._C is mqa._C is native
    assert mqa._EXT_AVAILABLE


def test_missing_native_jits_attention_and_gemm_once(isolated_runtime, monkeypatch):
    attention = _module("attention_jit", _ATTENTION, 2)
    gemm = _module("gemm_jit", _GEMM)
    modules = {"mqa_t06_verify": attention, "det_gemm_t06_verify": gemm}
    calls = []

    def load(name, sources, extra_include_paths=None):
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(cuda_runtime, "_load", load)
    assert cuda_runtime.ensure_native_kernels() == "jit"
    assert cuda_runtime.ensure_native_kernels() == "jit"
    assert calls == ["mqa_t06_verify", "det_gemm_t06_verify"]
    assert base._C is mqa._C is cuda_runtime._MERGED
    for module, symbols in ((attention, (*_ATTENTION, _VERSION)), (gemm, _GEMM)):
        for symbol in symbols:
            assert getattr(base._C, symbol) is getattr(module, symbol)


@pytest.mark.parametrize("explicit", [False, True])
def test_prepare_env_preserves_build_choices_and_refreshes_cuda_home(monkeypatch, explicit):
    from torch.utils import cpp_extension

    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/missing/cuda")
    monkeypatch.setattr(cuda_runtime, "resolve_cuda_home", lambda: "/resolved/cuda")
    for name in ("CUDA_HOME", "PATH", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(name, "initial")
    choices = {"CC": "clang", "CXX": "clang++", "TORCH_CUDA_ARCH_LIST": "9.0+PTX"}
    for name, value in choices.items():
        if explicit:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    cuda_runtime._prepare_env()
    assert cpp_extension.CUDA_HOME == "/resolved/cuda"
    assert cuda_runtime.os.environ["CUDA_HOME"] == "/resolved/cuda"
    for name, value in choices.items():
        assert cuda_runtime.os.environ.get(name) == (value if explicit else None)


@pytest.mark.parametrize("cxx, cc", [(None, None), ("/toolchain/clang++", None), ("g++", "gcc")])
def test_load_uses_only_explicit_host_compiler(monkeypatch, cxx, cc):
    from torch.utils import cpp_extension

    for name, value in (("CXX", cxx), ("CC", cc)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    recorded = {}
    result = object()

    def load(**kwargs):
        recorded.update(kwargs)
        return result

    monkeypatch.setattr(cpp_extension, "load", load)
    assert cuda_runtime._load("fixture", ["kernel.cu"]) is result
    compiler_flags = [flag for flag in recorded["extra_cuda_cflags"] if flag.startswith("-ccbin")]
    assert compiler_flags == ([f"-ccbin={cxx}"] if cxx and not cc else [])
    assert recorded["name"] == "fixture"
    assert recorded["sources"] == ["kernel.cu"]


@pytest.mark.parametrize("source", ["caller", "env", "cache", "path", "system", None])
def test_resolve_cuda_home_prioritizes_available_toolkits(monkeypatch, source):
    from torch.utils import cpp_extension

    order = ("caller", "env", "cache", "path", "system")
    homes = {name: f"/toolkits/{name}" for name in order}
    available = order[order.index(source):] if source is not None else ()
    files = {f"{homes[name]}/bin/nvcc" for name in available}
    monkeypatch.setenv("T06_CUDA_HOME", homes["caller"])
    monkeypatch.setenv("CUDA_HOME", homes["env"])
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", homes["cache"])
    monkeypatch.setattr(cuda_runtime.shutil, "which", lambda _name: f"{homes['path']}/bin/nvcc")
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) in files)
    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: [Path(homes["system"])])
    if source is None:
        with pytest.raises(RuntimeError, match="no CUDA toolkit"):
            cuda_runtime.resolve_cuda_home()
    else:
        assert cuda_runtime.resolve_cuda_home() == homes[source]
