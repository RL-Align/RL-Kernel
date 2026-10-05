# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for native-version selection and attention-only JIT refresh."""

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
    [(None, None), (0, None), (1, _ATTENTION[1]), (1, _ATTENTION[2])],
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
    attention = _module("attention_jit", _ATTENTION, 1)
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
    native = _module("_C", (*_ATTENTION, *_GEMM), 1)
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    monkeypatch.setattr(cuda_runtime, "_load", lambda *_args: pytest.fail("JIT requested"))
    assert cuda_runtime.ensure_native_kernels() == "native"
    assert base._C is mqa._C is native
    assert mqa._EXT_AVAILABLE


def test_missing_native_jits_attention_and_gemm_once(isolated_runtime, monkeypatch):
    attention = _module("attention_jit", _ATTENTION, 1)
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


def test_prepare_env_refreshes_cpp_extension_cuda_home(monkeypatch):
    from torch.utils import cpp_extension

    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/missing/cuda")
    monkeypatch.setattr(cuda_runtime, "resolve_cuda_home", lambda: "/resolved/cuda")
    for name in ("CUDA_HOME", "CC", "CXX", "TORCH_CUDA_ARCH_LIST", "PATH", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(name, "initial")
    cuda_runtime._prepare_env()
    assert cpp_extension.CUDA_HOME == "/resolved/cuda"
    assert cuda_runtime.os.environ["CUDA_HOME"] == "/resolved/cuda"
