# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for timestep compilation policy; no compiler is executed."""

import builtins
import importlib

import pytest
from torch.utils import cpp_extension

from rl_engine.backends.cuda.embedding import timestep_embed_mlp as timestep


@pytest.fixture
def jit_loader(monkeypatch):
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIMESTEP_CUDA_JIT_ONLY", "1")
    timestep._extension.cache_clear()
    yield timestep._extension
    timestep._extension.cache_clear()


@pytest.mark.parametrize(
    "variable", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS"]
)
@pytest.mark.parametrize(
    "flags",
    [
        "--use_fast_math",
        "--ftz=true",
        "--prec-div=false",
        "--prec-sqrt=false",
        "--fmad=true",
        "--fmad true",
        "--fmad=1",
        "--fmad",
        "--fmad=maybe",
        "'unterminated",
        "@external.flags",
        "--options-file=external.flags",
        "-optf external.flags",
    ],
)
def test_jit_rejects_unverifiable_math_before_compiler(jit_loader, monkeypatch, variable, flags):
    monkeypatch.setenv(variable, flags)

    def forbidden_load(**kwargs):
        pytest.fail("unsafe timestep request reached the compiler entry point")

    monkeypatch.setattr(cpp_extension, "load", forbidden_load)
    with pytest.raises(
        RuntimeError, match="Cannot JIT-build strict timestep CUDA extension"
    ) as error:
        jit_loader()
    assert variable in str(error.value)
    assert jit_loader.cache_info().currsize == 0


@pytest.mark.parametrize(
    "variable", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS"]
)
@pytest.mark.parametrize(
    "flags",
    [
        "",
        "-lineinfo",
        "--ftz=false --prec-div=true --prec-sqrt=true --fmad=false",
        "--ftz false;--prec-div true;--prec-sqrt true;--fmad false",
    ],
)
def test_jit_safe_math_keeps_exact_original_load_arguments(
    jit_loader, monkeypatch, variable, flags
):
    monkeypatch.setenv(variable, flags)
    calls = []
    compiled = object()

    def fake_load(**kwargs):
        calls.append(kwargs)
        return compiled

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    assert jit_loader() is compiled
    source = (
        timestep.Path(timestep.__file__).resolve().parents[4]
        / "csrc/cuda/embedding/timestep_embed_mlp.cu"
    )
    assert calls == [
        {
            "name": "timestep_official_cuda",
            "sources": [str(source)],
            "extra_cuda_cflags": ["-O3", "--fmad=false"],
            "extra_cflags": ["-O3"],
            "verbose": False,
        }
    ]


def test_safe_loaded_jit_module_does_not_recompile_when_environment_changes(
    jit_loader, monkeypatch
):
    calls = []
    compiled = object()

    def fake_load(**kwargs):
        calls.append(kwargs)
        return compiled

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    assert jit_loader() is compiled
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "--ftz=true --fmad=true")
    assert jit_loader() is compiled
    assert len(calls) == 1


def test_failed_guard_does_not_poison_later_safe_jit_request(jit_loader, monkeypatch):
    compiled = object()
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: compiled)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "--ftz=true")
    with pytest.raises(RuntimeError, match="Cannot JIT-build strict timestep"):
        jit_loader()
    monkeypatch.delenv("NVCC_APPEND_FLAGS")
    assert jit_loader() is compiled


def test_existing_prebuilt_import_needs_no_checkout_guard_or_new_compilation(
    jit_loader, monkeypatch
):
    monkeypatch.delenv("TIMESTEP_CUDA_JIT_ONLY")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "--ftz=true --fmad=true")
    prebuilt = object()
    imports = []
    original_import = builtins.__import__

    def import_prebuilt(name):
        imports.append(name)
        assert name == "rl_engine._timestep_cuda"
        return prebuilt

    def without_build_helpers(name, *args, **kwargs):
        if name.startswith("build_tools"):
            pytest.fail("prebuilt module import tried to load checkout-only build helpers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_prebuilt)
    monkeypatch.setattr(builtins, "__import__", without_build_helpers)
    monkeypatch.setattr(
        cpp_extension, "load", lambda **kwargs: pytest.fail("prebuilt import compiled")
    )
    assert jit_loader() is prebuilt
    assert jit_loader() is prebuilt
    assert imports == ["rl_engine._timestep_cuda"]


def test_missing_prebuilt_uses_checked_checkout_jit(jit_loader, monkeypatch):
    monkeypatch.delenv("TIMESTEP_CUDA_JIT_ONLY")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "--fmad=true")

    def missing_prebuilt(name):
        raise ModuleNotFoundError("prebuilt absent", name=name)

    monkeypatch.setattr(importlib, "import_module", missing_prebuilt)
    monkeypatch.setattr(
        cpp_extension, "load", lambda **kwargs: pytest.fail("unsafe fallback compiled")
    )
    with pytest.raises(RuntimeError, match="Cannot JIT-build strict timestep"):
        jit_loader()


def test_prebuilt_dependency_error_is_not_replaced_by_jit(jit_loader, monkeypatch):
    monkeypatch.delenv("TIMESTEP_CUDA_JIT_ONLY")

    def missing_dependency(name):
        raise ModuleNotFoundError("prebuilt dependency absent", name="prebuilt_dependency")

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    monkeypatch.setattr(
        cpp_extension, "load", lambda **kwargs: pytest.fail("dependency error compiled")
    )
    with pytest.raises(ModuleNotFoundError) as error:
        jit_loader()
    assert error.value.name == "prebuilt_dependency"


def test_missing_checkout_source_does_not_load_build_helpers(jit_loader, monkeypatch):
    original_import = builtins.__import__

    def without_build_helpers(name, *args, **kwargs):
        if name.startswith("build_tools"):
            pytest.fail("missing-source installation tried checkout build helpers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(timestep.Path, "is_file", lambda path: False)
    monkeypatch.setattr(builtins, "__import__", without_build_helpers)
    monkeypatch.setattr(
        cpp_extension, "load", lambda **kwargs: pytest.fail("missing source compiled")
    )
    with pytest.raises(RuntimeError, match="checkout CUDA source is absent"):
        jit_loader()
