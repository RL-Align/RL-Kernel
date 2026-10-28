# SPDX-License-Identifier: Apache-2.0
"""The explicit operator wheel profile must build native code or fail closed."""

from types import SimpleNamespace

import pytest
import torch

from build_tools import extensions


def test_rope_only_wheel_selects_production_sources(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROPE_ONLY", "1")
    fake_torch = SimpleNamespace(
        __file__=torch.__file__, version=SimpleNamespace(cuda="12.8", hip=None)
    )
    monkeypatch.setattr(
        extensions,
        "_load_torch_extension_tools",
        lambda: (fake_torch, None, lambda **kwargs: kwargs),
    )
    (extension,) = extensions.get_extensions()
    assert extension["name"] == "rl_engine._rope_cuda"
    assert set(extension["sources"]) == {
        "csrc/bindings/rope_module.cpp",
        "csrc/cuda/rope/multi_axis_rope.cu",
    }
    assert "--fmad=false" in extension["extra_compile_args"]["nvcc"]


@pytest.mark.parametrize("hip", [None, "6.2"])
def test_rope_only_wheel_rejects_non_cuda_builds(monkeypatch, hip):
    monkeypatch.setenv("RL_KERNEL_ROPE_ONLY", "1")
    fake_torch = SimpleNamespace(
        __file__=torch.__file__, version=SimpleNamespace(cuda=None, hip=hip)
    )
    monkeypatch.setattr(extensions, "_load_torch_extension_tools", lambda: (fake_torch, None, None))
    with pytest.raises(RuntimeError, match="requires CUDA-enabled PyTorch"):
        extensions.get_extensions()


def test_rope_only_wheel_rejects_missing_torch(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROPE_ONLY", "1")
    monkeypatch.setattr(extensions, "_load_torch_extension_tools", lambda: (None, None, None))
    with pytest.raises(RuntimeError, match="PyTorch is unavailable"):
        extensions.get_extensions()
