# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fused add RMSNorm registration, canonical identities, and import isolation."""

import importlib
import subprocess
import sys

import pytest
import torch

from rl_engine.reference.norm.fused_add_rmsnorm import NativeFusedAddRMSNormOp
from rl_engine.runtime.provenance.identity import stable_operator_path
from rl_engine.runtime.registry import OpBackend


@pytest.mark.parametrize(
    "backend",
    [OpBackend.PYTORCH_NATIVE_FUSED_ADD_RMSNORM, OpBackend.TRITON_FUSED_ADD_RMSNORM],
)
def test_new_operator_identity_uses_its_canonical_module(backend):
    if backend == OpBackend.TRITON_FUSED_ADD_RMSNORM:
        triton = pytest.importorskip("triton")
        if not hasattr(triton, "jit"):
            pytest.skip("A working Triton runtime is required; type stubs are insufficient")
    module_name, class_name = backend.value.rsplit(".", 1)
    operator = getattr(importlib.import_module(module_name), class_name)()
    assert type(operator).__module__ == module_name
    assert stable_operator_path(operator) == backend.value


def test_norm_packages_do_not_load_optional_triton_or_unrelated_operators():
    # Use a fresh interpreter so earlier tests cannot conceal an eager import.
    code = """
import importlib
import sys

class NoTriton:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton is unavailable in this import check")

sys.meta_path.insert(0, NoTriton())
importlib.import_module("rl_engine.backends.shared.triton.norm")
importlib.import_module("rl_engine.reference.norm.rms_norm")
assert "rl_engine.backends.shared.triton.norm.fused_add_rmsnorm" not in sys.modules
assert "rl_engine.reference.norm.fused_add_rmsnorm" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("platform", ["cpu", "musa", "npu"])
def test_registry_native_dispatch(platform, monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    op = registry.get_op("fused_add_rmsnorm")
    assert isinstance(op, NativeFusedAddRMSNormOp)
    generator = torch.Generator().manual_seed(434)
    inputs = tuple(torch.randn(shape, generator=generator) for shape in ((2, 7), (2, 7), (7,)))
    outputs = op(*inputs)
    assert len(outputs) == 2
    updated = inputs[0] + inputs[1]
    expected_y = (
        updated * torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + 1e-5) * inputs[2]
    )
    for actual, expected in zip(outputs, (expected_y, updated), strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_registry_fallback_when_triton_unavailable(platform, monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    load_backend = registry._load_backend
    attempted = []

    def without_triton(backend):
        attempted.append(backend)
        if backend == OpBackend.TRITON_FUSED_ADD_RMSNORM:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", without_triton)
    assert isinstance(registry.get_op("fused_add_rmsnorm"), NativeFusedAddRMSNormOp)
    assert attempted == [
        OpBackend.TRITON_FUSED_ADD_RMSNORM,
        OpBackend.PYTORCH_NATIVE_FUSED_ADD_RMSNORM,
    ]
