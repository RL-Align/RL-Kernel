# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the independent reference and strict dispatch gates."""
from types import SimpleNamespace

import pytest
import torch

from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import _ReferenceLinear


def test_reference_linear_backward_against_fp64_autograd():
    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(2, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(385, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(385, generator=gen, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_ReferenceLinear.apply, (x, weight, bias), fast_mode=True)


def test_fast_math_build_refused(monkeypatch):
    from rl_engine.backends.cuda.model_specific.qwenimage import adaln_mod_projection as module

    extension = SimpleNamespace(**{name: lambda: None for name in module._SYMBOLS})
    extension.qwenimage_strict_math_compiled = lambda: False
    monkeypatch.setattr(module, "_C", extension)
    monkeypatch.setattr(module, "_EXT_AVAILABLE", True)
    with pytest.raises(RuntimeError, match="without fast math"):
        module.QwenImageAdaLNModProjectionCudaOp()


def test_other_platform_slots_fail_closed():
    from rl_engine.runtime.registry import KernelRegistry

    registry = KernelRegistry()
    for platform in ("rocm", "npu", "musa"):
        assert registry._priority_map[platform]["qwenimage_adaln_mod_projection"] == []
