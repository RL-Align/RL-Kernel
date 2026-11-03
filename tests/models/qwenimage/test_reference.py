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
        for op in ("qwenimage_adaln_mod_projection", "qwenimage_final_adaln_out"):
            assert registry._priority_map[platform][op] == []


def test_reference_modulation_backward_against_fp64_autograd():
    from rl_engine.reference.model_specific.qwenimage.final_adaln_out import _ReferenceModulate

    gen = torch.Generator().manual_seed(2026)
    norm = torch.randn(2, 257, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    scale = torch.randn(2, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    shift = torch.randn(2, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda n, s, b: _ReferenceModulate.apply(n, s, b, torch.float64),
        (norm, scale, shift),
        fast_mode=True,
    )


def test_fp32_reference_retains_small_token_contributions():
    from rl_engine.reference.model_specific.qwenimage.final_adaln_out import _reference_token_sum

    values = torch.ones(1, 6032, 3, dtype=torch.float32)
    values[:, 0] = 2**26
    values[:, -1] = -(2**26)
    golden = values.double().sum(1).float()
    assert torch.equal(_reference_token_sum(values), golden)
