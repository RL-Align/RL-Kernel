# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Keep providers with incompatible weight semantics out of norm evidence."""

import sys
import types

import pytest
import torch

from scripts import qwen3_next_norm_reuse_check as reuse


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("uses_effective_weight", [False, True])
def test_megatron_zero_centered_admission(monkeypatch, dtype, uses_effective_weight):
    """Reject the 52fbcbc forward behavior while admitting a corrected provider."""
    module = types.ModuleType("megatron.core.transformer.custom_layers.batch_invariant_kernels")

    class BatchInvariantRMSNormFn:
        @staticmethod
        def apply(x, weight, eps, zero_centered_gamma):
            weight_eff = weight + 1.0 if zero_centered_gamma else weight
            scale = weight_eff if uses_effective_weight else weight
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            return (normalized * scale.float()).to(x.dtype)

    module.BatchInvariantRMSNormFn = BatchInvariantRMSNormFn
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(reuse, "DEV", "cpu")
    monkeypatch.setattr(reuse, "DT", dtype)
    factory = dict(reuse._c1_candidates(32))["megatron"]
    monkeypatch.setattr(reuse, "_c1_candidates", lambda hidden: [("megatron", factory)])

    candidate = reuse.candidates("qwen3_next_rms_norm")["megatron"]
    if not uses_effective_weight:
        assert "failed the zero-weight probe" in candidate["unavailable"]
        assert "fn" not in candidate
        return

    assert candidate["backward"] == "full"
    x = torch.ones(2, 32, dtype=dtype, requires_grad=True)
    weight = torch.zeros(32, dtype=dtype, requires_grad=True)
    output = candidate["fn"](x, weight)
    assert torch.all(output != 0)
    output.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None
