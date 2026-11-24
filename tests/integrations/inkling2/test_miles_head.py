# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from rl_engine.integrations.common import inkling2_head
from rl_engine.integrations.orchestrators.miles import inkling2


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_miles_forward_backward_and_restore(monkeypatch, platform):
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(5, 4))

        def forward(self, x):
            return x @ self.weight.t()

    class Model(nn.Module):
        unpadded_vocab_size = 200058
        final_logit_softcapping = 30.0
        _skip_lm_head = False

        def __init__(self):
            super().__init__()
            self.lm_head = Head()

    mod = ModuleType("miles.backends.torchtitan_utils.models.inkling2.model")
    mod.Fp32OutLinear = Head
    mod.Inkling2Model = Model
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    seen = []

    def compute(x, weight, *, platform):
        seen.append(platform)
        return x.float() @ weight.float().t()

    monkeypatch.setattr(inkling2, "raw_head", compute)
    model = Model()
    x = torch.ones(2, 3, 4, requires_grad=True)
    with inkling2.bind_miles_head(model, platform=platform):
        y = model.lm_head(x)
        assert y.shape == (2, 3, 5) and seen == [platform]
        y.sum().backward()
        assert torch.equal(x.grad, torch.full_like(x, 5))
        assert torch.equal(model.lm_head.weight.grad, torch.full_like(model.lm_head.weight, 6))
    assert "forward" not in vars(model.lm_head)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_real_gpu_head_gradient(monkeypatch, platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} GPU required")
    monkeypatch.setattr(inkling2_head, "HIDDEN", 32)
    monkeypatch.setattr(inkling2_head, "PADDED_VOCAB", 64)
    x = torch.ones(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.ones(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = inkling2_head.raw_head(x, w, platform=platform)
    out.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 64))
    assert torch.equal(w.grad, torch.full_like(w, 2))
