# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.integrations.common import inkling2_head
from rl_engine.integrations.engines.rollout.sglang import inkling2


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_sglang_actual_method_replacement(monkeypatch, platform):
    class Processor:
        use_fp32_lm_head = True
        shard_softcap = 30.0
        final_logit_softcapping = None
        logit_scale = None
        do_tensor_parallel_all_gather = False
        do_tensor_parallel_all_gather_dp_attn = False

        def _compute_lm_head(self, hidden, head, embedding_bias=None, *, out=None):
            return "native"

    mod = ModuleType("sglang.srt.models.inkling2")
    mod.Inkling2LogitsProcessor = Processor
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    lp = ModuleType("sglang.srt.layers.logits_processor")
    lp.should_apply_lm_head_quant_method = lambda head, method: False
    monkeypatch.setitem(sys.modules, lp.__name__, lp)
    observed = []

    def compute(hidden, weight, *, platform):
        observed.append(platform)
        return hidden.float() @ weight.float().t()

    monkeypatch.setattr(inkling2, "raw_head", compute)
    obj = Processor()
    x = torch.ones(1, 4).bfloat16()
    head = SimpleNamespace(weight=torch.full((3, 4), 15).bfloat16())
    with inkling2.bind_sglang_head(obj, platform=platform):
        buf = torch.empty(1, 3)
        assert obj._compute_lm_head(x, head, out=buf) is buf
        torch.testing.assert_close(buf, torch.full_like(buf, 30 * torch.tanh(torch.tensor(2.))))
        assert observed == [platform]
        obj.do_tensor_parallel_all_gather = True
        with pytest.raises(RuntimeError, match="TP1"):
            obj._compute_lm_head(x, head)
    assert obj._compute_lm_head(x, head) == "native"


def test_no_cpu_fallback():
    with pytest.raises(RuntimeError, match="real GPU"):
        inkling2_head.require_platform(torch.ones(1), "cuda")


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_shared_real_gpu_head(monkeypatch, platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} GPU required")
    monkeypatch.setattr(inkling2_head, "HIDDEN", 32)
    monkeypatch.setattr(inkling2_head, "PADDED_VOCAB", 64)
    x = torch.ones(2, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(64, 32, device="cuda", dtype=torch.bfloat16)
    y = inkling2_head.raw_head(x, w, platform=platform)
    assert y.dtype == torch.float32 and torch.equal(y, torch.full_like(y, 32))
