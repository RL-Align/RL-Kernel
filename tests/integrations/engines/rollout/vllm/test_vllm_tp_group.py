import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm import runtime as vllm_runtime


@pytest.mark.parametrize("tp_world", [1, 2, 4, 8])
def test_qwen_logprob_context_keeps_explicit_tp_group(monkeypatch, tp_world):
    """TP1 must not expand to the global CP group inside strict scoring."""
    monkeypatch.setattr(torch.version, "hip", None)
    group = object()
    published = []

    class Qwen:
        def __init__(self):
            self.lm_head = SimpleNamespace(
                weight=torch.ones(8, 3, dtype=torch.bfloat16),
                shard_indices=SimpleNamespace(padded_org_vocab_start_index=0),
                num_embeddings_padded=8 * tp_world,
                org_vocab_size=8 * tp_world,
            )

        def compute_logits(self, hidden):
            return hidden

    distributed = ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    distributed.get_tp_group = lambda: SimpleNamespace(world_size=tp_world, device_group=group)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    for name in ("qwen2", "qwen3"):
        module = ModuleType("vllm.model_executor.models." + name)
        setattr(module, name.capitalize() + "ForCausalLM", Qwen)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        vllm_runtime,
        "publish_rollout_linear_logp_context",
        lambda *args, **kwargs: published.append(kwargs),
    )
    integration = SimpleNamespace(record_installed_hook=lambda *args: None)
    vllm_runtime._patch_qwen_compute_logits(integration)
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    assert Qwen().compute_logits(hidden) is hidden
    assert len(published) == 1
    assert published[0]["tp_group"] is group
