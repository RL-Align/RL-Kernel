# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import pytest
import torch

from rl_engine.executors.stateless_executor import (
    StatelessForwardConfig,
    StatelessForwardExecutor,
    StatelessForwardInputs,
)

transformers = pytest.importorskip("transformers")
from transformers import (  # noqa: E402
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
)


def _inputs(device: torch.device | str = "cpu") -> StatelessForwardInputs:
    resolved_device = torch.device(device)
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 0],
        ],
        dtype=torch.long,
        device=resolved_device,
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, False],
        ],
        device=resolved_device,
    )
    completion_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, True, True, False],
        ],
        device=resolved_device,
    )
    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def _tiny_gpt2_reference_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
        use_cache=True,
    )
    return GPT2LMHeadModel(config).eval()


def _tiny_bert_reward_model() -> BertForSequenceClassification:
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
    )
    return BertForSequenceClassification(config).eval()


def test_stateless_reference_scores_real_hf_causal_lm_without_kv_cache():
    torch.manual_seed(47)
    model = _tiny_gpt2_reference_model()
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="reference"))

    result = executor.score(_inputs())

    assert result.reference_logps is not None
    assert result.reference_logps.shape == (2, 5)
    assert torch.equal(
        result.reference_logps[~_inputs().completion_mask],
        torch.zeros_like(result.reference_logps[~_inputs().completion_mask]),
    )
    assert result.metrics["use_cache_passed"] is True
    assert result.metrics["zero_kv_cache"] is True
    assert result.metrics["kv_cache_output_tensors"] == 0
    assert result.metrics["attention_backend_fallback"] is True
    assert result.metrics["attention_backend"] == "eager"
    assert model.config.use_cache is True


def test_stateless_reward_scores_real_hf_sequence_classifier():
    torch.manual_seed(48)
    model = _tiny_bert_reward_model()
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="reward"))

    result = executor.score(_inputs())

    assert result.rewards is not None
    assert result.rewards.shape == (2,)
    assert result.rewards.requires_grad is False
    assert result.metrics["zero_kv_cache"] is True
    assert result.metrics["attention_backend_fallback"] is True
    assert result.metrics["attention_backend"] == "eager"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stateless_reference_scores_real_hf_causal_lm_on_cuda_without_kv_cache():
    torch.manual_seed(49)
    device = torch.device("cuda")
    model = _tiny_gpt2_reference_model().to(device=device)
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="reference"))

    result = executor.score(_inputs(device))

    assert result.reference_logps is not None
    assert result.reference_logps.is_cuda
    assert result.metrics["zero_kv_cache"] is True
    assert result.metrics["kv_cache_output_mb"] == 0.0
    assert result.metrics["attention_backend_fallback"] is True
    assert result.metrics["attention_backend"] == "eager"
    assert "peak_allocated_mb" in result.metrics
