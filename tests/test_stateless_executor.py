# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from rl_engine.executors.stateless_executor import (
    StatelessForwardConfig,
    StatelessForwardExecutor,
    StatelessForwardInputs,
    StatelessForwardOutputs,
    default_reward_adapter,
    score_reference_logprobs,
    score_rewards,
)
from rl_engine.testing.reference_ops import selected_logprobs_reference


class FakeReferenceModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)
        self.config = SimpleNamespace(use_cache=True, _attn_implementation="eager")
        self.generation_config = SimpleNamespace(use_cache=True)
        self.use_cache_calls: list[bool | None] = []
        self.config_use_cache_calls: list[bool | None] = []
        self.generation_config_use_cache_calls: list[bool | None] = []
        self.attn_implementation_calls: list[str | None] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        del attention_mask
        self.use_cache_calls.append(use_cache)
        self.config_use_cache_calls.append(getattr(self.config, "use_cache", None))
        self.generation_config_use_cache_calls.append(
            getattr(self.generation_config, "use_cache", None)
        )
        self.attn_implementation_calls.append(getattr(self.config, "_attn_implementation", None))
        return SimpleNamespace(
            logits=self.fixed_logits[: input_ids.shape[0], : input_ids.shape[1]],
            past_key_values=None,
        )


class FakeRewardModel(torch.nn.Module):
    def __init__(self, rewards: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_rewards", rewards)
        self.use_cache_calls: list[bool | None] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        del attention_mask
        self.use_cache_calls.append(use_cache)
        return {"logits": self.fixed_rewards[: input_ids.shape[0]].unsqueeze(-1)}


class NoUseCacheModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch, seq_len = input_ids.shape
        vocab = 8
        return torch.zeros(batch, seq_len, vocab, device=input_ids.device)


class CacheReturningModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=None):
        del attention_mask, use_cache
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8, device=input_ids.device)
        key = torch.zeros(batch, 1, seq_len, 4, device=input_ids.device)
        value = torch.zeros_like(key)
        return {"logits": logits, "past_key_values": ((key, value),)}


def _inputs() -> StatelessForwardInputs:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [0, 2, 1, 4, 5],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    completion_mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, True],
        ]
    )
    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def _logits_for(inputs: StatelessForwardInputs, vocab_size: int = 8) -> torch.Tensor:
    logits = torch.full((*inputs.input_ids.shape, vocab_size), -3.0)
    for batch_index in range(inputs.input_ids.shape[0]):
        for pos in range(inputs.input_ids.shape[1] - 1):
            token = int(inputs.input_ids[batch_index, pos + 1].item())
            logits[batch_index, pos, token] = 4.0 + batch_index + pos
    return logits


def test_reference_scoring_matches_next_token_pytorch_reference_and_masks_prompt_tokens():
    inputs = _inputs()
    logits = _logits_for(inputs)

    actual = score_reference_logprobs(logits, inputs)
    shifted = selected_logprobs_reference(
        logits[:, :-1, :],
        inputs.input_ids[:, 1:],
        mask=inputs.completion_mask[:, 1:],
    )
    expected = torch.zeros_like(actual)
    expected[:, 1:] = shifted

    assert torch.allclose(actual, expected)
    assert torch.equal(
        actual[~inputs.completion_mask],
        torch.zeros_like(actual[~inputs.completion_mask]),
    )
    assert actual[0, 2] != 0.0
    assert actual[0, 1] == 0.0


def test_executor_runs_full_sequence_forward_with_use_cache_false_and_detaches_outputs():
    inputs = _inputs()
    logits = _logits_for(inputs).requires_grad_()
    model = FakeReferenceModel(logits)
    executor = StatelessForwardExecutor(
        model,
        StatelessForwardConfig(mode="reference", return_token_scores=True),
    )

    result = executor.score(inputs)

    assert model.use_cache_calls == [False]
    assert result.reference_logps is not None
    assert result.token_scores is not None
    assert result.reference_logps.requires_grad is False
    assert result.token_scores.requires_grad is False
    assert result.metrics["mode"] == "reference"
    assert result.metrics["active_completion_tokens"] == int(inputs.completion_mask.sum().item())
    assert result.metrics["use_cache"] is False
    assert result.metrics["use_cache_passed"] is True
    assert result.metrics["detached_outputs"] is True
    assert result.metrics["zero_kv_cache"] is True
    assert result.metrics["kv_cache_output_tensors"] == 0
    assert result.metrics["attention_backend"] == "flash_attention_2"
    assert result.metrics["attention_backend_configured"] is True
    assert result.metrics["model_config_use_cache_disabled"] is True
    assert model.config_use_cache_calls == [False]
    assert model.generation_config_use_cache_calls == [False]
    assert model.attn_implementation_calls == ["flash_attention_2"]
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True
    assert model.config._attn_implementation == "eager"
    assert not hasattr(model.config, "attn_implementation")
    assert not hasattr(model.generation_config, "attn_implementation")


def test_executor_falls_back_for_models_without_use_cache_argument():
    inputs = _inputs()
    executor = StatelessForwardExecutor(
        NoUseCacheModel(),
        StatelessForwardConfig(mode="reference"),
    )

    result = executor.score(inputs)

    assert result.reference_logps is not None
    assert result.metrics["use_cache_passed"] is False


def test_executor_rejects_models_that_return_kv_cache_outputs():
    inputs = _inputs()
    executor = StatelessForwardExecutor(
        CacheReturningModel(),
        StatelessForwardConfig(mode="reference"),
    )

    with pytest.raises(ValueError, match="KV-cache outputs"):
        executor.score(inputs)


def test_reward_mode_returns_one_scalar_per_sequence_with_default_adapter():
    inputs = _inputs()
    rewards = torch.tensor([1.5, -0.25])
    model = FakeRewardModel(rewards)
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="reward"))

    result = executor.score(inputs)

    assert model.use_cache_calls == [False]
    assert result.reference_logps is None
    assert result.rewards is not None
    assert torch.allclose(result.rewards, rewards)
    assert result.rewards.requires_grad is False
    assert result.metrics["mode"] == "reward"


def test_both_mode_runs_one_forward_and_returns_reference_logps_and_rewards():
    inputs = _inputs()
    logits = _logits_for(inputs)
    rewards = torch.tensor([0.2, 0.8])

    class FakeBothModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            del attention_mask
            assert use_cache is False
            self.calls += 1
            return {
                "logits": logits[: input_ids.shape[0], : input_ids.shape[1]],
                "rewards": rewards,
            }

    model = FakeBothModel()
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="both"))

    result = executor.score(inputs)

    assert model.calls == 1
    assert result.reference_logps is not None
    assert result.rewards is not None
    assert torch.allclose(result.rewards, rewards)


def test_custom_reward_adapter_and_shape_validation():
    inputs = _inputs()
    outputs = StatelessForwardOutputs(raw={"hidden": torch.ones(2, 3)}, logits=None)

    def adapter(model_outputs, batch_inputs):
        assert model_outputs is outputs
        return batch_inputs.attention_mask.float().sum(dim=1)

    rewards = score_rewards(outputs, inputs, reward_adapter=adapter)

    assert torch.equal(rewards, torch.tensor([4.0, 5.0]))

    with pytest.raises(ValueError, match="shape \\[B\\]"):
        score_rewards(outputs, inputs, reward_adapter=lambda _outputs, _inputs: torch.ones(2, 2))

    with pytest.raises(ValueError, match="batch size"):
        score_rewards(outputs, inputs, reward_adapter=lambda _outputs, _inputs: torch.ones(1))


def test_default_reward_adapter_rejects_non_scalar_logits():
    inputs = _inputs()
    outputs = StatelessForwardOutputs(raw={}, logits=torch.zeros(2, 3, 8))

    with pytest.raises(ValueError, match="reward tensor"):
        default_reward_adapter(outputs, inputs)


def test_executor_rejects_shape_mismatches_empty_masks_and_invalid_config():
    inputs = _inputs()
    model = FakeReferenceModel(_logits_for(inputs))
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="reference"))

    with pytest.raises(ValueError, match="attention_mask shape"):
        executor.score(
            StatelessForwardInputs(
                input_ids=inputs.input_ids,
                attention_mask=torch.ones(2, 4, dtype=torch.bool),
                completion_mask=inputs.completion_mask,
            )
        )

    invalid_start_mask = inputs.completion_mask.clone()
    invalid_start_mask[0, 0] = True
    with pytest.raises(ValueError, match=r"completion_mask\[:, 0\]"):
        executor.score(
            StatelessForwardInputs(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                completion_mask=invalid_start_mask,
            )
        )

    with pytest.raises(ValueError, match="completion_mask must contain"):
        executor.score(
            StatelessForwardInputs(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                completion_mask=torch.zeros_like(inputs.completion_mask),
            )
        )

    with pytest.raises(ValueError, match="use_cache"):
        StatelessForwardConfig(use_cache=True)

    with pytest.raises(ValueError, match="attention_backend"):
        StatelessForwardConfig(attention_backend="paged_attention")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="mode"):
        StatelessForwardConfig(mode="generate")  # type: ignore[arg-type]


def test_importing_stateless_executor_does_not_import_heavy_optional_runtimes(monkeypatch):
    for name in ("vllm", "deepspeed", "ray", "flash_attn"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("rl_engine.executors.stateless_executor")

    assert module.StatelessForwardExecutor is not None
    for name in ("vllm", "deepspeed", "ray", "flash_attn"):
        assert name not in sys.modules


def test_reference_scoring_rejects_completion_start_at_position_zero():
    inputs = _inputs()
    invalid_start_mask = inputs.completion_mask.clone()
    invalid_start_mask[0, 0] = True
    invalid_inputs = StatelessForwardInputs(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        completion_mask=invalid_start_mask,
    )

    with pytest.raises(ValueError, match=r"completion_mask\[:, 0\]"):
        score_reference_logprobs(_logits_for(inputs), invalid_inputs)


def test_attention_backend_fallback_does_not_swallow_unrelated_kernel_errors():
    class KernelFailureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True, _attn_implementation="eager")
            self.calls = 0

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            del input_ids, attention_mask, use_cache
            self.calls += 1
            raise RuntimeError("CUDA kernels launch failed")

    model = KernelFailureModel()
    executor = StatelessForwardExecutor(model, StatelessForwardConfig(mode="reference"))

    with pytest.raises(RuntimeError, match="CUDA kernels launch failed"):
        executor.score(_inputs())

    assert model.calls == 1
    assert model.config.use_cache is True
    assert model.config._attn_implementation == "eager"
    assert not hasattr(model.config, "attn_implementation")
