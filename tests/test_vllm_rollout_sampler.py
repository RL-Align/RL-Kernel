# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib

import pytest

from rl_engine.executors.rollout import RolloutExecutor
from rl_engine.executors.vllm_sampler import (
    VLLMSamplerConfig,
    VLLMSharedPrefixSampler,
    normalize_grouped_outputs,
)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generate_calls = []
        FakeLLM.instances.append(self)

    def generate(self, prompts, sampling_params):
        self.generate_calls.append((list(prompts), sampling_params))
        return [
            {
                "prompt": prompt,
                "candidate_index": index,
                "request_id": f"request-{index}",
                "prompt_token_ids": [1, 2, 3],
                "outputs": [
                    {
                        "text": f"text-{index}",
                        "token_ids": [index, index + 1],
                        "finish_reason": "length",
                        "cumulative_logprob": -float(index),
                        "logprobs": [{"token": index}],
                    }
                ],
            }
            for index, prompt in enumerate(prompts)
        ]


@pytest.fixture(autouse=True)
def reset_fake_llm():
    FakeLLM.instances.clear()


def test_sampler_config_enables_prefix_caching_by_default():
    config = VLLMSamplerConfig.from_model_config({"model": "tiny-model"})

    assert config.model == "tiny-model"
    assert config.enable_prefix_caching is True
    assert config.num_generations == 1


def test_sampler_config_allows_prefix_caching_disable():
    config = VLLMSamplerConfig.from_model_config(
        {
            "model": "tiny-model",
            "sampler": {"num_generations": 4, "enable_prefix_caching": False},
        }
    )

    assert config.enable_prefix_caching is False
    assert config.num_generations == 4


def test_sampler_construction_does_not_import_vllm(monkeypatch):
    def fail_import(name):
        if name == "vllm":
            raise AssertionError("vLLM should be imported lazily")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", fail_import)

    config = VLLMSamplerConfig(model="tiny-model")
    sampler = VLLMSharedPrefixSampler(config)

    assert sampler.config.enable_prefix_caching is True


def test_vllm_engine_receives_prefix_cache_flag_and_sampling_params():
    config = VLLMSamplerConfig(
        model="tiny-model",
        num_generations=2,
        sampling_params={"temperature": 0.7, "top_p": 0.9},
        engine_kwargs={"dtype": "float16"},
    )
    sampler = VLLMSharedPrefixSampler(
        config,
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
    )

    result = sampler.generate(["prompt-a"])

    assert FakeLLM.instances[0].kwargs == {
        "dtype": "float16",
        "model": "tiny-model",
        "enable_prefix_caching": True,
    }
    prompts, params = FakeLLM.instances[0].generate_calls[0]
    assert prompts == ["prompt-a", "prompt-a"]
    assert params.kwargs == {"temperature": 0.7, "top_p": 0.9}
    assert result["prefix_cache_enabled"] is True
    assert result["outputs"] == [
        [
            {
                "prompt": "prompt-a",
                "candidate_index": 0,
                "request_id": "request-0",
                "prompt_token_ids": [1, 2, 3],
                "outputs": [
                    {
                        "text": "text-0",
                        "token_ids": [0, 1],
                        "finish_reason": "length",
                        "cumulative_logprob": -0.0,
                        "logprobs": [{"token": 0}],
                    }
                ],
            },
            {
                "prompt": "prompt-a",
                "candidate_index": 1,
                "request_id": "request-1",
                "prompt_token_ids": [1, 2, 3],
                "outputs": [
                    {
                        "text": "text-1",
                        "token_ids": [1, 2],
                        "finish_reason": "length",
                        "cumulative_logprob": -1.0,
                        "logprobs": [{"token": 1}],
                    }
                ],
            },
        ]
    ]
    normalized = result["normalized_outputs"][0]
    assert normalized[0].to_dict() == {
        "prompt_index": 0,
        "candidate_index": 0,
        "request_id": "request-0",
        "prompt_token_ids": [1, 2, 3],
        "token_ids": [0, 1],
        "text": "text-0",
        "finish_reason": "length",
        "cumulative_logprob": -0.0,
        "logprobs": [{"token": 0}],
    }


def test_vllm_engine_receives_disabled_prefix_cache_flag():
    config = VLLMSamplerConfig(
        model="tiny-model",
        enable_prefix_caching=False,
    )
    sampler = VLLMSharedPrefixSampler(
        config,
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
    )

    sampler.generate("prompt-a")

    assert FakeLLM.instances[0].kwargs["enable_prefix_caching"] is False


def test_multi_prompt_candidates_preserve_prompt_grouping():
    config = VLLMSamplerConfig(model="tiny-model", num_generations=3)
    sampler = VLLMSharedPrefixSampler(
        config,
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
    )

    result = sampler.generate(["prompt-a", "prompt-b"])

    prompts, _ = FakeLLM.instances[0].generate_calls[0]
    assert prompts == [
        "prompt-a",
        "prompt-a",
        "prompt-a",
        "prompt-b",
        "prompt-b",
        "prompt-b",
    ]
    assert result["num_prompts"] == 2
    assert result["num_generations"] == 3
    assert [[item["prompt"] for item in group] for group in result["outputs"]] == [
        ["prompt-a", "prompt-a", "prompt-a"],
        ["prompt-b", "prompt-b", "prompt-b"],
    ]
    assert [[item.prompt_index for item in group] for group in result["normalized_outputs"]] == [
        [0, 0, 0],
        [1, 1, 1],
    ]
    assert [[item.candidate_index for item in group] for group in result["normalized_outputs"]] == [
        [0, 1, 2],
        [0, 1, 2],
    ]


class ObjectCompletion:
    text = "object text"
    token_ids = (7, 8)
    finish_reason = "stop"
    cumulative_logprob = -2.5
    logprobs = [{"token": 7}]


class ObjectRequestOutput:
    request_id = "object-request"
    prompt_token_ids = (4, 5, 6)
    outputs = [ObjectCompletion()]


def test_normalize_object_request_output():
    normalized = normalize_grouped_outputs([[ObjectRequestOutput()]])

    assert normalized[0][0].to_dict() == {
        "prompt_index": 0,
        "candidate_index": 0,
        "request_id": "object-request",
        "prompt_token_ids": [4, 5, 6],
        "token_ids": [7, 8],
        "text": "object text",
        "finish_reason": "stop",
        "cumulative_logprob": -2.5,
        "logprobs": [{"token": 7}],
    }


def test_rollout_executor_uses_vllm_sampler_config_by_default(monkeypatch):
    class FakeSampler:
        def __init__(self, config):
            self.config = config

        def generate(self, prompts, num_generations=None, sampling_params=None):
            return {
                "backend": "vllm",
                "prefix_cache_enabled": self.config.enable_prefix_caching,
                "prompts": prompts,
                "num_generations": num_generations,
                "sampling_params": sampling_params,
            }

    monkeypatch.setattr("rl_engine.executors.rollout.VLLMSharedPrefixSampler", FakeSampler)

    executor = RolloutExecutor(
        {
            "model": "tiny-model",
            "num_generations": 2,
            "enable_prefix_caching": True,
        }
    )
    result = executor.generate_candidates(["prompt-a"], sampling_params={"max_tokens": 8})

    assert executor.sampler_config.enable_prefix_caching is True
    assert executor.sampler_config.num_generations == 2
    assert result["prefix_cache_enabled"] is True
    assert result["num_generations"] is None
    assert result["sampling_params"] == {"max_tokens": 8}


def test_rollout_executor_defers_vllm_sampler_config_validation():
    executor = RolloutExecutor({"backend": "not-vllm"})

    assert executor.sampler_config is None
    with pytest.raises(ValueError, match="Unsupported rollout sampler backend"):
        executor.generate_candidates(["prompt-a"])
