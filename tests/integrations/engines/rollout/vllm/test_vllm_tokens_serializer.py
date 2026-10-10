# SPDX-License-Identifier: Apache-2.0
"""Keep native token-ID serialization while adapting the older vLLM API."""
import sys
from types import ModuleType, SimpleNamespace

from rl_engine.integrations.engines.rollout.vllm.runtime import _patch_tokens_api_top_logprobs


def _module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__path__ = []
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _protocol(monkeypatch):
    for name in [
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.serve",
    ]:
        _module(monkeypatch, name)
    _module(
        monkeypatch,
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionLogProb=SimpleNamespace,
    )


def test_native_tokens_serializer_is_not_wrapped(monkeypatch):
    _protocol(monkeypatch)
    monkeypatch.delitem(sys.modules, "vllm.entrypoints.serve.disagg.serving", raising=False)
    monkeypatch.delitem(sys.modules, "vllm.entrypoints.serve.disagg", raising=False)

    class ServingTokens:
        def _create_tokens_logprobs(self, *args):
            return "native"

    _module(
        monkeypatch,
        "vllm.entrypoints.scale_out.token_in_token_out.serving",
        ServingTokens=ServingTokens,
    )
    original = ServingTokens._create_tokens_logprobs
    _patch_tokens_api_top_logprobs()
    assert ServingTokens._create_tokens_logprobs is original


def test_legacy_tokens_serializer_keeps_id_repair(monkeypatch):
    _protocol(monkeypatch)
    _module(monkeypatch, "vllm.entrypoints.serve.disagg")

    class ServingTokens:
        def _create_tokens_logprobs(self, token_ids, top_logprobs, limit=None):
            return SimpleNamespace(content=[SimpleNamespace(top_logprobs=[])])

    _module(monkeypatch, "vllm.entrypoints.serve.disagg.serving", ServingTokens=ServingTokens)
    _patch_tokens_api_top_logprobs()
    wrapped = ServingTokens._create_tokens_logprobs
    _patch_tokens_api_top_logprobs()
    assert ServingTokens._create_tokens_logprobs is wrapped
    result = ServingTokens()._create_tokens_logprobs(
        [7],
        [{7: SimpleNamespace(logprob=-0.5), 9: SimpleNamespace(logprob=-2.0)}],
        1,
    )
    assert [(x.token, x.logprob) for x in result.content[0].top_logprobs] == [("token_id:7", -0.5)]
