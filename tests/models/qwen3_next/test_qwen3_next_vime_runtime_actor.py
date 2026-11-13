# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU check of which native setup path ``create_actor`` takes; no vime or Megatron needed."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from rl_engine.integrations.orchestrators.vime.providers import qwen3_next_vime_runtime as runtime


def _module(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def native(monkeypatch):
    """Stub every native import so only the branching in ``create_actor`` runs."""
    calls = []
    args = SimpleNamespace(vocab_size=151936, padded_vocab_size=151936)
    model = object()

    def setup_model_and_optimizer(args, role):
        calls.append("setup_model_and_optimizer")
        return [model], "optimizer", "scheduler"

    def get_model(provider, model_type, wrap_with_ddp):
        calls.append(("get_model", wrap_with_ddp))
        return [model]

    def load_checkpoint(model, optimizer, scheduler, checkpointing_context):
        calls.append(("load_checkpoint", optimizer, scheduler))

    stubs = {
        "vime": _module("vime"),
        "vime.backends": _module("vime.backends"),
        "vime.backends.megatron_utils": _module("vime.backends.megatron_utils"),
        "vime.backends.megatron_utils.initialize": _module("i", init=lambda args: None),
        "vime.backends.megatron_utils.checkpoint": _module("c", load_checkpoint=load_checkpoint),
        "vime.backends.megatron_utils.linear_logp_provider": _module(
            "l", linear_logp_provider_path=lambda args: None
        ),
        "vime.backends.megatron_utils.model": _module(
            "m",
            setup_model_and_optimizer=setup_model_and_optimizer,
            get_model_provider_func=lambda args, role: None,
            _install_linear_logp_capture=lambda model, args: calls.append("capture"),
        ),
        "vime.utils": _module("vime.utils"),
        "vime.utils.arguments": _module("a", parse_args=lambda: args),
        "vime.utils.distributed_utils": _module("d", init_gloo_group=lambda: None),
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core"),
        "megatron.core.enums": _module("e", ModelType=SimpleNamespace(encoder_or_decoder="eod")),
        "megatron.training": _module("megatron.training"),
        "megatron.training.training": _module("t", get_model=get_model),
        "rl_engine.models.qwen3_next.qwen3_next_logprob": _module(
            "p", install_vime_logprob_provider=lambda: calls.append("install")
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "prefix", "/venv")
    monkeypatch.setattr(sys, "base_prefix", "/base")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(runtime, "runtime_argv", lambda checkpoint, **kwargs: ["prog"])
    monkeypatch.setattr(runtime, "require_policy_configuration", lambda args: None)
    return calls, model


def test_inference_actor_loads_without_ddp_or_optimizer(native):
    calls, model = native
    _, actor_model, optimizer, scheduler = runtime.create_actor("ckpt", training=False)
    assert ("get_model", False) in calls
    assert "setup_model_and_optimizer" not in calls
    assert ("load_checkpoint", None, None) in calls
    assert (actor_model, optimizer, scheduler) == ([model], None, None)


def test_training_actor_still_builds_the_optimizer(native):
    calls, _ = native
    _, _, optimizer, scheduler = runtime.create_actor("ckpt", training=True)
    assert "setup_model_and_optimizer" in calls
    assert not any(isinstance(call, tuple) and call[0] == "get_model" for call in calls)
    assert (optimizer, scheduler) == ("optimizer", "scheduler")
