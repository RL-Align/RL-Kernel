# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field

import pytest

from rl_engine.integrations.vime.rollout_logp_probe import (
    ENV_ENABLED,
    ENV_STRICT,
    METADATA_KEY,
    custom_generate,
    run_logp_probe,
)


@dataclass
class FakeSample:
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    metadata: dict = field(default_factory=dict)


def _install_fake_vime(monkeypatch, generate_func):
    vime_mod = types.ModuleType("vime")
    rollout_pkg = types.ModuleType("vime.rollout")
    rollout_mod = types.ModuleType("vime.rollout.vllm_rollout")
    rollout_mod.generate = generate_func

    monkeypatch.setitem(sys.modules, "vime", vime_mod)
    monkeypatch.setitem(sys.modules, "vime.rollout", rollout_pkg)
    monkeypatch.setitem(sys.modules, "vime.rollout.vllm_rollout", rollout_mod)


async def _native_generate(args, sample, sampling_params):
    sample.tokens = [11, 12, 13]
    sample.response = "native"
    sample.response_length = 1
    return sample


async def _native_generate_with_evaluation(args, sample, sampling_params, evaluation=False):
    sample.tokens = [21, 22, 23]
    sample.response = "eval" if evaluation else "native"
    sample.response_length = 1
    sample.metadata["evaluation"] = evaluation
    return sample


async def _native_generate_with_kwargs(args, sample, sampling_params, **kwargs):
    sample.tokens = [31, 32, 33]
    sample.response = "kwargs" if kwargs.get("evaluation") else "native"
    sample.response_length = 1
    sample.metadata["evaluation"] = kwargs.get("evaluation")
    return sample


def test_run_logp_probe_invokes_dispatch_backend():
    result = run_logp_probe(strict=True)

    assert result.enabled is True
    assert result.invoked is True
    assert result.call_count >= 1
    assert result.op == "logp"
    assert result.backend
    assert result.output_shape == (2,)
    assert isinstance(result.output_sum, float)
    assert result.fallback is False


def test_custom_generate_records_disabled_probe_without_changing_native_sample(
    monkeypatch,
):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    _install_fake_vime(monkeypatch, _native_generate)

    sample = asyncio.run(custom_generate(object(), FakeSample(), {"temperature": 1.0}))

    assert sample.response == "native"
    assert sample.tokens == [11, 12, 13]
    probe = sample.metadata[METADATA_KEY]["vime_logp_probe"]
    assert probe["enabled"] is False
    assert probe["invoked"] is False


def test_custom_generate_invokes_probe_when_enabled(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    _install_fake_vime(monkeypatch, _native_generate_with_evaluation)

    sample = asyncio.run(
        custom_generate(object(), FakeSample(), {"temperature": 1.0}, evaluation=True)
    )

    assert sample.response == "eval"
    assert sample.metadata["evaluation"] is True
    probe = sample.metadata[METADATA_KEY]["vime_logp_probe"]
    assert probe["enabled"] is True
    assert probe["invoked"] is True
    assert probe["call_count"] >= 1
    assert probe["op"] == "logp"
    assert probe["backend"]
    assert probe["output_shape"] == (2,)


def test_custom_generate_passes_evaluation_to_kwargs_generate(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    _install_fake_vime(monkeypatch, _native_generate_with_kwargs)

    sample = asyncio.run(
        custom_generate(object(), FakeSample(), {"temperature": 1.0}, evaluation=True)
    )

    assert sample.response == "kwargs"
    assert sample.metadata["evaluation"] is True


def test_probe_falls_back_when_registry_is_unavailable(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setattr(
        "rl_engine.kernels.registry.kernel_registry.get_op",
        lambda _op_type: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
    )
    _install_fake_vime(monkeypatch, _native_generate)

    sample = asyncio.run(custom_generate(object(), FakeSample(), {"temperature": 1.0}))

    assert sample.response == "native"
    probe = sample.metadata[METADATA_KEY]["vime_logp_probe"]
    assert probe["enabled"] is True
    assert probe["invoked"] is False
    assert probe["fallback"] is True
    assert "backend unavailable" in probe["fallback_reason"]


def test_custom_generate_preserves_existing_rl_kernel_metadata(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    _install_fake_vime(monkeypatch, _native_generate)
    sample = FakeSample(metadata={METADATA_KEY: {"existing": "keep"}})

    sample = asyncio.run(custom_generate(object(), sample, {"temperature": 1.0}))

    assert sample.metadata[METADATA_KEY]["existing"] == "keep"
    assert "vime_logp_probe" in sample.metadata[METADATA_KEY]


def test_custom_generate_handles_non_dict_metadata(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    _install_fake_vime(monkeypatch, _native_generate)
    sample = FakeSample(metadata=None)

    sample = asyncio.run(custom_generate(object(), sample, {"temperature": 1.0}))

    assert isinstance(sample.metadata, dict)
    assert "vime_logp_probe" in sample.metadata[METADATA_KEY]


def test_probe_strict_mode_raises_on_backend_failure(monkeypatch):
    monkeypatch.setenv(ENV_STRICT, "1")
    monkeypatch.setattr(
        "rl_engine.kernels.registry.kernel_registry.get_op",
        lambda _op_type: (_ for _ in ()).throw(RuntimeError("strict failure")),
    )

    with pytest.raises(RuntimeError, match="strict failure"):
        run_logp_probe()
