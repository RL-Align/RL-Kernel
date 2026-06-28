# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import math
import os
import sys
import time

import pytest
import torch

from rl_engine.executors.bridge import LocalTensorCopyBridge, WeightBridgeUnavailableError
from rl_engine.executors.training_contract import RolloutStageResult


class FakeDeepSpeedEngine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.forward_calls = 0
        self.backward_calls = 0
        self.step_calls = 0
        self.zero_grad_calls = 0

    def __call__(self, *args, **kwargs):
        self.forward_calls += 1
        return self.model(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls += 1
        self.optimizer.zero_grad(*args, **kwargs)

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def step(self):
        self.step_calls += 1
        self.optimizer.step()


class FakeDeepSpeedModule:
    def __init__(self):
        self.initialize_calls = []
        self.engines = []

    def initialize(self, **kwargs):
        self.initialize_calls.append(kwargs)
        engine = FakeDeepSpeedEngine(kwargs["model"], kwargs["optimizer"])
        self.engines.append(engine)
        return engine, kwargs["optimizer"], None, None


class FakeGatheredParameters:
    calls = 0

    def __init__(self, parameters, modifier_rank=0):
        self.parameters = list(parameters)
        self.modifier_rank = modifier_rank

    def __enter__(self):
        type(self).calls += 1
        return self.parameters

    def __exit__(self, exc_type, exc, traceback):
        return False


def _install_fake_deepspeed(monkeypatch):
    fake = FakeDeepSpeedModule()
    monkeypatch.setitem(sys.modules, "deepspeed", fake)
    return fake


def _install_fake_deepspeed_with_gather(monkeypatch):
    fake = FakeDeepSpeedModule()
    FakeGatheredParameters.calls = 0
    fake.zero = type("FakeZeroNamespace", (), {"GatheredParameters": FakeGatheredParameters})()
    monkeypatch.setitem(sys.modules, "deepspeed", fake)
    return fake


def _rollout(iteration=2, weight_version=9):
    return RolloutStageResult(
        iteration=iteration,
        weight_version=weight_version,
        payload={
            "normalized_outputs": [
                [{"token_ids": [3, 4, 5], "text": "abc"}],
                [{"token_ids": [6, 7, 8], "text": "def"}],
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )


def test_importing_module_does_not_import_deepspeed(monkeypatch):
    monkeypatch.delitem(sys.modules, "deepspeed", raising=False)

    module = importlib.import_module("rl_engine.executors.deepspeed_trainer")

    assert module.DeepSpeedTrainingWorker is not None
    assert "deepspeed" not in sys.modules


def test_missing_deepspeed_raises_explicit_blocker(monkeypatch):
    from rl_engine.executors import deepspeed_trainer

    original_import_module = importlib.import_module

    def fail_import(name, package=None):
        if name == "deepspeed":
            raise ImportError("no deepspeed here")
        return original_import_module(name, package)

    monkeypatch.setattr(deepspeed_trainer.importlib, "import_module", fail_import)

    with pytest.raises(deepspeed_trainer.DeepSpeedUnavailableError, match="DeepSpeed"):
        deepspeed_trainer.DeepSpeedTrainingWorker()


def test_deepspeed_loader_preserves_explicit_cuda_home(monkeypatch):
    import torch.utils.cpp_extension as cpp_extension

    from rl_engine.executors import deepspeed_trainer

    monkeypatch.setenv("CUDA_HOME", "/custom/cuda")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    monkeypatch.setattr(
        deepspeed_trainer,
        "_python_cuda_home_candidates",
        lambda: pytest.fail("CUDA_HOME should short-circuit package probing"),
    )

    deepspeed_trainer._configure_cuda_home_from_python_packages()

    assert os.environ["CUDA_HOME"] == "/custom/cuda"
    assert os.environ["PATH"] == "/usr/bin"
    assert os.environ["LD_LIBRARY_PATH"] == "/usr/lib"
    assert cpp_extension.CUDA_HOME == os.path.normpath("/custom/cuda")


def test_deepspeed_loader_uses_python_cuda_toolkit(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp_extension

    from rl_engine.executors import deepspeed_trainer

    cuda_home = tmp_path / "site-packages" / "nvidia" / "cu13"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "include").mkdir()
    (cuda_home / "lib").mkdir()
    (cuda_home / "bin" / "nvcc").write_text("", encoding="utf-8")
    (cuda_home / "include" / "cuda.h").write_text("", encoding="utf-8")

    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    monkeypatch.setattr(
        deepspeed_trainer,
        "_python_cuda_home_candidates",
        lambda: [cuda_home],
    )

    deepspeed_trainer._configure_cuda_home_from_python_packages()

    assert os.environ["CUDA_HOME"] == str(cuda_home)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_home / "bin")
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(cuda_home / "lib")
    assert cpp_extension.CUDA_HOME == str(cuda_home)


def test_deepspeed_training_worker_uses_engine_backward_and_step(monkeypatch):
    fake = _install_fake_deepspeed(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=2,
            completion_len=3,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=5,
            deepspeed_config={
                "zero_optimization": {"stage": 1},
                "gradient_accumulation_steps": 2,
            },
        )
    )
    result = worker.train(_rollout())

    assert len(fake.initialize_calls) == 1
    init_call = fake.initialize_calls[0]
    assert init_call["model"] is worker.model
    assert init_call["optimizer"] is worker.optimizer
    assert init_call["config"]["zero_optimization"]["stage"] == 1
    assert init_call["config"]["gradient_accumulation_steps"] == 2

    engine = fake.engines[0]
    assert engine.forward_calls == 1
    assert engine.zero_grad_calls == 1
    assert engine.backward_calls == 1
    assert engine.step_calls == 1

    assert result.iteration == 2
    assert result.consumed_weight_version == 9
    assert result.published_weight_version == 10
    assert result.metrics["training_backend"] == "deepspeed"
    assert result.metrics["training_data_source"] == "rollout_payload"
    assert result.metrics["rollout_sequences"] == 2
    assert result.metrics["rollout_tokens"] == 6
    assert math.isfinite(result.metrics["loss"])
    assert "advantage_mean" not in result.metrics
    assert "advantage_std" not in result.metrics
    assert math.isfinite(result.metrics["active_advantage_mean_global"])
    assert result.metrics["active_advantage_std_global"] >= 0.0


def test_deepspeed_training_worker_synthetic_fallback(monkeypatch):
    _install_fake_deepspeed(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            num_prompts=1,
            samples_per_prompt=1,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            seed=11,
        )
    )
    result = worker.train(
        RolloutStageResult(
            iteration=0,
            weight_version=4,
            payload={"normalized_outputs": []},
            started_at=time.perf_counter(),
            finished_at=time.perf_counter(),
        )
    )

    assert result.iteration == 0
    assert result.consumed_weight_version == 4
    assert result.published_weight_version == 5
    assert result.metrics["training_backend"] == "deepspeed"
    assert result.metrics["training_data_source"] == "synthetic_fallback"


def test_deepspeed_worker_publishes_full_state_manifest(monkeypatch):
    _install_fake_deepspeed(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    bridge = LocalTensorCopyBridge(source_worker="training", source_rank=0)
    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            vocab_size=16,
            hidden_dim=8,
            zero_stage=2,
            seed=17,
        ),
        weight_bridge=bridge,
    )

    manifest = worker.publish_weights(
        weight_version=21,
        metadata={"iteration": 4},
    )
    imported = bridge.import_update(manifest)

    assert manifest.source_worker == "training"
    assert manifest.weight_version == 21
    assert manifest.metadata["iteration"] == 4
    assert manifest.metadata["layout"]["kind"] == "full-state"
    assert manifest.metadata["layout"]["zero_stage"] == 2
    assert manifest.metadata["layout"]["world_size"] == 1
    assert manifest.metadata["layout"]["rank"] == 0
    assert set(imported) == set(worker.model.state_dict())

    bridge.release(manifest.update_id)


def test_deepspeed_zero3_single_rank_publishes_full_state_manifest(monkeypatch):
    _install_fake_deepspeed(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    bridge = LocalTensorCopyBridge(source_worker="training", source_rank=0)
    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            vocab_size=16,
            hidden_dim=8,
            zero_stage=3,
            seed=19,
        ),
        weight_bridge=bridge,
    )

    manifest = worker.publish_weights(weight_version=31)
    imported = bridge.import_update(manifest)

    assert manifest.metadata["layout"]["kind"] == "full-state"
    assert manifest.metadata["layout"]["zero_stage"] == 3
    assert manifest.metadata["layout"]["world_size"] == 1
    assert manifest.metadata["layout"]["rank"] == 0
    assert manifest.metadata["deepspeed_zero3_full_state_export"] == {
        "method": "single-rank-state-dict",
        "rank": 0,
        "world_size": 1,
        "tensor_count": len(worker.model.state_dict()),
    }
    assert set(imported) == set(worker.model.state_dict())
    for name, original in worker.model.state_dict().items():
        assert torch.equal(imported[name], original)

    bridge.release(manifest.update_id)


def test_deepspeed_zero3_uses_gathered_parameters_when_available(monkeypatch):
    _install_fake_deepspeed_with_gather(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    bridge = LocalTensorCopyBridge(source_worker="training", source_rank=0)
    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            vocab_size=16,
            hidden_dim=8,
            zero_stage=3,
            seed=20,
        ),
        weight_bridge=bridge,
    )
    worker.engine.world_size = 2
    worker.engine.global_rank = 0

    manifest = worker.publish_weights(weight_version=32)
    imported = bridge.import_update(manifest)

    assert FakeGatheredParameters.calls == 1
    assert manifest.metadata["layout"]["world_size"] == 2
    assert manifest.metadata["layout"]["rank"] == 0
    assert manifest.metadata["deepspeed_zero3_full_state_export"]["method"] == (
        "deepspeed.zero.GatheredParameters"
    )
    assert manifest.metadata["deepspeed_zero3_full_state_export"]["tensor_count"] == len(
        worker.model.state_dict()
    )
    assert set(imported) == set(worker.model.state_dict())

    bridge.release(manifest.update_id)


def test_deepspeed_zero3_multi_rank_without_gather_api_is_blocked(monkeypatch):
    _install_fake_deepspeed(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            vocab_size=16,
            hidden_dim=8,
            zero_stage=3,
        )
    )
    worker.engine.world_size = 2
    worker.engine.global_rank = 0

    with pytest.raises(WeightBridgeUnavailableError, match="GatheredParameters"):
        worker.publish_weights(weight_version=1)


def test_deepspeed_published_versions_are_monotonic(monkeypatch):
    _install_fake_deepspeed(monkeypatch)
    from rl_engine.executors.deepspeed_trainer import (
        DeepSpeedTrainingConfig,
        DeepSpeedTrainingWorker,
    )

    worker = DeepSpeedTrainingWorker(
        DeepSpeedTrainingConfig(
            vocab_size=16,
            hidden_dim=8,
            seed=23,
        )
    )

    first = worker.train(_rollout(iteration=0, weight_version=5))
    second = worker.train(_rollout(iteration=1, weight_version=5))

    assert first.published_weight_version == 6
    assert second.published_weight_version == 7
