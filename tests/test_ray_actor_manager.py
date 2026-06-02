# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from rl_engine.executors.training_contract import RolloutStageResult, TrainingStageResult


@dataclass(frozen=True)
class IterationSpec:
    iteration: int
    weight_version: int = 0
    prompts: Sequence[Any] = ()


class FakeObjectRef:
    def __init__(self, value):
        self.value = value


class FakeRemoteMethod:
    def __init__(self, method):
        self.method = method
        self.remote_calls = []

    def remote(self, *args, **kwargs):
        self.remote_calls.append((args, kwargs))
        return FakeObjectRef(self.method(*args, **kwargs))


class FakeActorHandle:
    def __init__(self, instance, options):
        self.instance = instance
        self.options = options

    def __getattr__(self, name):
        return FakeRemoteMethod(getattr(self.instance, name))


class FakeRemoteActorClass:
    def __init__(self, fake_ray, actor_class, options=None):
        self.fake_ray = fake_ray
        self.actor_class = actor_class
        self._options = dict(options or {})

    def options(self, **kwargs):
        merged = dict(self._options)
        merged.update(kwargs)
        self.fake_ray.option_calls.append(merged)
        return FakeRemoteActorClass(self.fake_ray, self.actor_class, merged)

    def remote(self, *args, **kwargs):
        actor = FakeActorHandle(self.actor_class(*args, **kwargs), dict(self._options))
        self.fake_ray.actors.append(actor)
        return actor


class FakeRayModule:
    def __init__(self):
        self.initialized = False
        self.init_calls = []
        self.option_calls = []
        self.actors = []
        self.get_calls = []
        self.kill_calls = []
        self.shutdown_calls = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True
        self.init_calls.append(kwargs)

    def remote(self, actor_class):
        return FakeRemoteActorClass(self, actor_class)

    def get(self, ref):
        self.get_calls.append(ref)
        return ref.value

    def kill(self, actor, no_restart=True):
        self.kill_calls.append((actor, no_restart))

    def shutdown(self):
        self.shutdown_calls += 1
        self.initialized = False


class FakeRolloutWorker:
    def __init__(self, label="ray-rollout"):
        self.label = label

    def rollout(self, spec):
        started = time.perf_counter()
        return RolloutStageResult(
            iteration=spec.iteration,
            weight_version=spec.weight_version,
            payload={"label": self.label, "prompts": list(spec.prompts)},
            started_at=started,
            finished_at=time.perf_counter(),
            metrics={"worker": self.label},
        )


class FakeTrainingWorker:
    def __init__(self, publish_delta=1):
        self.publish_delta = publish_delta

    def train(self, rollout):
        started = time.perf_counter()
        return TrainingStageResult(
            iteration=rollout.iteration,
            consumed_weight_version=rollout.weight_version,
            published_weight_version=rollout.weight_version + self.publish_delta,
            metrics={"worker": "ray-training"},
            started_at=started,
            finished_at=time.perf_counter(),
        )


def test_importing_module_does_not_import_ray(monkeypatch):
    monkeypatch.delitem(sys.modules, "ray", raising=False)

    module = importlib.import_module("rl_engine.executors.ray_actor_manager")

    assert module.RayActorManager is not None
    assert "ray" not in sys.modules


def test_missing_ray_raises_explicit_blocker(monkeypatch):
    from rl_engine.executors import ray_actor_manager

    original_import_module = importlib.import_module

    def fail_import(name, package=None):
        if name == "ray":
            raise ImportError("no ray here")
        return original_import_module(name, package)

    monkeypatch.setattr(ray_actor_manager.importlib, "import_module", fail_import)

    manager = ray_actor_manager.RayActorManager()
    with pytest.raises(ray_actor_manager.RayUnavailableError, match="Ray"):
        manager.create_worker_actor(ray_actor_manager.RayWorkerSpec(FakeRolloutWorker))


def test_ray_manager_creates_actor_with_options_dispatches_and_cleans_up():
    from rl_engine.executors.ray_actor_manager import (
        RayActorManager,
        RayActorOptions,
        RayRuntimeConfig,
        RayWorkerSpec,
    )

    fake_ray = FakeRayModule()
    manager = RayActorManager(
        RayRuntimeConfig(
            auto_init=True,
            init_kwargs={"address": "local", "namespace": "RL-Kernel-test"},
            shutdown_ray_on_close=True,
        ),
        ray_module=fake_ray,
    )
    handle = manager.create_rollout_worker(
        RayWorkerSpec(
            FakeRolloutWorker,
            args=("remote-rollout",),
            actor_options=RayActorOptions(
                num_cpus=2,
                num_gpus=0.25,
                resources={"accelerator_type": 1},
                name="rollout-actor",
                max_restarts=1,
            ),
        )
    )
    result = handle.rollout(IterationSpec(iteration=3, weight_version=8, prompts=["a", "b"]))

    assert fake_ray.init_calls == [
        {
            "ignore_reinit_error": True,
            "address": "local",
            "namespace": "RL-Kernel-test",
        }
    ]
    assert fake_ray.option_calls == [
        {
            "num_cpus": 2,
            "num_gpus": 0.25,
            "name": "rollout-actor",
            "max_restarts": 1,
            "resources": {"accelerator_type": 1},
        }
    ]
    assert result.iteration == 3
    assert result.payload["label"] == "remote-rollout"
    assert len(fake_ray.get_calls) == 1
    assert manager.health_check() == [{"status": "ok", "worker_type": "FakeRolloutWorker"}]

    manager.shutdown()

    assert fake_ray.kill_calls == [(fake_ray.actors[0], True)]
    assert fake_ray.shutdown_calls == 1


def test_ray_actor_handles_support_direct_rollout_training_handoff():
    from rl_engine.executors.ray_actor_manager import (
        RayActorManager,
        RayRuntimeConfig,
        RayWorkerSpec,
    )

    fake_ray = FakeRayModule()
    manager = RayActorManager(
        RayRuntimeConfig(auto_init=True),
        ray_module=fake_ray,
    )
    rollout = manager.create_rollout_worker(RayWorkerSpec(FakeRolloutWorker, args=("r",)))
    training = manager.create_training_worker(
        RayWorkerSpec(FakeTrainingWorker, kwargs={"publish_delta": 1})
    )

    weight_version = 5
    results = []
    for iteration in range(2):
        rollout_result = rollout.rollout(
            IterationSpec(
                iteration=iteration,
                weight_version=weight_version,
                prompts=[f"p{iteration}"],
            )
        )
        result = training.train(rollout_result)
        results.append(result)
        if result.published_weight_version is not None:
            weight_version = result.published_weight_version

    assert [result.iteration for result in results] == [0, 1]
    assert [result.consumed_weight_version for result in results] == [5, 6]
    assert [result.published_weight_version for result in results] == [6, 7]
    assert len(fake_ray.actors) == 2
    assert len(fake_ray.get_calls) == 4
    assert manager.health_check() == [
        {"status": "ok", "worker_type": "FakeRolloutWorker"},
        {"status": "ok", "worker_type": "FakeTrainingWorker"},
    ]

    manager.shutdown()
    assert [actor for actor, _ in fake_ray.kill_calls] == list(reversed(fake_ray.actors))
