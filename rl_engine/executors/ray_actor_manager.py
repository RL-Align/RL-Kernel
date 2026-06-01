# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


class RayUnavailableError(RuntimeError):
    """Raised when the optional Ray runtime cannot be imported."""


@dataclass(frozen=True)
class RayRuntimeConfig:
    """Configuration for lazy Ray runtime initialization."""

    auto_init: bool = True
    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    shutdown_ray_on_close: bool = False

    def resolved_init_kwargs(self) -> dict[str, Any]:
        kwargs = {"ignore_reinit_error": True}
        kwargs.update(dict(self.init_kwargs))
        return kwargs


@dataclass(frozen=True)
class RayActorOptions:
    """Ray actor resource and lifecycle options."""

    num_cpus: Optional[float] = None
    num_gpus: Optional[float] = None
    resources: Mapping[str, float] = field(default_factory=dict)
    name: Optional[str] = None
    namespace: Optional[str] = None
    max_restarts: Optional[int] = None
    max_task_retries: Optional[int] = None
    scheduling_strategy: Any = None
    lifetime: Optional[str] = None

    def to_ray_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for key in (
            "num_cpus",
            "num_gpus",
            "name",
            "namespace",
            "max_restarts",
            "max_task_retries",
            "scheduling_strategy",
            "lifetime",
        ):
            value = getattr(self, key)
            if value is not None:
                options[key] = value
        if self.resources:
            options["resources"] = dict(self.resources)
        return options


@dataclass(frozen=True)
class RayWorkerSpec:
    """Factory and constructor arguments for a worker hosted in a Ray actor."""

    worker_factory: Any
    args: Sequence[Any] = field(default_factory=tuple)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    actor_options: RayActorOptions = field(default_factory=RayActorOptions)


class RayActorManager:
    """Create, wrap, and clean up Ray actors for RL-Kernel workers."""

    def __init__(
        self,
        runtime_config: Optional[RayRuntimeConfig] = None,
        *,
        ray_module: Any = None,
    ):
        self.runtime_config = runtime_config or RayRuntimeConfig()
        self._ray = ray_module
        self._actors: list[Any] = []

    def __enter__(self) -> RayActorManager:
        self.ensure_runtime()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()

    def ensure_runtime(self) -> Any:
        if self._ray is None:
            self._ray = _load_ray()
        if self.runtime_config.auto_init and not self._ray.is_initialized():
            self._ray.init(**self.runtime_config.resolved_init_kwargs())
        return self._ray

    def create_worker_actor(self, spec: RayWorkerSpec) -> Any:
        ray = self.ensure_runtime()
        remote_actor = ray.remote(_RayWorkerActor)
        options = spec.actor_options.to_ray_options()
        if options:
            remote_actor = remote_actor.options(**options)
        actor = remote_actor.remote(
            spec.worker_factory,
            *tuple(spec.args),
            **dict(spec.kwargs),
        )
        self._actors.append(actor)
        return actor

    def create_rollout_worker(self, spec: RayWorkerSpec) -> RayRolloutWorkerHandle:
        actor = self.create_worker_actor(spec)
        return RayRolloutWorkerHandle(actor, self.ensure_runtime())

    def create_training_worker(self, spec: RayWorkerSpec) -> RayTrainingWorkerHandle:
        actor = self.create_worker_actor(spec)
        return RayTrainingWorkerHandle(actor, self.ensure_runtime())

    def health_check(self) -> list[Mapping[str, Any]]:
        ray = self.ensure_runtime()
        return [ray.get(actor.health_check.remote()) for actor in self._actors]

    def shutdown(self) -> None:
        if self._ray is None:
            return
        for actor in reversed(self._actors):
            self._ray.kill(actor, no_restart=True)
        self._actors.clear()
        if self.runtime_config.shutdown_ray_on_close and hasattr(self._ray, "shutdown"):
            self._ray.shutdown()


class RayRolloutWorkerHandle:
    """Synchronous `RolloutWorker` protocol adapter for a Ray actor."""

    def __init__(self, actor: Any, ray_module: Any):
        self.actor = actor
        self._ray = ray_module

    def rollout(self, spec: Any) -> Any:
        return self._ray.get(self.actor.rollout.remote(spec))


class RayTrainingWorkerHandle:
    """Synchronous `TrainingWorker` protocol adapter for a Ray actor."""

    def __init__(self, actor: Any, ray_module: Any):
        self.actor = actor
        self._ray = ray_module

    def train(self, rollout: Any) -> Any:
        return self._ray.get(self.actor.train.remote(rollout))


class _RayWorkerActor:
    """Ray-hosted shim around an arbitrary local RL-Kernel worker."""

    def __init__(self, worker_factory: Any, *args: Any, **kwargs: Any):
        if callable(worker_factory):
            self.worker = worker_factory(*args, **kwargs)
        else:
            if args or kwargs:
                raise TypeError("constructor args require a callable worker_factory")
            self.worker = worker_factory

    def rollout(self, spec: Any) -> Any:
        return self.worker.rollout(spec)

    def train(self, rollout: Any) -> Any:
        return self.worker.train(rollout)

    def health_check(self) -> Mapping[str, Any]:
        return {
            "status": "ok",
            "worker_type": type(self.worker).__name__,
        }


def _load_ray() -> Any:
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        raise RayUnavailableError(
            "Ray is not installed or cannot be imported. Install a Ray runtime "
            "supported by the active Python/platform environment before creating "
            "Ray actors."
        ) from exc
