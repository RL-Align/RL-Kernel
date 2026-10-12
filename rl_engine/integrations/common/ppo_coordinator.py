# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Synchronous PPO across separate rollout, Actor TP and Critic TP groups."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Callable, Mapping, TypeVar

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import (
    ROLLOUT_FIELDS,
    TARGET_FIELDS,
    PPOBatch,
    PPOConfig,
    PPORollout,
    PPOTrainingWorker,
    build_ppo_batch,
)
from rl_engine.integrations.common.tp_scalar import TPModelInputs, TPScalarExecutor
from rl_engine.integrations.common.weights import TensorPacketBridge, synchronize_validation

T = TypeVar("T")


@dataclass(frozen=True)
class PPOProcessGroups:
    """Caller-created groups, created in the same order by all global ranks.

    control contains exactly the disjoint union of the three role groups.
    leaders contains exactly their designated global leaders. Every control
    rank enters coordinator methods in the same order. Other global ranks do
    not participate. Control and leaders use transport_device; local groups use the model device.
    """

    control: dist.ProcessGroup
    leaders: dist.ProcessGroup
    local: dist.ProcessGroup
    role: str
    rollout_leader: int
    actor_leader: int
    critic_leader: int

    @property
    def local_leader(self) -> int:
        return getattr(self, f"{self.role}_leader")

    def validate(self) -> None:
        if self.role not in ("rollout", "actor", "critic"):
            raise ValueError("role must be rollout, actor or critic")
        if self.control is None or self.local is None:
            raise ValueError("control and local process groups must be explicit")
        if dist.get_rank(self.control) < 0 or dist.get_rank(self.local) < 0:
            raise ValueError("this rank must belong to control and local groups")
        leaders = {self.rollout_leader, self.actor_leader, self.critic_leader}
        if len(leaders) != 3:
            raise ValueError("rollout, actor and critic leaders must be distinct")
        local = set(dist.get_process_group_ranks(self.local))
        control = set(dist.get_process_group_ranks(self.control))
        if local & leaders != {self.local_leader} or not local <= control:
            raise ValueError("local group must contain exactly its designated leader")
        if not leaders <= control:
            raise ValueError("all role leaders must belong to control")
        if dist.get_rank() in leaders:
            if self.leaders is None or set(dist.get_process_group_ranks(self.leaders)) != leaders:
                raise ValueError("leaders group must contain exactly the three role leaders")


@dataclass
class PPOServices:
    """Only the local role's services are supplied.

    sample runs on the rollout leader; install runs on all rollout ranks and
    returns True only after the version is visible. export runs on every Actor
    TP rank (so adapters may gather shards) and returns complete tensors on its
    leader. The weight field schema is fixed by the caller. These hooks are
    the integration boundary for serving runtimes; no private framework API is
    assumed. reference must be frozen and deterministic; it returns [B,S-1].
    """

    sample: Callable[[int, int], PPORollout] | None = None
    reference: Callable[[TPModelInputs], torch.Tensor] | None = None
    reference_model: torch.nn.Module | None = None
    reward: TPScalarExecutor | None = None
    critic: TPScalarExecutor | None = None
    worker: PPOTrainingWorker | None = None
    export: Callable[[], Mapping[str, torch.Tensor] | None] | None = None
    install: Callable[[Mapping[str, torch.Tensor], int], bool] | None = None


class PPOCoordinator:
    """One round is an all-control-rank operation; any failure poisons this instance.

    No optimizer rollback is attempted. After a failed round callers must
    restore a checkpoint and reconstruct the coordinator. Group timeouts bound
    failures inside model/adapter collectives. Versions advance only after all
    rollout workers acknowledge installation.
    """

    def __init__(
        self,
        *,
        groups: PPOProcessGroups,
        services: PPOServices,
        weight_fields: tuple[str, ...],
        device: torch.device | str,
        transport_device: torch.device | str | None = None,
        config: PPOConfig | None = None,
        actor_version: int = 0,
        critic_version: int = 0,
    ) -> None:
        self.groups, self.services = groups, services
        self.device = torch.device(device)
        self.transport_device = (
            torch.device(transport_device) if transport_device is not None else self.device
        )
        self.config = config or PPOConfig()
        self.weight_fields = weight_fields
        self.actor_version, self.critic_version = actor_version, critic_version
        self.initialized = False
        self.failed = False
        self.transport_metrics: list[dict[str, float | int]] = []
        self._phase(groups.validate)
        self._phase(self._validate_topology)
        self._phase(self._validate_services)

    def _validate_topology(self) -> None:
        members = dist.get_process_group_ranks(self.groups.control)
        local = dist.get_process_group_ranks(self.groups.local)
        roles = ("rollout", "actor", "critic")
        digest = hashlib.sha256(repr(self.weight_fields).encode()).digest()
        descriptor = torch.tensor(
            [
                roles.index(self.groups.role),
                self.groups.rollout_leader,
                self.groups.actor_leader,
                self.groups.critic_leader,
                self.actor_version,
                self.critic_version,
                *digest,
                *[int(rank in local) for rank in members],
            ],
            dtype=torch.int64,
            device=self.transport_device,
        )
        descriptors = [torch.empty_like(descriptor) for _ in members]
        dist.all_gather(descriptors, descriptor, group=self.groups.control)
        rows = torch.stack(descriptors).cpu()
        if not (rows[:, 1:38] == rows[0, 1:38]).all():
            raise ValueError("all ranks must agree on leaders, versions and weight schema")
        for row in rows:
            if not torch.equal(row[38:].bool(), rows[:, 0] == row[0]):
                raise ValueError("role groups must partition the control group")
        config = torch.tensor(
            [
                -1 if value is None else value
                for value in (getattr(self.config, f.name) for f in fields(self.config))
            ],
            dtype=torch.float64,
            device=self.transport_device,
        )
        configs = [torch.empty_like(config) for _ in members]
        dist.all_gather(configs, config, group=self.groups.control)
        if any(not torch.equal(config, other) for other in configs):
            raise ValueError("all PPO ranks must use the same configuration")

    def _validate_services(self) -> None:
        service, role = self.services, self.groups.role
        if self.actor_version < 0 or self.critic_version < 0:
            raise ValueError("initial versions must be nonnegative")
        if not self.weight_fields or len(set(self.weight_fields)) != len(self.weight_fields):
            raise ValueError("weight_fields must be a nonempty unique schema")
        if role == "rollout":
            if service.install is None or (
                dist.get_rank() == self.groups.rollout_leader and service.sample is None
            ):
                raise ValueError("rollout requires install and leader sample callbacks")
        else:
            if service.worker is None or service.worker.role != role:
                raise ValueError(f"{role} requires a matching PPO training worker")
            if (
                service.worker.group is not self.groups.local
                or service.worker.config != self.config
            ):
                raise ValueError("worker must use coordinator's local group and PPO config")
            if role == "actor":
                if (
                    service.export is None
                    or service.reference is None
                    or service.reference_model is None
                ):
                    raise ValueError("actor requires export, reference callback and model")
                if {id(p) for p in service.reference_model.parameters()} & {
                    id(p) for p in service.worker.model.parameters()
                }:
                    raise ValueError("reference and actor must have separate parameters")
                service.reference_model.requires_grad_(False).eval()
            if role == "critic":
                if service.reward is None or service.critic is None:
                    raise ValueError("critic requires reward and critic executors")
                for executor in (service.reward, service.critic):
                    if (
                        executor.group is not self.groups.local
                        or executor.leader != self.groups.critic_leader
                    ):
                        raise ValueError("scoring executors must use the critic group and leader")
                if service.critic.model is not service.worker.model:
                    raise ValueError("critic scoring and training must share the same model")
                if {id(p) for p in service.reward.model.parameters()} & {
                    id(p) for p in service.critic.model.parameters()
                }:
                    raise ValueError("reward and critic must have separate parameters")
                service.reward.model.requires_grad_(False)
                service.reward.model.eval()

    def _phase(self, operation: Callable[[], T], *, active: bool = True) -> T | None:
        result = None
        error = None
        if active:
            try:
                result = operation()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_validation(error, group=self.groups.control, device=self.transport_device)
        return result

    def _exchange(
        self,
        fields: tuple[str, ...],
        source: int,
        packet: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        is_leader = dist.get_rank() == self.groups.local_leader

        def stage():
            assert packet is not None
            return {
                name: value.detach().to(self.transport_device) for name, value in packet.items()
            }

        staged = self._phase(stage, active=dist.get_rank() == source and packet is not None)

        def relay():
            bridge = TensorPacketBridge(
                group=self.groups.leaders, source=source, device=self.transport_device
            )
            result = bridge.broadcast(fields, staged)
            self.transport_metrics.append(dict(bridge.metrics))
            return result

        relayed = self._phase(relay, active=is_leader)

        def to_device():
            assert relayed is not None
            return {name: value.to(self.device) for name, value in relayed.items()}

        local_packet = self._phase(to_device, active=is_leader)

        def local_broadcast():
            bridge = TensorPacketBridge(
                group=self.groups.local,
                source=self.groups.local_leader,
                device=self.device,
            )
            result = bridge.broadcast(fields, local_packet)
            self.transport_metrics.append(dict(bridge.metrics))
            return result

        result = self._phase(local_broadcast)
        assert result is not None  # Every member receives the packet or the phase raised.
        return result

    def _install_actor(self, version: int) -> None:
        def export():
            assert self.services.export is not None  # Checked for Actor ranks at construction.
            return self.services.export()

        weights = self._phase(export, active=self.groups.role == "actor")
        packet = self._exchange(self.weight_fields, self.groups.actor_leader, weights)

        def install():
            if self.services.install(packet, version) is not True:
                raise ValueError("rollout did not acknowledge installed Actor weights")

        self._phase(install, active=self.groups.role == "rollout")

    def run_round(self) -> dict[str, float | int]:
        """Sample and update once; return local training and shared version metrics."""
        if self.failed:
            raise RuntimeError("PPO coordinator failed; restore state before running again")
        try:
            return self._run_round()
        except Exception:
            self.failed = True
            raise

    def _run_round(self) -> dict[str, float | int]:
        self.transport_metrics.clear()
        if not self.initialized:
            self._install_actor(self.actor_version)
            self.initialized = True

        def sample():
            rollout = self.services.sample(self.actor_version, self.critic_version)
            rollout.validate(actor_version=self.actor_version, critic_version=self.critic_version)
            return rollout.packet()

        source_packet = self._phase(sample, active=dist.get_rank() == self.groups.rollout_leader)
        packet = self._exchange(ROLLOUT_FIELDS, self.groups.rollout_leader, source_packet)
        rollout = PPORollout(**packet)
        self._phase(
            lambda: rollout.validate(
                actor_version=self.actor_version, critic_version=self.critic_version
            )
        )
        if rollout.input_ids.shape[0] == 0:
            return {
                "updates": 0,
                "actor_version": self.actor_version,
                "critic_version": self.critic_version,
            }

        @torch.no_grad()
        def reference():
            scores = self.services.reference(rollout.inputs)
            if scores.shape != rollout.action_mask.shape:
                raise ValueError("reference adapter must return [B,S-1]")
            return {"reference_logps": scores}

        ref = self._phase(reference, active=self.groups.role == "actor")
        ref = self._exchange(("reference_logps",), self.groups.actor_leader, ref)

        @torch.no_grad()
        def targets():
            critic = self.services.critic
            modes = [(module, module.training) for module in critic.model.modules()]
            critic.model.eval()
            try:
                values = critic.values(rollout.inputs)
                rewards = self.services.reward.rewards(rollout.inputs)
            finally:
                for module, training in modes:
                    module.training = training
            if dist.get_rank() == self.groups.critic_leader:
                return build_ppo_batch(
                    rollout,
                    state_values=values,
                    rewards=rewards,
                    reference_logps=ref["reference_logps"],
                    config=self.config,
                ).targets()
            return None

        target_packet = self._phase(targets, active=self.groups.role == "critic")
        target_packet = self._exchange(TARGET_FIELDS, self.groups.critic_leader, target_packet)
        batch = PPOBatch(rollout, **target_packet)
        # Includes scalar rewards on every rollout rank, in original sample order.
        self.last_batch = batch

        def train():
            assert self.services.worker is not None  # Checked for trainer ranks at construction.
            return self.services.worker.train(batch)

        metrics = self._phase(train, active=self.groups.role in ("actor", "critic"))
        self._install_actor(self.actor_version + 1)
        self.actor_version += 1
        self.critic_version += 1
        return {
            **(metrics or {}),
            "actor_version": self.actor_version,
            "critic_version": self.critic_version,
            "sequences": rollout.input_ids.shape[0],
            "transport_payload_bytes": sum(m["payload_bytes"] for m in self.transport_metrics),
            "transport_elapsed_ms": sum(m["elapsed_ms"] for m in self.transport_metrics),
        }
