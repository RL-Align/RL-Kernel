# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PPO targets and independent Actor/Critic workers for explicit TP adapters."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Literal, Mapping

import torch
import torch.distributed as dist

from rl_engine.integrations.common.tp_scalar import TPModelInputs, clip_tp_grad_norm_
from rl_engine.integrations.common.weights import synchronize_validation


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_clip: float = 0.2
    kl_coefficient: float = 0.01
    epochs: int = 4
    minibatch_size: int = 8
    max_grad_norm: float = 1.0
    microbatch_size: int | None = None

    def __post_init__(self) -> None:
        """Raise ``ValueError`` on a non-finite float field or an out-of-range value.

        ``gamma`` and ``gae_lambda`` must lie in [0, 1] and ``kl_coefficient`` must be
        nonnegative; clips, ``max_grad_norm``, ``epochs`` and batch sizes must be positive.
        """
        import math

        if self.microbatch_size is not None and self.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")

        numeric = (
            self.gamma,
            self.gae_lambda,
            self.policy_clip,
            self.value_clip,
            self.kl_coefficient,
            self.max_grad_norm,
        )
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("PPO configuration must be finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.policy_clip <= 0 or self.value_clip <= 0 or self.kl_coefficient < 0:
            raise ValueError("clip ranges must be positive and KL coefficient nonnegative")
        if self.epochs < 1 or self.minibatch_size < 1 or self.max_grad_norm <= 0:
            raise ValueError("epochs, minibatch_size and max_grad_norm must be positive")


@dataclass(frozen=True)
class PPORollout:
    """Dense sequences and behavior scores, with action slots indexed by state t.

    action_mask[:, t] selects the sampled token input_ids[:, t+1]. Valid
    completion slots must be a nonempty contiguous suffix of each sequence.
    terminated=True means an actual terminal state; False means length cutoff.
    sampling is float [temperature, top_k, top_p], exactly [1, 0, 1] in v1.
    Version tensors are scalar int64; sample_ids preserves the original order.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    action_mask: torch.Tensor
    behavior_logps: torch.Tensor
    terminated: torch.Tensor
    sample_ids: torch.Tensor
    actor_version: torch.Tensor
    critic_version: torch.Tensor
    sampling: torch.Tensor

    @property
    def inputs(self) -> TPModelInputs:
        """Expose token, validity and position tensors without copying the rollout."""
        return TPModelInputs(self.input_ids, self.attention_mask, self.position_ids)

    def packet(self) -> dict[str, torch.Tensor]:
        """Return the ordered transport fields, sharing the underlying tensors."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def validate(self, *, actor_version: int, critic_version: int) -> None:
        """Reject malformed action slots, invalid sampling or stale model versions."""
        self.inputs.validate()
        batch, sequence = self.input_ids.shape
        if sequence < 2:
            raise ValueError("PPO sequences require at least two token slots")
        if any(t.device != self.input_ids.device for t in self.packet().values()):
            raise ValueError("all rollout tensors must share a device")
        if self.action_mask.shape != (batch, sequence - 1):
            raise ValueError("action_mask must be [B, S-1]")
        if self.action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be bool")
        if self.behavior_logps.shape != self.action_mask.shape:
            raise ValueError("behavior_logps must be [B, S-1]")
        if not self.behavior_logps.is_floating_point():
            raise ValueError("behavior_logps must be floating point")
        for name, dtype in (("terminated", torch.bool), ("sample_ids", torch.int64)):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != dtype:
                raise ValueError(f"{name} must be {dtype} [B]")
        if self.sample_ids.unique().numel() != batch:
            raise ValueError("sample_ids must be unique within the batch")
        for name, expected in (
            ("actor_version", actor_version),
            ("critic_version", critic_version),
        ):
            value = getattr(self, name)
            if value.ndim != 0 or value.dtype != torch.int64 or value.item() != expected:
                raise ValueError(f"stale or invalid {name}; expected {expected}")
        if self.sampling.shape != (3,) or not torch.equal(
            self.sampling, self.sampling.new_tensor([1, 0, 1])
        ):
            raise ValueError("PPO requires temperature=1, top_k=0, top_p=1")
        if batch == 0:
            return
        mask = self.action_mask
        if not mask.any(dim=1).all():
            raise ValueError("every sequence must have a nonempty completion")
        pairs = self.attention_mask[:, :-1] & self.attention_mask[:, 1:]
        if (mask & ~pairs).any():
            raise ValueError("actions must have valid state and target tokens")
        # Both the attention span and completion span must be contiguous.
        for span in (self.attention_mask, mask):
            starts = span[:, :1].sum(dim=1) + ((~span[:, :-1]) & span[:, 1:]).sum(dim=1)
            if (starts != 1).any():
                raise ValueError("attention and completion spans must be contiguous")
        positions = torch.arange(sequence, device=self.input_ids.device)
        last_token = positions.expand(batch, -1).masked_fill(~self.attention_mask, -1).max(1).values
        last_action = positions[:-1].expand(batch, -1).masked_fill(~mask, -1).max(1).values
        if not torch.equal(last_action + 1, last_token):
            raise ValueError("completion must end at the last valid token")
        if not torch.isfinite(self.behavior_logps[mask]).all():
            raise ValueError("behavior logprobs must be finite on valid actions")
        if (self.behavior_logps[mask] > 1e-6).any():
            raise ValueError("behavior logprobs cannot be positive")


ROLLOUT_FIELDS = tuple(field.name for field in fields(PPORollout))
TARGET_FIELDS = (
    "old_values",
    "bootstrap_values",
    "rewards",
    "reference_logps",
    "advantages",
    "returns",
)


@dataclass(frozen=True)
class PPOBatch:
    """Rollout and frozen targets, with [B, S-1] action slots and [B] rewards/bootstrap."""

    rollout: PPORollout
    old_values: torch.Tensor
    bootstrap_values: torch.Tensor
    rewards: torch.Tensor
    reference_logps: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def validate(self) -> None:
        """Require finite frozen targets on valid action slots and the rollout device."""
        if self.rollout.behavior_logps.requires_grad:
            raise ValueError("behavior logprobs must be frozen before PPO training")
        self.rollout.validate(
            actor_version=int(self.rollout.actor_version.item()),
            critic_version=int(self.rollout.critic_version.item()),
        )
        count = self.rollout.input_ids.shape[0]
        mask = self.rollout.action_mask
        for name, value in self.targets().items():
            expected = (count,) if name in ("bootstrap_values", "rewards") else mask.shape
            if value.shape != expected or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating tensor of shape {expected}")
            if value.device != mask.device or value.requires_grad:
                raise ValueError(f"{name} must be a frozen tensor on the rollout device")
            valid = value if value.ndim == 1 else value[mask]
            if not torch.isfinite(valid).all():
                raise ValueError(f"{name} must be finite on valid entries")

    def select(self, indices: torch.Tensor) -> PPOBatch:
        """Select sequence rows while retaining shared version and sampling metadata."""
        packet = self.rollout.packet()
        for name in ROLLOUT_FIELDS:
            if name not in ("actor_version", "critic_version", "sampling"):
                packet[name] = packet[name][indices]
        return PPOBatch(
            PPORollout(**packet), **{name: getattr(self, name)[indices] for name in TARGET_FIELDS}
        )

    def targets(self) -> dict[str, torch.Tensor]:
        """Expose the frozen target fields without copying or detaching them."""
        return {name: getattr(self, name) for name in TARGET_FIELDS}


@torch.no_grad()
def build_ppo_batch(
    rollout: PPORollout,
    *,
    state_values: torch.Tensor,
    rewards: torch.Tensor,
    reference_logps: torch.Tensor,
    config: PPOConfig,
) -> PPOBatch:
    """Freeze old scores and compute terminal-aware masked GAE in float32.

    state_values[:, t] estimates the value before sampling token t+1. The
    final valid state's value is used only for a length-truncated completion.
    """
    rollout.validate(
        actor_version=int(rollout.actor_version.item()),
        critic_version=int(rollout.critic_version.item()),
    )
    mask = rollout.action_mask
    batch, width = mask.shape
    if state_values.shape != rollout.input_ids.shape or rewards.shape != (batch,):
        raise ValueError("state_values must be [B,S] and rewards [B]")
    if reference_logps.shape != mask.shape:
        raise ValueError("reference_logps must be [B,S-1]")
    for value in (state_values, rewards, reference_logps):
        if not value.is_floating_point() or value.device != mask.device:
            raise ValueError("scoring tensors must be floating point on the rollout device")
    if not (
        torch.isfinite(state_values[rollout.attention_mask]).all()
        and torch.isfinite(rewards).all()
        and torch.isfinite(reference_logps[mask]).all()
    ):
        raise ValueError("non-finite reward, value or reference score")
    old_values = state_values[:, :-1].float().masked_fill(~mask, 0)
    reference = reference_logps.float().masked_fill(~mask, 0)
    behavior = rollout.behavior_logps.float().masked_fill(~mask, 0)
    shaped = -config.kl_coefficient * (behavior - reference)
    positions = torch.arange(width, device=mask.device).expand(batch, -1)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    rows = torch.arange(batch, device=mask.device)
    bootstrap = state_values[rows, last + 1].float().masked_fill(rollout.terminated, 0)
    shaped[rows, last] += rewards.float()
    advantages = torch.zeros_like(old_values)
    carry = torch.zeros(batch, device=mask.device)
    next_value = bootstrap
    for index in range(width - 1, -1, -1):
        delta = shaped[:, index] + config.gamma * next_value - old_values[:, index]
        current = delta + config.gamma * config.gae_lambda * carry
        carry = torch.where(mask[:, index], current, carry)
        advantages[:, index] = torch.where(mask[:, index], carry, 0)
        next_value = torch.where(mask[:, index], old_values[:, index], next_value)
    returns = advantages + old_values
    valid = advantages[mask]
    if valid.numel():
        advantages = (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8)
        advantages.masked_fill_(~mask, 0)
    snapshot = PPORollout(
        **{name: tensor.detach().clone() for name, tensor in rollout.packet().items()}
    )
    return PPOBatch(
        snapshot, old_values, bootstrap, rewards.detach().clone(), reference, advantages, returns
    )


def ppo_policy_loss(logps: torch.Tensor, batch: PPOBatch, clip: float) -> torch.Tensor:
    """Average the clipped surrogate over valid actions, using frozen behavior scores."""
    mask = batch.rollout.action_mask
    ratio = (logps[mask].float() - batch.rollout.behavior_logps[mask].float()).exp()
    advantages = batch.advantages[mask]
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


def ppo_value_loss(values: torch.Tensor, batch: PPOBatch, clip: float) -> torch.Tensor:
    """Average the larger clipped/unclipped squared error over valid action states."""
    mask = batch.rollout.action_mask
    current = values[mask].float()
    previous = batch.old_values[mask]
    targets = batch.returns[mask]
    clipped = previous + (current - previous).clamp(-clip, clip)
    return 0.5 * torch.maximum((current - targets).square(), (clipped - targets).square()).mean()


@dataclass(frozen=True)
class PPOGradientState:
    """Explicit model ownership and writable unscaled optimizer gradient buffers.

    The adapter verifies ownership against its real optimizer, including FP32
    master weights. Buffers follow ``parameters`` order and may be materialized
    after backward; clipping must modify the very buffers consumed by step.
    """

    parameters: tuple[torch.nn.Parameter, ...]
    gradients: Callable[[], list[torch.Tensor | None]]
    zero_grad: Callable[[], None]
    accumulate: Callable[[], None] | None = None


class PPOTrainingWorker:
    """One role's optimizer; the adapter owns model-internal TP synchronization.

    forward returns selected logprobs [B,S-1] for actor or values [B,S] for
    critic. Use a dropout-free/eval-mode adapter for a consistent PPO policy;
    eval does not disable autograd. Every TP rank executes identical minibatches.
    parameter_layouts must describe every optimizer parameter explicitly.
    Frameworks may supply paired backward/step callbacks; clipping still acts
    on the explicitly supplied unscaled optimizer buffers between those calls.
    update_observer, when used, must be set on every TP rank of the group.
    """

    def __init__(
        self,
        *,
        role: Literal["actor", "critic"],
        model: torch.nn.Module,
        forward: Callable[[TPModelInputs], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        parameter_layouts: Mapping[str, str],
        group: dist.ProcessGroup,
        config: PPOConfig,
        backward: Callable[[torch.Tensor], None] | None = None,
        step: Callable[[], None] | None = None,
        gradient_state: PPOGradientState | None = None,
        update_observer: Callable | None = None,
    ) -> None:
        """Validate the role, callbacks and parameter ownership, raising ``ValueError``.

        ``backward`` and ``step`` must be given together. The optimized parameters (from
        ``gradient_state.parameters`` when given) must be unique and exactly the trainable
        ones, each with a ``sharded`` or ``replicated`` layout. Without framework callbacks,
        ``backward`` is ``torch.Tensor.backward`` and ``step`` is ``optimizer.step``.
        """
        if role not in ("actor", "critic"):
            raise ValueError("role must be actor or critic")
        if (backward is None) != (step is None):
            raise ValueError("framework backward and step callbacks must be provided together")
        named = dict(model.named_parameters())
        optimized = (
            list(gradient_state.parameters)
            if gradient_state is not None
            else [p for entry in optimizer.param_groups for p in entry["params"]]
        )
        if len({id(p) for p in optimized}) != len(optimized):
            raise ValueError("optimizer model parameters must be unique")
        if set(parameter_layouts) != {n for n, p in named.items() if p.requires_grad}:
            raise ValueError("parameter_layouts must cover every trainable parameter")
        if {id(p) for p in optimized} != {id(p) for p in named.values() if p.requires_grad}:
            raise ValueError("optimizer must own exactly the trainable model parameters")
        if any(layout not in ("sharded", "replicated") for layout in parameter_layouts.values()):
            raise ValueError("unknown parameter layout")
        self.role, self.model, self.forward = role, model, forward
        self.optimizer, self.group, self.config = optimizer, group, config
        self.gradient_state = gradient_state
        self.update_observer = update_observer
        self.backward = backward or torch.Tensor.backward
        self.step = step or optimizer.step
        self.parameters = [(named[name], layout) for name, layout in parameter_layouts.items()]

    def _observe(self, observer, event, update, gradients, value, device):
        """Call ``observer`` if set and raise any rank's observer failure on the whole TP group."""
        if observer is None:
            return
        error = None
        try:
            observer(event, update, gradients, value)
        except Exception as exc:
            error = f"PPO evidence observer failed: {type(exc).__name__}: {exc}"
        synchronize_validation(error, group=self.group, device=device)

    def train(self, batch: PPOBatch) -> dict[str, float | int]:
        """Update this role on identical TP minibatches with token-weighted accumulation.

        Targets stay frozen; TP-global clipping modifies the optimizer's actual
        gradient buffers. Empty batches perform no steps. Restore module modes
        on exit and return update count, mean loss and mean pre-clip norm.
        """
        error = None
        try:
            batch.validate()
        except ValueError as exc:
            error = str(exc)
        device = batch.rollout.input_ids.device
        synchronize_validation(error, group=self.group, device=device)
        # Observer callbacks add synchronizations, so TP ranks must all have one or none.
        observer = self.update_observer
        present = int(observer is not None)
        flags = torch.tensor([present, -present], dtype=torch.int64, device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=self.group)
        if int(flags[0]) != -int(flags[1]):
            raise ValueError("update_observer must be set on every TP rank or on none")
        count = batch.rollout.input_ids.shape[0]
        if count == 0:
            return {"updates": 0, "loss": 0.0, "grad_norm": 0.0}
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        losses: list[float] = []
        norms: list[float] = []
        try:
            for _ in range(self.config.epochs):
                # Stable order is shared across independent TP groups without RNG coupling.
                for start in range(0, count, self.config.minibatch_size):
                    indices = torch.arange(
                        start,
                        min(start + self.config.minibatch_size, count),
                        device=batch.rollout.input_ids.device,
                    )
                    part = batch.select(indices)
                    if self.gradient_state is None:
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.gradient_state.zero_grad()
                    micro_size = self.config.microbatch_size or indices.numel()
                    token_count = part.rollout.action_mask.sum()
                    loss_value = 0.0
                    for offset in range(0, indices.numel(), micro_size):
                        micro = part.select(
                            torch.arange(
                                offset,
                                min(offset + micro_size, indices.numel()),
                                device=indices.device,
                            )
                        )
                        output = self.forward(micro.rollout.inputs)
                        expected = (
                            micro.rollout.action_mask.shape
                            if self.role == "actor"
                            else micro.rollout.input_ids.shape
                        )
                        error = None
                        if (
                            not isinstance(output, torch.Tensor)
                            or output.shape != expected
                            or not output.requires_grad
                        ):
                            error = (
                                f"{self.role} adapter must return differentiable scores {expected}"
                            )
                        synchronize_validation(error, group=self.group, device=indices.device)
                        loss = (
                            ppo_policy_loss(output, micro, self.config.policy_clip)
                            if self.role == "actor"
                            else ppo_value_loss(output[:, :-1], micro, self.config.value_clip)
                        )
                        synchronize_validation(
                            None if torch.isfinite(loss) else "non-finite PPO loss",
                            group=self.group,
                            device=indices.device,
                        )
                        weight = micro.rollout.action_mask.sum() / token_count
                        loss_value += float(loss.detach() * weight)
                        self.backward(loss * weight)
                        if (
                            self.gradient_state is not None
                            and self.gradient_state.accumulate is not None
                        ):
                            self.gradient_state.accumulate()
                    gradients = None
                    error = None
                    try:
                        if self.gradient_state is not None:
                            buffers = self.gradient_state.gradients()
                            if len(buffers) != len(self.gradient_state.parameters):
                                raise ValueError("optimizer gradient buffer count changed")
                            by_id = dict(
                                zip((id(p) for p in self.gradient_state.parameters), buffers)
                            )
                            gradients = [by_id[id(p)] for p, _ in self.parameters]
                    except (ValueError, RuntimeError, KeyError) as exc:
                        error = str(exc)
                    synchronize_validation(error, group=self.group, device=indices.device)
                    self._observe(
                        observer, "before_clip", len(losses), gradients, loss_value, indices.device
                    )
                    norm = clip_tp_grad_norm_(
                        self.parameters,
                        gradients=gradients,
                        group=self.group,
                        max_norm=self.config.max_grad_norm,
                        device=indices.device,
                    )
                    self._observe(
                        observer, "after_clip", len(losses), gradients, float(norm), indices.device
                    )
                    self.step()
                    self._observe(
                        observer, "after_step", len(losses), None, loss_value, indices.device
                    )
                    losses.append(loss_value)
                    norms.append(float(norm))
        finally:
            for module, training in modes:
                module.training = training
        return {"updates": len(losses), "loss": sum(losses) / len(losses), "grad_norm": max(norms)}
