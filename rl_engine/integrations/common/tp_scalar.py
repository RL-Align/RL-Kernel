# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Explicit TP scalar-head contracts for reward inference and critic training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

from rl_engine.integrations.common.weights import synchronize_validation


@dataclass(frozen=True)
class TPModelInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    def validate(self) -> None:
        """Raise ``ValueError`` unless the inputs form a well-typed ``[B, S]`` batch on one device.

        ``attention_mask`` must be bool and ``position_ids`` int64, both matching ``input_ids``;
        every sequence needs at least one valid token.
        """
        if self.input_ids.ndim != 2 or self.input_ids.dtype != torch.int64:
            raise ValueError("input_ids must be int64 [B, S]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if self.attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be bool")
        if self.position_ids.shape != self.input_ids.shape:
            raise ValueError("position_ids must match input_ids")
        if self.position_ids.dtype != torch.int64:
            raise ValueError("position_ids must be int64")
        if any(t.device != self.input_ids.device for t in (self.attention_mask, self.position_ids)):
            raise ValueError("model inputs must share a device")
        if self.input_ids.shape[0] and not self.attention_mask.any(dim=1).all():
            raise ValueError("each sequence must contain a valid token")

    def select(self, indices: torch.Tensor) -> TPModelInputs:
        """Return the rows chosen by ``indices`` from all three tensors."""
        return TPModelInputs(
            self.input_ids[indices], self.attention_mask[indices], self.position_ids[indices]
        )


class _SumLocalHead(torch.autograd.Function):
    """SUM forward, identity backward for a replicated loss on every TP rank.

    Each input is an additive head contribution, not a replicated parameter.
    Model adapters own backbone/replicated-parameter gradient synchronization.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """All-reduce (SUM) a clone of ``value`` over ``group``, leaving the input unchanged."""
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        """Pass the gradient through unchanged; ``group`` receives no gradient."""
        return gradient, None


def sum_tp_contributions(value: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sum additive outputs for identical losses on all ranks (identity backward)."""
    if group is None or dist.get_rank(group) < 0:
        raise ValueError("an explicit TP group containing this rank is required")
    return _SumLocalHead.apply(value, group) if dist.get_world_size(group) > 1 else value


class TPScalarExecutor:
    """Model returns a tensor [B, S] (critic) or [B]/[B, S] (reward).

    ``local_sum`` means disjoint additive head contributions: any shared bias
    must be included on exactly one rank. ``replicated`` means the model has
    already completed its own TP communication. No layout is inferred.
    All group members call each method; only leader receives a scalar reward.
    The model adapter accepts input_ids, attention_mask, position_ids, use_cache.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        group: dist.ProcessGroup,
        leader: int,
        output_layout: Literal["replicated", "local_sum"],
    ) -> None:
        """Raise ``ValueError`` unless this rank and ``leader`` belong to ``group``.

        ``output_layout`` must be ``replicated`` or ``local_sum``; reduction counters start at 0.
        """
        if group is None or dist.get_rank(group) < 0:
            raise ValueError("an explicit TP group containing this rank is required")
        if leader not in dist.get_process_group_ranks(group):
            raise ValueError("leader must be a global rank in the TP group")
        if output_layout not in ("replicated", "local_sum"):
            raise ValueError("output_layout must be replicated or local_sum")
        self.model = model
        self.group = group
        self.leader = leader
        self.output_layout = output_layout
        self.head_metrics = {"reduce_calls": 0, "payload_bytes": 0}

    def _forward(self, inputs: TPModelInputs, *, scalar_allowed: bool) -> torch.Tensor:
        """Validate inputs, run the model with ``use_cache=False`` and return its unreduced output.

        The output must be a finite floating tensor on the input device, shaped ``[B, S]`` or,
        when ``scalar_allowed``, ``[B]``. An input or output check failing on any rank raises
        ``ValueError`` on all ranks; exceptions from the model itself are not synchronized.
        """
        error = None
        try:
            inputs.validate()
        except ValueError as exc:
            error = str(exc)
        synchronize_validation(error, group=self.group, device=inputs.input_ids.device)
        output = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            position_ids=inputs.position_ids,
            use_cache=False,
        )
        shapes = [inputs.input_ids.shape]
        if scalar_allowed:
            shapes.append(inputs.input_ids.shape[:1])
        error = None
        if not isinstance(output, torch.Tensor):
            error = "scalar model adapter must return a tensor"
        elif output.shape not in shapes or not output.is_floating_point():
            error = "scalar model adapter returned an invalid shape or dtype"
        elif output.device != inputs.input_ids.device:
            error = "scalar model output must be on the input device"
        elif not torch.isfinite(output).all():
            error = "scalar model output must be finite"
        synchronize_validation(error, group=self.group, device=inputs.input_ids.device)
        return output

    def values(self, inputs: TPModelInputs) -> torch.Tensor:
        """Return differentiable [B, S] state values on every TP rank."""
        values = self._forward(inputs, scalar_allowed=False)
        if self.output_layout == "local_sum" and dist.get_world_size(self.group) > 1:
            self.head_metrics["reduce_calls"] += 1
            self.head_metrics["payload_bytes"] += values.numel() * values.element_size()
            values = sum_tp_contributions(values, self.group)
        return values

    @torch.no_grad()
    def rewards(self, inputs: TPModelInputs) -> torch.Tensor | None:
        """Pool locally before reducing [B]; preserve the caller's model modes."""
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            rewards = self._forward(inputs, scalar_allowed=True)
            if rewards.ndim == 2:
                positions = torch.arange(rewards.shape[1], device=rewards.device)
                last = positions.expand_as(rewards).masked_fill(~inputs.attention_mask, -1)
                rewards = rewards.gather(1, last.max(dim=1).values[:, None]).squeeze(1)
            if self.output_layout == "local_sum" and dist.get_world_size(self.group) > 1:
                rewards = rewards.clone()
                dist.reduce(rewards, dst=self.leader, group=self.group)
                self.head_metrics["reduce_calls"] += 1
                self.head_metrics["payload_bytes"] += rewards.numel() * rewards.element_size()
            return rewards if dist.get_rank() == self.leader else None
        finally:
            for module, training in modes:
                module.training = training


def clip_tp_grad_norm_(
    parameters: list[tuple[torch.nn.Parameter, str]],
    *,
    group: dist.ProcessGroup,
    max_norm: float,
    device: torch.device,
    gradients: list[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Clip TP gradients; layouts are 'sharded' or already-synchronized 'replicated'.

    Replicated gradients contribute once to the global norm, but are scaled on
    every rank. This function never synchronizes model parameter gradients.
    """
    if group is None or dist.get_rank(group) < 0:
        raise ValueError("an explicit TP group containing this rank is required")
    error = None
    grads = [p.grad for p, _ in parameters] if gradients is None else gradients
    if len(grads) != len(parameters):
        error = "gradient buffers must cover every parameter"
    elif any(g is not None and g.shape != p.shape for g, (p, _) in zip(grads, parameters)):
        error = "gradient buffer shapes must match parameters"
    if max_norm <= 0:
        error = "max_norm must be positive"
    if any(layout not in ("sharded", "replicated") for _, layout in parameters):
        error = "parameter layout must be sharded or replicated"
    if any(
        g is not None and (g.is_sparse or g.device != device or not g.is_floating_point())
        for g in grads
    ):
        error = "gradients must be dense and on the configured device"
    synchronize_validation(error, group=group, device=device)
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for grad, (_, layout) in zip(grads, parameters):
        if grad is not None and (layout == "sharded" or dist.get_rank(group) == 0):
            squared += grad.detach().to(torch.float64).square().sum()
    dist.all_reduce(squared, group=group)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise ValueError("non-finite TP gradient norm")
    scale = (max_norm / (norm + 1e-6)).clamp(max=1)
    with torch.no_grad():
        for grad in grads:
            if grad is not None:
                grad.mul_(scale.to(grad.dtype))
    return norm
