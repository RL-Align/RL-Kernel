# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Explicit TP scalar-head contracts for reward inference and critic training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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


def balanced_tree_sum(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum ``parts`` pairwise in the v0.1.0 fixed balanced tree, lower subtree first.

    For four inputs this is ``(p0 + p1) + (p2 + p3)``: the order used by
    ``DeterministicCollective.all_reduce`` across ranks. Contiguous leaf blocks
    therefore reduce to the same value whether one rank or several hold them.
    """
    nodes = list(parts)
    if not nodes:
        raise ValueError("balanced tree sum requires at least one input")
    stride = 1
    while stride < len(nodes):
        for index in range(0, len(nodes) - stride, 2 * stride):
            nodes[index] = nodes[index] + nodes[index + stride]
        stride *= 2
    return nodes[0]


def deterministic_tp_all_reduce_(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Replace ``tensor`` with its fixed-tree SUM over ``group`` on every rank.

    CUDA FP32/BF16/FP16 tensors use RL-Kernel's ``DeterministicCollective``
    (via the shared linear-logp helper). Other tensors, including FP64 norm
    partials and CPU tensors, are all-gathered (bit-exact transport) and summed
    locally in the same balanced tree, so every rank and every backend agrees.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return tensor
    if tensor.is_cuda and tensor.dtype in (torch.float32, torch.bfloat16, torch.float16):
        from rl_engine.reference.logprob.linear_logp import _deterministic_tp_all_reduce_

        if tensor.is_contiguous():
            return _deterministic_tp_all_reduce_(tensor, group)
        reduced = _deterministic_tp_all_reduce_(tensor.contiguous(), group)
        return tensor.copy_(reduced)
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    with torch.no_grad():
        return tensor.copy_(balanced_tree_sum(gathered))


class _SumLocalHead(torch.autograd.Function):
    """SUM forward, identity backward for a replicated loss on every TP rank.

    Each input is an additive head contribution, not a replicated parameter.
    Model adapters own backbone/replicated-parameter gradient synchronization.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """Fixed-tree SUM of a clone of ``value`` over ``group``, leaving the input unchanged."""
        return deterministic_tp_all_reduce_(value.clone(), group)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        """Pass the gradient through unchanged; ``group`` receives no gradient."""
        return gradient, None


def sum_tp_contributions(value: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sum additive outputs for identical losses on all ranks (identity backward).

    The cross-rank combine is the v0.1.0 fixed tree. Bitwise TP invariance also
    requires each rank's input to be its canonical subtree root (see
    ``canonical_feature_dot``).
    """
    if group is None or dist.get_rank(group) < 0:
        raise ValueError("an explicit TP group containing this rank is required")
    return _SumLocalHead.apply(value, group) if dist.get_world_size(group) > 1 else value


def canonical_chunks(tp_world: int, canonical_tp: int | None) -> int:
    """Return the canonical leaf count per rank; ``None`` means one leaf per rank."""
    canonical = tp_world if canonical_tp is None else canonical_tp
    if tp_world < 1 or canonical < tp_world or canonical % tp_world:
        raise ValueError("canonical TP must be a positive multiple of physical TP")
    chunks = canonical // tp_world
    if chunks & (chunks - 1):
        raise ValueError("canonical TP chunks must form a balanced binary tree")
    return chunks


def canonical_feature_dot(
    features: torch.Tensor, weight: torch.Tensor, *, tp_world: int, canonical_tp: int
) -> torch.Tensor:
    """FP32 ``features @ weight`` over this rank's feature shard as a canonical subtree root.

    The shard is split into ``canonical_tp / tp_world`` equal leaves, each an FP32
    product-sum over the same leaf width at every TP size, and the leaves are
    tree-summed. Followed by ``sum_tp_contributions`` the result is the same
    balanced tree over ``canonical_tp`` leaves at TP 1, 2, 4 or 8.
    """
    chunks = canonical_chunks(tp_world, canonical_tp)
    if weight.ndim != 1 or features.shape[-1] != weight.shape[0] or weight.shape[0] % chunks:
        raise ValueError("feature shard does not divide the canonical TP leaves")
    width = weight.shape[0] // chunks
    return balanced_tree_sum(
        [
            (
                features[..., i * width : (i + 1) * width].float()
                * weight[i * width : (i + 1) * width].float()
            ).sum(-1)
            for i in range(chunks)
        ]
    )


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
                rewards = deterministic_tp_all_reduce_(rewards.clone(), self.group)
                self.head_metrics["reduce_calls"] += 1
                self.head_metrics["payload_bytes"] += rewards.numel() * rewards.element_size()
            return rewards if dist.get_rank() == self.leader else None
        finally:
            for module, training in modes:
                module.training = training


def _canonical_square_sums(
    grad: torch.Tensor, parameter: torch.nn.Parameter, chunks: int, canonical: bool = True
) -> list[torch.Tensor]:
    """FP64 squared sums of a sharded gradient's canonical leaves, in leaf order.

    The shard is cut along Megatron's ``partition_dim`` into ``chunks`` pieces
    per ``partition_stride`` block; leaf ``i`` concatenates piece ``i`` of every
    block, so leaf contents and shapes are independent of the TP size. With
    ``canonical`` the sum runs block by block even when the rank holds one leaf,
    as smaller TP sizes sum it. Without ``canonical`` (canonical clipping off) a
    one-leaf rank sums the whole shard. A gradient without ``partition_dim`` is
    not a canonical shard; ``clip_tp_grad_norm_`` counts it separately.
    """
    dim = getattr(parameter, "partition_dim", None)
    if chunks == 1 and (dim is None or not canonical):
        return [grad.detach().to(torch.float64).square().sum()]
    stride = int(getattr(parameter, "partition_stride", 1))
    if dim is None or grad.shape[dim] % (stride * chunks):
        raise ValueError("canonical TP clipping requires divisible Megatron partition_dim shards")
    blocks = grad.detach().chunk(stride, dim=dim)
    sums = []
    for i in range(chunks):
        leaf = [block.chunk(chunks, dim=dim)[i].contiguous().to(torch.float64) for block in blocks]
        total = leaf[0].square().sum()
        for piece in leaf[1:]:
            total = total + piece.square().sum()
        sums.append(total)
    return sums


def clip_tp_grad_norm_(
    parameters: list[tuple[torch.nn.Parameter, str]],
    *,
    group: dist.ProcessGroup,
    max_norm: float,
    device: torch.device,
    gradients: list[torch.Tensor | None] | None = None,
    canonical_tp: int | None = None,
) -> torch.Tensor:
    """Clip TP gradients; layouts are 'sharded' or already-synchronized 'replicated'/'partial'.

    Replicated gradients contribute once to the global norm, but are scaled on
    every rank. This function never synchronizes model parameter gradients.

    The squared norm follows the v0.1.0 reduction tree: every rank forms
    ``canonical_tp / TP`` FP64 leaves (sharded gradients split along their
    Megatron ``partition_dim``; replicated ones added to global leaf 0 once),
    sums each leaf over parameters in order, tree-sums its leaves and combines
    ranks with the fixed tree. With a fixed ``canonical_tp`` and identical full
    gradients the norm is bitwise identical at every TP size; ``None`` uses one
    leaf per rank.

    With ``canonical_tp``, a sharded gradient without ``partition_dim`` (the scalar
    adapter's leader-only bias) is not split into leaves: its holder sums it, in
    parameter order, into a separate FP64 term. That term is reduced together with
    the leaf tree (one collective of two values) and added last. One rank holds it
    and the others contribute zero, so its reduction is exact, and the norm keeps
    the same bits at every TP size whichever rank the leader is.
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
    if any(layout not in ("sharded", "replicated", "partial") for _, layout in parameters):
        error = "parameter layout must be sharded, replicated or partial"
    if any(
        g is not None and (g.is_sparse or g.device != device or not g.is_floating_point())
        for g in grads
    ):
        error = "gradients must be dense and on the configured device"
    try:
        chunks = canonical_chunks(dist.get_world_size(group), canonical_tp)
    except ValueError as exc:
        error, chunks = str(exc), 1
    synchronize_validation(error, group=group, device=device)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    leaves = [zero.clone() for _ in range(chunks)]
    held = zero.clone()
    canonical = canonical_tp is not None
    first_rank = dist.get_rank(group) == 0
    error = None
    try:
        for grad, (parameter, layout) in zip(grads, parameters):
            if grad is None:
                continue
            if layout == "sharded":
                if canonical and getattr(parameter, "partition_dim", None) is None:
                    held = held + grad.detach().to(torch.float64).square().sum()
                    continue
                sums = _canonical_square_sums(grad, parameter, chunks, canonical=canonical)
                for i, value in enumerate(sums):
                    leaves[i] = leaves[i] + value
            elif first_rank:
                leaves[0] = leaves[0] + grad.detach().to(torch.float64).square().sum()
    except ValueError as exc:
        error = str(exc)
    synchronize_validation(error, group=group, device=device)
    if canonical:
        totals = deterministic_tp_all_reduce_(torch.stack([balanced_tree_sum(leaves), held]), group)
        squared = totals[0] + totals[1]
    else:
        squared = deterministic_tp_all_reduce_(balanced_tree_sum(leaves).reshape(1), group)[0]
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise ValueError("non-finite TP gradient norm")
    scale = (max_norm / (norm + 1e-6)).clamp(max=1)
    with torch.no_grad():
        for grad in grads:
            if grad is not None:
                grad.mul_(scale.to(grad.dtype))
    return norm
