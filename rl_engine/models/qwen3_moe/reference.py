# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B assembly. It reads ``Qwen3MoeSpec`` and does not replace WS1.

Expert MLP, shared experts, and GDN stay out until tasks 17–22 provide them.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from rl_engine.layers.moe import (
    NATIVE_OPS_WIRED,
    REFERENCE_BACKEND,
    LayerDims,
    OperatorDeferred,
    forward_layer,
    planned_layer_nodes,
)
from rl_engine.models.qwen3_moe.spec import TEXT_WEIGHT_PREFIX, Qwen3MoeSpec


class Qwen3MoeReferenceModel:
    """Teacher-forced text stack. Vision and MTP stay outside this forward."""

    def __init__(self, spec: Qwen3MoeSpec, weights: Mapping[str, torch.Tensor]):
        self.spec = spec
        self.weights = dict(weights)
        self.dims = LayerDims.from_spec(spec)
        self.native_ops_wired = NATIVE_OPS_WIRED
        self.backend = REFERENCE_BACKEND
        if spec.text_forward_includes_vision or spec.text_forward_includes_mtp:
            raise ValueError("this reference forward is the text stack only")
        check_weight_shapes(spec, self.weights, complete=True)

    def node_names(self) -> tuple[str, ...]:
        return self.spec.node_names()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        target_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        capture_nodes: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
        nodes: dict[str, torch.Tensor] | None = {} if capture_nodes else None
        hidden = F.embedding(input_ids, self.weights[f"{TEXT_WEIGHT_PREFIX}.embed_tokens.weight"])
        if nodes is not None:
            nodes["embedding"] = hidden
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.shape[0], -1)
        for index, layer_type in enumerate(self.spec.layer_types):
            layer_nodes: dict[str, torch.Tensor] | None = {} if capture_nodes else None
            layer_result = forward_layer(
                hidden,
                _layer_weights(self.weights, index),
                self.dims,
                layer_type,
                position_ids=position_ids,
                nodes=layer_nodes,
            )
            if nodes is not None and layer_nodes is not None:
                prefix = f"layers.{index}."
                nodes.update({prefix + name: value for name, value in layer_nodes.items()})
            if not layer_result.complete:
                raise OperatorDeferred(layer_result.pending_tasks)
        hidden = _final_norm(hidden, self.weights[f"{TEXT_WEIGHT_PREFIX}.norm.weight"], self.spec.rms_norm_eps)
        if nodes is not None:
            nodes["final_layernorm"] = hidden
        logits = F.linear(hidden, self.weights["lm_head.weight"])
        if nodes is not None:
            nodes["lm_head"] = logits
        result = {"logits": logits, "hidden": hidden}
        if target_ids is not None:
            logp = _selected_logp(logits, target_ids)
            if nodes is not None:
                nodes["logprob"] = logp
            result["selected_logp"] = logp
            if loss_mask is not None:
                result["loss"] = _masked_loss(logp, loss_mask, nodes)
        if nodes is not None:
            result["nodes"] = nodes
        return result


def check_weight_shapes(
    spec: Qwen3MoeSpec,
    weights: Mapping[str, torch.Tensor],
    *,
    complete: bool = False,
) -> None:
    """Reject a tensor whose shape is not the one pinned by the spec."""

    if complete:
        missing = [name for name in spec.parameter_names() if name not in weights]
        if missing:
            raise KeyError(f"missing {len(missing)} weights, first: {missing[0]}")
    for name, tensor in weights.items():
        expected = spec.parameter_shape(name)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected}")


def layer_suffixes(spec: Qwen3MoeSpec, layer_index: int) -> tuple[str, ...]:
    prefix = f"layers.{layer_index}."
    return tuple(name[len(prefix) :] for name in spec.node_names() if name.startswith(prefix))


def _layer_weights(weights: Mapping[str, torch.Tensor], layer_index: int) -> dict[str, torch.Tensor]:
    prefix = f"{TEXT_WEIGHT_PREFIX}.layers.{layer_index}."
    return {key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)}


def _final_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = values.float().pow(2).mean(dim=-1, keepdim=True)
    normed = values.float() * torch.rsqrt(variance + eps)
    return (normed * (1.0 + weight.float())).to(dtype=values.dtype)


def _selected_logp(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must match logits batch and sequence")
    if logits.shape[1] < 2:
        raise ValueError("causal selected-logprob requires sequence length >= 2")
    scores = logits[:, :-1, :].float()
    targets = target_ids[:, 1:]
    logp = torch.log_softmax(scores, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _masked_loss(
    logp: torch.Tensor,
    loss_mask: torch.Tensor,
    nodes: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    mask = loss_mask[:, 1:].to(dtype=torch.bool, device=logp.device)
    active = mask.sum().clamp_min(1)
    loss = -(logp.float().masked_fill(~mask, 0).sum() / active.float())
    if nodes is not None:
        nodes["loss"] = loss
    return loss


__all__ = [
    "Qwen3MoeReferenceModel",
    "check_weight_shapes",
    "layer_suffixes",
    "planned_layer_nodes",
]
