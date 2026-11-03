# SPDX-License-Identifier: Apache-2.0
"""Trainable P5 module for local dispatched rows; P4/P6 own transport and combine."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from rl_engine.contracts.operators.expert_training import (
    SCHEMA,
    ExpertTrainingBackend,
    ExpertTrainingConfig,
    PackedExpertWeight,
    require_tensor,
    validate_request,
)
from rl_engine.ops.moe.expert_training import routed_expert_training


class RoutedExpertMLP(nn.Module):
    """Frozen MXFP4 base and shared BF16 A1/B1/A2/B2 adapters over routed rows.

    Parameters are visible to standard optimizers. The packed base is registered
    as buffers; no base dW edge exists. Engine/optimizer wiring is task 72/73.
    """

    w1_codes: Tensor | None
    w1_scales: Tensor | None
    w2_codes: Tensor | None
    w2_scales: Tensor | None

    def __init__(
        self,
        config: ExpertTrainingConfig,
        *,
        backend: ExpertTrainingBackend,
        w1: PackedExpertWeight | None = None,
        w2: PackedExpertWeight | None = None,
        adapters: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        recompute: bool = False,
    ):
        super().__init__()
        self.config, self.backend, self.recompute = config, backend, recompute
        self._registered_config = config
        for name, weight in (("w1", w1), ("w2", w2)):
            if weight is not None:
                expect = (
                    (config.experts, 2 * config.ffn, config.hidden)
                    if name == "w1"
                    else (config.experts, config.hidden, config.ffn)
                )
                weight.validate(expect, weight.codes.device)
            self.register_buffer(
                name + "_codes", None if weight is None else weight.codes.detach().clone()
            )
            self.register_buffer(
                name + "_scales", None if weight is None else weight.scales.detach().clone()
            )
        if adapters is not None and len(adapters) != 4:
            raise ValueError("adapters must contain A1, B1, A2 and B2")
        for name, value in zip(("a1", "b1", "a2", "b2"), adapters or (None,) * 4):
            # Preserve intentional parameter freezing; supplied plain tensors are trainable.
            trainable = value.requires_grad if isinstance(value, nn.Parameter) else True
            self.register_parameter(
                name,
                (
                    None
                    if value is None
                    else nn.Parameter(value.detach().clone(), requires_grad=trainable)
                ),
            )
        roots = [t for t in self.buffers()] + [t for t in self.parameters()]
        if not roots:
            raise ValueError("expert module requires base weights or LoRA parameters")
        device = roots[0].device
        x = torch.empty((0, config.hidden), device=device, dtype=torch.bfloat16)
        p = torch.empty(0, device=device, dtype=torch.float32)
        offsets = torch.zeros(config.experts + 1, device=device, dtype=torch.int64)
        validate_request(config, x, p, offsets, *self.base_weights(), self.adapters())

    def adapters(self):
        return self.a1, self.b1, self.a2, self.b2

    def get_extra_state(self):
        return {"schema": SCHEMA, "config": asdict(self.config)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("expert checkpoint schema/config mismatch, including mode and alpha")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Validate the local contract before overwriting this module's tensors.
        self.set_extra_state(state_dict.get(prefix + "_extra_state"))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def base_weights(self) -> tuple[PackedExpertWeight | None, PackedExpertWeight | None]:
        c = self.config
        # Rebuild views after .to() / load_state_dict; never retain stale buffer references.
        weights: list[PackedExpertWeight | None] = []
        for codes, scales, shape in (
            (self.w1_codes, self.w1_scales, (c.experts, 2 * c.ffn, c.hidden)),
            (self.w2_codes, self.w2_scales, (c.experts, c.hidden, c.ffn)),
        ):
            if codes is None and scales is None:
                weights.append(None)
            elif codes is None or scales is None:
                raise ValueError("packed codes and scales must both be present")
            else:
                weights.append(PackedExpertWeight(codes, scales, shape))
        return weights[0], weights[1]

    def validate_trainable_parameters(self):
        if self.config != self._registered_config:
            raise RuntimeError("expert config changed; construct a new module for a new contract")
        allowed = {"a1", "b1", "a2", "b2"} if self.config.use_lora else set()
        unexpected = {name for name, p in self.named_parameters() if p.requires_grad} - allowed
        if unexpected:
            raise ValueError(f"unsupported trainable expert parameters: {sorted(unexpected)}")
        if any(t.requires_grad for t in self.buffers()):
            raise ValueError("expert base buffers must be frozen")

    def forward(self, x: Tensor, p_s: Tensor, expert_offsets: Tensor) -> Tensor:
        self.validate_trainable_parameters()
        return routed_expert_training(
            self.config,
            self.backend,
            x,
            p_s,
            expert_offsets,
            *self.base_weights(),
            self.adapters(),
            recompute=self.recompute,
        )


@dataclass(frozen=True)
class ExpertTrainingOutputs:
    routed: Tensor
    output_slot: Tensor
    shared: Tensor | None
    route_weight_applied: bool = True


class ExpertTrainingBlock(nn.Module):
    """One call per local layer execution; keep routed/shared outputs separate for P6.

    The supplied task 47 module owns shared-expert arithmetic and parameter
    validation. It receives original tokens, never expanded dispatched rows.
    This wrapper performs no TP/EP collectives, residual addition or slot reduction.
    """

    def __init__(self, routed: RoutedExpertMLP, shared: nn.Module | None = None):
        super().__init__()
        self.routed, self.shared = routed, shared

    def forward(self, x, p_s, offsets, output_slot, *, shared_x=None):
        if output_slot.dtype not in (torch.int32, torch.int64):
            raise TypeError("output_slot must contain integer slot IDs")
        require_tensor(output_slot, (x.shape[0],), output_slot.dtype, x.device, "output_slot")
        if bool((output_slot < 0).any()) or torch.unique(output_slot).numel() != x.shape[0]:
            raise ValueError("output_slot IDs must be nonnegative and unique")
        if (self.shared is None) != (shared_x is None):
            raise ValueError("provide original shared_x exactly when a shared module is configured")
        if shared_x is not None:
            if shared_x.ndim != 2:
                raise ValueError("shared_x must be [T, hidden]")
            require_tensor(
                shared_x,
                (shared_x.shape[0], self.routed.config.hidden),
                torch.bfloat16,
                x.device,
                "shared_x",
            )
        routed = self.routed(x, p_s, offsets)
        shared = None if self.shared is None else self.shared(shared_x)
        if shared is not None:
            require_tensor(shared, tuple(shared_x.shape), torch.bfloat16, x.device, "shared output")
            needs_grad = torch.is_grad_enabled() and (
                shared_x.requires_grad or any(p.requires_grad for p in self.shared.parameters())
            )
            if needs_grad and not shared.requires_grad:
                raise RuntimeError("task 47 shared module detached a required training branch")
        return ExpertTrainingOutputs(routed, output_slot.clone(), shared)
