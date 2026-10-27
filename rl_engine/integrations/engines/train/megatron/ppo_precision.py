# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 master updates for BF16 Megatron parameters without loss scaling."""

from __future__ import annotations

from typing import Callable

import torch

from rl_engine.integrations.common.ppo import PPOGradientState


class FP32MasterWeights:
    """Preserve sub-BF16 updates and expose the exact gradients consumed by step.

    Megatron's unfused local layers write parameter.grad. Fused main_grad buffers
    are deliberately rejected; this adapter must not silently read a stale grad.
    """

    def __init__(self, parameters, optimizer_factory: Callable):
        """Clone trainable BF16 ``parameters`` into FP32 masters and build ``gradient_state``.

        ``optimizer_factory(masters)`` must return an optimizer owning exactly those masters.
        Non-trainable parameters are skipped. Raises ``ValueError`` for no trainable, duplicate,
        non-BF16 or ``main_grad``-carrying parameters, or an optimizer not owning the masters.
        """
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        if not self.parameters or len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("master weights require unique trainable parameters")
        if any(p.dtype != torch.bfloat16 for p in self.parameters):
            raise ValueError("master weights require BF16 model parameters")
        if any(hasattr(p, "main_grad") for p in self.parameters):
            raise ValueError("fused main_grad buffers are not supported by this adapter")
        self.masters = tuple(
            torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters
        )
        self.optimizer = optimizer_factory(self.masters)
        owned = [p for group in self.optimizer.param_groups for p in group["params"]]
        if len(owned) != len(self.masters) or {id(p) for p in owned} != {
            id(p) for p in self.masters
        }:
            raise ValueError("optimizer must own exactly the FP32 masters")
        self._prepared = False
        self.gradient_state = PPOGradientState(
            self.parameters, self.gradients, self.zero_grad, self.accumulate
        )

    def zero_grad(self):
        """Clear model/master gradients and start a new accumulation window."""
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters:
            parameter.grad = None
        self._prepared = False

    def accumulate(self):
        """Add BF16 model gradients into FP32 masters and clear consumed model buffers."""
        if self._prepared:
            raise RuntimeError("cannot accumulate after clipping was prepared")
        for model, master in zip(self.parameters, self.masters):
            if hasattr(model, "main_grad"):
                raise ValueError("unexpected fused gradient buffer")
            if model.grad is not None:
                value = model.grad.detach().float()
                if master.grad is None:
                    master.grad = value.clone()
                else:
                    master.grad.add_(value)
                model.grad = None

    def gradients(self):
        """Prepare writable master gradients once for clipping before the next step."""
        if self._prepared:
            raise RuntimeError("master gradients may only be materialized once per update")
        self.accumulate()
        self._prepared = True
        return [p.grad for p in self.masters]

    @torch.no_grad()
    def step(self):
        """Update FP32 masters with prepared gradients, then copy weights back to BF16."""
        if not self._prepared:
            raise RuntimeError("materialize and clip FP32 gradients before step")
        self.optimizer.step()
        for model, master in zip(self.parameters, self.masters):
            model.copy_(master)
        self._prepared = False
