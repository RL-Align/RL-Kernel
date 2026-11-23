# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""A BF16 DeepSpeed step with all-zero gradients still applies AdamW, as the reference does."""

from __future__ import annotations

import pytest
import torch

from rl_engine.integrations.engines.train.deepspeed.ppo import _allow_zero_norm_step


class Optimizer:
    """The ``BF16_Optimizer`` attributes the zero-norm step reads, with one FP32 partition."""

    def __init__(self, grad: torch.Tensor):
        self.master = torch.nn.Parameter(torch.ones(4))
        self.grad = grad
        self.fp32_groups_flat_partition = [self.master]
        self.fp32_groups_gradient_flat_partition = [self.grad]
        self.optimizer = torch.optim.AdamW([self.master], lr=0.1, weight_decay=0.01)
        self.clip_grad, self.has_moe_layers, self.graph_harvesting = 0.0, False, False
        self.mpu, self.norm_type = None, 2
        self.calls: list[str] = []

    def get_grads_for_norm(self):
        return [self.grad], {}

    def step(self, closure=None):
        self.calls.append("deepspeed step")

    def _lazy_init_hp_params_optimizer_state(self):
        self.calls.append("link state")

    def update_lp_params(self):
        self.calls.append("copy to bf16")

    def clear_hp_grads(self):
        self.calls.append("clear grads")


def norm(input_tensors, mpu, norm_type, use_graph):
    return float(sum(t.double().square().sum() for t in input_tensors) ** 0.5)


def test_nonzero_norm_uses_deepspeed_step():
    optimizer = Optimizer(torch.full((4,), 0.5))
    _allow_zero_norm_step(optimizer, norm_fn=norm)
    optimizer.step()
    assert optimizer.calls == ["deepspeed step"]


def test_zero_norm_applies_adamw_and_finishes_the_step():
    optimizer = Optimizer(torch.zeros(4))
    _allow_zero_norm_step(optimizer, norm_fn=norm)
    optimizer.step()
    assert optimizer.calls == ["link state", "copy to bf16", "clear grads"]
    # Weight decay alone moves the parameters: lr * weight_decay * 1.0.
    torch.testing.assert_close(optimizer.master.detach(), torch.full((4,), 1 - 0.1 * 0.01))
    assert optimizer.master.grad is optimizer.grad


@pytest.mark.parametrize("attribute", ["clip_grad", "has_moe_layers"])
def test_zero_norm_with_clipping_or_moe_uses_deepspeed_step(attribute):
    optimizer = Optimizer(torch.zeros(4))
    setattr(optimizer, attribute, 1.0 if attribute == "clip_grad" else True)
    _allow_zero_norm_step(optimizer, norm_fn=norm)
    optimizer.step()
    assert optimizer.calls == ["deepspeed step"]
