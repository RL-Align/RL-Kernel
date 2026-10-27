# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek V4 gate adapter, shared by training, rollout, and recomputation."""

import torch

from rl_engine.ops.moe.router_gate import router_gate_projection


class DeepSeekV4Gate(torch.nn.Module):
    """Attach the loaded, unquantized gate Parameter without copying or recasting it.

    This is a local model adapter, not a complete vLLM/Megatron integration.
    Framework synchronization owns model_version and the supplied Parameter.
    """

    def __init__(self, weight: torch.nn.Parameter, *, model_version: str):
        super().__init__()
        if not isinstance(weight, torch.nn.Parameter):
            raise TypeError("DeepSeekV4Gate requires the loaded gate Parameter")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("DeepSeekV4Gate requires a synchronized model_version")
        self.weight = weight
        self.model_version = model_version
        self.last_execution = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        result = router_gate_projection(hidden, self.weight, model_version=self.model_version)
        self.last_execution = result.execution
        return result.logits
