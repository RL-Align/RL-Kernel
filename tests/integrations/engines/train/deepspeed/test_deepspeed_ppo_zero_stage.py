# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""deepspeed_ppo_worker accepts ZeRO-0 for FP32 and ZeRO-1 for BF16, nothing else."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.integrations.engines.train.deepspeed.ppo import deepspeed_ppo_worker

STAGE_ERROR = "requires ZeRO stage 0 for FP32 and stage 1 for BF16"
# Raised by the check after the stage gate: reaching it means the stage was accepted.
NEXT_CHECK = "disable DeepSpeed gradient_clipping"


class Engine:
    """The DeepSpeedEngine queries deepspeed_ppo_worker makes before it reads the optimizer."""

    def __init__(self, *, bf16: bool, stage: int):
        self.bf16 = bf16
        self.stage = stage
        dtype = torch.bfloat16 if bf16 else torch.float32
        self.weight = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
        self.dp_world_size = 1
        self.mpu = None

    def zero_optimization_stage(self) -> int:
        return self.stage

    def bfloat16_enabled(self) -> bool:
        return self.bf16

    def fp16_enabled(self) -> bool:
        return False

    def amp_enabled(self) -> bool:
        return False

    def torch_autocast_enabled(self) -> bool:
        return False

    def parameters(self):
        return iter([self.weight])

    def gradient_accumulation_steps(self) -> int:
        return 1

    def gradient_clipping(self) -> float:
        return 1.0


def _worker(engine: Any) -> None:
    group: Any = None
    deepspeed_ppo_worker(engine, role="actor", forward=lambda x: x, group=group, config=group)


@pytest.mark.parametrize(("bf16", "stage"), [(False, 0), (True, 1)])
def test_supported_stage_passes_the_stage_gate(bf16, stage):
    with pytest.raises(ValueError, match=NEXT_CHECK):
        _worker(Engine(bf16=bf16, stage=stage))


@pytest.mark.parametrize(
    ("bf16", "stage"), [(False, 1), (False, 2), (False, 3), (True, 0), (True, 2), (True, 3)]
)
def test_other_stages_are_rejected(bf16, stage):
    with pytest.raises(ValueError, match=STAGE_ERROR):
        _worker(Engine(bf16=bf16, stage=stage))
