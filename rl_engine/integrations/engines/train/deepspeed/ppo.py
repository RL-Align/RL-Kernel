# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSpeed PPO engine steps for Megatron TP models.

Validated configuration: DeepSpeed FP32, ZeRO-0, DP=1, accumulation=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import PPOConfig, PPOTrainingWorker
from rl_engine.integrations.common.tp_scalar import TPModelInputs
from rl_engine.integrations.engines.train.megatron.ppo import megatron_parameter_layouts

if TYPE_CHECKING:
    from deepspeed import DeepSpeedEngine


@dataclass(frozen=True)
class DeepSpeedTPGroups:
    """DeepSpeed MPU contract for TP-only models with explicit singleton DP.

    Megatron Core's parallel_state lacks DeepSpeed's legacy
    get_model_parallel_world_size API. This adapter maps the actual groups,
    without patching either framework's module globals.
    """

    tp: dist.ProcessGroup
    dp: dist.ProcessGroup

    def __post_init__(self) -> None:
        if self.tp is None or self.dp is None:
            raise ValueError("TP and DP groups must be explicit")
        if dist.get_rank(self.tp) < 0 or dist.get_rank(self.dp) < 0:
            raise ValueError("this rank must belong to TP and DP groups")
        if dist.get_world_size(self.dp) != 1:
            raise ValueError("DeepSpeed PPO currently requires singleton DP")

    def get_model_parallel_group(self):
        return self.tp

    def get_model_parallel_rank(self):
        return dist.get_rank(self.tp)

    def get_model_parallel_world_size(self):
        return dist.get_world_size(self.tp)

    get_tensor_model_parallel_group = get_model_parallel_group
    get_tensor_model_parallel_rank = get_model_parallel_rank
    get_tensor_model_parallel_world_size = get_model_parallel_world_size

    def get_tensor_model_parallel_src_rank(self):
        return dist.get_process_group_ranks(self.tp)[0]

    def get_data_parallel_group(self):
        return self.dp

    def get_data_parallel_rank(self):
        return dist.get_rank(self.dp)

    def get_data_parallel_world_size(self):
        return dist.get_world_size(self.dp)

    def get_data_parallel_group_ranks(self):
        return dist.get_process_group_ranks(self.dp)


def deepspeed_ppo_worker(
    engine: DeepSpeedEngine,
    *,
    role: Literal["actor", "critic"],
    forward: Callable[[TPModelInputs], torch.Tensor],
    group: dist.ProcessGroup,
    config: PPOConfig,
) -> PPOTrainingWorker:
    """Use engine.backward/step without bypassing DeepSpeed's lifecycle.

    This boundary deliberately rejects optimizer/gradient partitioning and loss
    scaling: PPO's TP norm clipping currently operates on ordinary FP32 .grad.
    DeepSpeed must receive an MPU describing the actual TP and singleton DP groups.
    """
    if engine.zero_optimization_stage() != 0:
        raise ValueError("DeepSpeed PPO currently requires ZeRO stage 0")
    if (
        engine.fp16_enabled()
        or engine.bfloat16_enabled()
        or engine.amp_enabled()
        or engine.torch_autocast_enabled()
    ):
        raise ValueError("DeepSpeed PPO currently requires FP32 without loss scaling")
    if any(p.dtype != torch.float32 for p in engine.parameters() if p.requires_grad):
        raise ValueError("DeepSpeed PPO requires FP32 trainable parameters")
    if engine.gradient_accumulation_steps() != 1 or engine.dp_world_size != 1:
        raise ValueError("DeepSpeed PPO currently requires accumulation=1 and DP=1")
    if engine.gradient_clipping() != 0:
        raise ValueError("disable DeepSpeed gradient_clipping; PPO performs TP-aware clipping")
    if engine.mpu is None or engine.mpu.get_model_parallel_group() is not group:
        raise ValueError("DeepSpeed MPU and PPO must use the same TP group")
    return PPOTrainingWorker(
        role=role,
        model=engine,
        forward=forward,
        optimizer=engine.optimizer,
        parameter_layouts=megatron_parameter_layouts(engine),
        group=group,
        config=config,
        backward=engine.backward,
        step=engine.step,
    )
