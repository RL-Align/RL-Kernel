# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSpeed PPO engine steps for Megatron TP models.

Supported configurations: FP32 or BF16 with FP32 gradients/masters, ZeRO-0, DP=1,
accumulation=1. Hardware qualification is recorded separately in experiment reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import PPOConfig, PPOGradientState, PPOTrainingWorker
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

    This boundary rejects optimizer partitioning and loss scaling. BF16 clipping
    operates on DeepSpeed's FP32 accumulation views, not low-precision .grad.
    DeepSpeed must receive an MPU describing the actual TP and singleton DP groups.
    """
    if engine.zero_optimization_stage() != 0:
        raise ValueError("DeepSpeed PPO currently requires ZeRO stage 0")
    if engine.fp16_enabled() or engine.amp_enabled() or engine.torch_autocast_enabled():
        raise ValueError("DeepSpeed PPO supports FP32/BF16 without FP16 loss scaling")
    expected_dtype = torch.bfloat16 if engine.bfloat16_enabled() else torch.float32
    if any(p.dtype != expected_dtype for p in engine.parameters() if p.requires_grad):
        raise ValueError("DeepSpeed PPO trainable parameter dtype disagrees with precision mode")
    if engine.gradient_accumulation_steps() != 1 or engine.dp_world_size != 1:
        raise ValueError("DeepSpeed PPO currently requires accumulation=1 and DP=1")
    if engine.gradient_clipping() != 0:
        raise ValueError("disable DeepSpeed gradient_clipping; PPO performs TP-aware clipping")
    if engine.mpu is None or engine.mpu.get_model_parallel_group() is not group:
        raise ValueError("DeepSpeed MPU and PPO must use the same TP group")
    gradient_state = _bf16_gradient_state(engine) if engine.bfloat16_enabled() else None
    return PPOTrainingWorker(
        role=role,
        model=engine,
        forward=forward,
        optimizer=engine.optimizer,
        parameter_layouts=megatron_parameter_layouts(engine),
        group=group,
        config=config,
        gradient_state=gradient_state,
        backward=engine.backward,
        step=engine.step,
    )


def _bf16_gradient_state(engine) -> PPOGradientState:
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

    optimizer = engine.optimizer
    if not isinstance(optimizer, BF16_Optimizer) or optimizer.grad_acc_dtype != torch.float32:
        raise ValueError("BF16 PPO requires DeepSpeed BF16_Optimizer with FP32 gradient buffers")
    parameters = tuple(p for group in optimizer.bf16_groups for p in group)
    if {id(p) for p in parameters} != {id(p) for p in engine.parameters() if p.requires_grad}:
        raise ValueError("DeepSpeed BF16 groups do not own the trainable model parameters")
    masters = [p for group in optimizer.optimizer.param_groups for p in group["params"]]
    if len(masters) != len(optimizer.fp32_groups_flat_partition) or any(
        actual is not expected or actual.dtype != torch.float32
        for actual, expected in zip(masters, optimizer.fp32_groups_flat_partition)
    ):
        raise ValueError("DeepSpeed optimizer does not own the expected FP32 masters")

    def gradients():
        if not (
            len(optimizer.bf16_groups)
            == len(optimizer.fp32_groups_gradients)
            == len(optimizer.fp32_groups_has_gradients)
        ):
            raise ValueError("DeepSpeed BF16 gradient group mapping changed")
        buffers = []
        for group, grads, flags in zip(
            optimizer.bf16_groups,
            optimizer.fp32_groups_gradients,
            optimizer.fp32_groups_has_gradients,
        ):
            if not (len(group) == len(grads) == len(flags)):
                raise ValueError("DeepSpeed BF16 gradient mapping changed")
            for parameter, grad, present in zip(group, grads, flags):
                if grad.dtype != torch.float32 or grad.numel() != parameter.numel():
                    raise ValueError("invalid DeepSpeed FP32 gradient view")
                buffers.append(grad.view_as(parameter) if present else None)
        return buffers

    return PPOGradientState(parameters, gradients, lambda: optimizer.zero_grad(set_to_none=True))
