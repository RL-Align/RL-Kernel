# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Attention and grouped output projection over recorded inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import (
    AttentionResult,
    MqaJointAttentionSinkOp,
)
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.contract import (
    ExecutionMode,
    HIDDEN_SIZE,
    N_Q_HEADS,
    HEAD_DIM,
)
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp, OProjResult
from rl_engine.kernels.dsv4.attention.state_gate import StateGateVerdict, require_state_gate


@dataclass
class RecordedBlockResult:
    y: Tensor
    attention: AttentionResult
    o_proj: OProjResult
    execution_mode: ExecutionMode


def recorded_attention_block(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    w_a: Tensor,
    w_b: Tensor,
    cos: Tensor,
    sin: Tensor,
    state_gate: StateGateVerdict,
    *,
    attn_backend: str = "oracle",
    o_proj_backend: str = "oracle",
    execution_mode: ExecutionMode = ExecutionMode.TRAINING,
) -> RecordedBlockResult:
    """Run the eager block; execution_mode labels the returned artifact."""

    require_state_gate(state_gate)
    attn = MqaJointAttentionSinkOp(backend=attn_backend).forward_fp32(
        q, k, v, sink, plan, compare=True, state_gate=state_gate, debug=True
    )
    if attn.o.shape[-2:] != (N_Q_HEADS, HEAD_DIM):
        raise RuntimeError(
            f"attention row must be [T,{N_Q_HEADS},{HEAD_DIM}], got {tuple(attn.o.shape)}"
        )
    o_proj_input = attn.o
    if o_proj_backend == "det_gemm" or (o_proj_backend == "auto" and attn.o.is_cuda):
        o_proj_input = o_proj_input.to(torch.bfloat16)
    o_proj = OProjGroupedOp(backend=o_proj_backend).forward_fp32(o_proj_input, w_a, w_b, cos, sin)
    if o_proj.y.shape[-1] != HIDDEN_SIZE:
        raise RuntimeError(f"block output must be [T,{HIDDEN_SIZE}], got {tuple(o_proj.y.shape)}")
    return RecordedBlockResult(
        y=o_proj.y, attention=attn, o_proj=o_proj, execution_mode=execution_mode
    )
