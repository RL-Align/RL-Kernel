# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Recorded-input P2 attention block: T06 attention then T06-B o-proj.

This is not a live transformer layer. Q/KV/sink/weights are recorded fixtures.
Compressor, indexer, and cache (T03–T05) are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import (
    AttentionResult,
    MqaJointAttentionSinkOp,
)
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.contract import ExecutionMode, HIDDEN_SIZE, N_Q_HEADS, HEAD_DIM
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp, OProjResult
from rl_engine.kernels.p2.state_gate import StateGateVerdict, require_state_gate


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
    """State-first eager recorded block; execution_mode is artifact metadata.

    Real mode comparisons, including capture, live in verify_recorded_four_modes.
    """

    require_state_gate(state_gate)
    attn = MqaJointAttentionSinkOp(backend=attn_backend).forward_fp32(
        q, k, v, sink, plan, compare=True, state_gate=state_gate, debug=True
    )
    if attn.o.shape[-2:] != (N_Q_HEADS, HEAD_DIM):
        raise RuntimeError(
            f"attention row must be [T,{N_Q_HEADS},{HEAD_DIM}], got {tuple(attn.o.shape)}"
        )
    o_proj = OProjGroupedOp(backend=o_proj_backend).forward_fp32(attn.o, w_a, w_b, cos, sin)
    if o_proj.y.shape[-1] != HIDDEN_SIZE:
        raise RuntimeError(f"block output must be [T,{HIDDEN_SIZE}], got {tuple(o_proj.y.shape)}")
    return RecordedBlockResult(
        y=o_proj.y, attention=attn, o_proj=o_proj, execution_mode=execution_mode
    )
