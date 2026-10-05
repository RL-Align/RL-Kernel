# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""P2 DSV4 CSA/HCA attention operators owned by T06 (and T06-B).

This package consumes recorded Q/KV/sink/candidate_plan/attention_row tensors.
It does not implement compressor, indexer, or cache state (T03–T05) and does
not register a second public RoPE primitive (T02).
"""

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.contract import (
    ATTENTION_SCALE,
    HEAD_DIM,
    N_Q_HEADS,
    SCHEMA_VERSION_ATTENTION,
    SCHEMA_VERSION_O_PROJ,
)
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.finite import CheckedCUDAGraph
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.p2.state_gate import StateGateVerdict, require_state_gate

__all__ = [
    "ATTENTION_SCALE",
    "HEAD_DIM",
    "N_Q_HEADS",
    "SCHEMA_VERSION_ATTENTION",
    "SCHEMA_VERSION_O_PROJ",
    "CandidatePlan",
    "CheckedCUDAGraph",
    "MqaJointAttentionSinkOp",
    "OProjGroupedOp",
    "P2FailClosedError",
    "P2Status",
    "StateGateVerdict",
    "require_state_gate",
]
