# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""DSv4 attention and grouped output projection for recorded inputs."""

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.contract import (
    ATTENTION_SCALE,
    HEAD_DIM,
    N_Q_HEADS,
    SCHEMA_VERSION_ATTENTION,
    SCHEMA_VERSION_O_PROJ,
)
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.finite import CheckedCUDAGraph
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.dsv4.attention.oracle import (
    AttentionBackwardTensors,
    AttentionForwardTensors,
    mqa_joint_attention_sink_bwd,
    mqa_joint_attention_sink_fwd,
)
from rl_engine.kernels.dsv4.attention.state_gate import StateGateVerdict, require_state_gate

__all__ = [
    "ATTENTION_SCALE",
    "HEAD_DIM",
    "N_Q_HEADS",
    "SCHEMA_VERSION_ATTENTION",
    "SCHEMA_VERSION_O_PROJ",
    "AttentionBackwardTensors",
    "AttentionForwardTensors",
    "CandidatePlan",
    "CheckedCUDAGraph",
    "MqaJointAttentionSinkOp",
    "OProjGroupedOp",
    "DSv4FailClosedError",
    "DSv4Status",
    "StateGateVerdict",
    "mqa_joint_attention_sink_bwd",
    "mqa_joint_attention_sink_fwd",
    "require_state_gate",
]
