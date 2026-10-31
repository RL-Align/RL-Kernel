# SPDX-License-Identifier: Apache-2.0
"""Opt-in native DSV4 sparse VJPs; keep all microbatch forwards in one session."""

from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd
from rl_engine.models.deepseek_v4.joint_backward import JointBackwardPlan
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

__all__ = ["JointAttentionAutograd", "JointBackwardPlan", "canonical_backward_session"]
