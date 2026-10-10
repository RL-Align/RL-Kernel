# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Linear-attention operators (Gated DeltaNet and friends)."""

from rl_engine.reference.linear_attn.causal_conv1d import CausalConv1dUpdateOp
from rl_engine.reference.linear_attn.gated_delta_rule import (
    NULL_BLOCK_ID,
    SOFTPLUS_THRESHOLD,
    GatedDeltaRuleRecurrentStepOp,
)

__all__ = [
    "CausalConv1dUpdateOp",
    "GatedDeltaRuleRecurrentStepOp",
    "NULL_BLOCK_ID",
    "SOFTPLUS_THRESHOLD",
]
