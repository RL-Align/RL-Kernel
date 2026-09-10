# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from rl_engine.kernels.ops.ascend.attention.deterministic_attn import DeterministicAttentionAscendOp
from rl_engine.kernels.ops.ascend.attention.prefix_shared_attn import (
    PrefixSharedAttentionAscendOp,
)

__all__ = ["DeterministicAttentionAscendOp", "PrefixSharedAttentionAscendOp"]
