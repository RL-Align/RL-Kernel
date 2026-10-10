# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from rl_engine.backends.ascend.activation.silu import SiLUAscendOp
from rl_engine.backends.ascend.activation.swiglu import SwiGLUAscendOp

__all__ = ["SiLUAscendOp", "SwiGLUAscendOp"]
