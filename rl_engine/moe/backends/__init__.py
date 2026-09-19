# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5 kernel backends. Each sub-issue registers its providers here."""

from rl_engine.moe.backends.mxfp8_act_quant import (
    CudaMXFP8ActQuantProvider,
    TritonMXFP8ActQuantProvider,
)

__all__ = ["CudaMXFP8ActQuantProvider", "TritonMXFP8ActQuantProvider"]
