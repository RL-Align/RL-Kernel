# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compatibility import for the NVIDIA-owned H3 projection adapter."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "rl_engine.backends.cuda.gemm.h3_projection_fp32"
)
