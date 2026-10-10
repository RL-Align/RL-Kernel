# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the ROCm-owned Triton FFN."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("rl_engine.backends.rocm.ffn.triton")
