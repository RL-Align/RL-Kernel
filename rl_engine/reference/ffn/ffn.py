# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the platform-owned deterministic FFN."""

import importlib
import sys

import torch

_TARGET = (
    "rl_engine.backends.rocm.ffn.ffn"
    if torch.version.hip is not None
    else "rl_engine.backends.cuda.ffn.ffn"
)
sys.modules[__name__] = importlib.import_module(_TARGET)
