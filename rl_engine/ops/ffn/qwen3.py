# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Import-time platform binding for the deterministic Qwen3 FFN."""

import importlib
import sys

import torch

_TARGET = (
    "rl_engine.backends.rocm.ffn.ffn"
    if torch.version.hip is not None
    else "rl_engine.backends.cuda.ffn.ffn"
)
_IMPL = importlib.import_module(_TARGET)

# Keep these names explicit for static analyzers and wildcard import users.  The
# module alias below preserves the historical behavior for private test hooks.
BACKEND_ID = _IMPL.BACKEND_ID
Qwen3FFNOp = _IMPL.Qwen3FFNOp
qwen3_ffn = _IMPL.qwen3_ffn

__all__ = ["BACKEND_ID", "Qwen3FFNOp", "qwen3_ffn"]

sys.modules[__name__] = _IMPL
