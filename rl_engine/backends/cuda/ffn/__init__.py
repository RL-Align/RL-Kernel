# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from importlib import import_module

__all__ = ["BACKEND_ID", "Qwen3FFNOp", "qwen3_ffn"]


def __getattr__(name: str):
    if name in __all__:
        return getattr(import_module("rl_engine.backends.cuda.ffn.ffn"), name)
    raise AttributeError(name)
