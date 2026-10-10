# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from importlib import import_module

__all__ = ["BACKEND_ID", "Qwen3FFNOp", "qwen3_ffn", "qwen3_ffn_training"]


def __getattr__(name: str):
    if name == "qwen3_ffn_training":
        return getattr(import_module("rl_engine.backends.rocm.ffn.training"), name)
    if name in {"BACKEND_ID", "Qwen3FFNOp", "qwen3_ffn"}:
        return getattr(import_module("rl_engine.backends.rocm.ffn.ffn"), name)
    raise AttributeError(name)
