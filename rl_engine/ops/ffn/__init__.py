# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Platform-selected deterministic FFN operators."""

from rl_engine.ops.ffn.qwen3 import BACKEND_ID, Qwen3FFNOp, qwen3_ffn

__all__ = ["BACKEND_ID", "Qwen3FFNOp", "qwen3_ffn"]
