# SPDX-License-Identifier: Apache-2.0
"""Megatron adapter; reuse native transport and RLK canonical backward lifetime."""

from rl_engine.integrations.common.kv_compressor_c4 import bind_c4_compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

__all__ = ["bind_c4_compressor", "canonical_backward_session"]
