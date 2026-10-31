# SPDX-License-Identifier: Apache-2.0
"""Opt-in raw ICV, not a replacement for fused MQA logits or Top-K.

The framework owns prefill/decode K gathering and de-swizzling of paged FP4.
Only the explicitly declared linear layout is accepted at this boundary.
"""

from rl_engine.integrations.common.indexer_icv import bind_global_icv_keys, bind_icv_queries

__all__ = ["bind_global_icv_keys", "bind_icv_queries"]
