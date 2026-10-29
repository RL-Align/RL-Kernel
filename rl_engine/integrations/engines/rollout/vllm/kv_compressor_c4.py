# SPDX-License-Identifier: Apache-2.0
"""vLLM rollout adapter for complete C4 groups; no live cache commit."""

from rl_engine.integrations.common.kv_compressor_c4 import bind_c4_compressor

__all__ = ["bind_c4_compressor"]
