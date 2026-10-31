# SPDX-License-Identifier: Apache-2.0
"""Cache state is non-differentiable; upstream transport must precede ingestion."""

from rl_engine.integrations.common.compressor_state import bind_compressor_state

__all__ = ["bind_compressor_state"]
