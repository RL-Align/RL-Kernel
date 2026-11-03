# SPDX-License-Identifier: Apache-2.0
"""Lifecycle boundary around upstream state/cache kernels, not full model registration."""

from rl_engine.integrations.common.compressor_state import bind_compressor_state

__all__ = ["bind_compressor_state"]
