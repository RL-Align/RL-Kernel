# SPDX-License-Identifier: Apache-2.0
"""Explicit MoE diagnostic references; never used as production fallbacks."""

from .output_combine import PreparedMoEOutputCombine, prepare_output_combine

__all__ = ["PreparedMoEOutputCombine", "prepare_output_combine"]
