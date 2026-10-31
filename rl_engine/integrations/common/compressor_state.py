# SPDX-License-Identifier: Apache-2.0
"""Explicit arithmetic handshake; no implicit C128 HCA-to-legacy substitution."""

from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache


def bind_compressor_state(contract, *, expected_arithmetic, **kwargs):
    return CompressorStateCache(contract, expected_arithmetic=expected_arithmetic, **kwargs)
