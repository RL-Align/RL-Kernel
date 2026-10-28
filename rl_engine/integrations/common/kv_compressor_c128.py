# SPDX-License-Identifier: Apache-2.0
"""Opt-in C128 operator boundary; does not pretend to register an entire model."""

from rl_engine.models.deepseek_v4.kv_compressor_c128 import C128Compressor


def bind_c128_compressor(*, contract, backend=None, expected_implementation=None):
    return C128Compressor(
        contract, backend=backend, expected_implementation=expected_implementation
    )
