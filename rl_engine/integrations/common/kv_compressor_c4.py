# SPDX-License-Identifier: Apache-2.0
"""Opt-in C4 operator boundary; item 21 owns state/cache lifetime."""

from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor


def bind_c4_compressor(*, contract, backend=None, expected_implementation=None):
    return C4Compressor(contract, backend=backend, expected_implementation=expected_implementation)
