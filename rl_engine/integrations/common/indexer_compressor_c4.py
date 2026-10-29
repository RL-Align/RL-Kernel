# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in Indexer boundary, not an entire framework registration."""

from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor


def bind_indexer_c4_compressor(*, contract, backend=None, expected_implementation=None):
    return IndexerC4Compressor(
        contract, backend=backend, expected_implementation=expected_implementation
    )
