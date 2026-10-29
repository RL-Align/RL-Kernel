# SPDX-License-Identifier: Apache-2.0
from rl_engine.models.deepseek_v4.indexer_state import IndexerStateCache


def bind_indexer_state(contract, *, expected_arithmetic, **kwargs):
    return IndexerStateCache(contract, expected_arithmetic=expected_arithmetic, **kwargs)
