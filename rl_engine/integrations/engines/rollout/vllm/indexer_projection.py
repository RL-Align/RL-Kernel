# SPDX-License-Identifier: Apache-2.0
"""Explicit item-13 hook before native Indexer RoPE, scales and scoring."""

from rl_engine.integrations.common.indexer_projection import bind_framework_indexer


def bind_indexer_projection(
    indexer,
    *,
    contract,
    layer_id,
    backend=None,
    packed_q=None,
    expected_implementation=None,
):
    return bind_framework_indexer(
        indexer,
        contract=contract,
        layer_id=layer_id,
        backend=backend,
        packed_q=packed_q,
        expected_implementation=expected_implementation,
    )
