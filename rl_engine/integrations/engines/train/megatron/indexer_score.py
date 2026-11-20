# SPDX-License-Identifier: Apache-2.0
"""Megatron CSAIndexer builds its projections with parallel_mode="duplicated", so TP is 1.

Its reference forward multiplies BF16 weights by the head scale only; this
contract takes unscaled FP32 weights and applies both scales itself.
"""

from rl_engine.contracts.operators.indexer_score import IndexerScoreContract


def contract_from_transformer_config(config):
    return IndexerScoreContract(
        num_heads=config.dsa_indexer_n_heads, head_dim=config.dsa_indexer_head_dim
    )


__all__ = ["contract_from_transformer_config"]
