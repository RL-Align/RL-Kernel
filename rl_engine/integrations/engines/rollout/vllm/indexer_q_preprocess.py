# SPDX-License-Identifier: Apache-2.0
"""vLLM DeepseekV4 reads ``qk_rope_head_dim`` for both the main and Indexer RoPE suffix.

vLLM's fused Indexer Q path uses the ceil E8M0 rule; this contract is #624's even rule.
"""

from rl_engine.contracts.operators.indexer_q_preprocess import IndexerQPreprocessContract
from rl_engine.contracts.operators.partial_rope import PartialRoPEContract


def contracts_from_hf_config(config):
    return (
        PartialRoPEContract(head_dim=config.head_dim, rope_dim=config.qk_rope_head_dim),
        IndexerQPreprocessContract(
            num_heads=config.index_n_heads,
            head_dim=config.index_head_dim,
            rope_dim=config.qk_rope_head_dim,
        ),
    )


__all__ = ["contracts_from_hf_config"]
