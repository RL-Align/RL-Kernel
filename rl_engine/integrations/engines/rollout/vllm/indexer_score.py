# SPDX-License-Identifier: Apache-2.0
"""vLLM DeepseekV4Indexer replicates its heads (ReplicatedLinear), so TP is 1 here.

vLLM's fused_indexer_q_rope_quant folds both scales into the head weights;
pass the unscaled FP32 weights and raw ICV instead.
"""

from rl_engine.contracts.operators.indexer_score import IndexerScoreContract


def contract_from_hf_config(config):
    return IndexerScoreContract(num_heads=config.index_n_heads, head_dim=config.index_head_dim)


__all__ = ["contract_from_hf_config"]
