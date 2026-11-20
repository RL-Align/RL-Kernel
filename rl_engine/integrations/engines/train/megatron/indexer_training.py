# SPDX-License-Identifier: Apache-2.0
"""Explicit native Indexer/compressor auxiliary-training boundary."""

from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
from rl_engine.models.deepseek_v4.indexer_training import (
    IndexerAuxiliary,
    IndexerAuxMetadata,
    canonical_indexer_auxiliary_loss,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

__all__ = [
    "IndexerAuxContract",
    "IndexerAuxiliary",
    "IndexerAuxMetadata",
    "canonical_indexer_auxiliary_loss",
    "canonical_backward_session",
]
