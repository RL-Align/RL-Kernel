# SPDX-License-Identifier: Apache-2.0
"""Items 10/14: global Indexer scales, then ReLU, weights and head sum."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.attention import AttentionContractError

SCORE_BACKEND_ID = "aiter.indexer_score.rowdot.v1"
ROW_DOT_MAX_WIDTH = 4096


class IndexerScoreError(AttentionContractError):
    pass


def require(ok, message):
    if not ok:
        raise IndexerScoreError(message)


@dataclass(frozen=True)
class IndexerScoreContract:
    num_heads: int
    head_dim: int
    tp_size: int = 1
    max_backward_keys: int = ROW_DOT_MAX_WIDTH
    graph_capture: bool = False
    backend_id: str = SCORE_BACKEND_ID

    def __post_init__(self):
        for field in ("num_heads", "head_dim", "tp_size", "max_backward_keys"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise IndexerScoreError(f"{field} must be a positive integer")
        if self.num_heads % self.tp_size:
            raise IndexerScoreError("global Indexer heads must divide evenly across TP")
        if self.max_backward_keys > ROW_DOT_MAX_WIDTH:
            raise IndexerScoreError(f"backward key rows are bounded by {ROW_DOT_MAX_WIDTH}")
        if self.graph_capture is not False:
            raise IndexerScoreError("Graph capture is not validated")
        if self.backend_id != SCORE_BACKEND_ID:
            raise IndexerScoreError("unknown backend; no silent fallback")

    @property
    def local_heads(self):
        return self.num_heads // self.tp_size

    @property
    def softmax_scale(self):
        return self.head_dim**-0.5

    @property
    def head_scale(self):
        # Always the global head count: a TP shard must not use local_heads here.
        return self.num_heads**-0.5

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
