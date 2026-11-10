# SPDX-License-Identifier: Apache-2.0
"""Item 15: Indexer Q RoPE -> Hadamard -> MXFP4, with an identity-STE VJP."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.partial_rope import PartialRoPEContract, require

INDEXER_Q_BACKEND = "aiter.indexer_q.rope_hadamard_mxfp4.v1"


@dataclass(frozen=True)
class IndexerQPreprocessContract:
    num_heads: int
    head_dim: int = 128
    rope_dim: int = 64
    precision: str = "mxfp4"
    scale_rule: str = "even"
    graph_capture: bool = False
    backend_id: str = INDEXER_Q_BACKEND

    def __post_init__(self):
        require(
            type(self.num_heads) is int and self.num_heads > 0,
            "num_heads must be a positive integer",
        )
        require(self.head_dim == 128, "the native Hadamard is D=128 only")
        require(
            self.precision == "mxfp4" and self.scale_rule == "even",
            "only #624's MXFP4 even-rounded E8M0 rule is validated",
        )
        require(self.graph_capture is False, "Graph capture is not validated")
        require(self.backend_id == INDEXER_Q_BACKEND, "unknown backend; no silent fallback")
        require(isinstance(self.rope, PartialRoPEContract), "invalid RoPE geometry")

    @property
    def rope(self):
        return PartialRoPEContract(head_dim=self.head_dim, rope_dim=self.rope_dim)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
