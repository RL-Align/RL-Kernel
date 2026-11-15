# SPDX-License-Identifier: Apache-2.0
"""Item 17: stable Indexer Top-K and the compressed-row read order."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.attention import AttentionContractError

TOPK_BACKEND_ID = "triton.indexer_topk.packed_key.v1"
SELECTION = "score_descending.logical_id_ascending"
READ_ORDER = "compressed_logical_id_descending.then_window_ascending"
MAX_TOPK = 512
MAX_KEYS = 2**31 - 1


class IndexerTopKError(AttentionContractError):
    pass


def require(ok, message):
    if not ok:
        raise IndexerTopKError(message)


@dataclass(frozen=True)
class IndexerTopKContract:
    topk: int = MAX_TOPK
    graph_capture: bool = False
    backend_id: str = TOPK_BACKEND_ID

    def __post_init__(self):
        require(
            type(self.topk) is int
            and 0 < self.topk <= MAX_TOPK
            and self.topk & (self.topk - 1) == 0,
            f"topk must be a power of two in [1, {MAX_TOPK}]",
        )
        require(self.graph_capture is False, "Graph capture is not validated")
        require(self.backend_id == TOPK_BACKEND_ID, "unknown backend; no silent fallback")

    @property
    def fingerprint(self):
        identity = dict(asdict(self), selection=SELECTION, read_order=READ_ORDER)
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
