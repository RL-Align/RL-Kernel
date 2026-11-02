# SPDX-License-Identifier: Apache-2.0
"""Typed DSV4 item-13 boundary: two projections, not Indexer scoring or RoPE."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.attention import AttentionContractError

INDEXER_BACKEND_ID = "aiter.indexer.fixed_k.v1"
INDEXER_VJP_ID = "rlk.canonical_rows.aiter.bf16_master.v1"


class IndexerProjectionError(AttentionContractError):
    pass


@dataclass(frozen=True)
class IndexerProjectionContract:
    hidden_size: int = 4096
    q_lora_rank: int = 1024
    num_heads: int = 64
    head_dim: int = 128
    q_precision: str = "bf16"
    q_output: str = "bf16"
    weights_output: str = "fp32"
    max_rows: int = 8192
    tp_size: int = 1
    cp_size: int = 1
    graph_capture: bool = False
    backend_id: str = INDEXER_BACKEND_ID
    backward_id: str = INDEXER_VJP_ID

    def __post_init__(self):
        for name in ("hidden_size", "q_lora_rank", "num_heads", "head_dim", "max_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise IndexerProjectionError(f"{name} must be a positive integer")
        if self.q_precision not in ("bf16", "block_fp8_ue8m0"):
            raise IndexerProjectionError("unknown Q precision; no implicit conversion")
        if self.q_output not in ("bf16", "fp32") or self.weights_output not in ("bf16", "fp32"):
            raise IndexerProjectionError("output must be bf16 or true fp32 accumulation output")
        if self.q_precision == "block_fp8_ue8m0":
            if self.q_lora_rank % 128 or self.q_width % 128:
                raise IndexerProjectionError("block-FP8 Q dimensions must be multiples of 128")
        if type(self.tp_size) is not int or self.tp_size != 1:
            raise IndexerProjectionError("TP>1 projection collectives require separate signoff")
        if type(self.cp_size) is not int or self.cp_size != 1:
            raise IndexerProjectionError("CP>1 ownership requires separate signoff")
        if self.graph_capture is not False:
            raise IndexerProjectionError("Graph capture is not validated")
        if self.backend_id != INDEXER_BACKEND_ID or self.backward_id != INDEXER_VJP_ID:
            raise IndexerProjectionError("backend/VJP identity mismatch; no fallback")

    @property
    def q_width(self):
        return self.num_heads * self.head_dim

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FixedGemmPlan:
    role: str
    k: int
    n: int
    precision: str
    config_items: tuple

    @property
    def config(self):
        # The upstream wrappers may mutate a configuration. Never share that dict.
        return dict(self.config_items)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
