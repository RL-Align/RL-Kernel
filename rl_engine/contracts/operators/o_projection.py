# SPDX-License-Identifier: Apache-2.0
"""Item 23: inverse RoPE -> eight-head groups -> wo_a -> concat -> wo_b."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.indexer_projection import IndexerProjectionError

O_PROJECTION_BACKEND = "aiter.o_projection.fixed_grouped.v1"
O_PROJECTION_VJP = "rlk.canonical_rows.aiter.grouped_bf16_master.v1"


class OProjectionError(IndexerProjectionError):
    pass


def require(value, message):
    if not value:
        raise OProjectionError(message)


@dataclass(frozen=True)
class OProjectionContract:
    # Geometry is required, rather than guessing a model variant's defaults.
    num_heads: int
    head_dim: int
    rope_dim: int
    o_lora_rank: int
    hidden_size: int
    heads_per_group: int = 8
    tp_size: int = 1
    tp_rank: int = 0
    a_precision: str = "bf16"
    b_precision: str = "bf16"
    max_rows: int = 8192
    graph_capture: bool = False
    backend_id: str = O_PROJECTION_BACKEND
    backward_id: str = O_PROJECTION_VJP

    def __post_init__(self):
        for name in (
            "num_heads",
            "head_dim",
            "rope_dim",
            "o_lora_rank",
            "hidden_size",
            "heads_per_group",
            "tp_size",
            "max_rows",
        ):
            require(
                type(getattr(self, name)) is int and getattr(self, name) > 0,
                f"{name} must be a positive integer",
            )
        require(type(self.tp_rank) is int and 0 <= self.tp_rank < self.tp_size, "invalid TP rank")
        require(self.heads_per_group == 8, "the checkpoint groups exactly eight heads")
        require(
            self.num_heads % (8 * self.tp_size) == 0, "TP must partition complete eight-head groups"
        )
        require(
            self.rope_dim <= self.head_dim and self.rope_dim % 2 == 0,
            "RoPE is an even-sized interleaved suffix",
        )
        require(self.head_dim % 2 == 0, "head dimension must be even")
        for name, n, k in (
            ("a_precision", self.o_lora_rank, self.group_width),
            ("b_precision", self.hidden_size, self.local_groups * self.o_lora_rank),
        ):
            require(
                getattr(self, name) in ("bf16", "block_fp8_ue8m0"),
                f"unsupported {name}; no implicit dequantization",
            )
            if getattr(self, name) == "block_fp8_ue8m0":
                require(n % 128 == 0 and k % 128 == 0, f"{name} requires 128x128 checkpoint blocks")
        require(self.graph_capture is False, "Graph capture is not validated")
        require(
            self.backend_id == O_PROJECTION_BACKEND and self.backward_id == O_PROJECTION_VJP,
            "backend/VJP identity mismatch",
        )

    @property
    def local_heads(self):
        return self.num_heads // self.tp_size

    @property
    def local_groups(self):
        return self.local_heads // self.heads_per_group

    @property
    def group_width(self):
        return self.heads_per_group * self.head_dim

    @property
    def head_ids(self):
        return tuple(range(self.tp_rank * self.local_heads, (self.tp_rank + 1) * self.local_heads))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def family_fingerprint(self):
        data = asdict(self)
        data.pop("tp_rank")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
