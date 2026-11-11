# SPDX-License-Identifier: Apache-2.0
"""Item 11: GPT-J interleaved RoPE on the trailing ``rope_dim`` of each head."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.attention import AttentionContractError

PARTIAL_ROPE_BACKEND = "aiter.rope_cached.gptj_interleaved_suffix.v1"
PARTIAL_ROPE_HEAD_DIMS = (128, 512)


class PartialRoPEError(AttentionContractError):
    pass


def require(ok, message):
    if not ok:
        raise PartialRoPEError(message)


@dataclass(frozen=True)
class PartialRoPEContract:
    head_dim: int
    rope_dim: int
    graph_capture: bool = False
    backend_id: str = PARTIAL_ROPE_BACKEND

    def __post_init__(self):
        for name in ("head_dim", "rope_dim"):
            require(type(getattr(self, name)) is int, f"{name} must be an integer")
        require(
            self.head_dim in PARTIAL_ROPE_HEAD_DIMS,
            "head_dim 128 (Indexer) or 512 (main branch) only",
        )
        require(self.rope_dim == 64, "only the DSV4 64-wide RoPE suffix is validated")
        require(self.graph_capture is False, "Graph capture is not validated")
        require(self.backend_id == PARTIAL_ROPE_BACKEND, "unknown backend; no silent fallback")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
