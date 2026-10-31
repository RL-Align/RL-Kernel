# SPDX-License-Identifier: Apache-2.0
"""One arithmetic profile for joint compressed/window/sink attention."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from rl_engine.contracts.operators.attention import AttentionContractError

JOINT_PROFILE = "aiter.326c.single_split.h32.k32.bf16_p.fp32_lse.v1"


class JointAttentionError(AttentionContractError):
    pass


def require(condition, message):
    if not condition:
        raise JointAttentionError(message)


@dataclass(frozen=True)
class JointAttentionContract:
    num_heads: int
    head_dim: int
    max_queries: int
    max_kv: int
    compressed_topk: int
    window_size: int
    compression_ratio: int = 4
    scale: float | None = None
    graph: bool = True
    max_workspace_bytes: int = 1 << 30
    profile: str = JOINT_PROFILE

    def __post_init__(self):
        for name in ("num_heads", "head_dim", "max_queries", "max_kv", "max_workspace_bytes"):
            require(
                type(getattr(self, name)) is int and getattr(self, name) > 0,
                f"{name} must be a positive integer",
            )
        for name in ("compressed_topk", "window_size"):
            require(
                type(getattr(self, name)) is int and getattr(self, name) >= 0,
                f"{name} must be nonnegative",
            )
        require(self.head_dim in (128, 512), "validated head dimensions are 128 and 512")
        require(self.num_heads <= 128, "unvalidated head count")
        require(self.compression_ratio in (4, 128), "compression ratio must be C4 or C128")
        require(
            self.selected_capacity <= 640, "longer fixed-loop profiles require separate signoff"
        )
        require(self.max_queries < 2**31 and self.max_kv < 2**31, "index ABI exceeds int32")
        require(type(self.graph) is bool, "graph must be bool")
        require(
            self.scale is None
            or (
                isinstance(self.scale, (int, float))
                and not isinstance(self.scale, bool)
                and math.isfinite(self.scale)
                and self.scale > 0
            ),
            "softmax scale must be finite and positive",
        )
        require(self.profile == JOINT_PROFILE, "arithmetic profile mismatch")
        require(self.workspace_bytes <= self.max_workspace_bytes, "workspace budget exceeded")

    @property
    def selected_capacity(self):
        # A sink-only query still has one invalid slot; no zero-sized pointer ABI.
        return max(1, self.compressed_topk + self.window_size)

    @property
    def softmax_scale(self):
        return self.head_dim**-0.5 if self.scale is None else float(self.scale)

    @property
    def workspace_bytes(self):
        # q, o, KV, sink, indices, active-row mask, LSE. Does not include
        # compilation, graph-pool allocations or returned forward-record clones.
        return (
            4 * self.max_queries * self.num_heads * self.head_dim
            + 2 * self.max_kv * self.head_dim
            + 4 * self.num_heads
            + 4 * self.max_queries * self.selected_capacity
            + self.max_queries
            + 4 * self.max_queries * self.num_heads
        )

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
