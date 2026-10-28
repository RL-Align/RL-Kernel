# SPDX-License-Identifier: Apache-2.0
"""Item 18: complete C128 groups, separate from projection and cache lifecycle."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from rl_engine.contracts.operators.attention import AttentionContractError

C128_BACKEND_ID = "aiter.hca.c128.gfx942.fp32_pool.fixed8x64.norm_wave1.v1"
MLITE_BACKWARD_SHA256 = "6a1e78ca89b83f56391b5e5ac409e765338f1e3a625251d0d4c877c1e80b4a68"
AITer_HCA_SHA256 = "183136a19a0c76fe19fd99b084b9dca9631b5e1fcee0ff2a5c440c271eea07d8"


class C128Error(AttentionContractError):
    pass


def require(ok, message):
    if not ok:
        raise C128Error(message)


def source_hash(path):
    require(isinstance(path, (str, Path)) and bool(path), "provider source is unavailable")
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise C128Error("provider source is unreadable") from exc


@dataclass(frozen=True)
class C128Contract:
    head_dim: int = 512
    ratio: int = 128
    rope_dim: int = 64
    rms_eps: float = 1e-6
    max_input_rows: int = 524288
    max_groups: int = 4096
    max_block_table_bytes: int = 16 * 1024 * 1024
    max_backward_groups: int = 128
    backend_id: str = C128_BACKEND_ID
    output_dtype: str = "bf16"
    graph_capture: bool = False

    def __post_init__(self):
        for name in (
            "head_dim",
            "ratio",
            "rope_dim",
            "max_input_rows",
            "max_groups",
            "max_block_table_bytes",
            "max_backward_groups",
        ):
            require(
                type(getattr(self, name)) is int and getattr(self, name) > 0,
                f"{name} must be a positive integer",
            )
        require(
            (self.head_dim, self.ratio, self.rope_dim) == (512, 128, 64),
            "validated C128 main-KV profile is D=512, ratio=128, RoPE=64",
        )
        require(
            type(self.rms_eps) in (int, float)
            and math.isfinite(self.rms_eps)
            and 1e-12 <= self.rms_eps <= 1,
            "invalid RMS epsilon",
        )
        require(
            self.max_input_rows < 2**31 and self.max_groups < 2**24,
            "int32 provider indexing capacity exceeded",
        )
        require(self.backend_id == C128_BACKEND_ID, "unknown C128 backend; no fallback")
        require(self.output_dtype == "bf16", "C128 emits BF16 independent rows")
        require(self.graph_capture is False, "Graph capture belongs to the runtime integration")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def arithmetic(self):
        return {
            "pool": "FP32 online softmax;8 waves;D-slice64;no N-based dispatch",
            "norm_rope": "FP32 RMSNorm;BF16 cos/sin;GPT-J tail;one wave;one BF16 output cast",
            "position": "global compressed group ID * 128",
            "backward": "mLite FP32 compact graph;one group per VJP;canonical FP32 left-fold",
        }
