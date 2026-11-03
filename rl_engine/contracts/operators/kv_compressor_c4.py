# SPDX-License-Identifier: Apache-2.0
"""Item 19: overlap-window C4 main-KV compressor contract."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar

from rl_engine.contracts.operators.attention import AttentionContractError

C4_BACKEND_ID = "aiter.csa.c4.gfx942.bf16.overlap.fixed1.v1"
AITer_C4_SHA256 = "53cdd2801b6b2f5866b02d18bff1b2d56a19289a8048cdc14fe8bb687e572a14"
AITer_C4_LEGACY_SHA256 = "0c2019ca3a4a0d6dc7d5fdd20ebcce658ff5f504fdc880dffd61c9c398029451"
MLITE_BACKWARD_SHA256 = "6a1e78ca89b83f56391b5e5ac409e765338f1e3a625251d0d4c877c1e80b4a68"


class C4Error(AttentionContractError):
    pass


def require(ok, message):
    if not ok:
        raise C4Error(message)


def source_hash(path):
    require(isinstance(path, (str, Path)) and bool(path), "provider source is unavailable")
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise C4Error("provider source is unreadable") from exc


@dataclass(frozen=True)
class C4Contract:
    expected_head_dim: ClassVar[int] = 512
    expected_backend_id: ClassVar[str] = C4_BACKEND_ID
    family: ClassVar[str] = "main_kv_c4"
    head_dim: int = 512
    ratio: int = 4
    overlap: int = 8
    rope_dim: int = 64
    rms_eps: float = 1e-6
    max_input_rows: int = 524288
    max_groups: int = 8192
    max_block_table_bytes: int = 16 * 1024 * 1024
    max_backward_groups: int = 128
    backend_id: str = C4_BACKEND_ID
    output_dtype: str = "bf16"
    graph_capture: bool = False

    def __post_init__(self):
        for name in (
            "head_dim",
            "ratio",
            "overlap",
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
            (self.head_dim, self.ratio, self.overlap, self.rope_dim)
            == (self.expected_head_dim, 4, 8, 64),
            f"validated {self.family} profile is D={self.expected_head_dim}, "
            "ratio=4, overlap=8, RoPE=64",
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
        require(self.backend_id == self.expected_backend_id, "unknown C4 backend; no fallback")
        require(self.output_dtype == "bf16", "C4 emits BF16 independent rows")
        require(self.graph_capture is False, "Graph capture belongs to runtime integration")

    @property
    def input_width(self):
        return 4 * self.head_dim

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps({"family": self.family, **asdict(self)}, sort_keys=True).encode()
        ).hexdigest()

    @property
    def arithmetic(self):
        return {
            "pool": (
                "aiter FP32 online softmax;K=8;overlap previous first4/current last4;fixed NW=1"
            ),
            "norm_rope": "FP32 RMSNorm;BF16 cos/sin;GPT-J tail;BF16 output",
            "position": "global group ID * 4; first group previous window is -inf padded",
            "backward": "mLite ratio4 compact graph;fixed logical row accumulation",
        }
