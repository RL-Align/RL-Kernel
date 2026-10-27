# SPDX-License-Identifier: Apache-2.0
"""Item-16 storage/arithmetic contract, distinct from fused Indexer scores."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from rl_engine.contracts.operators.attention import AttentionContractError

ICV_BACKEND_ID = "aiter.indexer_icv.gfx942.fixed_dot.v1"
MXFP4_LAYOUT = "e2m1.low_nibble_first.row_major.e8m0_group32"
FP8_LAYOUT = "e4m3.row_major.fp32_per_row_scale"


class IndexerICVError(AttentionContractError):
    pass


def source_sha256(path, *, label):
    """Fail closed when a numerical provider has no inspectable source file."""
    if not isinstance(path, (str, Path)) or not path:
        raise IndexerICVError(f"source fingerprint unavailable for {label}")
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise IndexerICVError(f"source fingerprint unavailable for {label}") from exc


@dataclass(frozen=True)
class IndexerICVContract:
    num_heads: int = 64
    head_dim: int = 128
    precision: str = "mxfp4"
    fp8_encoding: str | None = None
    scale_location: str = "operands"
    query_tile_rows: int = 32
    key_tile_rows: int = 256
    max_output_bytes: int = 256 * 1024 * 1024
    max_prepared_key_bytes: int = 256 * 1024 * 1024
    output_dtype: str = "fp32"
    postprocess: str = "none"
    graph_capture: bool = False
    backend_id: str = ICV_BACKEND_ID

    def __post_init__(self):
        for field in (
            "num_heads",
            "head_dim",
            "query_tile_rows",
            "key_tile_rows",
            "max_output_bytes",
            "max_prepared_key_bytes",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise IndexerICVError(f"{field} must be a positive integer")
        if self.num_heads > 128 or self.head_dim > 256:
            raise IndexerICVError("validated envelope is H<=128, D<=256")
        if self.precision not in ("mxfp4", "fp8"):
            raise IndexerICVError("precision must be explicitly mxfp4 or fp8")
        if self.precision == "fp8" and self.fp8_encoding not in ("e4m3fn", "e4m3fnuz"):
            raise IndexerICVError("FP8 requires an explicit OCP or FNUZ encoding")
        if self.precision == "mxfp4" and self.fp8_encoding is not None:
            raise IndexerICVError("MXFP4 does not use an FP8 encoding flag")
        if self.scale_location != "operands":
            raise IndexerICVError(
                "independent ICV requires Q/K scales, not scales folded into weights"
            )
        if self.output_dtype != "fp32" or self.postprocess != "none":
            raise IndexerICVError("ICV is raw FP32: no scaling, ReLU, weights, sum or masking")
        if self.graph_capture is not False:
            raise IndexerICVError("Graph capture is not validated")
        if self.backend_id != ICV_BACKEND_ID:
            raise IndexerICVError("unknown backend; no silent fallback")
        if self.query_tile_rows * self.num_heads * self.key_tile_rows * 4 > self.max_output_bytes:
            raise IndexerICVError("one output tile exceeds max_output_bytes")

    @property
    def padded_dim(self):
        return (self.head_dim + 31) // 32 * 32

    @property
    def layout(self):
        return MXFP4_LAYOUT if self.precision == "mxfp4" else FP8_LAYOUT

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def arithmetic(self):
        if self.precision == "mxfp4":
            return "exact_e2m1_e8m0_decode_bf16;dot_fp32;D_tiles64;split1"
        return "e4m3fnuz_dot_fp32;mul(q_scale*k_scale)_once;split1"
