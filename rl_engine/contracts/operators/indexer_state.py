# SPDX-License-Identifier: Apache-2.0
"""Task 22: independent FP32 Indexer state and row-major MXFP4 cached rows."""

from dataclasses import dataclass
from typing import ClassVar

from rl_engine.contracts.operators.compressor_state import CompressorStateContract
from rl_engine.contracts.operators.kv_compressor_c4 import require


@dataclass(frozen=True)
class IndexerStateContract(CompressorStateContract):
    head_dim: ClassVar[int] = 128
    cache_dtype: ClassVar[str] = "uint8"
    cache_element_bytes: ClassVar[int] = 1
    family: ClassVar[str] = "indexer_mxfp4"

    def __post_init__(self):
        super().__post_init__()
        require(self.ratio == 4, "Indexer state requires ratio=4, not main C128")

    @property
    def cache_width(self):
        return 64

    @property
    def extra_resident_bytes(self):
        rows = self.max_requests * self.pages_per_request * 64
        return rows * (128 * 2 + 4 + 1) + self.plan_rows * 8

    @property
    def arithmetic_id(self):
        return "aiter.indexer_c4.online_fp32.wave64.native_hadamard.mxfp4_group32.v1"

    @property
    def layout(self):
        return "e2m1.low_nibble_first.row_major.e8m0_group32"
