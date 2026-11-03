# SPDX-License-Identifier: Apache-2.0
"""Task 20's separate, typed Indexer profile reuses the C4 grouping contract."""

from dataclasses import dataclass
from typing import ClassVar

from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract, C4Error, require

INDEXER_C4_BACKEND_ID = "aiter.indexer_c4.gfx942.native_hadamard.v1"
IndexerC4Error = C4Error


@dataclass(frozen=True)
class IndexerC4Contract(C4Contract):
    expected_head_dim: ClassVar[int] = 128
    expected_backend_id: ClassVar[str] = INDEXER_C4_BACKEND_ID
    family: ClassVar[str] = "indexer_c4"
    head_dim: int = 128
    backend_id: str = INDEXER_C4_BACKEND_ID
    precision: str = "mxfp4"

    def __post_init__(self):
        super().__post_init__()
        require(self.precision in ("mxfp4", "fp8"), "explicit MXFP4 or FNUZ FP8 precision required")

    @property
    def layout(self):
        return (
            "e2m1.low_nibble_first.row_major.e8m0_group32"
            if self.precision == "mxfp4"
            else "e4m3.row_major.fp32_per_row_scale"
        )

    @property
    def arithmetic(self):
        return {
            **super().arithmetic,
            "rotate": "aiter.rotate_activation;normalized_Walsh_Hadamard128;BF16",
            "quantize": (
                "aiter.per_1x32_f4_quant_triton;unshuffled_group32_E8M0"
                if self.precision == "mxfp4"
                else "vllm.per_token_group_quant_fp8;FNUZ;UE8M0_FP32_multiplier"
            ),
            "backward_rotation": "native_Hadamard(BF16_dY);mLite_FP32_compact_VJP",
            "quantized_backward": "not_defined;train_on_explicit_BF16_values_only",
        }
