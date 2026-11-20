# SPDX-License-Identifier: Apache-2.0
"""Reuse V4 packed FP4 GEMM/VJP, using V4.1's distinct activation quantizer."""

from rl_engine.backends.rocm.gemm.v41_fp8 import quantize_fp8_block32
from rl_engine.backends.rocm.moe.packed_mx import RocmPackedMXBackend


class V41PackedMXBackend(RocmPackedMXBackend):
    profile = "dsv41-ceil-amax-over448-min1e-4-mxfp8-mxfp4-block32-v1"

    def quantize(self, x):
        return quantize_fp8_block32(x)
