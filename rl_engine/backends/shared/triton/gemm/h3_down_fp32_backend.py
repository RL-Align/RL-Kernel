# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""H3 BF16 adapter for the existing fixed-order Triton GEMM kernel."""

import hashlib

import torch

_BINARY_HASHES = {}


class FixedFP32Backend:
    """BF16 MMA tiles with FP32 accumulation and ordered FP32 tile additions.

    Reuses `_det_gemm_fp32_kernel` with PROMOTE_INPUTS=False and a BF16 output.
    Operands stay BF16; each 32-wide dot accumulates in FP32 before the ordered
    FP32 tile sum and one final BF16 cast. The class name is a historical alias
    for the accumulator precision, not FP32 multiplication.
    Fixed 64x64x32 tiles, four warps, two stages, no tuning or split-K.
    """

    h3_kernel_id = "rl_engine.kernels.ops.triton.matmul.det_gemm._det_gemm_fp32_kernel"

    def __init__(self):
        from rl_engine.backends.shared.triton.gemm import det_gemm

        if not det_gemm._TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for H3 BF16 MMA tiles; fallback is forbidden")
        self.kernel = det_gemm._det_gemm_fp32_kernel
        self.triton = det_gemm.triton

    def gemm(self, a, b):
        m, k = a.shape
        if b.shape[0] != k:
            raise ValueError(f"GEMM inner dimensions differ: {k} != {b.shape[0]}")
        n = b.shape[1]
        output = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        compiled = self.kernel[(self.triton.cdiv(m, 64), self.triton.cdiv(n, 64))](
            a,
            b,
            output,
            None,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            *output.stride(),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            PROMOTE_INPUTS=False,
            HAS_BIAS=False,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
        if compiled.hash not in _BINARY_HASHES:
            _BINARY_HASHES[compiled.hash] = hashlib.sha256(compiled.asm["cubin"]).hexdigest()
        self.h3_last_launch = {
            "kernel_id": self.h3_kernel_id,
            "triton_version": self.triton.__version__,
            "compiled_kernel_hash": compiled.hash,
            "cubin_sha256": _BINARY_HASHES[compiled.hash],
            "compiled_num_warps": compiled.metadata.num_warps,
            "grid": [self.triton.cdiv(m, 64), self.triton.cdiv(n, 64)],
        }
        return output

    def det_gemm_fwd_rhs_transposed(self, x, weight):
        return self.gemm(x, weight.t())

    def det_gemm_fwd(self, gradient, weight):
        return self.gemm(gradient, weight)

    def det_gemm_db_transposed(self, x, gradient):
        return self.gemm(gradient.t(), x)
