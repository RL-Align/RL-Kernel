# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""H3 adapter for the existing fixed-order Triton FP32 tile GEMM kernel."""

import hashlib

import torch

_BINARY_HASHES = {}


class FixedFP32Backend:
    """BF16 operands; FP32 dot tiles and ordered FP32 additions; one final cast.

    Reuses `_det_gemm_fp32_kernel`, bypassing its FP32-input/output wrapper.
    Fixed 64x64x32 tiles, four warps, two stages, no tuning or split-K.
    """

    h3_kernel_id = "rl_engine.kernels.ops.triton.matmul.det_gemm._det_gemm_fp32_kernel"

    def __init__(self):
        from rl_engine.kernels.ops.triton.matmul import det_gemm

        if not det_gemm._TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for H3 FP32 tiles; fallback is forbidden")
        self.kernel = det_gemm._det_gemm_fp32_kernel
        self.triton = det_gemm.triton

    def gemm(self, a, b):
        m, k = a.shape
        n = b.shape[1]
        output = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        compiled = self.kernel[(self.triton.cdiv(m, 64), self.triton.cdiv(n, 64))](
            a,
            b,
            output,
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
