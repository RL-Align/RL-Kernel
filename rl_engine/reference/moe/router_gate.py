# SPDX-License-Identifier: Apache-2.0
"""The two CUDA gate entries pinned by task 28, retained for separate comparison."""

import torch


def cuda_gate_reference(hidden, weight, *, entry):
    """No cross-platform bitwise claim; CuTe absence is an error, not a fallback.

    vLLM d262044021bda30273cb0833cd8bf05dd27deab1: GateLinear BI mm path.
    mLite 327a238239448de6771beea185dc6902a8de3eaa: small-M CuTe path.
    """
    if torch.version.hip is not None or not hidden.is_cuda:
        raise RuntimeError("CUDA gate references require an NVIDIA GPU")
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("CUDA gate reference requires BF16 operands")
    if entry == "mm":
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)
    if entry == "cute":
        if hidden.shape[0] > 16:
            raise ValueError("mLite CuTe gate reference requires M <= 16")
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
            is_available,
            ll_bf16_gemm,
        )

        if not is_available():
            raise RuntimeError("the mLite CuTe gate reference is unavailable")
        return ll_bf16_gemm(hidden, weight)
    raise ValueError(f"unknown CUDA gate reference entry: {entry}")
