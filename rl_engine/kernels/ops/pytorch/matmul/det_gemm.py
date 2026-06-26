# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native PyTorch GEMM -- NON-deterministic reference baseline (WS1).

WARNING: torch.matmul (cuBLAS) does NOT guarantee batch-invariance -- cuBLAS
selects kernels by shape and may use split-K. This op exists only as a
correctness reference and benchmark target, NOT as a fallback. It is
intentionally excluded from the det_gemm registry dispatch.
"""
import torch

from rl_engine.utils.logger import logger


class NativeGemmOp:
    """Plain torch.matmul. Non-deterministic; reference / benchmark use only."""

    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        logger.info("NativeGemmOp ready (non-deterministic torch.matmul reference).")

    def __call__(self, a, b):
        return torch.matmul(a, b)


def native_gemm(a, b):
    return torch.matmul(a, b)
