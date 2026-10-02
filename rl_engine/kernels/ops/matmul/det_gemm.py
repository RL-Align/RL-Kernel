# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Import-time platform binding for strict deterministic GEMM."""

from typing import TYPE_CHECKING

import torch

if not TYPE_CHECKING and torch.version.hip is not None:
    from rl_engine.kernels.ops.rocm.matmul.det_gemm import (
        DetGemmOp,
        det_gemm_backend,
        det_gemm_backend_id,
        det_gemm_fallback_reason,
        det_gemm_linear,
        det_gemm_linear_input_gradient,
        det_gemm_linear_weight_gradient,
        deterministic_gemm,
    )
else:
    from rl_engine.kernels.ops.cuda.matmul.det_gemm import (
        DetGemmOp,
        det_gemm_backend,
        det_gemm_backend_id,
        det_gemm_fallback_reason,
        det_gemm_linear,
        det_gemm_linear_input_gradient,
        det_gemm_linear_weight_gradient,
        deterministic_gemm,
    )

__all__ = [
    "DetGemmOp",
    "det_gemm_backend",
    "det_gemm_backend_id",
    "det_gemm_fallback_reason",
    "det_gemm_linear",
    "det_gemm_linear_input_gradient",
    "det_gemm_linear_weight_gradient",
    "deterministic_gemm",
]
