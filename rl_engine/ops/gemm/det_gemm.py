# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Platform-selected deterministic GEMM and explicit fixed-leaf linear dispatch."""

from typing import TYPE_CHECKING, Literal

import torch

if not TYPE_CHECKING and torch.version.hip is not None:
    from rl_engine.backends.rocm.gemm.det_gemm import (
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
    from rl_engine.backends.cuda.gemm.det_gemm import (
        DetGemmOp,
        det_gemm_backend,
        det_gemm_backend_id,
        det_gemm_fallback_reason,
        det_gemm_linear,
        det_gemm_linear_input_gradient,
        det_gemm_linear_weight_gradient,
        deterministic_gemm,
    )


def fixed_leaf_linear(
    a: torch.Tensor, weight: torch.Tensor, *, backend: Literal["cuda", "triton"]
) -> torch.Tensor:
    """Apply a native BF16 [N,K] weight using the strict 32-term reduction tree."""
    if backend == "cuda":
        from rl_engine.backends.cuda.gemm import det_gemm as cuda_gemm

        return cuda_gemm.fixed_leaf_linear(a, weight)
    if backend == "triton":
        from rl_engine.backends.shared.triton.gemm import det_gemm as triton_gemm

        return triton_gemm.fixed_leaf_linear(a, weight)
    raise ValueError(f"unknown fixed-leaf linear backend {backend!r}")


__all__ = [
    "DetGemmOp",
    "det_gemm_backend",
    "det_gemm_backend_id",
    "det_gemm_fallback_reason",
    "det_gemm_linear",
    "det_gemm_linear_input_gradient",
    "det_gemm_linear_weight_gradient",
    "deterministic_gemm",
    "fixed_leaf_linear",
]
