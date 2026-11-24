# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini model-specific numerical contracts."""

from .grouped_gemm import grouped_expert_gemm, grouped_expert_gemm_cuda
from .router import (
    RoutingResult,
    route_topk_sigmoid,
    route_topk_sigmoid_cuda,
    route_topk_sigmoid_rocm,
)
from .short_conv import (
    KERNEL_SIZE,
    STREAMS,
    causal_short_conv,
    causal_short_conv_cuda,
    causal_short_conv_rocm,
    causal_short_conv_streams,
)

__all__ = [
    "KERNEL_SIZE",
    "STREAMS",
    "RoutingResult",
    "causal_short_conv",
    "causal_short_conv_cuda",
    "causal_short_conv_rocm",
    "causal_short_conv_streams",
    "grouped_expert_gemm",
    "grouped_expert_gemm_cuda",
    "route_topk_sigmoid",
    "route_topk_sigmoid_cuda",
    "route_topk_sigmoid_rocm",
]
