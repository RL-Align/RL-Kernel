# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini model-specific numerical contracts."""

from .router import (
    RoutingResult,
    route_topk_sigmoid,
    route_topk_sigmoid_cuda,
    route_topk_sigmoid_rocm,
)
from .shared_weight_grad import (
    accumulate_shared_weight_gradients,
    accumulate_shared_weight_gradients_cuda,
)

__all__ = [
    "RoutingResult",
    "accumulate_shared_weight_gradients",
    "accumulate_shared_weight_gradients_cuda",
    "route_topk_sigmoid",
    "route_topk_sigmoid_cuda",
    "route_topk_sigmoid_rocm",
]
