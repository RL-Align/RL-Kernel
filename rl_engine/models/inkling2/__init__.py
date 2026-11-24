# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini model-specific numerical contracts."""

from .router import (
    RoutingResult,
    route_topk_sigmoid,
    route_topk_sigmoid_cuda,
    route_topk_sigmoid_rocm,
)
from .token_embedding import (
    PADDED_VOCAB_SIZE,
    VALID_VOCAB_SIZE,
    token_embedding,
    token_embedding_cuda,
    token_embedding_rocm,
)

__all__ = [
    "PADDED_VOCAB_SIZE",
    "VALID_VOCAB_SIZE",
    "RoutingResult",
    "route_topk_sigmoid",
    "route_topk_sigmoid_cuda",
    "route_topk_sigmoid_rocm",
    "token_embedding",
    "token_embedding_cuda",
    "token_embedding_rocm",
]
