# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini model-specific numerical contracts."""

from .grouped_gemm import grouped_expert_gemm, grouped_expert_gemm_cuda
from .expert_mlp import expert_gate_up, expert_gate_up_cuda, softcap
from .attention_residual import attention_o_residual, attention_o_residual_cuda
from .dynamic_kv_conv import dynamic_kv_conv, dynamic_kv_conv_cuda
from .full_attention import full_attention, full_attention_cuda
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
from .visit_cache import VisitKey, VisitState, VisitStateCache, clone_visit_state_cuda
from .shared_weight_grad import (
    accumulate_shared_weight_gradients,
    accumulate_shared_weight_gradients_cuda,
)
from .token_embedding import (
    PADDED_VOCAB_SIZE,
    VALID_VOCAB_SIZE,
    token_embedding,
    token_embedding_cuda,
)

__all__ = [
    "KERNEL_SIZE",
    "STREAMS",
    "PADDED_VOCAB_SIZE",
    "VALID_VOCAB_SIZE",
    "RoutingResult",
    "causal_short_conv",
    "causal_short_conv_cuda",
    "causal_short_conv_rocm",
    "causal_short_conv_streams",
    "grouped_expert_gemm",
    "grouped_expert_gemm_cuda",
    "expert_gate_up",
    "expert_gate_up_cuda",
    "attention_o_residual",
    "attention_o_residual_cuda",
    "VisitKey",
    "VisitState",
    "VisitStateCache",
    "clone_visit_state_cuda",
    "dynamic_kv_conv",
    "dynamic_kv_conv_cuda",
    "full_attention",
    "full_attention_cuda",
    "accumulate_shared_weight_gradients",
    "accumulate_shared_weight_gradients_cuda",
    "route_topk_sigmoid",
    "route_topk_sigmoid_cuda",
    "route_topk_sigmoid_rocm",
    "token_embedding",
    "token_embedding_cuda",
    "softcap",
]
