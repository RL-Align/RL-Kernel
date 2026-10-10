# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Explicit CUDA forward primitives for the Qwen3-Next shared TP4 profile.

These functions do not install framework hooks or advertise engine capabilities.
The caller owns parameter sharding, TP collectives, positions and cache identity.
Weights remain BF16; only the router explicitly computes in FP32, as in VIME's
Qwen3-Next configuration. No CPU or alternate-kernel fallback is permitted.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

FORWARD_PROVIDER_ID = "qwen3-next-shared-cuda-forward-v1"


@lru_cache(maxsize=1)
def _vllm_linear():
    if version("vllm") != "0.30.0":
        raise RuntimeError("Shared Qwen3-Next linear requires pinned vLLM 0.30.0")
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    return linear_batch_invariant


# The batch-invariant tile vLLM itself selects under VLLM_BATCH_INVARIANT=1
# (fused_moe.get_default_config), passed explicitly so no environment variable,
# tuned-config file or token count can change it.
_GROUPED_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "SPLIT_K": 1,
}


@lru_cache(maxsize=1)
def _vllm_grouped():
    if version("vllm") != "0.30.0":
        raise RuntimeError("Shared Qwen3-Next experts require pinned vLLM 0.30.0")
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
        dispatch_fused_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.utils import resolve_moe_use_td

    return _prepare_expert_assignment, dispatch_fused_moe_kernel, resolve_moe_use_td


def grouped_route_linear(rows, weight, route_indices):
    """Per-route ``rows @ weight[expert].T`` through vLLM's fused MoE Triton kernel.

    ``rows`` is [T, K] (one row per token, shared by its ten routes) or
    [T*10, K] (one row per route); the result is [T, 10, N] in route order.
    No routed weight is applied and nothing is summed: every output row has
    exactly one writer, and its K loop has the fixed order of the fixed tile.
    """
    prepare, dispatch, use_td = _vllm_grouped()
    if use_td():
        raise RuntimeError("VLLM_TRITON_USE_TD changes the MoE kernel; strict mode requires 0")
    import triton.language as tl

    tokens, top_k = route_indices.shape
    out = rows.new_empty((tokens, top_k, weight.shape[1]))
    if tokens == 0:
        return out
    indices = route_indices.to(torch.int32).contiguous()
    sorted_ids, expert_ids, padded = prepare(
        indices, _GROUPED_CONFIG, tokens, top_k, weight.shape[0], None, ignore_invalid_experts=True
    )
    dispatch(
        rows.contiguous(),
        weight,
        out,
        None,
        None,
        None,
        None,
        sorted_ids,
        expert_ids,
        padded,
        False,
        top_k if rows.shape[0] == tokens else 1,
        _GROUPED_CONFIG,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
    return out


def _cuda_tensors(*tensors):
    if torch.version.hip is not None or any(not value.is_cuda for value in tensors):
        raise ValueError("Shared Qwen3-Next forward requires CUDA tensors; no CPU fallback")
    if any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("Shared Qwen3-Next tensors must be on the same device")


class _SharedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        if x.shape[0] == 0:
            return x.new_empty((0, weight.shape[0]))
        with torch.cuda.device(x.device):
            return _vllm_linear()(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        if x.shape[0] == 0:
            return (
                torch.empty_like(x) if ctx.needs_input_grad[0] else None,
                torch.zeros_like(weight) if ctx.needs_input_grad[1] else None,
                weight.new_zeros(weight.shape[0]) if ctx.has_bias else None,
            )
        grad = grad_output.contiguous()
        with torch.cuda.device(x.device):
            linear = _vllm_linear()
            dx = linear(grad, weight.t()) if ctx.needs_input_grad[0] else None
            dw = linear(grad.t(), x.t()) if ctx.needs_input_grad[1] else None
            db = grad.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db


def shared_linear(x, weight, bias=None):
    """Pinned vLLM deterministic GEMM with an explicit first-order VJP.

    x is [..., K], weight is [N, K], output is [..., N]. Operands have one
    common BF16 or FP32 dtype. Empty token dimensions preserve autograd.
    Backward executes dX=dY@W and dW=dY.T@X through the same fixed provider.
    """
    _cuda_tensors(x, weight, *(() if bias is None else (bias,)))
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("Shared linear requires x[..., K] and weight[N, K]")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError("Shared linear operands must have one common BF16 or FP32 dtype")
    if x.dtype == torch.float32:
        import triton

        if triton.knobs.language.fp32_default != "ieee":
            raise RuntimeError(
                "FP32 shared linear requires TRITON_F32_DEFAULT=ieee at process startup"
            )
    if min(weight.shape) < 1:
        raise ValueError("Shared linear weight dimensions must be positive")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != x.dtype):
        raise ValueError("Shared linear bias must be [N] with the input dtype")
    out = _SharedLinear.apply(x.reshape(-1, x.shape[-1]), weight, bias)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def shared_router(x, weight):
    """Compute the 512-expert FP32 router from BF16 inputs/parameter storage."""
    _cuda_tensors(x, weight)
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Qwen3-Next router input and stored weight must be BF16")
    if x.ndim != 2 or weight.shape != (512, x.shape[-1]):
        raise ValueError("Qwen3-Next router requires x[T, H] and weight[512, H]")
    return shared_linear(x.float(), weight.float())


@dataclass(frozen=True)
class Routing:
    """Descending probability routes; equal scores prefer smaller expert IDs."""

    weights: torch.Tensor
    indices: torch.Tensor


def fixed_order_row_sum(values):
    """Sum the last dimension by a pairwise tree of elementwise adds.

    ``torch.sum`` picks its reduction order from the tensor shape, so the same
    row summed inside an 8-row and a 32-row tensor can differ by one ulp unless
    vLLM's batch-invariant aten overrides are installed. Qualification runs
    execute with ``VLLM_BATCH_INVARIANT=0``, where those overrides are absent.
    Elementwise adds have no such freedom: every row is summed in one fixed
    tree whatever the row count. Odd widths are padded with exact zeros.
    """
    while values.shape[-1] > 1:
        if values.shape[-1] % 2:
            values = torch.cat((values, values.new_zeros((*values.shape[:-1], 1))), dim=-1)
        values = values[..., 0::2] + values[..., 1::2]
    return values


def fixed_order_softmax(logits):
    """Row softmax from a max subtraction, exp and one fixed-order row sum."""
    shifted = logits - torch.amax(logits, dim=-1, keepdim=True)
    exponentials = torch.exp(shifted)
    return exponentials / fixed_order_row_sum(exponentials)


def stable_top10_routes(router_logits):
    """Select ten distinct experts and normalize their FP32 probabilities.

    The full 512-way softmax precedes selection, preserving Qwen's normalized
    top-k arithmetic. Stable sort fixes the otherwise ambiguous tie boundary.
    Indices are discrete; gradients flow through the selected probabilities.
    Both row sums use the fixed-order tree, so a token's route weights do not
    depend on how many other tokens share its forward pass.
    """
    _cuda_tensors(router_logits)
    if router_logits.ndim != 2 or router_logits.shape[1] != 512:
        raise ValueError("Qwen3-Next routing requires logits[T, 512]")
    if router_logits.dtype != torch.float32:
        raise ValueError("Qwen3-Next router logits must be FP32")
    if not torch.isfinite(router_logits).all():
        raise ValueError("Qwen3-Next router logits must be finite")
    probabilities = fixed_order_softmax(router_logits)
    indices = torch.argsort(router_logits, dim=-1, descending=True, stable=True)[:, :10]
    selected = probabilities.gather(1, indices)
    return Routing(selected / fixed_order_row_sum(selected), indices)


def combine_routes(expert_outputs, route_weights):
    """Combine [T, 10, H] BF16 outputs by ten ordered FP32 multiply-adds.

    There are no atomic sums. Each route's product and each addition is a
    separate eager operation; the single BF16 cast follows the final route.
    """
    _cuda_tensors(expert_outputs, route_weights)
    if expert_outputs.ndim != 3 or expert_outputs.shape[1] != 10:
        raise ValueError("Qwen3-Next expert outputs must have shape [T, 10, H]")
    if route_weights.shape != expert_outputs.shape[:2]:
        raise ValueError("Route weights must have shape [T, 10]")
    if expert_outputs.dtype != torch.bfloat16 or route_weights.dtype != torch.float32:
        raise ValueError("Combine requires BF16 expert outputs and FP32 route weights")
    output = expert_outputs[:, 0].float() * route_weights[:, 0, None]
    for slot in range(1, 10):
        output = output + expert_outputs[:, slot].float() * route_weights[:, slot, None]
    return output.to(torch.bfloat16)


def _expert_activation(gate_up_rows):
    """SwiGLU in FP32 from the BF16 [gate, up] projection; returns its parts too."""
    gate, up = gate_up_rows.float().chunk(2, dim=-1)
    silu_gate = F.silu(gate)
    return silu_gate * up, silu_gate, gate, up


class _RoutedExperts(torch.autograd.Function):
    """Grouped expert forward; per-expert backward into one dense gradient buffer."""

    @staticmethod
    def forward(ctx, x, gate_up, down, route_indices):
        ctx.save_for_backward(x, gate_up, down, route_indices)
        with torch.cuda.device(x.device):
            projected = grouped_route_linear(x, gate_up, route_indices)
            activated = _expert_activation(projected)[0]
            activated = activated.to(x.dtype).reshape(-1, activated.shape[-1])
            return grouped_route_linear(activated, down, route_indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, gate_up, down, route_indices = ctx.saved_tensors
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dgate_up = torch.zeros_like(gate_up) if ctx.needs_input_grad[1] else None
        ddown = torch.zeros_like(down) if ctx.needs_input_grad[2] else None
        top_k = route_indices.shape[1]
        slots_by_expert = [[] for _ in range(gate_up.shape[0])]
        for slot, expert in enumerate(route_indices.flatten().tolist()):
            slots_by_expert[expert].append(slot)
        grad_flat = grad_output.reshape(-1, x.shape[1])
        with torch.cuda.device(x.device):
            linear = _vllm_linear()
            projected = grouped_route_linear(x, gate_up, route_indices)
            projected = projected.reshape(-1, projected.shape[-1])
            for expert, slots in enumerate(slots_by_expert):
                if not slots:
                    continue
                slots = torch.tensor(slots, dtype=torch.long, device=x.device)
                tokens = slots // top_k
                activated, silu_gate, gate32, up32 = _expert_activation(
                    projected.index_select(0, slots)
                )
                activated = activated.to(x.dtype)
                grad = grad_flat.index_select(0, slots)
                if ddown is not None:
                    ddown[expert].copy_(linear(grad.t(), activated.t()))
                if dx is None and dgate_up is None:
                    continue
                rows = x.index_select(0, tokens)
                dactivated = linear(grad, down[expert].t()).float()
                sigmoid = gate32.sigmoid()
                dgate = dactivated * up32 * (sigmoid + gate32 * sigmoid * (1 - sigmoid))
                dup = dactivated * silu_gate
                dprojected = torch.cat((dgate, dup), dim=-1).to(x.dtype)
                if dgate_up is not None:
                    dgate_up[expert].copy_(linear(dprojected.t(), rows.t()))
                if dx is not None:
                    # Top-k has distinct experts, so tokens are unique within this
                    # launch. Ascending expert order fixes each token's sum order.
                    partial = linear(dprojected, gate_up[expert].t())
                    dx.index_copy_(0, tokens, dx.index_select(0, tokens) + partial)
        return dx, dgate_up, ddown, None


def shared_moe(x, router_weight, gate_up_weights, down_weights):
    """Evaluate TP-local routed experts; caller reduces the returned output.

    Weights are [512, 2*I_local, H] (gate then up) and [512, H, I_local].
    All experts belong to EP=1. Each (token, route) output row has exactly one
    writer and the ten routes are combined in route order. Shared-expert and
    TP-reduction ownership remains with the model adapter. The forward has no
    host synchronisation; the backward builds per-expert row lists on the host.
    """
    _cuda_tensors(x, router_weight, gate_up_weights, down_weights)
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("Qwen3-Next MoE input must be BF16 [T, H]")
    if gate_up_weights.ndim != 3 or down_weights.ndim != 3:
        raise ValueError("Qwen3-Next expert weights must have three dimensions")
    local_intermediate = down_weights.shape[-1]
    if local_intermediate < 1:
        raise ValueError("Expert intermediate dimension must be positive")
    if gate_up_weights.shape != (512, 2 * local_intermediate, x.shape[1]):
        raise ValueError("gate_up_weights must have shape [512, 2*I_local, H]")
    if down_weights.shape != (512, x.shape[1], local_intermediate):
        raise ValueError("down_weights must have shape [512, H, I_local]")
    if gate_up_weights.dtype != x.dtype or down_weights.dtype != x.dtype:
        raise ValueError("Qwen3-Next expert weights must remain BF16")
    routes = stable_top10_routes(shared_router(x, router_weight))
    outputs = _RoutedExperts.apply(x, gate_up_weights, down_weights, routes.indices)
    return combine_routes(outputs, routes.weights), routes
