# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image row-parallel output projection reusing WS1 #459 arithmetic."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
    require_device,
    require_extension,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32
from rl_engine.reference.gemm.attn_out_bias_gemm import ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION
from rl_engine.reference.model_specific.qwenimage.tp_attn_out_bias_gemm import (
    HIDDEN,
    TP_SIZES,
    shard_attn_out_parameters,
    validate_attn_out,
)

_CONTRACT = "qwenimage-attn-out-tp-tree"


def _tp_world(collective):
    world = 1 if collective is None else collective.world_size
    if type(world) is not int or world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    if collective is not None:
        if not 0 <= collective.rank < world:
            raise ValueError("collective rank must be in [0, TP size)")
        if collective.backend_id != "cuda_ipc_fixed_tree":
            raise ValueError(
                "TP attention output requires the existing CUDA deterministic collective"
            )
        if collective.max_size_bytes < HIDDEN * 4:
            raise ValueError("collective capacity cannot hold one FP32 output row")
    return world


def _forward(x, weight, bias, collective):
    """Compute WS1 subtree roots, complete the same tree, then add bias and cast."""
    weight_f = weight.float().contiguous()
    bias_f = None if bias is None else bias.float().contiguous()
    rows_per_call = x.shape[0]
    if collective is not None:
        rows_per_call = min(rows_per_call, collective.max_size_bytes // (HIDDEN * 4))
    output = torch.empty((x.shape[0], HIDDEN), dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[0], rows_per_call):
        stop = min(start + rows_per_call, x.shape[0])
        partial = _C.attn_out_tree_gemm_cuda(x[start:stop].float().contiguous(), weight_f)
        if collective is not None:
            partial = collective.all_reduce(partial)
        if bias_f is not None:
            partial.add_(bias_f)
        output[start:stop].copy_(partial.to(x.dtype))
    return output


class _TPAttnOut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, logical_keys, parameter_id, collective):
        ctx.save_for_backward(x, weight)
        ctx.session = active_session()
        ctx.weight_id = str(parameter_id) + ".weight"
        ctx.bias_id = str(parameter_id) + ".bias"
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.session is None:
                raise RuntimeError(
                    "trainable attention output parameters require a canonical backward session"
                )
            if logical_keys is None or not isinstance(parameter_id, str) or not parameter_id:
                raise ValueError(
                    "trainable attention output parameters require logical_keys and parameter_id"
                )
            if ctx.needs_input_grad[1]:
                ctx.weight_slot = ctx.session.register(ctx.weight_id, logical_keys)
            if ctx.needs_input_grad[2]:
                ctx.bias_slot = ctx.session.register(ctx.bias_id, logical_keys)
        return _forward(x, weight, bias, collective)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = dw = db = None
        executed = []
        if ctx.needs_input_grad[0]:
            dx = _C.attn_out_tree_gemm_cuda(
                grad.float().contiguous(), weight.t().float().contiguous()
            ).to(x.dtype)
            executed.append("attn_out_tree_gemm_cuda")
        if ctx.needs_input_grad[1]:

            def weight_reducer(rows, grads):
                executed.append("attn_out_dw_left_fold_cuda")
                return _C.attn_out_dw_left_fold_cuda(
                    grads.float().contiguous(), rows.float().contiguous()
                ).to(weight.dtype)

            dw = ctx.session.submit_linear(ctx.weight_id, ctx.weight_slot, x, grad, weight_reducer)
        if ctx.needs_input_grad[2]:

            def bias_reducer(rows):
                executed.append("reduce_rows_fp32_left_fold")
                return reduce_rows_fp32(rows).to(x.dtype)

            db = ctx.session.submit_rows(ctx.bias_id, ctx.bias_slot, grad, bias_reducer)
        if executed:
            record_backward(
                "qwenimage_tp_attn_out_bias_gemm",
                kernel_id="+".join("rl_engine._C." + name for name in executed),
                impl=_CONTRACT,
                family="cuda",
            )
        return dx, dw, db, None, None, None


class QwenImageTPAttnOutBiasGemmCudaOp:
    """One stream's input-column-sharded projection and strict first-order VJP.

    All ranks receive identical logical rows and complete upstream gradients.
    The output and bias are replicated; input and weight gradients stay local.
    Use distinct parameter IDs for image/text and layers, shared by microbatches.
    """

    op_class = "reduction"
    kernel_id = "rl_engine._C.attn_out_tree_gemm_cuda"

    def __init__(self, collective=None):
        require_extension()
        symbols = (
            "attn_out_tree_gemm_cuda",
            "attn_out_dw_left_fold_cuda",
            "reduce_rows_fp32_left_fold",
        )
        if not all(callable(getattr(_C, name, None)) for name in symbols):
            raise RuntimeError(
                "rebuild rl_engine._C with WS1 attention output entry points from #459"
            )
        self.collective = collective
        self.tp_world = _tp_world(collective)

    def forward(self, x, weight, bias=None, *, logical_keys=None, parameter_id=None):
        validate_attn_out(x, weight, bias, self.tp_world)
        require_device(x)
        if self.collective is not None and self.collective.device != x.device:
            raise ValueError("collective and attention output tensors must use the same device")
        if logical_keys is not None:
            if logical_keys.device != x.device or logical_keys.dtype != torch.int64:
                raise ValueError("logical_keys must be int64 on the input device")
            if logical_keys.ndim != 2 or logical_keys.shape[0] != x.shape[0]:
                raise ValueError("logical_keys must have one row per input row")
            if logical_keys.shape[1] not in (2, 3):
                raise ValueError("logical_keys must have two or three columns")
        if not torch.is_grad_enabled():
            return _forward(x, weight, bias, self.collective)
        return _TPAttnOut.apply(x, weight, bias, logical_keys, parameter_id, self.collective)

    __call__ = forward

    def implementation_info(self, x, weight, bias=None, **kwargs):
        validate_attn_out(x, weight, bias, self.tp_world)
        require_device(x)
        return {
            "operator": "qwenimage_tp_attn_out_bias_gemm",
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": False,
            "contract": _CONTRACT,
            "ws1_contract": ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION,
            "kernel_id": self.kernel_id,
            "tp_world_size": self.tp_world,
            "packing": "contiguous whole-head input columns; replicated output and bias",
            "forward": "32-channel FMA leaves; mid-split tree; FP32 fixed-tree AllReduce",
            "parameter_gradient": "canonical logical rows; WS1 ascending left folds",
            "transport": "none" if self.collective is None else self.collective.backend_id,
            "accumulation_dtype": "float32",
            "execution_dtype": str(x.dtype),
            "scope": "implementation configuration, not a per-call execution record",
        }


__all__ = ["QwenImageTPAttnOutBiasGemmCudaOp", "shard_attn_out_parameters"]
