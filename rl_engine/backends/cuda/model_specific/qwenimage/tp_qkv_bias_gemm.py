# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Head-sharded QKV using the merged Qwen-Image linear arithmetic (#537)."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
    aligned,
    require_device,
    require_extension,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.reference.model_specific.qwenimage.tp_qkv_bias_gemm import (
    HIDDEN,
    TP_SIZES,
    shard_qkv_parameters,
    validate_qkv,
)

# These are #537's native chunk width and a bound on independent input rows.
_INPUT_CHUNK = 64
_INPUT_ROWS = 128
_CONTRACT = "qwenimage-qkv-tp-det-linear-v1"


def _require_qkv_extension():
    require_extension()
    symbols = ("qwenimage_det_linear_input_partials", "qwenimage_det_linear_fold_input_partials")
    if not all(callable(getattr(_C, name, None)) for name in symbols):
        raise RuntimeError("rebuild rl_engine._C with Qwen-Image TP QKV entry points")


def _tp_world(collective):
    world = 1 if collective is None else collective.world_size
    if type(world) is not int or world not in TP_SIZES:
        raise ValueError("TP size must be 1, 2, 4 or 8")
    if collective is not None:
        if not 0 <= collective.rank < world:
            raise ValueError("collective rank must be in [0, TP size)")
        if collective.backend_id != "cuda_ipc_fixed_tree":
            raise ValueError("TP QKV requires the existing CUDA deterministic collective")
    return world


def _ordered_input_partials(gathered, tp_world):
    """Convert rank-major [Q_rank,K_rank,V_rank] chunks to global Q/K/V order."""
    chunks = HIDDEN // _INPUT_CHUNK
    rows = gathered.shape[1]
    return (
        gathered.view(tp_world, 3, chunks // tp_world, rows, HIDDEN)
        .permute(1, 0, 2, 3, 4)
        .reshape(3 * chunks, rows, HIDDEN)
        .contiguous()
    )


def _input_gradient(grad, weight, dtype, collective):
    world = _tp_world(collective)
    rows_per_call = _INPUT_ROWS
    if world > 1:
        bytes_per_row = (3 * HIDDEN // _INPUT_CHUNK // world) * HIDDEN * 4
        rows_per_call = min(rows_per_call, collective.max_size_bytes // bytes_per_row)
        if rows_per_call < 1:
            raise ValueError("collective capacity cannot hold one row of QKV input partials")
    output = torch.empty((grad.shape[0], HIDDEN), device=grad.device, dtype=dtype)
    for start in range(0, grad.shape[0], rows_per_call):
        stop = min(start + rows_per_call, grad.shape[0])
        partials = _C.qwenimage_det_linear_input_partials(aligned(grad[start:stop].float()), weight)
        if world > 1:
            gathered = collective.all_gather(partials)
            partials = _ordered_input_partials(gathered, world)
        output[start:stop].copy_(_C.qwenimage_det_linear_fold_input_partials(partials, dtype))
    return output


class _TPQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, logical_keys, parameter_id, collective):
        ctx.save_for_backward(x, weight)
        ctx.collective = collective
        ctx.session = active_session()
        ctx.weight_id = str(parameter_id) + ".weight"
        ctx.bias_id = str(parameter_id) + ".bias"
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.session is None:
                raise RuntimeError("trainable QKV parameters require a canonical backward session")
            if logical_keys is None or not isinstance(parameter_id, str) or not parameter_id:
                raise ValueError("trainable QKV parameters require logical_keys and parameter_id")
            if ctx.needs_input_grad[1]:
                ctx.weight_slot = ctx.session.register(ctx.weight_id, logical_keys)
            if ctx.needs_input_grad[2]:
                ctx.bias_slot = ctx.session.register(ctx.bias_id, logical_keys)
        return _C.qwenimage_det_linear_forward(x, weight, bias, 0, False)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = dw = db = None
        executed = []
        if ctx.needs_input_grad[0]:
            dx = _input_gradient(grad, weight, x.dtype, ctx.collective)
            executed.extend(("input_partials", "fold_input_partials"))
        if ctx.needs_input_grad[1]:

            def weight_reducer(rows, grads):
                executed.append("backward_weight")
                return _C.qwenimage_det_linear_backward_weight(
                    aligned(grads.float()), aligned(rows), weight.dtype, False
                )[0]

            dw = ctx.session.submit_linear(ctx.weight_id, ctx.weight_slot, x, grad, weight_reducer)
        if ctx.needs_input_grad[2]:

            def bias_reducer(rows):
                executed.append("backward_bias")
                return _C.qwenimage_det_linear_backward_bias(aligned(rows.float()), x.dtype)

            db = ctx.session.submit_rows(
                ctx.bias_id,
                ctx.bias_slot,
                grad,
                bias_reducer,
            )
        if executed:
            record_backward(
                "qwenimage_tp_qkv_bias_gemm",
                kernel_id="+".join(
                    "rl_engine._C.qwenimage_det_linear_" + name for name in executed
                ),
                impl=_CONTRACT,
                family="cuda",
            )
        return dx, dw, db, None, None, None


class QwenImageTPQKVBiasGemmCudaOp:
    """One stream's local [Q_heads;K_heads;V_heads] projection and strict VJP.

    Pass an existing DeterministicCollective for real TP. A missing collective
    selects TP1. Each rank owns local weights and receives replicated input rows.
    Use distinct parameter IDs for image and text, shared across microbatches.
    """

    op_class = "reduction"
    kernel_id = "rl_engine._C.qwenimage_det_linear_forward"

    def __init__(self, collective=None):
        _require_qkv_extension()
        self.collective = collective
        self.tp_world = _tp_world(collective)

    def forward(self, x, weight, bias=None, *, logical_keys=None, parameter_id=None):
        validate_qkv(x, weight, bias, self.tp_world)
        require_device(x)
        if self.collective is not None and self.collective.device != x.device:
            raise ValueError("collective and QKV tensors must use the same device")
        if logical_keys is not None:
            if logical_keys.device != x.device or logical_keys.dtype != torch.int64:
                raise ValueError("logical_keys must be int64 on the input device")
            if logical_keys.ndim != 2 or logical_keys.shape[0] != x.shape[0]:
                raise ValueError("logical_keys must have one row per input row")
            if logical_keys.shape[1] not in (2, 3):
                raise ValueError("logical_keys must have two or three columns")
        x, weight = aligned(x), aligned(weight)
        bias = None if bias is None else aligned(bias)
        if not torch.is_grad_enabled():
            return _C.qwenimage_det_linear_forward(x, weight, bias, 0, False)[0]
        return _TPQKV.apply(
            x,
            weight,
            bias,
            logical_keys,
            parameter_id,
            self.collective,
        )

    __call__ = forward

    def implementation_info(self, x, weight, bias=None, **kwargs):
        validate_qkv(x, weight, bias, self.tp_world)
        require_device(x)
        return {
            "operator": "qwenimage_tp_qkv_bias_gemm",
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": False,
            "contract": _CONTRACT,
            "kernel_id": self.kernel_id,
            "tp_world_size": self.tp_world,
            "packing": "Q_local;K_local;V_local; whole contiguous heads",
            "input_gradient": "64-column Neumaier partials; global ordered Kahan fold",
            "parameter_gradient": "canonical logical rows; #537 compensated native reducers",
            "transport": "none" if self.collective is None else self.collective.backend_id,
            "accumulation_dtype": "float32",
            "execution_dtype": str(x.dtype),
            "scope": "implementation configuration, not a per-call execution record",
        }


__all__ = ["QwenImageTPQKVBiasGemmCudaOp", "shard_qkv_parameters"]
