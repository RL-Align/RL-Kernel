# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Canonical training boundaries for Qwen-Image replicated parameters."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
    aligned,
    projection_forward,
    require_device,
    require_extension,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32, rmsnorm_dweight_rows_fp32


def _session(keys, parameter_id, rows, device):
    session = active_session()
    if session is None:
        raise RuntimeError("replicated parameter gradients require a canonical backward session")
    if not isinstance(parameter_id, str) or not parameter_id:
        raise ValueError("parameter_id must be a nonempty string")
    if keys is None or keys.ndim != 2 or keys.shape[1] not in (2, 3):
        raise ValueError("logical_keys must have two or three columns")
    if keys.dtype != torch.int64 or keys.device != device or keys.shape[0] != rows:
        raise ValueError("logical_keys must be int64 with one row per input on the same device")
    return session


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, keys, parameter_id, silu):
        if silu:
            output, act = projection_forward(x, weight, bias)
        else:
            act = x
            output = _C.qwenimage_det_linear_forward(
                aligned(x), aligned(weight), None if bias is None else aligned(bias), 0, False
            )[0]
        ctx.silu = silu
        ctx.save_for_backward(x, weight, act)
        ctx.ids = [str(parameter_id) + suffix for suffix in (".weight", ".bias")]
        ctx.slots = {}
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.session = _session(keys, parameter_id, x.shape[0], x.device)
            for index in (1, 2):
                if ctx.needs_input_grad[index]:
                    ctx.slots[index] = ctx.session.register(ctx.ids[index - 1], keys)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, act = ctx.saved_tensors
        grad = aligned(grad.float())
        dx = dw = db = None
        executed = []
        if ctx.needs_input_grad[0]:
            executed.append("qwenimage_det_linear_backward_input")
            dx = _C.qwenimage_det_linear_backward_input(
                grad, aligned(weight), torch.float32 if ctx.silu else x.dtype
            )
            if ctx.silu:
                # Preserve #537: FP32 dActivation and analytic SiLU VJP, then one cast.
                z = x.float()
                sigmoid = torch.sigmoid(z)
                dx = (dx * (sigmoid * (1 + z * (1 - sigmoid)))).to(x.dtype)
        if ctx.needs_input_grad[1]:
            executed.append("qwenimage_det_linear_backward_weight")
            dw = ctx.session.submit_linear(
                ctx.ids[0],
                ctx.slots[1],
                act,
                grad,
                lambda rows, grads: _C.qwenimage_det_linear_backward_weight(
                    aligned(grads.float()), aligned(rows), weight.dtype, False
                )[0],
            )
        if ctx.needs_input_grad[2]:
            executed.append("qwenimage_det_linear_backward_bias")
            db = ctx.session.submit_rows(
                ctx.ids[1],
                ctx.slots[2],
                grad,
                lambda rows: _C.qwenimage_det_linear_backward_bias(aligned(rows.float()), x.dtype),
            )
        record_backward(
            "qwenimage_replicated_linear",
            kernel_id="+".join("rl_engine._C." + name for name in executed),
            impl="canonical_replicated_linear",
            family="cuda",
        )
        return dx, dw, db, None, None, None


def _linear(x, weight, bias, logical_keys, parameter_id, silu):
    require_extension()
    require_device(x)
    if (
        x.ndim != 2
        or weight.ndim != 2
        or min(x.shape) < 1
        or weight.shape[0] < 1
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError("x and weight must be nonempty matrices with the same input width")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("replicated linear supports bfloat16 and float32")
    for tensor in (weight,) if bias is None else (weight, bias):
        if tensor.device != x.device or tensor.dtype != x.dtype:
            raise ValueError("linear inputs and parameters must share a device and dtype")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("bias must have one value per output channel")
    if not torch.is_grad_enabled():
        if silu:
            return projection_forward(x, weight, bias)[0]
        return _C.qwenimage_det_linear_forward(
            aligned(x), aligned(weight), None if bias is None else aligned(bias), 0, False
        )[0]
    return _Linear.apply(x, weight, bias, logical_keys, parameter_id, silu)


def replicated_linear(x, weight, bias=None, *, logical_keys=None, parameter_id=None):
    """Run the existing no-activation Linear with canonical parameter gradients.

    Complete inputs and upstream gradients are replicated over TP ranks. Each
    rank computes the complete gradient locally using the WS1 arithmetic.
    """
    return _linear(x, weight, bias, logical_keys, parameter_id, False)


def replicated_silu_projection(x, weight, bias, *, logical_keys, parameter_id):
    """Reuse #537 SiLU projection arithmetic with canonical parameter gradients."""
    if bias is None:
        raise ValueError("SiLU projection requires a bias")
    return _linear(x, weight, bias, logical_keys, parameter_id, True)


class _Broadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, condition, keys, parameter_id):
        ctx.save_for_backward(keys[:, 0])
        ctx.shape = condition.shape
        ctx.dtype = condition.dtype
        ctx.parameter_id = parameter_id
        ctx.session = _session(keys, parameter_id, keys.shape[0], condition.device)
        ctx.slot = ctx.session.register(parameter_id, keys)
        return condition.index_select(0, keys[:, 0].clamp_min(0))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (sample,) = ctx.saved_tensors

        def reducer(rows, grads):
            output = torch.zeros(ctx.shape, device=grad.device, dtype=ctx.dtype)
            for index in range(ctx.shape[0]):
                selected = grads[rows[:, 0] == index]
                if selected.shape[0]:
                    output[index] = reduce_rows_fp32(selected).to(ctx.dtype)
            return output

        result = ctx.session.submit_linear(
            ctx.parameter_id, ctx.slot, sample[:, None], grad.float(), reducer
        )
        return result, None, None


def broadcast_condition(condition, logical_keys, *, parameter_id):
    """Broadcast complete request conditions with a logical-row ordered VJP.

    The first key is the condition's request index. A negative first key marks
    padding. Uses of the same condition share a parameter ID. Keys include a stable
    consumer index as well as request and token identity when uses overlap.
    """
    if condition.ndim != 2 or condition.shape[0] < 1:
        raise ValueError("condition must be a nonempty matrix")
    if logical_keys is None or logical_keys.ndim != 2:
        raise ValueError("logical_keys must be a matrix")
    _session(logical_keys, parameter_id, logical_keys.shape[0], condition.device)
    valid = logical_keys[:, 0] >= 0
    if torch.any(logical_keys[valid, 0] >= condition.shape[0]):
        raise ValueError("logical sample index exceeds condition batch")
    if not condition.requires_grad or not torch.is_grad_enabled():
        return condition.index_select(0, logical_keys[:, 0].clamp_min(0))
    return _Broadcast.apply(condition, logical_keys, parameter_id)


def _head_keys(keys, heads):
    expanded = keys[:, None, :].expand(-1, heads, -1).clone()
    expanded[:, :, -1] = keys[:, None, -1] * heads + torch.arange(heads, device=keys.device)
    return expanded.reshape(-1, keys.shape[1])


class _QKNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, keys, parameter_id, collective, previous):
        local_heads = x.shape[1]
        world = 1 if collective is None else collective.world_size
        flat = x.reshape(-1, 128).contiguous()
        weight = aligned(weight)
        with torch.cuda.device(x.device):
            y, rstd = _C.rmsnorm_forward(flat, weight, 1e-6)
        ctx.save_for_backward(flat, weight, rstd)
        ctx.session = active_session()
        ctx.local_heads = local_heads
        ctx.collective = collective
        ctx.shape = x.shape
        if ctx.needs_input_grad[1]:
            ctx.session = _session(keys, parameter_id, x.shape[0], x.device)
            ctx.parameter_id = parameter_id
            ctx.slot = ctx.session.register(parameter_id, _head_keys(keys, local_heads * world))
        return y.reshape_as(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, rstd = ctx.saved_tensors
        dy = grad.reshape(-1, 128).contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            with torch.cuda.device(x.device):
                dx = _C.rmsnorm_backward_dx(dy, x, weight, rstd).reshape(ctx.shape)
        dw = None
        if ctx.collective is not None:
            ctx.session._qwenimage_qknorm_tails.pop(ctx.collective, None)
        if ctx.needs_input_grad[1]:
            rows = rmsnorm_dweight_rows_fp32(x, dy, rstd=rstd).reshape(-1, ctx.local_heads, 128)
            if ctx.collective is not None:
                world = ctx.collective.world_size
                step = ctx.collective.max_size_bytes // (ctx.local_heads * 128 * 4)
                if step < 1:
                    raise ValueError("collective capacity cannot hold one Q/K contribution row")
                gathered = []
                for start in range(0, rows.shape[0], step):
                    chunk = rows[start : start + step]
                    value = ctx.collective.all_gather(chunk.reshape(-1, 128).contiguous())
                    gathered.append(
                        value.reshape(world, -1, ctx.local_heads, 128)
                        .permute(1, 0, 2, 3)
                        .reshape(-1, 128)
                    )
                rows = torch.cat(gathered)
            dw = ctx.session.submit_rows(
                ctx.parameter_id,
                ctx.slot,
                rows.reshape(-1, 128),
                lambda ordered: reduce_rows_fp32(ordered).to(weight.dtype),
            )
        record_backward(
            "qwenimage_tp_qk_rmsnorm",
            kernel_id="rl_engine._C.rmsnorm_backward_dx+reduce_rows_fp32_left_fold",
            impl="canonical_global_head_rows",
            family="cuda",
        )
        return dx, dw, None, None, None, None


def replicated_qk_rmsnorm(x, weight, *, logical_keys, parameter_id, collective=None):
    """Normalize local heads and reduce shared gain gradients in global row order."""
    require_extension()
    require_device(x)
    world = 1 if collective is None else collective.world_size
    if (
        world not in (1, 2, 4, 8)
        or x.ndim != 3
        or x.shape[0] < 1
        or x.shape[1:] != (24 // world, 128)
    ):
        raise ValueError("Q/K input must be [M,24/TP,128] for TP1/2/4/8")
    if weight.shape != (128,) or weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("shared gain must have shape [128] on the input device and dtype")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("Q/K norm supports bfloat16 and float32")
    if not torch.is_grad_enabled():
        with torch.cuda.device(x.device):
            return _C.rmsnorm_forward(x.reshape(-1, 128).contiguous(), aligned(weight), 1e-6)[
                0
            ].reshape_as(x)
    session = _session(logical_keys, parameter_id, x.shape[0], x.device)
    previous = None
    if collective is not None:
        if collective.device != x.device or collective.backend_id != "cuda_ipc_fixed_tree":
            raise ValueError(
                "Q/K norm requires the CUDA deterministic collective on the input device"
            )
        if not hasattr(session, "_qwenimage_qknorm_tails"):
            session._qwenimage_qknorm_tails = {}
        previous = session._qwenimage_qknorm_tails.get(collective)
    output = _QKNorm.apply(x, weight, logical_keys, parameter_id, collective, previous)
    if collective is not None:
        session._qwenimage_qknorm_tails[collective] = output
    return output
