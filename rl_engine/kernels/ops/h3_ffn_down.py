# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""H3 BF16 down projection: an explicit SM90 reuse candidate for issue #420.

The default reuses the existing fixed-order FP32 Triton tile kernel. The native
BF16 midpoint tree is retained as an explicit legacy comparison backend.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops import base
from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.canonical_backward import active_session

H3_FFN_DIM = 14336
H3_HIDDEN_DIM = 5376
H3_MAX_ROWS = 32768
H3_DOWN_BACKEND = "rlkernel.h3_ffn_down.triton.fp32_tiles.v1"
H3_LEGACY_DOWN_BACKEND = "rlkernel.h3_ffn_down.sm90.midtree.v1"
H3_CHECKPOINT_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    for name, value in (("x", x), ("weight", weight)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must use strided layout")
        if value.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16")
    if x.ndim not in (2, 3) or x.shape[-1] != H3_FFN_DIM:
        raise ValueError("x must have shape [M,14336] or [B,S,14336]")
    if tuple(weight.shape) != (H3_HIDDEN_DIM, H3_FFN_DIM):
        raise ValueError("weight must have native shape [5376,14336]")
    rows = x.numel() // H3_FFN_DIM
    if not 1 <= rows <= H3_MAX_ROWS:
        raise ValueError(f"row count must be in [1,{H3_MAX_ROWS}]")
    if x.device != weight.device:
        raise ValueError("x and weight must share one device")
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("h3_ffn_down_gemm requires NVIDIA CUDA")


def _require_native(device: torch.device):
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("h3_ffn_down_gemm currently supports SM90 only")
    extension = base._C
    marker = getattr(extension, "det_gemm_sm90_compiled", None)
    symbols = ("det_gemm_fwd_rhs_transposed", "det_gemm_fwd", "det_gemm_db_transposed")
    if not base._EXT_AVAILABLE or not callable(marker) or not marker():
        raise RuntimeError("rebuild with KERNEL_ALIGN_DET_GEMM_SM90=1; fallback is forbidden")
    if any(not callable(getattr(extension, symbol, None)) for symbol in symbols):
        raise RuntimeError("native-weight GEMM symbols missing; rebuild the extension")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("disable TF32 before executing the H3 numerical profile")
    return extension


def _require_backend(device, backend=H3_DOWN_BACKEND):
    if backend == H3_LEGACY_DOWN_BACKEND:
        return _require_native(device)
    if backend != H3_DOWN_BACKEND:
        raise ValueError("unsupported H3 backend")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("h3_ffn_down_gemm currently supports SM90 only")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("disable TF32 before executing the H3 numerical profile")
    from rl_engine.kernels.ops.h3_down_fp32_backend import FixedFP32Backend

    return FixedFP32Backend()


def _register_rows(session, weight, keys, rows, parameter_id):
    if not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64:
        raise TypeError("logical_keys must be an int64 Tensor")
    if keys.ndim != 2 or keys.shape[0] != rows or keys.shape[1] not in (2, 3):
        raise ValueError("logical_keys must have shape [rows,2] or [rows,3]")
    if keys.device != weight.device:
        raise ValueError("logical_keys and weight must share one device")
    if not isinstance(parameter_id, str) or not parameter_id.strip():
        raise ValueError("parameter_id must be a nonempty stable parameter name")
    # Own the keys: caller mutation must not change a pending backward graph.
    keys = keys.detach().clone()
    bindings = getattr(session, "_h3_down_bindings", None)
    if bindings is None:
        bindings = {}
        session._h3_down_bindings = bindings
    previous = bindings.get(parameter_id)
    if previous is not None and previous is not weight:
        raise ValueError("parameter_id is already bound to another weight Tensor")
    if any(value is weight and name != parameter_id for name, value in bindings.items()):
        raise ValueError("one weight Tensor must use one parameter_id per session")
    entries = session.uses.get(parameter_id, ())
    if any(entry.keys.shape[1] != keys.shape[1] for entry in entries):
        raise ValueError("logical key width must remain fixed per parameter")
    all_keys = torch.cat([entry.keys for entry in entries] + [keys])
    active = all_keys[all_keys[:, 0] >= 0]
    if active.shape[0] > H3_MAX_ROWS:
        raise ValueError("canonical active row count exceeds the declared H3 row limit")
    if torch.unique(active, dim=0).shape[0] != active.shape[0]:
        raise ValueError("active logical keys must be unique across parameter uses")
    slot = session.register(parameter_id, keys)
    bindings[parameter_id] = weight
    return slot, keys


def _weight_gradient(extension, rows, grads):
    # Fix the token reduction extent from the SAME active logical row set.
    # Pad to BK=32 for a stable tail; also bypass legacy scalar/small-K routes.
    count = rows.shape[0]
    padded = ((count + 31) // 32) * 32
    if padded != count:
        x_pad = rows.new_zeros((padded, rows.shape[1]))
        g_pad = grads.new_zeros((padded, grads.shape[1]))
        x_pad[:count].copy_(rows)
        g_pad[:count].copy_(grads)
        rows, grads = x_pad, g_pad
    return extension.det_gemm_db_transposed(rows.contiguous(), grads.contiguous())


class _H3DownFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, session, slot, keys, parameter_id, trace):
        extension = _require_backend(x.device, trace["requested_backend"])
        flat = x.reshape(-1, H3_FFN_DIM).contiguous()
        native_weight = weight.contiguous()
        # The native extension uses the current device's stream and SM query.
        with torch.cuda.device(x.device):
            y = extension.det_gemm_fwd_rhs_transposed(flat, native_weight)
        # Retain original version counters even when layout normalization copies.
        ctx.save_for_backward(flat, native_weight, keys, x, weight)
        ctx.session, ctx.slot, ctx.parameter_id = session, slot, parameter_id
        ctx.extension, ctx.trace, ctx.input_shape = extension, trace, x.shape
        trace.update(actual_backend=trace["requested_backend"], forward_executed=True)
        trace["kernel_id"] = getattr(
            extension, "h3_kernel_id", "rl_engine._C.det_gemm_fwd_rhs_transposed"
        )
        if hasattr(extension, "h3_last_launch"):
            trace["forward_launch"] = extension.h3_last_launch
        return y.reshape(*x.shape[:-1], H3_HIDDEN_DIM)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, keys, _original_x, _original_weight = ctx.saved_tensors
        grad = grad_output.reshape(-1, H3_HIDDEN_DIM).contiguous()
        if grad.dtype != torch.bfloat16:
            raise TypeError("H3 upstream gradient must have dtype bfloat16")
        if keys is not None and torch.any(grad[keys[:, 0] < 0] != 0).item():
            raise ValueError("padding rows must have zero upstream gradient")
        dx = dw = None
        executed = []
        with torch.cuda.device(x.device):
            if ctx.needs_input_grad[0]:
                dx = ctx.extension.det_gemm_fwd(grad, weight).reshape(ctx.input_shape)
                executed.append(getattr(ctx.extension, "h3_kernel_id", "rl_engine._C.det_gemm_fwd"))
                ctx.trace["input_gradient_executed"] = True
                if hasattr(ctx.extension, "h3_last_launch"):
                    ctx.trace["input_gradient_launch"] = ctx.extension.h3_last_launch
            if ctx.needs_input_grad[1]:

                def reducer(rows, grads):
                    value = _weight_gradient(ctx.extension, rows, grads)
                    executed.append(
                        getattr(
                            ctx.extension, "h3_kernel_id", "rl_engine._C.det_gemm_db_transposed"
                        )
                    )
                    ctx.trace["weight_gradient_executed"] = True
                    ctx.trace["canonical_active_rows"] = rows.shape[0]
                    if hasattr(ctx.extension, "h3_last_launch"):
                        ctx.trace["weight_gradient_launch"] = ctx.extension.h3_last_launch
                    return value

                dw = ctx.session.submit_linear(ctx.parameter_id, ctx.slot, x, grad, reducer)
        if executed:
            record_backward(
                "h3_ffn_down_gemm",
                kernel_id="+".join(executed),
                impl=ctx.trace["requested_backend"],
                family="triton" if ctx.trace["requested_backend"] == H3_DOWN_BACKEND else "cuda",
            )
        return dx, dw, None, None, None, None, None


class H3FFNDownGemmOp:
    """Explicit eager CUDA operator; no backend auto-selection or fallback.

    Weight gradients require one canonical_backward_session spanning ALL
    forward uses and a single backward call. Supply unique, stable logical keys
    [sample_id, token_id] (optionally a third component); first key < 0 is padding.
    """

    def __init__(self, *, backend=H3_DOWN_BACKEND):
        if backend not in (H3_DOWN_BACKEND, H3_LEGACY_DOWN_BACKEND):
            raise ValueError("unsupported H3 backend")
        self.backend = backend
        self.last_execution: dict = {}

    def __call__(self, x, weight, *, logical_keys=None, parameter_id="h3.ffn.down.weight"):
        _validate_inputs(x, weight)
        if torch.compiler.is_compiling():
            raise RuntimeError("this H3 profile currently supports eager execution only")
        _require_backend(x.device, self.backend)
        session, slot, keys = None, None, None
        if torch.is_grad_enabled() and weight.requires_grad:
            session = active_session()
            if session is None:
                raise RuntimeError("weight gradients require an active canonical backward session")
            slot, keys = _register_rows(
                session, weight, logical_keys, x.numel() // H3_FFN_DIM, parameter_id
            )
        trace = {
            "requested_backend": self.backend,
            "actual_backend": None,
            "kernel_id": None,
            "fallback": False,
            "arithmetic": (
                "32-wide FP32 dot tiles, ordered FP32 additions, one BF16 epilogue cast"
                if self.backend == H3_DOWN_BACKEND
                else "32-wide FP32 MMA leaves, BF16 leaves and midpoint tree nodes"
            ),
            "tile": [64, 64, 32] if self.backend == H3_DOWN_BACKEND else [128, 64, 32],
            "num_warps": 4,
            "num_stages": 2,
            "fp_fusion": False if self.backend == H3_DOWN_BACKEND else "native build policy",
            "split_k": False,
            "split_policy": "none",
            "accumulator_dtype": "float32",
            "schedule_id": (
                "fixed-64x64x32-w4-s2-fusion0"
                if self.backend == H3_DOWN_BACKEND
                else "native-midpoint-128x64x32-w4-s2"
            ),
            "topology": {"world_size": 1, "tensor_parallel_size": 1},
            "checkpoint_revision": H3_CHECKPOINT_REVISION,
            "weight_layout": "[5376,14336]",
            "dw_order": "lexicographic active logical keys; trailing zero pad to 32",
            "forward_executed": False,
            "input_gradient_executed": False,
            "weight_gradient_executed": False,
            "qualification": "pending_h3_gpu_evidence",
        }
        self.last_execution = trace
        return _H3DownFn.apply(x, weight, session, slot, keys, parameter_id, trace)


def h3_ffn_down_gemm(
    x, weight, *, logical_keys=None, parameter_id="h3.ffn.down.weight", backend=H3_DOWN_BACKEND
):
    return H3FFNDownGemmOp(backend=backend)(
        x, weight, logical_keys=logical_keys, parameter_id=parameter_id
    )
