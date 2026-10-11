# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fixed-order CUDA Qwen-Image modulation projection, RFC #386."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import validate_projection

CONTRACT = "qwenimage-adaln-v2"
_SYMBOLS = (
    "qwenimage_det_linear_forward",
    "qwenimage_det_linear_backward_input",
    "qwenimage_det_linear_backward_weight",
    # Introduced with v2; reject stale v1 extensions with uncompensated dW/db.
    "qwenimage_det_linear_backward_bias",
    "qwenimage_strict_math_compiled",
)


def require_extension():
    if not _EXT_AVAILABLE or not all(hasattr(_C, name) for name in _SYMBOLS):
        raise RuntimeError("rebuild rl_engine._C with Qwen-Image strict CUDA kernels")
    if not _C.qwenimage_strict_math_compiled():
        raise RuntimeError("Qwen-Image requires a build without fast math")


def require_device(tensor):
    if tensor.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Qwen-Image CUDA requires NVIDIA CUDA tensors")
    if torch.cuda.get_device_capability(tensor.device) < (8, 0):
        raise RuntimeError("Qwen-Image CUDA requires SM80 or newer")


def projection_forward(temb, weight, bias):
    act = F.silu(temb).contiguous()
    table = _C.qwenimage_det_linear_forward(act, weight.contiguous(), bias.contiguous(), 0, False)[
        0
    ]
    return table, act


def parameter_gradients(grad, act, dtype):
    return _C.qwenimage_det_linear_backward_weight(grad, act, dtype, True)


def projection_backward(grad, temb, act, weight):
    grad = grad.float().contiguous()
    dw, db = parameter_gradients(grad, act, weight.dtype)
    da = _C.qwenimage_det_linear_backward_input(grad, weight.contiguous(), torch.float32)
    z = temb.float()
    sigmoid = torch.sigmoid(z)
    dt = da * (sigmoid * (1 + z * (1 - sigmoid)))
    return dt.to(temb.dtype), dw, db


def implementation_info(op, tensor):
    require_extension()
    require_device(tensor)
    return {
        "operator": op,
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "contract": CONTRACT,
        "fallback": False,
        "accumulation_dtype": "float32",
        "projection_kernel": "rl_engine._C.qwenimage_det_linear_forward",
        "projection_tree": ("mma_k16_ascending" if tensor.dtype == torch.bfloat16 else "warp_xor"),
        "backward": (
            "compensated_64_column_partials_kahan_fold; "
            "ascending_batch_neumaier_dW_product_residual_db"
        ),
        "parameter_gradient_kernel": "rl_engine._C.qwenimage_det_linear_backward_weight",
        "parameter_gradient_backend": "cuda_compensated",
        "backward_contract": "qwenimage-parameter-grad-neumaier-v2",
        "activation": "aten.silu; fp32 analytic VJP",
        "execution_dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "capability": list(torch.cuda.get_device_capability(tensor.device)),
    }


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, temb, weight, bias):
        table, act = projection_forward(temb, weight, bias)
        ctx.save_for_backward(temb, act, weight)
        return table

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return projection_backward(grad, *ctx.saved_tensors)


class QwenImageAdaLNModProjectionCudaOp:
    op_class = "reduction"
    kernel_id = "aten.silu+rl_engine._C.qwenimage_det_linear_forward"

    def __init__(self):
        require_extension()

    def forward(self, temb, weight, bias):
        validate_projection(temb, weight, bias, 6)
        require_device(temb)
        return _Projection.apply(temb, weight, bias)

    __call__ = forward

    def implementation_info(self, temb, weight, bias):
        validate_projection(temb, weight, bias, 6)
        return implementation_info("qwenimage_adaln_mod_projection", temb)
