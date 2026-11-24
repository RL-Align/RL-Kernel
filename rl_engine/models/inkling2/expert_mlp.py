# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini expert gate/up arithmetic contract (INK-26)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softcap(value: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply the declared smooth ``cap * tanh(value / cap)`` transform."""
    if cap <= 0:
        raise ValueError("softcap must be positive")
    return float(cap) * torch.tanh(value / float(cap))


def expert_gate_up(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    *,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Compute the FP32 softcapped SwiGLU gate/up projection.

    The public tensor keeps the input dtype, while both linear projections,
    softcaps and elementwise multiplication run in FP32. This gives the CUDA
    and ROCm paths one explicit numerical contract and leaves backward to
    PyTorch autograd.
    """
    if hidden.ndim < 2:
        raise ValueError("expert hidden states must have shape [..., hidden]")
    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate and up weights must share shape [intermediate, hidden]")
    if hidden.shape[-1] != gate_weight.shape[-1]:
        raise ValueError("expert hidden size does not match gate/up weights")
    for name, bias in (("gate", gate_bias), ("up", up_bias)):
        if bias is not None and bias.shape != (gate_weight.shape[0],):
            raise ValueError(f"{name} bias must have shape [intermediate]")
    devices = [hidden.device, gate_weight.device, up_weight.device]
    if gate_bias is not None:
        devices.append(gate_bias.device)
    if up_bias is not None:
        devices.append(up_bias.device)
    if any(device != hidden.device for device in devices):
        raise ValueError("expert gate/up tensors must share a device")

    hidden_fp32 = hidden.float()
    gate = F.linear(
        hidden_fp32, gate_weight.float(), None if gate_bias is None else gate_bias.float()
    )
    up = F.linear(hidden_fp32, up_weight.float(), None if up_bias is None else up_bias.float())
    return (F.silu(softcap(gate, gate_cap)) * softcap(up, up_cap)).to(hidden.dtype)


class _NativeExpertGateUpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, gate_weight, up_weight, gate_bias, up_bias, gate_cap, up_cap):
        from rl_engine.backends.extension import _C

        output, gate_pre, up_pre = _C.inkling2_expert_gate_up_forward(
            hidden.contiguous(),
            gate_weight.contiguous(),
            up_weight.contiguous(),
            gate_bias.contiguous(),
            up_bias.contiguous(),
            float(gate_cap),
            float(up_cap),
        )
        ctx.save_for_backward(
            hidden.contiguous(),
            gate_weight.contiguous(),
            up_weight.contiguous(),
            gate_pre,
            up_pre,
        )
        ctx.gate_cap = float(gate_cap)
        ctx.up_cap = float(up_cap)
        ctx.has_gate_bias = gate_bias.numel() != 0
        ctx.has_up_bias = up_bias.numel() != 0
        ctx.need_gate_bias = ctx.needs_input_grad[3]
        ctx.need_up_bias = ctx.needs_input_grad[4]
        return output

    @staticmethod
    def backward(ctx, dy):
        from rl_engine.backends.extension import _C

        hidden, gate_weight, up_weight, gate_pre, up_pre = ctx.saved_tensors
        d_hidden, d_gate_weight, d_up_weight, d_gate_bias, d_up_bias = (
            _C.inkling2_expert_gate_up_backward(
                dy.contiguous(),
                hidden,
                gate_weight,
                up_weight,
                gate_pre,
                up_pre,
                ctx.gate_cap,
                ctx.up_cap,
                bool(ctx.need_gate_bias),
                bool(ctx.need_up_bias),
            )
        )
        if not ctx.has_gate_bias or not ctx.need_gate_bias:
            d_gate_bias = None
        if not ctx.has_up_bias or not ctx.need_up_bias:
            d_up_bias = None
        return d_hidden, d_gate_weight, d_up_weight, d_gate_bias, d_up_bias, None, None


def expert_gate_up_cuda(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    *,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Run the native CUDA softcapped expert gate/up forward and backward."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_expert_gate_up_forward"):
        raise RuntimeError("Inkling 2 expert gate/up CUDA extension is unavailable")
    if not hidden.is_cuda or hidden.ndim != 2:
        raise ValueError("native expert hidden must be a 2-D CUDA tensor")
    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate and up weights must share shape [intermediate, hidden]")
    if hidden.shape[1] != gate_weight.shape[1]:
        raise ValueError("expert hidden size does not match gate/up weights")
    if hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("native expert gate/up supports float16, bfloat16, and float32")
    if gate_weight.dtype != hidden.dtype or up_weight.dtype != hidden.dtype:
        raise TypeError("native expert gate/up tensors must share dtype")
    if gate_bias is None:
        gate_bias = torch.empty(0, device=hidden.device, dtype=hidden.dtype)
    if up_bias is None:
        up_bias = torch.empty(0, device=hidden.device, dtype=hidden.dtype)
    for name, bias in (("gate", gate_bias), ("up", up_bias)):
        if bias.ndim != 1 or bias.shape[0] not in (0, gate_weight.shape[0]):
            raise ValueError(f"{name} bias must have shape [intermediate]")
        if bias.device != hidden.device or bias.dtype != hidden.dtype:
            raise ValueError("native expert gate/up tensors must share device and dtype")
    if gate_cap <= 0 or up_cap <= 0:
        raise ValueError("softcaps must be positive")
    return _NativeExpertGateUpFunction.apply(
        hidden, gate_weight, up_weight, gate_bias, up_bias, float(gate_cap), float(up_cap)
    )


def expert_gate_up_rocm(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    *,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Run the native ROCm (MI300X-tuned) softcapped expert gate/up forward and backward.

    Same contract and autograd function as the CUDA entry point; the ``_C``
    binding is backed by ``csrc/rocm/inkling2/expert_mlp.hip`` on ROCm builds.
    """
    if torch.version.hip is None:
        raise RuntimeError("expert_gate_up_rocm requires a ROCm (HIP) PyTorch build")
    if not hidden.is_cuda:
        raise ValueError("native expert hidden must be a 2-D ROCm device tensor")
    return expert_gate_up_cuda(
        hidden, gate_weight, up_weight, gate_bias, up_bias, gate_cap=gate_cap, up_cap=up_cap
    )
