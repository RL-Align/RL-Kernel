# SPDX-License-Identifier: Apache-2.0
"""INK-09: separate Q/K/V sources and fixed-order projection candidates."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QKVGeometry:
    hidden: int = 2304
    q_heads: int = 24
    kv_heads: int = 8
    head_dim: int = 128

    def __post_init__(self) -> None:
        if min(self.hidden, self.q_heads, self.kv_heads, self.head_dim) <= 0:
            raise ValueError("geometry dimensions must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("Q heads must be divisible by KV heads")


class _FixedFP32Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm

        ctx.save_for_backward(x, weight)
        return _triton_gemm(x, weight.t(), output_dtype=torch.float32).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm

        x, weight = ctx.saved_tensors
        grad = grad_output.float()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _triton_gemm(grad, weight.float(), output_dtype=torch.float32).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = _triton_gemm(grad.t(), x.float(), output_dtype=torch.float32).to(weight.dtype)
        return dx, dw


def project_qkv(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    geometry: QKVGeometry = QKVGeometry(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [T,Q,D], [T,KV,D], [T,KV,D] with one BF16 output cast.

    Distinct normalized Q/K/V streams must remain distinct after loop gates.
    GPU uses the shared fixed-tile FP32 no-split-K Triton GEMM, including VJP;
    CPU uses an explicit FP32 diagnostic reference. Weights may be FP32
    master parameters, but projection inputs and operands are BF16.
    Parameter VJP fixes physical token order only; WS2 owns cross-rank and
    cross-visit reduction. No native-engine parity or throughput is implied.
    """
    inputs = (q_input, k_input, v_input)
    weights = (q_weight, k_weight, v_weight)
    if q_input.ndim != 2 or q_input.shape[1] != geometry.hidden or not q_input.shape[0]:
        raise ValueError("expected nonempty [tokens, hidden] Q/K/V streams")
    if any(x.shape != q_input.shape or x.device != q_input.device for x in inputs):
        raise ValueError("Q/K/V streams must share geometry and device")
    if any(x.dtype != torch.bfloat16 for x in inputs):
        raise TypeError("Q/K/V inputs must be BF16")
    heads = (geometry.q_heads, geometry.kv_heads, geometry.kv_heads)
    result = []
    for x, weight, nheads in zip(inputs, weights, heads, strict=True):
        if weight.shape != (nheads * geometry.head_dim, geometry.hidden):
            raise ValueError("projection weight has incorrect [out, hidden] shape")
        if weight.device != x.device or weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("projection weight must be BF16/FP32 on the input device")
        w = weight.to(torch.bfloat16)
        if x.is_cuda:
            y = _FixedFP32Linear.apply(x, w)
        elif x.device.type == "cpu":
            with torch.autocast(device_type="cpu", enabled=False):
                y = F.linear(x.float(), w.float()).to(torch.bfloat16)
        else:
            raise RuntimeError("only CPU reference and CUDA/ROCm Triton paths are supported")
        result.append(y.reshape(x.shape[0], nheads, geometry.head_dim))
    return result[0], result[1], result[2]
