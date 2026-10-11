# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native PyTorch reference for the Qwen-Image MLP up projection + GELU (WS1).

The operator is the MMDiT feed-forward up projection of both streams
(``img_mlp.net.0`` / ``img_mlp.net.1`` and ``txt_mlp.net.0`` /
``txt_mlp.net.1``, ``[3072 -> 12288]`` with bias) fused with the
tanh-approximate GELU:

.. code-block:: text

    pre = x @ W.T + b                       x:[M, 3072]  W:[12288, 3072]  b:[12288]
    y   = single_cast(gelu_tanh(pre))

This module is the **independent FP32 CPU reference** the RFC asks for, and it
is also the contract's definition. Frozen reduction order
(``mlp-up-gemm-gelu-tree``) -- the same tree the down projection row froze,
because the operator is the same contraction:

1. the reduction length splits into 32-wide leaves (short tail allowed, missing
   k treated as ``+0.0``);
2. each leaf is an ascending-k FP32 chain from ``+0.0`` in which every
   multiply-into-add is one correctly-rounded FP32 FMA;
3. leaves combine through a mid-split tree ``T(l, r) = T(l, m) + T(m, r)``, so the
   tree depends only on the reduction length and a contiguous half-K split
   composes;
4. bias is added once, in FP32, after the complete tree -- into the fp32
   pre-activation;
5. the GELU is evaluated on that fp32 pre-activation (no intermediate cast, the
   operator is fused), with the frozen fp32 sequence below;
6. the single FP32 -> output-dtype round-to-nearest-even cast happens at the
   store. No other cast exists anywhere in the operator's forward.

The correctly-rounded FMA is emulated in fp64 (Figueroa's double-rounding
theorem makes this exact for the normal range), which is the same arithmetic the
device kernels reach with ``__fmaf_rn``/``fma.rn``.

GELU, frozen fp32 sequence (``S = sqrt(2/pi)`` as fp32, ``C = 0.044715`` as fp32)::

    q  = C * x                      one correctly-rounded multiply
    t  = fma(q, x * x, x)           = x + 0.044715 x^3
    th = tanh(S * t)
    y  = (0.5 * x) * (1 + th)

and its derivative, the free operand of the backward's gate::

    a = 1 + th
    b = 1 - th * th
    e = fma(K3, x * x, 1)           K3 = 0.134145 (= 3C) as fp32
    d = fma(0.5, a, 0.5 * S * x * b * e)

``tanh`` is the operator's only transcendental. The reference evaluates it
correctly rounded (fp64 ``tanh`` rounded once to fp32); the device kernels use
the platform's ``tanhf`` (max error 2 ulp), so the GELU *value* and the gate are
declared-tolerance comparisons, while everything else -- the tree, the
pre-activation, the polynomial, and the three backward contractions -- is
byte-equal to this reference.

Backward, frozen: the gate is the activation's output written at the gradient's
dtype boundary (bf16), exactly as the unfused chain would store it::

    gate = bf16_rne(grad_y * d)
    dx   = tree_gemm(gate, W)       reduction over N = 12288
    dW   = left_fold(gate, x)       ascending-row fp32 left fold
    db   = left_fold(gate)
"""

from __future__ import annotations

import math
import struct
from typing import Optional

import torch

from rl_engine.utils.logger import logger

LEAF_WIDTH = 32
MLP_UP_GEMM_GELU_CONTRACT = "mlp-up-gemm-gelu-tree"


def _f32(value: float) -> float:
    """The fp32 bit pattern of ``value``, as a Python float.

    The device kernels' ``constexpr float`` constants are the same bit patterns,
    so the reference's polynomial sees the same operands the kernels do.
    """

    return struct.unpack("<f", struct.pack("<f", value))[0]


#: GELU constants, as fp32 bit patterns (``S = sqrt(2/pi) == M_SQRT2 * M_2_SQRTPI * 0.5``,
#: the coefficient PyTorch's own tanh GELU uses on both CPU and CUDA; ``K3 = 3 * C``).
GELU_C_X3 = _f32(0.044715)
GELU_S = _f32(math.sqrt(2.0 / math.pi))
GELU_K3 = _f32(0.134145)


def _d(value) -> torch.Tensor:
    """An operand as fp64: tensors through ``.double()``, scalars as a python float."""

    return value.double() if torch.is_tensor(value) else float(value)


def _fma_rn(a, b, c) -> torch.Tensor:
    """One correctly-rounded FP32 FMA, emulated exactly via fp64."""

    return (_d(a) * _d(b) + _d(c)).float()


def _mul_rn(a, b) -> torch.Tensor:
    """One correctly-rounded FP32 multiply."""

    return (_d(a) * _d(b)).float()


def _add_rn(a, b) -> torch.Tensor:
    """One correctly-rounded FP32 add."""

    return (_d(a) + _d(b)).float()


def _sub_rn(a, b) -> torch.Tensor:
    """One correctly-rounded FP32 subtract."""

    return (_d(a) - _d(b)).float()


def _tanh_rn(t: torch.Tensor) -> torch.Tensor:
    """Correctly-rounded fp32 tanh: the fp64 function, rounded once."""

    return t.double().tanh().float()


def tree_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` under the frozen reduction tree, in FP32.

    ``a`` is ``[M, R]``; ``b`` is ``[R, N]``; the reduction runs over ``R`` and
    is split into leaves of :data:`LEAF_WIDTH` in leaf space.
    """

    a = a.float().contiguous()
    b = b.float().contiguous()
    reduction = a.size(1)
    leaves = math.ceil(reduction / LEAF_WIDTH)

    def leaf(index: int) -> torch.Tensor:
        start = index * LEAF_WIDTH
        end = min(start + LEAF_WIDTH, reduction)
        acc = torch.zeros(a.size(0), b.size(1), dtype=torch.float32, device=a.device)
        for k in range(start, end):
            acc = _fma_rn(a[:, k : k + 1], b[k : k + 1, :], acc)
        return acc

    def merge(lo: int, hi: int) -> torch.Tensor:
        if hi - lo == 1:
            return leaf(lo)
        mid = lo + (hi - lo) // 2
        return merge(lo, mid) + merge(mid, hi)

    if leaves == 0:
        return torch.zeros(a.size(0), b.size(1), dtype=torch.float32, device=a.device)
    return merge(0, leaves)


def left_fold_weight_gradient(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``dW``: ascending-row left fold, one correctly-rounded FMA per row."""

    grad = grad.float().contiguous()
    x = x.float().contiguous()
    acc = torch.zeros(grad.size(1), x.size(1), dtype=torch.float32, device=grad.device)
    for row in range(grad.size(0)):
        acc = _fma_rn(grad[row].unsqueeze(1), x[row].unsqueeze(0), acc)
    return acc


def left_fold_bias_gradient(grad: torch.Tensor) -> torch.Tensor:
    """``db``: ascending-row FP32 left fold."""

    grad = grad.float().contiguous()
    acc = torch.zeros(grad.size(1), dtype=torch.float32, device=grad.device)
    for row in range(grad.size(0)):
        acc = acc + grad[row]
    return acc


def gelu_tanh_argument(pre: torch.Tensor) -> torch.Tensor:
    """``S * (x + 0.044715 x^3)`` under the frozen fp32 sequence."""

    pre = pre.float()
    x2 = _mul_rn(pre, pre)
    return _mul_rn(_fma_rn(_mul_rn(pre, GELU_C_X3), x2, pre), GELU_S)


def gelu_tanh_fp32(pre: torch.Tensor) -> torch.Tensor:
    """The tanh-approximate GELU of an fp32 pre-activation, fp32 out."""

    pre = pre.float()
    th = _tanh_rn(gelu_tanh_argument(pre))
    return _mul_rn(_mul_rn(pre, 0.5), _add_rn(th, 1.0))


def gelu_tanh_grad_fp32(pre: torch.Tensor) -> torch.Tensor:
    """The derivative of :func:`gelu_tanh_fp32`, fp32, frozen sequence."""

    pre = pre.float()
    x2 = _mul_rn(pre, pre)
    th = _tanh_rn(gelu_tanh_argument(pre))
    a = _add_rn(th, 1.0)
    b = _sub_rn(1.0, _mul_rn(th, th))
    e = _fma_rn(x2, GELU_K3, 1.0)
    h = _mul_rn(_mul_rn(_mul_rn(pre, GELU_S), b), e)
    return _fma_rn(a, 0.5, _mul_rn(h, 0.5))


def mlp_up_gemm_gelu_reference_pre(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """FP32 tree result plus the bias added once in fp32 (no output cast).

    This is the operator's byte-equality anchor: the device tree kernel
    reproduces this tensor bit for bit.
    """

    pre = tree_gemm(x, weight.t().contiguous())
    if bias is not None:
        pre = pre + bias.float()
    return pre


def mlp_up_gemm_gelu_reference_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The fp32 GELU of the fp32 pre-activation (no output cast)."""

    return gelu_tanh_fp32(mlp_up_gemm_gelu_reference_pre(x, weight, bias))


def mlp_up_gemm_gelu_reference_gate(pre: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """``bf16_rne(grad_y * d)``: the activation's gradient at the dtype boundary.

    The device gate kernel and the Triton gate kernel both produce exactly these
    bytes on the same ``pre``/``grad_output`` (the bf16 cast is the only rounding
    between them and the correctly-rounded fp32 product).
    """

    grad = grad_output.reshape(-1, grad_output.size(-1)).float()
    return (grad * gelu_tanh_grad_fp32(pre.float())).to(torch.bfloat16)


def mlp_up_gemm_gelu_reference_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    pre: torch.Tensor,
    grad_output: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference VJP: ``dx`` reuses the tree, ``dW``/``db`` are left folds.

    ``gate`` overrides the activation gradient -- pass the device's gate to
    check the three contractions in isolation from the transcendental.
    """

    grad = grad_output.reshape(-1, grad_output.size(-1)).float().contiguous()
    x = x.reshape(-1, x.size(-1)).float().contiguous()
    weight = weight.float().contiguous()
    if gate is None:
        gate = mlp_up_gemm_gelu_reference_gate(pre, grad)
    gate = gate.float().contiguous()
    grad_x = tree_gemm(gate, weight)
    grad_w = left_fold_weight_gradient(gate, x)
    grad_b = left_fold_bias_gradient(gate)
    return grad_x, grad_w, grad_b


class _MlpUpGemmGeluTreeFunction(torch.autograd.Function):
    """Dtype path: fp32 tree and GELU internally, one RNE cast at the store."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        pre = mlp_up_gemm_gelu_reference_pre(x2d, weight, bias)
        out = gelu_tanh_fp32(pre).to(x.dtype)
        ctx.lead_shape = x.shape[:-1]
        ctx.save_for_backward(x2d, weight, bias, pre)
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x2d, weight, bias, pre = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.size(-1))
        grad_x, grad_w, grad_b = mlp_up_gemm_gelu_reference_backward(x2d, weight, pre, grad_2d)
        grad_x = grad_x.to(x2d.dtype).reshape(*ctx.lead_shape, weight.size(-1))
        grad_w = grad_w.to(weight.dtype)
        return grad_x, grad_w, None if bias is None else grad_b.to(bias.dtype)


class NativeMlpUpGemmGeluOp:
    """Independent fp32-CPU reference and PyTorch dispatch backend."""

    op_class = "reduction"
    is_batch_invariant = True

    def __init__(self) -> None:
        logger.info("NativeMlpUpGemmGeluOp ready (contract %s).", MLP_UP_GEMM_GELU_CONTRACT)

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, weight, bias=bias)

    def _check(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        if x.size(-1) != weight.size(-1):
            raise ValueError("x K must match weight K")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        if bias is not None and bias.numel() != weight.size(0):
            raise ValueError("bias must have N elements")

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dtype path: returns ``x.dtype`` with the single output cast."""

        self._check(x, weight, bias)
        return _MlpUpGemmGeluTreeFunction.apply(x, weight, bias)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Accuracy gold: the fp32 GELU result, before the output cast."""

        self._check(x, weight, bias)
        return mlp_up_gemm_gelu_reference_forward(x.reshape(-1, x.size(-1)), weight, bias).reshape(
            *x.shape[:-1], weight.size(0)
        )

    def pre_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Accuracy gold: the fp32 pre-activation (the tree, plus bias once)."""

        self._check(x, weight, bias)
        return mlp_up_gemm_gelu_reference_pre(x.reshape(-1, x.size(-1)), weight, bias).reshape(
            *x.shape[:-1], weight.size(0)
        )


def mlp_up_gemm_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convenience wrapper around the PyTorch reference."""

    return NativeMlpUpGemmGeluOp()(x, weight, bias=bias)
