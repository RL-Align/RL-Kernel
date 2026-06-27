# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.loss.linear_logp import chunked_linear_logp_backward

# Token / vocab / hidden tile sizes (forward Triton kernel).
_BLOCK_N = 32
_BLOCK_V = 64
_BLOCK_D = 64


@triton.jit
def _linear_logp_fwd_kernel(
    h_ptr,  # hidden [N, D]
    w_ptr,  # lm_head_weight [V, D]
    b_ptr,  # bias [V] (or dummy when HAS_BIAS=False)
    t_ptr,  # target_ids [N]
    logp_ptr,  # output [N]
    lse_ptr,  # output [N], saved for backward
    N,
    D,
    V,
    stride_hn,
    stride_hd,
    stride_wv,
    stride_wd,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per token-block. Streams the vocab in BLOCK_V tiles, folding
    each ``hidden @ Wblk^T`` tile into an online-softmax state without ever
    materializing the full [N, V] logits. Stores logp and the row log-sum-exp."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    target = tl.load(t_ptr + rows, mask=row_mask, other=0).to(tl.int32)

    m = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK_N,), tl.float32)
    z_t = tl.zeros((BLOCK_N,), tl.float32)

    for v0 in range(0, V, BLOCK_V):
        vcols = v0 + tl.arange(0, BLOCK_V)
        vmask = vcols < V

        acc = tl.zeros((BLOCK_N, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            d_mask = offs_d < D
            h = tl.load(
                h_ptr + rows[:, None] * stride_hn + offs_d[None, :] * stride_hd,
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + vcols[:, None] * stride_wv + offs_d[None, :] * stride_wd,
                mask=vmask[:, None] & d_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(h, tl.trans(w), input_precision="ieee")

        if HAS_BIAS:
            acc += tl.load(b_ptr + vcols, mask=vmask, other=0.0).to(tl.float32)[None, :]

        is_t = (vcols[None, :] == target[:, None]) & vmask[None, :]
        z_t += tl.sum(tl.where(is_t, acc, 0.0), axis=1)
        acc = tl.where(vmask[None, :], acc, float("-inf"))

        tile_max = tl.max(acc, axis=1)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(acc - new_m[:, None]), axis=1)
        m = new_m

    lse = m + tl.log(s)
    tl.store(logp_ptr + rows, z_t - lse, mask=row_mask)
    tl.store(lse_ptr + rows, lse, mask=row_mask)


class _LinearLogpFunction(torch.autograd.Function):
    """Autograd wrapper: fused forward + recompute-based backward."""

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.int32).contiguous()
        )
        n, d = hidden_2d.shape
        v = weight.shape[0]

        logp = torch.empty(n, device=hidden_2d.device, dtype=torch.float32)
        lse = torch.empty(n, device=hidden_2d.device, dtype=torch.float32)
        bias_t = bias.contiguous() if bias is not None else hidden_2d  # dummy ptr when no bias

        grid = (triton.cdiv(n, _BLOCK_N),)
        _linear_logp_fwd_kernel[grid](
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            logp,
            lse,
            n,
            d,
            v,
            hidden_2d.stride(0),
            hidden_2d.stride(1),
            weight.stride(0),
            weight.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_N=_BLOCK_N,
            BLOCK_V=_BLOCK_V,
            BLOCK_D=_BLOCK_D,
        )

        ctx.save_for_backward(hidden_2d, weight, bias_t, target_1d, lse)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d, _lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = chunked_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            target_1d,
            bias_t,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
        )
        # Inputs: hidden, lm_head_weight, bias, target_ids.
        return grad_hidden, grad_weight, grad_bias, None


class TritonLinearLogpOp:
    """Triton fused linear log-prob op.

    Computes per-token ``log_softmax(hidden @ W^T + b)[target]`` without
    materializing the ``[N, V]`` logits: the forward streams the vocab through an
    online softmax, the backward recomputes the logit tiles instead of storing
    them. Differentiable w.r.t. ``hidden``, ``lm_head_weight`` and ``bias``.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(hidden, lm_head_weight, target_ids, bias)

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.device.type not in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                "TritonLinearLogpOp requires a GPU tensor (CUDA / ROCm / XPU), got "
                f"device '{hidden.device}'."
            )
        if hidden.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], got [{t_min}, {t_max}]. "
                "Mask or filter padding / ignore-index values (e.g. -100) before this op."
            )
        return _LinearLogpFunction.apply(hidden, lm_head_weight, bias, target_ids)
