# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
    BWD_CHUNK_ELEMS,
    _linear_logits,
    _require_distributed_initialized,
    _use_fp32_matmul,
    _validate_global_targets,
    _validate_tp_vocab_partition,
    chunked_linear_logp_backward,
    should_use_tensor_parallel_linear_logp,
    tensor_parallel_linear_logp,
)
from rl_engine.utils.logger import logger

# Hidden-dim slice the SM90 forward streams per TMA load; D must be a multiple of
# it (mirrors `constexpr int BK` in csrc/cuda/fused_linear_logp_sm90.cu).
_SM90_BK = 32
_SM90_TP_PATH_LOGGED = False


def _sm90_supported(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> bool:
    """Whether the bf16 TMA+MMA forward can run these inputs directly."""
    if not (hidden.is_cuda and lm_head_weight.is_cuda):
        return False
    if hidden.device != lm_head_weight.device:
        return False
    cc_major, _ = torch.cuda.get_device_capability(hidden.device)
    return (
        cc_major == 9
        and hidden.dtype == torch.bfloat16
        and lm_head_weight.dtype == torch.bfloat16
        and hidden.size(-1) % _SM90_BK == 0
    )


def _fallback_op():
    """Portable op for inputs the SM90 forward cannot take (fp32/fp16, or a hidden
    dim not divisible by the kernel's K slice). Prefers Triton, else native."""
    try:
        from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

        return TritonLinearLogpOp()
    except Exception:  # pragma: no cover - Triton missing
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        return NativeLinearLogpOp()


class _FusedLinearLogpSM90Function(torch.autograd.Function):
    """SM90 TMA+WGMMA fused forward + Liger-style chunked backward.

    The forward calls the compiled ``_C.fused_linear_logp_sm90`` kernel (logits
    never materialized). The backward reuses the deterministic chunked cuBLAS
    path so gradients flow into ``hidden``, ``lm_head_weight`` and ``bias``.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.int32).contiguous()
        )
        logp, _lse = _C.fused_linear_logp_sm90(hidden_2d, weight, target_1d, bias)

        ctx.save_for_backward(hidden_2d, weight, bias if bias is not None else hidden_2d, target_1d)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d = ctx.saved_tensors
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


class _FusedTensorParallelLinearLogpSM90Function(torch.autograd.Function):
    """SM90 local-shard forward with tensor-parallel logsumexp reduction.

    Each rank runs the fused SM90 kernel over its local vocab shard to get local
    log-sum-exp and the owned target logit. TP ranks then merge those states into
    the global selected log-prob. Backward intentionally reuses the existing
    chunked TP path so training gradients keep the same contract as the portable
    implementation.
    """

    @staticmethod
    def forward(
        ctx,
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        vocab_start_index,
        global_vocab_size,
        tp_group,
    ):
        dist = _require_distributed_initialized()

        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        bias_t = bias.contiguous() if bias is not None else hidden_2d
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        _validate_global_targets(target_1d, global_vocab_size, tp_group)

        local_vocab = weight.size(0)
        local_target = target_1d - vocab_start_index
        owns_target = (local_target >= 0) & (local_target < local_vocab)
        kernel_target = torch.where(local_target >= 0, local_target, torch.zeros_like(local_target))
        kernel_target = torch.where(
            kernel_target < local_vocab, kernel_target, torch.zeros_like(kernel_target)
        )
        kernel_target = kernel_target.to(torch.int32).contiguous()

        local_logp, local_lse = _C.fused_linear_logp_sm90(hidden_2d, weight, kernel_target, bias)
        local_target_logit = torch.where(
            owns_target, local_logp + local_lse, torch.zeros_like(local_lse)
        )
        target_logit = local_target_logit.clone()
        dist.all_reduce(target_logit, op=dist.ReduceOp.SUM, group=tp_group)

        global_lse_max = local_lse.clone()
        dist.all_reduce(global_lse_max, op=dist.ReduceOp.MAX, group=tp_group)
        global_lse_sum = torch.exp(local_lse - global_lse_max)
        dist.all_reduce(global_lse_sum, op=dist.ReduceOp.SUM, group=tp_group)
        global_lse = global_lse_max + torch.log(global_lse_sum)

        ctx.save_for_backward(hidden_2d, weight, bias_t, target_1d, global_lse)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return (target_logit - global_lse).reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        dist = _require_distributed_initialized()
        hidden_2d, weight, bias_t, target_1d, global_lse = ctx.saved_tensors
        n, d = hidden_2d.shape
        local_vocab = weight.shape[0]
        dt = weight.dtype
        g = grad_logp.reshape(-1).to(torch.float32)

        grad_h = torch.empty_like(hidden_2d, dtype=torch.float32)
        grad_w = torch.zeros(local_vocab, d, device=weight.device, dtype=torch.float32)
        grad_b = (
            torch.zeros(local_vocab, device=weight.device, dtype=torch.float32)
            if ctx.has_bias
            else None
        )
        use_fp32 = _use_fp32_matmul(hidden_2d, weight)

        chunk = max(1, min(n, BWD_CHUNK_ELEMS // local_vocab))
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            x = hidden_2d[i0:i1]
            logits = _linear_logits(
                x,
                weight,
                bias_t if ctx.has_bias else None,
                use_fp32=use_fp32,
            )

            dz = -torch.exp(logits.float() - global_lse[i0:i1].unsqueeze(1))
            local_idx = target_1d[i0:i1] - ctx.vocab_start_index
            owns_target = (local_idx >= 0) & (local_idx < local_vocab)
            if bool(owns_target.any().item()):
                rows = torch.arange(i1 - i0, device=dz.device)[owns_target]
                dz[rows, local_idx[owns_target].long()] += 1.0
            dz *= g[i0:i1].unsqueeze(1)

            if use_fp32:
                grad_h[i0:i1] = torch.matmul(dz, weight.float()).float()
                grad_w += torch.matmul(dz.t(), x.float()).float()
            else:
                dz_dt = dz.to(dt)
                grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()
                grad_w += torch.matmul(dz_dt.t(), x).float()
            if grad_b is not None:
                grad_b += dz.sum(0)

        dist.all_reduce(grad_h, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        grad_hidden = grad_h.to(ctx.hidden_dtype).reshape((*ctx.lead_shape, d))
        grad_weight = grad_w.to(ctx.weight_dtype)
        grad_bias = grad_b.to(ctx.bias_dtype) if grad_b is not None else None
        return grad_hidden, grad_weight, grad_bias, None, None, None, None


def _sm90_tensor_parallel_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    tp_group: Any,
    vocab_start_index: int,
    global_vocab_size: Optional[int],
) -> torch.Tensor:
    global _SM90_TP_PATH_LOGGED
    if not _SM90_TP_PATH_LOGGED:
        logger.info("Using fused_linear_logp_sm90 tensor-parallel local-shard path.")
        _SM90_TP_PATH_LOGGED = True
    return _FusedTensorParallelLinearLogpSM90Function.apply(
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        int(vocab_start_index),
        None if global_vocab_size is None else int(global_vocab_size),
        tp_group,
    )


class FusedLinearLogpSM90Op:
    """SM90 (Hopper) TMA+WGMMA fused linear log-prob.

    Computes ``log_softmax(hidden @ W^T + b)[target]`` without materializing the
    ``[N, V]`` logits. Requires the extension built with ``KERNEL_ALIGN_FORCE_SM90=1``
    on an SM90 device; bfloat16 hidden/weight only.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_linear_logp_sm90"):
            raise RuntimeError(
                "fused_linear_logp_sm90 is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C.fused_linear_logp_sm90 kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        )

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        if lm_head_weight.device != hidden.device:
            raise ValueError(
                f"lm_head_weight device {lm_head_weight.device} must match hidden "
                f"device {hidden.device}"
            )
        n_tokens = hidden.numel() // hidden.size(-1)
        if target_ids.numel() != n_tokens:
            raise ValueError(
                f"target_ids must have one id per token: expected {n_tokens}, "
                f"got {target_ids.numel()}"
            )
        if bias is not None:
            if bias.numel() != lm_head_weight.size(0):
                raise ValueError(
                    f"bias must have V={lm_head_weight.size(0)} elements, got {bias.numel()}"
                )
            if bias.device != hidden.device:
                raise ValueError(
                    f"bias device {bias.device} must match hidden device {hidden.device}"
                )
        if should_use_tensor_parallel_linear_logp(
            tp_group,
            int(vocab_start_index),
            global_vocab_size,
            lm_head_weight.size(0),
        ):
            if _sm90_supported(hidden, lm_head_weight):
                return _sm90_tensor_parallel_linear_logp(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    bias,
                    tp_group=tp_group,
                    vocab_start_index=vocab_start_index,
                    global_vocab_size=global_vocab_size,
                )
            return tensor_parallel_linear_logp(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )
        if not _sm90_supported(hidden, lm_head_weight):
            return _fallback_op()(hidden, lm_head_weight, target_ids, bias)
        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], got [{t_min}, {t_max}]. "
                "Mask or filter padding / ignore-index values (e.g. -100) before this op."
            )
        return _FusedLinearLogpSM90Function.apply(hidden, lm_head_weight, bias, target_ids)
