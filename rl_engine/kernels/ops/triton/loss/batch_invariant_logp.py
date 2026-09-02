# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_V: int = 1024


@triton.jit
def _batch_invariant_logp_kernel(
    logits_ptr,  # logits [N, V]
    target_ptr,  # target_ids [N]
    output_ptr,  # selected logprob output [N]
    lse_ptr,  # per-row log-sum-exp, saved for backward [N]
    num_tokens,  # N
    vocab_size,  # V
    stride_row,  # stride between consecutive rows in logits
    ignore_index: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program = one token row. Computes selected logprob via online softmax.

    Algorithm (one-pass online log-sum-exp):
        m = -inf, s = 0, z_target = 0
        for each vocab tile [v0, v0+BLOCK_V):
            load tile, cast to fp32
            collect target logit from tile (if target falls in this tile)
            online softmax update: m, s
        lse = log(s) + m
        output = z_target - lse

    Also stores the per-row lse for use by the backward kernel.
    """
    row_id = tl.program_id(0)
    if row_id >= num_tokens:
        return

    target_id = tl.load(target_ptr + row_id)
    is_ignored = target_id == ignore_index
    safe_target = tl.where(is_ignored, 0, target_id)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    z_target = tl.zeros((), dtype=tl.float32)

    row_base = row_id.to(tl.int64) * stride_row

    for v0 in range(0, vocab_size, BLOCK_V):
        cols = v0 + tl.arange(0, BLOCK_V)
        mask = cols < vocab_size

        tile = tl.load(
            logits_ptr + row_base + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

        is_target = (cols == safe_target) & mask
        z_target += tl.sum(tl.where(is_target, tile, 0.0))

        tile_max = tl.max(tile)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m))
        m = new_m

    lse = m + tl.log(s)
    result = z_target - lse
    result = tl.where(is_ignored, 0.0, result)

    tl.store(output_ptr + row_id, result)
    tl.store(lse_ptr + row_id, lse)


@triton.jit
def _batch_invariant_logp_bwd_kernel(
    logits_ptr,  # logits [N, V]
    target_ptr,  # target_ids [N]
    lse_ptr,  # per-row log-sum-exp from forward [N]
    grad_out_ptr,  # upstream gradient [N]
    grad_logits_ptr,  # gradient output for logits [N, V]
    num_tokens,  # N
    vocab_size,  # V
    stride_row,  # stride between consecutive rows in logits / grad_logits
    ignore_index: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    cols = tile_id * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < vocab_size

    target = tl.load(target_ptr + row_id)
    ignored = target == ignore_index

    row_base = row_id.to(tl.int64) * stride_row

    logits = tl.load(
        logits_ptr + row_base + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    lse = tl.load(lse_ptr + row_id)
    grad_out = tl.load(grad_out_ptr + row_id).to(tl.float32)

    soft = tl.exp(logits - lse)
    onehot = tl.where(cols == target, 1.0, 0.0)
    grad = grad_out * (onehot - soft)
    grad = tl.where(ignored, 0.0, grad)

    tl.store(grad_logits_ptr + row_base + cols, grad, mask=mask)


class _BatchInvariantLogpFunction(torch.autograd.Function):
    """Autograd wrapper for the Triton batch-invariant logp kernel."""

    @staticmethod
    def forward(ctx, logits, target_ids, ignore_index):
        lead_shape = logits.shape[:-1]
        vocab_size = logits.size(-1)

        logits_2d = logits.reshape(-1, vocab_size).contiguous()
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.int64).contiguous()

        num_tokens = logits_2d.shape[0]
        output = torch.empty(num_tokens, device=logits.device, dtype=torch.float32)
        lse = torch.empty(num_tokens, device=logits.device, dtype=torch.float32)

        grid = (num_tokens,)
        _batch_invariant_logp_kernel[grid](
            logits_2d,
            target_1d,
            output,
            lse,
            num_tokens,
            vocab_size,
            logits_2d.stride(0),
            ignore_index=ignore_index,
            BLOCK_V=_BLOCK_V,
        )

        ctx.save_for_backward(logits_2d, target_1d, lse)
        ctx.ignore_index = ignore_index
        ctx.lead_shape = lead_shape
        ctx.vocab_size = vocab_size

        return output.reshape(lead_shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits_2d, target_1d, lse = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        vocab_size = ctx.vocab_size
        num_tokens = logits_2d.shape[0]

        grad_flat = grad_output.reshape(-1).contiguous().to(torch.float32)
        grad_logits = torch.empty_like(logits_2d)

        grid = (num_tokens, triton.cdiv(vocab_size, _BLOCK_V))
        _batch_invariant_logp_bwd_kernel[grid](
            logits_2d,
            target_1d,
            lse,
            grad_flat,
            grad_logits,
            num_tokens,
            vocab_size,
            logits_2d.stride(0),
            ignore_index=ignore_index,
            BLOCK_V=_BLOCK_V,
        )

        grad_logits = grad_logits.reshape(ctx.lead_shape + (vocab_size,))

        return grad_logits, None, None


class TritonBatchInvariantLogpOp:
    """Triton fused batch-invariant selected-token log-probability.

    Computes ``logits[t, target_ids[t]] - logsumexp(logits[t, :])`` using a
    one-pass online softmax Triton kernel with locked reduction order.

    Requires a GPU tensor (CUDA / ROCm / XPU / MUSA).
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = False,
    ) -> torch.Tensor:
        return self.apply(logits, target_ids, ignore_index=ignore_index, validate=validate)

    def apply(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = False,
    ) -> torch.Tensor:
        if logits.device.type not in ("cuda", "xpu", "hip", "musa"):
            raise RuntimeError(
                "TritonBatchInvariantLogpOp requires a GPU tensor "
                f"(CUDA / ROCm / XPU / MUSA), got device '{logits.device}'."
            )
        if logits.dim() < 2:
            raise ValueError(
                f"logits must be at least 2-D ([*lead, V]), got shape " f"{tuple(logits.shape)}"
            )
        if logits.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"logits leading shape {tuple(logits.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )

        if validate:
            vocab_size = logits.size(-1)
            target_flat = target_ids.reshape(-1)
            valid_targets = target_flat[target_flat != ignore_index]
            if valid_targets.numel() > 0 and (
                (valid_targets < 0).any() or (valid_targets >= vocab_size).any()
            ):
                bad = valid_targets[(valid_targets < 0) | (valid_targets >= vocab_size)]
                raise ValueError(
                    f"target_ids contains values outside [0, {vocab_size}): " f"{bad.tolist()}"
                )

        return _BatchInvariantLogpFunction.apply(logits, target_ids, ignore_index)
