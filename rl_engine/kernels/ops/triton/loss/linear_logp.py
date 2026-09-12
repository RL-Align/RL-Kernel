# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

from typing import Any, Optional

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
    _merge_tp_local_logp,
    _require_distributed_initialized,
    _validate_global_targets,
    _validate_tp_vocab_partition_cached,
    chunked_linear_logp_backward,
    should_use_tensor_parallel_linear_logp,
    tensor_parallel_linear_logp,
)

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
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.needs_input_grad[2],
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
        if hidden.device.type not in ("cuda", "xpu", "hip", "musa"):
            raise RuntimeError(
                "TritonLinearLogpOp requires a GPU tensor (CUDA / ROCm / XPU / MUSA), got "
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
        if should_use_tensor_parallel_linear_logp(
            tp_group,
            int(vocab_start_index),
            global_vocab_size,
            lm_head_weight.size(0),
        ):
            return tensor_parallel_linear_logp(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )
        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], got [{t_min}, {t_max}]. "
                "Mask or filter padding / ignore-index values (e.g. -100) before this op."
            )
        return _LinearLogpFunction.apply(hidden, lm_head_weight, bias, target_ids)


# ---------------------------------------------------------------------------
# Deterministic (strict-contract) Triton linear_logp -- the portable analogue
# of ``sm90_deterministic_linear_logp`` (see rl_engine/kernels/ops/cuda/loss/
# linear_logp.py and csrc/cuda/fused_linear_logp_sm90.cu). One source for CUDA
# and ROCm.
#
# Frozen numerical contract (bit-affecting; bump the version to change any):
#   * vocab splitting ....... TRITON_N_SPLIT_CONTRACT fixed splits over
#                             ceil(V / BLOCK_V) tiles, boundaries depend only
#                             on V -- never on batch shape, occupancy, or CU
#                             count
#   * within a split ........ ascending-v0 online-softmax chain over
#                             BLOCK_V-wide tiles; fixed BLOCK_D K-chain inside
#                             tl.dot with IEEE FP32 accumulation
#   * cross-split merge ..... ascending-split sequential scalar chains
#   * padding lanes ......... -inf to max, exp() -> exact 0 to sum
#   * temperature ........... multiplies stats and the selected logit by
#                             1/temperature in the FP32 epilogue, never the
#                             stored logits
#   * final clamp ........... logp = min(zt - lse, 0)
# Row-block size (_DET_BLOCK_N) only moves work along the token axis: rows are
# numerically independent, so it is bit-neutral and may be tuned freely.
# ---------------------------------------------------------------------------

TRITON_LINEAR_LOGP_CONTRACT_VERSION = "triton-fused-linear-logp-contract-v1"
TRITON_N_SPLIT_CONTRACT = 64

_DET_BLOCK_N = 32  # bit-neutral row tile
_DET_BLOCK_V = 64  # contract: vocab tile width
_DET_BLOCK_D = 64  # contract: K-chain step


@triton.jit
def _det_linear_logp_partial_kernel(
    h_ptr,  # hidden [N, D]
    w_ptr,  # lm_head_weight [V, D]
    b_ptr,  # bias [V] (dummy when HAS_BIAS=False)
    t_ptr,  # temperature [N] fp32 (dummy when HAS_TEMP=False)
    tgt_ptr,  # target_ids [N] int64 (global ids)
    logits_ptr,  # unscaled fp32 logits [N, V] (dummy when STORE_LOGITS=False)
    part_max_ptr,  # [n_split, N] fp32
    part_sum_ptr,  # [n_split, N] fp32
    part_zt_ptr,  # [n_split, N] fp32
    N,
    D,
    V,
    n_split,
    vocab_start,
    real_vocab_end,  # local column bound of the real vocabulary
    stride_hn,
    stride_hd,
    stride_wv,
    stride_wd,
    HAS_BIAS: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    STORE_LOGITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (row block, vocab split): fp32 (max, sumexp, zt) partials.

    The split owns a contiguous range of BLOCK_V tiles fixed by (V, n_split)
    alone and folds them in ascending order into an online-softmax state, so
    the reduction tree for any row never depends on the batch shape."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    target = tl.load(tgt_ptr + rows, mask=row_mask, other=-1)

    total_vtiles = tl.cdiv(V, BLOCK_V)
    vtiles_per_split = tl.cdiv(total_vtiles, n_split)
    vt_begin = split * vtiles_per_split
    vt_end = tl.minimum(vt_begin + vtiles_per_split, total_vtiles)

    inv_t = tl.full((BLOCK_N,), 1.0, tl.float32)
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(t_ptr + rows, mask=row_mask, other=1.0)

    m = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK_N,), tl.float32)
    zt = tl.zeros((BLOCK_N,), tl.float32)

    for vt in range(vt_begin, vt_end):
        vcols = vt * BLOCK_V + tl.arange(0, BLOCK_V)
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

        real_mask = vmask & (vcols < real_vocab_end)
        if STORE_LOGITS:
            # [contract] stored logits are unscaled: bias applied, padding -inf,
            # temperature never applied.
            tl.store(
                logits_ptr + rows[:, None].to(tl.int64) * V + vcols[None, :],
                tl.where(real_mask[None, :], acc, float("-inf")),
                mask=row_mask[:, None] & vmask[None, :],
            )

        val = tl.where(real_mask[None, :], acc, float("-inf"))
        if HAS_TEMP:
            val = val * inv_t[:, None]

        is_target = (vcols[None, :].to(tl.int64) + vocab_start) == target[:, None]
        zt += tl.sum(tl.where(is_target & real_mask[None, :], val, 0.0), axis=1)

        tile_max = tl.max(val, axis=1)
        new_m = tl.maximum(m, tile_max)
        finite = new_m != float("-inf")
        alpha = tl.where(finite, tl.exp(m - new_m), 1.0)
        p = tl.where(finite[:, None], tl.exp(val - new_m[:, None]), 0.0)
        s = s * alpha + tl.sum(p, axis=1)
        m = new_m

    base = split.to(tl.int64) * N + rows
    tl.store(part_max_ptr + base, m, mask=row_mask)
    tl.store(part_sum_ptr + base, s, mask=row_mask)
    tl.store(part_zt_ptr + base, zt, mask=row_mask)


@triton.jit
def _det_linear_logp_merge_kernel(
    part_max_ptr,  # [n_split, N] fp32
    part_sum_ptr,
    part_zt_ptr,
    zt_ptr,  # output [N] fp32: selected (scaled) target logit
    lse_ptr,  # output [N] fp32
    N,
    n_split,
    BLOCK_N: tl.constexpr,
):
    """[contract] Ascending-split sequential merge of the fp32 partials."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    rows64 = rows.to(tl.int64)
    m = tl.load(part_max_ptr + rows, mask=row_mask, other=float("-inf"))
    for split in range(1, n_split):
        base = rows64 + split * N
        m = tl.maximum(m, tl.load(part_max_ptr + base, mask=row_mask, other=float("-inf")))

    finite = m != float("-inf")
    s = tl.zeros((BLOCK_N,), tl.float32)
    zt = tl.zeros((BLOCK_N,), tl.float32)
    for split in range(0, n_split):
        base = rows64 + split * N
        pm = tl.load(part_max_ptr + base, mask=row_mask, other=float("-inf"))
        ps = tl.load(part_sum_ptr + base, mask=row_mask, other=0.0)
        term = tl.where(finite & (pm != float("-inf")), ps * tl.exp(pm - m), 0.0)
        s = s + term
        zt = zt + tl.load(part_zt_ptr + base, mask=row_mask, other=0.0)

    lse = m + tl.log(s)
    tl.store(zt_ptr + rows, zt, mask=row_mask)
    tl.store(lse_ptr + rows, lse, mask=row_mask)


def _det_linear_logp_local(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    target_1d: torch.Tensor,
    temperature: Optional[torch.Tensor],
    *,
    vocab_start: int,
    real_vocab_end: int,
    store_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run the frozen-contract kernels on one weight shard.

    Returns ``(zt, lse, logits)``: the temperature-scaled selected-target
    contribution (0 when this shard does not own the target), the shard-local
    log-sum-exp over its real-vocabulary columns (``-inf`` for an all-padding
    shard), and the unscaled fp32 logits when requested."""
    n, d = hidden_2d.shape
    v = weight.shape[0]
    device = hidden_2d.device
    n_split = TRITON_N_SPLIT_CONTRACT
    part = torch.empty(3, n_split, max(n, 1), device=device, dtype=torch.float32)
    zt = torch.empty(n, device=device, dtype=torch.float32)
    lse = torch.empty(n, device=device, dtype=torch.float32)
    logits = torch.empty(n, v, device=device, dtype=torch.float32) if store_logits else None
    if n == 0:
        return zt, lse, logits
    dummy = hidden_2d
    grid = (triton.cdiv(n, _DET_BLOCK_N), n_split)
    _det_linear_logp_partial_kernel[grid](
        hidden_2d,
        weight,
        bias if bias is not None else dummy,
        temperature if temperature is not None else dummy,
        target_1d,
        logits if logits is not None else dummy,
        part[0],
        part[1],
        part[2],
        n,
        d,
        v,
        n_split,
        int(vocab_start),
        int(real_vocab_end),
        hidden_2d.stride(0),
        hidden_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        HAS_BIAS=bias is not None,
        HAS_TEMP=temperature is not None,
        STORE_LOGITS=store_logits,
        BLOCK_N=_DET_BLOCK_N,
        BLOCK_V=_DET_BLOCK_V,
        BLOCK_D=_DET_BLOCK_D,
    )
    _det_linear_logp_merge_kernel[(triton.cdiv(n, 256),)](
        part[0],
        part[1],
        part[2],
        zt,
        lse,
        n,
        n_split,
        BLOCK_N=256,
    )
    return zt, lse, logits


def _det_prepare(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if hidden.device.type not in ("cuda", "hip", "xpu"):
        raise RuntimeError(
            "deterministic Triton linear_logp requires a GPU tensor, got "
            f"device '{hidden.device}'."
        )
    if hidden.dim() < 1:
        raise ValueError("hidden must have at least one dimension")
    if weight.size(-1) != hidden.size(-1):
        raise ValueError(
            f"hidden dim {hidden.size(-1)} must match lm_head_weight dim {weight.size(-1)}"
        )
    hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
    weight_c = weight.contiguous()
    target_1d = target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
    if target_1d.numel() != hidden_2d.size(0):
        raise ValueError("target_ids must have one id per hidden row")
    temp_arg = None
    if temperature is not None:
        temp_arg = temperature.to(device=hidden_2d.device, dtype=torch.float32).reshape(-1)
        if temp_arg.numel() == 1:
            temp_arg = temp_arg.expand(hidden_2d.size(0)).contiguous()
        else:
            temp_arg = temp_arg.contiguous()
        if temp_arg.numel() != hidden_2d.size(0) or bool((temp_arg <= 0).any().item()):
            raise ValueError("temperature must be positive and scalar or per-token")
    return hidden_2d, weight_c, target_1d, temp_arg


def _det_reference_dlogits(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    target_local: torch.Tensor,
    owned: Optional[torch.Tensor],
    lse: torch.Tensor,
    temp: Optional[torch.Tensor],
    grad_logp: Optional[torch.Tensor],
    grad_lse: Optional[torch.Tensor],
    *,
    real_vocab_end: int,
) -> torch.Tensor:
    """FP32 recompute of d(logits) for the strict backward (local shard).

    ``d(logp)/d(logits) = onehot(target) - softmax`` and
    ``d(lse)/d(logits) = softmax`` on the temperature-scaled logits; the chain
    through the scaling divides by the temperature once more at the end."""
    hidden_f = hidden_2d.float()
    weight_f = weight.float()
    logits = torch.nn.functional.linear(
        hidden_f, weight_f, bias.float() if bias is not None else None
    )
    if temp is not None:
        logits = logits / temp.reshape(-1, 1)
    if real_vocab_end < logits.size(1):
        columns = torch.arange(logits.size(1), device=logits.device)
        logits = logits.masked_fill(columns[None, :] >= real_vocab_end, float("-inf"))
    lse_f = lse.reshape(-1, 1).float()
    probs = torch.exp(logits - lse_f)
    logp_grad = torch.zeros_like(lse) if grad_logp is None else grad_logp.reshape(-1).float()
    lse_grad = torch.zeros_like(lse) if grad_lse is None else grad_lse.reshape(-1).float()
    dlogits = probs * (lse_grad - logp_grad).reshape(-1, 1)
    rows = torch.arange(target_local.numel(), device=target_local.device)
    if owned is None:
        dlogits[rows, target_local] += logp_grad
    else:
        hit = rows[owned]
        dlogits[hit, target_local[owned]] += logp_grad[owned]
    if temp is not None:
        dlogits = dlogits / temp.reshape(-1, 1)
    return dlogits


class _DetTritonLinearLogpAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, target, temperature, real_vocab_size):
        hidden_2d, weight_c, target_1d, temp = _det_prepare(hidden, weight, target, temperature)
        real_vocab = weight_c.size(0) if int(real_vocab_size) < 0 else int(real_vocab_size)
        zt, lse, _ = _det_linear_logp_local(
            hidden_2d,
            weight_c,
            bias.contiguous() if bias is not None else None,
            target_1d,
            temp,
            vocab_start=0,
            real_vocab_end=real_vocab,
            store_logits=False,
        )
        logp = torch.minimum(zt - lse, torch.zeros_like(lse))
        ctx.save_for_backward(
            hidden_2d,
            weight_c,
            bias.contiguous() if bias is not None else hidden_2d.new_empty(0),
            target_1d,
            lse,
            temp if temp is not None else hidden_2d.new_empty(0),
        )
        ctx.has_bias = bias is not None
        ctx.real_vocab_end = real_vocab
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(ctx.lead_shape), lse.reshape(ctx.lead_shape)

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        if grad_logp is None and grad_lse is None:
            return (None, None, None, None, None, None)
        hidden_2d, weight, bias, target, lse, temp = ctx.saved_tensors
        with torch.no_grad():
            dlogits = _det_reference_dlogits(
                hidden_2d,
                weight,
                bias if ctx.has_bias else None,
                target,
                None,
                lse,
                temp if temp.numel() else None,
                grad_logp,
                grad_lse,
                real_vocab_end=ctx.real_vocab_end,
            )
            grad_hidden = dlogits.matmul(weight.float())
            grad_weight = dlogits.transpose(0, 1).matmul(hidden_2d.float())
            grad_bias = dlogits.sum(0) if ctx.has_bias else None
        return (
            grad_hidden.reshape((*tuple(ctx.lead_shape), weight.size(1))).to(ctx.hidden_dtype),
            grad_weight.to(ctx.weight_dtype),
            None if grad_bias is None else grad_bias.to(ctx.bias_dtype),
            None,
            None,
            None,
        )


def triton_deterministic_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    temperature: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    real_vocab_size: int = -1,
):
    """Direct forward entry into the frozen-contract Triton fused linear_logp.

    Portable strict-mode boundary (CUDA and ROCm from one kernel source): two
    callers with byte-identical ``hidden`` / ``lm_head_weight`` /
    ``target_ids`` / ``temperature`` and the same
    ``TRITON_LINEAR_LOGP_CONTRACT_VERSION`` receive bitwise-identical FP32
    ``(logp, lse)`` -- for any batch shape that contains the row.

    * ``temperature`` (float32, scalar or one value per token, strictly
      positive) scales the streaming-softmax stats and the selected logit in
      the FP32 epilogue -- the same rounding point on every caller.
    * ``return_logits=True`` additionally returns the unscaled FP32 logits
      (bias applied, padded lanes ``-inf``) without changing the
      ``logp``/``lse`` bytes.
    * ``real_vocab_size`` masks padded vocabulary lanes out of the LSE: lanes
      at or beyond it contribute exactly ``-inf``/``0``. ``-1`` disables
      masking.
    """
    hidden_2d, weight, target_1d, temp_arg = _det_prepare(
        hidden, lm_head_weight, target_ids, temperature
    )
    lead_shape = hidden.shape[:-1]
    real_vocab = weight.size(0) if int(real_vocab_size) < 0 else int(real_vocab_size)
    if not 0 < real_vocab <= weight.size(0):
        raise ValueError(f"real_vocab_size must be in [1, {weight.size(0)}], got {real_vocab_size}")
    if target_1d.numel() and bool(((target_1d < 0) | (target_1d >= real_vocab)).any().item()):
        target_min = int(target_1d.min().item())
        target_max = int(target_1d.max().item())
        raise ValueError(
            f"target_ids must be in the real vocabulary [0, {real_vocab}), "
            f"got [{target_min}, {target_max}]"
        )
    if (
        torch.is_grad_enabled()
        and (
            hidden.requires_grad
            or lm_head_weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )
        and not return_logits
    ):
        return _DetTritonLinearLogpAutograd.apply(
            hidden, lm_head_weight, bias, target_ids, temp_arg, int(real_vocab_size)
        )
    zt, lse, logits = _det_linear_logp_local(
        hidden_2d,
        weight,
        bias.contiguous() if bias is not None else None,
        target_1d,
        temp_arg,
        vocab_start=0,
        real_vocab_end=real_vocab,
        store_logits=return_logits,
    )
    logp = torch.minimum(zt - lse, torch.zeros_like(lse)).reshape(lead_shape)
    lse = lse.reshape(lead_shape)
    if return_logits:
        assert logits is not None
        return logp, lse, logits.reshape(*lead_shape, weight.size(0))
    return logp, lse


def _det_tp_run(
    hidden,
    weight,
    bias,
    target,
    *,
    vocab_start,
    global_vocab,
    real_vocab,
    temperature,
    tp_group,
):
    hidden_2d, weight_c, target_1d, temp = _det_prepare(hidden, weight, target, temperature)
    global_vocab = _validate_tp_vocab_partition_cached(
        tp_group=tp_group,
        device=hidden_2d.device,
        vocab_start_index=int(vocab_start),
        local_vocab_size=weight_c.size(0),
        global_vocab_size=int(global_vocab),
    )
    if not 0 < int(real_vocab) <= global_vocab:
        raise ValueError(f"invalid real_vocab_size={real_vocab} for padded vocab={global_vocab}")
    _validate_global_targets(target_1d, int(real_vocab), tp_group)
    dist = _require_distributed_initialized()
    owners = (
        (target_1d >= int(vocab_start)) & (target_1d < int(vocab_start) + weight_c.size(0))
    ).to(torch.int32)
    dist.all_reduce(owners, op=dist.ReduceOp.SUM, group=tp_group)
    if bool((owners != 1).any().item()):
        raise ValueError("each selected target must have exactly one TP LM-head owner")
    local_real_end = max(0, min(weight_c.size(0), int(real_vocab) - int(vocab_start)))
    local_zt, local_lse, _ = _det_linear_logp_local(
        hidden_2d,
        weight_c,
        bias.contiguous() if bias is not None else None,
        target_1d,
        temp,
        vocab_start=int(vocab_start),
        real_vocab_end=local_real_end,
        store_logits=False,
    )
    # [contract] Rank merge is the shared explicit ascending-rank chain.
    logp, lse = _merge_tp_local_logp(local_lse, local_zt, tp_group=tp_group)
    return logp, lse, hidden_2d, weight_c, target_1d, temp


class _DetTritonTensorParallelLinearLogpAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        bias,
        target,
        vocab_start,
        global_vocab,
        real_vocab,
        temperature,
        tp_group,
    ):
        logp, lse, hidden_2d, weight_c, target_1d, temp = _det_tp_run(
            hidden,
            weight,
            bias,
            target,
            vocab_start=vocab_start,
            global_vocab=global_vocab,
            real_vocab=real_vocab,
            temperature=temperature,
            tp_group=tp_group,
        )
        ctx.save_for_backward(
            hidden_2d,
            weight_c,
            bias.contiguous() if bias is not None else hidden_2d.new_empty(0),
            target_1d,
            lse,
            temp if temp is not None else hidden_2d.new_empty(0),
        )
        ctx.has_bias = bias is not None
        ctx.vocab_start = int(vocab_start)
        ctx.real_vocab = int(real_vocab)
        ctx.tp_group = tp_group
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(ctx.lead_shape), lse.reshape(ctx.lead_shape)

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        if grad_logp is None and grad_lse is None:
            return (None, None, None, None, None, None, None, None, None)
        hidden_2d, weight, bias, target, lse, temp = ctx.saved_tensors
        local_vocab = weight.size(0)
        owned = (target >= ctx.vocab_start) & (target < ctx.vocab_start + local_vocab)
        local_idx = (target - ctx.vocab_start).clamp(0, max(local_vocab - 1, 0))
        local_real_end = max(0, min(local_vocab, ctx.real_vocab - ctx.vocab_start))
        with torch.no_grad():
            dlogits = _det_reference_dlogits(
                hidden_2d,
                weight,
                bias if ctx.has_bias else None,
                local_idx,
                owned,
                lse,
                temp if temp.numel() else None,
                grad_logp,
                grad_lse,
                real_vocab_end=local_real_end,
            )
            grad_hidden = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_hidden = dlogits.matmul(weight.float())
                dist = _require_distributed_initialized()
                dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)
                grad_hidden = grad_hidden.reshape((*tuple(ctx.lead_shape), weight.size(1))).to(
                    ctx.hidden_dtype
                )
            if ctx.needs_input_grad[1]:
                grad_weight = dlogits.transpose(0, 1).matmul(hidden_2d.float()).to(ctx.weight_dtype)
            if ctx.has_bias and ctx.needs_input_grad[2]:
                grad_bias = dlogits.sum(0).to(ctx.bias_dtype)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None


def triton_deterministic_linear_logp_tp(
    hidden: torch.Tensor,
    lm_head_weight_shard: torch.Tensor,
    target_ids: torch.Tensor,
    bias_shard: Optional[torch.Tensor] = None,
    *,
    vocab_start_index: int,
    global_vocab_size: Optional[int] = None,
    real_vocab_size: int = -1,
    temperature: Optional[torch.Tensor] = None,
    tp_group: Any = None,
):
    """Strict tensor-parallel Triton selected logprob.

    Each rank runs the frozen-contract local kernels on its vocab shard and the
    per-rank ``(lse, target-logit)`` stats merge through the shared explicit
    ascending-rank chain, so the result is bitwise-stable for a fixed TP
    topology and identical on every rank. The high-performance fused TP path is
    unchanged."""
    if tp_group is None:
        raise ValueError("strict TP linear_logp requires a TP process group")
    if global_vocab_size is None:
        dist = _require_distributed_initialized()
        global_vocab_size = lm_head_weight_shard.size(0) * dist.get_world_size(tp_group)
    real_vocab = int(global_vocab_size) if int(real_vocab_size) < 0 else int(real_vocab_size)
    if torch.is_grad_enabled() and (
        hidden.requires_grad
        or lm_head_weight_shard.requires_grad
        or (bias_shard is not None and bias_shard.requires_grad)
    ):
        return _DetTritonTensorParallelLinearLogpAutograd.apply(
            hidden,
            lm_head_weight_shard,
            bias_shard,
            target_ids,
            int(vocab_start_index),
            int(global_vocab_size),
            real_vocab,
            temperature,
            tp_group,
        )
    logp, lse, _hidden, _weight, _target, _temp = _det_tp_run(
        hidden,
        lm_head_weight_shard,
        bias_shard,
        target_ids,
        vocab_start=int(vocab_start_index),
        global_vocab=int(global_vocab_size),
        real_vocab=real_vocab,
        temperature=temperature,
        tp_group=tp_group,
    )
    return logp.reshape(hidden.shape[:-1]), lse.reshape(hidden.shape[:-1])
