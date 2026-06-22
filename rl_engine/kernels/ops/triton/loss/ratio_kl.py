# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fused Policy-ratio + KL-penalty Triton operator (PPO/GRPO front-end).
policy_ratio = exp(logp_policy - old_logp)
kl_penalty   = exp(d) - d - 1,   d = logp_ref - logp_policy   (k3 estimator)
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# Tile width
_MAX_BLOCK_V = 2048


@triton.jit
def _ratio_kl_fwd_kernel(
    policy_ptr,
    ref_ptr,
    action_ptr,
    mask_ptr,
    old_ptr,
    ratio_ptr,
    kl_ptr,
    diff_ptr,  # saved for backward: d = logp_ref - logp_policy
    logz_ptr,  # saved for backward: log-sum-exp of policy logits
    V,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    active = tl.load(mask_ptr + row) != 0

    if active:
        row_off = row.to(tl.int64) * V
        a = tl.load(action_ptr + row)

        max_p = -float("inf")
        sum_p = 0.0
        max_r = -float("inf")
        sum_r = 0.0
        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            cmask = cols < V
            p = tl.load(policy_ptr + row_off + cols, mask=cmask, other=-float("inf")).to(tl.float32)
            r = tl.load(ref_ptr + row_off + cols, mask=cmask, other=-float("inf")).to(tl.float32)

            tmax_p = tl.max(p, axis=0)
            nmax_p = tl.maximum(max_p, tmax_p)
            sum_p = sum_p * tl.exp(max_p - nmax_p) + tl.sum(tl.exp(p - nmax_p), axis=0)
            max_p = nmax_p

            tmax_r = tl.max(r, axis=0)
            nmax_r = tl.maximum(max_r, tmax_r)
            sum_r = sum_r * tl.exp(max_r - nmax_r) + tl.sum(tl.exp(r - nmax_r), axis=0)
            max_r = nmax_r

        logz_p = max_p + tl.log(sum_p)
        logz_r = max_r + tl.log(sum_r)
        pa = tl.load(policy_ptr + row_off + a).to(tl.float32)
        ra = tl.load(ref_ptr + row_off + a).to(tl.float32)
        logp_p = pa - logz_p
        logp_r = ra - logz_r

        old = tl.load(old_ptr + row).to(tl.float32)
        ratio = tl.exp(logp_p - old)
        d = logp_r - logp_p
        kl = tl.exp(d) - d - 1.0

        tl.store(ratio_ptr + row, ratio)
        tl.store(kl_ptr + row, kl)
        tl.store(diff_ptr + row, d)
        tl.store(logz_ptr + row, logz_p)
    else:
        # Inactive token
        tl.store(ratio_ptr + row, 1.0)
        tl.store(kl_ptr + row, 0.0)
        tl.store(diff_ptr + row, 0.0)
        tl.store(logz_ptr + row, 0.0)


@triton.jit
def _ratio_kl_bwd_kernel(
    policy_ptr,
    action_ptr,
    mask_ptr,
    ratio_ptr,
    diff_ptr,
    logz_ptr,
    grad_ratio_ptr,
    grad_kl_ptr,
    grad_policy_ptr,  # [N, V] fp32, pre-zeroed
    V,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    active = tl.load(mask_ptr + row) != 0
    if active:
        row_off = row.to(tl.int64) * V
        a = tl.load(action_ptr + row)
        ratio = tl.load(ratio_ptr + row)
        d = tl.load(diff_ptr + row)
        logz = tl.load(logz_ptr + row)
        g_ratio = tl.load(grad_ratio_ptr + row)
        g_kl = tl.load(grad_kl_ptr + row)

        # d(ratio)/d(logp_p) = ratio ; d(kl)/d(logp_p) = 1 - exp(d).
        # Both chain through (1[v==a] - softmax_policy(v)).
        c = g_ratio * ratio + g_kl * (1.0 - tl.exp(d))

        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            cmask = cols < V
            p = tl.load(policy_ptr + row_off + cols, mask=cmask, other=0.0).to(tl.float32)
            soft = tl.exp(p - logz)
            onehot = tl.where(cols == a, 1.0, 0.0)
            grad = c * (onehot - soft)
            tl.store(grad_policy_ptr + row_off + cols, grad, mask=cmask)


class _RatioKLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, policy_logits, ref_logits, action_ids, attention_mask, old_logps):
        if not policy_logits.is_cuda:
            raise RuntimeError("TritonRatioKLOp requires CUDA/ROCm tensors.")
        expected = tuple(policy_logits.shape[:-1])
        for name, t in (
            ("action_ids", action_ids),
            ("attention_mask", attention_mask),
            ("old_logps", old_logps),
        ):
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(t.shape)} does not match "
                    f"policy_logits.shape[:-1] {expected}."
                )
        V = policy_logits.shape[-1]
        pol = policy_logits.contiguous().view(-1, V)
        ref = ref_logits.contiguous().view(-1, V)
        n_rows = pol.shape[0]
        act = action_ids.contiguous().view(-1).clamp(0, V - 1).to(torch.int64)
        mask = attention_mask.contiguous().view(-1).to(torch.int32)
        old = old_logps.contiguous().view(-1).to(torch.float32)

        ratio = torch.empty(n_rows, device=pol.device, dtype=torch.float32)
        kl = torch.empty(n_rows, device=pol.device, dtype=torch.float32)
        diff = torch.empty(n_rows, device=pol.device, dtype=torch.float32)
        logz = torch.empty(n_rows, device=pol.device, dtype=torch.float32)

        block_v = min(_MAX_BLOCK_V, triton.next_power_of_2(V))
        _ratio_kl_fwd_kernel[(n_rows,)](
            pol, ref, act, mask, old, ratio, kl, diff, logz, V, BLOCK_V=block_v
        )

        ctx.save_for_backward(pol, act, mask, ratio, diff, logz)
        ctx.block_v = block_v
        ctx.policy_shape = tuple(policy_logits.shape)
        ctx.policy_dtype = policy_logits.dtype
        ctx.out_shape = tuple(attention_mask.shape)
        return ratio.view(ctx.out_shape), kl.view(ctx.out_shape)

    @staticmethod
    def backward(ctx, grad_ratio, grad_kl):
        pol, act, mask, ratio, diff, logz = ctx.saved_tensors
        n_rows, V = pol.shape
        gr = grad_ratio.contiguous().view(-1).to(torch.float32)
        gk = grad_kl.contiguous().view(-1).to(torch.float32)
        grad_pol = torch.zeros_like(pol, dtype=torch.float32)

        _ratio_kl_bwd_kernel[(n_rows,)](
            pol, act, mask, ratio, diff, logz, gr, gk, grad_pol, V, BLOCK_V=ctx.block_v
        )

        grad_pol = grad_pol.view(ctx.policy_shape).to(ctx.policy_dtype)
        # policy_logits, ref_logits, action_ids, attention_mask, old_logps
        return grad_pol, None, None, None, None


class TritonRatioKLOp:
    """Fused policy-ratio + KL-penalty op (Triton; CUDA & ROCm)."""

    def __call__(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _RatioKLFunction.apply(
            policy_logits, ref_logits, action_ids, attention_mask, old_logps
        )
