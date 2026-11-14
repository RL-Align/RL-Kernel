# SPDX-License-Identifier: Apache-2.0
"""Original learned Top6 probability primitive with selected-score VJP."""

import torch
import torch.nn.functional as F


class _Learned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, bias, scale):
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            vllm_topk_softplus_sqrt,
        )

        if (
            not torch.version.hip
            or not logits.is_cuda
            or logits.dtype != torch.float32
            or logits.ndim != 2
            or logits.shape[1] != 256
        ):
            raise ValueError("learned route requires original ROCm FP32 [M,256] logits")
        if (
            bias is None
            or bias.dtype != torch.float32
            or bias.shape != (256,)
            or bias.requires_grad
            or bias.device != logits.device
        ):
            raise ValueError(
                "selection bias must retain its original non-gradient FP32 representation"
            )
        ids = torch.empty((logits.shape[0], 6), device=logits.device, dtype=torch.int32)
        weights = torch.empty_like(ids, dtype=torch.float32)
        token_expert = torch.empty_like(ids)
        vllm_topk_softplus_sqrt(
            weights,
            ids,
            token_expert,
            logits.contiguous(),
            renormalize=True,
            e_score_correction_bias=bias,
            routed_scaling_factor=scale,
        )
        ctx.save_for_backward(logits, ids)
        ctx.scale = scale
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, grad, _):
        logits, ids = ctx.saved_tensors
        index = ids.long()
        z = logits.gather(1, index)
        score = F.softplus(z).sqrt()
        total = score.sum(-1, keepdim=True)
        values = (
            (grad - (grad * score).sum(-1, keepdim=True) / total)
            * (ctx.scale / total)
            * torch.sigmoid(z)
            / (2 * score)
        )
        return torch.zeros_like(logits).scatter_(1, index, values), None, None


def native_learned_route(logits, bias, scale):
    return _Learned.apply(logits, bias, float(scale))
