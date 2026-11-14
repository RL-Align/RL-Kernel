# SPDX-License-Identifier: Apache-2.0
"""Real vLLM hash-routing primitive with explicit normalized-score input VJP."""

import torch
import torch.nn.functional as F


class _Hash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, tokens, table, scale):
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            vllm_topk_softplus_sqrt,
        )

        if not torch.version.hip or not logits.is_cuda:
            raise RuntimeError("native hash routing requires a ROCm GPU")
        arch = torch.cuda.get_device_properties(logits.device).gcnArchName.split(":")[0]
        if arch != "gfx942":
            raise ValueError("native hash routing profile is validated on gfx942 only")
        if logits.dtype != torch.float32 or logits.ndim != 2 or logits.shape[1] != 256:
            raise ValueError("initial hash route requires original FP32 [M,256] logits")
        if (
            tokens.shape != (logits.shape[0],)
            or table.ndim != 2
            or table.shape[1] != 6
            or table.device != logits.device
            or tokens.device != logits.device
            or tokens.dtype not in (torch.int32, torch.int64)
            or table.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("hash routing requires original colocated integer IDs and Top6 table")
        if not bool(((tokens >= 0) & (tokens < table.shape[0])).all()):
            raise ValueError("token ID outside original hash table")
        selected = table.index_select(0, tokens.long())
        ordered = selected.sort(-1).values
        if not bool(((selected >= 0) & (selected < 256)).all()) or bool(
            (ordered[:, 1:] == ordered[:, :-1]).any()
        ):
            raise ValueError("original hash table must select six distinct valid expert IDs")
        ids = torch.empty((logits.shape[0], 6), device=logits.device, dtype=torch.int32)
        probs = torch.empty_like(ids, dtype=torch.float32)
        token_expert = torch.empty_like(ids)
        vllm_topk_softplus_sqrt(
            probs,
            ids,
            token_expert,
            logits.contiguous(),
            renormalize=True,
            input_tokens=tokens.to(torch.int32).contiguous(),
            hash_indices_table=table.to(torch.int32).contiguous(),
            routed_scaling_factor=scale,
        )
        ctx.save_for_backward(logits, ids)
        ctx.scale = scale
        ctx.mark_non_differentiable(ids)
        return probs, ids

    @staticmethod
    def backward(ctx, grad, _):
        logits, ids = ctx.saved_tensors
        index = ids.long()
        z = logits.gather(1, index)
        score = F.softplus(z).sqrt()
        total = score.sum(-1, keepdim=True)
        centered = grad - (grad * score).sum(-1, keepdim=True) / total
        values = centered * (ctx.scale / total) * torch.sigmoid(z) / (2 * score)
        result = torch.zeros_like(logits).scatter_(1, index, values)
        return result, None, None, None


def native_hash_route(logits, tokens, table, scale):
    return _Hash.apply(logits, tokens, table, float(scale))
