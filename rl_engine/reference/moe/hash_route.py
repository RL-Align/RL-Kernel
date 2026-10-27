# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 Hash weights and fixed-expert VJP; not a FP32 emulator."""

import torch


def hash_route_fp64(logits, token_ids, table, *, renormalize=True, scale=1.5):
    ids = table.long()[token_ids.long()]
    selected = logits.double().gather(1, ids)
    # Conservative mathematical tail representation, separate from production -16.
    scores = torch.where(
        selected < -36,
        torch.exp(selected / 2),
        torch.sqrt(torch.logaddexp(selected, torch.zeros_like(selected))),
    )
    total = scores.sum(dim=1, keepdim=True)
    denominator = total.clamp_min(1e-20) if renormalize else torch.ones_like(total)
    return (scores / denominator) * scale, ids


def hash_route_vjp_fp64(logits, token_ids, table, upstream, *, renormalize=True, scale=1.5):
    ids = table.long()[token_ids.long()]
    x = logits.double().gather(1, ids)
    scores = torch.where(
        x < -36, torch.exp(x / 2), torch.sqrt(torch.logaddexp(x, torch.zeros_like(x)))
    )
    total = scores.sum(dim=1, keepdim=True)
    denominator = total.clamp_min(1e-20) if renormalize else torch.ones_like(total)
    grad = upstream.double() * scale
    dot = (grad * (scores / denominator)).sum(dim=1, keepdim=True)
    adjoint = grad - torch.where(renormalize & (total >= 1e-20), dot, torch.zeros_like(dot))
    safe = torch.where(x < -36, torch.ones_like(scores), scores)
    slope = torch.where(x < -36, 0.5 * scores, torch.sigmoid(x) / (2 * safe))
    selected_grad = (adjoint / denominator) * slope
    result = torch.zeros_like(logits, dtype=torch.float64)
    result.scatter_(1, ids, selected_grad)
    return result
