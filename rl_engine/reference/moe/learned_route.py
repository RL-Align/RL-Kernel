# SPDX-License-Identifier: Apache-2.0
"""Independent mathematical router/capacity oracle and conditional FP64 VJP."""

import torch


def score_fp64(logits):
    x = logits.double()
    # Mathematical oracle, not a production-FP32 emulator: conservative -36
    # tail rather than production -16, avoiding intermediate FP64 underflow.
    return torch.where(
        x < -36, torch.exp(x / 2), torch.sqrt(torch.logaddexp(x, torch.zeros_like(x)))
    )


def weights_fp64(logits, columns, keep, scale=1.5):
    score = score_fp64(logits).gather(1, columns.long())
    survived = torch.where(keep, score, torch.zeros_like(score))
    denominator = survived.sum(dim=1, keepdim=True).clamp_min(1e-20)
    return survived / denominator * scale


def vjp_fp64(logits, columns, keep, upstream, scale=1.5):
    x = logits.double().gather(1, columns.long())
    score = score_fp64(x)
    survived = torch.where(keep, score, torch.zeros_like(score))
    total = survived.sum(dim=1, keepdim=True)
    denominator = total.clamp_min(1e-20)
    h = upstream.double() * scale
    dot = (h * (survived / denominator)).sum(dim=1, keepdim=True)
    numerator = torch.where(
        keep, h - torch.where(total >= 1e-20, dot, torch.zeros_like(dot)), torch.zeros_like(h)
    )
    safe = torch.where(x < -36, torch.ones_like(score), score)
    slope = torch.where(x < -36, 0.5 * score, torch.sigmoid(x) / (2 * safe))
    selected = (numerator / denominator) * slope
    dx = torch.zeros_like(logits, dtype=torch.float64)
    dx.scatter_(1, columns.long(), selected)
    return dx


def learned_route_reference(
    logits, logical_token_ids, bias, capacities, global_ids=None, scale=1.5
):
    """Normal CPU selection, then independently group tokens by each expert."""
    logits = logits.detach().cpu()
    bias = bias.detach().cpu()
    gids = torch.arange(256) if global_ids is None else global_ids.detach().cpu().long()
    scores = score_fp64(logits)
    choice = scores.float() + bias[gids][None, :]
    columns = torch.tensor(
        [sorted(range(256), key=lambda c: (-float(row[c]), int(gids[c])))[:6] for row in choice],
        dtype=torch.int64,
    ).reshape(-1, 6)
    ids = gids[columns]
    keep = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.full_like(ids, -1)
    token_ids = logical_token_ids.detach().cpu().long()
    # Independent expert-centric capacity oracle, unlike the implementation's
    # ascending-token pass with expert counters.
    for expert in range(256):
        matches = (ids == expert).nonzero().tolist()
        matches.sort(key=lambda pair: int(token_ids[pair[0]]))
        for position, (row, slot) in enumerate(matches[: capacities[expert]]):
            keep[row, slot] = True
            positions[row, slot] = position
    return weights_fp64(logits, columns, keep, scale), ids, columns, keep, positions
