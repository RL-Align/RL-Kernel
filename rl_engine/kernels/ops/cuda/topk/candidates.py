# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 two-level candidate selection for the sparse-attention indexer.

The candidate-source layer's indexer scores every causal compressed position of a query
row; its positions are grouped in blocks of ``block_size`` (8) and the top
``topk_blocks`` (2048) blocks by their maximum logit are published, the newest block
always kept (vLLM's ``select_candidate_blocks``). Later indexers set their logits outside
those blocks to ``-inf`` before their own top-k (vLLM's ``apply_candidate_mask``).

Every step is exact and row-local: a max, comparisons and masks (no floating-point
accumulation), and the selections run on ``stable_topk_per_row_cuda``, whose key
``(score bits, local index)`` breaks ties toward the larger index. So the result does not
depend on the batch, and a masked dense row selects the same positions as scoring only the
candidates. NaN in a block makes its score NaN, which ranks above every number, as in
vLLM (and ``torch.topk``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .stable_topk import stable_topk_per_row_cuda

BLOCK_SIZE = 8
TOPK_BLOCKS = 2048


def candidate_block_scores(
    logits: Tensor, row_ends: Tensor, block_size: int = BLOCK_SIZE
) -> Tensor:
    """FP32 ``[R, ceil(W / block_size)]``: each block's max over the row's causal columns
    ``[0, row_ends[r])``; ``-inf`` past the end, ``+inf`` for the newest block."""
    rows, width = logits.shape
    nblocks = (width + block_size - 1) // block_size
    cols = torch.arange(nblocks * block_size, device=logits.device)
    ends = row_ends.to(device=logits.device, dtype=torch.long)
    vals = torch.full(
        (rows, nblocks * block_size), -float("inf"), dtype=torch.float32, device=logits.device
    )
    vals[:, :width] = logits.float()
    vals = torch.where(cols[None] < ends[:, None], vals, torch.full_like(vals, -float("inf")))
    scores = vals.view(rows, nblocks, block_size).amax(-1)  # NaN propagates
    newest = ((ends - 1) // block_size).clamp(min=0)
    pin = (ends > 0)[:, None] & (
        torch.arange(nblocks, device=logits.device)[None] == newest[:, None]
    )
    return torch.where(pin, torch.full_like(scores, float("inf")), scores)


def select_candidate_blocks(
    logits: Tensor,
    row_ends: Tensor,
    topk_blocks: int = TOPK_BLOCKS,
    block_size: int = BLOCK_SIZE,
) -> Tensor:
    """int32 ``[R, topk_blocks]`` block ids, best first, ``-1`` padded."""
    scores = candidate_block_scores(logits, row_ends, block_size)
    nvalid = (row_ends.to(device=logits.device, dtype=torch.int32) + block_size - 1) // block_size
    zeros = torch.zeros_like(nvalid)
    return stable_topk_per_row_cuda(scores, zeros, nvalid, topk_blocks)


def mask_to_candidates(
    logits: Tensor, row_ends: Tensor, blocks: Tensor, block_size: int = BLOCK_SIZE
) -> Tensor:
    """``logits`` with every column outside the row's candidate ``blocks`` (int ``[R, K]``,
    ``-1`` padded) or past ``row_ends`` set to ``-inf`` (a new tensor, FP32)."""
    rows, width = logits.shape
    nblocks = (width + block_size - 1) // block_size
    flags = torch.zeros(rows, nblocks + 1, dtype=torch.bool, device=logits.device)
    ids = blocks.long().clone()
    ids[ids < 0] = nblocks
    flags.scatter_(1, ids, True)
    cols = torch.arange(width, device=logits.device)
    keep = flags[:, cols // block_size] & (
        cols[None] < row_ends.to(device=logits.device, dtype=torch.long)[:, None]
    )
    return torch.where(
        keep, logits.float(), torch.full_like(logits, -float("inf"), dtype=torch.float32)
    )
