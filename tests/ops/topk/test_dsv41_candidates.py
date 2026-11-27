# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 two-level candidates: block scores, the block selection and the mask agree with
a per-row reference, are batch-invariant, and a masked dense row selects the same
positions as scoring only the candidates."""

from __future__ import annotations

import pytest
import torch


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "stable_topk_per_row_out")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with stable_topk")


def _rows(r, w, seed=0, ties=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(r, w, generator=gen, device="cuda")
    if ties:
        logits = (logits * 4).round() / 4
    ends = torch.randint(0, w + 1, (r,), generator=gen, device="cuda")
    ends[0] = w
    return logits, ends.int()


def test_block_scores_match_reference():
    from rl_engine.kernels.ops.cuda.topk.candidates import candidate_block_scores

    logits, ends = _rows(5, 100, seed=1)
    logits[2, 3] = float("nan")
    ends[2] = 50
    got = candidate_block_scores(logits, ends)
    for r in range(5):
        e = int(ends[r])
        for b in range(got.shape[1]):
            seg = logits[r, b * 8 : min((b + 1) * 8, e)]
            if e > 0 and b == (e - 1) // 8:
                want = float("inf")
            elif seg.numel() == 0:
                want = -float("inf")
            elif bool(seg.isnan().any()):
                want = float("nan")
            else:
                want = float(seg.max())
            g = float(got[r, b])
            assert (g == want) or (g != g and want != want), (r, b, g, want)


@pytest.mark.parametrize("ties", [False, True])
def test_selection_matches_reference_and_is_batch_invariant(ties):
    from rl_engine.kernels.ops.cuda.topk.candidates import (
        candidate_block_scores,
        select_candidate_blocks,
    )

    logits, ends = _rows(9, 4000, seed=2, ties=ties)
    got = select_candidate_blocks(logits, ends, topk_blocks=64)
    scores = candidate_block_scores(logits, ends)
    for r in range(9):
        n = (int(ends[r]) + 7) // 8
        ids = list(range(n))
        ids.sort(key=lambda b: (float(scores[r, b]), b), reverse=True)
        want = ids[:64] + [-1] * (64 - min(64, n))
        assert got[r].tolist() == want, r
        alone = select_candidate_blocks(logits[r : r + 1].clone(), ends[r : r + 1].clone(), 64)
        assert torch.equal(alone[0], got[r])


def test_masked_dense_equals_gathered_candidates():
    from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda
    from rl_engine.kernels.ops.cuda.topk.candidates import (
        mask_to_candidates,
        select_candidate_blocks,
    )

    src, ends = _rows(7, 6000, seed=3)
    consumer = torch.randn_like(src)
    blocks = select_candidate_blocks(src, ends, topk_blocks=40)
    masked = mask_to_candidates(consumer, ends, blocks)
    zeros = torch.zeros_like(ends)
    dense = stable_topk_per_row_cuda(masked, zeros, ends, 512)
    for r in range(7):
        b = blocks[r][blocks[r] >= 0].long()
        cols = (b[:, None] * 8 + torch.arange(8, device="cuda")[None]).flatten()
        cols = cols[cols < int(ends[r])].sort().values
        vals = consumer[r, cols][None].contiguous()
        k = torch.tensor([cols.numel()], dtype=torch.int32, device="cuda")
        sel = stable_topk_per_row_cuda(vals, torch.zeros_like(k), k, 512)[0]
        want = torch.where(sel >= 0, cols[sel.clamp(min=0).long()].int(), sel)
        # Columns outside the candidates are -inf in the dense row; the top-512 of the
        # dense row stops before them when there are at least 512 candidates.
        n = min(512, cols.numel())
        assert torch.equal(dense[r, :n], want[:n]), r
