# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Simulate TP by splitting GEMM K, without a real process group.

TP=2 is one BF16 add of two shards (`a+b == b+a`), so it can match TP=1.
TP=8 left-fold is a different parenthesization from the kernel K-tree, so it
must not match. Real NCCL / custom AllReduce is out of scope here.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.gemm import deterministic_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="det_gemm requires CUDA SM80+",
)

DEV = "cuda"


def _rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


def _k_shards(a: torch.Tensor, b: torch.Tensor, tp: int) -> list[torch.Tensor]:
    k = a.shape[1]
    assert k % tp == 0
    width = k // tp
    return [
        deterministic_gemm(
            a[:, i * width : (i + 1) * width].contiguous(),
            b[i * width : (i + 1) * width].contiguous(),
        )
        for i in range(tp)
    ]


def _left_fold(parts: list[torch.Tensor]) -> torch.Tensor:
    acc = parts[0]
    for part in parts[1:]:
        acc = acc + part
    return acc


# NOTE: these two encode the *CUDA* det_gemm kernel's K-reduction tree. On ROCm
# det_gemm dispatches to TritonDetGemmOp, whose K-tree differs, so a K-split sum
# is not bitwise equal to the unsplit GEMM there. ``get_device_capability()``
# returns (9, 4) on gfx942, so the SM80 guard above does not exclude ROCm.
# Whether the Triton path *should* be K-split invariant is a separate question
# for the det_gemm owners; skipping here does not settle it.
@pytest.mark.cuda_only
def test_simulated_tp2_matches_full():
    # Two shards: AllReduce is a+b, and BF16 add is commutative.
    torch.manual_seed(8)
    m, k, n = 16, 256, 64
    a, b = _rand(m, k), _rand(k, n)
    left, right = _k_shards(a, b, 2)
    full = deterministic_gemm(a, b)
    assert torch.equal(full, left + right)
    assert torch.equal(left + right, right + left)


def test_simulated_tp8_left_fold_does_not_match_full():
    # Eight shards left-folded: ((((s0+s1)+s2)+...)+s7) is not the kernel tree
    # ((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)), so this must diverge.
    torch.manual_seed(8)
    m, k, n = 16, 256, 64
    a, b = _rand(m, k), _rand(k, n)
    full = deterministic_gemm(a, b)
    folded = _left_fold(_k_shards(a, b, 8))
    n_mismatch = int((full != folded).sum().item())
    assert n_mismatch > 0, "TP=8 left-fold unexpectedly matched TP=1"


@pytest.mark.cuda_only
def test_simulated_tp2_is_batch_invariant():
    torch.manual_seed(9)
    k, n = 256, 64
    b = _rand(k, n)
    row = _rand(1, k)
    big = _rand(32, k)
    big[0] = row[0]

    def tp2(x):
        left, right = _k_shards(x, b, 2)
        return left + right

    assert torch.equal(tp2(row)[0], tp2(big)[0])
    assert torch.equal(tp2(row)[0], deterministic_gemm(big, b)[0])
