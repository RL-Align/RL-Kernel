# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The pinned reduction trees (issue #2: every P1 reduction goes through these)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.mhc.reduction import (
    fixed_dot,
    fixed_sum,
    fixed_sumsq,
    stream4_max,
    stream4_sum,
    stream4_sum_dim,
)


def test_fixed_sum_is_the_ascending_left_fold() -> None:
    x = torch.tensor([[1e8, 1.0, -1e8, 1.0]], dtype=torch.float32)
    acc = torch.zeros(1, dtype=torch.float32)
    for i in range(4):
        acc = acc + x[:, i]
    assert torch.equal(fixed_sum(x, dim=1), acc)


def test_stream4_sum_is_the_balanced_tree_not_the_left_fold() -> None:
    """``(a0+a1)+(a2+a3)`` and the left fold disagree in FP32 -- that is the point."""
    a = [torch.tensor([v], dtype=torch.float32) for v in (1e-10, 1.0, -1.0, 1e-10)]
    balanced = (a[0] + a[1]) + (a[2] + a[3])
    left_fold = fixed_sum(torch.stack(a, dim=1), dim=1)
    assert torch.equal(stream4_sum(a), balanced)
    assert not torch.equal(balanced, left_fold)


def test_stream4_sum_dim_matches_stream4_sum() -> None:
    x = torch.randn(3, 4, 5, generator=torch.Generator().manual_seed(1))
    got = stream4_sum_dim(x, dim=1)
    want = stream4_sum([x[:, 0], x[:, 1], x[:, 2], x[:, 3]])
    assert torch.equal(got, want)


def test_fixed_dot_matches_a_manual_ascending_k_loop() -> None:
    g = torch.Generator().manual_seed(2)
    a = torch.randn(3, 9, generator=g)
    b = torch.randn(5, 9, generator=g)
    acc = torch.zeros(3, 5)
    for k in range(9):
        acc = acc + a[:, k].unsqueeze(1) * b[:, k].unsqueeze(0)
    assert torch.equal(fixed_dot(a, b), acc)


def test_fixed_dot_is_split_k_free() -> None:
    """A two-half Split-K merge changes bytes, so the reference must not do it."""
    g = torch.Generator().manual_seed(5)
    a = torch.randn(2, 64, generator=g) * 1e4
    b = torch.randn(2, 64, generator=g) * 1e-4
    split = fixed_dot(a[:, :32], b[:, :32]) + fixed_dot(a[:, 32:], b[:, 32:])
    assert not torch.equal(fixed_dot(a, b), split)


def test_fixed_sumsq_rounds_the_square_before_accumulating() -> None:
    x = torch.tensor([[1.0000001, 2.0, 3.0]], dtype=torch.float32)
    acc = torch.zeros(1, dtype=torch.float32)
    for i in range(3):
        acc = acc + x[:, i] * x[:, i]
    assert torch.equal(fixed_sumsq(x, dim=1), acc)


def test_stream4_max_is_the_balanced_tree() -> None:
    x = torch.tensor([[[1.0, 7.0, 3.0, 5.0]]])
    assert torch.equal(stream4_max(x, dim=2), torch.tensor([[7.0]]))


def test_wrong_arity_fails_closed() -> None:
    with pytest.raises(ValueError):
        stream4_sum([torch.zeros(1)] * 3)
    with pytest.raises(ValueError):
        stream4_sum_dim(torch.zeros(2, 5), dim=1)
    with pytest.raises(ValueError):
        fixed_dot(torch.zeros(2, 4), torch.zeros(3, 5))
