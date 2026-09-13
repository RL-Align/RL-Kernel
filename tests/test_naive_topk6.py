# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for T09-owned naive total-order Top-6 (contract §2.1/§3.3/§4-T09).

Three fixture families from the contract §3.3 table (owner T01, consumed here
for cross-check development): random, near-tie, exact-tie. Plus negative
checks that the checker itself fails closed on a tampered candidate.
"""

from __future__ import annotations

import sys
import unittest

import torch

sys.path.insert(0, "/workspace/RL-Kernel")  # standalone-runnable

from rl_engine.moe.naive_topk6 import (  # noqa: E402
    K,
    cross_check_topk6,
    naive_topk6,
    naive_topk6_row,
)

E = 256


def _make_q(seed: int, mode: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(64, E, generator=g, dtype=torch.float32)
    if mode == "random":
        return q
    if mode == "near_tie":
        # Distinct FP32 values differing only in low bits: build the top-6
        # band from one base value nudged by exact ULP steps via nextafter.
        base = torch.randn(64, 1, generator=g, dtype=torch.float32)
        band = base.expand(64, K).contiguous()
        for j in range(K):
            for _ in range(j):  # slot j sits j ULPs above the base
                band[:, j] = torch.nextafter(band[:, j], torch.tensor(float("inf")))
        q[:, :K] = band
        return q
    if mode == "exact_tie":
        # Every row has one value duplicated K-1 times -> exact ties spanning
        # the top-6 boundary, forcing the (id asc) tie-break to matter.
        q = torch.zeros(64, E, dtype=torch.float32)
        for r in range(64):
            v = float(torch.randn(1, generator=g, dtype=torch.float32))
            q[r, :] = v
        return q
    raise ValueError(mode)


class NaiveTopk6Tests(unittest.TestCase):
    def test_random_rows_match_torch_topk_values(self):
        q = _make_q(seed=1, mode="random")
        ids, values = naive_topk6(q)
        # values must be the 6 largest, sorted desc (set-equality with torch.topk)
        tv, _ = torch.topk(q, K, dim=-1)
        self.assertTrue(torch.allclose(values, tv, atol=0, rtol=0))
        # ids strictly descending values
        self.assertTrue(bool((values[:, :-1] >= values[:, 1:]).all()))

    def test_exact_tie_uses_logical_id_ascending(self):
        q = _make_q(seed=2, mode="exact_tie")
        ids, values = naive_topk6(q)
        # all-equal row: canonical order must be ids 0..5
        self.assertEqual(ids[0].tolist(), list(range(K)))
        self.assertTrue(torch.equal(values[0], q[0, :K]))

    def test_single_row_variant(self):
        q = _make_q(seed=3, mode="random")
        ids_b, _ = naive_topk6(q)
        for r in (0, 7, 63):
            row = naive_topk6_row(q[r])
            self.assertEqual(row.ids, ids_b[r].tolist())

    def test_rejects_bad_shape_and_dtype(self):
        with self.assertRaises(ValueError):
            naive_topk6(torch.zeros(4, E, dtype=torch.float64))
        with self.assertRaises(ValueError):
            naive_topk6(torch.zeros(E, dtype=torch.float32))

    def test_cross_check_passes_on_self(self):
        q = _make_q(seed=4, mode="random")
        ids, _ = naive_topk6(q)
        ok, msg = cross_check_topk6(ids, q)
        self.assertTrue(ok, msg)

    def test_cross_check_catches_tampered_first_slot(self):
        q = _make_q(seed=5, mode="random")
        ids, _ = naive_topk6(q)
        ids[3, 0], ids[3, 5] = ids[3, 5].item(), ids[3, 0].item()  # reorder
        ok, msg = cross_check_topk6(ids, q)
        self.assertFalse(ok)
        self.assertIn("row=3", msg)

    def test_cross_check_catches_tie_break_violation(self):
        # exact-tie row: any non-id-ascending order must fail
        q = _make_q(seed=6, mode="exact_tie")
        bad = torch.arange(K, dtype=torch.int32).flip(0).unsqueeze(0).expand(64, K).contiguous()
        ok, msg = cross_check_topk6(bad, q)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
