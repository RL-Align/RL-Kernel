# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the naive total-order Top-K cross-checker.

Three fixture families: random, near-tie, exact-tie. Plus negative
checks that the checker itself fails closed on a tampered candidate,
and K-generality (k != 6 needs no code change).
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # standalone-runnable

from rl_engine.moe.naive_topk import (
    K,
    cross_check_topk,
    naive_topk,
)

E = 256


def _make_q(seed: int, mode: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(64, E, generator=g, dtype=torch.float32)
    if mode == "random":
        return q
    if mode == "near_tie":
        # Distinct FP32 values differing only in low bits: build the top-k
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
        # the top-k boundary, forcing the (id asc) tie-break to matter.
        q = torch.zeros(64, E, dtype=torch.float32)
        for r in range(64):
            v = float(torch.randn(1, generator=g, dtype=torch.float32))
            q[r, :] = v
        return q
    raise ValueError(mode)


class NaiveTopKTests(unittest.TestCase):
    def test_random_rows_match_torch_topk_values(self):
        q = _make_q(seed=1, mode="random")
        ids, values = naive_topk(q)
        # values must be the 6 largest, sorted desc (set-equality with torch.topk)
        tv, _ = torch.topk(q, K, dim=-1)
        self.assertTrue(torch.allclose(values, tv, atol=0, rtol=0))
        # ids strictly descending values
        self.assertTrue(bool((values[:, :-1] >= values[:, 1:]).all()))

    def test_exact_tie_uses_logical_id_ascending(self):
        q = _make_q(seed=2, mode="exact_tie")
        ids, values = naive_topk(q)
        # all-equal row: canonical order must be ids 0..K-1
        self.assertEqual(ids[0].tolist(), list(range(K)))
        self.assertTrue(torch.equal(values[0], q[0, :K]))

    def test_single_row_uses_batch_api(self):
        q = _make_q(seed=3, mode="random")
        ids_b, _ = naive_topk(q)
        for r in (0, 7, 63):
            row_ids, _ = naive_topk(q[r].unsqueeze(0))
            self.assertEqual(row_ids[0].tolist(), ids_b[r].tolist())

    def test_rejects_bad_shape_and_dtype(self):
        with self.assertRaises(ValueError):
            naive_topk(torch.zeros(4, E, dtype=torch.float64))
        with self.assertRaises(ValueError):
            naive_topk(torch.zeros(E, dtype=torch.float32))

    def test_cross_check_passes_on_self(self):
        q = _make_q(seed=4, mode="random")
        ids, _ = naive_topk(q)
        ok, msg = cross_check_topk(ids, q)
        self.assertTrue(ok, msg)

    def test_cross_check_catches_tampered_first_slot(self):
        q = _make_q(seed=5, mode="random")
        ids, _ = naive_topk(q)
        ids[3, 0], ids[3, 5] = ids[3, 5].item(), ids[3, 0].item()  # reorder
        ok, msg = cross_check_topk(ids, q)
        self.assertFalse(ok)
        self.assertIn("row=3", msg)

    def test_cross_check_catches_tie_break_violation(self):
        # exact-tie row: any non-id-ascending order must fail
        q = _make_q(seed=6, mode="exact_tie")
        bad = torch.arange(K, dtype=torch.int32).flip(0).unsqueeze(0).expand(64, K).contiguous()
        ok, msg = cross_check_topk(bad, q)
        self.assertFalse(ok)

    def test_k_is_a_runtime_argument(self):
        # The checker is K-agnostic: k=1, k=8, k=E all work with no code
        # change; K=6 is only the DSV4-Flash convenience default.
        q = _make_q(seed=7, mode="random")
        for k in (1, 2, 8, 16, E):
            ids, values = naive_topk(q, k=k)
            self.assertEqual(ids.shape, (64, k))
            ok, msg = cross_check_topk(ids, q, k=k)
            self.assertTrue(ok, msg)
            tv, _ = torch.topk(q, k, dim=-1)
            self.assertTrue(torch.allclose(values, tv, atol=0, rtol=0))

    def test_rejects_out_of_range_k(self):
        q = _make_q(seed=8, mode="random")
        with self.assertRaises(ValueError):
            naive_topk(q, k=0)
        with self.assertRaises(ValueError):
            naive_topk(q, k=E + 1)

    def test_rejects_nan_input_fail_closed(self):
        """NaN breaks the value ordering (comparisons involving it are False),
        so the "canonical order" of a NaN row is sort-implementation-defined:
        torch.topk groups NaNs by input position, a total sort does not. The
        checker must refuse to judge rather than emit a wrong verdict."""
        q = _make_q(seed=9, mode="random")
        q[5, 7] = float("nan")
        with self.assertRaises(ValueError):
            naive_topk(q)
        # -inf is a legal ordered value: sorts last, never enters top-k
        q2 = _make_q(seed=9, mode="random")
        q2[5, 7] = float("-inf")
        ids, values = naive_topk(q2)
        self.assertNotIn(7, ids[5].tolist())
        tv, _ = torch.topk(q2, K, dim=-1)
        self.assertTrue(torch.allclose(values, tv, atol=0, rtol=0))

    def test_cross_check_refuses_nan_q(self):
        q = _make_q(seed=10, mode="random")
        q[3, 0] = float("nan")
        ids = torch.zeros(64, K, dtype=torch.int32)
        ok, msg = cross_check_topk(ids, q)
        self.assertFalse(ok)
        self.assertIn("NaN", msg)

    def test_cross_check_rejects_non_int32_candidate_ids(self):
        q = _make_q(seed=11, mode="random")
        ids, _ = naive_topk(q)
        ok, msg = cross_check_topk(ids.to(torch.float32), q)
        self.assertFalse(ok)
        self.assertIn("INT32", msg)


if __name__ == "__main__":
    unittest.main()
