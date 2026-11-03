# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the router comparison engine.

Covers: fixed stage order, fail-closed identity gate (later stages
unreachable), first-mismatch six-tuple localization, owner attribution,
byte-exactness incl. signed zero, padding exclusion via active mask.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # standalone-runnable

from rl_engine.moe.t01_verdicts import RouterVerdict
from rl_engine.moe.validation.comparison import (
    STAGE_IDENTITY,
    STAGE_DISCRETE,
    STAGE_SCORE_WEIGHT,
    STAGE_GRADIENT,
    TraceComparator,
    tensor_byte_exact,
)
from rl_engine.moe.validation.first_mismatch import (
    MismatchKey,
    TraceEvent,
    UnknownSiteError,
    first_mismatch,
)


def _meta(**overrides):
    base = {
        "case_id": "c1",
        "checkpoint_id": "ckpt-1",
        "weight_id": "w-1",
        "absolute_layer": 3,
        "router_mode": "learned",
        "table_fingerprint": "t-0",
        "bias_fingerprint": "b-0",
        "logit_round_point": "fp32_direct",
        "tie_break_policy": "q_desc_id_asc",
        "capacity_policy": "dropless_v1",
    }
    base.update(overrides)
    return base


def _ev(layer: int, site: str, idx: int, token: int, payload):
    return TraceEvent(
        key=MismatchKey(
            absolute_layer=layer,
            site=site,
            pass_direction="forward",
            event_index=idx,
            global_token_id=token,
            rank=0,
        ),
        payload={"value": payload},
    )


class FirstMismatchTests(unittest.TestCase):
    def test_identical_streams_pass(self):
        a = [_ev(3, "score", 0, 10, 1.0), _ev(3, "topk", 1, 10, [1, 2])]
        res = first_mismatch(a, list(a), phase="WS2", artifact_name="trace-x")
        self.assertFalse(res.found)
        self.assertEqual(res.phase, "WS2")
        self.assertEqual(res.artifact, "trace-x")

    def test_payload_divergence_localizes_six_tuple(self):
        a = [_ev(3, "score", 0, 10, 1.0), _ev(3, "topk", 1, 10, [1, 2, 3])]
        b = [_ev(3, "score", 0, 10, 1.0), _ev(3, "topk", 1, 10, [1, 3, 2])]
        res = first_mismatch(a, b)
        self.assertTrue(res.found)
        self.assertEqual(res.key.site, "topk")
        self.assertEqual(res.key.global_token_id, 10)
        self.assertEqual(res.owner, "stable-topk")
        self.assertEqual(res.issue, "#43")

    def test_length_mismatch_is_found_not_equal(self):
        a = [_ev(3, "score", 0, 10, 1.0)]
        b = [_ev(3, "score", 0, 10, 1.0), _ev(3, "topk", 1, 10, [1])]
        res = first_mismatch(a, b)
        self.assertTrue(res.found)
        self.assertIn("ended early", res.detail)
        self.assertIn("lhs=1 events", res.detail)
        self.assertIn("rhs=2 events", res.detail)
        # the next unmatched event of the longer stream is localized as
        # the first mismatch (fail-closed, never silently truncated)
        self.assertEqual(res.key.site, "topk")
        self.assertEqual(res.owner, "stable-topk")

    def test_key_divergence_owner_matches_reported_key(self):
        # The owner must follow the earlier reported key, not the lhs key.
        lhs = [_ev(3, "topk", 1, 10, [1, 2])]
        rhs = [_ev(3, "score", 0, 10, 1.0)]
        res = first_mismatch(lhs, rhs)
        self.assertTrue(res.found)
        self.assertEqual(res.key.site, "score")
        self.assertEqual(res.owner, "router-score")

    def test_unknown_site_fails_closed(self):
        bad = [TraceEvent(key=MismatchKey(3, "scheduler", "forward", 0, 1, 0), payload={})]
        with self.assertRaises(UnknownSiteError):
            first_mismatch(bad, bad)

    def test_backward_only_on_gradient_sites(self):
        bad = [TraceEvent(key=MismatchKey(3, "topk", "backward", 0, 1, 0), payload={})]
        with self.assertRaises(UnknownSiteError):
            first_mismatch(bad, bad)


class ByteExactTests(unittest.TestCase):
    def test_signed_zero_is_not_equal(self):
        a = torch.tensor([0.0], dtype=torch.float32)
        b = torch.tensor([-0.0], dtype=torch.float32)
        self.assertFalse(tensor_byte_exact(a, b))

    def test_equal_bytes_pass(self):
        a = torch.randn(4, 6, dtype=torch.float32)
        self.assertTrue(tensor_byte_exact(a, a.clone()))

    def test_noncontiguous_equal_bytes_pass(self):
        a = torch.randn(4, 6, dtype=torch.float32).transpose(0, 1)
        self.assertFalse(a.is_contiguous())
        self.assertTrue(tensor_byte_exact(a, a.clone()))

    def test_shape_dtype_mismatch_fails(self):
        a = torch.zeros(4, dtype=torch.float32)
        b = torch.zeros(4, dtype=torch.float64)
        self.assertFalse(tensor_byte_exact(a, b))


class ComparatorOrderTests(unittest.TestCase):
    def test_partial_walk_is_not_a_pass(self):
        cmp = TraceComparator("partial-case")
        cmp.check_identity(_meta(), _meta())
        self.assertFalse(cmp.report().passed)

    def test_full_pass_walks_four_stages(self):
        cmp = TraceComparator("ok-case")
        cmp.check_identity(_meta(), _meta())
        ev = [_ev(3, "topk", 0, 1, [0, 1, 2, 3, 4, 5])]
        cmp.check_discrete(ev, list(ev))
        w = torch.rand(2, 6, dtype=torch.float32)
        cmp.check_score_weight(w, w.clone())
        dz = torch.rand(2, 256, dtype=torch.float32)
        cmp.check_gradient(dz, dz.clone())
        rep = cmp.report()
        self.assertTrue(rep.passed)
        self.assertFalse(rep.stopped_early)
        self.assertEqual(
            [s.stage for s in rep.stages],
            [STAGE_IDENTITY, STAGE_DISCRETE, STAGE_SCORE_WEIGHT, STAGE_GRADIENT],
        )

    def test_identity_drift_halts_walk(self):
        cmp = TraceComparator("drift-case")
        cmp.check_identity(_meta(), _meta(checkpoint_id="ckpt-2"))
        rep = cmp.report()
        self.assertFalse(rep.passed)
        self.assertTrue(rep.stopped_early)
        self.assertEqual(rep.primary, RouterVerdict.IDENTITY_DRIFT)
        self.assertEqual(len(rep.stages), 1)  # later stages never ran
        with self.assertRaises(RuntimeError):
            cmp.check_discrete([], [])  # unreachable

    def test_round_point_policy_mismatch_keeps_dedicated_code(self):
        # both sides auditable but from different policy groups:
        # fp32_direct vs bf16_round_then_widen must not degrade to a
        # generic IDENTITY_DRIFT(10)
        cmp = TraceComparator("round-point-case")
        cmp.check_identity(_meta(), _meta(logit_round_point="bf16_round_then_widen"))
        rep = cmp.report()
        self.assertFalse(rep.passed)
        self.assertTrue(rep.stopped_early)
        self.assertEqual(rep.primary, RouterVerdict.LOGIT_ROUND_POINT_MISMATCH)
        self.assertEqual(len(rep.stages), 1)

    def test_tie_break_policy_mismatch_keeps_dedicated_code(self):
        # tie policy inconsistency keeps its dedicated code (56), not 10
        cmp = TraceComparator("tie-policy-case")
        cmp.check_identity(_meta(), _meta(tie_break_policy="q_desc_id_desc"))
        rep = cmp.report()
        self.assertFalse(rep.passed)
        self.assertTrue(rep.stopped_early)
        self.assertEqual(rep.primary, RouterVerdict.TIE_BREAK_POLICY_MISMATCH)
        self.assertEqual(len(rep.stages), 1)

    def test_missing_identity_field_is_missing_provenance(self):
        lhs = _meta()
        rhs = _meta()
        del rhs["bias_fingerprint"]
        cmp = TraceComparator("missing-case")
        cmp.check_identity(lhs, rhs)
        self.assertEqual(cmp.report().primary, RouterVerdict.MISSING_PROVENANCE)

    def test_discrete_mismatch_attributes_topk_owner(self):
        cmp = TraceComparator("topk-case")
        cmp.check_identity(_meta(), _meta())
        a = [_ev(3, "topk", 0, 7, [1, 2, 3, 4, 5, 6])]
        b = [_ev(3, "topk", 0, 7, [1, 2, 3, 4, 6, 5])]
        cmp.check_discrete(a, b)
        rep = cmp.report()
        # site-specific verdict, not a generic plan code (D2: attribution
        # is never downgraded while a specific code applies)
        self.assertEqual(rep.primary, RouterVerdict.TOPK_ORDER_MISMATCH)
        stage = rep.stages[-1]
        self.assertEqual(stage.mismatch.owner, "stable-topk")

    def test_weight_byte_mismatch_not_hidden_by_mask(self):
        cmp = TraceComparator("w-case")
        cmp.check_identity(_meta(), _meta())
        w1 = torch.zeros(3, 6, dtype=torch.float32)
        w2 = torch.zeros(3, 6, dtype=torch.float32)
        w2[2, 0] = 1e-12  # tiny but byte-different; must not be averaged away
        # all rows active on purpose: strict gate must fire even for tiny diffs
        cmp.check_score_weight(w1, w2, active_mask=None)
        rep = cmp.report()
        self.assertEqual(rep.primary, RouterVerdict.ROUTE_WEIGHT_BYTES_MISMATCH)
        self.assertIn("route_weight", rep.stages[-1].notes[0])

    def test_padding_rows_excluded_from_byte_gate(self):
        cmp = TraceComparator("pad-case")
        cmp.check_identity(_meta(), _meta())
        w1 = torch.zeros(3, 6, dtype=torch.float32)
        w2 = torch.zeros(3, 6, dtype=torch.float32)
        w2[0, 0] = 5.0  # divergence only on the padding row
        mask = torch.tensor([False, True, True], dtype=torch.bool)
        cmp.check_score_weight(w1, w2, active_mask=mask)
        rep = cmp.report()
        # stage 3 passed: padding rows are excluded from numeric comparison
        self.assertTrue(rep.stages[-1].passed)

    def test_gradient_mismatch_attributes_to_router_backward(self):
        cmp = TraceComparator("g-case")
        cmp.check_identity(_meta(), _meta())
        ev = [_ev(3, "score", 0, 1, 1.0)]
        cmp.check_discrete(ev, list(ev))
        w = torch.rand(2, 6, dtype=torch.float32)
        cmp.check_score_weight(w, w.clone())
        dz1 = torch.zeros(2, 256, dtype=torch.float32)
        dz2 = torch.zeros(2, 256, dtype=torch.float32)
        dz2[1, 5] = -0.0  # -0.0 vs +0.0: byte-different
        cmp.check_gradient(dz1, dz2)
        rep = cmp.report()
        self.assertEqual(rep.primary, RouterVerdict.GRADIENT_BYTES_MISMATCH)
        self.assertEqual(rep.stages[-1].mismatch.owner, "router-backward")

    def test_one_sided_score_evidence_fails_closed(self):
        cmp = TraceComparator("missing-score-case")
        cmp.check_identity(_meta(), _meta())
        weights = torch.zeros(2, 6, dtype=torch.float32)
        scores = torch.zeros(2, 6, dtype=torch.float32)
        cmp.check_score_weight(weights, weights.clone(), lhs_scores=scores)
        rep = cmp.report()
        self.assertEqual(rep.primary, RouterVerdict.MISSING_PROVENANCE)
        self.assertTrue(rep.stopped_early)

    def test_multidimensional_selection_gradient_is_located(self):
        cmp = TraceComparator("selection-gradient-case")
        cmp.check_identity(_meta(), _meta())
        dz = torch.zeros(2, 4, dtype=torch.float32)
        selection_grad = torch.zeros(2, 4, dtype=torch.float32)
        selection_grad[1, 2] = 1.0
        cmp.check_gradient(
            dz,
            dz.clone(),
            lhs_selection_grad=selection_grad,
        )
        rep = cmp.report()
        self.assertEqual(rep.primary, RouterVerdict.SELECTION_GRADIENT_PRESENT)
        self.assertEqual(rep.stages[-1].mismatch.key.event_index, 6)

    def test_summary_line_readable(self):
        cmp = TraceComparator("fmt-case")
        cmp.check_identity(_meta(), _meta())
        line = cmp.report().summary_line()
        self.assertIn("identity:ok", line)


if __name__ == "__main__":
    unittest.main()
