# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the P3 verdict code table (contract §6, frozen)."""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "/workspace/RL-Kernel")  # standalone-runnable

from rl_engine.moe.p3_verdicts import (  # noqa: E402
    DEVICE_WRITABLE,
    P3Verdict,
    classify_writable_band,
    is_valid_device_status,
    primary_verdict,
)


class VerdictTableTests(unittest.TestCase):
    def test_frozen_values_match_contract(self):
        # Spot-check every value against contract §6 to guard accidental edits.
        expected = {
            "PASS": 0, "NON_FINITE": 1, "HASH_TABLE_INDEX_OUT_OF_RANGE": 2,
            "IDENTITY_DRIFT": 10, "SCHEMA_MISMATCH": 11, "CORRUPT_ARTIFACT": 12,
            "INCOMPLETE_ARTIFACT": 13, "LOGIT_ROUND_POINT_MISMATCH": 14,
            "HASH_TABLE_MISMATCH": 15, "UNSUPPORTED_CAPABILITY": 16,
            "ZERO_ACTIVE_TOKENS": 17, "UPSTREAM_NON_FINITE": 18,
            "GATE_SHARDING_MISMATCH": 19, "MISSING_RANK": 20,
            "PRE_UPDATE_WEIGHT_DRIFT": 21, "STALE_RUN_METADATA": 22,
            "CASE_PASS": 50, "ROUTE_WEIGHT_BYTES_MISMATCH": 51,
            "SCORE_BYTES_MISMATCH": 52, "GRADIENT_BYTES_MISMATCH": 53,
            "BYTE_MISMATCH": 54, "TOPK_ORDER_MISMATCH": 55,
            "TIE_BREAK_POLICY_MISMATCH": 56, "INVALID_DISCRETE_PLAN": 57,
            "INVALID_PROFILE": 58, "ROUTE_SEMANTIC_FINGERPRINT_MISMATCH": 59,
            "ROUTE_ARTIFACT_FINGERPRINT_MISMATCH": 60,
            "SELECTION_GRADIENT_PRESENT": 61, "FORBIDDEN_LOCAL_SHARD_TOPK": 62,
            "AMBIGUOUS_GLOBAL_TOKEN_MAPPING": 63, "INVALID_PLACEMENT_MAP": 64,
            "PLACEMENT_MAP_VERSION_MISMATCH": 65, "SILENT_FALLBACK": 66,
            "MISSING_PROVENANCE": 67, "MISSING_BOUNDARY_TRACE": 68,
            "UPSTREAM_CONTRACT_MISMATCH": 69, "UPSTREAM_VERDICT_MISSING": 70,
            "UPSTREAM_EVIDENCE_MISSING": 71, "NATURAL_ROUTE_MISMATCH": 72,
        }
        for name, value in expected.items():
            self.assertEqual(int(P3Verdict[name]), value, name)

    def test_device_writable_only_two(self):
        self.assertEqual(
            DEVICE_WRITABLE,
            {P3Verdict.NON_FINITE, P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE},
        )

    def test_is_valid_device_status(self):
        self.assertTrue(is_valid_device_status(1))
        self.assertTrue(is_valid_device_status(2))
        for bad in (0, 3, 9, 10, 50, 55, 72, 90, -1, 1000):
            self.assertFalse(is_valid_device_status(bad), bad)

    def test_band_classification(self):
        self.assertEqual(classify_writable_band(10), "provider")
        self.assertEqual(classify_writable_band(22), "provider")
        self.assertEqual(classify_writable_band(50), "runner")
        self.assertEqual(classify_writable_band(72), "runner")
        self.assertEqual(classify_writable_band(90), "reserved")
        self.assertEqual(classify_writable_band(3), "reserved")

    def test_pass_vs_case_pass_not_conflated(self):
        self.assertNotEqual(P3Verdict.PASS, P3Verdict.CASE_PASS)

    def test_primary_verdict_priority(self):
        # identity/schema beats numeric beats fingerprint beats diagnostics
        vs = [
            P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,  # fingerprint
            P3Verdict.NON_FINITE,                            # numeric bytes
            P3Verdict.IDENTITY_DRIFT,                        # identity
            P3Verdict.NATURAL_ROUTE_MISMATCH,                # diagnostics
        ]
        self.assertIs(primary_verdict(vs), P3Verdict.IDENTITY_DRIFT)
        # upstream evidence beats discrete plan
        vs2 = [P3Verdict.TOPK_ORDER_MISMATCH, P3Verdict.UPSTREAM_VERDICT_MISSING]
        self.assertIs(primary_verdict(vs2), P3Verdict.UPSTREAM_VERDICT_MISSING)
        # empty -> None; single -> itself
        self.assertIsNone(primary_verdict([]))
        self.assertIs(primary_verdict([P3Verdict.SILENT_FALLBACK]), P3Verdict.SILENT_FALLBACK)


if __name__ == "__main__":
    unittest.main()
