# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared report type for every T09 ladder/ws2/paired runner.

All step-level runners (L1/L2/L3a/L3b, WS2 rank/cross-config, Torch paired
check) return a :class:`LadderReport` so the CLI and tests can treat them
uniformly. Keeping the type here (not in ``ladder``) avoids a cycle:
``ws2``/``paired_check`` need the report type without importing ladder
runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.first_mismatch import FirstMismatch


@dataclass
class LadderReport:
    """Outcome of one ladder run."""

    ladder: str                       # L1 | L2 | L3a | L3b | WS2-rank | WS2-cross | paired*
    case_id: str
    passed: bool
    verdict: P3Verdict | None         # primary; None iff passed
    detail: str = ""
    first_mismatch: FirstMismatch | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        flag = "PASS" if self.passed else f"FAIL({self.verdict.name if self.verdict else '?'})"
        return f"[{self.ladder}] case={self.case_id} {flag} {self.detail[:120]}"


def make_fail(ladder: str, case_id: str, verdict: P3Verdict, detail: str,
              fm: FirstMismatch | None = None, **extra: Any) -> LadderReport:
    """Build a failing report (strict verdict + optional first mismatch)."""
    return LadderReport(ladder=ladder, case_id=case_id, passed=False,
                        verdict=verdict, detail=detail, first_mismatch=fm,
                        extra=extra)


def make_pass(ladder: str, case_id: str, detail: str = "", **extra: Any) -> LadderReport:
    """Build a passing report (verdict None — pass is not a verdict code)."""
    return LadderReport(ladder=ladder, case_id=case_id, passed=True,
                        verdict=None, detail=detail, extra=extra)
