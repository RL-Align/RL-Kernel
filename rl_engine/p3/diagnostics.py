"""ULP and selection margins are diagnostics; strict byte verdicts never use them."""

import numpy as np

from .contract import P3Verdict
from .oracle import require

DIAGNOSTIC_SCHEMA = "p3-paired-diagnostic.v2"


def ulp_distance(left, right):
    require(
        left.dtype == right.dtype == np.float32 and left.shape == right.shape,
        P3Verdict.SCHEMA_MISMATCH,
        "ULP diagnostic requires matching FP32 tensors",
    )
    require(
        np.isfinite(left).all() and np.isfinite(right).all(),
        P3Verdict.NON_FINITE,
        "nonfinite diagnostic input",
    )

    def ordered(value):
        bits = value.view(np.uint32).astype(np.int64)
        return np.where(bits & 0x80000000, 0x80000000 - (bits & 0x7FFFFFFF), 0x80000000 + bits)

    return np.abs(ordered(left) - ordered(right))


def paired_diagnostics(trace, score, route, q, active):
    order = np.argsort(-q[active], axis=1, kind="stable")
    ranked = np.take_along_axis(q[active], order, axis=1)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "comparison_mode": "diagnostic_only",
        "strict_comparison": "byte_exact",
        "strict_tolerance_ulp": 0,
        "max_score_ulp": int(ulp_distance(trace["s"][active], score[active]).max()),
        "max_weight_ulp": int(
            ulp_distance(trace["weights"][active], route["weights"][active]).max()
        ),
        "max_abs_score_error": float(np.max(np.abs(trace["s"][active] - score[active]))),
        "ids_match": bool(np.array_equal(trace["ids"][active], route["ids"][active])),
        "cutoff_6_7_margin": ranked[:, 5] - ranked[:, 6],
        "rank_8_9_margin": ranked[:, 7] - ranked[:, 8],
        "cutoff_6_7_ulp": ulp_distance(ranked[:, 5].copy(), ranked[:, 6].copy()),
    }
