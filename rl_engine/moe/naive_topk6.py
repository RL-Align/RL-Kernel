# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T09-owned naive total-order Top-6 (contract §2.1 / §4-T09 / fixture table §3.3).

Purpose
-------
This is the *independent* re-implementation of stable Top-6 used **only** to
cross-check T01's ``stable_topk6`` golden. It is deliberately written in the
most naive way possible — a full total-order ``sorted()`` over all E experts —
so that any race, tie-mishandling or post-tie reorder inside faster
implementations shows up as a diff here.

Ownership (contract §3.3): ``naive total-order Top-6`` is owned by **T09** and
consumed by **T01**. It must NOT be used as a production selection kernel.

Semantics frozen by the contract
--------------------------------
* Order key: ``(q descending, logical_expert_id ascending)`` (§2.6: the P3
  canonical tie-break when Miles has no explicit deterministic tie).
* Slot order: slots 0..5 keep the order produced by the total sort; per §2.1
  Learned selection, ``ids = stable_topk6(q)`` with q descending and
  logical_expert_id ascending — six table slots keep original order, no
  re-sort / re-topk / reorder afterwards (that rule is stated for Hash but the
  same slot-order guarantee applies to the Top-6 output consumed by T04).
* Ties: a full total order handles exact ties naturally; near-ties (distinct
  FP32 q values that differ only in low bits) must still sort by value first.
* Padding rows are not selected here — the caller (assembler) owns padding
  canonicalization; this module never sees padding rows.

This module is CPU/FP32 pure-Python by design (auditability over speed): it
operates on Python floats converted from FP32 so the ordering decisions are
made on the exact FP32 bit patterns, not on double-precision artefacts.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

K: int = 6  # contract §2.1: K=6 fixed for DSV4-Flash
DEFAULT_E: int = 256  # contract §2.1: E=256


class NaiveTopk6Result(NamedTuple):
    """Total-order Top-6 output for one row.

    Attributes:
        ids: ``INT32 [K]`` logical expert ids in canonical order
            (q desc, logical_expert_id asc).
        values: ``FP32 [K]`` the q values corresponding to ``ids``.
    """

    ids: list[int]
    values: list[float]


def _fp32_key(value: float) -> tuple[int, float]:
    """Return a sort key that orders FP32 values exactly as FP32 compares.

    Python floats are doubles; two distinct FP32 values remain distinct and
    correctly ordered as doubles (FP32 -> double is exact), so a plain
    descending value sort is already faithful to FP32 semantics. We keep this
    helper to make the intent explicit and to give a single place to audit.
    """
    return (0, value)


def naive_topk6_row(q_row: torch.Tensor) -> NaiveTopk6Result:
    """Total-order Top-6 for a single row of q (FP32 ``[E]``)."""
    if q_row.ndim != 1:
        raise ValueError(f"q_row must be 1-D, got shape {tuple(q_row.shape)}")
    if q_row.numel() == 0:
        raise ValueError("q_row must be non-empty")
    if q_row.dtype != torch.float32:
        raise ValueError(f"q_row must be FP32, got {q_row.dtype}")

    # Work on exact FP32 values pulled out as doubles (lossless widening).
    values = q_row.tolist()
    e = len(values)
    if e < K:
        raise ValueError(f"need at least K={K} experts, got E={e}")

    # Total order: (q desc, logical_expert_id asc). A single sorted() pass over
    # (value, id) pairs with the id as the ascending tie-break achieves this.
    order = sorted(range(e), key=lambda i: (-values[i], i))
    top = order[:K]
    return NaiveTopk6Result(
        ids=[int(i) for i in top],
        values=[float(values[i]) for i in top],
    )


def naive_topk6(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch total-order Top-6.

    Args:
        q: FP32 ``[T, E]`` selection scores (``q = s + correction_bias``).

    Returns:
        ``(ids, top_values)`` where ``ids`` is INT32 ``[T, K]`` and
        ``top_values`` is FP32 ``[T, K]``, both in canonical order
        (q descending, logical_expert_id ascending), slots keep sort order.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D [T,E], got shape {tuple(q.shape)}")
    if q.dtype != torch.float32:
        raise ValueError(f"q must be FP32, got {q.dtype}")

    t, e = q.shape
    if e < K:
        raise ValueError(f"need at least K={K} experts, got E={e}")

    rows = q.tolist()  # exact FP32 -> double widening per element
    ids = torch.empty((t, K), dtype=torch.int32)
    top_values = torch.empty((t, K), dtype=torch.float32)
    for r, row in enumerate(rows):
        order = sorted(range(e), key=lambda i: (-row[i], i))
        for slot, idx in enumerate(order[:K]):
            ids[r, slot] = idx
            top_values[r, slot] = row[idx]
    return ids, top_values


def cross_check_topk6(
    candidate_ids: torch.Tensor,
    q: torch.Tensor,
) -> tuple[bool, str]:
    """Cross-check a candidate Top-6 implementation against the naive order.

    This is the T09 -> T01 cross-check entry point: T01 runs its
    ``stable_topk6`` fixtures (random / near-tie / exact-tie) through this
    checker; any mismatch is a defect in the candidate, never in this module.

    Args:
        candidate_ids: INT32 ``[T, K]`` ids produced by the implementation
            under test (slot order must be its own output order).
        q: FP32 ``[T, E]`` the exact scores the candidate was invoked with.

    Returns:
        ``(passed, message)``; ``message`` pinpoints the first mismatching
        ``(row, slot)`` with both id sequences — first-mismatch style, per
        contract §4-T09 (first mismatch must be locatable, not averaged away).
    """
    if candidate_ids.shape != q.shape[:-1] + (K,):
        return False, (
            f"shape mismatch: candidate_ids {tuple(candidate_ids.shape)} "
            f"vs expected {tuple(q.shape[:-1])}x{K}"
        )

    expected_ids, _ = naive_topk6(q)
    t = q.shape[0]
    cand = candidate_ids.to(torch.int64).tolist()
    exp = expected_ids.to(torch.int64).tolist()
    for r in range(t):
        if cand[r] != exp[r]:
            slot = next(i for i in (range(K)) if cand[r][i] != exp[r][i])
            return False, (
                f"first mismatch at (row={r}, slot={slot}): "
                f"candidate={cand[r]} expected={exp[r]}"
            )
    return True, "ok"
