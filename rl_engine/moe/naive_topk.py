# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Naive total-order Top-K checker.

Purpose
-------
This is the *independent* re-implementation of stable Top-K used **only** to
cross-check a production ``stable_topk``. It is deliberately written in the
most naive way possible — a full total-order ``sorted()`` over all E experts —
so that any race, tie-mishandling or post-tie reorder inside faster
implementations shows up as a diff here.

The algorithm is K-agnostic: ``k`` is a runtime argument, and the DSV4-Flash
router constant is exported solely as a convenience default for that model
family (``K = 6``). Cross-checking a router with a
different K requires no code change here — pass ``k`` through.

Ownership: ``naive total-order Top-K`` is owned by this validation package
and consumed by the golden's acceptance checks. It must NOT be used as a
production selection kernel.

Semantics frozen by the design
------------------------------
* Order key: ``(q descending, logical_expert_id ascending)`` (the router
  canonical tie-break when the engine has no explicit deterministic tie).
* Slot order: slots ``0..k-1`` keep the order produced by the total sort;
  selection computes ``ids = stable_topk(q)`` with q descending and
  logical_expert_id ascending — table slots keep original order, no
  re-sort / re-topk / reorder afterwards.
* Ties: a full total order handles exact ties naturally; near-ties (distinct
  FP32 q values that differ only in low bits) must still sort by value first.
* Non-finite q: a NaN breaks the value ordering (every comparison involving
  it is False), so the "canonical order" of a NaN row is undefined by the
  value key alone and depends on sort implementation details. Production
  kernels order NaN differently (e.g. ``torch.topk`` groups NaNs first by
  input position). Therefore NaN in q is a *contract violation of the
  caller*, not something this checker silently orders: ``naive_topk``
  raises ``ValueError`` so the divergence is never misattributed to a
  tie-break defect. ``-inf`` is a legal finite-ordered value and sorts
  normally (last).
* Padding rows are not selected here — the caller (assembler) owns padding
  canonicalization; this module never sees padding rows.

This module is CPU/FP32 pure-Python by design (auditability over speed): it
operates on Python floats converted from FP32 so the ordering decisions are
made on the exact FP32 bit patterns, not on double-precision artifacts.
"""

from __future__ import annotations

import torch

#: Top-K used by the DSV4-Flash MoE router (K=6). Convenience default only;
#: every API below takes ``k`` as a runtime argument.
K: int = 6

#: Expert count used by the DSV4-Flash MoE router (E=256). Convenience
#: default only; ``E`` is whatever ``q.shape[-1]`` is.
DEFAULT_E: int = 256


def naive_topk(q: torch.Tensor, k: int = K) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive total-order Top-K reference implementation.

    Args:
        q: FP32 ``[T, E]`` selection scores (``q = s + correction_bias``).
        k: number of experts to select (runtime argument; the DSV4-Flash
            router uses ``K = 6``).

    Returns:
        ``(ids, top_values)`` where ``ids`` is INT32 ``[T, k]`` and
        ``top_values`` is FP32 ``[T, k]``, both in canonical order
        (q descending, logical_expert_id ascending), slots keep sort order.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D [T,E], got shape {tuple(q.shape)}")
    if q.dtype != torch.float32:
        raise ValueError(f"q must be FP32, got {q.dtype}")

    t, e = q.shape
    if not 1 <= k <= e:
        raise ValueError(f"k must be in [1, E={e}], got k={k}")
    if torch.isnan(q).any():
        bad = torch.isnan(q).nonzero()
        r, c = bad[0].tolist()
        raise ValueError(
            f"q contains NaN at (row={r}, col={c}); ordering of "
            "NaN is undefined (caller contract violation, fail-closed)"
        )

    rows = q.tolist()

    ids = torch.empty((t, k), dtype=torch.int32)
    top_values = torch.empty((t, k), dtype=torch.float32)
    for r, row in enumerate(rows):
        order = sorted(range(e), key=lambda i: (-row[i], i))
        for slot, idx in enumerate(order[:k]):
            ids[r, slot] = idx
            top_values[r, slot] = row[idx]
    return ids, top_values


def cross_check_topk(
    candidate_ids: torch.Tensor,
    q: torch.Tensor,
    k: int = K,
) -> tuple[bool, str]:
    """Cross-check a candidate Top-K implementation against the naive order.

    This is the cross-check entry point: golden acceptance runs its
    ``stable_topk`` fixtures (random / near-tie / exact-tie) through this
    checker; any mismatch is a defect in the candidate, never in this module.

    Args:
        candidate_ids: INT32 ``[T, k]`` ids produced by the implementation
            under test (slot order must be its own output order).
        q: FP32 ``[T, E]`` the exact scores the candidate was invoked with.
        k: number of experts selected (runtime argument).

    Returns:
        ``(passed, message)``; ``message`` pinpoints the first mismatching
        ``(row, slot)`` with both id sequences — first-mismatch style (the
        first mismatch must be locatable, not averaged away).
    """
    if q.ndim != 2:
        return False, f"q must be 2-D [T,E], got shape {tuple(q.shape)}"
    if q.dtype != torch.float32:
        return False, f"q must be FP32, got {q.dtype}"
    if candidate_ids.dtype != torch.int32:
        return False, f"candidate_ids must be INT32, got {candidate_ids.dtype}"
    if candidate_ids.shape != q.shape[:-1] + (k,):
        return False, (
            f"shape mismatch: candidate_ids {tuple(candidate_ids.shape)} "
            f"vs expected {tuple(q.shape[:-1])}x{k}"
        )
    if torch.isnan(q).any():
        # NaN ordering is undefined (see module docstring); never emit a
        # possibly wrong verdict from an ill-formed input.
        return False, "q contains NaN: ordering undefined, refusing to judge"

    expected_ids, _ = naive_topk(q, k=k)
    t = q.shape[0]

    cand = candidate_ids.to(torch.int64).tolist()
    exp = expected_ids.to(torch.int64).tolist()
    for r in range(t):
        if cand[r] != exp[r]:
            slot = next(i for i in (range(k)) if cand[r][i] != exp[r][i])
            return False, (
                f"first mismatch at (row={r}, slot={slot}): candidate={cand[r]} expected={exp[r]}"
            )
    return True, "ok"
