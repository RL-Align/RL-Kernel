# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Test double for T02's canonical sampling record (requested/normalized/materialized/actual)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

DOUBLE_OF = "T02 canonical sampling record and sentinel normalization"


def normalize(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Framework values -> canonical sentinels (DISABLED / IDENTITY); temperature unchanged."""

    top_k = raw.get("top_k")
    top_p = raw.get("top_p")
    min_p = raw.get("min_p")
    return {
        "temperature": 1.0 if raw.get("temperature") is None else float(raw["temperature"]),
        "top_k": "DISABLED" if top_k in (None, -1, 0) else int(top_k),
        "top_p": "IDENTITY" if top_p in (None, 1.0) else float(top_p),
        "min_p": "DISABLED" if min_p in (None, 0.0) else float(min_p),
    }


def record(
    requested: Mapping[str, Any],
    *,
    materialized: Optional[Mapping[str, Any]] = None,
    actual: Optional[Mapping[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    normalized = normalize(requested)
    return {
        "requested": dict(requested),
        "normalized": normalized,
        "materialized": dict(materialized) if materialized is not None else dict(normalized),
        "actual": dict(actual) if actual is not None else dict(normalized),
    }


IDENTITY_REQUEST = {"temperature": 1.0, "top_k": -1, "top_p": 1.0, "min_p": 0.0}
