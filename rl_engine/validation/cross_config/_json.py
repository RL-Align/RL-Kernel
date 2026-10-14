# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fail-closed JSON decoding shared by configs and artifacts."""

from __future__ import annotations

import json
import math
from typing import Any


def strict_json_loads(value: str) -> Any:
    """Decode RFC JSON while rejecting duplicate keys and non-finite numbers."""

    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_float,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return result


__all__ = ["strict_json_loads"]
