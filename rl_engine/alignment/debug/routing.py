# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Canonical native MoE dispatch evidence, independent of all-to-all ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .core import bitdiff, snapshot


@dataclass(frozen=True)
class RoutedTensor:
    """Rows identified by (sample, token, top-k slot, GLOBAL expert ID)."""

    value: torch.Tensor
    routes: Sequence[tuple[int, int, int, int]]
    expected_routes: Sequence[tuple[int, int, int, int]]


def encode_routes(value: RoutedTensor) -> dict:
    if not isinstance(value.value, torch.Tensor):
        raise ValueError("native routed value must be a tensor")
    routes, expected = [tuple(r) for r in value.routes], [tuple(r) for r in value.expected_routes]
    if (
        value.value.ndim != 2
        or value.value.shape[0] != len(routes)
        or value.value.shape[1] < 1
        or len(set(routes)) != len(routes)
        or len(set(expected)) != len(expected)
        or not set(routes) <= set(expected)
        or any(
            len(r) != 4 or any(type(i) is not int or i < 0 for i in r) for r in routes + expected
        )
    ):
        raise ValueError("invalid native MoE route ownership")
    return {
        "kind": "routed",
        "value": snapshot(value.value),
        "routes": routes,
        "expected": sorted(expected),
    }


def merge_routes(trees: list[dict]) -> dict:
    first = trees[0]
    rows = {}
    for tree in trees:
        encode_routes(RoutedTensor(tree["value"], tree["routes"], tree["expected"]))
        if (
            tree["expected"] != first["expected"]
            or tree["value"].dtype != first["value"].dtype
            or tree["value"].shape[1] != first["value"].shape[1]
        ):
            raise ValueError("MoE route domains/shapes/dtypes differ")
        for route, row in zip(tree["routes"], tree["value"], strict=True):
            key = tuple(route)
            if key in rows and not bitdiff(rows[key], row).get("equal"):
                raise ValueError("MoE route replicas disagree")
            rows[key] = row
    if set(rows) != {tuple(r) for r in first["expected"]}:
        raise ValueError("incomplete MoE dispatch coverage")
    if not rows:
        return {
            "kind": "mapping",
            "items": {
                "empty": {"kind": "bool", "value": True},
                "width": {"kind": "int", "value": first["value"].shape[1]},
                "dtype": {"kind": "str", "value": str(first["value"].dtype)},
            },
        }
    return {
        "kind": "mapping",
        "items": {
            # Persist route identity too: equal numerical rows cannot hide a
            # different expert assignment.
            "routes": {"kind": "tensor", "value": torch.tensor(sorted(rows))},
            "values": {"kind": "tensor", "value": torch.stack([rows[r] for r in sorted(rows)])},
        },
    }


def moe_dispatch(
    dispatched: torch.Tensor,
    *,
    token_rows: Sequence[tuple[int, int]],
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    dropped: torch.Tensor,
    source_tokens: torch.Tensor,
    source_slots: torch.Tensor,
    num_experts: int,
) -> dict:
    """Read actual router/permutation metadata, never rerun top-k or dispatch.

    Router metadata covers the full logical dispatch domain on each EP worker.
    ``source_tokens/source_slots`` describe its local unpadded dispatched rows.
    A scheduler must explicitly strip capacity padding using its valid-row mask.
    A dropped route is recorded rather than silently counted as missing data.
    """
    shape = expert_indices.shape
    if (
        expert_indices.ndim != 2
        or shape[0] != len(token_rows)
        or shape[1] < 1
        or len(set(token_rows)) != len(token_rows)
        or any(len(r) != 2 or any(type(i) is not int or i < 0 for i in r) for r in token_rows)
        or routing_weights.shape != shape
        or dropped.shape != shape
        or dropped.dtype != torch.bool
        or expert_indices.dtype not in (torch.int32, torch.int64)
        or not routing_weights.is_floating_point()
        or type(num_experts) is not int
        or num_experts < 1
    ):
        raise ValueError("invalid native router metadata")
    if (
        source_tokens.ndim != 1
        or source_slots.shape != source_tokens.shape
        or source_tokens.dtype not in (torch.int32, torch.int64)
        or source_slots.dtype not in (torch.int32, torch.int64)
        or dispatched.ndim != 2
        or dispatched.shape[0] != source_tokens.numel()
    ):
        raise ValueError("invalid native dispatch permutation")
    experts, drops = expert_indices.cpu().tolist(), dropped.cpu().tolist()
    if any(not 0 <= e < num_experts for row in experts for e in row):
        raise ValueError("global expert ID out of bounds")
    expected = [
        (*token_rows[t], slot, expert)
        for t, row in enumerate(experts)
        for slot, expert in enumerate(row)
        if not drops[t][slot]
    ]
    routes = []
    for token, slot in zip(source_tokens.cpu().tolist(), source_slots.cpu().tolist(), strict=True):
        if not 0 <= token < shape[0] or not 0 <= slot < shape[1] or drops[token][slot]:
            raise ValueError("dispatched token/slot is out of bounds or declared dropped")
        routes.append((*token_rows[token], slot, experts[token][slot]))
    routed = RoutedTensor(dispatched, routes, expected)
    encode_routes(routed)
    canonical = sorted(range(len(token_rows)), key=lambda i: token_rows[i])
    return {
        "dispatch": routed,
        "token_rows": [token_rows[i] for i in canonical],
        "experts": expert_indices[canonical],
        "weights": routing_weights[canonical],
        "dropped": dropped[canonical],
        "num_experts": num_experts,
    }
