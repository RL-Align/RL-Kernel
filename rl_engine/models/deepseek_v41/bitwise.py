# SPDX-License-Identifier: Apache-2.0
"""Storage-bit comparison; signed zero and NaN payloads are significant."""

from __future__ import annotations

import math

import torch


def compare_bits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    token_dims: int | None = None,
    layer: int | None = None,
    operator: str | None = None,
    rank: int = 0,
) -> dict:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("bitwise comparison requires identical dtype and shape")
    if (
        actual.device != expected.device
        or actual.layout != torch.strided
        or expected.layout != torch.strided
    ):
        raise ValueError("bitwise comparison requires colocated strided tensors")
    if token_dims is not None and not 0 < token_dims <= actual.ndim:
        raise ValueError("token_dims must identify a nonempty shape prefix")
    width = actual.element_size()
    a = actual.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    b = expected.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    unequal = (a != b).any(dim=-1)
    count = int(unequal.sum().item())
    tokens = math.prod(actual.shape[:token_dims]) if token_dims is not None else None
    bad_tokens = None
    if tokens is not None:
        bad_tokens = int(unequal.reshape(tokens, -1).any(-1).sum().item()) if tokens else 0
    first = None
    if count:
        offset = int(unequal.nonzero()[0].item())
        index, rest = [], offset
        for size in reversed(actual.shape):
            index.append(rest % size)
            rest //= size

        def pattern(value):
            return "0x" + bytes(value[offset].cpu().tolist())[::-1].hex()

        first = {
            "flat_index": offset,
            "index": list(reversed(index)),
            "actual_bits": pattern(a),
            "expected_bits": pattern(b),
        }
    return {
        "dtype": str(actual.dtype),
        "shape": list(actual.shape),
        "storage_elements": actual.numel(),
        "compared_elements": actual.numel(),
        "mismatched_elements": count,
        "compared_bytes": a.numel(),
        "total_tokens": tokens,
        "mismatched_tokens": bad_tokens,
        "first_mismatch": first,
        "layer": layer,
        "operator": operator,
        "rank": rank,
        "status": "PASS" if count == 0 else "FAIL",
    }
