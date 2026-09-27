# SPDX-License-Identifier: Apache-2.0
"""Scalar CPU oracle. Every addition rounds to FP32; no NumPy/PyTorch dependency."""

import math
import struct
from .contract import ContractError, SavedForward, digest, require


def f32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ContractError("NON_FINITE", "FP32 overflow") from exc
    require(math.isfinite(result), "NON_FINITE", "active value/result")
    require(
        result == 0 or abs(result) >= 2**-126,
        "UNSUPPORTED_CAPABILITY",
        "subnormal is outside this draft profile",
    )
    return result


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def bf16_bits(value):
    bits = f32_bits(value)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    require(rounded & 0x7F80 != 0x7F80, "NON_FINITE", "BF16 overflow")
    return rounded


def bf16_value(value):
    return struct.unpack("<f", struct.pack("<I", bf16_bits(value) << 16))[0]


def add(a, b):
    return f32(a + b)


def matrix(value, rows, width, dtype, label, active=None):
    require(type(value) in (list, tuple) and len(value) == rows, "UNSUPPORTED_GEOMETRY", label)
    result = []
    for i, row in enumerate(value):
        require(type(row) in (list, tuple) and len(row) == width, "UNSUPPORTED_GEOMETRY", label)
        # Padding is never loaded into the arithmetic tree.
        if active is not None and not active[i]:
            result.append([0.0] * width)
            continue
        converted = []
        for x in row:
            require(type(x) in (int, float), "SCHEMA_MISMATCH", f"{label} value")
            v = f32(x)
            require(v == x, "DTYPE_MISMATCH", f"{label} not exact FP32")
            if dtype == "bf16":
                require(f32_bits(v) == (bf16_bits(v) << 16), "DTYPE_MISMATCH", f"{label} not BF16")
            converted.append(v)
        result.append(converted)
    return result


def matrix_hex(rows, dtype="fp32"):
    if dtype == "bf16":
        return b"".join(struct.pack("<H", bf16_bits(v)) for row in rows for v in row).hex()
    return b"".join(struct.pack("<f", f32(v)) for row in rows for v in row).hex()


def fold(plan, rows):
    """Return canonical rows, six running partials and routed accumulator."""
    n, h = len(plan.token_ids), plan.hidden_size
    token_index = {t: i for i, t in enumerate(plan.token_ids)}
    canonical = [[[0.0] * h for _ in range(6)] for _ in range(n)]
    for i, (token, slot, valid) in enumerate(plan.inverse_map):
        if valid:
            canonical[token_index[token]][slot] = list(rows[i])
    acc = [[0.0] * h for _ in range(n)]
    seen = [False] * n
    partials = []
    for slot in range(6):
        for t in range(n):
            if plan.valid_slots[t][slot]:
                if seen[t]:
                    acc[t] = [add(a, b) for a, b in zip(acc[t], canonical[t][slot], strict=True)]
                else:
                    acc[t] = list(canonical[t][slot])
                    seen[t] = True
        partials.append([list(r) for r in acc])
    return canonical, partials, acc


def forward(plan, rows, shared, residual, context):
    plan.validate(context)
    n, h = len(plan.token_ids), plan.hidden_size
    rows = matrix(
        rows, len(plan.inverse_map), h, "bf16", "routed", [r[2] for r in plan.inverse_map]
    )
    shared = matrix(shared, n, h, "bf16", "shared")
    residual = matrix(residual, n, h, "bf16", "residual")
    canonical, partials, routed = fold(plan, rows)
    after_shared = [
        [add(a, b) for a, b in zip(r, s, strict=True)] for r, s in zip(routed, shared, strict=True)
    ]
    precast = [
        [add(a, b) for a, b in zip(r, s, strict=True)]
        for r, s in zip(after_shared, residual, strict=True)
    ]
    output = [[bf16_value(v) for v in r] for r in precast]
    stages = {
        "canonical_fp32": matrix_hex([r for token in canonical for r in token]),
        "slot_partials_fp32": [matrix_hex(p) for p in partials],
        "routed_fp32": matrix_hex(routed),
        "after_shared_fp32": matrix_hex(after_shared),
        "precast_fp32": matrix_hex(precast),
        "output_bf16": matrix_hex(output, "bf16"),
    }
    return {
        "output": output,
        "stages": stages,
        "saved": SavedForward.capture(plan, context),
        "trace": {
            "schema": "p6-local-trace.v1",
            "phase": "forward",
            "plan_fingerprint": plan.fingerprint,
            "order_hash": plan.order_hash,
            "boundary_hashes": {k: digest(v) for k, v in stages.items()},
        },
    }


def backward(saved, dx_rows, dx_shared, context, expected_fingerprint, shared_boundary):
    plan = saved.restore(context, expected_fingerprint)
    require(
        shared_boundary == plan.gradient_boundary, "GRADIENT_BOUNDARY_MISMATCH", shared_boundary
    )
    n, h = len(plan.token_ids), plan.hidden_size
    rows = matrix(
        dx_rows, len(plan.inverse_map), h, "fp32", "routed dx", [r[2] for r in plan.inverse_map]
    )
    shared = matrix(dx_shared, n, h, "fp32", "shared dx")
    canonical, partials, routed = fold(plan, rows)
    output = [
        [add(a, b) for a, b in zip(r, s, strict=True)] for r, s in zip(routed, shared, strict=True)
    ]
    return {
        "output": output,
        "output_boundary": plan.gradient_boundary,
        "stages": {
            "canonical_fp32": matrix_hex([r for token in canonical for r in token]),
            "slot_partials_fp32": [matrix_hex(p) for p in partials],
            "routed_fp32": matrix_hex(routed),
            "output_fp32": matrix_hex(output),
        },
        "trace": {
            "phase": "backward",
            "plan_fingerprint": plan.fingerprint,
            "order_hash": plan.order_hash,
        },
    }


def gradient_dispatch(saved, dy, context, expected_fingerprint):
    """P4 mock only: same dy gathered for every valid slot, no route weighting."""
    plan = saved.restore(context, expected_fingerprint)
    dy = matrix(dy, len(plan.token_ids), plan.hidden_size, "fp32", "dy")
    table = dict(zip(plan.token_ids, dy, strict=True))
    return [
        list(table[t]) if valid else [0.0] * plan.hidden_size for t, slot, valid in plan.inverse_map
    ]


def mock_return(plan, rows, arrival, context):
    """P4 mock delivers by logical receive index, independent of readiness."""
    plan.validate(context)
    require(len(rows) == len(plan.inverse_map), "UNSUPPORTED_GEOMETRY", "return rows")
    require(
        type(arrival) in (list, tuple) and all(type(i) is int for i in arrival),
        "INVALID_DISCRETE_PLAN",
        "arrival type",
    )
    require(
        sorted(arrival) == list(range(len(rows))),
        "INVALID_DISCRETE_PLAN",
        "incomplete/duplicate arrival",
    )
    result = [None] * len(rows)
    for logical_index in arrival:
        result[logical_index] = list(rows[logical_index])
    return result
