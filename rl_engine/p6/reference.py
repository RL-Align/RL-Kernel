# SPDX-License-Identifier: Apache-2.0
"""Independent T02-T06 CPU entry points; no production kernel registration."""

from .contract import require
from .oracle import add, backward, bf16_value, forward, matrix, matrix_hex


def canonical_unpermute_fwd(plan, rows, context):
    plan.validate(context)
    rows = matrix(
        rows,
        len(plan.inverse_map),
        plan.hidden_size,
        "bf16",
        "routed",
        [r[2] for r in plan.inverse_map],
    )
    lookup = {token: i for i, token in enumerate(plan.token_ids)}
    result = [[[0.0] * plan.hidden_size for _ in range(6)] for _ in plan.token_ids]
    for row, (token, slot, valid) in zip(rows, plan.inverse_map, strict=True):
        if valid:
            result[lookup[token]][slot] = list(row)
    return {
        "canonical_rows": result,
        "canonical_fp32": matrix_hex([row for slots in result for row in slots]),
    }


def fixed_order_combine_fwd(plan, canonical_rows, context):
    plan.validate(context)
    require(len(canonical_rows) == len(plan.token_ids), "UNSUPPORTED_GEOMETRY", "token rows")
    values = [
        matrix(slots, 6, plan.hidden_size, "fp32", "canonical", mask)
        for slots, mask in zip(canonical_rows, plan.valid_slots, strict=True)
    ]
    result = [[0.0] * plan.hidden_size for _ in plan.token_ids]
    seen = [False] * len(plan.token_ids)
    partials = []
    for slot in range(6):
        for i, mask in enumerate(plan.valid_slots):
            if mask[slot]:
                result[i] = (
                    [add(a, b) for a, b in zip(result[i], values[i][slot], strict=True)]
                    if seen[i]
                    else list(values[i][slot])
                )
                seen[i] = True
        partials.append(matrix_hex(result))
    return {"routed": result, "routed_fp32": matrix_hex(result), "slot_partials_fp32": partials}


def shared_residual_merge_fwd(plan, routed, shared, residual, context):
    plan.validate(context)
    n, h = len(plan.token_ids), plan.hidden_size
    routed = matrix(routed, n, h, "fp32", "routed accumulator")
    shared = matrix(shared, n, h, "bf16", "shared")
    residual = matrix(residual, n, h, "bf16", "residual")
    middle = [
        [add(a, b) for a, b in zip(x, y, strict=True)] for x, y in zip(routed, shared, strict=True)
    ]
    precast = [
        [add(a, b) for a, b in zip(x, y, strict=True)]
        for x, y in zip(middle, residual, strict=True)
    ]
    output = [[bf16_value(x) for x in row] for row in precast]
    return {
        "output": output,
        "after_shared_fp32": matrix_hex(middle),
        "precast_fp32": matrix_hex(precast),
        "output_bf16": matrix_hex(output, "bf16"),
    }


# These expose the T05/T06 contract names for the scalar oracle only.
fused_moe_combine_fwd = forward
fused_dx_fanin_bwd = backward
