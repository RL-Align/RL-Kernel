# SPDX-License-Identifier: Apache-2.0
"""Synthetic start-kit inputs. Goldens are serialized and reviewed separately."""

from dataclasses import asdict
from pathlib import Path
import json
from .contract import (
    Context,
    CombinePlan,
    POLICY,
    SCHEMA,
    SavedForward,
    digest,
    exact_keys,
    require,
)
from .oracle import bf16_value, forward, backward, gradient_dispatch, matrix_hex


class Generator:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def value(self):
        return bf16_value(((self.next() >> 8) % 4097 - 2048) / 64.0)

    def shuffle(self, rows):
        rows = list(rows)
        for i in range(len(rows) - 1, 0, -1):
            j = self.next() % (i + 1)
            rows[i], rows[j] = rows[j], rows[i]
        return rows


def make_case(name, n=3, h=17, seed=7, mode="full"):
    rng = Generator(seed)
    tokens = tuple(2**40 + 17 * (n - i) for i in range(n))
    masks = tuple(
        tuple(mode != "zero-routes" and (mode != "mixed" or (i + s) % 3 != 0) for s in range(6))
        for i in range(n)
    )
    keys = [
        (t, s, True) for t, mask in zip(tokens, masks, strict=True) for s in range(6) if mask[s]
    ]
    if mode == "mixed":
        keys += [(-1, -1, False), (-1, -1, False)]
    keys = tuple(rng.shuffle(keys))
    context = Context("synthetic-run", "microbatch-0", name + "-forward", "synthetic-checkpoint")
    plan = CombinePlan(
        SCHEMA,
        context,
        ("synthetic-route.draft.v1", digest([tokens, masks])),
        ("synthetic-exchange.draft.v1", digest(keys)),
        tokens,
        masks,
        keys,
        h,
        "synthetic.expert_input",
        digest(POLICY),
    )
    plan.validate(context)
    rows = [[rng.value() for _ in range(h)] for _ in keys]
    shared = [[rng.value() for _ in range(h)] for _ in tokens]
    residual = [[rng.value() for _ in range(h)] for _ in tokens]
    dx = [[rng.value() / 2.0 for _ in range(h)] for _ in keys]
    dx_shared = [[rng.value() / 2.0 for _ in range(h)] for _ in tokens]
    dy = [[rng.value() for _ in range(h)] for _ in tokens]
    return {
        "name": name,
        "plan": plan.to_dict(),
        "rows": rows,
        "shared": shared,
        "residual": residual,
        "dx_rows": dx,
        "dx_shared": dx_shared,
        "dy": dy,
    }


def cases():
    result = [
        make_case("regular-tail-17"),
        make_case("mixed-invalid-overflow", mode="mixed"),
        make_case("zero-tokens", n=0),
        make_case("zero-routes", mode="zero-routes"),
        make_case("one-token-tail-33", n=1, h=33),
        make_case("batch-tail-65", n=7, h=65, seed=99),
    ]
    cancellation = make_case("cancellation-slot-order", n=1, h=1)
    # Sequential FP32 gives 2; the tested rank-local regrouping gives 3.
    vals = [2**24, 1, -(2**24), 3, -3, 2]
    for i, (_, s, _) in enumerate(cancellation["plan"]["inverse_map"]):
        cancellation["rows"][i] = [float(vals[s])]
        cancellation["dx_rows"][i] = [float(vals[s])]
    cancellation["shared"] = [[0.0]]
    cancellation["residual"] = [[0.0]]
    cancellation["dx_shared"] = [[0.0]]
    result.append(cancellation)
    rounding = make_case("one-round-bf16-ties", n=1, h=3)
    for i, (_, s, _) in enumerate(rounding["plan"]["inverse_map"]):
        rounding["rows"][i] = (
            [1.0, 1.0078125, 1.0]
            if s == 0
            else [0.00390625, 0.00390625, 0.0]
            if s == 1
            else [0.0] * 3
        )
    rounding["shared"] = [[0.0, 0.0, 0.00390625]]
    rounding["residual"] = [[0.0, 0.0, 0.00390625]]
    result.append(rounding)
    signed = make_case("signed-zero", n=1, h=2)
    for key in ("rows", "shared", "residual", "dx_rows", "dx_shared"):
        signed[key] = [[-0.0, 0.0] for _ in signed[key]]
    result.append(signed)
    return result


def evaluate(case):
    plan = CombinePlan.from_dict(case["plan"])
    fwd = forward(plan, case["rows"], case["shared"], case["residual"], plan.context)
    bwd = backward(
        fwd["saved"],
        case["dx_rows"],
        case["dx_shared"],
        plan.context,
        plan.fingerprint,
        plan.gradient_boundary,
    )
    gathered = gradient_dispatch(fwd["saved"], case["dy"], plan.context, plan.fingerprint)
    return {
        "forward": fwd["stages"],
        "backward": bwd["stages"],
        "gradient_dispatch_fp32": matrix_hex(gathered),
        "saved_forward": asdict(fwd["saved"]),
        "trace": fwd["trace"],
    }


def make_golden():
    payload = {"schema": SCHEMA, "policy": POLICY, "cases": []}
    for case in cases():
        payload["cases"].append({"input": case, "expected": evaluate(case)})
    return {"payload": payload, "payload_sha256": digest(payload)}


def load_golden(path):
    value = json.loads(Path(path).read_text())
    exact_keys(value, ("payload", "payload_sha256"), "golden envelope")
    require(
        digest(value["payload"]) == value["payload_sha256"], "CORRUPT_ARTIFACT", "golden checksum"
    )
    require(
        value["payload"]["schema"] == SCHEMA and value["payload"]["policy"] == POLICY,
        "SCHEMA_MISMATCH",
        "golden policy/schema",
    )
    return value


class RecordedProvider:
    """Fixture-only stub: input-bound; never registered as a live provider."""

    def __init__(self, case, expected):
        require(evaluate(case) == expected, "CORRUPT_ARTIFACT", "recorded expected stages")
        self.case_json = json.dumps(case, sort_keys=True, allow_nan=False)
        self.input_hash = digest(case)
        self.expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)

    def run(self, case):
        require(digest(case) == self.input_hash, "IDENTITY_DRIFT", "recorded input mismatch")
        result = json.loads(self.expected_json)
        plan = CombinePlan.from_dict(case["plan"])
        SavedForward(**result["saved_forward"]).restore(plan.context, plan.fingerprint)
        return result
