"""Synthetic versioned cases, fixed token identities and source checksums."""

from dataclasses import dataclass, field, replace

import numpy as np

from .contract import FIXTURE_VERSION, ROUND_POLICIES, E, H
from .serialization import fingerprint


@dataclass(frozen=True)
class RouterCase:
    case_id: str
    z: np.ndarray
    input_token_id: np.ndarray
    global_token_id: np.ndarray
    row_active: np.ndarray
    table: np.ndarray
    bias: np.ndarray
    dweights: np.ndarray
    absolute_layer: int = 3
    router_mode: str = "learned"
    round_policy: str = "fp32_direct"
    checkpoint_id: str = "synthetic-checkpoint.v1"
    weight_id: str = "synthetic-pre-update.v1"
    weight_fingerprint: str = fingerprint("synthetic-gate-weight.v1")
    source_dtype: str = "float32"
    hidden_dtype: str = "bfloat16"
    hidden_shape: tuple = (4, H)
    hidden: np.ndarray = field(default_factory=lambda: np.zeros((4, H), dtype=np.uint16))

    def identity(self):
        active = self.row_active
        pairs = sorted(
            zip(
                self.global_token_id[active].tolist(),
                self.input_token_id[active].tolist(),
                strict=False,
            )
        )
        return {
            "case_id": self.case_id,
            "checkpoint_id": self.checkpoint_id,
            "weight_id": self.weight_id,
            "weight_fingerprint": self.weight_fingerprint,
            "tokens": pairs,
            "layer": self.absolute_layer,
            "mode": self.router_mode,
            "round_policy": self.round_policy,
            "table": fingerprint(self.table) if self.router_mode == "hash" else None,
            "bias": fingerprint(self.bias) if self.router_mode == "learned" else None,
        }

    def checksum(self):
        return fingerprint(
            {
                "identity": self.identity(),
                "z": self.z,
                "input_token_id": self.input_token_id,
                "global_token_id": self.global_token_id,
                "row_active": self.row_active,
                "table": self.table,
                "bias": self.bias,
                "dweights": self.dweights,
                "hidden_shape": self.hidden_shape,
                "hidden_dtype": self.hidden_dtype,
                "source_dtype": self.source_dtype,
                "hidden_bf16_storage": self.hidden,
            }
        )

    def rows(self, indices):
        return replace(
            self,
            z=self.z[indices].copy(),
            input_token_id=self.input_token_id[indices].copy(),
            global_token_id=self.global_token_id[indices].copy(),
            row_active=self.row_active[indices].copy(),
            dweights=self.dweights[indices].copy(),
            hidden_shape=(len(indices), H),
            hidden=self.hidden[indices].copy(),
        )


def catalog():
    rng = np.random.default_rng(43001)
    z = rng.standard_normal((4, E)).astype(np.float32)
    table = np.array(
        [[9, 1, 9, 200, 4, 18], [8, 7, 6, 5, 4, 3], [255, 0, 200, 3, 17, 66], [0, 1, 2, 3, 4, 5]],
        dtype=np.int32,
    )
    base = RouterCase(
        "random",
        z,
        np.array([1, 2, 1, 0], np.int64),
        np.array([101, 203, 305, -1], np.int64),
        np.array([1, 1, 1, 0], np.bool_),
        table,
        rng.uniform(-0.5, 0.5, E).astype(np.float32),
        rng.standard_normal((4, 6)).astype(np.float32),
    )
    tie = np.zeros_like(z)
    near = tie.copy()
    near[:, 7] = np.nextafter(np.float32(1), np.float32(2))
    near[:, 8] = 1
    threshold = z.copy()
    threshold[:, :8] = [
        np.nextafter(np.float32(20), np.float32(0)),
        20,
        np.nextafter(np.float32(20), np.float32(21)),
        -104,
        -100,
        -80,
        0,
        80,
    ]
    cases = [
        base,
        replace(base, case_id="exact-tie", z=tie, bias=np.zeros(E, np.float32)),
        replace(base, case_id="near-tie", z=near, bias=np.zeros(E, np.float32)),
        replace(base, case_id="threshold-extremes", z=threshold),
        replace(base, case_id="hash-slots", router_mode="hash", absolute_layer=0),
        replace(
            base,
            case_id="hash-duplicates",
            router_mode="hash",
            absolute_layer=2,
            input_token_id=np.array([0, 0, 2, 0], np.int64),
        ),
        replace(
            base,
            case_id="zero-active",
            row_active=np.zeros(4, np.bool_),
            global_token_id=np.full(4, -1, np.int64),
        ),
        replace(base.rows([]), case_id="empty"),
    ]
    original = [
        replace(c, case_id=f"{c.case_id}.{policy}", round_policy=policy)
        for policy in ROUND_POLICIES
        for c in cases
    ]
    cutoff = np.zeros(E, np.float32)
    cutoff[:5] = 2
    cutoff[5] = 1
    cutoff[6] = np.float32(1 + 2**-22)
    eighth = np.zeros(E, np.float32)
    eighth[:7] = 2
    eighth[7] = 1
    eighth[8] = np.float32(1.001)
    prebias = np.zeros(E, np.float32)
    prebias[200] = 100
    additions = [
        replace(base, case_id="cutoff-6-7-bias-ulp", z=tie.copy(), bias=cutoff),
        replace(base, case_id="bias-precision-8-9", z=tie.copy(), bias=eighth),
        replace(base, case_id="pre-bias-weight", bias=prebias),
        replace(
            base,
            case_id="dropless-hotspot-padding",
            router_mode="hash",
            absolute_layer=1,
            input_token_id=np.zeros(4, np.int64),
            table=np.tile(np.arange(6, dtype=np.int32), (4, 1)),
        ),
    ]
    return original + [
        replace(c, case_id=f"{c.case_id}.{policy}", round_policy=policy)
        for policy in ROUND_POLICIES
        for c in additions
    ]


def fixture_manifest():
    return {
        "fixture_version": FIXTURE_VERSION,
        "miles_anchor_status": "anchor_pending",
        "source": "synthetic-only; not recorded Megatron/Miles",
        "cases": [
            {"case_id": c.case_id, "checksum": c.checksum(), "identity": c.identity()}
            for c in catalog()
        ],
    }
