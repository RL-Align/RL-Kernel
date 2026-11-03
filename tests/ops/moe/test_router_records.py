# SPDX-License-Identifier: Apache-2.0
"""CPU adversarial record loading and discrete-before-numeric comparison."""

from dataclasses import replace
import json

import pytest
import torch

from rl_engine.ops.moe.learned_route import _digest
from rl_engine.ops.moe.router_training import (
    RouteRecord,
    compare_route_records,
    load_route_record,
    save_route_record,
)


def record():
    data = {
        "schema": "rlkernel.route_record.v1",
        "kind": "learned",
        "mode": "normal",
        "absolute_layer": 3,
        "versions": {
            "model_version": "m1",
            "bias_version": "b1",
            "domain_version": "d1",
            "plan_version": "p1",
        },
        "record_version": "r1",
        "rank": 0,
        "world_size": 1,
        "global_ids": list(range(256)),
        "renormalize": True,
        "scale": 1.5,
        "source_record_sha256": None,
        "execution": {"actual_backend": "cpu_test_fixture"},
        "tokens": [
            {
                "logical_id": 10,
                "experts": list(range(6)),
                "columns": list(range(6)),
                "keep": [True] * 6,
                "positions": [0] * 6,
                "weight_bits": [1048576000] * 6,
                "logit_sha256": "0" * 64,
            }
        ],
    }
    return make_record(data)


def make_record(data):
    return RouteRecord(json.dumps(data, sort_keys=True, separators=(",", ":")), _digest(data))


def test_raw_bits_and_discrete_failure_priority():
    baseline = record()
    data = baseline.payload()
    data["tokens"][0]["experts"] = list(range(6, 12))
    data["tokens"][0]["columns"] = list(range(6, 12))
    data["tokens"][0]["weight_bits"][0] += 1
    difference = compare_route_records(make_record(data), baseline)
    assert difference["status"] == "discrete_mismatch" and not difference["numeric_compared"]
    assert difference["first_difference"] == {
        "logical_id": 10,
        "slot": 0,
        "field": "experts",
        "actual": 6,
        "expected": 0,
    }
    data = baseline.payload()
    data["tokens"][0]["weight_bits"][0] += 1
    difference = compare_route_records(make_record(data), baseline)
    assert difference["status"] == "value_mismatch" and difference["numeric_compared"]
    # Signed zero is numerically equal but its original payload differs.
    data["tokens"][0]["weight_bits"][0] = 0
    positive_zero = make_record(data)
    data["tokens"][0]["weight_bits"][0] = -(2**31)
    assert compare_route_records(make_record(data), positive_zero)["status"] == "value_mismatch"


@pytest.mark.parametrize("field", ["keep", "positions"])
def test_mask_and_dispatch_differences_precede_values(field):
    base = record()
    data = base.payload()
    if field == "keep":
        data["tokens"][0]["keep"][0] = False
        data["tokens"][0]["positions"][0] = -1
    else:
        data["tokens"][0]["positions"][0] = 1
    assert compare_route_records(make_record(data), base)["status"] == "discrete_mismatch"


@pytest.mark.parametrize(
    "bad",
    [
        "schema",
        "rank",
        "world_size",
        "layer",
        "dtype_bits",
        "nan",
        "duplicate_experts",
        "column_map",
        "global_map",
        "slot_count",
        "mask_type",
        "position",
        "logical_id",
        "missing_version",
        "empty_version",
        "scale",
        "renormalize",
        "source",
        "logit_hash",
    ],
)
def test_self_consistent_checksum_does_not_bypass_structural_validation(bad):
    data = record().payload()
    row = data["tokens"][0]
    if bad == "schema":
        data["schema"] = "other"
    if bad == "rank":
        data["rank"] = True
    if bad == "world_size":
        data["world_size"] = 0
    if bad == "layer":
        data["absolute_layer"] = 0
    if bad == "dtype_bits":
        row["weight_bits"][0] = 1.0
    if bad == "nan":
        row["weight_bits"][0] = int(torch.tensor(float("nan")).view(torch.int32))
    if bad == "duplicate_experts":
        row["experts"][0] = 1
    if bad == "column_map":
        row["columns"][0] = 255
    if bad == "global_map":
        data["global_ids"][0] = 1
    if bad == "slot_count":
        row["keep"].pop()
    if bad == "mask_type":
        row["keep"][0] = 1
    if bad == "position":
        row["positions"][0] = -1
    if bad == "logical_id":
        row["logical_id"] = -1
    if bad == "missing_version":
        del data["versions"]["bias_version"]
    if bad == "empty_version":
        data["versions"]["bias_version"] = " "
    if bad == "scale":
        data["scale"] = float("inf")
    if bad == "renormalize":
        data["renormalize"] = False
    if bad == "source":
        data["source_record_sha256"] = "f" * 64
    if bad == "logit_hash":
        row["logit_sha256"] = "x" * 64
    with pytest.raises(ValueError):
        make_record(data).payload()


def test_checksum_snapshot_roundtrip_and_identity_barriers(tmp_path):
    base = record()
    with pytest.raises(ValueError, match="checksum"):
        replace(base, sha256="f" * 64).payload()
    path = tmp_path / "record.json"
    save_route_record(base, path)
    assert load_route_record(path) == base
    for field, value in [("rank", 1), ("record_version", "r2"), ("scale", 2.0)]:
        data = base.payload()
        if field == "rank":
            data["world_size"] = 2
        data[field] = value
        with pytest.raises(ValueError, match="identity mismatch"):
            compare_route_records(make_record(data), base)
    data = base.payload()
    data["execution"]["actual_backend"] = "other_backend"
    with pytest.raises(ValueError, match="execution actual_backend identity"):
        compare_route_records(make_record(data), base)


def test_diagnostics_and_empty_rows_never_sign_off_acceptance():
    base = record()
    data = base.payload()
    data["mode"] = "diagnostic_fixed"
    data["source_record_sha256"] = base.sha256
    diagnostic = make_record(data)
    with pytest.raises(ValueError, match="diagnostic only"):
        compare_route_records(diagnostic, base)
    assert not compare_route_records(diagnostic, base, require_normal=False)["acceptance"]
    data = base.payload()
    data["tokens"] = []
    empty = make_record(data)
    result = compare_route_records(empty, empty)
    assert result["status"] == "empty" and not result["acceptance"]


def test_gradient_payload_identity_and_first_difference(tmp_path):
    from rl_engine.ops.moe.router_training import (
        RouteVJPRecord,
        compare_vjp_records,
        load_vjp_record,
        save_vjp_record,
    )

    base = record()
    data = {
        "schema": "rlkernel.route_vjp_record.v1",
        "forward": base.payload(),
        "forward_sha256": base.sha256,
        "loss_version": "loss.v1",
        "upstream_bits": [[0] * 6],
        "dlogits_bits": [[0] * 256],
    }

    def build(payload):
        return RouteVJPRecord(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), _digest(payload)
        )

    original = build(data)
    path = tmp_path / "vjp.json"
    save_vjp_record(original, path)
    assert load_vjp_record(path) == original
    data["dlogits_bits"][0][200] = 1
    different = build(data)
    result = compare_vjp_records(different, original)
    assert result["status"] == "gradient_mismatch" and result["first_difference"] == {
        "logical_id": 10,
        "global_expert_id": 200,
        "actual_bits": 1,
        "expected_bits": 0,
    }
    data["forward_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="forward record identity"):
        build(data).payload()
