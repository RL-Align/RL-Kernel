# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1 schema, fingerprint, and trace tests (issue #2 contracts)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from rl_engine.mhc import fixtures
from rl_engine.mhc.contract import (
    PROD_CONTROLLER_N,
    PROD_HIDDEN,
    SCHEMA_VERSION,
    SINKHORN_ITERS,
    LayerContract,
    tensor_sha256,
)
from rl_engine.mhc.trace import MHCTrace, first_divergence


def test_fixture_batches_validate() -> None:
    for name in fixtures.BLOCK_CASES:
        batch = fixtures.make_batch(name)
        assert batch.contract.schema_version == SCHEMA_VERSION
        batch.validate()
        fixtures.make_grads(name, batch).validate(batch)


def test_production_contract_constants_are_frozen() -> None:
    prod = LayerContract()
    prod.validate()
    prod.assert_production()
    assert (prod.hidden, prod.controller_n, prod.flat_k) == (PROD_HIDDEN, PROD_CONTROLLER_N, 16384)
    assert prod.hc_mult == 4 and prod.sinkhorn_iters == SINKHORN_ITERS == 20
    assert prod.mhc_eps == prod.rmsnorm_eps == 1e-6
    with pytest.raises(ValueError):
        LayerContract(hidden=128).assert_production()


def test_frozen_constants_cannot_be_overridden() -> None:
    for kwargs in (
        {"hc_mult": 2},
        {"sinkhorn_iters": 10},
        {"mhc_eps": 1e-5},
        {"rmsnorm_eps": 1e-5},
        {"controller_n": 20},
    ):
        with pytest.raises(ValueError):
            LayerContract(**kwargs).validate()


def test_default_contract_is_the_canonical_unfused_mode() -> None:
    """Train/infer byte-equality is the goal, so unfused is what you get unless
    an engine explicitly declares it cannot expose the intermediate."""
    from rl_engine.mhc.contract import CANONICAL_FUSION_MODE

    assert LayerContract().fusion_mode == CANONICAL_FUSION_MODE == "unfused"
    assert LayerContract().trainability == "full"


def test_unknown_modes_fail_closed() -> None:
    for kwargs in (
        {"fusion_mode": "fused-everything"},
        {"trainability": "everything-trainable"},
        {"placement": "expert-parallel"},
    ):
        with pytest.raises(ValueError):
            LayerContract(**kwargs).validate()


def test_weight_fingerprint_detects_tampering() -> None:
    batch = fixtures.make_batch("packed_t16")
    batch.validate()
    batch.controller.weight[0, 0] += 1.0  # tamper one checkpoint value
    with pytest.raises(ValueError, match="fingerprint"):
        batch.validate()


def test_gamma_tampering_is_also_caught() -> None:
    batch = fixtures.make_batch("packed_t16")
    batch.norm.gamma[0] = torch.tensor(9.0, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="fingerprint"):
        batch.validate()


def test_bad_dtypes_and_shapes_fail_closed() -> None:
    batch = fixtures.make_batch("packed_t16")
    with pytest.raises(TypeError):
        dataclasses.replace(batch, r_old=batch.r_old.float()).validate()
    with pytest.raises(TypeError):
        dataclasses.replace(batch, token_id=batch.token_id.to(torch.int32)).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(batch, y_sublayer=batch.y_sublayer[:, :-1]).validate()
    with pytest.raises(ValueError, match="one-row"):
        dataclasses.replace(batch, row_geometry="one-row").validate()


def test_controller_params_must_be_fp32() -> None:
    batch = fixtures.make_batch("packed_t16")
    bad = dataclasses.replace(batch.controller, weight=batch.controller.weight.to(torch.bfloat16))
    with pytest.raises(TypeError, match="FP32"):
        bad.validate(batch.contract)


def test_grad_boundary_shape_checks() -> None:
    batch = fixtures.make_batch("packed_t16")
    grads = fixtures.make_grads("packed_t16", batch)
    with pytest.raises(ValueError):
        dataclasses.replace(grads, d_normalized=grads.d_normalized[:-1]).validate(batch)
    with pytest.raises(TypeError):
        dataclasses.replace(grads, d_residual=grads.d_residual.float()).validate(batch)


def test_batch_serialization_roundtrip(tmp_path) -> None:
    batch = fixtures.make_batch("packed_t16")
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    loaded = torch.load(path, weights_only=False)
    loaded.validate()
    assert tensor_sha256(loaded.r_old) == tensor_sha256(batch.r_old)
    assert loaded.weight_fingerprint == batch.weight_fingerprint


def test_contract_fingerprint_moves_with_every_frozen_field() -> None:
    base = LayerContract(hidden=128)
    for kwargs in (
        {"hidden": 256},
        {"fusion_mode": "fused-pre-norm"},
        {"trainability": "mixer-frozen"},
    ):
        assert dataclasses.replace(base, **kwargs).fingerprint() != base.fingerprint()
    # layer_index is identity, not arithmetic: it must NOT move the fingerprint.
    assert dataclasses.replace(base, layer_index=17).fingerprint() == base.fingerprint()


def test_trace_first_divergence() -> None:
    a = MHCTrace(numeric_profile="p")
    b = MHCTrace(numeric_profile="p")
    t1 = torch.arange(4, dtype=torch.float32)
    t2 = t1 + 1
    for trace, second in ((a, t1), (b, t2)):
        trace.record("s1", t1)
        trace.record("s2", second)
    assert first_divergence(a, b) == "s2"
    b.records[1] = a.records[1]
    assert first_divergence(a, b) is None


def test_trace_records_arithmetic_provenance() -> None:
    from rl_engine.mhc import oracle

    batch = fixtures.make_batch("packed_t16")
    trace = MHCTrace(numeric_profile="p")
    oracle.mhc_block_forward(batch, trace)
    for key in ("reduction_tree", "fma", "rsqrt", "downcast_points", "weight_fingerprint"):
        assert key in trace.notes, f"trace is missing required provenance note {key!r}"
    assert "rsqrt(mean+eps)" in trace.notes["rsqrt"]
    assert trace.to_dict()["records"][0]["name"] == "controller.p"
