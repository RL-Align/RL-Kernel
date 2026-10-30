# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P2a: execution evidence, configured canonical TP, target ownership, missing ledger."""

import dataclasses
import json

import pytest
import torch
from p7_doubles.fixtures import vime_request
from p7_doubles.harness import (
    FIXTURES,
    TP_GROUP,
    expect,
    make_context,
    patch_tp2,
    run,
    strict_observer,
)
from p7_doubles.strict_linear_logp import CpuStrictLinearLogpDouble

from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate

pytestmark = pytest.mark.unit


@pytest.fixture
def observer(monkeypatch):
    with strict_observer(monkeypatch) as instance:
        yield instance


@pytest.fixture
def tp2(monkeypatch):
    return patch_tp2(monkeypatch)


ALWAYS = {
    "gemm.executed_spec",
    "gemm.harness_fingerprint",
    "reduction.lse",
    "reduction.global_max_sum",
    "reduction.padded_vocab_mask_applied",
}
TP_ONLY = {
    "reduction.canonical_split_actual",
    "reduction.redistribution_collective",
    "reduction.merge_order_actual",
}


def _tp2_call(observer, tp2, *, rank=1, request=None):
    fixture = FIXTURES["padded_vocab_tp2"]
    tp2.rank = rank
    request = request or vime_request(fixture, tp_rank=rank)
    request.tensor_parallel_group = TP_GROUP
    return fixture, run(observer, request)


def test_missing_ledger_non_tp(observer):
    fixture = FIXTURES["normal"]
    evidence = gate.validate_training_call(
        run(observer, vime_request(fixture)), make_context(fixture)
    )
    names = {item.field for item in evidence.missing_evidence}
    assert names == ALWAYS
    for item in evidence.missing_evidence:
        assert item.status in {"MISSING_PROVENANCE", "UPSTREAM_EVIDENCE_MISSING"}
        assert item.required_interface and item.depends_on
    assert evidence.canonical_configured["applicable"] is False


def test_missing_ledger_tp_adds_canonical_items(observer, tp2):
    fixture, snapshot = _tp2_call(observer, tp2)
    evidence = gate.validate_training_call(snapshot, make_context(fixture))
    assert {item.field for item in evidence.missing_evidence} == ALWAYS | TP_ONLY


def test_canonical_defaults_to_physical_tp_and_is_configured_only(observer, tp2, monkeypatch):
    monkeypatch.delenv("RL_KERNEL_STRICT_CANONICAL_TP", raising=False)
    monkeypatch.delenv("RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE", raising=False)
    fixture, snapshot = _tp2_call(observer, tp2)
    canonical = gate.validate_training_call(snapshot, make_context(fixture)).canonical_configured
    assert canonical == {
        "applicable": True,
        "source": "configured",
        "canonical_tp": 2,
        "canonical_vocab_size": 64,
        "summaries_per_rank": 1,
        "canonical_local_vocab": 32,
        "layout_matches_if_configured": True,
    }
    assert snapshot.field("configured.canonical_tp").source.value == "configured"


def test_canonical_tp4_on_tp2(observer, tp2, monkeypatch):
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "4")
    fixture, snapshot = _tp2_call(observer, tp2)
    canonical = gate.validate_training_call(snapshot, make_context(fixture)).canonical_configured
    assert (canonical["summaries_per_rank"], canonical["canonical_local_vocab"]) == (2, 32)
    assert {"reduction.canonical_split_actual"} <= {
        item.field
        for item in gate.validate_training_call(snapshot, make_context(fixture)).missing_evidence
    }


@pytest.mark.parametrize(
    "env, value, field",
    [
        ("RL_KERNEL_STRICT_CANONICAL_TP", "3", "configured.canonical_tp"),
        ("RL_KERNEL_STRICT_CANONICAL_TP", "x", "configured.canonical_tp"),
        ("RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE", "60", "configured.canonical_vocab_size"),
    ],
)
def test_illegal_canonical_configuration(observer, tp2, monkeypatch, env, value, field):
    monkeypatch.setenv(env, value)
    fixture, snapshot = _tp2_call(observer, tp2)
    expect("IDENTITY_DRIFT", snapshot, make_context(fixture), field)


def test_canonical_env_is_not_applicable_without_tp(observer, monkeypatch):
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "4")
    fixture = FIXTURES["normal"]
    evidence = gate.validate_training_call(
        run(observer, vime_request(fixture)), make_context(fixture)
    )
    assert evidence.canonical_configured == {
        "applicable": False,
        "canonical_tp": "4",
        "canonical_vocab_size": None,
    }


def test_target_owner_ranks_come_from_observed_targets(observer, tp2):
    fixture, snapshot = _tp2_call(observer, tp2, rank=0)
    evidence = gate.validate_training_call(snapshot, make_context(fixture))
    rows = list(fixture.cp_local_rows(0))
    active = fixture.target_ids[rows][fixture.active_mask[rows]]
    assert evidence.target_owner_ranks == tuple(int(t) // 32 for t in active)
    assert set(evidence.target_owner_ranks) == {0, 1}


def test_active_target_in_padded_tail_is_identity_drift(observer, tp2):
    fixture = FIXTURES["padded_vocab_tp2"]
    request = vime_request(fixture, tp_rank=1)
    row = int(fixture.active_mask.nonzero()[0])
    request.target_ids[row] = 62  # real vocab is 60; [60, 64) is padding
    _, snapshot = _tp2_call(observer, tp2, request=request)
    expect("IDENTITY_DRIFT", snapshot, make_context(fixture), "target_ids")


def test_missing_target_values_is_missing_provenance(observer):
    fixture = FIXTURES["normal"]
    snapshot = dataclasses.replace(run(observer, vime_request(fixture)), target_ids=None)
    expect("MISSING_PROVENANCE", snapshot, make_context(fixture), "target_ids")


def test_build_identity_is_recorded_with_sources(observer):
    fixture = FIXTURES["normal"]
    snapshot = run(observer, vime_request(fixture))
    evidence = gate.validate_training_call(snapshot, make_context(fixture))
    assert (
        evidence.build["build.strict_contract_version"]
        == "cuda-det-gemm-linear-logp-sm90-contract-v2"
    )
    assert evidence.build["build.sm90_n_split_contract"] == 64
    assert len(evidence.build["build.wrapper_module_sha256"]) == 64
    assert evidence.build["build.torch"]["version"] == torch.__version__
    assert snapshot.field("build.strict_contract_version").source.value == "declared"
    assert snapshot.field("build.extension").source.value == "observed"


def test_rng_consumption_is_recorded_not_gated(observer):
    class RandomConsumingDouble(CpuStrictLinearLogpDouble):
        def __call__(self, *args, **kwargs):
            torch.rand(1)
            return super().__call__(*args, **kwargs)

    fixture = FIXTURES["normal"]
    clean = gate.validate_training_call(run(observer, vime_request(fixture)), make_context(fixture))
    assert clean.runtime["runtime.rng_consumed"] is False
    dirty = gate.validate_training_call(
        run(observer, vime_request(fixture), RandomConsumingDouble()), make_context(fixture)
    )
    assert dirty.runtime["runtime.rng_consumed"] is True
    assert dirty.runtime["runtime.rng_before"] != dirty.runtime["runtime.rng_after"]
    assert dirty.runtime["runtime.stream"]["stream"] is None  # CPU tensors: no CUDA stream


def test_evidence_is_json_serializable(observer, tp2):
    fixture, snapshot = _tp2_call(observer, tp2)
    payload = gate.validate_training_call(snapshot, make_context(fixture)).to_dict()
    assert json.loads(json.dumps(payload)) == payload
