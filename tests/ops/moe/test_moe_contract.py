# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5 schema, fingerprint, and trace tests (issue #8 contracts)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from rl_engine.moe import fixtures
from rl_engine.moe.contract import SCHEMA_VERSION, tensor_sha256
from rl_engine.moe.trace import ExpertTrace, first_divergence


def test_fixture_batches_validate() -> None:
    for name in fixtures.E2E_CASES:
        batch = fixtures.make_expert_batch(name)
        assert batch.schema_version == SCHEMA_VERSION
        batch.validate()
    for name in fixtures.SHARED_CASES:
        fixtures.make_shared_batch(name).validate()


def test_weight_fingerprint_detects_tampering() -> None:
    batch = fixtures.make_expert_batch("base_plus_lora")
    batch.validate()
    batch.w1.codes[0, 0, 0] ^= 0xFF  # tamper one packed byte
    with pytest.raises(ValueError, match="fingerprint"):
        batch.validate()


def test_bad_offsets_and_dtypes_fail_closed() -> None:
    batch = fixtures.make_expert_batch("base_only_packed")
    bad = dataclasses.replace(batch, expert_offsets=torch.tensor([0, 30, 24], dtype=torch.int32))
    with pytest.raises(ValueError):
        bad.validate()
    bad2 = dataclasses.replace(batch, p_s=batch.p_s.to(torch.bfloat16))
    with pytest.raises(TypeError):
        bad2.validate()


def test_batch_serialization_roundtrip(tmp_path) -> None:
    batch = fixtures.make_expert_batch("base_plus_lora")
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    loaded = torch.load(path, weights_only=False)
    loaded.validate()
    assert tensor_sha256(loaded.x) == tensor_sha256(batch.x)
    assert loaded.weight_fingerprint == batch.weight_fingerprint


def test_trace_first_divergence() -> None:
    a = ExpertTrace(numeric_profile="p")
    b = ExpertTrace(numeric_profile="p")
    t1 = torch.arange(4, dtype=torch.float32)
    t2 = torch.arange(4, dtype=torch.float32) + 1
    a.record("s1", t1)
    a.record("s2", t1)
    b.record("s1", t1)
    b.record("s2", t2)
    assert first_divergence(a, b) == "s2"
    b.records[1] = a.records[1]
    assert first_divergence(a, b) is None
