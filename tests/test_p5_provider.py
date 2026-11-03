# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Provider protocol, fail-closed stub, and golden-manifest anchor tests."""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import fixtures, oracle
from rl_engine.moe.provider import ReferenceProvider, StubProvider, resolve_provider
from rl_engine.moe.trace import ExpertTrace, first_divergence


def test_reference_provider_matches_oracle_bytes() -> None:
    batch = fixtures.make_expert_batch("base_plus_lora")
    gold, cand = ExpertTrace("a"), ExpertTrace("b")
    _, saved_g = oracle.routed_expert_forward(batch, gold)
    _, saved_c = oracle.routed_expert_forward(batch, cand, ops=ReferenceProvider())
    assert first_divergence(gold, cand) is None
    dy = fixtures.make_grad_output("p", (batch.rows, batch.hidden))
    grads_g = oracle.routed_expert_backward(batch, saved_g, dy)
    grads_c = oracle.routed_expert_backward(batch, saved_c, dy, ops=ReferenceProvider())
    for key, grad in grads_g.items():
        other = grads_c[key]
        assert (grad is None) == (other is None)
        if grad is not None:
            assert torch.equal(grad, other)


def test_stub_provider_fails_closed() -> None:
    stub = StubProvider()
    with pytest.raises(NotImplementedError, match="#60"):
        stub.mxfp8_act_quant_fwd(torch.zeros(1, 32, dtype=torch.bfloat16))
    batch = fixtures.make_expert_batch("base_only_one_row")
    with pytest.raises(NotImplementedError):
        oracle.routed_expert_forward(batch, ops=stub)


def test_resolve_provider() -> None:
    assert resolve_provider("reference").name == "reference"
    assert resolve_provider("rl_engine.moe.provider:StubProvider").name == "stub"
    with pytest.raises(ValueError):
        resolve_provider("not-a-spec")
    prov = resolve_provider("reference").provenance()
    assert prov["requested_backend"] == prov["actual_backend"] == "reference"


def test_golden_manifest_anchor() -> None:
    """CI anchor: regenerated golden hashes must match the committed manifest.

    A failure here means the oracle's bytes drifted (torch RNG/libm change or
    an intentional contract change) — regenerate with
    ``python -m rl_engine.moe.fixtures --write-manifest`` and review the diff.
    """
    committed = fixtures.load_manifest()
    regenerated = fixtures.golden_manifest()
    assert committed == regenerated
