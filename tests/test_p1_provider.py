# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Provider protocol, fail-closed stub, and golden-manifest anchor tests."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from rl_engine.mhc import fixtures, oracle
from rl_engine.mhc.contract import LayerContract
from rl_engine.mhc.provider import (
    ReferenceProvider,
    StubProvider,
    check_capability,
    resolve_provider,
)
from rl_engine.mhc.trace import MHCTrace, first_divergence


def test_reference_provider_matches_oracle_bytes() -> None:
    batch = fixtures.make_batch("packed_t16")
    grads = fixtures.make_grads("packed_t16", batch)
    gold, cand = MHCTrace("a"), MHCTrace("b")
    _, saved_g = oracle.mhc_block_forward(batch, gold)
    _, saved_c = oracle.mhc_block_forward(batch, cand, ops=ReferenceProvider())
    assert first_divergence(gold, cand) is None
    out_g = oracle.mhc_block_backward(batch, saved_g, grads, gold)
    out_c = oracle.mhc_block_backward(batch, saved_c, grads, cand, ops=ReferenceProvider())
    assert first_divergence(gold, cand) is None
    for key, grad in out_g.items():
        other = out_c[key]
        assert (grad is None) == (other is None)
        if grad is not None:
            assert torch.equal(grad, other), f"{key} diverged"


def test_stub_provider_fails_closed() -> None:
    stub = StubProvider()
    with pytest.raises(NotImplementedError, match=r"P1-1 \(#14\)"):
        stub.hc_split_sinkhorn_fwd(torch.zeros(1, 24))
    # #15 absorbs the fixed-K GEMM reference, so it points at P1-2, not its own task
    with pytest.raises(NotImplementedError, match=r"P1-2 \(#15\)"):
        stub.fixed_k_gemm_fwd(torch.zeros(1, 4), torch.zeros(2, 4))
    with pytest.raises(NotImplementedError, match=r"P1-5 \(#18\)"):
        stub.rmsnorm_residual_fwd(torch.zeros(1, 8), torch.zeros(8), 1e-6)
    batch = fixtures.make_batch("one_row")
    with pytest.raises(NotImplementedError):
        oracle.mhc_block_forward(batch, ops=stub)


def test_a_partial_provider_keeps_the_rest_on_the_oracle() -> None:
    """The pattern each D1..D6 PR uses: override one op, run full acceptance."""

    calls: list[str] = []

    class OnlyPost(ReferenceProvider):
        name = "only-post"
        numeric_profile = "test-only-post"

        @staticmethod
        def mhc_post_fwd(r_old, y, c, post):
            calls.append("fwd")
            return oracle.mhc_post_fwd(r_old, y, c, post)

    batch = fixtures.make_batch("packed_t16")
    gold, cand = MHCTrace("a"), MHCTrace("b")
    oracle.mhc_block_forward(batch, gold)
    oracle.mhc_block_forward(batch, cand, ops=OnlyPost())
    assert calls == ["fwd"]
    assert first_divergence(gold, cand) is None


def test_resolve_provider() -> None:
    assert resolve_provider("reference").name == "reference"
    assert resolve_provider("rl_engine.mhc.provider:StubProvider").name == "stub"
    with pytest.raises(ValueError):
        resolve_provider("not-a-spec")
    prov = resolve_provider("reference").provenance()
    assert prov["requested_backend"] == prov["actual_backend"] == "reference"


def test_check_capability_fails_closed_on_unsupported_placement() -> None:
    provider = ReferenceProvider()
    check_capability(provider, LayerContract(hidden=128))
    sharded = dataclasses.replace(LayerContract(hidden=128), placement="tp-sharded")
    with pytest.raises(NotImplementedError, match="fail-closed"):
        check_capability(provider, sharded)


def test_reference_provider_declares_every_supported_mode() -> None:
    caps = ReferenceProvider().capabilities()
    assert set(caps["fusion_modes"]) == set(oracle.SUPPORTED_FUSION)
    assert set(caps["trainability"]) == set(oracle.SUPPORTED_TRAINABILITY)


def test_golden_manifest_anchor() -> None:
    """CI anchor: regenerated golden hashes must match the committed manifest.

    A failure here means the oracle's bytes drifted (a torch RNG/libm change,
    or an intentional contract change) -- regenerate with
    ``python -m rl_engine.mhc.fixtures --write-manifest`` and review the diff.
    """
    assert fixtures.load_manifest() == fixtures.golden_manifest()
