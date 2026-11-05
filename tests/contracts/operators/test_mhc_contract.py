# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU tests for the DeepSeek-V4 mHC contract.

The arithmetic it records is measured on ROCm in tests/backends/rocm/test_rocm_mhc_*.py.
"""

import inspect

import pytest
import torch

from rl_engine.backends.rocm.mhc import aiter_mhc
from rl_engine.contracts.operators import mhc as contract
from rl_engine.reference.mhc import NativeMHCHeadOp, NativeMHCPreOp, sinkhorn_dsv4
from rl_engine.runtime.registry import _default_semantic_descriptors


def _defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


@pytest.mark.parametrize("op", contract.OPS)
def test_describe_reports_provenance(op):
    info = contract.describe(op)

    assert info["profile"] == contract.PROFILE
    assert info["op"] == op
    assert info["aiter_commit"] == contract.AITER_COMMIT
    assert info["hc_mult"] == 4
    assert info["batch_invariance"]["forward"] is True
    assert info["batch_invariance"]["backward"] is False
    assert info["batch_invariance"]["validated_on"] == "gfx942"
    assert len(info["open_items"]) == 2
    assert info["arithmetic"]["downcast_points"]


def test_describe_returns_copies():
    info = contract.describe("mhc_pre")
    info["shapes"]["residual"] = "changed"
    info["arithmetic"]["bit_params"].append("changed")
    info["batch_invariance"]["forward"] = False

    fresh = contract.describe("mhc_pre")
    assert fresh["shapes"]["residual"] == "[..., n, C] bf16"
    assert fresh["arithmetic"]["bit_params"] == list(contract.PRE_BIT_PARAMS)
    assert fresh["batch_invariance"]["forward"] is True


def test_describe_rejects_unknown_ops():
    with pytest.raises(contract.MHCContractError, match="unknown mHC operator"):
        contract.describe("mhc_tail")


def test_bit_params_match_the_rocm_backend():
    assert contract.PRE_BIT_PARAMS == aiter_mhc.PRE_BIT_PARAMS
    assert contract.HEAD_BIT_PARAMS == aiter_mhc.HEAD_BIT_PARAMS
    assert contract.HC_MULT == aiter_mhc.SUPPORTED_N


def test_defaults_match_the_reference_and_the_rocm_backend():
    pre = {
        "rms_eps": contract.RMS_EPS,
        "hc_pre_eps": contract.HC_PRE_EPS,
        "hc_sinkhorn_eps": contract.HC_SINKHORN_EPS,
        "hc_post_mult_value": contract.HC_POST_MULT_VALUE,
        "sinkhorn_repeat": contract.SINKHORN_REPEAT,
    }
    head = {"rms_eps": contract.RMS_EPS, "hc_pre_eps": contract.HC_PRE_EPS}

    assert _defaults(NativeMHCPreOp._pre) == pre
    assert _defaults(aiter_mhc.AiterMHCPreOp.__call__) == pre
    assert _defaults(NativeMHCHeadOp.forward) == head
    assert _defaults(aiter_mhc.AiterMHCHeadOp.__call__) == head


def test_pre_fn_rows_matches_the_reference_shape_check():
    n, c = contract.HC_MULT, 8
    rows = contract.pre_fn_rows()
    assert rows == 24
    residual = torch.zeros(1, n, c)
    NativeMHCPreOp()(residual, torch.zeros(rows, n * c), torch.ones(3), torch.zeros(rows))
    with pytest.raises(ValueError, match="fn must have shape"):
        NativeMHCPreOp()(residual, torch.zeros(rows + 1, n * c), torch.ones(3), torch.zeros(rows))


def test_reference_sinkhorn_is_the_recorded_asymmetric_form():
    logits = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(0))
    eps = contract.HC_SINKHORN_EPS
    comb = torch.softmax(logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    assert torch.equal(sinkhorn_dsv4(logits, 2, eps), comb)


@pytest.mark.parametrize("op", contract.OPS)
def test_semantic_descriptor_agrees_on_batch_invariance(op):
    (descriptor,) = [d for d in _default_semantic_descriptors() if d.semantic_op == op]
    props = descriptor.determinism_or_alignment_properties

    assert descriptor.supported_devices == frozenset({"rocm"})
    assert descriptor.supported_dtypes == frozenset({"bfloat16"})
    assert props["batch_invariant"] is contract.BATCH_INVARIANCE["forward"]
    assert props["backward_batch_invariant"] is contract.BATCH_INVARIANCE["backward"]
