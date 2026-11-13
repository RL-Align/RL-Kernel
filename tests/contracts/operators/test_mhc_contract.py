# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU tests for the DeepSeek-V4 mHC contract.

The arithmetic it records is measured on ROCm in tests/backends/rocm/test_rocm_mhc_*.py.
"""

import inspect

import pytest
import torch

from rl_engine.backends.rocm.mhc import triton_mhc
from rl_engine.contracts.operators import mhc as contract
from rl_engine.ops.mhc import training
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
    assert info["batch_invariance"]["backward"] is True
    assert info["batch_invariance"]["validated_on"] == "gfx942"
    assert info["arithmetic"]["downcast_points"]
    assert info["arithmetic"]["backward"]


def test_describe_returns_copies():
    info = contract.describe("mhc_pre")
    info["shapes"]["residual"] = "changed"
    info["arithmetic"]["arithmetic_constants"]["NUM_KSPLIT"] = 1
    info["batch_invariance"]["forward"] = False

    fresh = contract.describe("mhc_pre")
    assert fresh["shapes"]["residual"] == "[..., n, C] bf16"
    assert fresh["arithmetic"]["arithmetic_constants"] == contract.ARITHMETIC_CONSTANTS
    assert fresh["batch_invariance"]["forward"] is True


def test_describe_rejects_unknown_ops():
    with pytest.raises(contract.MHCContractError, match="unknown mHC operator"):
        contract.describe("mhc_tail")


def test_arithmetic_constants_match_the_rocm_backend():
    assert contract.ARITHMETIC_CONSTANTS == {
        "NUM_KSPLIT": triton_mhc.NUM_KSPLIT,
        "BLOCK_K": triton_mhc.BLOCK_K,
        **triton_mhc.DOT_OPTIONS,
    }
    assert contract.HC_MULT == triton_mhc.SUPPORTED_N
    assert contract.BACKWARD_ARITHMETIC_CONSTANTS == {"DOT_BLOCK": triton_mhc.DOT_BLOCK}


def test_defaults_match_the_reference_and_the_rocm_backend():
    # The ROCm ops also take logical_keys and parameter_id (default None), used only
    # to train fn, scale and base; they are not arithmetic.
    def arithmetic_defaults(fn):
        return {
            k: v for k, v in _defaults(fn).items() if k not in ("logical_keys", "parameter_id")
        }

    pre = {
        "rms_eps": contract.RMS_EPS,
        "hc_pre_eps": contract.HC_PRE_EPS,
        "hc_sinkhorn_eps": contract.HC_SINKHORN_EPS,
        "hc_post_mult_value": contract.HC_POST_MULT_VALUE,
        "sinkhorn_repeat": contract.SINKHORN_REPEAT,
    }

    assert arithmetic_defaults(triton_mhc.TritonMHCPreOp.__call__) == pre
    # The reference fixes eps, the post multiplier and n; its default round count is 20.
    assert (training.EPS, training.N) == (contract.RMS_EPS, contract.HC_MULT)
    assert contract.RMS_EPS == contract.HC_PRE_EPS == contract.HC_SINKHORN_EPS
    assert _defaults(training.mhc_pre)["sinkhorn_repeat"] == contract.SINKHORN_REPEAT


def test_pre_fn_rows_matches_the_reference_shape_check():
    n, c = contract.HC_MULT, 8
    rows = contract.pre_fn_rows()
    assert rows == 24
    residual = torch.zeros(1, n, c, dtype=torch.bfloat16)
    training.mhc_pre(
        residual, torch.zeros(rows, n * c), torch.ones(3), torch.zeros(rows), backend="reference"
    )
    with pytest.raises(ValueError, match="expected FP32 parameter"):
        training.mhc_pre(
            residual,
            torch.zeros(rows + 1, n * c),
            torch.ones(3),
            torch.zeros(rows),
            backend="reference",
        )


def test_reference_sinkhorn_is_the_recorded_asymmetric_form():
    logits = torch.randn(4, 4, generator=torch.Generator().manual_seed(0))
    eps = contract.HC_SINKHORN_EPS
    comb = torch.softmax(logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # With scale 1, base 0 and r 1 the comb logits are the last 16 projection values.
    proj = torch.cat((torch.zeros(8), logits.flatten()))
    got = training.mix_state(torch.tensor(1.0), proj, torch.ones(3), torch.zeros(24), False, 2)[4]
    torch.testing.assert_close(got, comb, atol=1e-6, rtol=0)


@pytest.mark.parametrize("op", contract.OPS)
def test_semantic_descriptor_agrees_on_batch_invariance(op):
    (descriptor,) = [d for d in _default_semantic_descriptors() if d.semantic_op == op]
    props = descriptor.determinism_or_alignment_properties

    assert descriptor.supported_devices == frozenset({"rocm"})
    assert descriptor.supported_dtypes == frozenset({"bfloat16"})
    assert props["batch_invariant"] is contract.BATCH_INVARIANCE["forward"]
    assert props["backward_batch_invariant"] is contract.BATCH_INVARIANCE["backward"]
