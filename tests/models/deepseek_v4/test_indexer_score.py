# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.indexer_score import IndexerScoreContract, IndexerScoreError
from rl_engine.integrations.engines.rollout.vllm.indexer_score import contract_from_hf_config
from rl_engine.integrations.engines.train.megatron.indexer_score import (
    contract_from_transformer_config,
)
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore
from tests.models.deepseek_v4.score_helpers import (
    ReferenceScoreBackend,
    normalized_error,
    score_fp64,
    vjp_fp64,
)

pytestmark = pytest.mark.unit

FLASH = IndexerScoreContract(num_heads=64, head_dim=128)


def operator(**kwargs):
    return IndexerScore(IndexerScoreContract(**kwargs), backend=ReferenceScoreBackend())


def inputs(m, h, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, h, n, generator=g) * 4, torch.randn(m, h, generator=g)


@pytest.mark.parametrize(
    "change",
    [
        {"num_heads": 0},
        {"head_dim": -1},
        {"tp_size": 0},
        {"num_heads": 64, "tp_size": 3},
        {"max_backward_keys": 4097},
        {"graph_capture": True},
        {"backend_id": "other"},
        {"num_heads": 64.0},
    ],
)
def test_contract_rejects_invalid_fields(change):
    with pytest.raises(IndexerScoreError):
        replace(FLASH, **change)


def test_contract_has_no_model_defaults():
    with pytest.raises(TypeError):
        IndexerScoreContract()
    with pytest.raises(TypeError):
        IndexerScoreContract(num_heads=64)


def test_scales_come_from_config_and_global_heads():
    c = FLASH
    assert c.softmax_scale == 128**-0.5
    assert c.head_scale == 0.125
    tp = replace(c, tp_size=2)
    assert tp.local_heads == 32 and tp.head_scale == 0.125
    assert replace(c, head_dim=64).softmax_scale == 64**-0.5


def test_framework_adapters_read_their_own_config_fields():
    vllm = contract_from_hf_config(SimpleNamespace(index_n_heads=64, index_head_dim=128))
    mlm = contract_from_transformer_config(
        SimpleNamespace(dsa_indexer_n_heads=64, dsa_indexer_head_dim=128)
    )
    assert vllm == mlm == FLASH


def test_forward_matches_fp64_and_keeps_intermediates():
    op = operator(num_heads=8, head_dim=16)
    icv, w = inputs(5, 8, 13)
    record = op.forward(icv, w)
    assert record.scores.shape == (5, 13) and record.scores.dtype == torch.float32
    assert normalized_error(record.scores, score_fp64(icv, w, op.contract)) < 1e-6
    assert torch.equal(record.relu, torch.relu(icv * 16**-0.5))
    assert torch.equal(record.scaled_weights, w * 8**-0.5)


def test_tp_shard_uses_global_head_scale():
    full = operator(num_heads=8, head_dim=16)
    shard = operator(num_heads=8, head_dim=16, tp_size=2)
    icv, w = inputs(4, 8, 9)
    partials = [shard.forward(icv[:, i : i + 4], w[:, i : i + 4]).scores for i in (0, 4)]
    torch.testing.assert_close(partials[0] + partials[1], full.forward(icv, w).scores)
    with pytest.raises(IndexerScoreError, match="local_heads"):
        shard.forward(icv, w)


@pytest.mark.parametrize(
    "icv, weights, match",
    [
        (torch.zeros(2, 4, 3, dtype=torch.float64), torch.zeros(2, 4), "FP32"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 4, dtype=torch.bfloat16), "FP32"),
        (torch.zeros(2, 4), torch.zeros(2, 4), "ICV must be"),
        (torch.zeros(2, 4, 3), torch.zeros(3, 4), "weights must be"),
        (torch.full((2, 4, 3), float("nan")), torch.zeros(2, 4), "non-finite"),
        (torch.zeros(2, 4, 3), torch.full((2, 4), float("inf")), "non-finite"),
    ],
)
def test_forward_rejects_invalid_inputs(icv, weights, match):
    with pytest.raises(IndexerScoreError, match=match):
        operator(num_heads=4, head_dim=8).forward(icv, weights)


def test_backward_matches_fp64_autograd():
    op = operator(num_heads=8, head_dim=16)
    icv, w = inputs(6, 8, 11, seed=1)
    grad = torch.randn(6, 11)
    d_icv, d_w = op.backward(op.forward(icv, w), grad)
    ref_icv, ref_w = vjp_fp64(icv, w, grad, op.contract)
    assert normalized_error(d_icv.flatten(1), ref_icv.flatten(1)) < 1e-6
    assert normalized_error(d_w, ref_w) < 1e-6


def test_relu_derivative_is_zero_at_zero_and_after_underflow():
    op = operator(num_heads=2, head_dim=128)
    icv = torch.tensor([[[0.0, -1.0, 2.0, 1e-45]] * 2])
    record = op.forward(icv, torch.ones(1, 2))
    assert record.relu[0, 0, 3] == 0
    d_icv, _ = op.backward(record, torch.ones(1, 4))
    assert d_icv[0, :, [0, 1, 3]].eq(0).all() and d_icv[0, :, 2].gt(0).all()


def test_backward_rejects_mutation_foreign_records_and_wide_keys():
    op = operator(num_heads=4, head_dim=8, max_backward_keys=4)
    icv, w = inputs(2, 4, 4)
    record = op.forward(icv, w)
    with pytest.raises(IndexerScoreError, match="queries, keys"):
        op.backward(record, torch.zeros(2, 3))
    other = operator(num_heads=4, head_dim=16)
    with pytest.raises(IndexerScoreError, match="different implementation"):
        other.backward(record, torch.zeros(2, 4))
    record.relu.add_(1)
    with pytest.raises(IndexerScoreError, match="changed after forward"):
        op.backward(record, torch.zeros(2, 4))
    wide = op.forward(*inputs(2, 4, 5))
    with pytest.raises(IndexerScoreError, match="exceed the contract"):
        op.backward(wide, torch.zeros(2, 5))


def test_expected_fingerprint_must_match():
    op = operator(num_heads=4, head_dim=8)
    with pytest.raises(IndexerScoreError, match="fingerprint"):
        IndexerScore(op.contract, backend=ReferenceScoreBackend(), expected_implementation="x")
    same = IndexerScore(
        op.contract,
        backend=ReferenceScoreBackend(),
        expected_implementation=op.implementation_fingerprint,
    )
    assert same.implementation_fingerprint == op.implementation_fingerprint
