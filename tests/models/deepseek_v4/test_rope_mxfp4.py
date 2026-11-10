# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.indexer_q_preprocess import (
    IndexerQPreprocessContract,
    IndexerQPreprocessError,
)
from rl_engine.contracts.operators.partial_rope import PartialRoPEContract, PartialRoPEError
from rl_engine.integrations.engines.rollout.vllm.indexer_q_preprocess import (
    contracts_from_hf_config,
)
from rl_engine.models.deepseek_v4.indexer_q_preprocess import IndexerQPreprocess
from rl_engine.models.deepseek_v4.partial_rope import PartialRoPE
from tests.models.deepseek_v4.icv_helpers import decode_fp4
from tests.models.deepseek_v4.rope_mxfp4_helpers import (
    ReferenceIndexerQBackend,
    ReferenceRoPEBackend,
    angles,
    mxfp4_even,
    rope_fp64,
)

pytestmark = pytest.mark.unit

ROPE = PartialRoPEContract(head_dim=512, rope_dim=64)
QPRE = IndexerQPreprocessContract(num_heads=4)


@pytest.mark.parametrize(
    "change",
    [
        {"head_dim": 256},
        {"rope_dim": 32},
        {"head_dim": 128.0},
        {"graph_capture": True},
        {"backend_id": "other"},
    ],
)
def test_rope_contract_rejects_invalid_fields(change):
    with pytest.raises(PartialRoPEError):
        replace(ROPE, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"num_heads": 0},
        {"head_dim": 512},
        {"rope_dim": 32},
        {"scale_rule": "ceil"},
        {"precision": "fp8"},
        {"graph_capture": True},
        {"backend_id": "other"},
    ],
)
def test_indexer_q_contract_rejects_invalid_fields(change):
    with pytest.raises(PartialRoPEError):
        replace(QPRE, **change)


def test_contracts_have_no_model_defaults_and_adapter_reads_config():
    with pytest.raises(TypeError):
        PartialRoPEContract()
    with pytest.raises(TypeError):
        IndexerQPreprocessContract()
    config = SimpleNamespace(
        head_dim=512, qk_rope_head_dim=64, index_n_heads=64, index_head_dim=128
    )
    rope, q = contracts_from_hf_config(config)
    assert rope == ROPE and q == replace(QPRE, num_heads=64)
    assert QPRE.rope == PartialRoPEContract(head_dim=128, rope_dim=64)
    assert issubclass(IndexerQPreprocessError, PartialRoPEError)


def test_reference_rope_autograd_and_padding():
    op = PartialRoPE(ROPE, backend=ReferenceRoPEBackend())
    cos, sin = angles(16)
    x = torch.randn(4, 2, 512).bfloat16().requires_grad_()
    positions = torch.tensor([3, -1, 15, 0])
    y = op(x, positions, cos, sin)
    torch.testing.assert_close(
        y.double(), rope_fp64(x, positions, cos, sin), rtol=0.016, atol=0.016
    )
    assert torch.equal(y[[0, 2, 3], :, :448], x[[0, 2, 3], :, :448]) and y[1].eq(0).all()
    y.backward(torch.ones_like(y))
    assert x.grad[1].eq(0).all()


def test_expected_implementation_is_enforced():
    op = PartialRoPE(ROPE, backend=ReferenceRoPEBackend())
    same = PartialRoPE(
        ROPE, backend=ReferenceRoPEBackend(), expected_implementation=op.implementation_fingerprint
    )
    assert same.implementation_fingerprint == op.implementation_fingerprint
    with pytest.raises(PartialRoPEError, match="fingerprint"):
        PartialRoPE(ROPE, backend=ReferenceRoPEBackend(), expected_implementation="other")
    with pytest.raises(PartialRoPEError, match="identity"):
        PartialRoPE(ROPE, backend=ReferenceIndexerQBackend())


def test_reference_indexer_q_outputs_and_frozen_codes():
    op = IndexerQPreprocess(QPRE, backend=ReferenceIndexerQBackend())
    cos, sin = angles(8)
    q = torch.randn(3, 4, 128).bfloat16().requires_grad_()
    out = op(q, torch.tensor([0, 7, -1]), cos, sin)
    assert out.packed.shape == (3, 4, 64) and out.scales.shape == (3, 4, 4)
    assert not out.packed.requires_grad and out.values.requires_grad
    out.values.sum().backward()
    assert q.grad[2].eq(0).all()
    with pytest.raises(PartialRoPEError, match="non-finite"):
        op(torch.full_like(q, float("nan")), torch.tensor([0, 1, 2]), cos, sin)


def test_mxfp4_oracle_roundtrip_and_known_codes():
    values = torch.zeros(1, 128)
    values[0, :6] = torch.tensor([6.5, -0.75, 1.25, 0.0, 3.0, 2.0])
    packed, scales = mxfp4_even(values.bfloat16())
    assert scales.tolist() == [[127, 0, 0, 0]]
    assert packed[0, :3].tolist() == [0xA7, 0x02, 0x45]
    decoded = decode_fp4(packed, scales, 128)[0, :6].tolist()
    assert decoded == [6.0, -1.0, 1.0, 0.0, 3.0, 2.0]
