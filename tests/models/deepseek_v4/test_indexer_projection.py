# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.indexer_projection import (
    INDEXER_BACKEND_ID,
    IndexerProjectionContract,
    IndexerProjectionError,
)
from rl_engine.integrations.engines.rollout.vllm.indexer_projection import (
    bind_indexer_projection as bind_vllm,
)
from rl_engine.integrations.engines.train.megatron.indexer_projection import (
    bind_indexer_projection as bind_megatron,
)
from rl_engine.models.deepseek_v4.indexer_projection import IndexerLogicalRows, IndexerProjection
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

pytestmark = pytest.mark.unit


class ReferenceBackend:
    """CPU arithmetic for contract tests only; never selected as a production fallback."""

    backend_id = INDEXER_BACKEND_ID
    device_type = "cpu"

    def __init__(self):
        self.calls = 0

    def linear(self, x, weight, *, output_dtype, role):
        self.calls += 1
        return (x.float() @ weight.float().T).to(output_dtype)

    def input_gradient(self, grad, weight):
        return (grad.float() @ weight.float()).to(torch.bfloat16)

    def weight_gradient(self, rows, grads):
        return (grads.float().T @ rows.float()).to(torch.bfloat16)


def small_contract(**kwargs):
    return IndexerProjectionContract(
        hidden_size=16, q_lora_rank=8, num_heads=2, head_dim=4, **kwargs
    )


def values(m=7, train=False):
    generator = torch.Generator().manual_seed(1313)
    tensors = [
        torch.randn(shape, generator=generator, dtype=torch.bfloat16)
        for shape in ((m, 8), (m, 16), (8, 8), (2, 16))
    ]
    return [tensor.requires_grad_(train) for tensor in tensors]


def keys(m):
    return IndexerLogicalRows.bind(
        torch.stack(
            (
                torch.zeros(m, dtype=torch.int64),
                torch.arange(m, dtype=torch.int64),
            ),
            dim=1,
        )
    )


@pytest.mark.parametrize("q_output", ["bf16", "fp32"])
@pytest.mark.parametrize("weights_output", ["bf16", "fp32"])
@pytest.mark.parametrize("m", [0, 1, 7, 33])
def test_output_geometry_dtype_and_cast_location(m, q_output, weights_output):
    c = small_contract(q_output=q_output, weights_output=weights_output)
    x, h, wq, ww = values(m)
    op = IndexerProjection(c, layer_id="layer0", backend=ReferenceBackend())
    result = op(x, h, wq, ww)
    expected_q = (x.float() @ wq.float().T).to(
        getattr(torch, "bfloat16" if q_output == "bf16" else "float32")
    )
    expected_w = (h.float() @ ww.float().T).to(
        getattr(torch, "bfloat16" if weights_output == "bf16" else "float32")
    )
    assert result.q.shape == (m, 2, 4)
    assert torch.equal(result.q.reshape(m, 8), expected_q)
    assert torch.equal(result.weights, expected_w)
    assert result.contract_fingerprint == c.fingerprint
    if m and weights_output == "fp32":
        assert not torch.equal(result.weights, result.weights.bfloat16().float())


@pytest.mark.parametrize(
    "bad",
    [
        {"hidden_size": 0},
        {"q_lora_rank": True},
        {"num_heads": -1},
        {"head_dim": 0},
        {"max_rows": 0},
        {"tp_size": 2},
        {"cp_size": 2},
        {"tp_size": True},
        {"graph_capture": True},
        {"q_precision": "auto"},
        {"q_output": "fp16"},
        {"weights_output": "auto"},
        {"backend_id": "fallback"},
        {"backward_id": "native_te_auto"},
        {"q_precision": "block_fp8_ue8m0"},
    ],
)
def test_contract_fail_closed(bad):
    with pytest.raises(IndexerProjectionError):
        replace(small_contract(), **bad)


def test_wrong_shapes_dtype_layout_capacity_fail_before_gemm():
    backend = ReferenceBackend()
    op = IndexerProjection(small_contract(max_rows=7), layer_id="layer0", backend=backend)
    x, h, wq, ww = values(8)
    with pytest.raises(IndexerProjectionError, match="capacity"):
        op(x, h, wq, ww)
    x, h, wq, ww = values()
    for args in ((x.float(), h, wq, ww), (x, h[:1], wq, ww), (x, h, wq.t(), ww)):
        with pytest.raises(IndexerProjectionError):
            op(*args)
    assert backend.calls == 0


def test_fp8_metadata_is_not_silently_used_as_plain_bf16():
    backend = ReferenceBackend()
    op = IndexerProjection(small_contract(), layer_id="layer0", backend=backend)
    x, h, wq, ww = values()
    wq._fp8_source_scales = torch.ones(1)
    with pytest.raises(IndexerProjectionError, match="FP8 source"):
        op(x, h, wq, ww)
    assert backend.calls == 0


def test_training_requires_canonical_metadata_and_session():
    op = IndexerProjection(small_contract(), layer_id="layer0", backend=ReferenceBackend())
    tensors = values(train=True)
    with pytest.raises(IndexerProjectionError, match="canonical"):
        op(*tensors, logical_rows=keys(7))
    with canonical_backward_session():
        with pytest.raises(IndexerProjectionError, match="logical rows"):
            op(*tensors)


def test_master_mutation_is_rejected_by_saved_tensor_versions():
    op = IndexerProjection(small_contract(), layer_id="layer0", backend=ReferenceBackend())
    x, h, wq, ww = values(train=True)
    with canonical_backward_session():
        result = op(x, h, wq, ww, logical_rows=keys(7))
        with torch.no_grad():
            wq.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace"):
            (result.q.float().sum() + result.weights.sum()).backward()


def test_logical_snapshot_and_duplicate_keys():
    source = keys(7).keys
    bound = IndexerLogicalRows.bind(source)
    source.fill_(-1)
    bound.validate(7, torch.device("cpu"))
    bound.keys[0, 0] = 4
    with pytest.raises(IndexerProjectionError, match="mutated"):
        bound.validate(7, torch.device("cpu"))
    with pytest.raises(IndexerProjectionError, match="duplicate"):
        IndexerLogicalRows.bind(torch.zeros((2, 2), dtype=torch.int64))
    with pytest.raises(IndexerProjectionError):
        IndexerLogicalRows.bind(torch.zeros((2, 2), dtype=torch.float32))


def test_duplicate_tokens_across_microbatches_rejected():
    op = IndexerProjection(small_contract(), layer_id="layer0", backend=ReferenceBackend())
    tensors = values(train=True)
    with canonical_backward_session():
        op(*tensors, logical_rows=keys(7))
        with pytest.raises(IndexerProjectionError, match="repeated"):
            op(*tensors, logical_rows=keys(7))


def test_same_parameter_id_cannot_refer_to_different_weights():
    op = IndexerProjection(small_contract(), layer_id="layer0", backend=ReferenceBackend())
    tensors = values(train=True)
    with canonical_backward_session():
        op(*tensors, logical_rows=keys(7))
        other = values(train=True)
        rows = IndexerLogicalRows.bind(torch.tensor([[1, i] for i in range(7)], dtype=torch.int64))
        with pytest.raises(IndexerProjectionError, match="another weight"):
            op(*other, logical_rows=rows)


@pytest.mark.parametrize("binder", [bind_vllm, bind_megatron])
def test_framework_adapters_use_the_same_contract_and_parameters(binder):
    x, h, wq, ww = values()
    indexer = SimpleNamespace(
        wq_b=SimpleNamespace(weight=wq, bias=None),
        weights_proj=SimpleNamespace(weight=ww, bias=None),
    )
    bound = binder(
        indexer, contract=small_contract(), layer_id="layer0", backend=ReferenceBackend()
    )
    expected = IndexerProjection(small_contract(), layer_id="layer0", backend=ReferenceBackend())(
        x, h, wq, ww
    )
    actual = bound(x, h)
    assert bound.q_weight is wq and bound.weights_weight is ww
    assert torch.equal(actual.q, expected.q) and torch.equal(actual.weights, expected.weights)
    indexer.wq_b.bias = torch.zeros(8)
    with pytest.raises(IndexerProjectionError, match="bias"):
        binder(indexer, contract=small_contract(), layer_id="layer0", backend=ReferenceBackend())


def test_dtype_fp32_outputs_keep_bf16_master_vjp_rule():
    x, h, wq, ww = values(train=True)
    op = IndexerProjection(
        small_contract(q_output="fp32"), layer_id="layer0", backend=ReferenceBackend()
    )
    gq, gw = torch.randn((7, 8)), torch.randn((7, 2))
    with canonical_backward_session() as session:
        out = op(x, h, wq, ww, logical_rows=keys(7))
        torch.autograd.backward((out.q, out.weights), (gq.view(7, 2, 4), gw))
        session.validate_complete()
    assert torch.equal(x.grad, (gq.bfloat16().float() @ wq.detach().float()).bfloat16())
    assert torch.equal(ww.grad, (gw.bfloat16().float().T @ h.detach().float()).bfloat16())


@pytest.mark.parametrize("m", [0, 3])
def test_empty_and_all_padding_backward(m):
    x, h, wq, ww = values(m, train=True)
    op = IndexerProjection(small_contract(), layer_id="layer0", backend=ReferenceBackend())
    bound = IndexerLogicalRows.bind(torch.full((m, 2), -1, dtype=torch.int64))
    with canonical_backward_session() as session:
        out = op(x, h, wq, ww, logical_rows=bound)
        (out.q.float().sum() + out.weights.sum()).backward()
        session.validate_complete()
    assert not torch.count_nonzero(x.grad)
    assert not torch.count_nonzero(h.grad)
    assert not torch.count_nonzero(wq.grad)
    assert not torch.count_nonzero(ww.grad)


def test_implementation_fingerprint_is_a_real_binding_guard():
    c = small_contract()
    first = IndexerProjection(c, layer_id="L0", backend=ReferenceBackend())
    second = IndexerProjection(
        c,
        layer_id="L1",
        backend=ReferenceBackend(),
        expected_implementation=first.implementation_fingerprint,
    )
    assert second.implementation_fingerprint == first.implementation_fingerprint
    with pytest.raises(IndexerProjectionError, match="fingerprint mismatch"):
        IndexerProjection(
            replace(c, weights_output="bf16"),
            layer_id="L0",
            backend=ReferenceBackend(),
            expected_implementation=first.implementation_fingerprint,
        )
