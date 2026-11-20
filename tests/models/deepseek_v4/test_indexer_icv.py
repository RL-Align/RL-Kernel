# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.indexer_icv import IndexerICVContract, IndexerICVError
from rl_engine.integrations.common.indexer_icv import bind_global_icv_keys, bind_icv_queries
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV, QuantizedICVRows
from tests.models.deepseek_v4.icv_helpers import ReferenceICVBackend, ids, payload, byte_equal

pytestmark = pytest.mark.unit


def small(**kwargs):
    return IndexerICVContract(
        num_heads=3, head_dim=33, query_tile_rows=3, key_tile_rows=5, **kwargs
    )


def bind_pair(op, m=7, n=11, request=7, version="v1"):
    c = op.contract
    q = bind_icv_queries(
        op, *payload(c, m, q=True), ids(request, m), layout=c.layout, version=version
    )
    k = bind_global_icv_keys(
        op,
        *payload(c, n, seed=1),
        ids(request, n),
        layout=c.layout,
        version=version,
        request_key_counts={request: n},
    )
    return q, k


@pytest.mark.parametrize(
    "change",
    [
        {"num_heads": 0},
        {"num_heads": 129},
        {"head_dim": 0},
        {"head_dim": True},
        {"head_dim": 257},
        {"precision": "bf16"},
        {"precision": "fp8"},
        {"fp8_encoding": "e4m3fn"},
        {"scale_location": "weights"},
        {"postprocess": "relu"},
        {"output_dtype": "bf16"},
        {"graph_capture": True},
        {"backend_id": "auto"},
        {"key_tile_rows": 0},
        {"max_prepared_key_bytes": 0},
        {"max_output_bytes": 1},
    ],
)
def test_contract_rejects_unsupported(change):
    with pytest.raises(IndexerICVError):
        replace(small(), **change)


@pytest.mark.parametrize("m,n", [(0, 0), (0, 11), (7, 0), (1, 1), (7, 11), (9, 17)])
def test_tiled_and_materialized_outputs(m, n):
    op = IndexerICV(small(), backend=ReferenceICVBackend())
    q, k = bind_pair(op, m, n)
    blocks = op.materialize(q, k)
    if not m:
        assert blocks == ()
        assert list(op.iter_tiles(q, k)) == []
        return
    assert len(blocks) == 1 and blocks[0].values.shape == (m, 3, n)
    expected = op.backend.dot(
        op.backend.prepare(q.operand.data, q.operand.scales, op.contract)[0],
        k.requests[7].values,
        None,
        None,
        op.contract,
    )
    byte_equal(blocks[0].values, expected)
    for tile in op.iter_tiles(q, k):
        byte_equal(
            tile.values, expected[list(tile.query_physical_rows), :, tile.key_start : tile.key_end]
        )
        assert tile.values.numel() * 4 <= op.contract.max_output_bytes


def test_materialization_budget_checked_before_any_gemm():
    c = small(max_output_bytes=3 * 3 * 5 * 4)
    backend = ReferenceICVBackend()
    op = IndexerICV(c, backend=backend)
    q, k = bind_pair(op)
    with pytest.raises(IndexerICVError, match="use iter_tiles"):
        op.materialize(q, k)
    assert backend.calls == 0
    assert len(list(op.iter_tiles(q, k))) == 9


def test_prepared_key_budget():
    c = small(max_prepared_key_bytes=32)
    op = IndexerICV(c, backend=ReferenceICVBackend())
    with pytest.raises(IndexerICVError, match="prepared K"):
        bind_pair(op)


def test_incomplete_global_keys_duplicate_keys_and_cp_owner_fail():
    c = small()
    op = IndexerICV(c, backend=ReferenceICVBackend())
    data, scale = payload(c, 4)
    common = dict(layout=c.layout, version="v1", request_key_counts={7: 8})
    with pytest.raises(IndexerICVError, match="global K"):
        bind_global_icv_keys(op, data, scale, ids(7, 4), **common)
    common["request_key_counts"] = {7: 4}
    with pytest.raises(IndexerICVError, match="duplicate"):
        bind_global_icv_keys(op, data, scale, torch.tensor([[7, 0]] * 4), **common)
    with pytest.raises(IndexerICVError, match="ownership"):
        bind_global_icv_keys(op, data, scale, ids(7, 4), cp_size=2, **common)
    with pytest.raises(IndexerICVError, match="owner"):
        bind_global_icv_keys(
            op,
            data,
            scale,
            ids(7, 4),
            cp_size=2,
            key_owners=torch.tensor([0, 1, 2, 0]),
            **common,
        )


def test_cp_global_order_is_logical_not_rank_or_physical():
    c = small()
    op = IndexerICV(c, backend=ReferenceICVBackend())
    data, scale = payload(c, 11)
    key_ids = ids(7, 11)
    order = torch.tensor([8, 0, 4, 9, 1, 5, 10, 2, 6, 3, 7])
    a = bind_global_icv_keys(
        op, data, scale, key_ids, layout=c.layout, version="v1", request_key_counts={7: 11}
    )
    b = bind_global_icv_keys(
        op,
        data[order],
        scale[order],
        key_ids[order],
        layout=c.layout,
        version="v1",
        request_key_counts={7: 11},
        cp_size=4,
        key_owners=order % 4,
    )
    byte_equal(a.requests[7].values, b.requests[7].values)
    assert b.cp_size == 4


def test_alias_mutation_version_mismatch_and_workspace_poisoning():
    op = IndexerICV(small(), backend=ReferenceICVBackend())
    q, k = bind_pair(op)
    q.operand.data[0, 0, 0] ^= 1
    with pytest.raises(IndexerICVError, match="changed"):
        op.materialize(q, k)
    q, k = bind_pair(op)
    k.requests[7].values[0, 0] = 0
    with pytest.raises(IndexerICVError, match="workspace"):
        op.materialize(q, k)
    q, k = bind_pair(op)
    q_other, _ = bind_pair(op, version="v2")
    with pytest.raises(IndexerICVError, match="version mismatch"):
        op.materialize(q_other, k)


def test_stream_detects_input_mutation_between_tiles():
    op = IndexerICV(small(), backend=ReferenceICVBackend())
    q, k = bind_pair(op)
    iterator = op.iter_tiles(q, k)
    next(iterator)
    k.operand.scales[0, 0] = 127
    with pytest.raises(IndexerICVError, match="changed"):
        next(iterator)


def test_inference_tensors_are_snapshotted_safely():
    c = small()
    op = IndexerICV(c, backend=ReferenceICVBackend())
    with torch.inference_mode():
        data, scale = payload(c, 3, q=True)
        q = bind_icv_queries(op, data, scale, ids(7, 3), layout=c.layout, version="v1")
        data.zero_()
    q.validate()
    assert not q.operand.data.is_inference()


def test_padding_and_two_requests_do_not_mix():
    c = small()
    op = IndexerICV(c, backend=ReferenceICVBackend())
    q = bind_icv_queries(
        op,
        *payload(c, 4, q=True),
        torch.tensor([[7, 4], [-1, -1], [8, 2], [7, 5]]),
        layout=c.layout,
        version="v1",
    )
    k = bind_global_icv_keys(
        op,
        *payload(c, 4),
        torch.tensor([[8, 0], [7, 1], [-1, -1], [7, 0]]),
        layout=c.layout,
        version="v1",
        request_key_counts={7: 2, 8: 1},
    )
    blocks = op.materialize(q, k)
    assert [(b.request_id, b.values.shape) for b in blocks] == [(7, (2, 3, 2)), (8, (1, 3, 1))]
    assert blocks[0].query_positions == (4, 5)
    assert blocks[0].query_physical_rows == (0, 3)


@pytest.mark.parametrize("bad_scale", [0, 1, 253, 254, 255])
def test_mxfp4_unsupported_exponents_rejected(bad_scale):
    c = small()
    data, scale = payload(c, 1)
    scale[0, 0] = bad_scale
    with pytest.raises(IndexerICVError, match="exact-decode"):
        QuantizedICVRows.bind(data, scale, role="k", version="v1", layout=c.layout, contract=c)


def test_fused_scales_swizzle_and_gradient_are_not_silently_accepted():
    c = small(precision="fp8", fp8_encoding="e4m3fnuz")
    op = IndexerICV(c, backend=ReferenceICVBackend())
    data, scale = payload(c, 1, q=True)
    with pytest.raises(IndexerICVError, match="independent Q scales"):
        bind_icv_queries(op, data, None, ids(7, 1), layout=c.layout, version="v1")
    with pytest.raises(IndexerICVError, match="swizzled"):
        bind_icv_queries(op, data, scale, ids(7, 1), layout="gfx950_preshuffled", version="v1")
    scale.requires_grad_(True)
    with pytest.raises(IndexerICVError, match="item 26"):
        bind_icv_queries(op, data, scale, ids(7, 1), layout=c.layout, version="v1")


@pytest.mark.parametrize("value", [0.0, -1.0, 1.5, float("inf"), float("nan"), 2.0**31])
def test_fp8_invalid_scale(value):
    c = small(precision="fp8", fp8_encoding="e4m3fnuz")
    data, scale = payload(c, 1)
    scale[0, 0] = value
    with pytest.raises(IndexerICVError):
        QuantizedICVRows.bind(data, scale, role="k", version="v1", layout=c.layout, contract=c)


def test_fingerprint_blocks_mismatched_consumers():
    a = IndexerICV(small(), backend=ReferenceICVBackend())
    b = IndexerICV(
        small(), backend=ReferenceICVBackend(), expected_implementation=a.implementation_fingerprint
    )
    assert a.implementation_fingerprint == b.implementation_fingerprint
    with pytest.raises(IndexerICVError, match="fingerprint"):
        IndexerICV(
            replace(small(), head_dim=32),
            backend=ReferenceICVBackend(),
            expected_implementation=a.implementation_fingerprint,
        )
