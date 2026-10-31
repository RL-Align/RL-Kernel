# SPDX-License-Identifier: Apache-2.0
"""CPU-safe complete-group, contract and autograd lifetime tests."""

from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.kv_compressor_c128 import C128Contract, C128Error
from rl_engine.integrations.common.kv_compressor_c128 import bind_c128_compressor
from rl_engine.models.deepseek_v4.kv_compressor_c128 import C128Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c128_helpers import (
    ReferenceC128Backend,
    byte_equal,
    fixture,
    reference,
)

pytestmark = pytest.mark.unit


def operator(**kwargs):
    return C128Compressor(C128Contract(**kwargs), backend=ReferenceC128Backend())


@pytest.mark.parametrize("rows", [0, 1, 63, 127, 128, 129, 255, 256, 257, 384])
def test_only_complete_128_groups_produce_rows(rows):
    op = operator()
    x, ids, a, n, c, s = fixture(rows)
    bound = op.bind(x, ids, version="v1")
    out = op(bound, a, n, c, s, version="v1")
    assert out.values.shape == (rows // 128, 512)
    assert bound.logical_keys == tuple((7, i) for i in range(rows // 128))
    assert bound.incomplete_groups == (() if rows % 128 == 0 else ((7, rows // 128),))
    assert out.version == "v1" and out.contract_fingerprint == op.contract.fingerprint
    assert torch.isfinite(out.values).all()


def test_nonzero_global_start_preserves_rope_position_and_ignores_partial_edges():
    op = operator()
    x, ids, a, n, co, si = fixture(256, start=64)
    b = op.bind(x, ids, version="v1")
    assert b.logical_keys == ((7, 1),)
    assert b.incomplete_groups == ((7, 0), (7, 2))
    expected = reference(x[64:192].float(), a, n, torch.tensor([1]), co, si, 1e-6).bfloat16()
    byte_equal(op(b, a, n, co, si, version="v1").values, expected)


@pytest.mark.parametrize("rank,expected", [(0, ()), (1, ((7, 0),)), (2, ((7, 1),))])
def test_complete_cross_cp_groups_have_one_last_token_owner(rank, expected):
    op = operator()
    x, ids, *_ = fixture()
    owners = torch.cat((torch.zeros(80), torch.ones(80), torch.full((96,), 2))).long()
    b = op.bind(x, ids, version="v1", owners=owners, cp_size=3, rank=rank)
    assert b.logical_keys == expected
    assert len(b.logical_keys) + len(b.unowned_groups) == 2
    missing = op.bind(
        x[80:160], ids[80:160], version="v1", owners=owners[80:160], cp_size=3, rank=rank
    )
    assert missing.logical_keys == ()
    assert len(missing.incomplete_groups) == 2


def test_padding_permutation_requests_and_snapshot_isolation():
    op = operator()
    x, ids, a, n, c, s = fixture()
    expected = op(op.bind(x, ids, version="v1"), a, n, c, s, version="v1").values
    ids2 = ids.clone()
    ids2[:, 0] = 99
    combined = torch.cat((x, x, torch.full((3, 1024), float("nan")).bfloat16()))
    metadata = torch.cat((ids, ids2, torch.full((3, 2), -1, dtype=torch.int64)))
    perm = torch.randperm(combined.shape[0], generator=torch.Generator().manual_seed(2))
    combined, metadata = combined[perm].contiguous(), metadata[perm]
    b = op.bind(combined, metadata, version="v1")
    combined.data.zero_()  # owned pack isolates even an untracked caller alias
    metadata.zero_()
    actual = op(b, a, n, c, s, version="v1").values
    byte_equal(actual[:2], expected)
    byte_equal(actual[2:], expected)
    b.values.zero_()
    with pytest.raises(C128Error, match="storage changed"):
        op(b, a, n, c, s, version="v1")


@pytest.mark.parametrize(
    "change",
    [
        {"head_dim": 128},
        {"ratio": 4},
        {"rope_dim": 32},
        {"rms_eps": 0},
        {"rms_eps": float("nan")},
        {"rms_eps": True},
        {"max_groups": False},
        {"max_input_rows": 2**31},
        {"max_groups": 2**24},
        {"max_backward_groups": 0},
        {"max_block_table_bytes": -1},
        {"backend_id": "auto"},
        {"output_dtype": "fp32"},
        {"graph_capture": True},
    ],
)
def test_invalid_contract(change):
    with pytest.raises(C128Error):
        replace(C128Contract(), **change)


@pytest.mark.parametrize(
    "bad",
    [
        "float",
        "width",
        "strides",
        "id_dtype",
        "id_shape",
        "partial_padding",
        "negative",
        "overflow",
        "duplicate",
        "cp_missing",
        "cp_bool",
        "rank",
        "owner_dtype",
        "owner_range",
        "owner_shape",
        "version",
        "nan",
        "infinite",
        "too_large",
    ],
)
def test_bad_input_or_metadata_fail_before_provider(bad):
    op = operator()
    x, ids, *_ = fixture()
    args = {"version": "v1"}
    if bad == "float":
        x = x.float()
    elif bad == "width":
        x = x[:, :512].contiguous()
    elif bad == "strides":
        x = x.T.contiguous().T
    elif bad == "id_dtype":
        ids = ids.int()
    elif bad == "id_shape":
        ids = ids.flatten()
    elif bad == "partial_padding":
        ids[0, 0] = -1
    elif bad == "negative":
        ids[0, 1] = -2
    elif bad == "overflow":
        ids[0, 1] = 2**31
    elif bad == "duplicate":
        ids[1] = ids[0]
    elif bad == "cp_missing":
        args["cp_size"] = 2
    elif bad == "cp_bool":
        args["cp_size"] = True
    elif bad == "rank":
        args["rank"] = 1
    elif bad == "owner_dtype":
        args["owners"] = torch.zeros(256)
    elif bad == "owner_range":
        args["owners"] = torch.ones(256, dtype=torch.int64)
    elif bad == "owner_shape":
        args["owners"] = torch.zeros((256, 1), dtype=torch.int64)
    elif bad == "version":
        args["version"] = ""
    else:
        x[0, 0] = {"nan": float("nan"), "infinite": float("inf"), "too_large": 2.0**31}[bad]
    with pytest.raises(C128Error):
        op.bind(x, ids, **args)
    assert op.backend.calls == 0


@pytest.mark.parametrize(
    "change",
    [
        {"max_groups": 1},
        {"max_input_rows": 255},
        {"max_block_table_bytes": 4},
    ],
)
def test_allocation_budgets(change):
    x, ids, *_ = fixture()
    with pytest.raises(C128Error, match="capacity|budget"):
        operator(**change).bind(x, ids, version="v1")


@pytest.mark.parametrize(
    "bad",
    [
        "ape_dtype",
        "ape_shape",
        "ape_inf",
        "norm_shape",
        "norm_nan",
        "rope_shape",
        "rope_dtype",
        "rope_short",
        "rope_range",
        "rope_nan",
        "rope_grad",
        "version",
    ],
)
def test_parameter_and_rope_contract(bad):
    op = operator()
    x, ids, a, n, c, s = fixture()
    b = op.bind(x, ids, version="v1")
    version = "v1"
    if bad == "ape_dtype":
        a = a.bfloat16()
    elif bad == "ape_shape":
        a = a[:127]
    elif bad == "ape_inf":
        a[0, 0] = float("inf")
    elif bad == "norm_shape":
        n = n[:511]
    elif bad == "norm_nan":
        n[0] = float("nan")
    elif bad == "rope_shape":
        c = c[:, :16]
    elif bad == "rope_dtype":
        c = c.float()
    elif bad == "rope_short":
        c, s = c[:1], s[:1]
    elif bad == "rope_range":
        c[128, 0] = 2
    elif bad == "rope_nan":
        s[0, 0] = float("nan")
    elif bad == "rope_grad":
        c.requires_grad_()
    else:
        version = "v2"
    with pytest.raises(C128Error):
        op(b, a, n, c, s, version=version)
    assert op.backend.calls == 0


def run_partitioned(op, chunks, *, reverse=False, rows=384):
    x, ids, a, n, co, si = fixture(rows, requires_grad=True)
    upstream = torch.arange((rows // 128) * 512).reshape(-1, 512).remainder(13).bfloat16() / 13
    with canonical_backward_session() as session:
        calls = []
        for start, end in chunks:
            part = x[start:end]
            b = op.bind(part.contiguous(), ids[start:end], version="v1")
            out = op(b, a, n, co, si, version="v1", parameter_id="layer3.main.c128")
            calls.append((out, upstream[start // 128 : end // 128]))
        for out, grad in reversed(calls) if reverse else calls:
            out.values.backward(grad)
        session.validate_complete()
    return x.grad, a.grad, n.grad


def test_backward_canonical_session_is_bitwise_across_microbatches_and_order():
    op = operator()
    baseline = run_partitioned(op, [(0, 384)])
    for chunks in ([(0, 128), (128, 384)], [(256, 384), (0, 256)]):
        for reverse in (False, True):
            actual = run_partitioned(op, chunks, reverse=reverse)
            for a, b in zip(actual, baseline, strict=True):
                byte_equal(a, b)


def test_empty_backward_and_input_only_backward():
    op = operator()
    x, ids, a, n, co, si = fixture(127, requires_grad=True)
    with canonical_backward_session() as session:
        out = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1", parameter_id="empty")
        out.values.sum().backward()
        session.validate_complete()
    assert all(torch.count_nonzero(g) == 0 for g in (x.grad, a.grad, n.grad))
    x, ids, a, n, co, si = fixture(129)
    x.requires_grad_()
    out = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1")
    out.values.sum().backward()
    assert torch.count_nonzero(x.grad[:128]) > 0
    assert torch.count_nonzero(x.grad[128:]) == 0


def test_canonical_session_requires_identity_version_and_unique_groups():
    op = operator()
    x, ids, a, n, c, s = fixture(requires_grad=True)
    b = op.bind(x, ids, version="v1")
    with pytest.raises(C128Error, match="canonical_backward_session"):
        op(b, a, n, c, s, version="v1", parameter_id="layer0")
    with canonical_backward_session():
        op(b, a, n, c, s, version="v1", parameter_id="layer0")
        with pytest.raises(C128Error, match="duplicate"):
            op(b, a, n, c, s, version="v1", parameter_id="layer0")
        with pytest.raises(C128Error, match="identity"):
            op(b, a.clone(), n, c, s, version="v1", parameter_id="layer0")
    limited = operator(max_backward_groups=1)
    with pytest.raises(C128Error, match="budget"):
        limited(limited.bind(x, ids, version="v1"), a, n, c, s, version="v1")
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(C128Error, match="training source"):
        b.validate()


def test_implementation_binding_and_real_adapter_exports():
    from rl_engine.integrations.engines.rollout.vllm.kv_compressor_c128 import (
        bind_c128_compressor as rollout,
    )
    from rl_engine.integrations.engines.train.megatron.kv_compressor_c128 import (
        bind_c128_compressor as train,
    )

    op = operator()
    for factory in (rollout, train, bind_c128_compressor):
        other = factory(
            contract=op.contract,
            backend=ReferenceC128Backend(),
            expected_implementation=op.implementation_fingerprint,
        )
        assert other.implementation_fingerprint == op.implementation_fingerprint
    with pytest.raises(C128Error, match="fingerprint"):
        C128Compressor(op.contract, backend=ReferenceC128Backend(), expected_implementation="wrong")
    assert op.contract.arithmetic["position"] == "global compressed group ID * 128"
