# SPDX-License-Identifier: Apache-2.0
"""CPU-safe C4 complete-group, overlap, ownership and VJP tests."""

from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract, C4Error
from rl_engine.integrations.common.kv_compressor_c4 import bind_c4_compressor
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c4_helpers import ReferenceC4Backend, byte_equal, fixture, reference

pytestmark = pytest.mark.unit


def operator(**kwargs):
    return C4Compressor(C4Contract(**kwargs), backend=ReferenceC4Backend())


@pytest.mark.parametrize("groups", [0, 1, 2, 3, 4, 7, 16])
def test_only_complete_current_groups_emit_and_first_group_pads_previous(groups):
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(groups)
    tail, _, _, _, _, _ = fixture(1, seed=77)
    tail_ids = torch.stack(
        (
            torch.full((3,), 7, dtype=torch.int64),
            torch.arange(groups * 4, groups * 4 + 3, dtype=torch.int64),
        ),
        dim=-1,
    )
    values = torch.cat((values, tail[:3]), dim=0)
    ids = torch.cat((ids, tail_ids), dim=0)
    bound = op.bind(values, ids, version="v1")
    output = op(bound, ape, norm, cos, sin, version="v1")
    assert output.values.shape == (groups, 512)
    assert bound.logical_keys == tuple((7, group) for group in range(groups))
    assert (7, groups) in bound.incomplete_groups
    expected = (
        reference(values[: groups * 4].float(), ape, norm, range(groups), cos, sin, 1e-6).bfloat16()
        if groups
        else torch.empty((0, 512), dtype=torch.bfloat16)
    )
    byte_equal(output.values, expected)


def test_overlap_uses_previous_group_first_half_and_current_group_last_half():
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(3, start=0)
    values[0:4, :1024] = 1
    values[0:4, 1024:] = -1
    values[4:8, :1024] = 2
    values[4:8, 1024:] = 3
    values[8:12, :1024] = 4
    values[8:12, 1024:] = 5
    output = op(op.bind(values, ids, version="v1"), ape, norm, cos, sin, version="v1").values
    expected = reference(values, ape, norm, range(3), cos, sin, 1e-6).bfloat16()
    byte_equal(output, expected)
    assert not torch.equal(output[1], output[0])
    assert not torch.equal(output[2], output[1])


@pytest.mark.parametrize("rank,expected", [(0, ((7, 0),)), (1, ((7, 1),)), (2, ((7, 2),))])
def test_cp_final_token_ownership_and_previous_context(rank, expected):
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(3)
    owners = torch.arange(12).remainder(3).long()
    bound = op.bind(values, ids, version="v1", owners=owners, cp_size=3, rank=rank)
    assert bound.logical_keys == expected
    assert len(bound.unowned_groups) == 2
    assert bound.context_indices.shape == (1, 8)
    if rank == 0:
        assert tuple(bound.context_indices_host[0][:4]) == (-1, -1, -1, -1)
    else:
        assert all(index >= 0 for index in bound.context_indices_host[0])


def test_permuted_rows_padding_duplicate_requests_and_snapshot_alias():
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(3)
    expected = op(op.bind(values, ids, version="v1"), ape, norm, cos, sin, version="v1").values
    second, second_ids, *_ = fixture(3, request=99, seed=29)
    merged = torch.cat((values, second, torch.full((3, 2048), float("nan")).bfloat16()))
    metadata = torch.cat((ids, second_ids, torch.full((3, 2), -1, dtype=torch.int64)))
    permutation = torch.randperm(len(metadata), generator=torch.Generator().manual_seed(4))
    bound = op.bind(merged[permutation].contiguous(), metadata[permutation], version="v1")
    merged.data.zero_()
    output = op(bound, ape, norm, cos, sin, version="v1").values
    byte_equal(output[:3], expected)
    bound.context_values.zero_()
    with pytest.raises(C4Error, match="storage changed"):
        op(bound, ape, norm, cos, sin, version="v1")


@pytest.mark.parametrize(
    "change",
    [
        {"head_dim": 128},
        {"ratio": 128},
        {"overlap": 4},
        {"rope_dim": 32},
        {"rms_eps": 0},
        {"rms_eps": float("nan")},
        {"max_groups": 2**24},
        {"max_input_rows": 2**31},
        {"max_backward_groups": 0},
        {"backend_id": "auto"},
        {"output_dtype": "fp32"},
        {"graph_capture": True},
    ],
)
def test_invalid_contract(change):
    with pytest.raises(C4Error):
        replace(C4Contract(), **change)


@pytest.mark.parametrize(
    "bad",
    [
        "dtype",
        "width",
        "ids",
        "partial",
        "negative",
        "duplicate",
        "overflow",
        "version",
        "cp",
        "cp_bool",
        "owner_dtype",
        "owner_range",
        "owner_shape",
        "nan",
        "inf",
        "stride",
    ],
)
def test_invalid_input_never_reaches_backend(bad):
    op = operator()
    values, ids, *_ = fixture(2)
    args = {"version": "v1"}
    if bad == "dtype":
        values = values.float()
    elif bad == "width":
        values = values[:, :1024].contiguous()
    elif bad == "ids":
        ids = ids.int()
    elif bad == "partial":
        ids[0, 0] = -1
    elif bad == "negative":
        ids[0, 1] = -1
    elif bad == "duplicate":
        ids[1] = ids[0]
    elif bad == "overflow":
        ids[0, 1] = 2**31
    elif bad == "version":
        args["version"] = ""
    elif bad == "cp":
        args["cp_size"] = 2
    elif bad == "cp_bool":
        args["cp_size"] = True
    elif bad == "owner_dtype":
        args["owners"] = torch.zeros(8)
    elif bad == "owner_range":
        args["owners"] = torch.full((8,), 3, dtype=torch.int64)
    elif bad == "owner_shape":
        args["owners"] = torch.zeros((8, 1), dtype=torch.int64)
    elif bad in ("nan", "inf"):
        values[0, 0] = float("nan") if bad == "nan" else float("inf")
    else:
        wide = torch.empty((values.shape[0], values.shape[1] * 2), dtype=values.dtype)
        values = wide[:, ::2]
        values.copy_(fixture(2)[0])
    with pytest.raises(C4Error):
        op.bind(values, ids, **args)
    assert op.backend.calls == 0


@pytest.mark.parametrize(
    "bad",
    [
        "ape_dtype",
        "ape_shape",
        "ape_nan",
        "norm_shape",
        "norm_inf",
        "rope_shape",
        "rope_dtype",
        "rope_short",
        "rope_bad",
        "version",
    ],
)
def test_parameter_rope_and_version_contract(bad):
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(2)
    bound = op.bind(values, ids, version="v1")
    version = "v1"
    if bad == "ape_dtype":
        ape = ape.bfloat16()
    elif bad == "ape_shape":
        ape = ape[:3]
    elif bad == "ape_nan":
        ape[0, 0] = float("nan")
    elif bad == "norm_shape":
        norm = norm[:511]
    elif bad == "norm_inf":
        norm[0] = float("inf")
    elif bad == "rope_shape":
        cos = cos[:, :16]
    elif bad == "rope_dtype":
        cos = cos.float()
    elif bad == "rope_short":
        cos, sin = cos[:1], sin[:1]
    elif bad == "rope_bad":
        sin[0, 0] = 2
    else:
        version = "v2"
    with pytest.raises(C4Error):
        op(bound, ape, norm, cos, sin, version=version)
    assert op.backend.calls == 0


def train_partition(op, chunks, *, reverse=False):
    values, ids, ape, norm, cos, sin = fixture(8, requires_grad=True)
    upstream = torch.arange(8 * 512).reshape(8, 512).remainder(17).bfloat16() / 17
    with canonical_backward_session() as session:
        calls = []
        for start, end in chunks:
            input_start = max(0, start - 1)
            bound = op.bind(
                values[input_start * 4 : end * 4].contiguous(),
                ids[input_start * 4 : end * 4],
                version="v1",
                emit_group_ids=tuple((7, group) for group in range(start, end)),
            )
            output = op(bound, ape, norm, cos, sin, version="v1", parameter_id="layer.c4")
            calls.append((output, upstream[start:end]))
        order = list(range(len(calls)))
        if reverse:
            order.reverse()
        for index in order:
            calls[index][0].values.backward(calls[index][1])
        session.validate_complete()
    return values.grad, ape.grad, norm.grad


def test_canonical_backward_fixed_order_and_overlap_accumulation():
    op = operator()
    baseline = train_partition(op, [(0, 4), (4, 8)])
    for chunks, reverse in [
        ([(0, 2), (2, 4), (4, 8)], False),
        ([(4, 8), (0, 4)], True),
        ([(0, 1), (1, 3), (3, 8)], True),
    ]:
        actual = train_partition(operator(), chunks, reverse=reverse)
        for got, expected in zip(actual, baseline, strict=True):
            byte_equal(got, expected)


def test_input_only_backward_and_empty_groups():
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(1, requires_grad=True)
    with canonical_backward_session() as session:
        output = op(
            op.bind(values, ids, version="v1"), ape.detach(), norm.detach(), cos, sin, version="v1"
        )
        output.values.sum().backward()
        session.validate_complete()
    assert torch.count_nonzero(values.grad) > 0
    values, ids, ape, norm, cos, sin = fixture(0, requires_grad=True)
    output = op(op.bind(values, ids, version="v1"), ape, norm, cos, sin, version="v1")
    assert output.values.shape == (0, 512)


def test_session_identity_version_and_adapters():
    op = operator()
    values, ids, ape, norm, cos, sin = fixture(2, requires_grad=True)
    bound = op.bind(values, ids, version="v1")
    with pytest.raises(C4Error, match="canonical_backward_session"):
        op(bound, ape, norm, cos, sin, version="v1", parameter_id="layer")
    with canonical_backward_session():
        op(bound, ape, norm, cos, sin, version="v1", parameter_id="layer")
        with pytest.raises(C4Error, match="duplicate"):
            op(bound, ape, norm, cos, sin, version="v1", parameter_id="layer")
    from rl_engine.integrations.engines.rollout.vllm.kv_compressor_c4 import (
        bind_c4_compressor as rollout,
    )
    from rl_engine.integrations.engines.train.megatron.kv_compressor_c4 import (
        bind_c4_compressor as train,
    )

    for factory in (rollout, train, bind_c4_compressor):
        assert (
            factory(
                contract=op.contract,
                backend=ReferenceC4Backend(),
                expected_implementation=op.implementation_fingerprint,
            ).implementation_fingerprint
            == op.implementation_fingerprint
        )
