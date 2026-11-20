from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.indexer_projection import IndexerProjectionError
from rl_engine.contracts.operators.o_projection import OProjectionContract, OProjectionError
from rl_engine.models.deepseek_v4.o_projection import (
    OProjection,
    OProjectionRows,
    merge_tp_partials,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from .o_projection_helpers import (
    ReferenceBackend,
    make_case,
    oracle_backward,
    oracle_forward,
    rotation,
    select_rows,
    same_bytes,
)


@pytest.mark.parametrize(
    "change",
    [
        {"num_heads": 0},
        {"num_heads": True},
        {"num_heads": 17},
        {"tp_size": 3},
        {"tp_rank": -1},
        {"tp_rank": 1},
        {"head_dim": 63},
        {"rope_dim": 65},
        {"rope_dim": 256},
        {"o_lora_rank": 0},
        {"hidden_size": 0},
        {"heads_per_group": 4},
        {"max_rows": False},
        {"graph_capture": True},
        {"backend_id": "other"},
        {"backward_id": "other"},
        {"a_precision": "fp16"},
        {"b_precision": "fp8"},
        {"a_precision": "block_fp8_ue8m0", "o_lora_rank": 64},
        {"b_precision": "block_fp8_ue8m0", "hidden_size": 64},
    ],
)
def test_contract_rejects(change):
    with pytest.raises(OProjectionError):
        replace(OProjectionContract(16, 128, 64, 128, 128), **change)


def test_contract_partition_and_fingerprints():
    c = OProjectionContract(32, 128, 64, 128, 128, tp_size=2)
    d = replace(c, tp_rank=1)
    assert c.head_ids == tuple(range(16)) and d.head_ids == tuple(range(16, 32))
    assert c.fingerprint != d.fingerprint and c.family_fingerprint == d.family_fingerprint


@pytest.mark.parametrize("m", [0, 1, 5, 33])
def test_explicit_reference_forward_and_backward(m):
    op, x, w, rows = make_case(m=m, train=True, padding=True)
    with canonical_backward_session() as session:
        result = op(x, w, rows=rows)
        inv, z, y = oracle_forward(op.contract, x, w, rows)
        assert same_bytes(inv, result.inverse_rope) and same_bytes(z, result.groups)
        assert same_bytes(y, result.partial)
        grad = torch.randn_like(y)
        expected = oracle_backward(op.contract, result, w, grad)
        result.partial.backward(grad)
        session.validate_complete()
    for actual, want in zip((x.grad, w.a.grad, w.b.grad), expected, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "head",
        "padding",
        "duplicate",
        "position",
        "dtype",
        "angles",
        "norm",
        "requires_grad",
        "capacity",
    ],
)
def test_metadata_rejected(fault):
    c = OProjectionContract(8, 128, 64, 128, 128, max_rows=2)
    keys = torch.tensor([[0, 0], [0, 1]])
    cos, sin = torch.ones(2, 32), torch.zeros(2, 32)
    heads = c.head_ids
    if fault == "head":
        heads = tuple(reversed(heads))
    if fault == "padding":
        keys[-1] = torch.tensor([-1, 1])
    if fault == "duplicate":
        keys[-1] = keys[0]
    if fault == "position":
        keys[-1, 1] = 2
    if fault == "dtype":
        keys = keys.int()
    if fault == "angles":
        cos[1, 0] = float("nan")
    if fault == "norm":
        cos[1, 0] = 2
    if fault == "requires_grad":
        cos.requires_grad_()
    if fault == "capacity":
        keys = torch.zeros(3, 2, dtype=torch.int64)
    with pytest.raises(IndexerProjectionError):
        OProjectionRows.bind(c, keys, cos, sin, head_ids=heads, version="v1")


@pytest.mark.parametrize(
    "fault",
    [
        "weight",
        "cos",
        "keys",
        "version",
        "geometry",
        "layer",
        "missing_session",
        "fp8_master",
        "implementation",
    ],
)
def test_lifetime_and_abi_rejected(fault):
    op, x, w, rows = make_case(train=fault == "missing_session")
    with pytest.raises(IndexerProjectionError):
        if fault == "weight":
            w.a.add_(1)
        if fault == "cos":
            rows.cos.add_(1)
        if fault == "keys":
            rows.logical.keys[0, 0] = 999
        if fault == "version":
            rows = replace(rows, version="v0")
        if fault == "geometry":
            x = x[:, :8].contiguous()
        if fault == "layer":
            OProjection(op.contract, layer_id="", backend=ReferenceBackend())
        if fault == "fp8_master":
            w.a._fp8_source_scales = torch.ones(1)
            op.prepare_weights(w.a, w.b, version="v2")
        if fault == "implementation":
            OProjection(
                op.contract, layer_id="L", backend=ReferenceBackend(), expected_implementation="bad"
            )
        op(x, w, rows=rows)


def test_owned_metadata_does_not_alias_source():
    c = OProjectionContract(8, 128, 64, 128, 128)
    keys = torch.tensor([[0, 0]])
    cos, sin = torch.ones(2, 32), torch.zeros(2, 32)
    rows = OProjectionRows.bind(c, keys, cos, sin, head_ids=c.head_ids, version="v1")
    keys.fill_(-1)
    cos.zero_()
    sin.fill_(1)
    assert rows.logical.keys.tolist() == [[0, 0]] and rows.cos.eq(1).all()


def test_duplicate_microbatch_and_stale_parameter():
    op, x, w, rows = make_case(train=True)
    with canonical_backward_session():
        op(x, w, rows=rows)
        with pytest.raises(IndexerProjectionError, match="repeated"):
            op(x, w, rows=rows)
    with canonical_backward_session():
        op(x[:2].contiguous(), w, rows=select_rows(op.contract, rows, slice(0, 2)))
        other = op.prepare_weights(
            w.a.detach().clone().requires_grad_(),
            w.b.detach().clone().requires_grad_(),
            version="v1",
        )
        with pytest.raises(IndexerProjectionError, match="another weight"):
            op(x[2:].contiguous(), other, rows=select_rows(op.contract, rows, slice(2, None)))


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "version", "keys", "layer", "impl", "cos"]
)
def test_tp_validation(fault):
    results = []
    for rank in range(2):
        c = OProjectionContract(16, 128, 64, 128, 128, tp_size=2, tp_rank=rank)
        op, x, w, rows = make_case(contract=c)
        results.append(op(x, w, rows=rows))
    if fault == "missing":
        results = results[:1]
    if fault == "duplicate":
        results[1] = results[0]
    if fault == "version":
        results[1] = replace(results[1], rows=replace(results[1].rows, version="v0"))
    if fault == "keys":
        results[1].rows.logical.keys[0, 0] = 99
    if fault == "layer":
        results[1] = replace(results[1], layer_id="other")
    if fault == "impl":
        results[1] = replace(results[1], implementation_fingerprint="other")
    if fault == "cos":
        results[1].rows.cos.add_(1)
    with pytest.raises(IndexerProjectionError):
        merge_tp_partials(results)


def test_inverse_rotation_finite_difference_and_adjoint():
    # Real-valued rotation, before BF16 rounding: central finite difference.
    x = torch.randn(2, 8, 128, dtype=torch.float64)
    angle = torch.randn(2, 1, 1, 32)
    c, s = angle.cos(), angle.sin()
    dy = torch.randn_like(x)
    direction = torch.randn_like(x)
    eps = 1e-5
    fd = (
        (
            rotation(x + eps * direction, c, s, inverse=True)
            - rotation(x - eps * direction, c, s, inverse=True)
        )
        * dy
    ).sum() / (2 * eps)
    adjoint = (rotation(dy, c, s, inverse=False) * direction).sum()
    torch.testing.assert_close(fd, adjoint, rtol=1e-8, atol=1e-8)


def test_framework_hooks_share_exact_provider():
    from rl_engine.integrations.engines.train.megatron.o_projection import (
        bind_output_projection as train,
    )
    from rl_engine.integrations.engines.rollout.vllm.o_projection import (
        bind_output_projection as rollout,
    )

    op, x, w, rows = make_case()
    kwargs = dict(
        contract=op.contract,
        layer_id=op.layer_id,
        wo_a=w.a,
        wo_b=w.b,
        version="v1",
        backend=ReferenceBackend(),
    )
    a = train(**kwargs)
    b = rollout(**kwargs, expected_implementation=a.operator.implementation_fingerprint)
    assert same_bytes(a(x, rows=rows).partial, b(x, rows=rows).partial)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "geometry_a",
        "geometry_b",
        "precision",
        "version",
        "inference",
        "unexpected_pack",
        "shape_pack",
    ],
)
def test_weight_binding_validation(fault):
    op, x, w, rows = make_case()
    a, b, version, pack = w.a, w.b, "v1", None
    if fault == "missing":
        a = None
    if fault == "geometry_a":
        a = a[:, :-1].contiguous()
    if fault == "geometry_b":
        b = b[:, :-1].contiguous()
    if fault == "precision":
        a = a.float()
    if fault == "version":
        version = ""
    if fault == "inference":
        with torch.inference_mode():
            a = a.clone()
    if fault == "unexpected_pack":
        pack = (a, torch.ones(1))
    if fault == "shape_pack":
        pack = (a,)
    with pytest.raises(IndexerProjectionError):
        op.prepare_weights(a, b, version=version, a_checkpoint=pack)


def test_mutation_after_forward_is_rejected_in_backward():
    op, x, w, rows = make_case(train=True)
    with canonical_backward_session():
        out = op(x, w, rows=rows)
        rows.logical.keys[0, 0] = 999
        with pytest.raises(IndexerProjectionError, match="mutated"):
            out.partial.sum().backward()
