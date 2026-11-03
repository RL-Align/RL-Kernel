# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract, C4Error
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor, C4Groups
from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c4_helpers import ReferenceC4Backend, byte_equal
from tests.models.deepseek_v4.indexer_c4_helpers import (
    fixture,
    ReferenceIndexerC4Backend,
    rotate_oracle,
    pre_rotation_reference,
)

pytestmark = pytest.mark.unit


def operator(**changes):
    return IndexerC4Compressor(IndexerC4Contract(**changes), backend=ReferenceIndexerC4Backend())


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
@pytest.mark.parametrize("groups", [0, 1, 2, 8, 17])
def test_shared_binding_and_explicit_quantized_boundary(precision, groups):
    op = operator(precision=precision)
    x, ids, a, n, co, si = fixture(groups)
    b = op.bind(x, ids, version="model1-indexer-epoch1")
    assert isinstance(b, C4Groups)
    assert b.contract.family == "indexer_c4"
    out = op(b, a, n, co, si, version=b.version)
    assert out.values.shape == (groups, 128)
    assert out.packed.shape == (groups, 64 if precision == "mxfp4" else 128)
    assert out.scales.shape == (groups, 4 if precision == "mxfp4" else 1)
    assert out.layout == op.contract.layout and not out.packed.requires_grad
    if groups:
        expected = torch.stack(
            [
                pre_rotation_reference(ctx.float(), a, n, g, co, si, 1e-6)
                for ctx, (_, g) in zip(b.context_values, b.logical_keys, strict=True)
            ]
        ).bfloat16()
        byte_equal(out.values, rotate_oracle(expected))


@pytest.mark.parametrize(
    "change",
    [
        {"head_dim": 512},
        {"ratio": 128},
        {"overlap": 4},
        {"rope_dim": 32},
        {"precision": "bf16"},
        {"precision": "auto"},
        {"backend_id": C4Contract().backend_id},
        {"output_dtype": "fp32"},
        {"graph_capture": True},
        {"max_groups": False},
    ],
)
def test_reject_mixed_or_unknown_profiles(change):
    with pytest.raises(C4Error):
        replace(IndexerC4Contract(), **change)


def test_main_and_indexer_contract_state_and_parameter_isolation():
    from tests.models.deepseek_v4.c4_helpers import fixture as main_fixture

    main = C4Compressor(C4Contract(), backend=ReferenceC4Backend())
    index = operator()
    mx, mi, ma, mn, co, si = main_fixture(2)
    ix, ii, ia, inn, co, si = fixture(2)
    mb, ib = main.bind(mx, mi, version="v1"), index.bind(ix, ii, version="v1")
    with pytest.raises(C4Error, match="typed"):
        C4Compressor(IndexerC4Contract(), backend=ReferenceIndexerC4Backend())
    with pytest.raises(C4Error, match="typed"):
        IndexerC4Compressor(C4Contract(), backend=ReferenceC4Backend())
    for op, b, a, n in [
        (main, ib, ia, inn),
        (index, mb, ma, mn),
        (main, mb, ia, inn),
        (index, ib, ma, mn),
    ]:
        with pytest.raises(C4Error):
            op(b, a, n, co, si, version="v1")
    before = index(ib, ia, inn, co, si, version="v1").values.clone()
    mb.context_values.zero_()
    byte_equal(index(ib, ia, inn, co, si, version="v1").values, before)


def train_partition(op, chunks, *, order=None):
    x, ids, a, n, co, si = fixture(8, requires_grad=True)
    dy = torch.randn((8, 128), generator=torch.Generator().manual_seed(37)).bfloat16()
    with canonical_backward_session() as session:
        calls = []
        for first, last in chunks:
            lo = max(0, first - 1)
            b = op.bind(
                x[lo * 4 : last * 4],
                ids[lo * 4 : last * 4],
                version="v1",
                emit_group_ids=[(7, i) for i in range(first, last)],
            )
            out = op(b, a, n, co, si, version="v1", parameter_id="same-layer-prefix")
            assert out.packed is not None and not out.packed.requires_grad
            calls.append((out.values, dy[first:last]))
        for i in range(len(calls)) if order is None else order:
            calls[i][0].backward(calls[i][1])
        session.validate_complete()
    return x.grad, a.grad, n.grad


def test_backward_reuses_canonical_rows_across_chunks_and_orders():
    baseline = train_partition(operator(), [(0, 8)])
    for chunks, order in [([(0, 4), (4, 8)], [1, 0]), ([(4, 8), (0, 2), (2, 4)], [0, 2, 1])]:
        for a, b in zip(baseline, train_partition(operator(), chunks, order=order), strict=True):
            byte_equal(a, b)


def test_shared_session_with_identical_layer_prefix_keeps_branches_separate():
    from tests.models.deepseek_v4.c4_helpers import fixture as main_fixture

    main = C4Compressor(C4Contract(), backend=ReferenceC4Backend())
    index = operator()
    m = main_fixture(2, requires_grad=True)
    i = fixture(2, requires_grad=True)
    with canonical_backward_session() as session:
        outputs = []
        for op, (x, ids, a, n, co, si) in [(main, m), (index, i)]:
            outputs.append(
                op(
                    op.bind(x, ids, version="v1"), a, n, co, si, version="v1", parameter_id="layer0"
                ).values
            )
        outputs[1].sum().backward()
        assert m[0].grad is None and m[2].grad is None
        outputs[0].sum().backward()
        session.validate_complete()
        assert len(session.uses) == 4
    assert m[2].grad.shape == (4, 1024) and i[2].grad.shape == (4, 256)


def test_framework_adapters_are_the_same_operator_not_duplicate_arithmetic():
    from rl_engine.integrations.engines.rollout.vllm.indexer_compressor_c4 import (
        bind_indexer_c4_compressor as rollout,
    )
    from rl_engine.integrations.engines.train.megatron.indexer_compressor_c4 import (
        bind_indexer_c4_compressor as train,
    )

    assert train is rollout
    op = operator()
    assert train(
        contract=op.contract,
        backend=ReferenceIndexerC4Backend(),
        expected_implementation=op.implementation_fingerprint,
    )
    with pytest.raises(C4Error, match="fingerprint"):
        train(
            contract=op.contract, backend=ReferenceIndexerC4Backend(), expected_implementation="old"
        )


def test_backward_budget_covers_all_registered_chunks():
    op = operator(max_backward_groups=2)
    x, ids, a, n, co, si = fixture(3, requires_grad=True)
    with canonical_backward_session():
        b = op.bind(x, ids, version="v1", emit_group_ids=[(7, 0), (7, 1)])
        op(b, a, n, co, si, version="v1", parameter_id="budget")
        b2 = op.bind(x, ids, version="v1", emit_group_ids=[(7, 2)])
        with pytest.raises(C4Error, match="budget"):
            op(b2, a, n, co, si, version="v1", parameter_id="budget")
