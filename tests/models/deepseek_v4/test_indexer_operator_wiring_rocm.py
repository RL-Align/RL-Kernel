# SPDX-License-Identifier: Apache-2.0
"""Items 10/11/14/15 called from Indexer training and o-projection, byte-equal to the old code."""

import pytest
import torch

from rl_engine.backends.rocm.attention.partial_rope import ROPE_LAUNCH
from rl_engine.models.deepseek_v4.indexer_q_preprocess import IndexerQPreprocess
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore
from rl_engine.models.deepseek_v4.indexer_training import canonical_indexer_auxiliary_loss
from rl_engine.models.deepseek_v4.partial_rope import PartialRoPE
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.rope_mxfp4_helpers import angles, selected
from tests.models.deepseek_v4.rope_mxfp4_helpers import byte_equal as same
from tests.models.deepseek_v4.test_indexer_auxiliary_rocm import make_case

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm GPU"
)


def tables():
    return angles(16, device="cuda")


def counting(monkeypatch, cls, name):
    calls = []
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        calls.append(name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, wrapper)
    return calls


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_auxiliary_score_uses_items_10_14(monkeypatch, precision):
    forward = counting(monkeypatch, IndexerScore, "forward")
    backward = counting(monkeypatch, IndexerScore, "backward")
    op, q, k, w, tq, tk, meta = make_case(precision, 5)
    for t in (q.master, k.master, w):
        t.requires_grad_()
    with canonical_backward_session() as session:
        result = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        assert forward == ["forward"]
        loss = canonical_indexer_auxiliary_loss([result])
        loss.backward()
        session.validate_complete()
    assert backward == ["backward"]
    assert torch.isfinite(loss).all()
    assert op.backend.provenance["score"] == op.backend.score.implementation_fingerprint


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_auxiliary_score_bytes_match_previous_inline_composition(precision):
    op, q, k, w, tq, tk, meta = make_case(precision, 5)
    backend, c = op.backend, op.contract
    saved = op.forward_record(q, k, w, tq, tk, metadata=meta, version="v1")
    indices = meta.selected_indices
    raw = saved.raw_icv
    relu = torch.relu(raw * (128**-0.5))
    weighted = torch.where(meta.valid_rows[:, None], saved.weights, 0.0) * (c.num_heads**-0.5)
    scores = backend.row_dot(
        relu.transpose(1, 2).contiguous(), weighted[:, None, :].expand(-1, indices.shape[1], -1)
    )
    same(saved.relu, relu)
    same(saved.score.scores, scores)
    same(saved.predicted, backend.masked_softmax(scores, (indices >= 0) & meta.valid_rows[:, None]))

    ds = torch.randn_like(scores)
    ds = torch.where((indices >= 0) & meta.valid_rows[:, None], ds, 0.0)
    ddot, dw = backend.score.backward(saved.score, ds)
    old_dw = backend.row_dot(relu, ds[:, None, :].expand_as(relu)) * (c.num_heads**-0.5)
    old_ddot = ds[:, None, :] * weighted[:, :, None]
    old_ddot = torch.where(raw > 0, old_ddot * (128**-0.5), 0.0)
    same(dw, old_dw)
    same(ddot, old_ddot)


@pytest.mark.parametrize("rows", [0, 1, 3, 9])
def test_mxfp4_query_preprocess_matches_previous_inline_composition(monkeypatch, rows):
    calls = counting(monkeypatch, IndexerQPreprocess, "__call__")
    op, *_ = make_case("mxfp4", 1)
    backend = op.backend
    torch.manual_seed(1115 + rows)
    q = torch.randn(rows, op.contract.num_heads, 128, device="cuda").bfloat16()
    positions = torch.tensor([0, 3, -1, 7, 15, 2, 9, 11, 1], device="cuda")[:rows]
    cos_cache, sin_cache = tables()
    q.requires_grad_()
    values = op.transform_queries(q, positions, cos_cache, sin_cache)
    assert calls == ["__call__"]

    cos, sin = selected(cos_cache, positions), selected(sin_cache, positions)
    masked = torch.where((positions >= 0)[:, None, None], q.detach(), 0)
    rope = backend.query_preprocess.backend
    if rows:
        rotated = rope._forward(masked[:, None], cos, sin, *ROPE_LAUNCH.values()).squeeze(1)
        expected = backend.codec._rotate(rotated.reshape(-1, 128)).reshape_as(q)
    else:
        expected = torch.empty_like(q.detach())
    same(values.detach(), expected)

    result = op.preprocess_queries(q.detach(), positions, cos_cache, sin_cache)
    data, scales = backend.codec.quantize(expected.reshape(-1, 128), backend.icv_contract)
    same(result.packed.reshape(-1, 64), data.reshape(-1, 64))
    same(result.scales.reshape(-1, 4), scales.reshape(-1, 4))

    grad = torch.randn_like(values)
    values.backward(grad)
    if rows:
        unrotated = backend.codec._rotate(grad.reshape(-1, 128)).reshape_as(grad)
        old = rope._backward(unrotated[:, None], cos, sin, *ROPE_LAUNCH.values()).squeeze(1)
        same(q.grad, torch.where((positions >= 0)[:, None, None], old, 0))


def test_fp8_query_uses_item_11_rope(monkeypatch):
    calls = counting(monkeypatch, PartialRoPE, "__call__")
    op, *_ = make_case("fp8", 1)
    assert not hasattr(op.backend, "query_preprocess")
    torch.manual_seed(1116)
    q = torch.randn(4, op.contract.num_heads, 128, device="cuda").bfloat16()
    positions = torch.tensor([0, 5, -1, 12], device="cuda")
    cos_cache, sin_cache = tables()
    values = op.transform_queries(q, positions, cos_cache, sin_cache)
    assert calls == ["__call__"]
    rope = op.backend.query_rope
    expected = op.backend.codec._rotate(
        rope(q, positions, cos_cache, sin_cache).reshape(-1, 128)
    ).reshape_as(q)
    same(values, expected)
    with pytest.raises(Exception, match="MXFP4 only"):
        op.preprocess_queries(q, positions, cos_cache, sin_cache)


def test_mxfp4_rejects_non_finite_query():
    op, *_ = make_case("mxfp4", 1)
    q = torch.zeros(2, op.contract.num_heads, 128, device="cuda").bfloat16()
    q[1, 0, 0] = float("nan")
    with pytest.raises(Exception, match="non-finite"):
        op.transform_queries(q, torch.tensor([0, 1], device="cuda"), *tables())


def test_o_projection_inverse_rope_uses_item_11_backend():
    from rl_engine.backends.rocm.attention.partial_rope import AiterPartialRoPEBackend
    from rl_engine.backends.rocm.gemm.o_projection import AiterOProjectionBackend

    backend = AiterOProjectionBackend()
    assert isinstance(backend.rope, AiterPartialRoPEBackend)
    assert backend.provenance["rope"] == backend.rope.provenance
    torch.manual_seed(1117)
    x = torch.randn(6, 4, 512, device="cuda").bfloat16()
    angles = torch.rand(6, 1, 1, 32, device="cuda") * 6
    cos, sin = angles.cos(), angles.sin()
    old_inverse = backend.rope._backward(x[:, None], cos, sin, 1, True, True).squeeze(1)
    old_adjoint = backend.rope._forward(x[:, None], cos, sin, 1, True, True).squeeze(1)
    same(backend.inverse_rope(x, cos, sin), old_inverse)
    same(backend.inverse_rope_gradient(x, cos, sin), old_adjoint)
    empty = x[:0]
    assert backend.inverse_rope(empty, cos[:0], sin[:0]).shape == empty.shape
