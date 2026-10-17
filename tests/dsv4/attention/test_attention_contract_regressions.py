# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regressions for the sequential attention arithmetic contract."""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.dsv4.attention.oracle import (
    mqa_joint_attention_sink_bwd,
    mqa_joint_attention_sink_fwd,
)
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.contract import ATTENTION_SCALE
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.fixtures.catalog import make_attn_case


def _case():
    return make_attn_case(
        "attention_contract",
        layer_type="C4",
        tokens=2,
        n_compressed=2,
        n_recent=2,
        invalid_prefix=1,
        invalid_recent=1,
        sink_mode="shared",
    )


def _forward(case, **kwargs):
    return mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan, **kwargs)


def test_oracle_forward_and_backward_do_not_use_matmul_or_softmax(monkeypatch):
    case = _case()

    def forbidden(*args, **kwargs):
        pytest.fail("sequential attention oracle called matmul or softmax")

    for owner, names in (
        (torch, ("matmul", "mm", "bmm", "softmax")),
        (torch.Tensor, ("matmul", "mm", "bmm", "__matmul__", "__rmatmul__", "softmax")),
        (F, ("softmax",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)

    saved = _forward(case)
    grads = mqa_joint_attention_sink_bwd(torch.ones_like(saved.o), saved, sink_was_shared=True)
    for tensor in (saved.o, saved.z, grads.dq, grads.dk, grads.dv, grads.dsink):
        assert torch.isfinite(tensor).all()
    assert grads.dsink.shape == case.sink.shape


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize(
    "bad_scale",
    [0.0, 2.0 * ATTENTION_SCALE, float("nan"), float("inf"), -float("inf")],
    ids=["zero", "double", "nan", "positive_inf", "negative_inf"],
)
def test_oracle_rejects_noncontract_scale(direction, bad_scale):
    case = _case()
    saved = _forward(case) if direction == "backward" else None
    with pytest.raises(DSv4FailClosedError) as exc:
        if direction == "forward":
            _forward(case, scale=bad_scale)
        else:
            mqa_joint_attention_sink_bwd(
                torch.ones_like(saved.o), saved, scale=bad_scale, sink_was_shared=True
            )
    assert exc.value.status is DSv4Status.ROUND_POINT_MISMATCH


@pytest.mark.parametrize("field", ["q", "k", "v", "sink"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_oracle_forward_rejects_nonfinite_arithmetic(field, nonfinite):
    case = _case()
    tensor = getattr(case, field)
    if field in {"k", "v"}:
        tensor[1, 0] = nonfinite
    else:
        tensor.reshape(-1)[0] = nonfinite
    with pytest.raises(DSv4FailClosedError) as exc:
        _forward(case)
    assert exc.value.status is DSv4Status.NON_FINITE


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_oracle_backward_rejects_nonfinite_gradient(nonfinite):
    saved = _forward(_case())
    grad_out = torch.ones_like(saved.o)
    grad_out[0, 0, 0] = nonfinite
    with pytest.raises(DSv4FailClosedError) as exc:
        mqa_joint_attention_sink_bwd(grad_out, saved, sink_was_shared=True)
    assert exc.value.status is DSv4Status.NON_FINITE


@pytest.mark.parametrize("field", ["k", "v"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_rows_match_removed_candidates(field, nonfinite):
    case = _case()
    valid = case.plan.valid
    getattr(case, field)[~valid] = nonfinite
    compact_plan = CandidatePlan(
        layer_type="C4", n_compressed=1, n_recent=1, valid=torch.ones(2, dtype=torch.bool)
    )
    expected = mqa_joint_attention_sink_fwd(
        case.q, case.k[valid], case.v[valid], case.sink, compact_plan
    )
    actual = _forward(case)
    assert torch.equal(actual.o, expected.o)
    assert torch.equal(actual.p[:, :, valid], expected.p)
    assert torch.equal(actual.p_sink, expected.p_sink)
    grad_out = torch.ones_like(actual.o)
    grads = mqa_joint_attention_sink_bwd(grad_out, actual, sink_was_shared=True)
    reference = mqa_joint_attention_sink_bwd(grad_out, expected, sink_was_shared=True)
    for result, target in (
        (grads.dq, reference.dq),
        (grads.dsink, reference.dsink),
        (grads.dk[valid], reference.dk),
        (grads.dv[valid], reference.dv),
    ):
        assert torch.equal(result, target)
    assert torch.count_nonzero(grads.dk[~valid]) == 0
    assert torch.count_nonzero(grads.dv[~valid]) == 0
