# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import math

import pytest
import torch

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.p2.attention.oracle import (
    mqa_joint_attention_sink_bwd,
    mqa_joint_attention_sink_fwd,
)
from rl_engine.kernels.p2.contract import ATTENTION_SCALE, HEAD_DIM, N_Q_HEADS
from rl_engine.kernels.p2.fixtures.catalog import named_attn_catalog
from rl_engine.kernels.p2.fixtures.recorded import tensor_checksum


@pytest.fixture
def op():
    return MqaJointAttentionSinkOp(backend="oracle")


@pytest.mark.parametrize("name", list(named_attn_catalog().keys()))
def test_named_catalog_forward_finite(op, name):
    case = named_attn_catalog()[name]
    result = op.forward_fp32(
        case.q, case.k, case.v, case.sink, case.plan,
        compare=True, state_gate=case.state_gate, debug=True,
    )
    assert result.o.shape == (case.q.shape[0], N_Q_HEADS, HEAD_DIM)
    assert torch.isfinite(result.o).all()
    assert result.debug is not None
    assert result.debug["z"].shape == (case.q.shape[0], N_Q_HEADS)
    tensor_checksum(result.o)


def test_empty_candidates_output_zero(op):
    case = named_attn_catalog()["candidate_empty"]
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    assert torch.equal(saved.o, torch.zeros_like(saved.o))
    assert torch.allclose(saved.p_sink, torch.ones_like(saved.p_sink))
    bwd = mqa_joint_attention_sink_bwd(torch.ones_like(saved.o), saved)
    assert torch.equal(bwd.dq, torch.zeros_like(bwd.dq))
    assert bwd.dkv.numel() == 0
    assert torch.equal(bwd.dsink, torch.zeros_like(bwd.dsink))


def test_sink_dominates_mass_on_sink():
    case = named_attn_catalog()["sink_dominates"]
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    assert torch.all(saved.p_sink > 0.999)
    assert saved.o.abs().max() < 1e-3


def test_one_denominator_includes_sink():
    case = named_attn_catalog()["c0_recent_1"]
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    # p + p_sink == 1 for every valid row
    mass = saved.p.sum(dim=-1) + saved.p_sink
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-6)


def test_invalid_candidates_have_zero_prob():
    case = named_attn_catalog()["candidate_partial"]
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    assert torch.equal(
        saved.p[:, :, ~case.plan.valid], torch.zeros_like(saved.p[:, :, ~case.plan.valid])
    )


def test_debug_does_not_change_production_bytes(op):
    case = named_attn_catalog()["csa_selected_c4"]
    a = op.forward_fp32(case.q, case.k, case.v, case.sink, case.plan, debug=False)
    b = op.forward_fp32(case.q, case.k, case.v, case.sink, case.plan, debug=True)
    assert torch.equal(a.o, b.o)
    assert b.debug is not None
    assert a.debug is None


def test_scale_is_512_inv_sqrt():
    assert math.isclose(ATTENTION_SCALE, HEAD_DIM**-0.5, rel_tol=0, abs_tol=0)


def test_backward_zero_dout_zero_grads():
    case = named_attn_catalog()["c0_recent_only"]
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    bwd = mqa_joint_attention_sink_bwd(torch.zeros_like(saved.o), saved)
    assert torch.equal(bwd.dq, torch.zeros_like(bwd.dq))
    assert torch.equal(bwd.dkv, torch.zeros_like(bwd.dkv))
    assert torch.equal(bwd.dsink, torch.zeros_like(bwd.dsink))


def test_shared_sink_reduces_over_tokens():
    case = named_attn_catalog()["shared_sink"]
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    bwd = mqa_joint_attention_sink_bwd(torch.ones_like(saved.o), saved, sink_was_shared=True)
    assert bwd.dsink.shape == (N_Q_HEADS,)


def test_autograd_returns_dq_dk_dv():
    case = named_attn_catalog()["c0_recent_1"]
    q = case.q.clone().requires_grad_(True)
    k = case.k.clone().requires_grad_(True)
    v = case.v.clone().requires_grad_(True)
    sink = case.sink.clone().requires_grad_(True)
    op = MqaJointAttentionSinkOp(backend="oracle")
    out = op.apply_autograd(q, k, v, sink, case.plan, output_fp32=True)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and v.grad is not None


def _independent_softmax_attention(q, k, v, sink, valid):
    """Independent Jacobian: sink is an extra softmax logit with no V."""

    qf, kf, vf = q.float(), k.float(), v.float()
    logits = ATTENTION_SCALE * torch.einsum("thd,nd->thn", qf, kf)
    logits = logits.masked_fill(~valid.view(1, 1, -1), float("-inf"))
    if sink.dim() == 1:
        sink_col = sink.float().view(1, -1, 1).expand(q.shape[0], q.shape[1], 1)
    else:
        sink_col = sink.float().unsqueeze(-1)
    probs = torch.softmax(torch.cat([logits, sink_col], dim=-1), dim=-1)
    return torch.einsum("thn,nd->thd", probs[..., :-1], vf)


@pytest.mark.parametrize(
    "name", ["c0_recent_1", "csa_selected_c4", "candidate_partial", "shared_sink"]
)
def test_backward_matches_independent_softmax_autograd(name):
    case = named_attn_catalog()[name]
    q = case.q.clone().requires_grad_(True)
    k = case.k.clone().requires_grad_(True)
    v = case.v.clone().requires_grad_(True)
    sink = case.sink.clone().requires_grad_(True)
    out = _independent_softmax_attention(q, k, v, sink, case.plan.valid)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    saved = mqa_joint_attention_sink_fwd(
        q.detach(), k.detach(), v.detach(), sink.detach(), case.plan
    )
    bwd = mqa_joint_attention_sink_bwd(grad_out, saved, sink_was_shared=sink.dim() == 1)
    assert torch.allclose(bwd.dq, q.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(bwd.dk, k.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(bwd.dv, v.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(bwd.dsink, sink.grad, atol=1e-5, rtol=1e-4)
