# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reject non-finite CUDA values in eager execution and graph replay."""

from dataclasses import replace

import pytest
import torch

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import (
    MqaJointAttentionSinkOp,
    cuda_forward_into,
    make_cuda_fwd_workspace,
)
from rl_engine.kernels.p2.cuda_runtime import ensure_t06_cuda_kernel
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.finite import CheckedCUDAGraph
from rl_engine.kernels.p2.fixtures.catalog import make_attn_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


@pytest.fixture(scope="module", autouse=True)
def cuda_runtime():
    if torch.cuda.is_available():
        ensure_t06_cuda_kernel()


def _case(*, n_recent=2):
    return make_attn_case(
        "cuda-nonfinite",
        layer_type="C0",
        tokens=2,
        n_compressed=0,
        n_recent=n_recent,
        device="cuda",
        seed=19,
    )


@pytest.mark.parametrize("path", ["forward", "autograd", "forward_into"])
@pytest.mark.parametrize("field", ["q", "k", "v", "sink"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_cuda_forward_rejects_nonfinite(path, field, value):
    case = _case()
    getattr(case, field).reshape(-1)[0] = value
    op = MqaJointAttentionSinkOp(backend="cuda")
    with pytest.raises(P2FailClosedError) as exc:
        if path == "forward_into":
            workspace = make_cuda_fwd_workspace(case.q, case.k, case.sink, case.plan.valid)
            cuda_forward_into(case.q, case.k, case.v, workspace)
        elif path == "autograd":
            op.apply_autograd(
                case.q,
                case.k,
                case.v,
                case.sink,
                case.plan,
                output_fp32=True,
                compare=True,
                state_gate=case.state_gate,
            )
        else:
            op.forward_fp32(
                case.q,
                case.k,
                case.v,
                case.sink,
                case.plan,
                compare=True,
                state_gate=case.state_gate,
            )
    assert exc.value.status is P2Status.NON_FINITE


@pytest.mark.parametrize("n_recent", [0, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_cuda_backward_rejects_nonfinite_upstream_gradient(n_recent, value):
    case = _case(n_recent=n_recent)
    inputs = tuple(t.detach().requires_grad_() for t in (case.q, case.k, case.v, case.sink))
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        *inputs,
        case.plan,
        output_fp32=True,
    )
    grad = torch.ones_like(out)
    grad[0, 0, 0] = value
    with pytest.raises(P2FailClosedError) as exc:
        out.backward(grad)
    assert exc.value.status is P2Status.NON_FINITE
    assert all(t.grad is None for t in inputs)


def test_cuda_forward_rejects_overflow_from_finite_inputs():
    case = _case()
    case.q.fill_(1e20)
    case.k.fill_(1e20)
    with pytest.raises(P2FailClosedError) as exc:
        MqaJointAttentionSinkOp(backend="cuda").forward_fp32(
            case.q,
            case.k,
            case.v,
            case.sink,
            case.plan,
        )
    assert exc.value.status is P2Status.NON_FINITE


def test_cuda_backward_rejects_overflow_from_finite_upstream_gradient():
    case = _case()
    for tensor in (case.q, case.k, case.v, case.sink):
        tensor.zero_()
    inputs = tuple(t.detach().requires_grad_() for t in (case.q, case.k, case.v, case.sink))
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        *inputs,
        case.plan,
        output_fp32=True,
    )
    with pytest.raises(P2FailClosedError) as exc:
        out.backward(torch.full_like(out, 1e38))
    assert exc.value.status is P2Status.NON_FINITE
    assert all(t.grad is None for t in inputs)


@pytest.mark.parametrize("path", ["forward", "forward_into", "backward"])
def test_graph_checks_changed_inputs_and_recovers_after_nonfinite_replay(path):
    case = _case()
    case.plan.validate_for_kv(case.k, case.v)
    plan = replace(case.plan, kind=None)
    inputs = tuple(
        t.detach().clone().requires_grad_(path == "backward")
        for t in (case.q, case.k, case.v, case.sink)
    )
    q, k, v, sink = inputs
    grad = torch.ones_like(q) if path == "backward" else None
    workspace = make_cuda_fwd_workspace(q, k, sink, plan.valid) if path == "forward_into" else None
    op = MqaJointAttentionSinkOp(backend="cuda")

    def run():
        if path == "forward_into":
            cuda_forward_into(q, k, v, workspace)
            return (workspace.out,)
        if path == "backward":
            out = op.apply_autograd(*inputs, plan, output_fp32=True)
            return torch.autograd.grad(out, inputs, grad_outputs=grad)
        return (op.forward_fp32(*inputs, plan).o,)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = CheckedCUDAGraph()
    with graph.capture(stream=stream):
        outputs = run()
    graph.replay()
    expected = tuple(t.detach().clone() for t in outputs)
    bad_input = grad if path == "backward" else q
    original = bad_input.detach().clone()
    with torch.no_grad():
        bad_input.reshape(-1)[0] = float("nan")
    with pytest.raises(P2FailClosedError) as exc:
        graph.replay()
    assert exc.value.status is P2Status.NON_FINITE
    with torch.no_grad():
        bad_input.copy_(original)
    graph.replay()
    for actual, ref in zip(outputs, expected, strict=True):
        assert torch.equal(actual, ref)
    graph.reset()


def test_unchecked_graph_capture_fails_closed():
    case = _case()
    case.plan.validate_for_kv(case.k, case.v)
    plan = replace(case.plan, kind=None)
    op = MqaJointAttentionSinkOp(backend="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.forward_fp32(case.q, case.k, case.v, case.sink, plan)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(P2FailClosedError) as exc:
        with torch.cuda.graph(graph, stream=stream):
            op.forward_fp32(case.q, case.k, case.v, case.sink, plan)
    assert exc.value.status is P2Status.UNSUPPORTED_CAPABILITY
    assert torch.isfinite(op.forward_fp32(case.q, case.k, case.v, case.sink, plan).o).all()
