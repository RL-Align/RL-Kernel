# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import (
    MqaJointAttentionSinkOp,
    cuda_kernel_available,
)
from rl_engine.kernels.dsv4.attention.oracle import (
    mqa_joint_attention_sink_bwd,
    mqa_joint_attention_sink_fwd,
)
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.contract import (
    ATTENTION_SCALE,
    CUDA_VS_ORACLE_BWD_ATOL,
    CUDA_VS_ORACLE_FWD_ATOL,
)
from rl_engine.kernels.dsv4.attention.fixtures.catalog import named_attn_catalog

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")

_WORKSPACE_FIELDS = ("out", "scores", "p_sink", "m", "z")
_BACKWARD_FIELDS = ("dO", "valid", "P", "p_sink")


def _native_forward_case(n_candidates, output_fp32=True):
    q = torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(n_candidates, 512, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    sink = torch.zeros(2, 64, device=q.device)
    valid = torch.ones(n_candidates, dtype=torch.bool, device=q.device)
    shapes = {"out": q.shape, "scores": (2, 64, n_candidates)}
    buffers = {
        field: torch.full(
            shapes.get(field, (2, 64)),
            19,
            dtype=q.dtype if field == "out" and not output_fp32 else torch.float32,
            device=q.device,
        )
        for field in _WORKSPACE_FIELDS
    }
    return (q, k, v, sink, valid, float(ATTENTION_SCALE), output_fp32), buffers


def _native_forward_into(native, inputs, buffers):
    native.mqa_joint_attention_sink_forward_into(
        *inputs, *(buffers[field] for field in _WORKSPACE_FIELDS)
    )


def _native_backward_saved(inputs, buffers):
    return {
        "dO": torch.zeros_like(inputs[0]),
        "valid": inputs[4],
        "P": buffers["scores"],
        "p_sink": buffers["p_sink"],
    }


def _native_backward(native, inputs, saved):
    return native.mqa_joint_attention_sink_backward(
        saved["dO"], *inputs[:4], saved["valid"], saved["P"], saved["p_sink"], inputs[5], False
    )


@pytest.fixture(scope="module")
def validated_native():
    from rl_engine.kernels.ops.base import _C

    assert getattr(_C, "mqa_joint_attention_sink_workspace_validation_version", 0) == 3
    # Require native dtype validation before testing malformed workspaces.
    inputs, buffers = _native_forward_case(0)
    buffers["scores"] = buffers["scores"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="scores dtype mismatch"):
        _native_forward_into(_C, inputs, buffers)
    saved = _native_backward_saved(inputs, buffers)
    saved["P"] = saved["P"].float()
    saved["p_sink"] = saved["p_sink"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="p_sink dtype mismatch"):
        _native_backward(_C, inputs, saved)
    torch.cuda.synchronize()
    return _C


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("entry", ["forward", "forward_into", "backward"])
def test_native_rejects_empty_query_before_launch(validated_native, n_candidates, entry):
    inputs, buffers = _native_forward_case(n_candidates)
    inputs = (inputs[0][:0], *inputs[1:3], inputs[3][:0], *inputs[4:])
    buffers = {field: tensor[:0] for field, tensor in buffers.items()}
    with pytest.raises(RuntimeError, match="non-empty query batch required"):
        if entry == "forward":
            validated_native.mqa_joint_attention_sink_forward(*inputs)
        elif entry == "forward_into":
            _native_forward_into(validated_native, inputs, buffers)
        else:
            _native_backward(validated_native, inputs, _native_backward_saved(inputs, buffers))
    torch.cuda.synchronize()
    normal_inputs, _ = _native_forward_case(n_candidates)
    output = validated_native.mqa_joint_attention_sink_forward(*normal_inputs)[0]
    assert torch.isfinite(output).all()


def _assert_workspace_rejected(native, inputs, buffers, field, replacement, message):
    replacement.fill_(19)
    invalid = dict(buffers, **{field: replacement})
    with pytest.raises(RuntimeError, match=message):
        _native_forward_into(native, inputs, invalid)
    torch.cuda.synchronize()
    # Validation must complete before any output buffer is written.
    for buffer in (*buffers.values(), replacement):
        assert torch.equal(buffer, torch.full_like(buffer, 19))


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("field", _WORKSPACE_FIELDS)
@pytest.mark.parametrize("device", ["cpu", "peer"])
def test_native_forward_into_rejects_workspace_device(
    validated_native, n_candidates, field, device
):
    inputs, buffers = _native_forward_case(n_candidates)
    if device == "peer":
        if torch.cuda.device_count() < 2:
            pytest.skip("requires two GPUs")
        device = f"cuda:{(inputs[0].device.index + 1) % torch.cuda.device_count()}"
    _assert_workspace_rejected(
        validated_native, inputs, buffers, field, buffers[field].to(device),
        f"{field} device mismatch",
    )


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("output_fp32", [False, True])
@pytest.mark.parametrize("field", _WORKSPACE_FIELDS)
def test_native_forward_into_rejects_workspace_dtype(
    validated_native, n_candidates, output_fp32, field
):
    inputs, buffers = _native_forward_case(n_candidates, output_fp32)
    dtype = torch.float32 if field == "out" and not output_fp32 else torch.bfloat16
    _assert_workspace_rejected(
        validated_native, inputs, buffers, field, buffers[field].to(dtype),
        f"{field} dtype mismatch",
    )


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("field", _WORKSPACE_FIELDS)
def test_native_forward_into_rejects_workspace_shape(validated_native, n_candidates, field):
    inputs, buffers = _native_forward_case(n_candidates)
    _assert_workspace_rejected(
        validated_native, inputs, buffers, field, buffers[field].reshape(-1),
        f"{field} must be contiguous",
    )


@pytest.mark.parametrize("field", _WORKSPACE_FIELDS)
def test_native_forward_into_rejects_workspace_strides(validated_native, field):
    inputs, buffers = _native_forward_case(2)
    shape = buffers[field].shape
    padded = torch.empty(
        *shape[:-1], shape[-1] * 2, dtype=buffers[field].dtype, device=inputs[0].device
    )
    replacement = padded[..., ::2]
    assert replacement.shape == shape and not replacement.is_contiguous()
    _assert_workspace_rejected(
        validated_native, inputs, buffers, field, replacement, f"{field} must be contiguous"
    )


@pytest.mark.parametrize("entry", ["forward", "forward_into", "backward"])
@pytest.mark.parametrize("field", ["k", "v"])
def test_native_attention_rejects_kv_device_mismatch(validated_native, entry, field):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    inputs, buffers = _native_forward_case(2)
    mismatched = list(inputs)
    index = 1 if field == "k" else 2
    peer = f"cuda:{(inputs[0].device.index + 1) % torch.cuda.device_count()}"
    mismatched[index] = inputs[index].to(peer)
    with pytest.raises(RuntimeError, match="Q/K/V device mismatch"):
        if entry == "forward_into":
            _native_forward_into(validated_native, mismatched, buffers)
        elif entry == "forward":
            validated_native.mqa_joint_attention_sink_forward(*mismatched)
        else:
            validated_native.mqa_joint_attention_sink_backward(
                torch.zeros_like(inputs[0]), *mismatched[:5],
                buffers["scores"], buffers["p_sink"], float(ATTENTION_SCALE), False,
            )
    torch.cuda.synchronize()


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("field", _BACKWARD_FIELDS)
@pytest.mark.parametrize("device", ["cpu", "peer"])
def test_native_backward_rejects_saved_device(validated_native, n_candidates, field, device):
    inputs, buffers = _native_forward_case(n_candidates)
    saved = _native_backward_saved(inputs, buffers)
    if device == "peer":
        if torch.cuda.device_count() < 2:
            pytest.skip("requires two GPUs")
        device = f"cuda:{(inputs[0].device.index + 1) % torch.cuda.device_count()}"
    saved[field] = saved[field].to(device)
    message = "dO must be on Q's CUDA device" if field == "dO" else f"{field} device mismatch"
    with pytest.raises(RuntimeError, match=message):
        _native_backward(validated_native, inputs, saved)
    torch.cuda.synchronize()


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("field", _BACKWARD_FIELDS)
def test_native_backward_rejects_saved_shape(validated_native, n_candidates, field):
    inputs, buffers = _native_forward_case(n_candidates)
    saved = _native_backward_saved(inputs, buffers)
    saved[field] = saved[field].unsqueeze(0)
    with pytest.raises(RuntimeError, match=f"{field} shape mismatch"):
        _native_backward(validated_native, inputs, saved)
    torch.cuda.synchronize()


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("field", ["valid", "P", "p_sink"])
def test_native_backward_rejects_saved_dtype(validated_native, n_candidates, field):
    inputs, buffers = _native_forward_case(n_candidates)
    saved = _native_backward_saved(inputs, buffers)
    saved[field] = saved[field].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match=f"{field} dtype mismatch"):
        _native_backward(validated_native, inputs, saved)
    torch.cuda.synchronize()


def test_native_backward_accepts_strided_saved_tensors(validated_native):
    inputs, _ = _native_forward_case(2)
    _, p, p_sink, _, _ = validated_native.mqa_joint_attention_sink_forward(*inputs)
    saved = {"dO": torch.randn_like(inputs[0]), "valid": inputs[4], "P": p, "p_sink": p_sink}
    expected = _native_backward(validated_native, inputs, saved)
    strided = {}
    for field, tensor in saved.items():
        padded = torch.empty(
            *tensor.shape[:-1], tensor.shape[-1] * 2, dtype=tensor.dtype, device=tensor.device
        )
        strided[field] = padded[..., ::2]
        strided[field].copy_(tensor)
        assert not strided[field].is_contiguous()
    actual = _native_backward(validated_native, inputs, strided)
    for gradient, reference in zip(actual, expected, strict=True):
        assert torch.equal(gradient, reference)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_native_masked_nonfinite_v_matches_zeroed_row(validated_native, value, output_fp32):
    inputs, buffers = _native_forward_case(2, output_fp32)
    inputs[4][1] = False
    inputs[2][1].zero_()
    expected = validated_native.mqa_joint_attention_sink_forward(*inputs)
    saved = {
        "dO": torch.randn_like(inputs[0]), "valid": inputs[4],
        "P": expected[1], "p_sink": expected[2],
    }
    expected_gradients = _native_backward(validated_native, inputs, saved)
    inputs[2][1].fill_(value)
    actual = validated_native.mqa_joint_attention_sink_forward(*inputs)
    _native_forward_into(validated_native, inputs, buffers)
    for field, result, reference in zip(_WORKSPACE_FIELDS, actual, expected, strict=True):
        assert torch.isfinite(result).all()
        assert torch.equal(result, reference)
        assert torch.equal(buffers[field], reference)
    saved["P"], saved["p_sink"] = actual[1], actual[2]
    for gradient, reference in zip(
        _native_backward(validated_native, inputs, saved), expected_gradients, strict=True
    ):
        assert torch.isfinite(gradient).all()
        assert torch.equal(gradient, reference)


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_native_forward_into_static_graph_matches_forward(
    validated_native, n_candidates, output_fp32
):
    inputs, buffers = _native_forward_case(n_candidates, output_fp32)
    expected = validated_native.mqa_joint_attention_sink_forward(*inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _native_forward_into(validated_native, inputs, buffers)
    torch.cuda.current_stream().wait_stream(stream)
    for field, reference in zip(_WORKSPACE_FIELDS, expected, strict=True):
        assert torch.equal(buffers[field], reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _native_forward_into(validated_native, inputs, buffers)
    graph.replay()
    for field, reference in zip(_WORKSPACE_FIELDS, expected, strict=True):
        assert torch.equal(buffers[field], reference)
    inputs[0].add_(0.5)
    expected = validated_native.mqa_joint_attention_sink_forward(*inputs)
    graph.replay()
    for field, reference in zip(_WORKSPACE_FIELDS, expected, strict=True):
        assert torch.equal(buffers[field], reference)
    graph.reset()


def test_runtime_cuda_extension_exposes_t06_symbols(validated_native):
    assert hasattr(validated_native, "mqa_joint_attention_sink_forward")
    assert hasattr(validated_native, "mqa_joint_attention_sink_backward")
    assert cuda_kernel_available()


@pytest.mark.parametrize("n_candidates", [0, 2])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_native_shared_sink_matches_expanded_sink(
    validated_native, n_candidates, output_fp32
):
    inputs, _ = _native_forward_case(n_candidates, output_fp32)
    sink = torch.linspace(-0.5, 0.5, 64, device=inputs[0].device)
    expanded = sink.expand(inputs[0].size(0), -1).contiguous()
    shared_inputs = (*inputs[:3], sink, *inputs[4:])
    expanded_inputs = (*inputs[:3], expanded, *inputs[4:])
    shared = validated_native.mqa_joint_attention_sink_forward(*shared_inputs)
    per_token = validated_native.mqa_joint_attention_sink_forward(*expanded_inputs)
    for actual, expected in zip(shared, per_token, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(shared[0])
    shared_grad = validated_native.mqa_joint_attention_sink_backward(
        grad, *shared_inputs[:5], shared[1], shared[2], inputs[5], True
    )
    per_token_grad = validated_native.mqa_joint_attention_sink_backward(
        grad, *expanded_inputs[:5], per_token[1], per_token[2], inputs[5], False
    )
    for actual, expected in zip(shared_grad[:3], per_token_grad[:3], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(shared_grad[3], per_token_grad[3].sum(0), rtol=0, atol=0)


def _to_cuda(case):
    plan = CandidatePlan(
        layer_type=case.plan.layer_type,
        n_compressed=case.plan.n_compressed,
        n_recent=case.plan.n_recent,
        valid=case.plan.valid.cuda(),
        kind=None if case.plan.kind is None else case.plan.kind.cuda(),
    )
    return case.q.cuda(), case.k.cuda(), case.v.cuda(), case.sink.cuda(), plan


@pytest.mark.parametrize(
    "name", ["candidate_empty", "c0_recent_1", "csa_selected_c4", "candidate_partial"]
)
def test_cuda_matches_oracle_fp32(name):
    cpu = named_attn_catalog(device="cpu")[name]
    oracle = mqa_joint_attention_sink_fwd(cpu.q, cpu.k, cpu.v, cpu.sink, cpu.plan)
    q, k, v, sink, plan = _to_cuda(cpu)
    out = MqaJointAttentionSinkOp(backend="cuda").forward_fp32(
        q, k, v, sink, plan, compare=True, state_gate=cpu.state_gate
    )
    # CPU torch.exp vs CUDA expf is not bitwise; budget is in contract.py.
    assert torch.allclose(out.o.cpu(), oracle.o, atol=CUDA_VS_ORACLE_FWD_ATOL, rtol=0)
    assert out.provenance.backend == "cuda"
    assert "pragma_unroll_8" in out.provenance.unroll
    assert out.provenance.fallback is False


def test_cuda_forward_is_bitwise_deterministic():
    cpu = named_attn_catalog(device="cpu")["csa_selected_c4"]
    q, k, v, sink, plan = _to_cuda(cpu)
    op = MqaJointAttentionSinkOp(backend="cuda")
    a = op.forward_fp32(q, k, v, sink, plan, compare=True, state_gate=cpu.state_gate)
    b = op.forward_fp32(q, k, v, sink, plan, compare=True, state_gate=cpu.state_gate)
    assert torch.equal(a.o, b.o)


@pytest.mark.parametrize("name", ["c0_recent_1", "candidate_empty", "csa_selected_c4"])
def test_cuda_backward_matches_oracle(name):
    cpu = named_attn_catalog(device="cpu")[name]
    saved = mqa_joint_attention_sink_fwd(cpu.q, cpu.k, cpu.v, cpu.sink, cpu.plan)
    d_o = torch.randn_like(saved.o)
    ref = mqa_joint_attention_sink_bwd(d_o, saved, sink_was_shared=cpu.sink.dim() == 1)
    q, k, v, sink, plan = _to_cuda(cpu)
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    sink = sink.clone().requires_grad_(True)
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        q, k, v, sink, plan, output_fp32=True, compare=True, state_gate=cpu.state_gate
    )
    out.backward(d_o.cuda())
    assert torch.allclose(q.grad.cpu(), ref.dq, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
    if k.numel():
        assert torch.allclose(k.grad.cpu(), ref.dk, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
        assert torch.allclose(v.grad.cpu(), ref.dv, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
    assert torch.allclose(sink.grad.cpu(), ref.dsink, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)


def test_cuda_shared_1d_sink_backward():
    cpu = named_attn_catalog(device="cpu")["shared_sink"]
    assert cpu.sink.dim() == 1
    saved = mqa_joint_attention_sink_fwd(cpu.q, cpu.k, cpu.v, cpu.sink, cpu.plan)
    d_o = torch.randn_like(saved.o)
    ref = mqa_joint_attention_sink_bwd(d_o, saved, sink_was_shared=True)
    q, k, v, sink, plan = _to_cuda(cpu)
    assert sink.dim() == 1
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    sink = sink.clone().requires_grad_(True)
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        q, k, v, sink, plan, output_fp32=True, compare=True, state_gate=cpu.state_gate
    )
    out.backward(d_o.cuda())
    assert sink.grad.shape == (64,)
    assert torch.allclose(sink.grad.cpu(), ref.dsink, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
    assert torch.allclose(q.grad.cpu(), ref.dq, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)


def test_cuda_bf16_q_fp32_output_backward():
    cpu = named_attn_catalog(device="cpu")["c0_recent_1"]
    q, k, v, sink, plan = _to_cuda(cpu)
    q = q.to(torch.bfloat16).clone().requires_grad_(True)
    k = k.to(torch.bfloat16).clone().requires_grad_(True)
    v = v.to(torch.bfloat16).clone().requires_grad_(True)
    sink = sink.clone().requires_grad_(True)
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        q, k, v, sink, plan, output_fp32=True, compare=True, state_gate=cpu.state_gate
    )
    assert out.dtype == torch.float32
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert q.grad.dtype == torch.bfloat16


def test_eager_cuda_graph_byte_equal_or_fail_closed():
    """Capture in a child process so a failed capture cannot poison later tests."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    script = r"""
import torch
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.cuda_runtime import ensure_t06_cuda_kernel
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.fixtures.catalog import named_attn_catalog
from rl_engine.kernels.dsv4.attention.four_mode import eager_vs_cuda_graph_attention

ensure_t06_cuda_kernel()
cpu = named_attn_catalog(device="cpu")["c0_recent_1"]
q = cpu.q.cuda(); k = cpu.k.cuda(); v = cpu.v.cuda(); sink = cpu.sink.cuda()
plan = CandidatePlan(
    layer_type=cpu.plan.layer_type,
    n_compressed=cpu.plan.n_compressed,
    n_recent=cpu.plan.n_recent,
    valid=cpu.plan.valid.cuda(),
)
eager, graph = eager_vs_cuda_graph_attention(q, k, v, sink, plan, cpu.state_gate)
assert torch.equal(eager, graph)
print("GRAPH_STATUS=BYTE_EQUAL")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "GRAPH_STATUS=BYTE_EQUAL" in proc.stdout
