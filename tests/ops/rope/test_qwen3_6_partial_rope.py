# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Partial RoPE contracts on CPU and, when available, real CUDA/ROCm devices.

The same cases exercise the Native and Triton implementations. GPU-only cases
skip on CPU rather than substituting a fake Triton implementation.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.rope.qwen3_6_partial_rope import NativeQwen3_6PartialRoPEOp
from rl_engine.validation.operators.op_checks import CandidateSpec, OperatorCase, run_operator_suite
from rl_engine.validation.operators.operator_inputs import make_operator_inputs


@pytest.fixture(params=("native_cpu", "native_gpu", "triton_gpu"))
def implementation(request):
    """Return an actual implementation and device; never time/claim a CPU fallback."""
    if request.param == "native_cpu":
        return NativeQwen3_6PartialRoPEOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm GPU")
    if request.param == "native_gpu":
        return NativeQwen3_6PartialRoPEOp(), "cuda"
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.rope.qwen3_6_partial_rope import (
        TritonQwen3_6PartialRoPEOp,
    )

    return TritonQwen3_6PartialRoPEOp(), "cuda"


def _inputs(dtype, device, *, tokens=7, heads=2):
    args = argparse.Namespace(
        batch=1, seq=tokens, n_heads=heads, input_mode="random", seed=642, position_offset=11
    )
    return make_operator_inputs("qwen3_6_partial_rope", args, dtype, torch.device(device))


def _evaluate(op, inputs, grad_y):
    leaf = inputs["x"].detach().requires_grad_(True)
    y = op(**{**inputs, "x": leaf})
    (dx,) = torch.autograd.grad(y, leaf, grad_outputs=grad_y)
    return y.detach(), dx


def _assert_bits(actual, expected):
    integer_dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(integer_dtype), expected.contiguous().view(integer_dtype)
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("heads", (2, 16))
@pytest.mark.parametrize("tokens", (1, 257))
def test_forward_backward_accuracy(implementation, dtype, heads, tokens):
    """Q/K model head counts meet the unchanged forward and gradient thresholds."""
    op, device = implementation
    inputs = _inputs(dtype, device, heads=heads, tokens=tokens)
    report = run_operator_suite(
        "qwen3_6_partial_rope",
        candidates=[
            CandidateSpec(
                name=type(op).__name__,
                fn=op,
                backend="pytorch" if isinstance(op, NativeQwen3_6PartialRoPEOp) else "triton",
            )
        ],
        cases=[
            OperatorCase(
                name="text-qk",
                op_class="elementwise",
                dtype=dtype,
                inputs=inputs,
                gold_fn=NativeQwen3_6PartialRoPEOp().forward_fp32,
                grad_input_names=("x",),
            )
        ],
        check_grad=True,
    )
    assert report.passed, report.to_dict()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("chunk_size", (1, 3))
def test_prefill_decode_and_chunked_bits(implementation, dtype, chunk_size):
    """Retain absolute positions across calls, including a packed sequence reset."""
    op, device = implementation
    inputs = _inputs(dtype, device)
    inputs["positions"] = torch.tensor([11, 12, 13, 0, 1, 2, 3], device=device, dtype=torch.int32)
    grad_y = torch.linspace(-2, 2, inputs["x"].numel(), device=device).reshape_as(inputs["x"])
    grad_y = grad_y.to(dtype)
    whole_y, whole_dx = _evaluate(op, inputs, grad_y)
    for start in range(0, len(inputs["x"]), chunk_size):
        selection = slice(start, start + chunk_size)
        chunk = {**inputs, "x": inputs["x"][selection], "positions": inputs["positions"][selection]}
        y, dx = _evaluate(op, chunk, grad_y[selection])
        _assert_bits(y, whole_y[selection])
        _assert_bits(dx, whole_dx[selection])
    # Extra unrelated tokens or a narrower head shard cannot affect existing rows.
    padded = {
        **inputs,
        "x": torch.cat((inputs["x"], inputs["x"][:2])),
        "positions": torch.cat((inputs["positions"], inputs["positions"][:2])),
    }
    _assert_bits(op(**padded)[:7], whole_y)
    shard = {**inputs, "x": inputs["x"][:, :1]}
    y, dx = _evaluate(op, shard, grad_y[:, :1])
    _assert_bits(y, whole_y[:, :1])
    _assert_bits(dx, whole_dx[:, :1])


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("layout", ("strided", "broadcast"))
def test_strides_and_tail_bits(implementation, dtype, layout):
    """Exercise every input stride, zero-stride dy, and unmodified tail bit patterns."""
    op, device = implementation
    inputs = _inputs(dtype, device)
    for name, value in inputs.items():
        storage = torch.empty(
            (*value.shape[:-1], value.shape[-1] * 2), device=device, dtype=value.dtype
        )
        storage[..., 1::2] = value
        inputs[name] = storage[..., 1::2]
    x = inputs["x"]
    x[..., 64] = -0.0
    x[..., 65] = float("nan")
    # A distinct NaN payload must survive a copy, even though torch.equal(NaN, NaN) is false.
    integer_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    x.view(integer_dtype)[..., 66] = 0x7FC1 if dtype == torch.bfloat16 else 0x7FC00001
    if layout == "broadcast":
        inputs["x"] = x = x[:1].expand(7, -1, -1)
    saved = {name: value.clone() for name, value in inputs.items()}
    if layout == "broadcast":
        grad_y = torch.full((1, 1, 256), -0.0, device=device, dtype=dtype).expand_as(x)
    else:
        grad_storage = torch.linspace(-2, 2, x.numel() * 2, device=device).to(dtype)
        grad_y = grad_storage.reshape(*x.shape[:-1], 512)[..., 1::2]
        grad_y[..., 64] = -0.0
        grad_y.view(integer_dtype)[..., 65] = 0x7FC1 if dtype == torch.bfloat16 else 0x7FC00001
    y, dx = _evaluate(op, inputs, grad_y)
    expected_y, expected_dx = _evaluate(NativeQwen3_6PartialRoPEOp(), inputs, grad_y)
    _assert_bits(y, expected_y)
    _assert_bits(dx, expected_dx)
    _assert_bits(y[..., 64:], x[..., 64:])
    _assert_bits(dx[..., 64:], grad_y[..., 64:])
    assert y.is_contiguous() and dx.is_contiguous()
    for name in ("x", "cos", "sin"):
        _assert_bits(inputs[name], saved[name])
    assert torch.equal(inputs["positions"], saved["positions"])


@pytest.mark.parametrize("tokens,heads", ((0, 2), (3, 0)))
def test_empty_input(implementation, tokens, heads):
    op, device = implementation
    x = torch.empty((tokens, heads, 256), device=device, requires_grad=True)
    table = torch.empty((0, 32), device=device)
    positions = torch.zeros(tokens, device=device, dtype=torch.int32)
    y = op(x, table, table, positions)
    (dx,) = torch.autograd.grad(y, x, torch.empty_like(y))
    assert y.shape == dx.shape == x.shape
    assert y.numel() == dx.numel() == 0


@pytest.mark.parametrize(
    "name,change,error,match",
    [
        ("x", lambda t: t[..., :128], ValueError, "x must have shape"),
        ("x", lambda t: t.half(), TypeError, "x must have dtype"),
        ("cos", lambda t: t[:, :16], ValueError, "cos must have shape"),
        ("sin", lambda t: t.bfloat16(), TypeError, "sin must have dtype"),
        ("sin", lambda t: t[:-1], ValueError, "share shape"),
        ("cos", lambda t: t.requires_grad_(), ValueError, "must not require gradients"),
        ("sin", lambda t: t.requires_grad_(), ValueError, "must not require gradients"),
        ("positions", lambda t: t[:, None], ValueError, "positions must have shape"),
        ("positions", lambda t: t.float(), TypeError, "positions must have dtype"),
        ("cos", lambda t: t.to("meta"), ValueError, "must be on x.device"),
        ("positions", lambda t: t.to("meta"), ValueError, "must be on x.device"),
    ],
)
def test_invalid_metadata(implementation, name, change, error, match):
    op, device = implementation
    inputs = _inputs(torch.float32, device)
    inputs[name] = change(inputs[name])
    with pytest.raises(error, match=match):
        op(**inputs)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_native_against_independent_rotation_matrix(dtype):
    """FP64 2x2 matrices check pair/sign/position selection and the transpose VJP."""
    inputs = _inputs(dtype, "cpu")
    inputs["positions"] = torch.tensor([17, 0, 11, 17, 1, 2, 13])
    x, cos, sin, positions = (inputs[k] for k in ("x", "cos", "sin", "positions"))
    c, s = cos[positions].double(), sin[positions].double()
    matrix = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2)
    pairs = x[..., :64].double().reshape(7, 2, 2, 32).transpose(-1, -2)
    expected = torch.einsum("tpij,thpj->thpi", matrix, pairs)
    expected = torch.cat((expected.transpose(-1, -2).flatten(-2), x[..., 64:].double()), -1)
    grad_y = torch.linspace(-1, 1, x.numel()).reshape_as(x).to(dtype)
    grad_pairs = grad_y[..., :64].double().reshape(7, 2, 2, 32).transpose(-1, -2)
    expected_dx = torch.einsum("tpji,thpj->thpi", matrix, grad_pairs)
    expected_dx = torch.cat(
        (expected_dx.transpose(-1, -2).flatten(-2), grad_y[..., 64:].double()), -1
    )
    y, dx = _evaluate(NativeQwen3_6PartialRoPEOp(), inputs, grad_y)
    for actual, gold, judgment in (
        (y, expected, "forward_accuracy"),
        (dx, expected_dx, "gradient_accuracy"),
    ):
        tol = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="elementwise", dtype=dtype
        )
        torch.testing.assert_close(actual.double(), gold, atol=tol.atol, rtol=tol.rtol)
    fp32 = NativeQwen3_6PartialRoPEOp().forward_fp32(**inputs)
    assert fp32.dtype == torch.float32
    _assert_bits(y, fp32.to(dtype))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_noncurrent_device_restored(dtype):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA/ROCm GPUs")
    from rl_engine.backends.shared.triton.rope.qwen3_6_partial_rope import (
        TritonQwen3_6PartialRoPEOp,
    )

    with torch.cuda.device(0):
        inputs = _inputs(dtype, "cuda:1")
        grad_y = torch.ones_like(inputs["x"])
        leaf = inputs["x"].detach().requires_grad_(True)
        y = TritonQwen3_6PartialRoPEOp()(leaf, inputs["cos"], inputs["sin"], inputs["positions"])
        assert torch.cuda.current_device() == 0
        (dx,) = torch.autograd.grad(y, leaf, grad_y)
        assert torch.cuda.current_device() == 0
        expected = _evaluate(NativeQwen3_6PartialRoPEOp(), inputs, grad_y)
        _assert_bits(y, expected[0])
        _assert_bits(dx, expected[1])
