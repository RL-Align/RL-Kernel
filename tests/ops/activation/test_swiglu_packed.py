# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Common packed SwiGLU: both orders, shared arithmetic and layout invariance."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.contracts.operators.activation import SwiGLUPackedOrder
from rl_engine.reference.activation.swiglu import NativeSwiGLUOp

pytest.importorskip("triton")
from rl_engine.backends.shared.triton.activation.swiglu import TritonSwiGLUOp  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm required")
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _input(layout: str, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(713)
    shapes = {
        "vector": (28672,),
        "empty": (2, 0, 28672),
        "channel_stride": (2, 3, 57344),
        "row_stride": (4, 3, 28672),
        "transpose": (3, 2, 28672),
        "odd_width": (2, 3, 2050),
        "narrow": (2, 3, 14),
        "block_tail": (2, 3, 2046),
        "block_aligned": (2, 3, 2048),
        "many_rows": (256, 256, 2),
        "contiguous": (2, 7, 28672),
    }
    x = torch.randn(shapes[layout], device="cuda", dtype=dtype)
    if layout == "channel_stride":
        x = x[..., ::2]
    elif layout == "row_stride":
        x = x[::2]
    elif layout == "transpose":
        x = x.transpose(0, 1)
    return x.detach().requires_grad_(True)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize(
    "layout",
    [
        "vector",
        "empty",
        "channel_stride",
        "row_stride",
        "transpose",
        "odd_width",
        "narrow",
        "block_tail",
        "block_aligned",
        "many_rows",
        "contiguous",
    ],
)
@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_packed_matches_shared_bits_and_native_accuracy(dtype, method, layout, order):
    x = _input(layout, dtype)
    before = x.detach().clone()
    first, second = x.chunk(2, dim=-1)
    gate, up = (first, second) if order is SwiGLUPackedOrder.GATE_UP else (second, first)
    shared = getattr(TritonSwiGLUOp(), method)(input=gate, up=up)
    packed = getattr(TritonSwiGLUOp(), method)(input=x, packed_order=order)
    _assert_bits(packed, shared)
    upstream = torch.randn(
        (*packed.shape[:-1], packed.shape[-1] * 2), device="cuda", dtype=packed.dtype
    )[..., ::2]
    shared_grad = torch.autograd.grad(shared, x, upstream)[0]
    packed_grad = torch.autograd.grad(packed, x, upstream)[0]
    _assert_bits(packed_grad, shared_grad)
    gold_x = x.detach().float().requires_grad_(True)
    gold = NativeSwiGLUOp().forward_fp32(gold_x, packed_order=order)
    gold_grad = torch.autograd.grad(gold, gold_x, upstream.float())[0]
    # Bitwise agreement above is separate from independent FP32 accuracy.
    for judgment, actual, expected in (
        ("forward_accuracy", packed, gold),
        ("gradient_accuracy", packed_grad, gold_grad),
    ):
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="elementwise", dtype=dtype
        )
        torch.testing.assert_close(
            actual.float(), expected, rtol=tolerance.rtol, atol=tolerance.atol
        )
    _assert_bits(x.detach(), before)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_training_inference_batch_padding_and_repeated_backward(dtype, method, order):
    x = _input("contiguous", dtype)
    fn = partial(getattr(TritonSwiGLUOp(), method), packed_order=order)
    y = fn(x)
    with torch.no_grad():
        _assert_bits(fn(x), y)
        padded = torch.cat((x, torch.zeros_like(x)), dim=0)
        _assert_bits(fn(padded)[: x.shape[0]], y)
    with torch.inference_mode():
        _assert_bits(fn(x), y)
    upstream = torch.randn_like(y)
    full_grad = torch.autograd.grad(y, x, upstream, retain_graph=True)[0]
    _assert_bits(torch.autograd.grad(y, x, upstream)[0], full_grad)
    one = x[:1, :1].detach().requires_grad_(True)
    one_y = fn(one)
    _assert_bits(one_y, y[:1, :1])
    _assert_bits(torch.autograd.grad(one_y, one, upstream[:1, :1])[0], full_grad[:1, :1])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_saturation_and_zero_match_shared_bits(dtype, order):
    up = torch.linspace(-8, 8, 4097, device="cuda", dtype=dtype)
    gate = torch.linspace(-100, 100, 4097, device="cuda", dtype=dtype)
    operands = (gate, up) if order is SwiGLUPackedOrder.GATE_UP else (up, gate)
    x = torch.cat(operands).repeat(2, 1).requires_grad_(True)
    first, second = x.chunk(2, dim=-1)
    gate, up = (first, second) if order is SwiGLUPackedOrder.GATE_UP else (second, first)
    op = TritonSwiGLUOp()
    shared, packed = op(input=gate, up=up), op(input=x, packed_order=order)
    _assert_bits(packed, shared)
    _assert_bits(
        torch.autograd.grad(packed, x, torch.ones_like(packed))[0],
        torch.autograd.grad(shared, x, torch.ones_like(shared))[0],
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", ["forward", "backward", "forward_backward"])
@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_cuda_graph_replay_matches_eager(dtype, mode, order):
    x = _input("contiguous", dtype)
    upstream = torch.randn_like(x[..., :14336])
    op = partial(TritonSwiGLUOp(), packed_order=order)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = op(x)
        if mode == "backward":

            def run():
                return torch.autograd.grad(y, x, upstream, retain_graph=True)[0]

        elif mode == "forward_backward":

            def run():
                return torch.autograd.grad(op(x), x, upstream)[0]

        else:

            def run():
                with torch.no_grad():
                    return op(x)

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for _ in range(3):
            graph.replay()
        _assert_bits(captured, run())
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


@pytest.mark.parametrize("shape", [(), (2, 0), (2, 3)])
def test_rejects_invalid_shape(shape):
    with pytest.raises(ValueError):
        TritonSwiGLUOp()(torch.empty(shape, device="cuda"), packed_order=SwiGLUPackedOrder.GATE_UP)


def test_rejects_unsupported_dtype_and_cpu():
    with pytest.raises(TypeError, match="torch.int32"):
        TritonSwiGLUOp()(
            torch.ones(2, 4, device="cuda", dtype=torch.int32),
            packed_order=SwiGLUPackedOrder.GATE_UP,
        )
    with pytest.raises(RuntimeError, match="cpu"):
        TritonSwiGLUOp()(torch.ones(2, 4), packed_order=SwiGLUPackedOrder.GATE_UP)


@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
def test_rejects_ambiguous_or_inapplicable_order(method):
    fn = getattr(TritonSwiGLUOp(), method)
    x = torch.ones(2, 8, device="cuda")
    with pytest.raises(ValueError, match="packed_order"):
        fn(x)
    with pytest.raises(ValueError, match="single packed input"):
        fn(x, x, packed_order=SwiGLUPackedOrder.UP_GATE)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_noncurrent_device_and_caller_restoration(dtype, method, order):
    with torch.cuda.device(0):
        x = torch.randn(2, 3, 28672, device="cuda:1", dtype=dtype).requires_grad_(True)
        fn = partial(getattr(TritonSwiGLUOp(), method), packed_order=order)
        y = fn(x)
        assert torch.cuda.current_device() == 0
        grad = torch.autograd.grad(y, x, torch.ones_like(y))[0]
        assert torch.cuda.current_device() == 0
        with torch.cuda.device(1):
            expected = fn(x)
            expected_grad = torch.autograd.grad(expected, x, torch.ones_like(expected))[0]
        _assert_bits(y, expected)
        _assert_bits(grad, expected_grad)
