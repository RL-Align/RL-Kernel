# SPDX-License-Identifier: Apache-2.0
"""Accuracy is independent FP64 complex algebra; invariance compares raw bytes."""

import gc
import math

import pytest
import torch

from rl_engine.ops.rope.multi_axis_rope import (
    build_multi_axis_cos_sin,
    get_multi_axis_rope_op,
    multi_axis_rope,
    qwen_image_positions,
)


def assert_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def complex_reference(x, cos, sin, *, transpose=False):
    pairs = x.double().reshape(*x.shape[:-1], 64, 2)
    z = torch.view_as_complex(pairs.contiguous())
    phase = torch.complex(cos.double(), sin.double()).unsqueeze(-2)
    if transpose:
        phase = phase.conj()
    return torch.view_as_real(z * phase).flatten(-2)


@pytest.fixture(params=["reference", "cuda", "triton"])
def implementation(request):
    backend = request.param
    device = "cpu" if backend == "reference" else "cuda"
    if device == "cuda" and (not torch.cuda.is_available() or torch.version.hip is not None):
        pytest.skip("NVIDIA CUDA required")
    if backend == "cuda":
        from rl_engine.backends.cuda.rope.multi_axis_rope import native_available

        if not native_available():
            pytest.skip("build native multi_axis_rope or use the focused CUDA validation runner")
    if backend == "triton":
        pytest.importorskip("triton")
    return get_multi_axis_rope_op(backend, device=device), device


def inputs(device, dtype=torch.float32, *, batch=2, seq=7, heads=3, batched=False):
    generator = torch.Generator().manual_seed(386)
    x = torch.randn(batch, seq, heads, 128, generator=generator).to(device, dtype)
    shape = (batch, seq, 3) if batched else (seq, 3)
    pos = torch.randint(-42, 53, shape, generator=generator)
    cos, sin = build_multi_axis_cos_sin(pos, device=device)
    return x, cos, sin


@pytest.mark.parametrize("height,width", [(2, 4), (3, 5), (64, 64), (83, 83), (104, 58)])
def test_positions_and_axis_frequencies(height, width):
    image, text = qwen_image_positions(height, width, 3)
    # Explicit flattened endpoints and a row boundary catch swapped H/W and odd-grid shifts.
    assert image[0].tolist() == [0, -((height + 1) // 2), -((width + 1) // 2)]
    assert image[-1].tolist() == [0, height // 2 - 1, width // 2 - 1]
    assert image[width].tolist() == [0, -((height + 1) // 2) + 1, -((width + 1) // 2)]
    assert text[:, 0].tolist() == list(
        range(max(height // 2, width // 2), max(height // 2, width // 2) + 3)
    )
    cos, sin = build_multi_axis_cos_sin(image[[0, -1]])
    # Independent scalar FP64 trig oracle, per-axis exponents reset at 8 and 36 pairs.
    for row, position in enumerate(image[[0, -1]].tolist()):
        offset = 0
        for axis, width_axis in enumerate((16, 56, 56)):
            for pair in range(width_axis // 2):
                angle = position[axis] / (10000.0 ** (2 * pair / width_axis))
                assert abs(cos[row, offset + pair].item() - math.cos(angle)) < 4e-6
                assert abs(sin[row, offset + pair].item() - math.sin(angle)) < 4e-6
            offset += width_axis // 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
def test_forward_backward_accuracy(implementation, dtype, batched):
    op, device = implementation
    x, cos, sin = inputs(device, dtype, batched=batched)
    x.requires_grad_()
    grad = torch.randn_like(x)
    result = op(x, cos, sin)
    dx = torch.autograd.grad(result, x, grad)[0]
    tolerance = {torch.float32: 8e-7, torch.float16: 0.002, torch.bfloat16: 0.02}[dtype]
    torch.testing.assert_close(
        result.double(), complex_reference(x, cos, sin), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        dx.double(),
        complex_reference(grad, cos, sin, transpose=True),
        atol=tolerance,
        rtol=tolerance,
    )


def test_adjacent_pair_layout_not_rotate_half(implementation):
    op, device = implementation
    x = torch.arange(128, dtype=torch.float32, device=device).reshape(1, 1, 1, 128)
    cos = torch.zeros(1, 64, device=device)
    sin = torch.ones_like(cos)
    y = op(x, cos, sin)
    assert_bits(y[..., 0::2], -x[..., 1::2])
    assert_bits(y[..., 1::2], x[..., 0::2])


def test_batch_position_padding_and_chunk_invariance(implementation):
    op, device = implementation
    x, cos, sin = inputs(device, batched=True, batch=3, seq=19)
    x.requires_grad_()
    grad = torch.randn_like(x)
    y = op(x, cos, sin)
    dx = torch.autograd.grad(y, x, grad)[0]
    # Reordering samples with different positions must not reuse sample zero's table.
    permutation = torch.tensor([2, 0, 1], device=device)
    assert_bits(op(x[permutation], cos[permutation], sin[permutation]), y[permutation])
    for b in range(3):
        for lo, hi in [(0, 1), (1, 8), (8, 19)]:
            xx = x[b : b + 1, lo:hi].detach().requires_grad_()
            yy = op(xx, cos[b, lo:hi], sin[b, lo:hi])
            dd = torch.autograd.grad(yy, xx, grad[b : b + 1, lo:hi])[0]
            assert_bits(yy, y[b : b + 1, lo:hi])
            assert_bits(dd, dx[b : b + 1, lo:hi])
    padded = torch.cat((x.detach(), torch.randn(3, 5, 3, 128, device=device)), dim=1)
    pc = torch.cat((cos, torch.ones(3, 5, 64, device=device)), dim=1)
    ps = torch.cat((sin, torch.zeros(3, 5, 64, device=device)), dim=1)
    assert_bits(op(padded, pc, ps)[:, :19], y)


def test_noncontiguous_input_tables_and_gradient(implementation):
    op, device = implementation
    base, cos, sin = inputs(device, batched=True, seq=14, heads=4)
    x = base[:, ::2, ::2].detach().requires_grad_()
    c, s = cos[:, ::2], sin[:, ::2]
    grad = torch.randn_like(base)[:, ::2, ::2]
    result = op(x, c, s)
    dx = torch.autograd.grad(result, x, grad)[0]
    xx = x.detach().contiguous().requires_grad_()
    yy = op(xx, c.contiguous(), s.contiguous())
    dd = torch.autograd.grad(yy, xx, grad.contiguous())[0]
    assert_bits(result, yy)
    assert_bits(dx, dd)


@pytest.mark.parametrize("batch,seq", [(0, 7), (2, 0), (0, 0)])
def test_empty_forward_backward(implementation, batch, seq):
    op, device = implementation
    x, cos, sin = inputs(device, batch=batch, seq=seq)
    x.requires_grad_()
    result = op(x, cos, sin)
    result.sum().backward()
    assert result.shape == x.shape and x.grad.shape == x.shape


def test_second_derivative(implementation):
    op, device = implementation
    x, cos, sin = inputs(device, batch=1, seq=2)
    x.requires_grad_()
    y = op(x, cos, sin)
    first = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    # R^T R = (cos^2 + sin^2) I, including the small table rounding error.
    expected = (
        (2 * (cos.double().square() + sin.double().square()))
        .repeat_interleave(2, -1)[None, :, None, :]
        .expand_as(x)
    )
    torch.testing.assert_close(second.double(), expected, atol=8e-7, rtol=8e-7)


@pytest.mark.parametrize(
    "problem", ["shape", "dtype", "table_shape", "table_dtype", "table_grad", "device"]
)
def test_reject_invalid_inputs(implementation, problem):
    op, device = implementation
    x, cos, sin = inputs(device)
    if problem == "shape":
        x = x[..., :64]
    elif problem == "dtype":
        x = x.double()
    elif problem == "table_shape":
        sin = sin[:-1]
    elif problem == "table_dtype":
        cos = cos.half()
    elif problem == "table_grad":
        cos.requires_grad_()
    elif problem == "device":
        sin = torch.empty_like(sin, device="meta")
    with pytest.raises((ValueError, TypeError)):
        op(x, cos, sin)


def test_explicit_backend_and_reference_dispatch(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("multi_axis_rope", device="cpu")
    assert op.trace()["actual_backend"].startswith("rlkernel.reference.")
    from rl_engine.backends.cuda.rope import multi_axis_rope as cuda_module

    monkeypatch.setattr(cuda_module, "native_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        get_multi_axis_rope_op("cuda")
    with pytest.raises(ValueError, match="unknown"):
        get_multi_axis_rope_op("typo")
    x, cos, sin = inputs("cpu")
    assert_bits(multi_axis_rope(x, cos, sin), op(x, cos, sin))


def test_cuda_registry_uses_native_when_present():
    from rl_engine.backends.cuda.rope.multi_axis_rope import native_available
    from rl_engine.runtime.registry import KernelRegistry

    if not torch.cuda.is_available() or not native_available():
        pytest.skip("native CUDA multi_axis_rope extension required")
    op = KernelRegistry().get_op("multi_axis_rope", device="cuda")
    assert op.trace()["actual_backend"] == "rlkernel.cuda.qwenimage_multi_axis_rope.v1"
    x, cos, sin = inputs("cuda")
    torch.testing.assert_close(
        op(x, cos, sin).double(), complex_reference(x, cos, sin), atol=8e-7, rtol=8e-7
    )


def test_cuda_registry_falls_back_to_triton(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.rope import multi_axis_rope as cuda_module

    monkeypatch.setattr(cuda_module, "native_available", lambda: False)
    op = KernelRegistry().get_op("multi_axis_rope", device="cuda")
    assert op.trace()["actual_backend"] == "rlkernel.triton.qwenimage_multi_axis_rope.v1"
    x, cos, sin = inputs("cuda")
    torch.testing.assert_close(
        op(x, cos, sin).double(), complex_reference(x, cos, sin), atol=8e-7, rtol=8e-7
    )


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.int64), torch.full((2, 3), 4096)],
)
def test_reject_invalid_positions(bad):
    with pytest.raises((TypeError, ValueError)):
        build_multi_axis_cos_sin(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_triton_byte_parity_and_graph(dtype):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    from rl_engine.backends.cuda.rope.multi_axis_rope import native_available

    if not native_available():
        pytest.skip("native multi_axis_rope extension required")
    pytest.importorskip("triton")
    cuda, triton = get_multi_axis_rope_op("cuda"), get_multi_axis_rope_op("triton")
    x, cos, sin = inputs("cuda", dtype, batched=True, seq=17, heads=24)
    x.requires_grad_()
    grad = torch.randn_like(x)
    yc, yt = cuda(x, cos, sin), triton(x, cos, sin)
    dc = torch.autograd.grad(yc, x, grad)[0]
    dt = torch.autograd.grad(yt, x, grad)[0]
    assert_bits(yc, yt)
    assert_bits(dc, dt)
    reference = get_multi_axis_rope_op("reference")
    yr = reference(x, cos, sin)
    dr = torch.autograd.grad(yr, x, grad)[0]
    assert_bits(yc, yr)
    assert_bits(dc, dr)
    # Capture forward + autograd after warmup on a side stream.
    for op in (cuda, triton):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            static_x = x.detach().clone().requires_grad_()
            static_grad = grad.clone()
            for _ in range(3):
                torch.autograd.grad(op(static_x, cos, sin), static_x, static_grad)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.graph(graph, stream=stream):
                result = op(static_x, cos, sin)
                derivative = torch.autograd.grad(result, static_x, static_grad)[0]
        finally:
            if was_enabled:
                gc.enable()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert_bits(result, yc)
        assert_bits(derivative, dc)
        # Replay must consume updated static buffers, not just retain captured outputs.
        with torch.no_grad():
            static_x.copy_(x * 2)
            static_grad.copy_(grad * 0.5)
        graph.replay()
        torch.cuda.synchronize()
        expected = op(static_x, cos, sin)
        expected_dx = torch.autograd.grad(expected, static_x, static_grad)[0]
        assert_bits(result, expected)
        assert_bits(derivative, expected_dx)


def test_native_entrypoint_checks_and_nondefault_stream():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    from rl_engine.backends.cuda.rope.multi_axis_rope import native_available

    if not native_available():
        pytest.skip("native multi_axis_rope extension required")
    x, cos, sin = inputs("cuda")
    native = torch.ops.rl_kernel_native.multi_axis_rope
    for c, s in [(cos.half(), sin), (cos, sin[:-1]), (cos.cpu(), sin)]:
        with pytest.raises(RuntimeError):
            native(x, c, s, False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        xx = x * 3
        y = native(xx, cos, sin, False)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(y.double(), complex_reference(xx, cos, sin), atol=2e-6, rtol=2e-6)


def test_device_guard():
    if torch.cuda.device_count() < 2 or torch.version.hip is not None:
        pytest.skip("requires two NVIDIA GPUs")
    from rl_engine.backends.cuda.rope.multi_axis_rope import native_available

    if not native_available():
        pytest.skip("native multi_axis_rope extension required")
    with torch.cuda.device(0):
        x, cos, sin = inputs("cuda:1")
        for backend in ("cuda", "triton"):
            y = get_multi_axis_rope_op(backend)(x, cos, sin)
            assert y.device == x.device
            assert torch.cuda.current_device() == 0
            torch.testing.assert_close(
                y.double(), complex_reference(x, cos, sin), atol=8e-7, rtol=8e-7
            )


@pytest.mark.parametrize("default_device", ["meta", "cuda"])
def test_canonical_tables_ignore_default_device(default_device):
    if default_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA context requires CUDA")
    positions, text = qwen_image_positions(3, 5, 7)
    expected = build_multi_axis_cos_sin(positions)
    with torch.device(default_device):
        image_again, text_again = qwen_image_positions(3, 5, 7)
        actual = build_multi_axis_cos_sin(positions, device="cpu")
    assert image_again.device.type == "cpu" and text_again.device.type == "cpu"
    assert_bits(image_again, positions)
    assert_bits(text_again, text)
    for a, b in zip(actual, expected, strict=True):
        assert_bits(a, b)


def test_packed_qkv_and_independent_table_strides(implementation):
    op, device = implementation
    packed = torch.randn(2, 17, 3, 3, 256, device=device)
    x = packed[:, :, 0, :, ::2].detach().requires_grad_()
    cos_base = torch.randn(2, 17, 128, device=device)
    cos = cos_base[..., ::2]
    sin = torch.randn(17, 2, 64, device=device).transpose(0, 1)
    grad = torch.randn_like(x)
    result = op(x, cos, sin)
    dx = torch.autograd.grad(result, x, grad)[0]
    assert result.is_contiguous()
    torch.testing.assert_close(
        result.double(), complex_reference(x, cos, sin), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        dx.double(), complex_reference(grad, cos, sin, transpose=True), atol=2e-6, rtol=2e-6
    )
    if device == "cuda":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            op(x, cos, sin)
        copies = {event.key for event in profile.key_averages()} & {
            "aten::contiguous",
            "aten::clone",
            "aten::copy_",
        }
        assert not copies, f"RoPE materialized its strided inputs: {copies}"


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_dispatcher_fake_autograd_and_fullgraph_compile(backend):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    if backend == "cuda":
        from rl_engine.backends.cuda.rope.multi_axis_rope import _rotation, native_available

        if not native_available():
            pytest.skip("build native RoPE library")
    else:
        pytest.importorskip("triton")
        from rl_engine.backends.shared.triton.rope.multi_axis_rope import _rotation
    op = get_multi_axis_rope_op(backend)
    x, cos, sin = inputs("cuda", batched=True, seq=5, heads=2)
    x.requires_grad_()
    checks = torch.library.opcheck(_rotation, (x, cos, sin, False))
    assert all(value == "SUCCESS" for value in checks.values())
    compiled = torch.compile(op, fullgraph=True)
    grad = torch.randn_like(x)
    expected = op(x, cos, sin)
    expected_dx = torch.autograd.grad(expected, x, grad)[0]
    result = compiled(x, cos, sin)
    dx = torch.autograd.grad(result, x, grad)[0]
    assert_bits(result, expected)
    assert_bits(dx, expected_dx)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.ws1
def test_standard_ws1_four_judgments(backend):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    if backend == "cuda":
        from rl_engine.backends.cuda.rope.multi_axis_rope import native_available

        if not native_available():
            pytest.skip("build native RoPE library")
    else:
        pytest.importorskip("triton")
    from rl_engine.validation.operators.qwen_image_rope import run_multi_axis_rope_ws1

    result = run_multi_axis_rope_ws1(backend)
    assert result["passed"]
    assert result["trace"]["actual_backend"].startswith(f"rlkernel.{backend}.")
    for phase in ("forward", "gradient"):
        assert result[phase]["accuracy_reports"]
        assert result[phase]["invariance_reports"]
        assert all(report["passed"] for report in result[phase]["invariance_reports"])
