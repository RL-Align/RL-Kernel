# SPDX-License-Identifier: Apache-2.0
"""Target-device checks. RLK_REQUIRE_ROCM=1 turns missing hardware into failure."""

import os

import pytest
import torch

from rl_engine.ops.moe.route_weight_grad import route_weight_grad
from rl_engine.reference.moe.route_weight_grad import route_weight_grad_reference as reference


@pytest.fixture(scope="module", autouse=True)
def require_device():
    available = bool(torch.version.hip and torch.cuda.is_available())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if available else "none"
    if arch not in ("gfx942", "gfx950"):
        reason = f"task 49 needs gfx942/gfx950, found {arch}"
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def same_bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def inputs(m, f, dtype, seed=492):
    rng = torch.Generator().manual_seed(seed)
    dh = torch.randn(m, 2 * f, generator=rng).to(dtype)[:, ::2]
    silu = torch.randn(m, 2 * f, generator=rng)[:, ::2]
    up = (torch.randn(m, 2 * f, generator=rng) * 20).to(dtype)[:, ::2]
    # Preserve strides on device as well.
    result = []
    for value in (dh, silu, up):
        target = torch.empty((m, 2 * f), dtype=value.dtype, device="cuda")[:, ::2]
        target.copy_(value)
        result.append(target)
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 255, 256, 257, 769, 2049])
def test_ordered_accuracy_batch_positions_and_strides(dtype, width):
    row = inputs(1, width, dtype)
    expected = route_weight_grad(*row)
    same_bytes(expected.cpu(), reference(*(x.cpu() for x in row)))
    for m in (1, 16, 17, 64, 65):
        for pos in sorted({0, m // 2, m - 1}):
            values = inputs(m, width, dtype, seed=493)
            for x, r in zip(values, row):
                x[pos] = r[0]
            actual = route_weight_grad(*values)
            same_bytes(expected, actual[pos : pos + 1])
            same_bytes(actual, route_weight_grad(*values))
            same_bytes(actual, route_weight_grad(*(x.contiguous() for x in values)))


def test_empty_batch_and_zero_stride_inputs():
    x = torch.empty((0, 33), device="cuda")
    assert route_weight_grad(x, x, x).shape == (0,)
    x = torch.tensor([[2.0]], device="cuda").expand(17, 257)
    expected = torch.full((17,), 8.0 * 257, device="cuda")
    same_bytes(route_weight_grad(x, x, x), expected)


def test_overflow_clamp_and_reduction_order():
    dh = torch.zeros((3, 769), device="cuda")
    s, u = torch.ones_like(dh), torch.ones_like(dh)
    dh[0, :4] = torch.tensor([1e20, 1, -1e20, 1], device="cuda")
    dh[1, [0, 256, 512]] = torch.tensor([1e20, -1e20, 3], device="cuda")
    dh[2, 0], s[2, 0], u[2, 0] = torch.finfo(torch.float32).max, 2, 0.25
    actual = route_weight_grad(dh, s, u)
    same_bytes(actual.cpu(), reference(dh.cpu(), s.cpu(), u.cpu()))
    assert actual[0] == 0 and actual[1] == 3 and torch.isposinf(actual[2])
    u[2, 0] = 0  # inf * 0 must propagate NaN, not recover a finite value.
    assert torch.isnan(route_weight_grad(dh, s, u)[2])
    x = torch.ones((1, 4), device="cuda")
    up = torch.tensor([[-11, -10, 10, 11]], device="cuda", dtype=torch.float32)
    assert route_weight_grad(x, x, up).item() == 0


def test_streams_graph_replay_and_input_updates():
    a, b = inputs(17, 513, torch.float32), inputs(3, 769, torch.bfloat16, seed=494)
    expected_a, expected_b = route_weight_grad(*a), route_weight_grad(*b)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    current = torch.cuda.current_stream()
    outputs = []
    for stream, values in zip(streams, (a, b)):
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            outputs.append(route_weight_grad(*values))
    for stream in streams:
        current.wait_stream(stream)
    for actual, expected in zip(outputs, (expected_a, expected_b)):
        same_bytes(actual, expected)
    # Warm both kernels before capture; each capture owns its workspace.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = route_weight_grad(*a)
    for factor in (0.5, -2.0):
        a[0].mul_(factor)
        graph.replay()
        same_bytes(captured, route_weight_grad(*a))
