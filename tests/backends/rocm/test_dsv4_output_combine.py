# SPDX-License-Identifier: Apache-2.0
"""MI300X/MI355X acceptance; RLK_REQUIRE_ROCM=1 forbids missing-device skips."""

import os

import pytest
import torch

from rl_engine.contracts.operators.fixed_output_combine import (
    prepare_fixed_output_combine as prepare,
)
from rl_engine.ops.moe.output_combine import output_combine
from rl_engine.reference.moe.fixed_output_combine import fixed_output_combine_reference as reference
from tests.helpers.dsv4_output_combine import move, request


@pytest.fixture(scope="module", autouse=True)
def require_device():
    available = bool(torch.version.hip and torch.cuda.is_available())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if available else "none"
    if arch not in ("gfx942", "gfx950"):
        reason = f"P6 acceptance requires gfx942/gfx950, found {arch}"
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def raw_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def check(plan, cpu):
    cp = prepare(plan, cpu, expected_mapping_run_id="run-53", num_experts=16)
    expected = reference(cp, cpu.expert_output, cpu.shared_output, mapping_run_id="run-53")
    gpu = move(cpu, "cuda")
    p = prepare(plan, gpu, expected_mapping_run_id="run-53", num_experts=16)
    actual = output_combine(
        p, gpu.expert_output, gpu.shared_output, mapping_run_id="run-53", debug=True
    )
    for name in ("output", "slot_steps", "before_cast"):
        raw_equal(getattr(actual, name).cpu(), getattr(expected, name))
    for _ in range(2):
        raw_equal(
            output_combine(p, gpu.expert_output, gpu.shared_output, mapping_run_id="run-53"),
            actual.output,
        )
    return actual.output, p, gpu


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("h", [1, 255, 257, 2049])
def test_trace_debug_toggle_tail_and_output_vjp(dtype, shared, h):
    plan, cpu = request(t=3, h=h, dtype=dtype, shared=shared, index_dtype=torch.int32)
    _, p, gpu = check(plan, cpu)
    x = gpu.expert_output.requires_grad_()
    s = None if gpu.shared_output is None else gpu.shared_output.requires_grad_()
    y = output_combine(p, x, s, mapping_run_id="run-53")
    dy = torch.linspace(-1, 1, y.numel(), device="cuda").reshape(y.shape).to(dtype)
    grads = torch.autograd.grad(y, (x,) if s is None else (x, s), dy)
    raw_equal(grads[0], dy[gpu.token_indices.long()])
    if shared:
        raw_equal(grads[1], dy)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_same_token_across_batches_positions_and_arrival_orders(dtype):
    plan, one = request(t=1, h=257, dtype=dtype)
    base, _, _ = check(plan, one)
    for t in (1, 16, 17, 64, 65):
        for pos in sorted({0, t // 2, t - 1}):
            plan, cpu = request(t=t, h=257, dtype=dtype)
            for slot in range(6):
                cpu.expert_output[(cpu.token_indices == pos) & (cpu.slot_indices == slot)] = (
                    one.expert_output[one.slot_indices == slot]
                )
            cpu.shared_output[pos] = one.shared_output[0]
            actual, _, _ = check(plan, cpu)
            raw_equal(actual[pos : pos + 1], base)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cancellation_signed_zero_and_shared_rounding(dtype):
    plan, cpu = request(t=1, h=4, dtype=dtype)
    p = prepare(plan, cpu, expected_mapping_run_id="run-53")
    rows = p.canonical_rows()[0]
    cpu.expert_output.zero_()
    cpu.shared_output.zero_()
    cpu.expert_output[rows[:4], 0] = torch.tensor([2**24, 1, -(2**24), 1], dtype=dtype)
    cpu.expert_output[:, 1] = -0.0
    cpu.expert_output[rows[:3], 2] = torch.tensor([1, 2**-24, 2**-24], dtype=dtype)
    cpu.expert_output[rows[:2], 3] = torch.tensor([1, 2**-8], dtype=dtype)
    cpu.shared_output[0, 3] = 2**-8
    check(plan, cpu)


def test_graph_value_updates_and_concurrent_execution():
    plan, cpu = request(t=3, h=257)
    _, p, gpu = check(plan, cpu)
    x, shared = gpu.expert_output, gpu.shared_output
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = output_combine(p, x, shared, mapping_run_id="run-53")
    for factor in (0.5, -2):
        x.mul_(factor)
        shared.mul_(factor)
        graph.replay()
        expected = output_combine(p, x, shared, mapping_run_id="run-53")
        raw_equal(captured, expected)
    current = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            outputs.append(output_combine(p, x, shared, mapping_run_id="run-53"))
    for stream in streams:
        current.wait_stream(stream)
    for out in outputs:
        raw_equal(out, expected)
