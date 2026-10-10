# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regressions for benchmark ordering and backward timing boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_engine.validation.models import h3_report


@pytest.fixture
def cuda_clock(monkeypatch):
    clock = SimpleNamespace(time=0.0, active=False, log=[], allocated=0, peak=0, events=0)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.start = clock.events % 2 == 0
            clock.events += 1

        def record(self):
            clock.active = self.start
            clock.log.append("start" if self.start else "end")
            self.time = clock.time

        def synchronize(self):
            assert not clock.active

        def elapsed_time(self, end):
            return end.time - self.time

    def reset_peak():
        clock.log.append("reset_peak")
        clock.peak = clock.allocated

    def allocated():
        clock.log.append("memory_baseline")
        return clock.allocated

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: clock.log.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: clock.peak)
    return clock


def test_setup_is_outside_backward_events_and_memory_baseline(cuda_clock):
    states = []

    def setup():
        assert not cuda_clock.active
        cuda_clock.log.append("setup")
        cuda_clock.time += 50.0
        cuda_clock.allocated += 64 * 2**20
        return object()

    def backward(state):
        states.append(state)
        cuda_clock.log.append("backward")
        cuda_clock.time += 2.0
        cuda_clock.allocated += 3 * 2**20
        cuda_clock.peak = cuda_clock.allocated

    assert h3_report.time_us(backward, warmup=1, iters=2, setup=setup) == 2000.0
    assert len({id(state) for state in states}) == 3
    assert cuda_clock.log.count("start") == 2
    for index, entry in enumerate(cuda_clock.log):
        if entry == "start":
            assert cuda_clock.log[index - 1] == "setup"
            assert cuda_clock.log[index + 1 : index + 3] == ["backward", "end"]

    cuda_clock.log.clear()
    assert h3_report.peak_mib(backward, setup=setup) == 3.0
    assert cuda_clock.log == [
        "setup",
        "synchronize",
        "reset_peak",
        "memory_baseline",
        "backward",
        "synchronize",
    ]


@pytest.mark.parametrize("keys", [("candidate", "provider"), h3_report.TIMED_KEYS])
def test_measure_interleaves_and_records_both_orders(cuda_clock, keys):
    calls = []
    case = {"op": "test", "case": "tiny", "backend": "cpu", "bytes": 4096}
    for position, key in enumerate(keys, start=1):

        def run(key=key, duration=position):
            calls.append(key)
            cuda_clock.time += duration

        case[key] = run

    row = h3_report.measure(case, warmup=2, iters=4)
    first, second = list(keys), list(reversed(keys))
    assert calls == (first + second) * 3 + first  # warmup, samples, then memory calls
    assert row["execution_order"] == {
        "policy": "alternating",
        "iteration_0": first,
        "iteration_1": second,
    }
    for position, key in enumerate(keys, start=1):
        assert row[f"{key}_us"] == position * 1000.0
        assert row[f"{key}_gbps"] == pytest.approx(4096 / (position * 1e-3) / 1e9)
        assert row[f"{key}_peak_mib"] == 0.0
    assert not any(key.endswith("_setup") for key in row)


@pytest.mark.parametrize("operator", ["norm", "gather", "gate", "final"])
def test_backward_perf_cases_prepare_fresh_cpu_graphs_before_events(
    monkeypatch, cuda_clock, operator
):
    prepared_leaves = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def track_backward(outputs):
        tensors = outputs if isinstance(outputs, tuple) else (outputs,)

        def backward_hook(grad):
            cuda_clock.time += 2.0 / len(tensors)
            return grad

        for tensor in tensors:
            tensor.register_hook(backward_hook)
        return outputs

    def track(fn, leaves, *args):
        if leaves[0].requires_grad:
            assert not cuda_clock.active
            prepared_leaves.append([leaf if leaf.is_leaf else leaf._base for leaf in leaves])
        cuda_clock.time += 1.0
        outputs = fn(*leaves, *args)
        return track_backward(outputs) if leaves[0].requires_grad else outputs

    if operator == "norm":

        def inputs(seq):
            return (
                torch.randn(1, seq, 4),
                torch.ones(4),
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.arange(seq) % 3,
            )

        provider = h3_report.provider_norm_modulate

        def forward(x, weight, shift, scale, index, **kwargs):
            return track(provider, [x, weight, shift, scale], index)

        monkeypatch.setattr(h3_report, "NORM_SEQ_LENS", (2, 3))
        monkeypatch.setattr(h3_report, "_norm_inputs", inputs)
        monkeypatch.setattr(h3_report, "provider_norm_modulate", forward)
        op = SimpleNamespace(forward_modulated=forward)
        build = h3_report._norm_perf
    elif operator == "gate":

        def inputs(seq):
            return (
                torch.randn(3, 24),
                torch.randn(1, seq, 4),
                torch.randn(1, seq, 4),
                torch.arange(seq) % 3,
            )

        provider = h3_report.provider_gate_residual

        def forward(residual, y, gate, index, **kwargs):
            return track(lambda r, y_, g: provider(r, g, index, y_), [residual, y, gate])

        monkeypatch.setattr(h3_report, "NORM_SEQ_LENS", (2, 3))
        monkeypatch.setattr(h3_report, "_gate_inputs", inputs)
        monkeypatch.setattr(
            h3_report, "provider_gate_residual", lambda r, g, i, y: forward(r, y, g, i)
        )
        op = SimpleNamespace(forward=forward)
        build = h3_report._gate_perf
    elif operator == "final":

        def inputs(seq):
            return (
                torch.randn(1, seq, 4),
                torch.ones(4),
                torch.randn(3, 2),
                torch.randn(8, 2),
                torch.randn(8),
                torch.arange(seq) % 3,
            )

        def forward(x, nw, temb, w, b, ti):
            def compute(x, nw, temb, w, b):
                rows = torch.nn.functional.linear(torch.nn.functional.silu(temb), w, b)
                shift, scale = rows.chunk(2, dim=-1)
                norm = torch.nn.functional.rms_norm(x, (4,), nw)
                return norm * (1 + scale[ti]) + shift[ti]

            return track(compute, [x, nw, temb, w, b])

        monkeypatch.setattr(h3_report, "NORM_SEQ_LENS", (2, 3))
        monkeypatch.setattr(h3_report, "_final_inputs", inputs)
        monkeypatch.setattr(h3_report, "provider_final_adaln_out", forward)
        op = forward
        build = h3_report._final_perf
    else:
        randn = torch.randn

        def cpu_randn(*args, **kwargs):
            kwargs["device"] = "cpu"
            return randn(*args, **kwargs)

        provider = h3_report.provider_adaln_row_gather

        def forward(rows, ti, tags, **kwargs):
            def gather(leaf, ti, tags):
                return provider(leaf.chunk(6, dim=-1), ti, tags)

            return track(gather, [rows], ti, tags)

        def provider_forward(chunks, ti, tags):
            if chunks[0].requires_grad:
                assert not cuda_clock.active
                prepared_leaves.append([chunks[0]._base])
            cuda_clock.time += 1.0
            outputs = provider(chunks, ti, tags)
            return track_backward(outputs) if chunks[0].requires_grad else outputs

        monkeypatch.setattr(torch, "randn", cpu_randn)
        monkeypatch.setattr(h3_report, "GATHER_SEQ_LENS", (2, 3))
        monkeypatch.setattr(
            h3_report,
            "h3_packed_layout",
            lambda seq, num, seed: (torch.arange(seq) % num, torch.arange(seq) % 3),
        )
        monkeypatch.setattr(h3_report, "provider_adaln_row_gather", provider_forward)
        op = SimpleNamespace(forward=forward)
        build = h3_report._gather_perf

    try:
        registry = SimpleNamespace(get_op=lambda *args, **kwargs: op)
        for case in build(registry):
            row = h3_report.measure(case, warmup=1, iters=2)
            assert row["backward_timing_scope"] == "backward_only"
            assert row["candidate_backward_us"] == pytest.approx(2000.0)
            assert row["provider_backward_us"] == pytest.approx(2000.0)
    finally:
        torch.set_num_threads(previous_threads)

    assert prepared_leaves
    assert len({id(leaves[0]) for leaves in prepared_leaves}) == len(prepared_leaves)
    assert all(leaf.grad is not None for leaves in prepared_leaves for leaf in leaves)
