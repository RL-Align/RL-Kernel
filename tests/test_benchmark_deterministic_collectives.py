# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for collective benchmark timing boundaries."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_deterministic_collectives as benchmark


@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("with_prepare", [False, True])
def test_paired_median_us_excludes_preparation(monkeypatch, warmup, with_prepare):
    """Alternate each sample's order and reset NCCL inputs outside timing events."""
    clock_us = 0
    calls = []
    value = 1

    class Event:
        """Model CUDA events on a stream with a deterministic virtual clock."""

        def __init__(self, *, enable_timing):
            """Accept the same timing flag as a CUDA event."""
            assert enable_timing
            self.timestamp = None

        def record(self):
            """Record the stream's current virtual timestamp."""
            self.timestamp = clock_us
            calls.append("event")

        def synchronize(self):
            """Represent completion of the recorded event."""
            calls.append("end_sync")

        def elapsed_time(self, end):
            """Return elapsed milliseconds, matching the CUDA event API."""
            return (end.timestamp - self.timestamp) / 1e3

    def prepare():
        """Spend time resetting the mutable collective input."""
        nonlocal clock_us, value
        clock_us += 1000
        value = 1
        calls.append("prepare")

    def deterministic():
        """Model an out-of-place reduction with seven microseconds of work."""
        nonlocal clock_us
        clock_us += 7
        calls.append("deterministic")

    def nccl():
        """Model an in-place reduction with eleven microseconds of work."""
        nonlocal clock_us, value
        if with_prepare:
            assert value == 1
        value *= 2
        clock_us += 11
        calls.append("nccl")

    monkeypatch.setattr(benchmark.torch.cuda, "Event", Event)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: calls.append("sync"))
    measured = benchmark._paired_median_us(
        deterministic, nccl, warmup, 3, prepare_nccl=prepare if with_prepare else None
    )

    assert measured == pytest.approx((7, 11))
    expected = []
    for count, timed in ((warmup, False), (3, True)):
        if timed:
            expected.append("sync")
        for sample in range(count):
            order = ("deterministic", "nccl") if sample % 2 == 0 else ("nccl", "deterministic")
            for name in order:
                if name == "nccl" and with_prepare:
                    expected.append("prepare")
                if timed:
                    expected.append("event")
                expected.append(name)
                if timed:
                    expected.extend(("event", "end_sync"))
    assert calls == expected


def test_reduction_callbacks_reuse_outputs(monkeypatch):
    """Reuse separate reduction buffers while preserving and resetting inputs."""
    outputs = {name: [] for name in ("reduce_scatter", "all_reduce")}
    nccl_outputs = {name: [] for name in outputs}
    inputs = []

    class Collective:
        """Model deterministic reductions on CPU with caller-provided outputs."""

        def __init__(self, **kwargs):
            """Accept the distributed benchmark configuration."""

        def all_gather(self, *args, **kwargs):
            """Reject gather calls in this reductions-only benchmark."""
            pytest.fail("only reduction callbacks should run")

        all_gather_many = all_gather

        def reduce_scatter(self, value, *, out):
            """Write a two-rank reduction into the supplied output buffer."""
            outputs["reduce_scatter"].append(out)
            return out.copy_(value.view(2, -1).sum(dim=0))

        def all_reduce(self, value, *, out):
            """Preserve the source while writing a simulated two-rank sum."""
            inputs.append(value.clone())
            assert out is not value
            outputs["all_reduce"].append(out)
            return out.copy_(value * 2)

        def close(self):
            """Release the simulated collective."""

    def nccl_scatter(out, value):
        """Model NCCL's out-of-place reduce-scatter contract."""
        nccl_outputs["reduce_scatter"].append(out)
        out.copy_(value.view(2, -1).sum(dim=0))

    def nccl_reduce(value):
        """Require a reset before each simulated in-place NCCL reduction."""
        torch.testing.assert_close(value, inputs[-1])
        nccl_outputs["all_reduce"].append(value)
        value.mul_(2)

    def measure(deterministic, nccl, warmup, iterations, *, prepare_nccl):
        """Invoke the real benchmark callbacks repeatedly without GPU timing."""
        assert (warmup, iterations) == (1, 2)
        for _ in range(warmup + iterations):
            deterministic()
            if prepare_nccl is not None:
                prepare_nccl()
            nccl()
        return 7.0, 11.0

    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda rank: None),
        device=torch.device,
        empty=lambda *shape, device=None, **kwargs: torch.empty(*shape, **kwargs),
        randn=lambda *shape, device=None: torch.randn(*shape),
        empty_like=torch.empty_like,
    )
    cpu_dist = SimpleNamespace(
        init_process_group=lambda *args, **kwargs: None,
        get_world_size=lambda: 2,
        all_gather_into_tensor=lambda *args: None,
        reduce_scatter_tensor=nccl_scatter,
        all_reduce=nccl_reduce,
        destroy_process_group=lambda: None,
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(benchmark, "torch", cpu_torch)
    monkeypatch.setattr(benchmark, "dist", cpu_dist)
    monkeypatch.setattr(benchmark, "DeterministicCollective", Collective)
    monkeypatch.setattr(benchmark, "_paired_median_us", measure)
    monkeypatch.setattr(benchmark, "_slowest", lambda value: value)
    benchmark.main(
        [
            "--size-bytes",
            "32",
            "--operations",
            "reduce_scatter",
            "all_reduce",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ]
    )

    for name in outputs:
        assert len(outputs[name]) == len(nccl_outputs[name]) == 3
        assert all(out is outputs[name][0] for out in outputs[name])
        assert all(out is nccl_outputs[name][0] for out in nccl_outputs[name])
        assert outputs[name][0] is not nccl_outputs[name][0]
        torch.testing.assert_close(outputs[name][0], nccl_outputs[name][0])
    assert all(torch.equal(value, inputs[0]) for value in inputs)
