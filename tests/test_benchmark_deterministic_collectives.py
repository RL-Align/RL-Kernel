# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for collective benchmark timing boundaries."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_deterministic_collectives as benchmark


@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("with_prepare", [False, True])
def test_paired_samples_us_excludes_preparation(monkeypatch, warmup, with_prepare):
    """Alternate each sample's order and reset NCCL inputs outside timing events."""
    clock_us = 0
    calls = []
    value = 1
    deterministic_durations = [17, 5, 9, 4, 12]
    nccl_durations = [19, 8, 15, 6, 11]
    deterministic_calls = 0
    nccl_calls = 0

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
        """Model an out-of-place reduction with varying sample durations."""
        nonlocal clock_us, deterministic_calls
        clock_us += deterministic_durations[deterministic_calls]
        deterministic_calls += 1
        calls.append("deterministic")

    def nccl():
        """Model an in-place reduction with varying sample durations."""
        nonlocal clock_us, value, nccl_calls
        if with_prepare:
            assert value == 1
        value *= 2
        clock_us += nccl_durations[nccl_calls]
        nccl_calls += 1
        calls.append("nccl")

    monkeypatch.setattr(benchmark.torch.cuda, "Event", Event)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: calls.append("sync"))
    measured = benchmark._paired_samples_us(
        deterministic, nccl, warmup, 3, prepare_nccl=prepare if with_prepare else None
    )

    assert measured[0] == pytest.approx(deterministic_durations[warmup : warmup + 3])
    assert measured[1] == pytest.approx(nccl_durations[warmup : warmup + 3])
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


def test_gather_samples_us_preserves_rank_order_and_precision(monkeypatch):
    """Keep each rank's complete ordered samples without float32 rounding."""
    samples_by_rank = [[1.123456789, 3.987654321], [7.876543219, 2.234567891]]

    def gather(outputs, samples):
        """Populate gathered tensors in global-rank order."""
        assert samples.dtype == torch.float64
        assert samples.tolist() == samples_by_rank[1]
        for output, rank_samples in zip(outputs, samples_by_rank, strict=True):
            output.copy_(torch.tensor(rank_samples, dtype=torch.float64))

    cpu_torch = SimpleNamespace(
        tensor=lambda values, *, device, dtype: torch.tensor(values, dtype=dtype),
        float64=torch.float64,
        empty_like=torch.empty_like,
    )
    cpu_dist = SimpleNamespace(get_world_size=lambda: 2, all_gather=gather)
    monkeypatch.setattr(benchmark, "torch", cpu_torch)
    monkeypatch.setattr(benchmark, "dist", cpu_dist)

    assert benchmark._gather_samples_us(samples_by_rank[1]) == samples_by_rank


def test_reduction_callbacks_reuse_outputs_and_report_samples(monkeypatch, tmp_path):
    """Reuse reduction buffers and report all ranks with their maximum medians."""
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
        return [7.0, 19.0], [19.0, 17.0]

    def gather_samples(samples):
        """Model different slowest ranks for deterministic and NCCL timings."""
        if samples == [7.0, 19.0]:
            return [samples, [20.0, 10.0]]
        assert samples == [19.0, 17.0]
        return [samples, [21.0, 13.0]]

    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            set_device=lambda rank: None,
            get_device_properties=lambda rank: SimpleNamespace(name="test-gpu"),
        ),
        device=torch.device,
        empty=lambda *shape, device=None, **kwargs: torch.empty(*shape, **kwargs),
        randn=lambda *shape, device=None: torch.randn(*shape),
        empty_like=torch.empty_like,
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda"),
    )
    cpu_dist = SimpleNamespace(
        init_process_group=lambda *args, **kwargs: None,
        get_world_size=lambda: 2,
        get_rank=lambda: 0,
        all_gather_into_tensor=lambda *args: None,
        reduce_scatter_tensor=nccl_scatter,
        all_reduce=nccl_reduce,
        destroy_process_group=lambda: None,
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(benchmark, "torch", cpu_torch)
    monkeypatch.setattr(benchmark, "dist", cpu_dist)
    monkeypatch.setattr(benchmark, "DeterministicCollective", Collective)
    monkeypatch.setattr(benchmark, "_paired_samples_us", measure)
    monkeypatch.setattr(benchmark, "_gather_samples_us", gather_samples)
    report_path = tmp_path / "nested" / "timings.json"
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
            "--output",
            str(report_path),
        ]
    )

    for name in outputs:
        assert len(outputs[name]) == len(nccl_outputs[name]) == 3
        assert all(out is outputs[name][0] for out in outputs[name])
        assert all(out is nccl_outputs[name][0] for out in nccl_outputs[name])
        assert outputs[name][0] is not nccl_outputs[name][0]
        torch.testing.assert_close(outputs[name][0], nccl_outputs[name][0])
    assert all(torch.equal(value, inputs[0]) for value in inputs)

    report = json.loads(report_path.read_text())
    assert report["world_size"] == 2
    assert [row["operation"] for row in report["rows"]] == ["reduce_scatter", "all_reduce"]
    assert (report["warmup"], report["iterations"]) == (1, 2)
    assert [row["input_bytes_per_rank"] for row in report["rows"]] == [64, 32]
    for row in report["rows"]:
        assert row["deterministic_samples_us"] == [[7.0, 19.0], [20.0, 10.0]]
        assert row["nccl_samples_us"] == [[19.0, 17.0], [21.0, 13.0]]
        assert row["deterministic_us"] == 15.0
        assert row["nccl_us"] == 18.0


@pytest.mark.parametrize(
    "argv",
    [["--iterations", "0"], ["--size-bytes", "0"], ["--size-bytes", "33"]],
)
def test_parse_args_rejects_invalid_workloads(argv):
    """Reject workloads that would produce no samples or truncated tensors."""
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv)
