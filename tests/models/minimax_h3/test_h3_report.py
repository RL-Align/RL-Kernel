# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Optional checkpoint handling preserves standalone operator measurements, and
comparative timing balances backend order and preserves its raw evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rl_engine.validation.models import h3_chain, h3_report
from rl_engine.validation.models.h3_cases import h3_packed_layout
from rl_engine.validation.models.h3_provider import (
    provider_adaln_row_gather,
    provider_norm_modulate,
)
from rl_engine.validation.models.h3_weights import WEIGHTS_ENV


@pytest.fixture
def small_cpu_report(monkeypatch):
    """Exercise actual report/autograd logic with small CPU tensors and provider ops."""

    def dimension(size):
        return {5376: 4, 6 * 5376: 24, 257: 160, 777: 160, 4097: 160, 32768: 160}.get(size, size)

    class CpuTorch:
        nn = SimpleNamespace(
            functional=SimpleNamespace(
                rms_norm=lambda x, _shape, weight, eps: F.rms_norm(x, (x.shape[-1],), weight, eps)
            )
        )

        def __getattr__(self, name):
            return getattr(torch, name)

        def Generator(self, device):
            return torch.Generator(device="cpu")

        def randn(self, *shape, **kwargs):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                shape = shape[0]
            kwargs["device"] = "cpu"
            return torch.randn(*(dimension(size) for size in shape), **kwargs)

    class Gather:
        def __call__(self, rows, ti, tags):
            return provider_adaln_row_gather(rows.chunk(6, dim=-1), ti, tags)

        forward_fp32 = __call__

    class Norm:
        def __call__(self, x, weight):
            return F.rms_norm(x, (x.shape[-1],), weight, 1e-5)

        forward_modulated = staticmethod(provider_norm_modulate)

    gather = Gather()
    ops = {"adaln_row_gather": gather, "h3_rmsnorm": Norm()}
    registry = SimpleNamespace(
        get_op=lambda name, device: ops[name],
        _priority_map={"cpu": {"adaln_row_gather": ["gather"]}},
        _get_or_create_backend=lambda _backend: gather,
    )
    monkeypatch.setattr(h3_report, "torch", CpuTorch())
    monkeypatch.setattr(
        h3_report,
        "h3_packed_layout",
        lambda seq, num, seed: h3_packed_layout(dimension(seq), num, seed=seed, device="cpu"),
    )

    def norm_inputs(seq, seed=0):
        generator = torch.Generator().manual_seed(seed)
        weight = h3_report.h3_params(["transformer_blocks.0.norm1.weight"], [(5376,)])[0]
        shift, scale = [torch.randn(9, 4, generator=generator).bfloat16() for _ in range(2)]
        x = torch.randn(1, dimension(seq), 4, generator=generator).bfloat16()
        ti, tags = h3_packed_layout(dimension(seq), 3, seed=seed, device="cpu")
        return x, weight.bfloat16(), shift, scale, ti * 3 + tags

    monkeypatch.setattr(h3_report, "_norm_inputs", norm_inputs)
    return registry


def test_gather_without_weights_preserves_op_results(small_cpu_report, monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)

    def unexpected_load(*args, **kwargs):
        pytest.fail("an unset checkpoint must not be loaded")

    monkeypatch.setattr(h3_report, "load_h3_conditioning_weights", unexpected_load)
    result = h3_report._gather_accuracy(small_cpu_report)
    assert all(result["forward_bitwise_vs_index_select"].values())
    assert set(result["op_backward"]) == {"cuda", "provider"}
    assert all(row["repeat_bitwise_equal"] for row in result["op_backward"].values())
    assert result["chain_backward"] == []
    assert WEIGHTS_ENV in result["chain_backward_skipped"]


@pytest.mark.parametrize("configured", [False, True])
def test_norm_reports_weight_source(small_cpu_report, monkeypatch, tmp_path, configured):
    if configured:
        monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    calls = []

    def load_weights(device, names):
        assert configured, "an unset checkpoint must not be loaded"
        calls.append(names)
        return {name: torch.ones(4, dtype=torch.bfloat16) for name in names}

    monkeypatch.setattr(h3_report, "load_h3_conditioning_weights", load_weights)
    result = h3_report._norm_accuracy(small_cpu_report)
    assert result["weight_source"] == ("pinned_checkpoint" if configured else "synthetic")
    assert len(result["plain_bitwise_vs_nn_rmsnorm"]) == 4
    assert all(result["plain_bitwise_vs_nn_rmsnorm"].values())
    assert result["modulated_bitwise_vs_diffusers"]
    assert result["rows_batch_invariant"]
    assert set(result["backward"]) == {"cuda", "provider"}
    assert bool(calls) is configured


def test_gather_with_weights_runs_chain(small_cpu_report, monkeypatch, tmp_path):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    weights = {"weight": torch.ones(1)}
    monkeypatch.setattr(h3_report, "load_h3_conditioning_weights", lambda device: weights)

    def run_chain(registry, loaded, **case):
        assert registry is small_cpu_report
        assert loaded is weights
        return case

    monkeypatch.setattr(h3_chain, "run_backward_case", run_chain)
    result = h3_report._gather_accuracy(small_cpu_report)
    assert len(result["chain_backward"]) == 5
    assert result["chain_backward"][-1] == {"num_timesteps": 4, "seq_len": 32768}
    assert "chain_backward_skipped" not in result


@pytest.mark.parametrize("accuracy", [h3_report._gather_accuracy, h3_report._norm_accuracy])
def test_configured_missing_weights_raise(small_cpu_report, monkeypatch, tmp_path, accuracy):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="missing; run tools/weights/prepare_h3_weights.py"):
        accuracy(small_cpu_report)


def test_plot_gather_without_chain(small_cpu_report, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    script = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "validation"
        / "models"
        / "plot_h3_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("plot_h3_evidence_test", script)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    accuracy = h3_report._gather_accuracy(small_cpu_report)
    report = {
        "environment": {"gpu": "CPU fixture", "torch": torch.__version__, "cuda": "none"},
        "rl_kernel_commit": "testcommit",
        "model_revision": "pinned_revision",
        "accuracy": accuracy,
        "perf": [
            {
                "case": "S=160",
                "candidate_us": 1,
                "provider_us": 2,
                "candidate_backward_us": 3,
                "provider_backward_us": 4,
            }
        ],
    }
    figure = plot.plot_gather(report)
    output = tmp_path / "figure.png"
    figure.savefig(output)
    assert output.stat().st_size > 0
    assert figure.axes[1].texts[0].get_text() == accuracy["chain_backward_skipped"]
    plot.plt.close(figure)


def test_measure_interleaves_backends_and_records_samples(monkeypatch):
    """Balance candidate/provider timing order and preserve samples and derived bandwidth."""

    elapsed_us = 0.0
    calls = []

    class Event:
        def __init__(self, *, enable_timing):
            """Require timing-enabled events in the deterministic timing stub."""

            assert enable_timing

        def record(self):
            """Capture the synthetic elapsed clock at this event."""

            self.timestamp = elapsed_us

        def synchronize(self):
            """Leave synchronization inert because the timing stub has no pending GPU work."""

            pass

        def elapsed_time(self, end):
            """Return elapsed synthetic event time in CUDA milliseconds."""

            return (end.timestamp - self.timestamp) / 1e3

    def candidate():
        """Record a candidate call and advance the synthetic clock by two microseconds."""

        nonlocal elapsed_us
        calls.append("candidate")
        elapsed_us += 2.0

    def provider():
        """Record a provider call and advance the synthetic clock by four microseconds."""

        nonlocal elapsed_us
        calls.append("provider")
        elapsed_us += 4.0

    monkeypatch.setattr(h3_report.torch.cuda, "Event", Event)
    monkeypatch.setattr(h3_report.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(h3_report, "peak_mib", lambda fn, **kwargs: 0.0)
    report = h3_report.measure(
        {"bytes": 1000, "candidate": candidate, "provider": provider}, warmup=1, iters=4
    )
    assert calls == [
        "candidate",
        "provider",  # warmup
        "candidate",
        "provider",
        "provider",
        "candidate",
        "candidate",
        "provider",
        "provider",
        "candidate",
    ]
    assert report["timing_order"] == [
        ["candidate", "provider"],
        ["provider", "candidate"],
        ["candidate", "provider"],
        ["provider", "candidate"],
    ]
    assert report["timing_samples_us"] == {"candidate": [2.0] * 4, "provider": [4.0] * 4}
    assert report["candidate_us"] == pytest.approx(2.0)
    assert report["provider_us"] == pytest.approx(4.0)
    assert report["candidate_gbps"] == pytest.approx(0.5)
    assert report["provider_gbps"] == pytest.approx(0.25)
