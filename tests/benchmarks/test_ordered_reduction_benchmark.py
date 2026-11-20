# SPDX-License-Identifier: Apache-2.0
"""Benchmark planning, validation, timeout recovery and GPU smoke coverage."""

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks.backends.shared.triton.benchmark_ordered_reduction import (
    MODULE,
    accuracy,
    check_replay,
    main,
    markdown,
    parser,
    plan,
    provenance,
    run_isolated,
    summarize,
)
from rl_engine.contracts.numerical import load_contract, resolve_tolerance


def test_dry_run_needs_no_triton_or_gpu(monkeypatch, capsys):
    def unexpected_gpu_query():
        pytest.fail("dry run must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_gpu_query)
    assert main(["--dry-run", "--shapes", "1x1025", "--dtypes", "bf16"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["case_count"] == 1
    assert result["measurement_count"] == 10


def test_gluon_plan_is_opt_in_and_keeps_the_same_arithmetic(monkeypatch):
    """The dry run stays usable without installing or initializing either frontend."""

    def unexpected_gpu_query(*args, **kwargs):
        pytest.fail("planning must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_gpu_query)
    ordinary = plan(parser().parse_args(["--shapes", "128x4105", "--dtypes", "bf16"]))
    assert not any(
        p["family"].startswith("ordered_gluon") for p in ordinary["cases"][0]["providers"]
    )
    candidate = plan(
        parser().parse_args(["--shapes", "128x4105", "--dtypes", "bf16", "--include-gluon"])
    )["cases"][0]
    assert candidate["tree"] == ordinary["cases"][0]["tree"]
    schedules = [p for p in candidate["providers"] if p["family"].startswith("ordered_gluon")]
    assert {p["family"] for p in schedules} == {"ordered_gluon", "ordered_gluon_parallel"}
    assert len(schedules) == 2


@pytest.mark.parametrize("warp_size,hip", [(32, None), (64, "test-rocm")])
@pytest.mark.parametrize("family", ["ordered_gluon", "ordered_gluon_parallel"])
def test_gluon_launchers_use_input_target_width_on_cuda_and_rocm(
    monkeypatch, warp_size, hip, family
):
    """Exercise host wiring without a GPU; these doubles do not validate GPU arithmetic."""
    import rl_engine.backends.shared.triton as shared

    launches = []
    active_devices = []

    class Input:
        device = torch.device("cuda:1")
        shape = (2, 1025)

        def __getitem__(self, key):
            return self

        def stride(self, axis):
            return self.shape[1]

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **options):
                launches.append((self.name, args[-1], options))

            return launch

    @contextmanager
    def device_scope(device):
        assert device == Input.device
        active_devices.append(device)
        try:
            yield
        finally:
            active_devices.pop()

    def current_target():
        assert active_devices == [Input.device]
        return SimpleNamespace(warp_size=warp_size)

    triton = ModuleType("triton")
    language = ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    triton.jit = lambda function: function
    triton.cdiv = lambda size, block: (size + block - 1) // block
    triton.runtime = SimpleNamespace(
        driver=SimpleNamespace(active=SimpleNamespace(get_current_target=current_target))
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setattr(shared, "reduction", ModuleType("reduction"), raising=False)
    kernels = ModuleType("ordered_reduction_gluon_kernels")
    for name in ("row_sum", "partial_sum", "merge_sum"):
        setattr(kernels, name, Kernel(name))
    monkeypatch.setitem(
        sys.modules, "benchmarks.backends.shared.triton.ordered_reduction_gluon_kernels", kernels
    )
    monkeypatch.setattr(torch.cuda, "device", device_scope)
    monkeypatch.setattr(torch.version, "hip", hip)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda shape, **kwargs: empty(shape, dtype=kwargs["dtype"]))
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/backends/shared/triton/ordered_reduction_kernels.py"
    )
    spec = spec_from_file_location("_ordered_launch_probe", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = dict(
        family=family, warps=4, merge_warps=2, tiles_per_batch=32, elements_per_thread=8
    )
    call, scratch = module.make_call(
        Input(), object(), 1025, provider, dict(block_size=1024, max_group_size=8)
    )
    call()
    assert [name for name, _, _ in launches] == (
        ["row_sum"] if family == "ordered_gluon" else ["partial_sum", "merge_sum"]
    )
    assert scratch == (0 if family == "ordered_gluon" else 16)
    for name, width, options in launches:
        assert width == warp_size
        assert options["num_warps"] == (2 if name == "merge_sum" else 4)
        assert options["enable_fp_fusion"] is False
        assert options.get("allow_flush_denorm") is (False if hip else None)


def test_comparison_has_one_config_per_schedule_and_bounds_full_row_compilation():
    args = parser().parse_args(
        [
            "--shapes",
            "1x1024",
            "1x262144",
            "--dtypes",
            "fp32",
            "--warps",
            "1",
            "--merge-warps",
            "2",
            "--pipeline-stages",
            "3",
            "--padding",
            "37",
            "--include-gluon",
            "--gluon-elements-per-thread",
            "4",
            "--gluon-tiles-per-batch",
            "8",
        ]
    )
    report = plan(args)
    for case in report["cases"]:
        assert case["tree"] == dict(block_size=1024, max_group_size=8)
        assert case["padding"] == 37
        schedules = case["providers"]
        assert len({p["family"] for p in schedules}) == len(schedules)
        assert all(p["warps"] == 1 for p in schedules if p["family"] != "torch")
        assert all(p["merge_warps"] == 2 for p in schedules if p["family"].endswith("parallel"))
        assert next(p for p in schedules if p["family"] == "ordered_serial")["pipeline"] == 3
        gluon = next(p for p in schedules if p["family"] == "ordered_gluon")
        assert gluon["elements_per_thread"] == 4
        assert gluon["tiles_per_batch"] == 8
        assert (
            next(p for p in schedules if p["family"] == "ordered_gluon_parallel")[
                "elements_per_thread"
            ]
            == 4
        )
    assert any(p["family"] == "tl_row" for p in report["cases"][0]["providers"])
    assert not any(p["family"] == "tl_row" for p in report["cases"][1]["providers"])


@pytest.mark.parametrize(
    "option,value",
    [
        ("--shapes", "0x32"),
        ("--warps", "3"),
        ("--repeat", "0"),
        ("--case-timeout", "0"),
        ("--merge-warps", "0"),
        ("--gluon-elements-per-thread", "3"),
        ("--gluon-tiles-per-batch", "128"),
    ],
)
def test_invalid_workloads_are_rejected(option, value):
    with pytest.raises(SystemExit):
        parser().parse_args([option, value])


def test_accuracy_uses_the_shared_contract_and_rejects_nonfinite_outputs():
    tolerance = resolve_tolerance(
        load_contract(), judgment="forward_accuracy", op_class="reduction", dtype="float32"
    )
    gold = torch.tensor([1.0, -1.0], dtype=torch.float64)
    assert accuracy(gold.float(), gold, tolerance)["passed"]
    assert not accuracy(torch.tensor([float("nan"), -1]), gold, tolerance)["passed"]
    assert accuracy(torch.tensor([1.1, -1]), gold, tolerance)["failures"] == 1


def test_round_aggregation_keeps_raw_samples_and_exposes_variation():
    result = summarize([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert result["median_ms"] == 3
    assert result["round_spread"] == 2
    assert result["samples_ms"] == [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
    for samples in ([], [[0.0]], [[float("nan")]]):
        with pytest.raises(ValueError):
            summarize(samples)


def test_replay_check_rejects_stale_output():
    """Test the failure gate directly; real capture/replay is covered on GPU below."""
    expected = torch.tensor([1.0, -0.0])
    output = expected.clone()
    with pytest.raises(AssertionError, match="graph replay changed output bits"):
        check_replay(lambda: None, output, expected, "no-op")
    check_replay(lambda: output.copy_(expected), output, expected, "valid")
    wrong_sign = torch.tensor([1.0, 0.0])
    with pytest.raises(AssertionError, match="graph replay changed output bits"):
        check_replay(lambda: output.copy_(wrong_sign), output, expected, "wrong-result")


def test_worker_timeout_returns_control_and_preserves_log(tmp_path):
    started = time.monotonic()
    log = tmp_path / "worker.log"
    code, timed_out = run_isolated(
        [sys.executable, "-u", "-c", "import time; print('compile'); time.sleep(30)"],
        log,
        1,
    )
    assert code != 0 and timed_out
    assert time.monotonic() - started < 10
    assert "compile" in log.read_text()


@pytest.mark.parametrize("git_failure", ("missing", "not-checkout", "timeout"))
def test_unavailable_git_keeps_benchmark_running_and_marks_metadata_unknown(
    monkeypatch, tmp_path, git_failure
):
    """A completed worker stays reportable even when source Git metadata is unavailable."""
    from benchmarks.backends.shared.triton import benchmark_ordered_reduction as benchmark

    def unavailable_git(command, **kwargs):
        if git_failure == "missing":
            raise FileNotFoundError("git executable is not installed")
        if git_failure == "not-checkout":
            raise subprocess.CalledProcessError(128, command, stderr="not a git repository")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    def completed_worker(command, log, timeout):
        # Stub only GPU execution; exercise real planning, provenance and report writing.
        spec = json.loads(command[-2])
        Path(command[-1]).write_text(json.dumps(dict(spec=spec, complete=True, results=[])))
        return 0, False

    monkeypatch.setattr(subprocess, "check_output", unavailable_git)
    monkeypatch.setattr(benchmark, "run_isolated", completed_worker)
    folder = tmp_path / "results"
    assert main(["--shapes", "1x1025", "--dtypes", "bf16", "--output-dir", str(folder)]) == 0
    report = json.loads((folder / "results.json").read_text())
    assert report["complete"] and len(report["cases"]) == 1
    source = report["source"]
    assert source["commit"] is None and source["worktree"] is None
    assert set(source["git_errors"]) == {"commit", "worktree"}
    assert all(source["git_errors"].values())
    assert source["source_sha256"]
    assert all(len(digest) == 64 for digest in source["source_sha256"].values())
    rendered = (folder / "report.md").read_text()
    assert "Source commit: `unavailable`; dirty worktree: unknown." in rendered
    assert "dirty worktree: False" not in rendered
    assert "Git metadata unavailable (commit)" in rendered


@pytest.mark.parametrize("worktree", ("", " M measured.py\n", None))
def test_provenance_retains_available_git_fields(monkeypatch, worktree):
    """A failed status query must not erase a known revision or imply a clean worktree."""

    def git_output(command, **kwargs):
        if command[1:] == ["rev-parse", "HEAD"]:
            return "measured-revision\n"
        if worktree is None:
            raise subprocess.CalledProcessError(128, command)
        return worktree

    monkeypatch.setattr(subprocess, "check_output", git_output)
    source = provenance()
    assert source["commit"] == "measured-revision"
    assert source["worktree"] == worktree
    assert set(source["git_errors"]) == ({"worktree"} if worktree is None else set())
    rendered = markdown(dict(complete=True, cases=[], source=source))
    expected_dirty = "unknown" if worktree is None else str(bool(worktree))
    assert f"dirty worktree: {expected_dirty}." in rendered


def test_markdown_compares_matching_timing_modes_and_hides_incomplete_results():
    rows = []
    for timing, factor in (("eager", 10), ("graph", 1)):
        for family, ms in (("torch", 8), ("tl_row", 4), ("tl_parallel", 2), ("ordered_serial", 1)):
            rows.append(
                dict(
                    provider=dict(family=family, name=family),
                    timing=timing,
                    median_ms=ms * factor,
                    effective_gb_s=100,
                    scratch_bytes=0,
                    extra_peak_bytes=0,
                    round_spread=1,
                )
            )
    case = dict(spec=dict(dtype="fp32", rows=1, length=1024), complete=True, results=rows)
    report = dict(complete=True, cases=[case], source=dict(commit="test", worktree=""))
    text = markdown(report)
    assert "| 8.00x | 2.00x |" in text
    assert "ordered_serial | eager | 10.000000 | 8.00x | 2.00x" in text
    assert "ordered_serial | graph | 1.000000 | 8.00x | 2.00x" in text
    case.update(complete=False, error="accuracy failed")
    text = markdown(report)
    assert "accuracy failed" in text
    assert "8.00x" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm GPU is required")
@pytest.mark.parametrize("timings,include_gluon", ((["graph"], False), (["eager", "graph"], True)))
def test_gpu_benchmark_smoke_checks_all_providers_and_replays(tmp_path, timings, include_gluon):
    if include_gluon:
        pytest.importorskip("triton.experimental.gluon")
    args = parser().parse_args(
        [
            "--shapes",
            "3x1025",
            "--dtypes",
            "bf16",
            "--padding",
            "17",
            "--rounds",
            "1",
            "--repeat",
            "2",
            "--warmup",
            "1",
            "--graph-calls",
            "2",
            "--pipeline-stages",
            "0",
            *(["--include-gluon"] if include_gluon else []),
            "--timings",
            *timings,
        ]
    )
    spec = plan(args)["cases"][0]
    path, log = tmp_path / "result.json", tmp_path / "log.txt"
    code, timed_out = run_isolated(
        [sys.executable, "-m", MODULE, "--worker", json.dumps(spec), str(path)],
        log,
        90,
    )
    assert not timed_out and code == 0, log.read_text()
    result = json.loads(path.read_text())
    assert result["complete"]
    assert len(result["results"]) == len(spec["providers"]) * len(timings)
    expected_keys = {(p["name"], timing) for p in spec["providers"] for timing in timings}
    assert len(result["round_orders"]) == spec["rounds"]
    assert all({tuple(key) for key in order} == expected_keys for order in result["round_orders"])
    assert all(row["accuracy"]["passed"] and row["median_ms"] > 0 for row in result["results"])
    for row in result["results"]:
        assert len(row["samples_ms"]) == spec["rounds"]
        assert all(len(samples) == spec["repeat"] for samples in row["samples_ms"])
        assert "rounds" not in row  # Raw samples have one report field, not two copies.
