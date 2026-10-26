# SPDX-License-Identifier: Apache-2.0
"""Separate eager latency, graph device time, and cross-process autotune cache reuse."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _summary(values: list[float]) -> dict[str, float]:
    """Return the median, min and max of ``values``."""
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1]}


def worker(directory: str, shape: tuple[int, int], scan_candidates: bool) -> dict:
    """Time legacy SiLU tuning in this process: first call, eager wall and graph device time."""
    import torch

    from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp, _silu_fwd_kernel
    from rl_engine.backends.shared.triton.autotuning import TuneState, tuning_scope

    os.environ["RL_KERNEL_MODE"] = "auto"
    os.environ["RL_KERNEL_AUTOTUNE"] = "1"
    os.environ["RL_KERNEL_AUTOTUNE_CACHE_DIR"] = directory
    torch.manual_seed(2026)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    op = TritonSiLUOp()
    state = TuneState(True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with tuning_scope(state):
        tuned_output = op(x)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000
    with tuning_scope(TuneState(False)):
        fixed_output = op(x)
    if not torch.equal(fixed_output.view(torch.uint8), tuned_output.view(torch.uint8)):
        raise RuntimeError("Selected configuration differs from the fixed output")

    # A CUDA event around one Python call includes GPU idle time while Python
    # prepares and submits the kernel. Capture many launches to remove that gap.
    repeats = 64
    graphs = {}
    for enabled in (False, True):
        with tuning_scope(TuneState(enabled)):
            for _ in range(5):
                op(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    output = op(x)
            graphs[f"operator_{'tuned' if enabled else 'fixed'}"] = graph
    del output

    if not state.records:
        raise RuntimeError(
            "the tuned call recorded no selection; the device identity is unavailable"
        )
    configs = {"kernel_fixed": {}, "kernel_selected": state.records[0]["config"]}
    if scan_candidates:
        for warps in (4, 8):
            for stages in (1, 2):
                configs[f"kernel_w{warps}_s{stages}"] = {
                    "num_warps": warps,
                    "num_stages": stages,
                }
        for block in (256, 512, 2048):
            configs[f"kernel_block{block}"] = {"BLOCK": block, "num_warps": 4}
    buffer = torch.empty_like(x)
    for name, config in configs.items():
        options = {"BLOCK": 1024, **config}
        grid = ((x.numel() + options["BLOCK"] - 1) // options["BLOCK"],)
        runner = _silu_fwd_kernel.kernel[grid]
        runner(x, buffer, x.numel(), **options)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                runner(x, buffer, x.numel(), **options)
        graphs[name] = graph
    for graph in graphs.values():
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    device_times = {name: [] for name in graphs}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for sample in range(11):
        names = list(graphs)
        if sample % 2:
            names.reverse()
        for name in names:
            start_event.record()
            graphs[name].replay()
            end_event.record()
            end_event.synchronize()
            device_times[name].append(start_event.elapsed_time(end_event) * 1000 / repeats)

    wall_times = {"fixed": [], "tuned": []}
    for sample in range(11):
        order = (False, True) if sample % 2 == 0 else (True, False)
        for enabled in order:
            with tuning_scope(TuneState(enabled)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    op(x)
                torch.cuda.synchronize()
                elapsed_us = (time.perf_counter() - start) * 1e6 / 100
            wall_times["tuned" if enabled else "fixed"].append(elapsed_us)

    return {
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "shape": list(shape),
        "dtype": str(x.dtype),
        "first_call_ms": first_ms,
        "eager_wall_us": {name: _summary(times) for name, times in wall_times.items()},
        "graph_device_us": {name: _summary(times) for name, times in device_times.items()},
        "raw_kernel_configs": configs,
        "launches": state.records,
        "methodology": {
            "eager": "11 alternating batches of 100 calls, wall time through final synchronization",
            "graph": "11 alternating samples, 64 captured launches per replay, CUDA event time",
            "cache": "warm repeated input; raw kernels reuse a preallocated output",
            "first_call": "includes tuning/cache lookup, Triton initialization and compilation",
            "speedup": "Compare eager_wall_us for eager users and graph_device_us for graph users",
        },
    }


def main():
    """Parse arguments and run an experiment worker, the operator matrix or the SiLU benchmark."""
    # Keep worker stdout a JSON-only protocol, including core backend startup logs.
    os.environ["RL_KERNEL_LOG_STREAM"] = "stderr"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--shape", type=int, nargs=2, default=(1024, 4096), metavar=("ROWS", "COLS")
    )
    parser.add_argument("--scan-candidates", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--operator",
        choices=[
            "all",
            "silu",
            "swiglu",
            "rope",
            "embedding",
            "rmsnorm",
            "ratio_kl",
            "grpo",
            "gemm",
            "attention",
            "logp",
        ],
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--phase", choices=["forward", "backward", "forward_backward", "all"], default="all"
    )
    parser.add_argument("--preset", choices=["small", "throughput", "all"], default="all")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment-worker", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-mode", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.experiment_worker:
        from benchmarks.tuning.autotuning_experiment import experiment_worker

        print(
            json.dumps(
                experiment_worker(
                    json.loads(args.experiment_worker),
                    str(args.cache_dir),
                    args.experiment_mode,
                    validate_only=args.experiment_validate_only,
                )
            )
        )
        return
    if args.operator:
        from benchmarks.tuning.autotuning_experiment import run_experiments

        if args.trials < 1:
            parser.error("--trials must be positive")
        if tuple(args.shape) != (1024, 4096) or args.scan_candidates:
            parser.error(
                "--shape/--scan-candidates belong to the legacy SiLU benchmark; use --preset"
            )
        if args.cache_dir:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(prefix="rlkernel-matrix-", dir=args.cache_dir)
        result = run_experiments(args, directory)
        if args.output:
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "output": str(args.output),
                        "cases": len(result["aggregates"]),
                    }
                )
            )
        else:
            print(json.dumps(result, indent=2))
        return
    if any(dimension <= 0 for dimension in args.shape):
        parser.error("shape dimensions must be positive")
    if args.worker:
        print(json.dumps(worker(str(args.cache_dir), tuple(args.shape), args.scan_candidates)))
        return
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix="rlkernel-autotune-", dir=args.cache_dir)
    command = [
        sys.executable,
        "-m",
        "benchmarks.tuning.benchmark_autotuning",
        "--worker",
        "--cache-dir",
        directory,
        "--shape",
        *(str(dimension) for dimension in args.shape),
    ]
    if args.scan_candidates:
        command.append("--scan-candidates")
    cold = json.loads(subprocess.check_output(command, text=True))
    restarted = json.loads(subprocess.check_output(command, text=True))
    result = {"cache_directory": directory, "cold_process": cold, "restarted_process": restarted}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
