# SPDX-License-Identifier: Apache-2.0
"""Multi-operator cold/hot/restart experiments; invoked by benchmark_autotuning."""

import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.tuning.autotuning_cases import (
    OPERATORS,
    fixed_case,
    make_case,
    preset_shape,
    supported,
)


def summary(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def ensure_idle():
    """Reject contaminated formal measurements; never terminate other GPU jobs."""
    import torch

    if torch.version.hip:
        # PyTorch uses AMD SMI here; require a working process listing rather than
        # silently accepting a device whose occupancy could not be established.
        listing = torch.cuda.list_gpu_processes(0)
        processes = re.findall(r"^process\s+(\d+)\s+uses", listing, re.MULTILINE)
        if not processes and "no processes are running" not in listing.lower():
            raise RuntimeError(f"Cannot establish ROCm occupancy: {listing}")
        if any(pid != str(os.getpid()) for pid in processes):
            raise RuntimeError(f"GPU is occupied; postpone formal experiment: {listing}")
        return
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    listing = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    others = [pid.strip() for pid in listing.splitlines() if pid.strip() != str(os.getpid())]
    if others:
        raise RuntimeError(f"GPU is occupied; postpone formal experiment (PIDs: {others})")


def experiment_case(spec, directory, mode, *, validate_only=False):
    import torch

    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(caller)
    try:
        with torch.cuda.stream(stream):
            return _experiment_case(spec, directory, mode, validate_only=validate_only)
    finally:
        caller.wait_stream(stream)


def _experiment_case(spec, directory, mode, *, validate_only=False):
    import torch

    from rl_engine.backends.shared.triton import autotuning as adapter
    from rl_engine.platforms.autotuner import PersistentAutotuner, TuneConfig

    results = []
    torch.manual_seed(2026)
    dtype = getattr(torch, spec["dtype"])
    rows, width = preset_shape(spec["operator"], spec["preset"])
    case = make_case(spec["operator"], dtype, rows=rows, width=width)
    fixed = fixed_case(case, spec["operator"], dtype)
    phase = spec["phase"]
    selected_records = []
    profile_ms = []
    tuner = PersistentAutotuner(Path(directory) / spec["id"])
    adapter._TUNER = tuner
    original_select = tuner.select
    original_profile = adapter._profile_candidates

    def select(signature, *args, **kwargs):
        selected = original_select(signature, *args, **kwargs)
        selected_records.append(
            {
                "kernel": signature["kernel"],
                "phase": signature["phase"],
                "config": selected.config.to_dict(),
                "source": selected.source,
            }
        )
        return selected

    tuner.select = select

    def timed_profile(*args, **kwargs):
        start = time.perf_counter()
        result = original_profile(*args, **kwargs)
        profile_ms.append((time.perf_counter() - start) * 1000)
        return result

    adapter._profile_candidates = timed_profile
    # Backward-only builds its graph before measurement; no forward tuning.
    os.environ["RL_KERNEL_AUTOTUNE"] = "0"
    outputs = ()
    backward_graphs = {}
    if phase == "backward":
        for enabled in (False, True):
            with adapter.tuning_scope(adapter.TuneState(False)):
                outputs = case.forward()
            # Public operators may add view nodes around the custom Function.
            pending = [out.grad_fn for out in outputs if out.grad_fn is not None]
            visited = set()
            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                if hasattr(node, "autotune_state"):
                    node.autotune_state = adapter.TuneState(enabled, core=True)
                pending.extend(child for child, _ in node.next_functions if child is not None)
            backward_graphs[enabled] = outputs
    grad_outputs = tuple(torch.ones_like(x) for x in outputs) if outputs else None

    def run(enabled):
        nonlocal grad_outputs
        if phase == "backward":
            return torch.autograd.grad(
                backward_graphs[enabled], case.variables, grad_outputs, retain_graph=True
            )
        value = (case if enabled else fixed).forward()
        if phase == "forward_backward":
            if grad_outputs is None:
                grad_outputs = tuple(torch.ones_like(x) for x in value)
            return torch.autograd.grad(value, case.variables, grad_outputs)
        return value

    def call(enabled):
        with adapter.tuning_scope(adapter.TuneState(enabled, core=True)):
            return run(enabled)

    os.environ["RL_KERNEL_AUTOTUNE"] = "1"
    if mode == "precompiled":

        def compile_only(candidates, args, outputs, inout, launch):
            from triton.runtime.errors import OutOfResources

            times = {}
            for config in candidates:
                private = list(args)
                for i in outputs:
                    private[i] = args[i].clone()
                try:
                    launch(config, private)
                    times[config] = 1.0 if config == TuneConfig() else 2.0
                except OutOfResources:
                    if config == TuneConfig():
                        raise
                    times[config] = None
            return times

        adapter._profile_candidates = compile_only
        # Precompile into a separate nonpersistent tuner; no real selection survives.
        temporary = PersistentAutotuner()
        temporary_select = temporary.select
        temporary.select = lambda *a, **kw: temporary_select(*a, **{**kw, "persistent": False})
        adapter._TUNER = temporary
        call(True)
        torch.cuda.synchronize()
        adapter._TUNER = tuner
        adapter._profile_candidates = timed_profile

    torch.cuda.synchronize()
    start = time.perf_counter()
    selected_value = call(True)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000
    first_records = list(selected_records)
    # Check selected phase output, including gradients, against the fixed implementation.
    os.environ["RL_KERNEL_AUTOTUNE"] = "0"
    fixed_value = call(False)
    for a, b in zip(selected_value, fixed_value, strict=True):
        if not adapter._bitwise_equal(a, b):
            raise AssertionError(f"Selected output mismatch: {spec['id']}")
    before_hot = len(selected_records)
    os.environ["RL_KERNEL_AUTOTUNE"] = "1"
    call(True)
    if len(selected_records) != before_hot:
        raise AssertionError("Hot call entered selector")
    if mode == "restart" and any(x["source"] != "disk" for x in first_records):
        raise AssertionError(f"Restart missed disk cache: {first_records}")
    if not first_records:
        raise AssertionError(f"No tuned launches: {spec}")

    graphs = {}
    graph_values = {}
    for enabled in (False, True):
        os.environ["RL_KERNEL_AUTOTUNE"] = "1" if enabled else "0"
        for _ in range(3):
            call(enabled)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            for _ in range(32):
                graph_value = call(enabled)
        graphs[enabled] = graph
        graph_values[enabled] = graph_value
    del graph_value
    if validate_only:
        for enabled, graph in graphs.items():
            graph.replay()
            for actual, expected in zip(graph_values[enabled], fixed_value, strict=True):
                if not adapter._bitwise_equal(actual, expected):
                    raise AssertionError(f"Benchmark graph output mismatch: {spec['id']}")
        adapter._profile_candidates = original_profile
        return {
            "status": "validated",
            "launches": first_records,
            "hot_selection_calls": len(selected_records) - before_hot,
        }
    times = {kind: {False: [], True: []} for kind in ("eager", "graph")}
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for sample in range(11):
        for enabled in (False, True) if sample % 2 == 0 else (True, False):
            os.environ["RL_KERNEL_AUTOTUNE"] = "1" if enabled else "0"
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                call(enabled)
            torch.cuda.synchronize()
            times["eager"][enabled].append((time.perf_counter() - start) * 1e6 / 50)
            event_start.record()
            graphs[enabled].replay()
            event_end.record()
            event_end.synchronize()
            times["graph"][enabled].append(event_start.elapsed_time(event_end) * 1000 / 32)
    adapter._profile_candidates = original_profile
    results.append(
        {
            **spec,
            "shape_parameters": {"rows": rows, "width": width},
            "first_call_ms": first_ms,
            "profile_ms": sum(profile_ms),
            "launches": first_records,
            "hot_selection_calls": len(selected_records) - before_hot,
            "timings_us": {
                kind: {"fixed": summary(pair[False]), "tuned": summary(pair[True])}
                for kind, pair in times.items()
            },
        }
    )
    return results[0]


def experiment_worker(specs, directory, mode, *, validate_only=False):
    import torch

    from rl_engine.backends.shared.triton import autotuning as adapter

    os.environ["RL_KERNEL_MODE"] = "auto"
    os.environ["RL_KERNEL_AUTOTUNE"] = "1"
    os.environ["RL_KERNEL_AUTOTUNE_CACHE_DIR"] = directory
    results = []
    for spec in specs:
        if not validate_only:
            ensure_idle()
        original_profile = adapter._profile_candidates
        try:
            results.append(experiment_case(spec, directory, mode, validate_only=validate_only))
            if not validate_only:
                ensure_idle()
        finally:
            adapter._profile_candidates = original_profile
        torch.cuda.empty_cache()
    return {
        "mode": mode,
        "status": "validation_only" if validate_only else "measured",
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "triton": adapter.version("triton"),
        "runtime": torch.version.hip or torch.version.cuda,
        "identity": adapter._device_identity(torch.device("cuda:0"), False),
        "cases": results,
    }


def run_experiments(args, directory):
    import torch

    names = OPERATORS if args.operator == "all" else (args.operator,)
    presets = ("small", "throughput") if args.preset == "all" else (args.preset,)
    specs, unsupported = [], []
    for name in names:
        for preset in presets:
            phases = (
                ("forward", "backward", "forward_backward")
                if args.phase == "all"
                else (args.phase,)
            )
            for phase in phases:
                spec = {"operator": name, "preset": preset, "phase": phase, "dtype": args.dtype}
                spec["id"] = "-".join(spec.values())
                if not supported(name, getattr(torch, args.dtype)) or (
                    phase != "forward"
                    and (
                        name in ("embedding", "grpo")
                        or (name == "gemm" and args.dtype != "bfloat16")
                    )
                ):
                    unsupported.append({**spec, "status": "unsupported"})
                else:
                    specs.append(spec)
    try:
        ensure_idle()
    except RuntimeError as error:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": str(error),
                        "planned_cases": specs,
                        "unsupported": unsupported,
                        "trials_requested": args.trials,
                        "trials": [],
                        "performance_claim": None,
                    },
                    indent=2,
                )
            )
        raise
    trials = []
    for trial in range(args.trials):
        records = {}
        for mode in ("cold", "restart", "precompiled"):
            cache = (
                Path(directory)
                / f"trial-{trial}"
                / ("precompiled" if mode == "precompiled" else "shared")
            )
            command = [
                sys.executable,
                "-m",
                "benchmarks.tuning.benchmark_autotuning",
                "--experiment-worker",
                json.dumps(specs),
                "--cache-dir",
                str(cache),
                "--experiment-mode",
                mode,
            ]
            try:
                records[mode] = json.loads(
                    subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
                )
            except subprocess.CalledProcessError as error:
                reason = error.stderr or str(error)
                if args.output:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.output).write_text(
                        json.dumps(
                            {
                                "status": "blocked" if "GPU is occupied" in reason else "failed",
                                "reason": reason,
                                "trials": trials,
                                "completed_workers": records,
                                "planned_cases": specs,
                                "unsupported": unsupported,
                                "performance_claim": None,
                            },
                            indent=2,
                        )
                    )
                raise
        trials.append(records)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps({"status": "running", "trials": trials}, indent=2)
            )
    aggregates = aggregate_trials(specs, trials)
    result = {
        "status": "complete",
        "trials": trials,
        "aggregates": aggregates,
        "unsupported": unsupported,
        "methodology": {
            "seeds": [2026],
            "trials": args.trials,
            "samples": 11,
            "eager": "50 calls/batch; synchronized wall time including allocation/dispatch",
            "graph": "32 phase calls/replay; CUDA events; independent autograd.grad outputs",
            "precompiled": "precompiled candidates; fresh tuner; first call excludes JIT",
            "cold": "setup excluded; backward-only excludes fixed forward; phase includes JIT",
            "benefit": "confirmed only when restarted process median ranges do not overlap",
            "raw_kernel": "Use legacy --scan-candidates for isolated SiLU kernel timing",
            "baseline": "ordinary: tuning disabled; core: fixed public Triton backend",
        },
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    return result


def aggregate_trials(specs, trials):
    aggregates = []
    for index, spec in enumerate(specs):
        cost = statistics.median(t["precompiled"]["cases"][index]["first_call_ms"] for t in trials)
        entry = {**spec, "precompiled_first_call_ms": cost}
        for kind in ("eager", "graph"):
            fixed = [
                t["restart"]["cases"][index]["timings_us"][kind]["fixed"]["median"] for t in trials
            ]
            tuned = [
                t["restart"]["cases"][index]["timings_us"][kind]["tuned"]["median"] for t in trials
            ]
            savings = statistics.median(fixed) - statistics.median(tuned)
            confirmed = min(fixed) > max(tuned)
            entry[kind] = {
                "fixed_us": summary(fixed),
                "tuned_us": summary(tuned),
                "benefit": "confirmed" if confirmed else "unconfirmed",
                "break_even_calls": (
                    math.ceil(cost * 1000 / savings) if confirmed and savings > 0 else None
                ),
                "amortization": (
                    "finite" if confirmed and savings > 0 else "no confirmed finite break-even"
                ),
            }
        aggregates.append(entry)
    return aggregates
