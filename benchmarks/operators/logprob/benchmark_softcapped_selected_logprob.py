# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare public softcapped logprob Ops and optionally tune forward schedules."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.profiler import GPUTargetInfo, PerformanceProfiler  # noqa: E402
from rl_engine.contracts.numerical import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.contracts.operators.softcapped_selected_logprob import REDUCTION_CONFIG  # noqa: E402
from rl_engine.reference.logprob.softcapped_selected_logprob import (  # noqa: E402
    NativeSoftcappedSelectedLogprobOp,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
DEFAULT_SHAPES = ("1x1025", "1x262144", "16x262144", "128x262144", "1024x262144")
DEFAULT_PIPELINE_STAGES = (1, 2, 3, 4, 5, 6, 8)


@dataclass(frozen=True)
class _Provider:
    """One public Op and an optional benchmark-only pipeline depth override."""

    name: str
    op: object
    pipeline_stages: int | None = None


@contextmanager
def _pipeline_depth(core, stages):
    """Temporarily tune the public pipelined Op outside timed calls.

    This standalone benchmark is single-threaded. Save and restore the table
    and fallback even on failure; clear the selector cache at both boundaries.
    The normal public validation, selector and autograd path remain in use.
    No production rule is changed on disk, and AUTO runs with the original table.
    """
    if stages is None:
        yield
        return
    original_rules = core.ROW_PIPELINE_CONFIGS
    original_default = core.DEFAULT_ROW_PIPELINE_CONFIG
    config = core.RowPipelineConfig(pipeline_stages=stages)
    try:
        core.ROW_PIPELINE_CONFIGS = {}
        core.DEFAULT_ROW_PIPELINE_CONFIG = config
        core.select_softcapped_logprob_pipeline_config.cache_clear()
        yield
    finally:
        core.ROW_PIPELINE_CONFIGS = original_rules
        core.DEFAULT_ROW_PIPELINE_CONFIG = original_default
        core.select_softcapped_logprob_pipeline_config.cache_clear()


def _configured_check_op(core, provider):
    """Apply a provider's temporary config for untimed correctness calls only."""

    def checked(logits, token_ids, *, softcap):
        with _pipeline_depth(core, provider.pipeline_stages):
            return provider.op(logits, token_ids, softcap=softcap)

    return checked


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _shape(value: str) -> tuple[int, int]:
    try:
        rows, vocab = map(int, value.split("x"))
        if rows <= 0 or vocab <= 0:
            raise ValueError
        return rows, vocab
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be positive MxV, e.g. 16x262144") from exc


def _make_workload(op, logits, token_ids, upstream, mode, *, softcap: float):
    """Prepare inputs/graphs outside timing; discard results without accumulating .grad."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(logits, token_ids, softcap=softcap)

        return forward

    leaf = logits.detach().requires_grad_(True)
    if mode == "backward":
        output = op(leaf, token_ids, softcap=softcap)

        def backward():
            torch.autograd.grad(output, leaf, grad_outputs=upstream, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaf, token_ids, softcap=softcap), leaf, grad_outputs=upstream)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(
    reference_op,
    candidate,
    logits,
    token_ids,
    upstream,
    *,
    softcap: float,
    strategy_candidates=(),
    check_saved_statistics=False,
):
    """Check against Native FP32, preserving input quantization and reference gradients.

    Timing still uses the public Native Op at the original input dtype. Only
    the untimed accuracy reference promotes inputs and upstreams to FP32.
    Also require identical output/gradient bytes across all supplied strategies.
    """

    def evaluate(op, values, *, capture_statistics=False):
        leaf = values.detach().requires_grad_(True)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            output = op(leaf, token_ids, softcap=softcap)
            saved = ()
            if capture_statistics:
                _, _, row_max, log_sum_exp_shifted = output.grad_fn.saved_tensors
                saved = (row_max.detach(), log_sum_exp_shifted.detach())
            gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream.float())[0]
        return output.detach(), gradient, saved

    expected = evaluate(reference_op, logits.float())
    actual = evaluate(candidate, logits, capture_statistics=check_saved_statistics)
    errors = {}
    for label, judgment, shape, dtype, reference, result in zip(
        ("output", "gradient"),
        ("forward_accuracy", "gradient_accuracy"),
        (logits.shape[:1], logits.shape),
        (torch.float32, logits.dtype),
        expected[:2],
        actual[:2],
        strict=True,
    ):
        assert result.shape == shape and result.dtype == dtype
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="logprob", dtype=dtype
        )
        torch.testing.assert_close(
            result.float(), reference, atol=tolerance.atol, rtol=tolerance.rtol
        )
        errors[label] = {
            "max_abs_error": (result.float() - reference.float()).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            inferred = candidate(logits, token_ids, softcap=softcap)
        _assert_same_bits(inferred, actual[0], "training/inference output")
    for name, op in strategy_candidates:
        other = evaluate(op, logits, capture_statistics=check_saved_statistics)
        for label, result, gold in zip(("output", "gradient"), other[:2], actual[:2], strict=True):
            _assert_same_bits(result, gold, f"{name} {label}")
        if check_saved_statistics:
            for label, result, gold in zip(
                ("row_max", "log_sum_exp_shifted"), other[2], actual[2], strict=True
            ):
                _assert_same_bits(result, gold, f"{name} {label}")
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                inferred = op(logits, token_ids, softcap=softcap)
            _assert_same_bits(inferred, other[0], f"{name} training/inference output")
    return errors


def _assert_same_bits(actual, expected, label):
    """Compare all bytes without rounding; singleton strided views need a copy."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual = actual.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    expected = expected.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    if not torch.equal(actual, expected):
        raise AssertionError(f"Bitwise mismatch: {label}")


def _case_plan(args):
    stages = list(dict.fromkeys(args.pipeline_stages))
    explicit_names = ["row", "parallel", *[f"row_pipelined_s{stage}" for stage in stages]]
    providers = (
        ["native", "triton", *explicit_names] if args.compare_strategies else ["native", "triton"]
    )
    per_case = sum(
        len(providers) if args.compare_strategies and mode != "backward" else 2
        for mode in args.modes
    )
    return {
        "softcap": args.softcap,
        "dtypes": args.dtypes,
        "shapes": [list(shape) for shape in args.shapes],
        "modes": args.modes,
        "case_count": len(args.dtypes) * len(args.shapes),
        "measurement_count": len(args.dtypes) * len(args.shapes) * len(args.modes),
        "compare_forward_strategies": args.compare_strategies,
        "forward_timing_rounds": args.rounds if args.compare_strategies else 1,
        "pipeline_stages": stages if args.compare_strategies else [],
        "comparison_providers": providers,
        "provider_measurement_count": len(args.dtypes) * len(args.shapes) * per_case,
    }


def _measure(profiler, fn):
    """Use the shared event timer, then measure allocation in a separate call."""
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    # Inputs and any retained backward graph are already part of the baseline.
    # fn returns None, so outputs do not remain live between invocations.
    profiler._sync()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    fn()
    profiler._sync()
    peak = torch.cuda.max_memory_allocated(profiler.device) - baseline
    return {"median_ms": median_ms, "std_ms": std_ms, "peak_extra_mib": peak / 1024**2}


def _measure_strategy_comparison(
    profiler,
    providers,
    logits,
    ids,
    upstream,
    *,
    softcap,
    core=None,
    mode="forward",
    rounds=4,
):
    """Rotate public providers, including AUTO and each explicit pipeline depth.

    Config changes and graph preparation happen outside timing. Warmup fills
    the selector cache after each override. Offsets span the provider list;
    each provider occupies every position when rounds equals provider count.
    Report all round samples and their median; backward-only uses one shared kernel.
    """
    samples = {provider.name: [] for provider in providers}
    for round_index in range(rounds):
        start = (round_index * len(providers) // rounds) % len(providers)
        for position, provider in enumerate(providers[start:] + providers[:start]):
            with _pipeline_depth(core, provider.pipeline_stages):
                fn = _make_workload(provider.op, logits, ids, upstream, mode, softcap=softcap)
                sample = _measure(profiler, fn)
                del fn
            samples[provider.name].append(dict(sample, round=round_index, position=position))
    measurements = {}
    for name, round_samples in samples.items():
        deviations = [sample["std_ms"] for sample in round_samples if sample["std_ms"] is not None]
        measurements[name] = {
            "median_ms": statistics.median(sample["median_ms"] for sample in round_samples),
            "std_ms": max(deviations) if deviations else None,
            "peak_extra_mib": max(sample["peak_extra_mib"] for sample in round_samples),
            "round_spread_ratio": max(sample["median_ms"] for sample in round_samples)
            / min(sample["median_ms"] for sample in round_samples),
            "rounds": round_samples,
        }
    return measurements


def _command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(device, args, triton_version, block_v, target: GPUTargetInfo):
    """Record this benchmark's settings and the profiler's CUDA or ROCm target."""
    props = torch.cuda.get_device_properties(device)
    is_rocm = target.backend == "rocm"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "backend": target.backend,
        "runtime": target.driver_version,
        "gpu": props.name,
        "architecture": target.architecture,
        "compute_capability": None if is_rocm else list(torch.cuda.get_device_capability(device)),
        "gpu_name_driver": _command_output(
            ["rocm-smi", "--showproductname", "--showdriverversion"]
            if is_rocm
            else ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "device": str(device),
        "total_memory_gib": props.total_memory / 1024**3,
        "multiprocessors": props.multi_processor_count,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
        "block_v": block_v,
        "reduction_group_size": REDUCTION_CONFIG.group_size,
        "reduction_section_count": REDUCTION_CONFIG.section_count,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "layout": "contiguous",
        "timer": (
            f"PerformanceProfiler {'HIP' if is_rocm else 'CUDA'} events; "
            "median and sample standard deviation"
        ),
        "operator": "softcapped_selected_logprob",
        "softcap": args.softcap,
        "baseline": (
            "eager NativeSoftcappedSelectedLogprobOp; "
            "FP32 device-tensor division; FP32 log_softmax"
        ),
        "candidate": "TritonSoftcappedSelectedLogprobOp with automatic forward selection",
        "triton_arithmetic": "FP32 whole-row maximum shift; ordered tile sums",
        "cache_policy": "reused contiguous inputs, warm selector cache, no explicit eviction",
        "scope": "Public wrappers, including allocation and autograd dispatch; no graph capture",
        "backward": "Random FP32 upstream gradient; prebuilt retained graph; no forward timing",
        "memory": "Extra peak allocated MiB per call; excludes inputs and any prebuilt graph",
    }


def _markdown(report):
    env = report["environment"]
    lines = [
        "# Softcapped selected logprob benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Status: {'complete' if report['complete'] else 'incomplete'}; "
        f"{len(report['results'])}/{report['case_plan']['measurement_count']} measurements.",
        f"Softcap: {env['softcap']}. "
        f"Warmup: {env['warmup']}; measured repetitions: {env['repeat']}; seed: {env['seed']}.",
        "",
        "Eager NativeSoftcappedSelectedLogprobOp vs TritonSoftcappedSelectedLogprobOp "
        "with automatic strategy selection; no torch.compile or CUDA Graph capture.",
        "The native Op is both the accuracy reference and the timed baseline. It uses "
        "FP32 log_softmax and an FP32 device-tensor divisor; "
        "its creation and transfer are included in forward "
        "and forward+backward timing. Backward reuses the divisor saved in the prebuilt graph.",
        "Untimed accuracy checks promote the same quantized inputs to FP32 and keep "
        "Native reference gradients in FP32. Timing uses the original input dtype. "
        "Independent FP64 mathematical checks are available through the optional "
        "tools.validation.operators.check_softcapped_math diagnostic.",
        "Contiguous inputs are reused without explicit cache eviction. Median accelerator-event "
        "latency includes public wrapper allocation and autograd dispatch. Input generation, "
        "accuracy checks and JIT compilation are outside timing; the selector cache is warm. "
        "Small cases can be dominated by host dispatch gaps.",
        "Forward uses no_grad. Backward reuses a prebuilt graph; forward+backward builds a "
        "fresh graph per call. Extra peak allocation excludes inputs and prebuilt graphs.",
        "Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower. Every "
        "reported case passed output and random-upstream gradient checks before timing. "
        "AUTO outputs also match training/inference bitwise; output and gradient bytes "
        "match all retained forward strategies. Triton uses an FP32 whole-row maximum "
        "shift and saves both the maximum and log of the shifted sum for backward. "
        "JSON also contains sample standard deviations and error bounds.",
        "",
        "| Input | Shape | Triton strategy | Mode | Native ms | Triton ms | Speedup | "
        "Native extra MiB | Triton extra MiB |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        native, triton = row["native"], row["triton"]
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | "
            f"{row['triton_strategy']} | {row['mode']} | {native['median_ms']:.6f} | "
            f"{triton['median_ms']:.6f} | {row['speedup']:.2f}x | "
            f"{native['peak_extra_mib']:.2f} | {triton['peak_extra_mib']:.2f} |"
        )
    comparisons = [row for row in report["results"] if "strategy_comparison" in row]
    if comparisons:
        lines.extend(
            [
                "",
                "## Forward schedule tuning",
                "",
                "Configured rounds rotate PyTorch, AUTO, ROW, PARALLEL and explicit "
                "ROW_PIPELINED depths for forward and, when requested, forward+backward. "
                "Values are medians of round medians; JSON retains every round, its sample "
                "standard deviation and peak allocation. Aggregate std_ms is the maximum "
                "within-round deviation, not a confidence interval. All providers pass "
                "reference output/gradient checks before timing. Saved maximum/log-sum, "
                "outputs and gradients must match AUTO bitwise at every depth; each "
                "candidate also passes training/no_grad/inference_mode checks. Pipeline "
                "overrides and cache resets happen outside timed calls and are restored "
                "before AUTO. Backward-only timing uses the shared automatic Op.",
                "AUTO / best explicit compares public-call latency, including AUTO dispatch; "
                "values above 1 expose overhead or a slower choice. Near ties need repeated "
                "measurements; this check does not automatically rewrite the policy.",
                "",
                "| Input | Shape | Mode | Provider | ms | PyTorch / provider | Round max/min |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in comparisons:
            comparison = row["strategy_comparison"]
            for name, measurement in comparison["providers"].items():
                lines.append(
                    f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
                    f"{name} | {measurement['median_ms']:.6f} | "
                    f"{row['native']['median_ms'] / measurement['median_ms']:.2f}x | "
                    f"{measurement['round_spread_ratio']:.3f}x |"
                )
        lines.extend(
            [
                "",
                "Best observed candidates are timing observations, not accepted policy rules.",
                "",
                "| Input | Shape | Mode | AUTO choice | Best explicit | AUTO / best explicit |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for row in comparisons:
            comparison = row["strategy_comparison"]
            lines.append(
                f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
                f"{row['triton_strategy']} | {comparison['best_explicit']} | "
                f"{comparison['auto_over_best_explicit']:.3f}x |"
            )
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--softcap", type=_positive_float, required=True)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--shapes", nargs="+", type=_shape, default=list(map(_shape, DEFAULT_SHAPES))
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Time ROW, PARALLEL and candidate ROW_PIPELINED depths in forward/forward+backward",
    )
    parser.add_argument(
        "--pipeline-stages",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_PIPELINE_STAGES),
        help="Candidate depths for --compare-strategies (default: 1 2 3 4 5 6 8)",
    )
    parser.add_argument("--rounds", type=_positive_int, default=4, help="Rotated comparison rounds")
    parser.add_argument(
        "--list-cases", action="store_true", help="Print the plan without a GPU run"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/softcapped-selected-logprob")
    )
    return parser


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    # Delay GPU imports so --help and --list-cases also work on CPU-only hosts.
    import triton

    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as core
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_pipeline_config,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        native = NativeSoftcappedSelectedLogprobOp()
        candidate = TritonSoftcappedSelectedLogprobOp()
        strategy_candidates = [
            (strategy.value, TritonSoftcappedSelectedLogprobOp(forward_impl=strategy))
            for strategy in SoftcappedLogprobStrategy
        ]
        explicit = []
        if args.compare_strategies and any(mode != "backward" for mode in args.modes):
            explicit = [
                _Provider(strategy.value, TritonSoftcappedSelectedLogprobOp(forward_impl=strategy))
                for strategy in (SoftcappedLogprobStrategy.ROW, SoftcappedLogprobStrategy.PARALLEL)
            ]
            explicit.extend(
                _Provider(
                    name=f"row_pipelined_s{stage}",
                    op=TritonSoftcappedSelectedLogprobOp(
                        forward_impl=SoftcappedLogprobStrategy.ROW_PIPELINED
                    ),
                    pipeline_stages=stage,
                )
                for stage in dict.fromkeys(args.pipeline_stages)
            )
            strategy_candidates.extend(
                (provider.name, _configured_check_op(core, provider))
                for provider in explicit
                if provider.pipeline_stages is not None
            )
        env = _environment(
            device, args, triton.__version__, REDUCTION_CONFIG.tile_size, profiler.gpu_info
        )
        report = {
            "environment": env,
            "case_plan": _case_plan(args),
            "complete": False,
            "results": [],
        }
        _write_report(report, args.output_dir)
        case_number = 0
        started = time.perf_counter()
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                case_started = time.perf_counter()
                case_number += 1
                print(
                    f"Case {case_number}/{report['case_plan']['case_count']}: "
                    f"checking {dtype_name} {shape}, softcap={args.softcap}...",
                    flush=True,
                )
                generator = torch.Generator(device=device).manual_seed(args.seed)
                logits = (30.0 * torch.randn(shape, device=device, generator=generator)).to(
                    DTYPES[dtype_name]
                )
                ids = torch.randint(shape[1], (shape[0],), device=device, generator=generator)
                upstream = torch.randn(
                    shape[0], device=device, generator=generator, dtype=torch.float32
                )
                accuracy = _check_accuracy(
                    native,
                    candidate,
                    logits,
                    ids,
                    upstream,
                    softcap=args.softcap,
                    strategy_candidates=strategy_candidates,
                    check_saved_statistics=True,
                )
                # Descriptive metadata only: the timed Op still selects its strategy itself.
                strategy = select_softcapped_logprob_strategy(
                    softcapped_logprob_device_key(logits.device), logits.dtype, *shape
                ).value
                pipeline_config = select_softcapped_logprob_pipeline_config(
                    softcapped_logprob_device_key(logits.device), logits.dtype, *shape
                )
                for mode in args.modes:
                    measurements = {}
                    if explicit and mode != "backward":
                        print(
                            f"  {mode}: {len(explicit) + 2} providers, {args.rounds} rounds",
                            flush=True,
                        )
                        measurements = _measure_strategy_comparison(
                            profiler,
                            [
                                _Provider("native", native),
                                _Provider("triton", candidate),
                                *explicit,
                            ],
                            logits,
                            ids,
                            upstream,
                            softcap=args.softcap,
                            core=core,
                            mode=mode,
                            rounds=args.rounds,
                        )
                    else:
                        for name, op in (("native", native), ("triton", candidate)):
                            fn = _make_workload(
                                op, logits, ids, upstream, mode, softcap=args.softcap
                            )
                            measurements[name] = _measure(profiler, fn)
                            del fn  # Free the retained graph before the next backend.
                    row = {
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "mode": mode,
                        "triton_strategy": strategy,
                        "accuracy": accuracy,
                        "train_inference_bitwise": True,
                        "strategy_bitwise": [name for name, _ in strategy_candidates],
                        "saved_statistics_bitwise": True,
                        "row_pipeline_stages": pipeline_config.pipeline_stages,
                        "native": measurements["native"],
                        "triton": measurements["triton"],
                        "speedup": measurements["native"]["median_ms"]
                        / measurements["triton"]["median_ms"],
                    }
                    if explicit and mode != "backward":
                        best = min(
                            explicit, key=lambda provider: measurements[provider.name]["median_ms"]
                        )
                        row["strategy_comparison"] = {
                            "providers": {
                                provider.name: measurements[provider.name] for provider in explicit
                            },
                            "pipeline_stages": {
                                provider.name: provider.pipeline_stages for provider in explicit
                            },
                            "best_explicit": best.name,
                            "auto_over_best_explicit": measurements["triton"]["median_ms"]
                            / measurements[best.name]["median_ms"],
                        }
                    report["results"].append(row)
                    # Retain completed cases on interruption; disk I/O is outside timing.
                    _write_report(report, args.output_dir)
                    print(
                        f"{dtype_name} {shape} {mode} ({strategy}): {row['speedup']:.2f}x",
                        flush=True,
                    )
                    if "strategy_comparison" in row:
                        comparison = row["strategy_comparison"]
                        print(
                            f"  best observed: {comparison['best_explicit']}; "
                            f"AUTO/best={comparison['auto_over_best_explicit']:.3f}x",
                            flush=True,
                        )
                del logits, ids, upstream
                print(
                    f"Case finished in {time.perf_counter() - case_started:.1f}s; "
                    f"elapsed {(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )
        report["complete"] = True
        _write_report(report, args.output_dir)
    return report


def _write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    pending = json_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(json_path)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")


def main():
    args = build_arg_parser().parse_args()
    print(json.dumps(_case_plan(args), indent=2), flush=True)
    if args.list_cases:
        return
    report = run_benchmark(args)
    print(_markdown(report))
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
