# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare eager PyTorch, split Triton, and two fused forward implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the softcap benchmark's profiler timing, peak-memory measurement and
# environment fingerprint. Only the workloads and report differ here.
from benchmarks.benchmark_final_logit_softcap import (  # noqa: E402
    DTYPES,
    MODES,
    _environment,
    _measure,
    _positive_int,
)
from benchmarks.profiler import PerformanceProfiler  # noqa: E402
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.kernels.ops.pytorch.loss import NativeSoftcappedSelectedLogprobOp  # noqa: E402

DEFAULT_SHAPES = ("1x1025", "1x262144", "4x262144", "16x262144", "64x262144")
# Cross row counts with vocabulary widths: row-level parallelism also changes
# the point where adding vocabulary-level parallelism pays for a second launch.
DISPATCH_ROWS = (1, 4, 16, 64, 256)
DISPATCH_VOCAB_SIZES = (
    512,
    1023,
    1024,
    1025,
    2048,
    4096,
    4097,
    8192,
    16384,
    32768,
    32769,
    65536,
    128256,
    131072,
    151936,
    256000,
    262144,
)
IMPLEMENTATIONS = ("native", "split_triton", "triton", "triton_parallel")
DISPATCH_MIN_SPEEDUP = 1.05
DISPATCH_MAX_RELATIVE_NOISE = 0.10


def _shape(value: str) -> tuple[int, int]:
    try:
        dims = tuple(int(dim) for dim in value.split("x"))
        if len(dims) != 2 or any(dim <= 0 for dim in dims):
            raise ValueError
        return dims
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive MxV, e.g. 16x262144") from exc


def _resolve_cases(args):
    """Expand a preset or explicit geometry without importing the GPU backend."""
    if getattr(args, "_cases_resolved", False):
        return args
    dispatch = args.suite == "dispatch"
    if args.shapes is not None and (args.rows is not None or args.vocab_sizes is not None):
        raise ValueError("use either --shapes or --rows/--vocab-sizes, not both")
    if args.shapes is None:
        if dispatch or args.rows is not None or args.vocab_sizes is not None:
            args.shapes = [
                (rows, vocab)
                for rows in (args.rows if args.rows is not None else DISPATCH_ROWS)
                for vocab in (
                    args.vocab_sizes if args.vocab_sizes is not None else DISPATCH_VOCAB_SIZES
                )
            ]
        else:
            args.shapes = list(map(_shape, DEFAULT_SHAPES))
    args.shapes = sorted(set(args.shapes))
    args.dtypes = list(dict.fromkeys(args.dtypes))
    args.modes = list(dict.fromkeys(args.modes or (["forward"] if dispatch else MODES)))
    if args.warmup is None:
        args.warmup = 25 if dispatch else 10
    if args.repeat is None:
        args.repeat = 100 if dispatch else 50
    if args.rounds is None:
        args.rounds = 4 if dispatch else 1
    args._cases_resolved = True
    return args


def _case_plan(args):
    return {
        "suite": args.suite,
        "dtypes": args.dtypes,
        "shapes": [list(shape) for shape in args.shapes],
        "modes": args.modes,
        "correctness_cases": len(args.dtypes) * len(args.shapes),
        "timing_cases": len(args.dtypes) * len(args.shapes) * len(args.modes),
        "implementations": list(IMPLEMENTATIONS),
        "rounds": args.rounds,
        "warmup_per_round": args.warmup,
        "repeat_per_round": args.repeat,
    }


class _SplitTritonOp:
    """Materialize FP32 softcap output before the existing Triton logprob op."""

    def __init__(self):
        from rl_engine.kernels.ops.triton.activation import TritonFinalLogitSoftcapOp
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        self.softcap = TritonFinalLogitSoftcapOp()
        self.logprob = TritonBatchInvariantLogpOp()

    def __call__(self, logits, token_ids):
        return self.logprob(self.softcap(logits), token_ids)


def _make_workload(op, logits, token_ids, upstream, mode):
    """Prepare inputs/graphs outside timing; never accumulate into leaf.grad."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(logits, token_ids)

        return forward

    leaf = logits.detach().requires_grad_(True)
    if mode == "backward":
        output = op(leaf, token_ids)

        def backward():
            torch.autograd.grad(output, leaf, grad_outputs=upstream, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaf, token_ids), leaf, grad_outputs=upstream)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(native, candidate, logits, token_ids, upstream):
    def evaluate(op):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, token_ids)
        gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
        return output.detach(), gradient

    expected, actual = evaluate(native), evaluate(candidate)
    errors = {}
    for label, judgment, shape, dtype, reference, result in zip(
        ("output", "gradient"),
        ("forward_accuracy", "gradient_accuracy"),
        (logits.shape[:1], logits.shape),
        (torch.float32, logits.dtype),
        expected,
        actual,
        strict=True,
    ):
        assert result.shape == shape and result.dtype == dtype
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="logprob", dtype=dtype
        )
        torch.testing.assert_close(result, reference, atol=tolerance.atol, rtol=tolerance.rtol)
        errors[label] = {
            "max_abs_error": (result.float() - reference.float()).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def _check_forward_variants(row_op, parallel_op, logits, token_ids, upstream):
    """Reject changes to the output, saved row statistics or resulting gradient."""

    def evaluate(op):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, token_ids)
        _, _, log_sum_exp = output.grad_fn.saved_tensors
        gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
        return output.detach(), log_sum_exp, gradient

    expected = evaluate(row_op)
    actual = evaluate(parallel_op)
    checks = {}
    for name, reference, result in zip(
        ("output", "log_sum_exp", "gradient"), expected, actual, strict=True
    ):
        # Compare stored bits, including signed zeros, rather than relaxing the
        # tolerance for the experimental forward's extra global-memory round trip.
        if not torch.equal(
            reference.contiguous().view(torch.uint8), result.contiguous().view(torch.uint8)
        ):
            raise AssertionError(f"Row and parallel forward variants differ in {name}")
        checks[name] = True
    return checks


def _measurement_order(round_index):
    offset = round_index % len(IMPLEMENTATIONS)
    return IMPLEMENTATIONS[offset:] + IMPLEMENTATIONS[:offset]


def _measure_case(profiler, ops, logits, ids, upstream, mode, rounds):
    measurements = {name: [] for name in IMPLEMENTATIONS}
    for round_index in range(rounds):
        # Over four rounds each implementation occupies each timing position.
        # Only one retained graph is live at a time, as in the original benchmark.
        for name in _measurement_order(round_index):
            fn = _make_workload(ops[name], logits, ids, upstream, mode)
            measurements[name].append(_measure(profiler, fn))
            del fn
    return {name: _summarize_measurements(runs) for name, runs in measurements.items()}


def _summarize_measurements(runs):
    medians = [run["median_ms"] for run in runs]
    stds = [run["std_ms"] for run in runs]
    median = statistics.median(medians)
    return {
        "median_ms": median,
        # This is deliberately not called a pooled sample standard deviation.
        "std_ms": max(stds) if all(std is not None for std in stds) else None,
        "round_median_relative_spread": (max(medians) - min(medians)) / median,
        "peak_extra_mib": max(run["peak_extra_mib"] for run in runs),
        "rounds": runs,
    }


def _forward_choice(measurements):
    """Conservative per-shape evidence, not an installed runtime dispatch rule."""
    row = measurements["triton"]["rounds"]
    parallel = measurements["triton_parallel"]["rounds"]
    ratios = [r["median_ms"] / p["median_ms"] for r, p in zip(row, parallel, strict=True)]
    noise = [
        run["std_ms"] / run["median_ms"] for run in row + parallel if run["std_ms"] is not None
    ]
    result = {
        "candidate": "inconclusive",
        "row_over_parallel_per_round": ratios,
        "max_relative_std": max(noise) if noise else None,
    }
    if len(row) < 4 or len(row) % 4 or len(noise) != len(row + parallel):
        result["reason"] = "need complete four-round rotations and repeat >= 2"
    elif max(noise) > DISPATCH_MAX_RELATIVE_NOISE or any(
        measurements[name]["round_median_relative_spread"] > DISPATCH_MAX_RELATIVE_NOISE
        for name in ("triton", "triton_parallel")
    ):
        result["reason"] = "timing noise exceeds the screening limit; rerun"
    elif min(ratios) >= DISPATCH_MIN_SPEEDUP:
        result.update(candidate="parallel", reason="parallel wins every round by at least 5%")
    elif max(ratios) <= 1 / DISPATCH_MIN_SPEEDUP:
        result.update(candidate="row", reason="row wins every round by at least 5%")
    else:
        result["reason"] = "winner changes or the speedup is below 1.05x"
    return result


def _dispatch_summary(report):
    """Do not infer a vocabulary threshold or hide dependence on the row count."""
    groups = {}
    for row in report["results"]:
        if row["mode"] != "forward" or "forward_choice" not in row:
            continue
        key = (row["dtype"], row["shape"][1])
        group = groups.setdefault(key, {"row": [], "parallel": [], "inconclusive": []})
        group[row["forward_choice"]["candidate"]].append(row["shape"][0])
    summary = []
    for (dtype, vocab), group in sorted(groups.items()):
        expected_rows = {m for m, v in report["case_plan"]["shapes"] if v == vocab}
        measured_rows = set(group["row"] + group["parallel"] + group["inconclusive"])
        if group["row"] and group["parallel"]:
            candidate = "depends_on_rows"
        elif group["inconclusive"] or measured_rows != expected_rows:
            candidate = "inconclusive"
        else:
            candidate = "row" if group["row"] else "parallel"
        summary.append(
            {
                "dtype": dtype,
                "vocab_size": vocab,
                "candidate": candidate,
                "rows_by_candidate": {name: sorted(rows) for name, rows in group.items()},
                "missing_rows": sorted(expected_rows - measured_rows),
            }
        )
    return summary


def _markdown(report):
    env = report["environment"]
    lines = [
        "# Softcapped selected logprob benchmark",
        "",
        f"GPU: {env['gpu']}; {env['backend']} runtime: {env['runtime']}; "
        f"PyTorch: {env['torch']}; Triton: {env['triton']}.",
        f"Commit: `{env['git_commit']}`; tracked changes: "
        f"`{env['git_tracked_changes'] or 'none'}`.",
        f"Warmup: {env['warmup']}; repetitions: {env['repeat']}; "
        f"vocab tile: {env['block_size']}; warps: {env['num_warps']}.",
        f"Timing rounds: {env.get('rounds', 1)}; report complete: "
        f"`{report.get('complete', True)}`.",
        "",
        "Native = eager PyTorch. Split = the existing Triton final_logit_softcap followed "
        "by TritonBatchInvariantLogpOp. Fused row = the original per-row loop. "
        "Fused parallel = per-tile partial sums followed by an ordered merge. "
        "Both fused paths share the same tiled backward. No torch.compile or graph capture. "
        "All Triton paths pass output and gradient checks against native before timing; "
        "the fused variants also pass bitwise output, saved log_sum_exp and gradient checks. "
        "The JSON contains errors, tolerances and exact-match results.",
        "Latency is the median of round medians ± the maximum within-round sample "
        "standard deviation in ms (N/A for one repetition), using the profiler's "
        "CUDA/HIP event timer. Public-wrapper allocation and autograd dispatch are included. "
        "The parallel forward allocates scratch and launches both kernels inside the timed "
        "call. Input generation, correctness checks and JIT compilation are outside timing.",
        "Forward uses no_grad. Backward reuses a retained graph; forward+backward builds "
        "a fresh graph per call. Extra peak memory excludes inputs and retained graphs.",
        "Native/row and split/row compare the baselines with fused row. Row/parallel > 1 "
        "means the parallel version is faster; < 1 means it is slower. "
        "Small cases may include substantial host dispatch gaps.",
    ]
    if "case_plan" in report:
        plan = report["case_plan"]
        lines += [
            f"Suite: `{plan['suite']}`; {plan['correctness_cases']} planned dtype/shape "
            f"checks; {len(report['results'])}/{plan['timing_cases']} completed timing cases. "
            "Each round rotates implementation order; JSON retains every round's "
            "median, standard deviation and extra peak allocation.",
        ]
    if report.get("dispatch_summary"):
        lines += [
            "",
            "## Forward selection evidence",
            "",
            "Candidates apply only to the measured GPU/software, contiguous inputs and "
            "listed row counts. A candidate needs complete four-round rotations, at least "
            "1.05x speedup in every round, and both within-round std/median and round-median "
            "relative spread <= 10%. This is a screening heuristic, not a confidence "
            "interval. Inconclusive cases need more measurements. Backward is shared and "
            "does not select a forward implementation. No runtime dispatch rule is changed; "
            "do not extrapolate a vocabulary threshold from this table.",
            "",
            "| Dtype | Vocab | Candidate across tested rows | Row faster: M | "
            "Parallel faster: M | Inconclusive: M | Missing: M |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
        for item in report["dispatch_summary"]:
            groups = item["rows_by_candidate"]
            row_sets = [groups[key] for key in ("row", "parallel", "inconclusive")]
            row_sets.append(item["missing_rows"])
            rows_text = " | ".join(", ".join(map(str, rows)) if rows else "—" for rows in row_sets)
            lines.append(
                f"| {item['dtype']} | {item['vocab_size']} | {item['candidate']} | {rows_text} |"
            )
    lines += [
        "",
        "| Dtype | Shape | Mode | Native ms | Split ms | Fused row ms | Fused parallel ms | "
        "Native/row | Split/row | Row/parallel |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        timings = []
        for name in ("native", "split_triton", "triton", "triton_parallel"):
            measurement = row[name]
            std_ms = measurement["std_ms"]
            std_text = f"{std_ms:.6f}" if std_ms is not None else "N/A"
            timings.append(f"{measurement['median_ms']:.6f} ± {std_text}")
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
            + " | ".join(timings)
            + f" | {row['speedup']:.2f}x | {row['split_speedup']:.2f}x | "
            f"{row['parallel_speedup']:.2f}x |"
        )
    lines += [
        "",
        "| Dtype | Shape | Mode | Native extra MiB | Split extra MiB | "
        "Fused row extra MiB | Fused parallel extra MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        memory = " | ".join(
            f"{row[name]['peak_extra_mib']:.3f}"
            for name in ("native", "split_triton", "triton", "triton_parallel")
        )
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | {memory} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--suite",
        choices=("standard", "dispatch"),
        default="standard",
        help="standard: original cases/all modes; dispatch: dtype x row x vocab forward sweep",
    )
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--shapes", nargs="+", type=_shape, help="explicit MxV cases")
    parser.add_argument(
        "--rows",
        nargs="+",
        type=_positive_int,
        help="cross these row counts with --vocab-sizes (defaults to the dispatch widths)",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=_positive_int,
        help="cross these widths with --rows (defaults to the dispatch row counts)",
    )
    parser.add_argument("--warmup", type=_positive_int, help="per round: standard 10, dispatch 25")
    parser.add_argument("--repeat", type=_positive_int, help="per round: standard 50, dispatch 100")
    parser.add_argument("--rounds", type=_positive_int, help="standard 1, dispatch 4")
    parser.add_argument("--list-cases", action="store_true", help="print the plan without a GPU")
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--output-dir", type=Path, default=Path("../softcapped-logprob-results"))
    return parser


def run_benchmark(args):
    args = _resolve_cases(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    import triton

    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        _BLOCK_V,
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        ops = {
            "native": NativeSoftcappedSelectedLogprobOp(),
            "split_triton": _SplitTritonOp(),
            "triton": TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW),
            "triton_parallel": TritonSoftcappedSelectedLogprobOp(
                forward_impl=SoftcappedLogprobStrategy.PARALLEL
            ),
        }
        env = _environment(device, args, triton.__version__, _BLOCK_V, profiler.gpu_info)
        env.update(
            {
                "operator": "softcapped_selected_logprob",
                "softcap": 30.0,
                "num_warps": 4,
                "baseline": "eager NativeSoftcappedSelectedLogprobOp; no torch.compile",
                "split_baseline": "TritonFinalLogitSoftcapOp + TritonBatchInvariantLogpOp",
                "fused_schedule": (
                    "forward: one program per row, ascending 1024-element FP32 tile sums; "
                    "backward: one program per row/vocabulary tile using saved log_sum_exp; "
                    "fp_fusion=False"
                ),
                "parallel_fused_schedule": (
                    "forward: one program per 1024-element vocabulary tile, FP32 scratch, "
                    "then one program per row accumulating tile sums in ascending order; "
                    "same backward as fused row; fp_fusion=False"
                ),
                "parallel_scratch_bytes": "4 * M * ceil(V / 1024), forward only",
                "modes": args.modes,
                "rounds": args.rounds,
                "measurement_order": [list(_measurement_order(r)) for r in range(args.rounds)],
                "dispatch_screening": {
                    "min_speedup_every_round": DISPATCH_MIN_SPEEDUP,
                    "max_relative_std_and_round_spread": DISPATCH_MAX_RELATIVE_NOISE,
                    "requires_complete_four_round_rotations": True,
                    "scope": "forward only; measured hardware, shapes and contiguous inputs",
                    "runtime_dispatch_changed": False,
                },
            }
        )
        # Include source hashes even if the measured operator files are untracked.
        sources = [
            "benchmarks/benchmark_softcapped_selected_logprob.py",
            "benchmarks/benchmark_final_logit_softcap.py",
            "benchmarks/profiler.py",
            "rl_engine/kernels/ops/pytorch/loss/softcapped_selected_logprob.py",
            "rl_engine/kernels/ops/triton/loss/softcapped_selected_logprob.py",
            "rl_engine/kernels/ops/triton/activation/final_logit_softcap.py",
            "rl_engine/kernels/ops/triton/loss/batch_invariant_logp.py",
        ]
        env["source_sha256"] = {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in sources
        }
        report = {
            "environment": env,
            "case_plan": _case_plan(args),
            "complete": False,
            "results": [],
            "dispatch_summary": [],
        }
        _write_report(report, args.output_dir)
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                print(f"Checking {dtype_name} {shape} before timing...", flush=True)
                generator = torch.Generator(device=device).manual_seed(args.seed)
                logits = (torch.randn(shape, device=device, generator=generator) * 30).to(
                    DTYPES[dtype_name]
                )
                ids = torch.randint(shape[1], (shape[0],), device=device, generator=generator)
                upstream = torch.randn(shape[0], device=device, generator=generator)
                accuracy = {
                    name: _check_accuracy(ops["native"], ops[name], logits, ids, upstream)
                    for name in ("split_triton", "triton", "triton_parallel")
                }
                exact_match = _check_forward_variants(
                    ops["triton"], ops["triton_parallel"], logits, ids, upstream
                )
                for mode in args.modes:
                    measurements = _measure_case(
                        profiler, ops, logits, ids, upstream, mode, args.rounds
                    )
                    fused_ms = measurements["triton"]["median_ms"]
                    row = {
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "mode": mode,
                        "accuracy": accuracy,
                        "forward_variants_bitwise_equal": exact_match,
                        **measurements,
                        "speedup": measurements["native"]["median_ms"] / fused_ms,
                        "split_speedup": measurements["split_triton"]["median_ms"] / fused_ms,
                        "parallel_speedup": fused_ms / measurements["triton_parallel"]["median_ms"],
                    }
                    if mode == "forward":
                        row["forward_choice"] = _forward_choice(measurements)
                    report["results"].append(row)
                    report["dispatch_summary"] = _dispatch_summary(report)
                    # Keep completed evidence if a later correctness gate or GPU run fails.
                    # Disk I/O is outside all timed workloads.
                    _write_report(report, args.output_dir)
                    print(
                        f"{dtype_name} {shape} {mode}: native/row={row['speedup']:.2f}x, "
                        f"split/row={row['split_speedup']:.2f}x, "
                        f"row/parallel={row['parallel_speedup']:.2f}x"
                        + (
                            f", candidate={row['forward_choice']['candidate']}"
                            if mode == "forward"
                            else ""
                        ),
                        flush=True,
                    )
                del logits, ids, upstream
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
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args = _resolve_cases(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(_case_plan(args), indent=2), flush=True)
    if args.list_cases:
        return
    report = run_benchmark(args)
    print(_markdown(report))
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
