# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark H3 partial RoPE, including the original eager model as a baseline.

Run from the repository root with ``python -m benchmarks.models.benchmark_h3_mm_rope``.
Inputs use packed H3 positions; table generation, validation and compilation are
outside timing. The public wrappers are timed without torch.compile or CUDA Graphs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from functools import partial
from pathlib import Path

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.rope.h3_mm_rope import NativeH3MMRoPEOp, model_boundary_reference
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout
from rl_engine.validation.models.h3_provider import provider_apply_rotary, provider_rope

LAYOUTS = {
    **H3_BLOCK_LAYOUTS,
    "long": dict(
        num_text_tokens=40,
        num_latent_frames=72,
        latent_height=48,
        latent_width=84,
        num_audio_latents=405,
    ),
}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")


def parse_args(argv=None):
    """Select model layouts and timing effort; dry-run requires no GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--heads", nargs="+", type=int, choices=(14, 56), default=[14, 56])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--seed", type=int, default=656)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/h3-mm-rope"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeat < 2:
        parser.error("warmup must be >= 1 and repeat must be >= 2")
    return args


def case_plan(args):
    """Enumerate one Q or K tensor per case, with full or head-sharded geometry."""
    cases = []
    for layout in args.layouts:
        config = LAYOUTS[layout]
        length = (
            config["num_text_tokens"]
            + 2 * config["num_audio_latents"]
            + config["num_latent_frames"]
            * (config["latent_height"] // 2)
            * (config["latent_width"] // 2)
        )
        for heads in args.heads:
            for dtype in args.dtypes:
                cases.append(dict(layout=layout, dtype=dtype, shape=[1, length, heads, 128]))
    return cases


def _evaluate(op, x, cos, sin, upstream):
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, cos, sin)
    return output.detach(), torch.autograd.grad(output, leaf, upstream)[0]


def _record_error(summary, actual, reference, tolerance, token_offset):
    """Accumulate full-domain diagnostics, including global outlier coordinates."""
    error = (actual.float() - reference).abs()
    limit = tolerance.atol + tolerance.rtol * reference.abs()
    violations = (error > limit) | ~torch.isfinite(error)
    summary["elements"] += error.numel()
    summary["outside_tolerance"] += int(violations.sum())
    summary["max_abs_error"] = max(summary["max_abs_error"], float(error.max()))
    summary["max_error_over_tolerance"] = max(
        summary["max_error_over_tolerance"], float((error / limit).max())
    )
    if len(summary["examples"]) < 8:
        for index in violations.nonzero()[: 8 - len(summary["examples"])].tolist():
            local = tuple(index)
            index[1] += token_offset
            summary["examples"].append(
                dict(
                    index=index,
                    actual=float(actual[local]),
                    reference=float(reference[local]),
                    abs_error=float(error[local]),
                    allowed_error=float(limit[local]),
                )
            )


def check_case(operators, x, cos, sin, upstream):
    """Gate on model boundaries; separately report unrounded FP32 mathematical error.

    Validate in token chunks to bound temporary memory on small GPUs. The subsequent
    full-shape check compares both candidate outputs/gradients with the Native path.
    No tolerance is inferred from the benchmark results.
    """
    contract = load_contract()
    diagnostics = {
        "reference": "original_fp32_coefficients_and_unrounded_fp32_math",
        "is_acceptance_gate": False,
        "forward": {},
        "backward": {},
        "provider_backward": {},
    }
    for name in ("forward", "backward", "provider_backward"):
        diagnostics[name] = dict(
            elements=0,
            outside_tolerance=0,
            max_abs_error=0.0,
            max_error_over_tolerance=0.0,
            examples=[],
        )
    acceptance = dict(reference="model_boundary_reference", passed=True)
    reference_op = NativeH3MMRoPEOp()
    acceptance_fn = partial(model_boundary_reference, storage_dtype=x.dtype)
    for start in range(0, x.shape[1], 1024):
        view, dy = x[:, start : start + 1024], upstream[:, start : start + 1024]
        c, s = cos[start : start + 1024], sin[start : start + 1024]
        actual, grad = _evaluate(operators["triton"], view, c, s, dy)
        expected, d_expected = _evaluate(operators["native"], view, c, s, dy)
        for result, reference in (
            (actual, expected),
            (grad, d_expected),
            (actual, provider_apply_rotary(view, c, s)),
        ):
            assert torch.equal(
                result.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
            )  # noqa: E501
        golden, d_golden = _evaluate(
            acceptance_fn,
            view.float(),
            c,
            s,
            dy.float(),
        )
        for judgment, result, reference in (
            ("forward_accuracy", actual, golden),
            ("gradient_accuracy", grad, d_golden),
        ):
            tolerance = resolve_tolerance(
                contract, judgment=judgment, op_class="elementwise", dtype=x.dtype
            )
            try:
                torch.testing.assert_close(
                    result.float(), reference, atol=tolerance.atol, rtol=tolerance.rtol
                )
            except AssertionError as error:
                raise AssertionError(
                    f"model_boundary_reference {judgment}, "
                    f"tokens [{start}, {start + view.shape[1]}): {error}"
                ) from error
        math_y, math_dx = _evaluate(reference_op.forward_fp32, view.float(), c, s, dy.float())
        _, provider_dx = _evaluate(provider_apply_rotary, view, c, s, dy)
        for name, judgment, result, reference in (
            ("forward", "forward_accuracy", actual, math_y),
            ("backward", "gradient_accuracy", grad, math_dx),
            ("provider_backward", "gradient_accuracy", provider_dx, math_dx),
        ):
            tolerance = resolve_tolerance(
                contract, judgment=judgment, op_class="elementwise", dtype=x.dtype
            )
            diagnostics[name].update(atol=tolerance.atol, rtol=tolerance.rtol)
            _record_error(diagnostics[name], result, reference, tolerance, start)
    actual, grad = _evaluate(operators["triton"], x, cos, sin, upstream)
    expected, d_expected = _evaluate(operators["native"], x, cos, sin, upstream)
    for result, reference in ((actual, expected), (grad, d_expected)):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
    return dict(acceptance=acceptance, unrounded_math_diagnostic=diagnostics)


def timed_call(op, mode, x, cos, sin, upstream):
    """Build a timed public call without accumulating .grad or timing input creation.

    Backward reuses a prebuilt graph. Forward+backward builds a fresh graph each
    call. All returned callables return their result so execution stays eager.
    """
    if mode == "forward":
        return torch.no_grad()(lambda: op(x, cos, sin))
    leaf = x.detach().requires_grad_(True)
    if mode == "backward":
        output = op(leaf, cos, sin)
        return lambda: torch.autograd.grad(output, leaf, upstream, retain_graph=True)[0]
    if mode == "forward_backward":
        return lambda: torch.autograd.grad(op(leaf, cos, sin), leaf, upstream)[0]
    raise ValueError(f"unknown mode: {mode}")


def measure(calls, warmup, repeat):
    """Alternate provider order and record event latency plus extra peak allocation.

    Eager event latency includes host dispatch gaps. Inputs, upstream gradients
    and prebuilt graphs are excluded from incremental peak allocation. Raw samples
    are retained so latency variability is visible rather than hidden in a ratio.
    """
    names = list(calls)
    samples = {name: [] for name in names}
    for _ in range(warmup):
        for call in calls.values():
            call()
    torch.cuda.synchronize()
    for iteration in range(repeat):
        for name in names if iteration % 2 == 0 else names[::-1]:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    results = {}
    for name, call in calls.items():
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        call()
        torch.cuda.synchronize()
        results[name] = dict(
            median_us=statistics.median(samples[name]),
            stdev_us=statistics.stdev(samples[name]),
            extra_peak_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20,
            samples_us=samples[name],
        )
    return results


def write_report(folder, report):
    """Checkpoint JSON and readable tables after each completed shape."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# H3 partial MM-RoPE benchmark",
        "",
        "Eager public wrappers; event medians; no torch.compile or CUDA Graphs. "
        "Backward reuses a graph; forward+backward creates a fresh graph. "
        "Inputs, table generation, compilation and checks are outside timing.",
        "",
        "Provider = original model expression/autograd. Native = matched forward "
        "with explicit FP32 backward. BF16 provider backward rounds products earlier, "
        "so its gradient is not required to be bitwise equal.",
        "",
        f"Environment: `{report['environment']}`",
        "",
        f"Complete: {report['complete']}",
        "",
        "Acceptance: model coefficient/product storage boundaries, independent FP32 VJP. "
        "Unrounded FP32 mathematical error is diagnostic only; its outlier counts below "
        "are not waived or interpreted as a pure FP32 accuracy pass.",
        "",
        "| Dtype / shape | Mode | Provider us | Native us | Triton us | Provider / Triton | Native / Triton |",  # noqa: E501
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        p, n, t = (row["timings"][key]["median_us"] for key in ("provider", "native", "triton"))
        lines.append(
            f"| {row['dtype']} {'x'.join(map(str, row['shape']))} | {row['mode']} | "
            f"{p:.2f} | {n:.2f} | {t:.2f} | {p/t:.2f}x | {n/t:.2f}x |"
        )
    if report["failures"]:
        lines.extend(["", "Model-boundary acceptance failures (not timed):", ""])
        for failure in report["failures"]:
            lines.extend(
                [
                    f"- {failure['dtype']} {failure['shape']}",
                    "",
                    "```text",
                    failure["error"],
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "## Unrounded FP32 mathematical diagnostics (not an acceptance gate)",
            "",
            "| Dtype / shape | Direction | Outside tolerance / elements | Max abs error | Max error / limit |",  # noqa: E501
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case in report["validation"]:
        for direction in ("forward", "backward", "provider_backward"):
            metric = case["unrounded_math_diagnostic"][direction]
            lines.append(
                f"| {case['dtype']} {'x'.join(map(str, case['shape']))} | {direction} | "
                f"{metric['outside_tolerance']} / {metric['elements']} | "
                f"{metric['max_abs_error']:.8g} | {metric['max_error_over_tolerance']:.5g} |"
            )
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    plan = case_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("requires an NVIDIA CUDA GPU; the ROCm path is not qualified yet")
    import triton

    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.runtime.semantic_registry import OperatorRequirements

    # Limit CPU worker overhead in eager autograd; GPU arithmetic is unchanged.
    torch.set_num_threads(1)
    session = KernelRegistry().semantic.session()
    report = dict(
        schema_version=2,
        complete=False,
        acceptance_reference="model_boundary_reference",
        mathematical_reference="forward_fp32 (diagnostic only, not a timing gate)",
        case_plan=plan,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            hip=torch.version.hip,
            gpu=torch.cuda.get_device_name(),
            compute_capability=torch.cuda.get_device_capability(),
            cpu_threads=torch.get_num_threads(),
        ),
        backends={},
        results=[],
        validation=[],
        failures=[],
    )
    write_report(args.output_dir, report)
    for index, case in enumerate(plan, 1):
        print(f"Case {index}/{len(plan)}: {case}", flush=True)
        resolution = session.resolve(
            semantic_op="h3_mm_rope_3axis_partial",
            requested_backend="rlkernel.h3_mm_rope.triton",
            target="training",
            requirements=OperatorRequirements(
                device="cuda",
                dtype=str(DTYPES[case["dtype"]]).removeprefix("torch."),
                topology={"world_size": 1, "tensor_parallel_size": 1, "context_parallel_size": 1},
            ),
        )
        candidate = session.instantiate(resolution)
        report["backends"][case["dtype"]] = dict(
            resolution=resolution.to_dict(),
            instance=session.instance_provenance(resolution, candidate).to_dict(),
        )
        operators = dict(
            provider=provider_apply_rotary, native=NativeH3MMRoPEOp(), triton=candidate
        )
        positions = h3_fl2va_layout(**LAYOUTS[case["layout"]], device="cuda")["position_ids"]
        cos, sin = provider_rope(positions)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        x = torch.randn(
            case["shape"], device="cuda", dtype=DTYPES[case["dtype"]], generator=generator
        )
        upstream = torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator)
        try:
            validation = check_case(operators, x, cos, sin, upstream)
        except AssertionError as error:
            report["failures"].append(dict(**case, error=str(error)))
            print(f"  Model-boundary acceptance failed; skipping timing: {error}", flush=True)
        else:
            report["validation"].append(dict(**case, **validation))
            diagnostic = validation["unrounded_math_diagnostic"]
            print(
                "  Model-boundary acceptance passed; diagnostic-only unrounded FP32 "
                f"outliers: forward={diagnostic['forward']['outside_tolerance']}, "
                f"backward={diagnostic['backward']['outside_tolerance']}",
                flush=True,
            )
            for mode in MODES:
                calls = {
                    name: timed_call(op, mode, x, cos, sin, upstream)
                    for name, op in operators.items()
                }
                timings = measure(calls, args.warmup, args.repeat)
                report["results"].append(
                    dict(**case, mode=mode, checks_passed=True, timings=timings)
                )
                print(
                    f"  {mode}: "
                    + ", ".join(f"{n}={v['median_us']:.2f}us" for n, v in timings.items()),
                    flush=True,
                )
                del calls
        finally:
            write_report(args.output_dir, report)
            del x, upstream, cos, sin
            torch.cuda.empty_cache()
    report["complete"] = not report["failures"]
    write_report(args.output_dir, report)
    print(f"Reports saved to {args.output_dir}")
    if report["failures"]:
        raise SystemExit("model-boundary acceptance failures recorded; benchmark is incomplete")


if __name__ == "__main__":
    main()
