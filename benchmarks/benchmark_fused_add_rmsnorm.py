# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare eager PyTorch and automatic Triton fused add RMSNorm on the same GPU."""

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profiler import PerformanceProfiler  # noqa: E402
from rl_engine.kernels.ops.pytorch.norm import NativeFusedAddRMSNormOp  # noqa: E402

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_MODES = ("forward", "backward", "forward_backward")
_DEFAULT_SHAPES = [(1, 2688), (32, 2688), (8192, 2688), (16384, 2688), (65536, 2688)]


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _shape(value):
    try:
        rows, cols = map(int, value.split("x"))
        if rows <= 0 or cols <= 0:
            raise ValueError
        return rows, cols
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be positive MxD, e.g. 16384x2688") from exc


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=tuple(_DTYPES), default=list(_DTYPES))
    parser.add_argument("--shapes", nargs="+", type=_shape, default=_DEFAULT_SHAPES)
    parser.add_argument("--modes", nargs="+", choices=_MODES, default=list(_MODES))
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fused-add-rmsnorm"))
    parser.add_argument("--list-cases", "--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("dtypes", "shapes", "modes"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


def _case_plan(args):
    case_count = len(args.dtypes) * len(args.shapes)
    return {
        "dtypes": args.dtypes,
        "shapes": [list(shape) for shape in args.shapes],
        "modes": args.modes,
        "case_count": case_count,
        # One record contains both providers' measurements and their ratio.
        "measurement_count": case_count * len(args.modes),
    }


def _make_workload(op, inputs, upstream, mode):
    """Keep inputs/graphs alive, but do not retain timed returns between repetitions."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(*inputs)

        return forward

    leaves = tuple(value.detach().requires_grad_(True) for value in inputs)
    if mode == "backward":
        outputs = op(*leaves)

        def backward():
            torch.autograd.grad(outputs, leaves, upstream, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def combined():
            torch.autograd.grad(op(*leaves), leaves, upstream)

        return combined
    raise ValueError(f"Unknown mode: {mode}")


def _check_accuracy(native, candidate, inputs, upstream):
    def evaluate(op):
        leaves = tuple(value.detach().requires_grad_(True) for value in inputs)
        outputs = op(*leaves)
        gradients = torch.autograd.grad(outputs, leaves, upstream)
        return tuple(output.detach() for output in outputs) + gradients

    expected, actual = evaluate(native), evaluate(candidate)
    shapes = (inputs[0].shape, inputs[0].shape, *(value.shape for value in inputs))
    dtypes = (torch.float32, torch.float32, *(value.dtype for value in inputs))
    n_rows = inputs[0].numel() // inputs[0].shape[-1]
    errors = {}
    for label, result, reference, shape, dtype in zip(
        ("y", "updated_residual", "grad_x", "grad_residual", "grad_weight"),
        actual,
        expected,
        shapes,
        dtypes,
        strict=True,
    ):
        assert result.shape == shape and result.dtype == dtype
        rtol = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
        atol = 2e-5 * math.sqrt(n_rows) if label == "grad_weight" else rtol
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
        errors[label] = {
            "max_abs_error": (result.float() - reference.float()).abs().max().item(),
            "atol": atol,
            "rtol": rtol,
        }
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            inference = candidate(*inputs)
        for training, inferred in zip(actual[:2], inference, strict=True):
            assert training.shape == inferred.shape and training.dtype == inferred.dtype
            if not torch.equal(training.view(torch.uint8), inferred.view(torch.uint8)):
                raise AssertionError("Training/inference mismatch")
    return errors


def _measure(profiler, fn):
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"Invalid measured latency: {median_ms}")
    profiler._sync()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    fn()
    profiler._sync()
    extra = torch.cuda.max_memory_allocated(profiler.device) - baseline
    return {"median_ms": median_ms, "std_ms": std_ms, "peak_extra_mib": extra / 1024**2}


def _report(payload):
    env, config, plan = payload["environment"], payload["config"], payload["case_plan"]
    lines = [
        "# Fused add RMSNorm benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Status: {'complete' if payload['complete'] else 'incomplete'}; "
        f"{len(payload['results'])}/{plan['measurement_count']} comparisons.",
        f"Warmup: {config['warmup']}; measured repetitions: {config['repeat']}; "
        f"seed: {config['seed']}.",
        "",
        "Eager NativeFusedAddRMSNormOp vs TritonFusedAddRMSNormOp with automatic selection. "
        "Inputs x/residual use the listed dtype; weight and both random upstream gradients "
        "are FP32. eps=1e-5; both outputs are FP32. No torch.compile or CUDA Graph timing.",
        "",
        "The shared PerformanceProfiler accelerator-event timer reports median latency "
        "and sample standard deviation. Timings include public wrapper allocation and "
        "autograd dispatch. Input generation, accuracy checks and JIT compilation are "
        "outside timing; the selector cache is warm. The profiler releases unused allocator "
        "cache before timing, without explicit GPU data-cache eviction. Small cases can "
        "be dominated by host dispatch gaps.",
        "",
        "Forward uses no_grad. Backward reuses a prebuilt graph; forward+backward builds "
        "a fresh graph per call. Extra peak allocation excludes inputs and prebuilt graphs. "
        "Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower.",
        "",
        "Every reported case passed output and random-upstream gradient checks before timing, "
        "plus byte equality of grad-enabled/no_grad/inference_mode outputs. FP32 weight "
        "gradient tolerance is rtol=2e-5, atol=2e-5*sqrt(M). JSON records all error bounds "
        "and measured errors. This is operator validation, not full-model or microbatch parity.",
        "",
        "| Input | Shape | Triton strategy | Mode | Native ms | Triton ms | "
        "Speedup | Native extra MiB | Triton extra MiB |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload["results"]:
        native, triton, selected = result["native"], result["triton"], result["triton_plan"]
        shape = "x".join(map(str, result["shape"]))
        lines.append(
            f"| {result['dtype']} | {shape} | {selected['strategy']} | {result['mode']} | "
            f"{native['median_ms']:.6f} | {triton['median_ms']:.6f} | "
            f"{result['speedup']:.2f}x | {native['peak_extra_mib']:.2f} | "
            f"{triton['peak_extra_mib']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _save(payload, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(_report(payload), encoding="utf-8")


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires a CUDA or ROCm GPU")
    import triton

    from rl_engine.kernels.ops.triton.norm import TritonFusedAddRMSNormOp

    native, candidate = NativeFusedAddRMSNormOp(), TritonFusedAddRMSNormOp()
    with torch.cuda.device(device):
        device = torch.device("cuda", torch.cuda.current_device())
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        target = profiler.gpu_info
        payload = {
            "complete": False,
            "environment": {
                "python": platform.python_version(),
                "gpu": target.name,
                "torch": torch.__version__,
                "triton": triton.__version__,
                "backend": target.backend,
                "runtime": target.driver_version,
                "architecture": target.architecture,
                "device": str(device),
                "total_memory_gib": target.total_memory_gb,
            },
            "config": {
                **vars(args),
                "shapes": [list(shape) for shape in args.shapes],
                "output_dir": str(args.output_dir),
            },
            "case_plan": _case_plan(args),
            "results": [],
        }
        _save(payload, args.output_dir)
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                print(f"Checking {dtype_name} {shape} before timing...", flush=True)
                torch.manual_seed(args.seed)
                x = torch.randn(shape, device=device, dtype=_DTYPES[dtype_name])
                residual = torch.randn_like(x)
                weight = torch.randn(shape[-1], device=device, dtype=torch.float32)
                inputs = (x, residual, weight)
                upstream = tuple(
                    torch.randn(shape, device=device, dtype=torch.float32) for _ in range(2)
                )
                errors = _check_accuracy(native, candidate, inputs, upstream)
                leaves = tuple(value.detach().requires_grad_(True) for value in inputs)
                outputs = candidate(*leaves)
                ctx = outputs[0].grad_fn
                selected = {
                    "strategy": ctx.weight_grad_strategy.value,
                    "block_rows": ctx.weight_grad_config.block_rows,
                    "block_cols": ctx.weight_grad_config.block_cols,
                    "num_warps": ctx.weight_grad_config.num_warps,
                }
                del outputs, leaves, ctx
                for mode in args.modes:
                    measured = {}
                    for name, op in (("native", native), ("triton", candidate)):
                        fn = _make_workload(op, inputs, upstream, mode)
                        measured[name] = _measure(profiler, fn)
                        del fn
                    speedup = measured["native"]["median_ms"] / measured["triton"]["median_ms"]
                    payload["results"].append(
                        {
                            "dtype": dtype_name,
                            "shape": list(shape),
                            "mode": mode,
                            "triton_plan": selected,
                            "errors": errors,
                            "train_inference_bitwise": True,
                            "speedup": speedup,
                            **measured,
                        }
                    )
                    _save(payload, args.output_dir)
                    print(
                        f"{dtype_name} {shape} {mode} ({selected['strategy']}): {speedup:.2f}x",
                        flush=True,
                    )
                del inputs, upstream, x, residual, weight
        assert len(payload["results"]) == payload["case_plan"]["measurement_count"]
        payload["complete"] = True
        _save(payload, args.output_dir)
        print(f"Reports saved to {args.output_dir}", flush=True)
        return payload


def main(argv=None):
    args = _parse_args(argv)
    print(json.dumps(_case_plan(args), indent=2))
    if not args.list_cases:
        run_benchmark(args)


if __name__ == "__main__":
    main()
