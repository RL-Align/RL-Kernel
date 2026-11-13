# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark separate-input SwiGLU and the packed-input extension against PyTorch.

CUDA events measure forward, retained-graph backward, and fresh forward+backward.
Eager timing includes launch gaps; CUDA Graph replay removes most host dispatch.
Every shape passes shared-path bitwise and independent FP32 accuracy checks first.
See docs/operators/activation.md for the numerical and measurement contracts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_engine.contracts.numerical import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.contracts.operators.activation import SwiGLUPackedOrder  # noqa: E402
from rl_engine.reference.activation.swiglu import NativeSwiGLUOp  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
PROVIDERS = ("pytorch", "adapter", "packed")
SEPARATE_PROVIDERS = ("pytorch", "triton")
METHOD = (
    "CUDA-event per-call latency; median of sample means, then median across randomized "
    "provider rounds. Warm cache, no explicit eviction, no torch.compile. Forward uses "
    "no_grad; backward reuses a graph; forward_backward builds a fresh graph each call. "
    "Input generation, accuracy checks, JIT and CUDA Graph capture are excluded. "
    "Extra peak allocation is measured eagerly, excluding inputs and any prebuilt graph."
)


def positive_int(value: str) -> int:
    """Reject empty benchmark cases and zero-count timing settings."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    """Parse layout, shape and timing options, rejecting duplicate case dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-layout",
        choices=["separate", "packed"],
        default="packed",
        help="separate gate/up regression, or packed input with a chunk adapter baseline",
    )
    parser.add_argument(
        "--rows", nargs="+", type=positive_int, default=[1, 32, 256, 1024, 4096, 16384]
    )
    parser.add_argument(
        "--width", type=positive_int, default=14336, help="FFN intermediate width I"
    )
    parser.add_argument(
        "--packed-order",
        choices=[order.value for order in SwiGLUPackedOrder],
        default=SwiGLUPackedOrder.GATE_UP.value,
        help="order of the two equal halves in the input's last dimension",
    )
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--timings", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--warmup", type=positive_int, default=4)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=3)
    parser.add_argument("--sample-ms", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/swiglu-packed"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for field in ("rows", "dtypes", "timings"):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error(f"--{field} must not contain duplicates")
    return args


def case_plan(args):
    """Describe input shapes, providers and timing-round counts without using a GPU."""
    cases = len(args.rows) * len(args.dtypes)
    packed = args.input_layout == "packed"
    providers = PROVIDERS if packed else SEPARATE_PROVIDERS
    return {
        "case_count": cases,
        "measurement_count": cases * len(args.timings) * len(MODES) * len(providers) * args.rounds,
        "input_layout": args.input_layout,
        "input_count": 1 if packed else 2,
        "input_shapes": [[1, rows, (2 if packed else 1) * args.width] for rows in args.rows],
        "packed_order": args.packed_order if packed else None,
        "providers": providers,
    }


def assert_bits(actual, expected):
    """Require matching shapes, dtypes and bytes for tensors or tuples of tensors."""
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_bits(actual_item, expected_item)
        return
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    ):
        raise AssertionError("compared paths differ in shape, dtype, or bits")


def check_case(providers, x, upstream, packed_order):
    """Separate shared-path equivalence from independent FP32 accuracy."""
    packed = providers["packed"](packed_input=x)
    grad = torch.autograd.grad(packed, x, upstream)[0]
    shared = providers["adapter"](packed_input=x)
    assert_bits(packed, shared)
    assert_bits(grad, torch.autograd.grad(shared, x, upstream)[0])
    with torch.no_grad():
        assert_bits(packed, providers["packed"](packed_input=x))
    gold_x = x.detach().float().requires_grad_(True)
    gold = NativeSwiGLUOp().forward_fp32(packed_input=gold_x, packed_order=packed_order)
    gold_grad = torch.autograd.grad(gold, gold_x, upstream.float())[0]
    errors = {}
    for judgment, actual, expected in (
        ("forward_accuracy", packed, gold),
        ("gradient_accuracy", grad, gold_grad),
    ):
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="elementwise", dtype=x.dtype
        )
        torch.testing.assert_close(
            actual.float(), expected, rtol=tolerance.rtol, atol=tolerance.atol
        )
        errors[judgment] = {
            "max_abs_error": (actual.float() - expected).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def check_separate_case(providers, inputs, upstream):
    """Gate the existing two-input path on train/infer bits and FP32 output/gradients."""
    gate, up = inputs
    actual = providers["triton"](gate=gate, up=up)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    with torch.no_grad():
        assert_bits(actual, providers["triton"](gate=gate, up=up))
    gold_inputs = tuple(value.detach().float().requires_grad_(True) for value in inputs)
    gold_gate, gold_up = gold_inputs
    gold = NativeSwiGLUOp().forward_fp32(gate=gold_gate, up=gold_up)
    gold_gradients = torch.autograd.grad(gold, gold_inputs, upstream.float())
    errors = {}
    for name, judgment, result, expected in (
        ("forward", "forward_accuracy", actual, gold),
        ("gate", "gradient_accuracy", gradients[0], gold_gradients[0]),
        ("up", "gradient_accuracy", gradients[1], gold_gradients[1]),
    ):
        tolerance = resolve_tolerance(
            load_contract(),
            judgment=judgment,
            op_class="elementwise",
            dtype=inputs[0].dtype,
        )
        torch.testing.assert_close(
            result.float(), expected, rtol=tolerance.rtol, atol=tolerance.atol
        )
        errors[name] = {
            "max_abs_error": (result.float() - expected).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def make_call(op, inputs, upstream, mode):
    """Build backward state on the same stream used for measurement and capture."""
    if len(inputs) == 2:
        gate, up = inputs
        arguments = {"gate": gate, "up": up}
    else:
        (packed_input,) = inputs
        arguments = {"packed_input": packed_input}
    if mode == "backward":
        y = op(**arguments)
        return lambda: torch.autograd.grad(y, inputs, upstream, retain_graph=True)
    if mode == "forward_backward":
        return lambda: torch.autograd.grad(op(**arguments), inputs, upstream)

    def forward():
        """Measure forward execution without building an autograd graph."""
        with torch.no_grad():
            return op(**arguments)

    return forward


def measure(op, inputs, upstream, mode, timing, args):
    """Keep captured autograd state private; measure eager memory after replay timing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call = make_call(op, inputs, upstream, mode)
        for _ in range(args.warmup):
            call()
        stream.synchronize()
        graph = None
        if timing == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            graph.replay()
            stream.synchronize()
            assert_bits(captured, call())
            timed_call = graph.replay
        else:
            timed_call = call
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        timed_call()
        end.record()
        end.synchronize()
        repeats = max(5, min(100, int(args.sample_ms / max(start.elapsed_time(end), 0.001))))
        samples = []
        for _ in range(args.samples):
            start.record()
            for _ in range(repeats):
                timed_call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / repeats)
        # Graph-private pools are still alive but excluded from the baseline;
        # this call runs eagerly and only its extra allocation is reported.
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = call()
        stream.synchronize()
        extra_mib = (torch.cuda.max_memory_allocated() - allocated) / 2**20
        del output
    torch.cuda.current_stream().wait_stream(stream)
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "repeats": repeats,
        "extra_peak_mib": extra_mib,
    }


def write_report(report, folder):
    """Write checkpoints with an explicit completion flag, including failed gates."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    plan = report["case_plan"]
    providers = plan["providers"]
    lines = [
        f"# SwiGLU benchmark: {plan['input_layout']} inputs",
        "",
        f"GPU: {report.get('hardware', 'not started')}. {METHOD}",
        "",
        f"Input count: {plan['input_count']}; packed order: {plan['packed_order']}; "
        f"shape of each input: {plan['input_shapes']}.",
        "",
        f"Complete: {report['complete']}; {len(report['results'])}/"
        f"{report['case_plan']['measurement_count']} timing-round records.",
        "",
        "| Dtype | M | Timing | Mode | " + " | ".join(f"{p} ms" for p in providers) + " |",
        "| --- | ---: | --- | --- | " + " | ".join("---:" for _ in providers) + " |",
    ]
    groups = {}
    for row in report["results"]:
        key = (row["dtype"], row["rows"], row["timing"], row["mode"])
        groups.setdefault(key, {}).setdefault(row["provider"], []).append(row["median_ms"])
    for key, values in groups.items():
        if all(len(values.get(p, [])) == report["settings"]["rounds"] for p in providers):
            medians = [f"{statistics.median(values[p]):.6f}" for p in providers]
            lines.append("| " + " | ".join([*(str(v) for v in key), *medians]) + " |")
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    """Run correctness-gated measurements and save reports even when a case fails.

    Dry runs only print the case plan. GPU runs randomize provider order in
    each round and record latency, extra peak allocation and source hashes.
    """
    args = parse_args(argv)
    plan = case_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/ROCm GPU is required; --dry-run works without one")
    import triton

    from rl_engine.backends.shared.triton.activation.swiglu import TritonSwiGLUOp

    shared = TritonSwiGLUOp()
    packed_order = SwiGLUPackedOrder(args.packed_order)

    def adapter(*, packed_input):
        """Benchmark packed input through chunk views and the existing two-input path."""
        first, second = packed_input.chunk(2, dim=-1)
        gate, up = (first, second) if packed_order is SwiGLUPackedOrder.GATE_UP else (second, first)
        return shared(gate=gate, up=up)

    if args.input_layout == "separate":
        providers = {"pytorch": NativeSwiGLUOp(), "triton": shared}
    else:
        providers = {
            "pytorch": partial(NativeSwiGLUOp(), packed_order=packed_order),
            "adapter": adapter,
            "packed": partial(shared, packed_order=packed_order),
        }
    sources = [
        Path(__file__),
        ROOT / "rl_engine/backends/shared/triton/activation/swiglu.py",
        ROOT / "rl_engine/contracts/operators/activation.py",
        ROOT / "rl_engine/reference/activation/swiglu.py",
    ]
    report = {
        "hardware": torch.cuda.get_device_name(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "width": args.width,
        "method": METHOD,
        "settings": {k: v for k, v in vars(args).items() if k != "output_dir"},
        "case_plan": plan,
        "sources_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "complete": False,
        "checks": [],
        "results": [],
    }
    rng = random.Random(args.seed)
    try:
        for dtype_name in args.dtypes:
            for rows in args.rows:
                torch.manual_seed(args.seed)
                dtype = DTYPES[dtype_name]
                input_width = (2 if args.input_layout == "packed" else 1) * args.width
                inputs = tuple(
                    torch.randn((1, rows, input_width), device="cuda", dtype=dtype).requires_grad_(
                        True
                    )
                    for _ in range(plan["input_count"])
                )
                upstream = torch.randn((1, rows, args.width), device="cuda", dtype=dtype)
                print(f"Checking {dtype_name} M={rows}, I={args.width}", flush=True)
                if args.input_layout == "separate":
                    errors = check_separate_case(providers, inputs, upstream)
                else:
                    errors = check_case(providers, inputs[0], upstream, packed_order)
                report["checks"].append({"dtype": dtype_name, "rows": rows, **errors})
                for timing in args.timings:
                    for mode in MODES:
                        for round_id in range(args.rounds):
                            names = list(providers)
                            rng.shuffle(names)
                            for name in names:
                                gc.collect()
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                result = measure(
                                    providers[name],
                                    inputs,
                                    upstream,
                                    mode,
                                    timing,
                                    args,
                                )
                                report["results"].append(
                                    {
                                        "dtype": dtype_name,
                                        "rows": rows,
                                        "shape": list(inputs[0].shape),
                                        "timing": timing,
                                        "mode": mode,
                                        "provider": name,
                                        "round": round_id,
                                        **result,
                                    }
                                )
                        write_report(report, args.output_dir)
                        print(f"  {timing}/{mode} done", flush=True)
                del inputs, upstream
        report["complete"] = len(report["results"]) == plan["measurement_count"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report(report, args.output_dir)
    print(f"Reports saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
