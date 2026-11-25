# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rl_engine.contracts.numerical import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.reference.guidance import NativeCfgCombineRenormOp  # noqa: E402


def shape_arg(value):
    dims = tuple(int(n) for n in value.lower().split("x"))
    if len(dims) != 3 or min(dims) < 1:
        raise argparse.ArgumentTypeError("expected positive BxSxD")
    return dims


def check_accuracy(native, candidate, inputs, dy, mode, contract):
    fp32 = [x.detach().float().requires_grad_() for x in inputs]
    gold = native.forward_fp32(*fp32, renorm_type=mode)
    gold_grads = torch.autograd.grad(gold, fp32, dy.float())
    output = candidate(*inputs, renorm_type=mode)
    grads = torch.autograd.grad(output, inputs, dy)
    errors = {}
    for name, actual, reference in zip(
        ("output", "d_main", "d_text", "d_img"), (output, *grads), (gold, *gold_grads), strict=True
    ):
        if actual.dtype != inputs[0].dtype:
            raise TypeError(f"{name}: expected {inputs[0].dtype}, got {actual.dtype}")
        judgment = "forward_accuracy" if name == "output" else "gradient_accuracy"
        tol = resolve_tolerance(
            contract, judgment=judgment, op_class="reduction", dtype=actual.dtype
        )
        torch.testing.assert_close(actual.float(), reference, atol=tol.atol, rtol=tol.rtol)
        errors[name] = (actual.float() - reference).abs().max().item()
    return errors


def measure(profiler, op, inputs, dy, mode, phase):
    if phase == "backward":
        output = op(*inputs, renorm_type=mode)

    def run():
        if phase == "forward":
            with torch.no_grad():
                op(*inputs, renorm_type=mode)
        elif phase == "backward":
            torch.autograd.grad(output, inputs, dy, retain_graph=True)
        else:
            torch.autograd.grad(op(*inputs, renorm_type=mode), inputs, dy)

    _, median_ms, _ = profiler._time_kernel(run)
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    run()
    torch.cuda.synchronize(profiler.device)
    peak = torch.cuda.max_memory_allocated(profiler.device)
    return {"median_ms": median_ms, "peak_bytes": peak, "extra_peak_bytes": peak - baseline}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BAGEL CFG native and Triton on NVIDIA CUDA"
    )
    parser.add_argument(
        "--shapes", type=shape_arg, nargs="+", default=[(1, 1024, 64), (2, 4096, 64)]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repeat) < 1:
        parser.error("--warmup and --repeat must both be positive")
    if torch.version.cuda is None or not torch.cuda.is_available():
        parser.error("this benchmark requires NVIDIA CUDA")

    import triton

    from benchmarks.common.profiler import PerformanceProfiler
    from rl_engine.backends.shared.triton.guidance import TritonCfgCombineRenormOp

    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
    native = NativeCfgCombineRenormOp()
    candidate = TritonCfgCombineRenormOp()
    rng = torch.Generator(device=device).manual_seed(435)
    contract = load_contract()
    rows = []
    lines = [
        "# CFG combine and renorm benchmark",
        "",
        f"GPU {torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}, "
        f"torch {torch.__version__}, "
        f"Triton {triton.__version__}. Three BF16 branches, scales 4.0 and 1.5, seed 435.",
        f"Median CUDA-event latency over {args.repeat} calls after {args.warmup} warmups. "
        "Extra peak allocation excludes inputs and the graph retained for backward-only timing.",
        "",
        "| Shape | Renorm | Phase | Native ms | Triton ms | Speedup | "
        "Native extra MiB | Triton extra MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for shape in args.shapes:
        inputs = [
            torch.randn(shape, device=device, dtype=torch.bfloat16, generator=rng).requires_grad_()
            for _ in range(3)
        ]
        dy = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=rng)
        for mode in ("global", "channel", "text_channel"):
            errors = check_accuracy(native, candidate, inputs, dy, mode, contract)
            for phase in ("forward", "backward", "forward_backward"):
                n = measure(profiler, native, inputs, dy, mode, phase)
                t = measure(profiler, candidate, inputs, dy, mode, phase)
                speedup = n["median_ms"] / t["median_ms"]
                rows.append(
                    dict(
                        shape=shape,
                        renorm_type=mode,
                        phase=phase,
                        native=n,
                        triton=t,
                        max_abs_errors=errors,
                        speedup=speedup,
                    )
                )
                lines.append(
                    f"| {'x'.join(map(str, shape))} | {mode} | {phase} | {n['median_ms']:.4f} | "
                    f"{t['median_ms']:.4f} | {speedup:.3f} | {n['extra_peak_bytes']/2**20:.3f} | "
                    f"{t['extra_peak_bytes']/2**20:.3f} |"
                )
    report = dict(
        gpu=torch.cuda.get_device_name(device),
        cuda=torch.version.cuda,
        torch=torch.__version__,
        triton=triton.__version__,
        seed=435,
        dtype="bfloat16",
        warmup=args.warmup,
        repeat=args.repeat,
        results=rows,
    )
    args.output_dir.mkdir(exist_ok=True, parents=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
