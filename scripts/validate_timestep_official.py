#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Official reduction-contract accuracy, invariance, observed kernels and timing.

Does not alter tolerances. JSON preserves failures; nonzero exit means failure.
Use --device cpu --backends pytorch for plumbing only, never GPU certification.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_engine.kernels.gtest import CandidateSpec, OperatorCase, run_operator_suite  # noqa: E402
from rl_engine.kernels.gtest.tolerance import BackendProvenance, load_contract, resolve_tolerance  # noqa: E402
from rl_engine.kernels.ops.timestep_embed_mlp import TimestepEmbedMLPOp  # noqa: E402
from rl_engine.kernels.ops.pytorch.timestep_embed_mlp import NativeTimestepEmbedMLPOp  # noqa: E402

NAMES = ("timestep", "weight1", "bias1", "weight2", "bias2")


def inputs(batch, hidden, dtype, device, seed):
    gen = torch.Generator().manual_seed(seed)
    shapes = ((batch,), (hidden, 256), (hidden,), (hidden, hidden), (hidden,))
    scales = (1, 16, 16, hidden**0.5, hidden**0.5)
    result = {}
    for name, shape, scale in zip(NAMES, shapes, scales, strict=True):
        x = torch.randn(shape, generator=gen) / scale
        if name != "timestep":
            x = x.to(dtype)
        result[name] = x.to(device)
    return result


def cloned(values):
    return {k: v.detach().clone().requires_grad_() for k, v in values.items()}


def run_grad(op, values, upstream, **kwargs):
    v = cloned(values)
    y = op(**v, **kwargs)
    g = torch.autograd.grad(y, tuple(v.values()), upstream)
    return [y.detach(), *[x.detach() for x in g]]


def bitwise(a, b):
    # torch.equal alone considers +0 and -0 equal. Compare actual storage bits.
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def invariance(op, values, dtype, seed):
    contract = load_contract()
    for judgment in ("forward_invariance", "gradient_invariance"):
        tol = resolve_tolerance(contract, judgment=judgment, op_class="reduction", dtype=dtype)
        assert tol.mode == "bitwise" and tol.atol == tol.rtol == 0
    t = values["timestep"]
    n, h = t.numel(), values["weight1"].shape[0]
    gen = torch.Generator().manual_seed(seed + 100)
    dy = torch.randn((n, h), generator=gen).to(device=t.device, dtype=dtype)
    base = run_grad(op, values, dy)
    comparisons = {}

    def compare(name, result):
        comparisons[name] = {
            label: {
                "bitwise": bitwise(a, b),
                "max_abs": (a.float() - b.float()).abs().max().item() if a.numel() else 0.0,
            }
            for label, a, b in zip(("output", *NAMES), base, result, strict=True)
        }

    compare("repeat", run_grad(op, values, dy))
    compare("chunk_1", run_grad(op, values, dy, chunk_size=1))
    compare("chunk_2", run_grad(op, values, dy, chunk_size=2))
    perm = torch.randperm(n, generator=gen).to(t.device)
    pv = dict(values, timestep=t[perm])
    result = run_grad(op, pv, dy[perm], sample_ids=perm)
    result[0], result[1] = result[0][torch.argsort(perm)], result[1][torch.argsort(perm)]
    compare("permuted_canonical_ids", result)
    pt = torch.cat((t[:1] * 0 + 19, t, t[:1] * 0 - 3))
    pv = dict(values, timestep=pt)
    mask = torch.ones(n + 2, dtype=torch.bool, device=t.device)
    mask[0] = mask[-1] = False
    ids = torch.arange(n + 2, device=t.device)
    pdy = torch.cat((dy[:1] * 0 + 1, dy, dy[:1] * 0 + 1))
    result = run_grad(op, pv, pdy, active_mask=mask, sample_ids=ids, chunk_size=2)
    padding_zero = bool((result[0][~mask] == 0).all() and (result[1][~mask] == 0).all())
    result[0], result[1] = result[0][mask], result[1][mask]
    compare("padding", result)
    strided = {}
    for name, x in values.items():
        buf = x.new_empty((*x.shape, 2))
        buf[..., 0] = x
        strided[name] = buf[..., 0]
    # run_grad clone preserves ordinary noncontiguous matrices only when dense;
    # exercise actual strides directly and make them leaves here.
    sv = {k: v.detach().requires_grad_() for k, v in strided.items()}
    sy = op(**sv)
    sg = torch.autograd.grad(sy, tuple(sv.values()), dy)
    compare("strided", [sy.detach(), *sg])
    # Row-local singleton semantics: parameter gradients deliberately excluded.
    singleton = []
    singleton_dt = []
    for i in range(n):
        r = run_grad(op, dict(values, timestep=t[i : i + 1]), dy[i : i + 1])
        singleton.append(r[0])
        singleton_dt.append(r[1])
    comparisons["singleton_rows"] = {
        "output": {"bitwise": bitwise(base[0], torch.cat(singleton))},
        "timestep": {"bitwise": bitwise(base[1], torch.cat(singleton_dt))},
    }
    passed = padding_zero and all(
        x["bitwise"] for rows in comparisons.values() for x in rows.values()
    )
    return {
        "passed": passed,
        "padding_zero": padding_zero,
        "comparisons": comparisons,
        "parameter_scope": (
            "same complete active sample set in canonical ID order; "
            "no external microbatch .grad sum"
        ),
    }


def observed_trace(op, values, output):
    # Warm compilation first, then require real CUDA events, not launch labels.
    v = cloned(values)
    y, trace = op(**v, return_trace=True)
    torch.autograd.grad(y, tuple(v.values()), torch.ones_like(y))
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        v = cloned(values)
        y, trace = op(**v, return_trace=True)
        torch.autograd.grad(y, tuple(v.values()), torch.ones_like(y))
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output))
    kernels = sorted(
        {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    )
    required = ("timestep_mm", "timestep_embedding", "timestep_silu", "timestep_dt")
    passed = all(any(r in k for k in kernels) for r in required)
    return {
        "passed": passed,
        "observed_cuda_kernels": kernels,
        "launch_record": asdict(trace),
        "chrome_trace": output.name,
    }


def eager_batched(timestep, weight1, bias1, weight2, bias2):
    """Conventional batched FP32 compute baseline, independent of fixed-order kernels."""
    frequency = torch.exp(-math.log(10000.0) * torch.arange(128, dtype=torch.float32) / 128)
    phase = (timestep.float()[:, None] * frequency.to(timestep.device)) * 1000.0
    e = torch.cat((phase.cos(), phase.sin()), dim=1)
    z = torch.nn.functional.linear(e, weight1.float(), bias1.float())
    h = torch.nn.functional.silu(z)
    return torch.nn.functional.linear(h, weight2.float(), bias2.float()).to(weight1.dtype)


def benchmark(op, values, iterations):
    dy = torch.ones(
        (values["timestep"].numel(), values["weight1"].shape[0]),
        device=values["timestep"].device,
        dtype=values["weight1"].dtype,
    )
    timings = {}
    for name in ("forward", "forward_backward"):
        v = cloned(values) if name == "forward_backward" else values

        def call(v=v, name=name):
            y = op(**v)
            if name == "forward_backward":
                torch.autograd.grad(y, tuple(v.values()), dy)

        for _ in range(3):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        timings[name] = {"median_ms": sorted(samples)[len(samples) // 2], "samples_ms": samples}
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--backends", nargs="+", default=["cuda", "triton"])
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 3, 16])
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--seeds", nargs="+", type=int, default=[386, 9386])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "args": vars(args).copy(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "contract_sha256": hashlib.sha256(
                (ROOT / "rl_engine/kernels/gtest/tolerance_contract.json").read_bytes()
            ).hexdigest(),
            "base_sha": "32b765ec992cd1206517104ec66506881203c91c",
            "tf32": False,
        },
        "source_sha256": {},
        "cases": [],
        "passed": False,
    }
    paths = [
        *ROOT.glob("rl_engine/kernels/ops/**/timestep_embed_mlp.py"),
        ROOT / "csrc/cuda/timestep_embed_mlp.cu",
        Path(__file__).resolve(),
        ROOT / "rl_engine/kernels/gtest/operator_specs.py",
        ROOT / "rl_engine/kernels/gtest/operator_inputs.py",
    ]
    for p in paths:
        report["source_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    save()
    gold = NativeTimestepEmbedMLPOp()
    for backend in args.backends:
        op = TimestepEmbedMLPOp(backend)
        for dtype_name in args.dtypes:
            dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
            for batch in args.batches:
                for seed in args.seeds:
                    start = time.perf_counter()
                    row = {
                        "backend": backend,
                        "dtype": dtype_name,
                        "batch": batch,
                        "hidden": args.hidden,
                        "seed": seed,
                        "passed": False,
                    }
                    report["cases"].append(row)
                    try:
                        values = inputs(batch, args.hidden, dtype, args.device, seed)

                        def cpu_gold(**kw):
                            return gold.forward_fp32(**{k: v.cpu() for k, v in kw.items()}).to(
                                args.device
                            )

                        provenance = None
                        if dtype == torch.bfloat16 and backend != "pytorch":
                            provenance = BackendProvenance(
                                backend_profile="cuda_bf16"
                                if backend == "cuda"
                                else "triton_cuda_bf16",
                                requested_backend=backend,
                                actual_backend=backend,
                                execution_dtype="bfloat16",
                                accumulation_dtype="float32",
                                output_dtype="bfloat16",
                                reference_dtype="float32",
                                candidate_tf32_enabled=False,
                                reference_tf32_enabled=False,
                            )
                        case = OperatorCase(
                            "timestep-cpu-gold", "reduction", dtype, values, cpu_gold, NAMES
                        )
                        suite = run_operator_suite(
                            "timestep_embed_mlp",
                            candidates=[
                                CandidateSpec(backend, op, backend=backend, provenance=provenance)
                            ],
                            cases=[case],
                            check_grad=True,
                            grad_seed=seed + 1,
                        )
                        row["accuracy"] = suite.to_dict()
                        row["launch_record"] = asdict(op.last_trace)
                        if backend != "pytorch":
                            assert op.last_trace.actual_backend == backend
                            assert op.last_trace.fallback_reason is None
                            row["invariance"] = invariance(op, values, dtype, seed)
                        if args.trace and batch == args.batches[0] and seed == args.seeds[0]:
                            trace_path = args.output.with_name(
                                f"{args.output.stem}-trace-{backend}-{dtype_name}.json"
                            )
                            row["trace"] = observed_trace(op, values, trace_path)
                        if args.benchmark and seed == args.seeds[0]:
                            row["timing"] = benchmark(op, values, args.iterations)
                            row["native_batched_timing"] = benchmark(
                                eager_batched, values, args.iterations
                            )
                        row["passed"] = (
                            suite.passed
                            and row.get("invariance", {}).get("passed", True)
                            and row.get("trace", {}).get("passed", True)
                        )
                    except Exception as exc:
                        import traceback

                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["traceback"] = traceback.format_exc()
                    row["elapsed_seconds"] = time.perf_counter() - start
                    save()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "backend",
                                    "dtype",
                                    "batch",
                                    "seed",
                                    "passed",
                                    "elapsed_seconds",
                                )
                            }
                        ),
                        flush=True,
                    )
    report["passed"] = bool(report["cases"]) and all(r["passed"] for r in report["cases"])
    save()
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
