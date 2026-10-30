from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_engine.mhc import oracle
from rl_engine.mhc.provider import ReferenceProvider

EPS = 1.0e-6
D = 4096


def torch_fwd(x, gamma, eps=EPS):
    x32 = x.float()
    r = torch.rsqrt((x32 * x32).sum(dim=1) / float(x.shape[1]) + eps)
    y = ((x32 * r[:, None]) * gamma.float()[None, :]).to(x.dtype)
    return y, x.clone(), {"r": r, "d": x.shape[1]}


def torch_bwd(dy, d_residual, x, gamma, saved):
    x32, dy32 = x.float(), dy.float()
    r, d = saved["r"], saved["d"]
    u = dy32 * gamma.float()[None, :]
    q = (u * x32).sum(dim=1)
    r3 = (r * r) * r
    dx_norm = (r[:, None] * u) - (((x32 * r3[:, None]) * q[:, None]) / float(d))
    dgamma = ((dy32 * x32) * r[:, None]).sum(dim=0)
    return dx_norm + d_residual.float(), dgamma


def load_backends():
    from rl_engine.kernels.ops.cuda.norm.rmsnorm_residual import (
        cuda_rmsnorm_residual_bwd,
        cuda_rmsnorm_residual_fwd,
    )
    from rl_engine.kernels.ops.triton.rmsnorm_residual_triton import (
        triton_rmsnorm_residual_bwd,
        triton_rmsnorm_residual_fwd,
    )

    return {
        "torch-native": (torch_fwd, torch_bwd),
        "triton": (triton_rmsnorm_residual_fwd, triton_rmsnorm_residual_bwd),
        "cuda": (cuda_rmsnorm_residual_fwd, cuda_rmsnorm_residual_bwd),
    }


def make_inputs(t, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed + t)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator).to("cuda", dtype)

    return rand(t, D), rand(D, dtype=torch.float32), rand(t, D), rand(t, D)


def collect_outputs(fwd, bwd, inputs):
    x, gamma, dy, d_residual = inputs
    y, residual, saved = fwd(x, gamma, EPS)
    if residual.data_ptr() == x.data_ptr():
        raise AssertionError("residual must be a copy, not an alias of x")
    if saved["d"] != x.shape[1]:
        raise AssertionError("saved D does not match x")
    dx, dgamma = bwd(dy, d_residual, x, gamma, saved)
    return {"y": y, "residual": residual, "r": saved["r"], "dx": dx, "dgamma": dgamma}


def bit_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def validate(backends, inputs):
    gold = collect_outputs(
        oracle.rmsnorm_residual_fwd, ReferenceProvider.rmsnorm_residual_bwd, inputs
    )
    checks = {}
    for (
        name,
        (fwd, bwd),
    ) in backends.items():
        got = collect_outputs(fwd, bwd, inputs)
        checks[name] = {}
        for key, want in gold.items():
            actual = got[key]
            if (actual.shape, actual.dtype, actual.device) != (
                want.shape,
                want.dtype,
                want.device,
            ):
                raise AssertionError(f"{name}.{key}: shape/dtype/device mismatch")
            if (
                not torch.isfinite(actual).all().item()
                or not torch.isfinite(want).all().item()
            ):
                raise AssertionError(f"{name}.{key}: non-finite output")
            equal = bit_equal(actual, want)
            checks[name][key] = {
                "bit_equal": equal,
                "max_abs": (actual.float() - want.float()).abs().max().item(),
            }
            if (name != "torch-native" or key == "residual") and not equal:
                raise AssertionError(f"{name}.{key}: raw bytes differ from oracle")
    return checks


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    mean_ms = start.elapsed_time(end) / iterations

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    del result
    return {"mean_ms": mean_ms, "peak_extra_allocated_bytes": extra}


def measure_backend(fwd, bwd, inputs, warmup, iterations):
    x, gamma, dy, d_residual = inputs
    y, residual, saved = fwd(x, gamma, EPS)
    del y, residual
    return measure(lambda: bwd(dy, d_residual, x, gamma, saved), warmup, iterations)


def benchmark_backend(fwd, bwd, inputs, warmup, iterations):
    x, gamma, dy, d_residual = inputs

    def forward_backward():
        y, residual, saved = fwd(x, gamma, EPS)
        dx, dgamma = bwd(dy, d_residual, x, gamma, saved)
        return y, residual, dx, dgamma

    forward = measure(lambda: fwd(x, gamma, EPS), warmup, iterations)
    backward = measure_backend(fwd, bwd, inputs, warmup, iterations)
    combined = measure(forward_backward, warmup, iterations)
    return {"forward": forward, "backward": backward, "forward_backward": combined}


def command_output(*args):
    try:
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def environment():
    import triton

    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "device_total_memory_bytes": props.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "nvcc": command_output("nvcc", "--version"),
        "gpu_layout": command_output("nvidia-smi", "-L"),
        "driver_and_mig": command_output(
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,mig.mode.current",
            "--format=csv",
        ),
    }


def print_table(rows):
    print(
        "| T | Backend | Forward ms | Backward ms | Fwd+Bwd ms | Extra MiB (F/B/FB) |"
    )
    print("|---:|---|---:|---:|---:|---|")
    for row in rows:
        modes = [row[key] for key in ("forward", "backward", "forward_backward")]
        f, b, fb = [mode["mean_ms"] for mode in modes]
        memory = "/".join(
            f'{mode["peak_extra_allocated_bytes"] / 2**20:.4f}' for mode in modes
        )
        print(
            f'| {row["T"]} | {row["backend"]} | {f:.6f} | {b:.6f} | {fb:.6f} | {memory} |'
        )


def main():
    parser = argparse.ArgumentParser(
        description="P1-5 strict candidates vs eager torch-native on NVIDIA CUDA"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.iterations < 1
        or any(t < 1 or t > 2**31 - 1 for t in args.tokens)
    ):
        parser.error("warmup/iterations must be positive; T must be in [1, INT_MAX]")
    if len(set(args.tokens)) != len(args.tokens):
        parser.error("tokens must not contain duplicates")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("NVIDIA CUDA is required; CPU/ROCm results are not accepted")
    torch.cuda.set_device(0)

    backends = load_backends()
    report = {
        "environment": environment(),
        "config": {
            "dtype": "bfloat16",
            "gamma_dtype": "float32",
            "gradient_dtype": "float32",
            "D": D,
            "eps": EPS,
            "tokens": args.tokens,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "backends": list(backends),
            "timing": "eager explicit fwd/bwd, CUDA Event mean, no CUDA Graph",
        },
        "validation": {},
        "results": [],
    }
    with torch.no_grad():
        for t in args.tokens:
            print(
                f"T={t}, D={D}: validating before timing", file=sys.stderr, flush=True
            )
            inputs = make_inputs(t, args.seed)
            report["validation"][str(t)] = validate(backends, inputs)
            for name, (fwd, bwd) in backends.items():
                result = benchmark_backend(
                    fwd, bwd, inputs, args.warmup, args.iterations
                )
                report["results"].append({"T": t, "backend": name, **result})
            del inputs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print_table(report["results"])
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
