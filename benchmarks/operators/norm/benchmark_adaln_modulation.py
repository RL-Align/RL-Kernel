# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare AdaLN backends on Qwen-Image image-token lengths.

Run with --real for S=4096/6889/6032, H=3072. These S values are derived
from 8x VAE downsampling and 2x2 latent packing, not specified by #386.
"""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def layer_norm_baseline(x, modulation, *, eps=1e-6):
    """Ordinary PyTorch math for accuracy/performance, without a replay claim."""
    shift, scale, _ = modulation.float().chunk(3, dim=-1)
    norm = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return (
        (norm * (1.0 + scale[:, None]) + shift[:, None]).to(x.dtype),
        modulation[:, None, 2 * x.shape[-1] :],
    )


def _bench(op, x, modulation, *, backward, warmup, repeat):
    def run():
        x.grad = modulation.grad = None
        if backward:
            y, gate = op(x, modulation)
            torch.autograd.backward((y, gate), (torch.ones_like(y), torch.ones_like(gate)))
        else:
            with torch.no_grad():
                op(x, modulation)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "allocated_before_samples_bytes": allocated_before,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="run all three Qwen-Image resolutions")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--json", action="store_true", help="emit one JSON report on stdout")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")
    return args


def _command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path):
    if not path or not Path(path).is_file():
        return None
    with open(path, "rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    # Imports and extension diagnostics must not corrupt the JSON stdout stream.
    with contextlib.redirect_stdout(sys.stderr):
        from rl_engine.runtime.adaln import SHARED_PROFILE, AdaLNDispatch
        from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp

        backends = [("pytorch", NativeAdaLNModulationOp())]
        skipped = []
        try:
            from rl_engine.backends.shared.triton.norm.adaln_modulation import (
                TritonAdaLNModulationOp,
            )

            backends.append(("triton", TritonAdaLNModulationOp()))
        except ImportError as exc:
            skipped.append({"backend": "triton", "reason": str(exc)})
        try:
            from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

            backends.append(("cuda", CudaAdaLNModulationOp()))
        except RuntimeError as exc:
            skipped.append({"backend": "cuda", "reason": str(exc)})
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = None
        environment = {
            "gpu": torch.cuda.get_device_name(),
            "device": str(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability()),
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "triton": triton_version,
            "driver_inventory": _command_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
            ),
            "source_revision": _command_output(["git", "rev-parse", "HEAD"]),
            "source_status": _command_output(["git", "status", "--porcelain=v1"]),
        }
        source_paths = [
            "benchmarks/operators/norm/benchmark_adaln_modulation.py",
            "rl_engine/runtime/adaln.py",
            "rl_engine/reference/norm/adaln_modulation.py",
            "rl_engine/backends/cuda/norm/adaln_modulation.py",
            "rl_engine/backends/shared/triton/norm/adaln_modulation.py",
            "rl_engine/ops/autograd/vjp_fp32.py",
            "csrc/cuda/norm/adaln_modulation.cu",
            "csrc/bindings/ops.cpp",
            "setup.py",
            "build_tools/extensions.py",
        ]
        environment["source_sha256"] = {path: _sha256(REPO_ROOT / path) for path in source_paths}
        backends = [
            (
                name,
                AdaLNDispatch(
                    op,
                    {
                        "requested_backend": name,
                        "fallback": False,
                        "fallback_reason": None,
                        "arithmetic_profile": SHARED_PROFILE,
                    },
                    hidden=3072,
                ),
            )
            for name, op in backends
        ]
        backends.append(("layer_norm_baseline", layer_norm_baseline))
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        results = []
        sizes = (4096, 6889, 6032) if args.real else (64,)
        for seq in sizes:
            x = torch.randn(
                1, seq, 3072, device="cuda", dtype=dtype, generator=generator, requires_grad=True
            )
            modulation = torch.randn(
                1, 9216, device="cuda", dtype=dtype, generator=generator, requires_grad=True
            )
            for name, op in backends:
                # Observe outside timing; retain the original bare-operator event interval.
                with torch.no_grad():
                    op(x, modulation)
                trace = (
                    dict(op.trace)
                    if isinstance(op, AdaLNDispatch)
                    else {
                        "actual_backend": "pytorch",
                        "arithmetic_profile": "ordinary-layer-norm",
                        "observation": "successful_forward",
                        "strict": False,
                        "profiler_observed": False,
                        "fallback": False,
                    }
                )
                trace["binary_sha256"] = _sha256(trace.get("binary_path"))
                timed_op = op.implementation if isinstance(op, AdaLNDispatch) else op
                for backward in (False, True):
                    measured = _bench(
                        timed_op,
                        x,
                        modulation,
                        backward=backward,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    results.append(
                        {
                            "backend": name,
                            "shape": list(x.shape),
                            "backward": backward,
                            "trace": trace,
                            **measured,
                        }
                    )
    report = {
        "schema_version": "adaln-benchmark-v1",
        "environment": environment,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timing": "CUDA events; forward or autograd forward+backward including ones_like upstreams",
        "skipped_backends": skipped,
        "results": results,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps({key: value for key, value in report.items() if key != "results"}))
        for result in results:
            print(
                f"S={result['shape'][1]} backend={result['backend']} "
                f"backward={result['backward']} median_ms={result['median_ms']:.3f} "
                f"peak_MB={result['peak_allocated_bytes'] / 2**20:.1f}"
            )


if __name__ == "__main__":
    main()
