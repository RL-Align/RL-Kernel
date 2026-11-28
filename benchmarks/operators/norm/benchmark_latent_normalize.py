# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU-reference accuracy and CUDA timing evidence for issue #386.

Run from the checkout: PYTHONPATH=. python benchmarks/operators/norm/benchmark_latent_normalize.py
"""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch

from rl_engine.backends.cuda.norm.latent_normalize import (
    LatentDenormalizeCudaOp,
    LatentNormalizeCudaOp,
)
from rl_engine.backends.extension import _C
from rl_engine.contracts.numerical import BackendProvenance
from rl_engine.reference.norm.latent_normalize import (
    NativeLatentDenormalizeOp,
    NativeLatentNormalizeOp,
)
from rl_engine.validation.operators.op_checks import CandidateSpec, OperatorCase, run_operator_suite

ROOT = Path(__file__).resolve().parents[3]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def byte_mismatches(a, b):
    return int(
        (
            a.detach().cpu().contiguous().view(torch.uint8)
            != b.detach().cpu().contiguous().view(torch.uint8)
        ).sum()
    )


def latency_us(fn, iterations):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/latent-normalize/evidence.json")
    )
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    shapes = [(2, 16, 3, 7), (1, 16, 1, 128, 128), (1, 16, 1, 166, 166), (1, 16, 1, 208, 116)]
    for dtype in (torch.float32, torch.bfloat16):
        for inverse in (False, True):
            op = LatentDenormalizeCudaOp() if inverse else LatentNormalizeCudaOp()
            ref = NativeLatentDenormalizeOp() if inverse else NativeLatentNormalizeOp()
            for shape in shapes:
                generator = torch.Generator().manual_seed(386)
                cpu_x = torch.randn(shape, generator=generator).to(dtype).float().requires_grad_()
                mean = torch.linspace(-1.1, 1.7, 16)
                scale = torch.linspace(0.31, 0.89, 16)
                grad = torch.randn(shape, generator=generator).to(dtype)
                gold = ref.forward_fp32(cpu_x, mean, scale)
                gold_dx = torch.autograd.grad(gold, cpu_x, grad.float())[0]
                x = cpu_x.detach().to(device="cuda", dtype=dtype).requires_grad_()
                mu, si, dy = (v.cuda() for v in (mean, scale, grad))
                y = op(x, mu, si)
                dx = torch.autograd.grad(y, x, dy, retain_graph=True)[0]
                mismatch_fwd = byte_mismatches(y, gold.to(dtype))
                mismatch_bwd = byte_mismatches(dx, gold_dx.to(dtype))
                if mismatch_fwd or mismatch_bwd:
                    raise AssertionError(f"CPU arithmetic mismatch: {shape}, {dtype}, {inverse}")

                # The official harness checks the unrounded FP32 reference using
                # the shared accuracy thresholds, independently of raw-byte checks.
                def gold_fn(x, mean, std_inv, _ref=ref):
                    return _ref.forward_fp32(x.float().cpu(), mean.cpu(), std_inv.cpu()).to(
                        x.device
                    )

                provenance = None
                if dtype == torch.bfloat16:
                    provenance = BackendProvenance(
                        backend_profile="cuda_bf16",
                        requested_backend="cuda",
                        actual_backend="cuda",
                        execution_dtype="bfloat16",
                        accumulation_dtype="float32",
                        output_dtype="bfloat16",
                        reference_dtype="float32",
                        candidate_tf32_enabled=False,
                        reference_tf32_enabled=False,
                    )
                accuracy = run_operator_suite(
                    "latent_denormalize" if inverse else "latent_normalize",
                    candidates=[
                        CandidateSpec(name="cuda", fn=op, backend="cuda", provenance=provenance)
                    ],
                    cases=[
                        OperatorCase(
                            name=str(shape),
                            op_class="elementwise",
                            dtype=dtype,
                            inputs={"x": x, "mean": mu, "std_inv": si},
                            gold_fn=gold_fn,
                            grad_input_names=("x",),
                        )
                    ],
                    check_grad=True,
                )
                if not accuracy.passed:
                    raise AssertionError(accuracy.to_dict())
                native_us = latency_us(
                    partial(_C.latent_normalize_forward, x, mu, si, inverse, 256), args.iterations
                )
                cuda_backward_us = latency_us(
                    partial(torch.autograd.grad, y, x, dy, retain_graph=True), args.iterations
                )
                ref_y = ref(x, mu, si)
                eager_backward_us = latency_us(
                    partial(torch.autograd.grad, ref_y, x, dy, retain_graph=True), args.iterations
                )
                with torch.no_grad():
                    cuda_us = latency_us(partial(op, x, mu, si), args.iterations)
                    eager_us = latency_us(partial(ref, x, mu, si), args.iterations)
                records.append(
                    {
                        "shape": shape,
                        "dtype": str(dtype),
                        "inverse": inverse,
                        "forward_byte_mismatches": mismatch_fwd,
                        "backward_byte_mismatches": mismatch_bwd,
                        "trace": dict(op.last_trace),
                        "accuracy": accuracy.to_dict(),
                        "cuda_forward_us": cuda_us,
                        "eager_forward_us": eager_us,
                        "native_forward_us": native_us,
                        "cuda_backward_us": cuda_backward_us,
                        "eager_backward_us": eager_backward_us,
                        "forward_speedup": eager_us / cuda_us,
                    }
                )
    source_paths = [
        "csrc/cuda/norm/latent_normalize.cu",
        "rl_engine/backends/cuda/norm/latent_normalize.py",
        "rl_engine/reference/norm/latent_normalize.py",
        "rl_engine/ops/norm/latent_normalize.py",
    ]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "nvcc": command("/usr/local/cuda/bin/nvcc", "--version"),
        "base_commit": command("git", "rev-parse", "HEAD"),
        "working_tree": command("git", "status", "--short"),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths
        },
        "seed": 386,
        "statistics": "synthetic FP32 channel vectors; not a full-model fixture",
        "timing": {
            "warmup": 20,
            "groups": 5,
            "iterations_per_group": args.iterations,
            "aggregation": "median of CUDA-event group means; includes launch overhead",
            "backward": "autograd.grad on retained graph; forward excluded",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(records)} forward/backward cases; evidence: {args.output}")
    for record in records:
        print(
            f"{record['dtype']} inverse={record['inverse']} {record['shape']}: "
            f"CUDA {record['cuda_forward_us']:.2f} us, "
            f"eager {record['eager_forward_us']:.2f} us, "
            f"speedup {record['forward_speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
