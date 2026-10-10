# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the Qwen-Image joint-attention softmax backends.

The default key lengths are the three issue #386 image shapes after VAE stride
8, 2x2 latent packing, and 512 text positions:

* 1024x1024 -> 4096 image + 512 text = 4608 keys
* 1328x1328 -> 6889 image + 512 text = 7401 keys
* 1664x928  -> 6032 image + 512 text = 6544 keys

Examples:
    python benchmarks/benchmark_joint_attn_softmax.py
    python benchmarks/benchmark_joint_attn_softmax.py --backward
    python benchmarks/benchmark_joint_attn_softmax.py --valid-text-keys 73
    python benchmarks/benchmark_joint_attn_softmax.py --rows 24576 --backends cuda,triton

``--backward`` times forward plus ``torch.autograd.grad``, not an isolated
backward kernel.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

# Allow direct execution from a source checkout without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import (  # noqa: E402
    JointAttnSoftmaxCudaOp,
)
from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import (  # noqa: E402
    NativeJointAttnSoftmaxOp,
)
from rl_engine.testing.bitwise import tensor_bytes_equal  # noqa: E402

DEFAULT_CASES = {
    "1024x1024": 4608,
    "1328x1328": 7401,
    "1664x928": 6544,
}
_TEXT_KEY_SLOTS = 512


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _parse_cases(raw: str | None) -> dict[str, int]:
    if raw is None:
        return dict(DEFAULT_CASES)
    cases: dict[str, int] = {}
    for item in raw.split(";"):
        name, key_length = item.split(",", maxsplit=1)
        parsed_length = int(key_length)
        if not name.strip() or parsed_length <= 0:
            raise ValueError("cases must use non-empty '<name>,<positive keys>' entries")
        cases[name.strip()] = parsed_length
    return cases


def _load_backends(names: list[str]) -> dict[str, object]:
    factories = {
        "cuda": JointAttnSoftmaxCudaOp,
        "pytorch": NativeJointAttnSoftmaxOp,
    }
    unknown = sorted(set(names) - {"cuda", "triton", "pytorch"})
    if unknown:
        raise ValueError(f"unknown backends: {unknown}")
    if "triton" in names:
        from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import (
            TritonJointAttnSoftmaxOp,
        )

        factories["triton"] = TritonJointAttnSoftmaxOp
    return {name: factories[name]() for name in names}


def _time_cuda(call: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _forward_call(
    op: object,
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    with torch.no_grad():
        return op.forward(  # type: ignore[attr-defined]
            scores,
            key_padding_mask=key_padding_mask,
        )


def _backward_call(
    op: object,
    scores: torch.Tensor,
    grad_output: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    differentiable_scores = scores.detach().requires_grad_(True)
    probabilities = op.forward(  # type: ignore[attr-defined]
        differentiable_scores,
        key_padding_mask=key_padding_mask,
    )
    (grad_scores,) = torch.autograd.grad(probabilities, differentiable_scores, grad_output)
    return grad_scores


def _environment() -> dict[str, Any]:
    try:
        triton_version = version("triton")
    except PackageNotFoundError:
        triton_version = None

    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "triton": triton_version,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability()),
        "git_commit": _git_commit(),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("joint_attn_softmax benchmark requires an NVIDIA CUDA GPU")
    if args.rows <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("rows and iterations must be positive; warmup must be non-negative")
    if args.valid_text_keys is not None and not 0 <= args.valid_text_keys <= _TEXT_KEY_SLOTS:
        raise ValueError(f"valid-text-keys must be between 0 and {_TEXT_KEY_SLOTS}")

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    backends = _load_backends(args.backends)
    if "cuda" not in backends:
        raise ValueError("the CUDA bit-reference backend must be included")

    records: list[dict[str, Any]] = []
    for shape_name, key_length in args.cases.items():
        if args.valid_text_keys is not None and key_length < _TEXT_KEY_SLOTS:
            raise ValueError(
                f"{shape_name} has {key_length} keys; prompt masking requires at least "
                f"{_TEXT_KEY_SLOTS}"
            )
        score_shape = (1, args.rows, 1, key_length)
        generator = torch.Generator(device="cuda").manual_seed(386 + key_length + args.rows)
        scores = torch.randn(
            score_shape,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        grad_output = torch.randn(
            score_shape,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        key_padding_mask = None
        masked_prompt_keys = 0
        if args.valid_text_keys is not None:
            key_padding_mask = torch.ones((1, key_length), device="cuda", dtype=torch.bool)
            key_padding_mask[:, args.valid_text_keys : _TEXT_KEY_SLOTS] = False
            masked_prompt_keys = _TEXT_KEY_SLOTS - args.valid_text_keys
        reference = (
            _backward_call(backends["cuda"], scores, grad_output, key_padding_mask)
            if args.backward
            else _forward_call(backends["cuda"], scores, key_padding_mask)
        )

        for backend_name, op in backends.items():
            call: Callable[[], torch.Tensor]
            if args.backward:
                call = partial(
                    _backward_call,
                    op,
                    scores,
                    grad_output,
                    key_padding_mask,
                )
            else:
                call = partial(_forward_call, op, scores, key_padding_mask)
            actual = call()
            if not tensor_bytes_equal(actual, reference):
                raise AssertionError(
                    f"{backend_name} differs from CUDA for {shape_name}: bytes or metadata differ"
                )

            latency_ms = _time_cuda(call, args.warmup, args.iterations)
            fingerprint = op.provenance["kernel_fingerprint"]  # type: ignore[attr-defined]
            records.append(
                {
                    "image_shape": shape_name,
                    "rows": args.rows,
                    "keys": key_length,
                    "dtype": args.dtype,
                    "direction": "backward" if args.backward else "forward",
                    "backend": backend_name,
                    "valid_text_keys": args.valid_text_keys,
                    "masked_prompt_keys": masked_prompt_keys,
                    "latency_ms": latency_ms,
                    "cuda_speed_ratio": None,
                    "kernel_fingerprint": fingerprint,
                    "byte_equal_to_cuda": True,
                }
            )

    cuda_latencies = {
        (record["image_shape"], record["direction"]): record["latency_ms"]
        for record in records
        if record["backend"] == "cuda"
    }
    for record in records:
        baseline = cuda_latencies[(record["image_shape"], record["direction"])]
        record["cuda_speed_ratio"] = baseline / record["latency_ms"]

    return {
        "schema_version": "rlkernel.joint_attn_softmax_benchmark.v3",
        "environment": _environment(),
        "configuration": {
            "rows": args.rows,
            "dtype": args.dtype,
            "direction": "backward" if args.backward else "forward",
            "valid_text_keys": args.valid_text_keys,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=24, help="flattened B * H * Q rows")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--backward", action="store_true", help="time forward plus autograd.grad")
    parser.add_argument(
        "--valid-text-keys",
        type=int,
        help="mask prompt key slots from this count up to position 512",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--backends",
        type=lambda raw: [item.strip() for item in raw.split(",") if item.strip()],
        default=["cuda", "triton", "pytorch"],
        help="comma-separated subset of cuda,triton,pytorch (CUDA is required)",
    )
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=None,
        help="semicolon-separated '<name>,<keys>' entries",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.cases = _parse_cases(None) if args.cases is None else args.cases
    return args


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
