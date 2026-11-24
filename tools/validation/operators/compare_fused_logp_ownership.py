#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare Python ownership refactoring against pinned main using the same CUDA extension."""

import argparse
import hashlib
import importlib.machinery
import json
import subprocess
from pathlib import Path
from types import ModuleType

import torch

from rl_engine.backends.cuda.logprob import logp
from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.runtime.registry import KernelRegistry

BASE = "e9d52e95226b62d826f0a224bafad2cab3ed7538"
ROOT = Path(__file__).resolve().parents[3]


def tensor_hash(value):
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    digest = tensor_hash(actual)
    assert digest == tensor_hash(expected), "baseline/refactor tensor bytes differ"
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "requires a real CUDA device"
    assert torch.version.hip is None, "requires NVIDIA CUDA, not HIP"
    assert torch.cuda.get_device_capability()[0] == 9, "requires Hopper"
    assert _EXT_AVAILABLE and hasattr(_C, "fused_logp"), "build the CUDA extension first"
    assert hasattr(_C, "fused_logp_backward"), "requires the merged #495 native backward"
    assert any(
        str(_C.__file__).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ), "requires a binary extension"
    assert (
        Path(_C.__file__).resolve().parent == ROOT / "rl_engine"
    ), "extension from another checkout"
    assert type(KernelRegistry().get_op("logp", device="cuda")) is logp.FusedLogpGenericOp
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:rl_engine/backends/cuda/logprob/logp.py"],
        cwd=ROOT,
        text=True,
    )
    baseline = ModuleType("logp_before_task19")
    exec(compile(source, "<main-before-task19>", "exec"), baseline.__dict__)
    old, new = baseline.FusedLogpGenericOp(), logp.FusedLogpGenericOp()
    assert old._backend is _C and new._backend is _C
    torch.manual_seed(19)
    cases = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in ((6, 257), (2, 3, 4097), (1, 2, 151936)):
            for strided in (False, True):
                storage_shape = (*shape[:-1], shape[-1] * (2 if strided else 1))
                x = torch.randn(storage_shape, device="cuda", dtype=dtype)
                if strided:
                    x = x[..., ::2]
                labels = torch.randint(shape[-1], shape[:-1], device="cuda", dtype=torch.int32)
                upstream = torch.randn(shape[:-1], device="cuda", dtype=dtype)
                outputs, gradients = [], []
                for op in (old, new):
                    logits = x.detach().requires_grad_()
                    output = op(logits, labels)
                    outputs.append(output)
                    gradients.append(torch.autograd.grad(output, logits, upstream)[0])
                case = {
                    "dtype": str(dtype),
                    "shape": shape,
                    "strided": strided,
                    "output_sha256": compare(outputs[1], outputs[0]),
                    "gradient_sha256": compare(gradients[1], gradients[0]),
                }
                reference_x = x.detach().float().requires_grad_()
                reference = (
                    reference_x.log_softmax(-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
                )
                reference_grad = torch.autograd.grad(reference, reference_x, upstream.float())[0]
                tolerance = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
                torch.testing.assert_close(
                    outputs[1].float(), reference, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    gradients[1].float(),
                    reference_grad.to(dtype).float(),
                    atol=tolerance,
                    rtol=tolerance,
                )
                indices = torch.tensor([0], device="cuda", dtype=torch.int32)
                variant_hashes = {}
                for method in (
                    "apply_fp32",
                    "out",
                    "indexed_out",
                    "indexed_fp32",
                    "online_out",
                    "online_fp32",
                    "online_indexed_out",
                    "online_indexed_fp32",
                ):
                    results = []
                    for op in (old, new):
                        call_args = [indices] if "indexed" in method else []
                        if method.endswith("out"):
                            output = torch.full(shape[:-1], 123.0, device="cuda", dtype=dtype)
                            call_args.append(output)
                        result = getattr(op, method)(x, labels, *call_args)
                        if method.endswith("out"):
                            assert result.data_ptr() == output.data_ptr()
                        results.append(result)
                    variant_hashes[method] = compare(results[1], results[0])
                case["variant_sha256"] = variant_hashes
                cases.append(case)
    torch.cuda.synchronize()
    extension_path = Path(_C.__file__)
    report = {
        "baseline": args.baseline,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "extension": str(extension_path),
        "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_python_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "python_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(logp.__file__),
                ROOT / "rl_engine/backends/musa/logprob/logp.py",
                ROOT / "rl_engine/ops/logprob/fused_logp.py",
            )
        },
        "backward": "native fused_logp_backward; FP32 upstream, input-dtype gradient",
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(cases)} CUDA cases; exact forward, VJP and eight variant comparisons")
    print(args.output)


if __name__ == "__main__":
    main()
