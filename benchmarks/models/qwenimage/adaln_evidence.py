#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-weight accuracy, deterministic replay, and alternating CUDA measurements.

Run from the repository root with PYTHONPATH=. and a compiled local extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.models.qwenimage.streams import STREAMS
from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
    QwenImageAdaLNModProjectionCudaOp,
)
from rl_engine.backends.extension import _C
from rl_engine.contracts.numerical import load_contract
from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import (
    QwenImageAdaLNModProjectionReference,
    reference_projection,
)

CHECKPOINT_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
DIFFUSERS_REVISION = "d961a388fd02e4db38d17350c8dd9b8abe642e05"


def equal_bytes(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


QWENIMAGE_SOURCE_ROOTS = (
    "csrc/cuda/model_specific/qwenimage",
    "rl_engine/backends/cuda/model_specific/qwenimage",
    "rl_engine/reference/model_specific/qwenimage",
    "benchmarks/models/qwenimage",
)


def source_state():
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    hashes = {}
    for name in untracked:
        if not name:
            continue
        path = Path(name.decode())
        if path.suffix in (".py", ".cu", ".cpp", ".h", ".pyi"):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(diff + json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"])
    sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in QWENIMAGE_SOURCE_ROOTS
        for path in sorted(Path(root).rglob("*"))
        if path.suffix in (".py", ".cu") and "__pycache__" not in path.parts
    }
    return {
        "base_sha": base,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_source_sha256": hashes,
        "workspace_source_sha256": fingerprint,
        "qwenimage_source_sha256": sources,
        "extension_sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        "clean_commit": not status.strip(),
    }


def load_weights(model, stem):
    index = json.loads(
        (model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    tensors, manifest = {}, {}
    for suffix in ("weight", "bias"):
        name = f"{stem}.{suffix}"
        shard = index["weight_map"][name]
        metadata = model / ".cache/huggingface/download/transformer" / (shard + ".metadata")
        revision, etag, *_ = metadata.read_text().splitlines()
        if revision != CHECKPOINT_REVISION:
            raise ValueError(f"unverified local checkpoint revision: {revision}")
        with safe_open(model / "transformer" / shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        tensors[suffix] = tensor
        manifest[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_hash(tensor),
            "shard": shard,
            "download_etag": etag,
        }
    return tensors, manifest


def compare(actual, expected, dtype, judgment):
    spec = load_contract()["judgments"][judgment]["by_op_class"]["reduction"][
        str(dtype).split(".")[-1]
    ]
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    error = (a - b).abs()
    limit = spec["atol"] + spec["rtol"] * b.abs()
    return {
        "passed": bool(
            torch.isfinite(a).all() and torch.isfinite(b).all() and (error <= limit).all()
        ),
        "max_abs": error.max().item(),
        "max_tolerance_ratio": (error / limit).max().item(),
        "violations": (error > limit).sum().item(),
        "atol": spec["atol"],
        "rtol": spec["rtol"],
    }


def correctness(candidate, args, kind):
    dtype = args["temb"].dtype
    reference = reference_projection
    cpu = {k: v.detach().cpu().float().requires_grad_() for k, v in args.items()}
    y = candidate(**args)
    gold = reference(**cpu, execution_dtype=dtype)
    gen = torch.Generator(device=y.device).manual_seed(127)
    grad = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=gen)
    grads = torch.autograd.grad(y, tuple(args.values()), grad)
    gold_grads = torch.autograd.grad(gold, tuple(cpu.values()), grad.float().cpu())
    metrics = {
        "forward": compare(y, gold, dtype, "forward_accuracy"),
        "gradients": {
            k: compare(a, b, dtype, "gradient_accuracy") for k, a, b in zip(args, grads, gold_grads)
        },
    }
    again = candidate(**args)
    repeat = torch.autograd.grad(again, tuple(args.values()), grad)
    metrics["repeat_bitwise"] = bool(
        equal_bytes(y, again) and all(equal_bytes(a, b) for a, b in zip(grads, repeat))
    )
    del gold, gold_grads, cpu
    return metrics


def measure(function, args, mode):
    values = tuple(args.values())
    if mode == "backward":
        output = function(**args)
        upstream = torch.ones_like(output)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    if mode == "forward":
        with torch.no_grad():
            output = function(**args)
    elif mode == "backward":
        torch.autograd.grad(output, values, upstream)
    else:
        output = function(**args)
        torch.autograd.grad(output, values, torch.ones_like(output))
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def benchmark(functions, args, repeats):
    result = {}
    for mode in ("forward", "backward", "forward_backward"):
        for fn in functions.values():
            for _ in range(3):
                measure(fn, args, mode)
        samples = {name: [] for name in functions}
        orders = []
        for iteration in range(repeats):
            order = list(functions)
            if iteration % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                samples[name].append(measure(functions[name], args, mode))
        result[mode] = {
            "samples_ms": samples,
            "order": orders,
            "median_ms": {k: statistics.median(v) for k, v in samples.items()},
        }
    result["peak_increment_bytes"] = {}
    for name, fn in functions.items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        measure(fn, args, "forward_backward")
        result["peak_increment_bytes"][name] = torch.cuda.max_memory_allocated() - before
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    opts = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(71)
    report = {
        "checkpoint": "Qwen/Qwen-Image",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "diffusers_revision": DIFFUSERS_REVISION,
        "diffusers_source_sha256": {
            "transformer_qwenimage.py": (
                "0bad11901deae69f05aec01fd464e51e38582d455068f2851ee665036179321f"
            ),
            "normalization.py": "a3c40cee9308a0e339cca21c14bf51f394c61ae68e5b9a578e5636441f5eaa84",
        },
        "source": source_state(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0],
            "gpu": torch.cuda.get_device_name(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "weights": {},
        "cases": [],
    }
    for label, stem in STREAMS:
        weights, manifest = load_weights(opts.model, stem)
        report["weights"].update(manifest)
        kind = "projection"
        candidate = QwenImageAdaLNModProjectionCudaOp()
        native = QwenImageAdaLNModProjectionReference()
        for dtype in (torch.bfloat16, torch.float32):
            for batch, seq in [(1, 1), (2, 1), (4, 1), (9, 1)]:
                args = {
                    "temb": torch.randn(batch, 3072, device="cuda", dtype=dtype).requires_grad_(),
                    **{
                        k: v.to(device="cuda", dtype=dtype).requires_grad_()
                        for k, v in weights.items()
                    },
                }
                entry = {
                    "stream": label,
                    "dtype": str(dtype),
                    "batch": batch,
                    "seq": seq,
                    "readback": candidate.implementation_info(**args),
                    "correctness": correctness(candidate, args, kind),
                    "timing": benchmark(
                        {"candidate": candidate, "native": native}, args, opts.repeats
                    ),
                }
                report["cases"].append(entry)
                opts.output.parent.mkdir(parents=True, exist_ok=True)
                opts.output.write_text(json.dumps(report, indent=2) + "\n")
                print(label, str(dtype), batch, seq, entry["correctness"], flush=True)
                del args
    report["passed"] = all(
        c["correctness"]["forward"]["passed"]
        and c["correctness"]["repeat_bitwise"]
        and all(v["passed"] for v in c["correctness"]["gradients"].values())
        and all(
            v["passed"]
            for group in c["correctness"].get("fp64_diagnostic", {}).values()
            for v in group.values()
        )
        for c in report["cases"]
    )
    opts.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("accuracy gate failed; see report")


if __name__ == "__main__":
    main()
