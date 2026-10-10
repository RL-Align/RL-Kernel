#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fail-closed H3 GPU qualification with JSON evidence and performance samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rl_engine.kernels.gtest.tolerance import load_contract  # noqa: E402
from rl_engine.kernels.ops import base  # noqa: E402
from rl_engine.kernels.ops.h3_down_validation import (  # noqa: E402
    comparison,
    logical_keys,
    reference_projection,
    training_projection,
)
from rl_engine.kernels.ops.h3_ffn_down import (  # noqa: E402
    H3_CHECKPOINT_REVISION,
    H3_DOWN_BACKEND,
    H3_LEGACY_DOWN_BACKEND,
    H3_MAX_ROWS,
    H3FFNDownGemmOp,
    _require_backend,
    _weight_gradient,
)


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def provenance(args):
    contract_path = ROOT / "rl_engine/kernels/gtest/tolerance_contract.json"
    result = {
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--porcelain"]),
        "git_diff_sha256": hashlib.sha256(command(["git", "diff", "HEAD"]).encode()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "contract_version": load_contract()["version"],
        "contract_sha256": sha256(contract_path),
        "checkpoint_revision": H3_CHECKPOINT_REVISION,
        "args": vars(args),
        "environment": {
            name: os.getenv(name)
            for name in (
                "KERNEL_ALIGN_USE_FAST_MATH",
                "KERNEL_ALIGN_DET_GEMM_SM90",
                "RL_KERNEL_DET_GEMM_BACKEND",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUBLASLT_WORKSPACE_SIZE",
                "TORCH_CUDA_ARCH_LIST",
            )
        },
        "source_hashes": {},
        "build_flags_verified_at_runtime": False,
    }
    for name in (
        "csrc/cuda/gemm/det_gemm_kernel.cu",
        "csrc/cuda/gemm/det_gemm_tma.cuh",
        "rl_engine/kernels/ops/h3_ffn_down.py",
        "rl_engine/kernels/ops/h3_down_fp32_backend.py",
        "rl_engine/kernels/ops/triton/matmul/det_gemm.py",
        "rl_engine/kernels/ops/h3_down_validation.py",
        "rl_engine/kernels/ops/canonical_backward.py",
        "rl_engine/kernels/registry.py",
        "scripts/validate_h3_ffn_down.py",
    ):
        result["source_hashes"][name] = sha256(ROOT / name)
    if base._EXT_AVAILABLE:
        result["extension_sha256"] = sha256(base._C.__file__)
    if args.build_log:
        result["build_log_sha256"] = sha256(args.build_log)
    return result


def timed(fn, repeats):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    # Wall clock includes host-side canonical-key validation, copies and reduction prep.
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del value
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - base_memory,
        "timing_scope": "synchronized wall time, includes wrapper/copies/padding",
    }


def add_checks(report, actual, expected, transform, *, accuracy=False):
    for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
        judgment = "forward_" if i == 0 else "gradient_"
        judgment += "accuracy" if accuracy else "invariance"
        row = comparison(left, right, judgment)
        row.update(tensor=("y", "dx", "dw")[i], transform=transform)
        report["checks"].append(row)


def row_local(op, x, w, dy):
    x = x.detach().clone().requires_grad_(True)
    y = op(x, w.detach())
    (dx,) = torch.autograd.grad(y, x, dy)
    return y.detach(), dx.detach()


def vllm_comparison(x, w, dy, reference, repeats):
    """Optional forward and explicit VJP prior-art probe, without a vLLM install.

    This probe is not an autograd integration or a weight-gradient chunking claim.
    """
    result = {"status": "running", "checks": [], "benchmarks": {}}
    try:
        import triton

        from benchmarks.vllm_batch_invariant_matmul import (
            VLLM_BATCH_INVARIANT_SOURCE_SHA,
            matmul_config_metadata,
            matmul_persistent,
        )

        result.update(
            source_commit=VLLM_BATCH_INVARIANT_SOURCE_SHA,
            triton_version=triton.__version__,
            source_sha256=sha256(ROOT / "benchmarks/vllm_batch_invariant_matmul.py"),
            configs_sha256=sha256(ROOT / "benchmarks/vllm_batch_invariant_configs.py"),
        )
        pairs = ((x, w.t()), (dy, w), (dy.t(), x))
        result["configs"] = [
            matmul_config_metadata(a.shape[0], b.shape[1], a.shape[1], a.dtype) for a, b in pairs
        ]
        actual = tuple(matmul_persistent(a, b) for a, b in pairs)
        add_checks(result, actual, reference, "fp32_reference", accuracy=True)
        for i, (a, b) in enumerate(pairs):
            result["benchmarks"][("forward", "dx", "dw")[i]] = timed(
                lambda a=a, b=b: matmul_persistent(a, b), repeats
            )
            if i < 2:
                chunked = torch.cat(
                    [matmul_persistent(a[lo : lo + 31], b) for lo in range(0, a.shape[0], 31)]
                )
                judgment = "forward_invariance" if i == 0 else "gradient_invariance"
                for name, value, expected in (
                    ("chunk31", chunked, actual[i]),
                    ("singleton", matmul_persistent(a[:1], b), actual[i][:1]),
                ):
                    check = comparison(value, expected, judgment)
                    check.update(tensor=("y", "dx")[i], transform=name)
                    result["checks"].append(check)
        result["status"] = (
            "passed_probe" if all(row["passed"] for row in result["checks"]) else "failed_probe"
        )
        result["qualification"] = "partial_prior_art_probe_only"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    return result


def run_case(x, w, dy, label, repeats, *, compare_vllm=False, backend=H3_DOWN_BACKEND):
    rows = x.shape[0]
    keys = logical_keys(rows, x.device)
    op = H3FFNDownGemmOp(backend=backend)
    case = {"rows": rows, "input_family": label, "checks": [], "benchmarks": {}}
    canonical, traces = training_projection(op, x, w, dy, keys)
    case["execution_traces"] = traces
    case["output_digests"] = {
        name: tensor_digest(value) for name, value in zip(("y", "dx", "dw"), canonical, strict=True)
    }
    reference = reference_projection(x, w, dy)
    add_checks(case, canonical, reference, "fp32_reference", accuracy=True)
    if compare_vllm:
        case["vllm_comparison"] = vllm_comparison(x, w, dy, reference, repeats)
    del reference
    repeat, _ = training_projection(op, x, w, dy, keys)
    add_checks(case, repeat, canonical, "repeat")
    del repeat
    order = torch.arange(rows - 1, -1, -1, device=x.device)
    inverse = torch.argsort(order)
    permuted, _ = training_projection(op, x[order], w, dy[order], keys[order])
    add_checks(
        case,
        (permuted[0][inverse], permuted[1][inverse], permuted[2]),
        canonical,
        "permuted_logical_rows",
    )
    del permuted
    chunked, _ = training_projection(op, x, w, dy, keys, chunk_size=31)
    add_checks(case, chunked, canonical, "chunk31_single_backward")
    del chunked
    # Force non-contiguous activations AND native-shaped non-contiguous weights.
    layout_x, layout_w = x.t().contiguous().t(), w.t().contiguous().t()
    layout, _ = training_projection(op, layout_x, layout_w, dy, keys)
    add_checks(case, layout, canonical, "strided_input_and_weight")
    del layout, layout_x, layout_w
    pad = 17
    padded_x = torch.cat((x, torch.ones(pad, 14336, device=x.device, dtype=x.dtype)))
    padded_dy = torch.cat((dy, torch.zeros(pad, 5376, device=x.device, dtype=dy.dtype)))
    padded_keys = torch.cat((keys, torch.full((pad, 2), -1, device=x.device, dtype=torch.int64)))
    if rows + pad <= H3_MAX_ROWS:
        padded, _ = training_projection(op, padded_x, w, padded_dy, padded_keys)
        add_checks(
            case, (padded[0][:rows], padded[1][:rows], padded[2]), canonical, "inactive_padding"
        )
        del padded
    del padded_x, padded_dy, padded_keys
    if rows % 2 == 0:
        batched, _ = training_projection(
            op, x.reshape(2, rows // 2, 14336), w, dy.reshape(2, rows // 2, 5376), keys
        )
        add_checks(
            case,
            (batched[0].reshape(rows, 5376), batched[1].reshape(rows, 14336), batched[2]),
            canonical,
            "batch2_reshape",
        )
        del batched
    # Compare only row-local tensors when the effective logical row set changes.
    one = row_local(op, x[:1], w, dy[:1])
    mutated_x, mutated_dy = x.clone(), dy.clone()
    mutated_x[1:].neg_()
    mutated_dy[1:].zero_()
    mutated = row_local(op, mutated_x, w, mutated_dy)
    for i in (0, 1):
        for name, value in (
            ("singleton_vs_batch", canonical[i][:1]),
            ("other_rows_mutated", mutated[i][:1]),
        ):
            check = comparison(
                value, one[i], "forward_invariance" if i == 0 else "gradient_invariance"
            )
            check.update(tensor=("y", "dx")[i], transform=name)
            case["checks"].append(check)
    del one, mutated, mutated_x, mutated_dy, canonical
    extension = _require_backend(x.device, backend)
    case["benchmarks"]["candidate_forward"] = timed(lambda: op(x, w), repeats)
    case["benchmarks"]["candidate_dx"] = timed(
        lambda: extension.det_gemm_fwd(dy.contiguous(), w.contiguous()), repeats
    )
    case["benchmarks"]["candidate_dw_canonical_order"] = timed(
        lambda: _weight_gradient(extension, x, dy), repeats
    )
    case["benchmarks"]["candidate_forward_backward"] = timed(
        lambda: training_projection(op, x, w, dy, keys), repeats
    )
    case["benchmarks"]["torch_bf16_forward"] = timed(
        lambda: torch.nn.functional.linear(x, w), repeats
    )

    def torch_training():
        tx = x.detach().requires_grad_(True)
        tw = w.detach().requires_grad_(True)
        ty = torch.nn.functional.linear(tx, tw)
        return torch.autograd.grad(ty, (tx, tw), dy)

    case["benchmarks"]["torch_bf16_forward_backward"] = timed(torch_training, repeats)
    case["passed"] = all(check["passed"] for check in case["checks"])
    return case


def make_inputs(rows, family, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 14336, device="cuda", generator=generator)
    w = torch.randn(5376, 14336, device="cuda", generator=generator) / 14336**0.5
    if family == "swiglu_like":
        up = torch.randn(x.shape, device="cuda", generator=generator)
        x = torch.nn.functional.silu(x) * up
    elif family == "cancellation":
        # Adjacent +/- activation pairs and nearly identical weights create near-zero sums.
        x[:, 1::2] = -x[:, ::2]
        w[:, 1::2] = w[:, ::2] + 1e-5
    dy = torch.randn(rows, 5376, device="cuda", generator=generator) * 0.1
    return x.bfloat16(), w.bfloat16(), dy.bfloat16()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,31,32,33,127,128,129,256,513,1024,2048,4096")
    parser.add_argument("--families", default="random,swiglu_like,cancellation")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-log")
    parser.add_argument(
        "--backend", choices=(H3_DOWN_BACKEND, H3_LEGACY_DOWN_BACKEND), default=H3_DOWN_BACKEND
    )
    parser.add_argument(
        "--compare-vllm",
        action="store_true",
        help="probe the repository's pinned vLLM Triton matrix kernel",
    )
    parser.add_argument(
        "--fixture", help="local weights_only torch file with x/weight/grad_output/metadata"
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "h3-down-qualification-v2",
        "backend": args.backend,
        "status": "running",
        "cases": [],
        "acceptance_complete": False,
        "scope": "one CUDA device, eager, operator-only; no TP or full-model acceptance",
    }
    exit_code = 2

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["provenance"] = provenance(args)
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("an NVIDIA SM90 GPU is required; CPU runs are not GPU acceptance")
        rows = [int(value) for value in args.rows.split(",")]
        families = args.families.split(",")
        if not rows or any(not 1 <= value <= H3_MAX_ROWS for value in rows):
            raise ValueError("rows must be in [1,32768]")
        if args.repeats <= 0 or not set(families) <= {"random", "swiglu_like", "cancellation"}:
            raise ValueError("invalid repeats or input family")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        report["provenance"].update(
            device=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            candidate_tf32_enabled=torch.backends.cuda.matmul.allow_tf32,
            reference_tf32_enabled=False,
            torch_baseline_bf16_reduced_precision_reduction=(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            nvidia_smi=command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        )
        _require_backend(torch.device("cuda"), args.backend)
        save()
        for family in families:
            for count in rows:
                values = make_inputs(count, family, args.seed)
                case = run_case(
                    *values,
                    family,
                    args.repeats,
                    compare_vllm=args.compare_vllm,
                    backend=args.backend,
                )
                report["cases"].append(case)
                save()
                print(f"{family} M={count}: {'PASS' if case['passed'] else 'FAIL'}", flush=True)
                del values, case
        if args.fixture:
            fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
            metadata = fixture["metadata"]
            if metadata.get("checkpoint_revision") != H3_CHECKPOINT_REVISION:
                raise ValueError("fixture checkpoint revision does not match the H3 contract")
            if not metadata.get("weight_key") or not metadata.get("activation_source"):
                raise ValueError("fixture must declare weight_key and activation_source")
            if metadata.get("activation_kind") not in {"captured", "synthetic"}:
                raise ValueError("fixture activation_kind must be captured or synthetic")
            # Metadata must remain serializable even on a later validation failure.
            json.dumps(metadata, allow_nan=False)
            report["fixture"] = {"sha256": sha256(args.fixture), "metadata": metadata}
            values = [fixture[key].to("cuda") for key in ("x", "weight", "grad_output")]
            if any(value.dtype != torch.bfloat16 for value in values):
                raise TypeError("fixture tensors must already be quantized BF16")
            values[0] = values[0].reshape(-1, 14336)
            values[2] = values[2].reshape(-1, 5376)
            report["cases"].append(
                run_case(
                    *values,
                    "checkpoint_fixture",
                    args.repeats,
                    compare_vllm=args.compare_vllm,
                    backend=args.backend,
                )
            )
        passed = all(case["passed"] for case in report["cases"])
        report["status"] = "passed" if passed else "failed"
        report["acceptance_complete"] = False
        report["remaining_acceptance"] = [
            "maintainer review of arithmetic contract and covered matrix",
            "independent build/environment repeat and prior-art performance decision",
        ]
        if not args.fixture:
            report["remaining_acceptance"].append("real checkpoint weights and captured inputs")
        elif report["fixture"]["metadata"]["activation_kind"] != "captured":
            report["remaining_acceptance"].append("real captured SwiGLU activations")
        if not args.build_log and args.backend == H3_LEGACY_DOWN_BACKEND:
            report["remaining_acceptance"].append("auditable no-fast-math extension build log")
        exit_code = 0 if passed else 1
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        save()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
