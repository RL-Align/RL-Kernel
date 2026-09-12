# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Offline tile/occupancy sweep for the strict ROCm deterministic GEMM leaf.

The benchmark launches the production Triton kernels directly with preallocated
buffers. Every candidate is checked against the pinned 64x64/4-warp baseline at
both the complete leaf workspace and final tree root before timings are reported.

Example:

    python benchmarks/benchmark_rocm_det_gemm_leaf.py \
      --device 4 \
      --cases fwd_gate_m32,fwd_down_m32 \
      --configs 16x128x4,32x64x2xn,32x128x4xn,64x64x4 \
      --output /tmp/rlk_leaf_sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import triton

from rl_engine.kernels.ops.triton.matmul.det_gemm import (
    _copy_tree_root_kernel,
    _copy_tree_root_transposed_kernel,
    _det_gemm_tree_leaf_kernel,
    _det_gemm_tree_reduce_kernel,
    _device_tree_plan,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_KERNEL_SOURCE = _REPO_ROOT / "rl_engine/kernels/ops/triton/matmul/det_gemm.py"


@dataclass(frozen=True)
class LeafConfig:
    block_m: int
    block_n: int
    num_warps: int
    order: str = "leaf"
    waves_per_eu: int = 0

    @property
    def slug(self) -> str:
        return (
            f"{self.block_m}x{self.block_n}x{self.num_warps}x" f"{self.order}xw{self.waves_per_eu}"
        )


@dataclass(frozen=True)
class LeafCase:
    name: str
    m_size: int
    k_size: int
    n_size: int
    transposed_a: bool = False
    transpose_output: bool = False


def _leaf_cases() -> dict[str, LeafCase]:
    cases: list[LeafCase] = []
    for token_count in (1, 8, 16, 32):
        cases.extend(
            (
                LeafCase(f"fwd_gate_m{token_count}", token_count, 4096, 12288),
                LeafCase(f"fwd_down_m{token_count}", token_count, 12288, 4096),
                LeafCase(
                    f"wgrad_gate_m{token_count}",
                    4096,
                    token_count,
                    12288,
                    transposed_a=True,
                    transpose_output=True,
                ),
                LeafCase(
                    f"wgrad_down_m{token_count}",
                    12288,
                    token_count,
                    4096,
                    transposed_a=True,
                    transpose_output=True,
                ),
            )
        )
        for tp_size in (2, 4, 8):
            local_intermediate = 12288 // tp_size
            suffix = f"tp{tp_size}_m{token_count}"
            cases.extend(
                (
                    LeafCase(
                        f"fwd_gate_{suffix}",
                        token_count,
                        4096,
                        local_intermediate,
                    ),
                    LeafCase(
                        f"fwd_down_{suffix}",
                        token_count,
                        local_intermediate,
                        4096,
                    ),
                    LeafCase(
                        f"wgrad_gate_{suffix}",
                        4096,
                        token_count,
                        local_intermediate,
                        transposed_a=True,
                        transpose_output=True,
                    ),
                    LeafCase(
                        f"wgrad_down_{suffix}",
                        local_intermediate,
                        token_count,
                        4096,
                        transposed_a=True,
                        transpose_output=True,
                    ),
                )
            )
    return {case.name: case for case in cases}


_CASES = _leaf_cases()

_BASELINE = LeafConfig(64, 64, 4)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_config(value: str) -> LeafConfig:
    try:
        parts = value.split("x")
        if len(parts) not in (3, 4, 5):
            raise ValueError
        block_m, block_n, num_warps = (int(part) for part in parts[:3])
        order = parts[3] if len(parts) >= 4 else "leaf"
        waves_per_eu = int(parts[4]) if len(parts) == 5 else 0
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "config must be BLOCK_MxBLOCK_NxNUM_WARPS[xORDER[xWAVES_PER_EU]], " f"got {value!r}"
        ) from error
    if (
        block_m <= 0
        or block_n <= 0
        or num_warps not in (1, 2, 4, 8)
        or order not in ("leaf", "n")
        or waves_per_eu not in (0, 1, 2, 4)
    ):
        raise argparse.ArgumentTypeError(f"invalid leaf config {value!r}")
    return LeafConfig(block_m, block_n, num_warps, order, waves_per_eu)


def _git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _command_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            list(arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(samples_ms: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _measure(
    launch: Callable[[], None],
    *,
    warmup: int,
    samples: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(samples)
    ]
    for start, end in events:
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()
    return _summary([float(start.elapsed_time(end)) for start, end in events])


def _inputs(case: LeafCase, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(4100 + case.k_size + case.m_size)
    if case.transposed_a:
        source = torch.randn(
            (case.k_size, case.m_size),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        a = source.t()
        if any(stride <= 0 for stride in a.stride()):
            raise RuntimeError("wgrad benchmark requires a positive-stride transpose view")
        if case.k_size > 1 and a.is_contiguous():
            raise RuntimeError("non-degenerate wgrad benchmark requires a transpose view")
    else:
        a = torch.randn(
            (case.m_size, case.k_size),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
    b = torch.randn(
        (case.k_size, case.n_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return a, b


def _launch_leaf(
    case: LeafCase,
    config: LeafConfig,
    a: torch.Tensor,
    b: torch.Tensor,
    workspace: torch.Tensor,
    plan,
) -> None:
    tiles_m = triton.cdiv(case.m_size, config.block_m)
    tiles_n = triton.cdiv(case.n_size, config.block_n)
    grid = (
        (tiles_n, tiles_m, len(plan.host.leaf_nodes))
        if config.order == "n"
        else (len(plan.host.leaf_nodes), tiles_m, tiles_n)
    )
    launch_options = {"num_warps": config.num_warps}
    if config.waves_per_eu:
        launch_options["waves_per_eu"] = config.waves_per_eu
    _det_gemm_tree_leaf_kernel[grid](
        a,
        b,
        workspace,
        plan.leaf_starts,
        plan.leaf_lengths,
        plan.leaf_nodes,
        M=case.m_size,
        N=case.n_size,
        K=case.k_size,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=32,
        N_FASTEST=config.order == "n",
        **launch_options,
    )


def _launch_tree(
    case: LeafCase,
    config: LeafConfig,
    a: torch.Tensor,
    b: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    plan,
) -> None:
    _launch_leaf(case, config, a, b, workspace, plan)
    reduction_block = 256
    for operations, (lower, upper, result) in zip(
        plan.host.reduction_levels,
        plan.reduction_levels,
        strict=True,
    ):
        grid = (
            len(operations),
            triton.cdiv(case.m_size * case.n_size, reduction_block),
        )
        _det_gemm_tree_reduce_kernel[grid](
            workspace,
            lower,
            upper,
            result,
            M=case.m_size,
            N=case.n_size,
            BLOCK=reduction_block,
        )

    if case.transpose_output:
        block = 32
        grid = (
            triton.cdiv(case.m_size, block),
            triton.cdiv(case.n_size, block),
        )
        _copy_tree_root_transposed_kernel[grid](
            workspace,
            output,
            plan.host.root,
            M=case.m_size,
            N=case.n_size,
            BLOCK_M=block,
            BLOCK_N=block,
        )
    else:
        block = 256
        _copy_tree_root_kernel[(triton.cdiv(output.numel(), block),)](
            workspace,
            output,
            plan.host.root,
            output.numel(),
            BLOCK=block,
        )


def _same_raw_bytes(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    return bool(torch.equal(actual.contiguous().view(torch.uint8), expected.view(torch.uint8)))


def _run_case(
    case: LeafCase,
    configs: list[LeafConfig],
    *,
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    print(
        f"{case.name}: A=({case.m_size}, {case.k_size}), " f"B=({case.k_size}, {case.n_size})",
        flush=True,
    )
    a, b = _inputs(case, device)
    plan = _device_tree_plan(case.k_size, device)
    workspace_shape = (plan.host.node_count, case.m_size, case.n_size)
    output_shape = (
        (case.n_size, case.m_size) if case.transpose_output else (case.m_size, case.n_size)
    )
    reference_workspace = torch.empty(workspace_shape, dtype=torch.bfloat16, device=device)
    reference_output = torch.empty(output_shape, dtype=torch.bfloat16, device=device)
    _launch_tree(
        case,
        _BASELINE,
        a,
        b,
        reference_workspace,
        reference_output,
        plan,
    )
    torch.cuda.synchronize()
    leaf_indices = plan.leaf_nodes.to(torch.int64)
    reference_leaves = reference_workspace.index_select(0, leaf_indices)
    reference_fingerprints = {
        "leaf_workspace_sha256_raw_bytes": _tensor_sha256(reference_leaves),
        "root_sha256_raw_bytes": _tensor_sha256(reference_output),
        "leaf_workspace_nbytes": reference_leaves.numel() * reference_leaves.element_size(),
        "root_nbytes": reference_output.numel() * reference_output.element_size(),
    }

    results: list[dict[str, object]] = []
    for config in configs:
        workspace = torch.empty(workspace_shape, dtype=torch.bfloat16, device=device)
        output = torch.empty(output_shape, dtype=torch.bfloat16, device=device)
        _launch_tree(case, config, a, b, workspace, output, plan)
        _launch_leaf(case, config, a, b, workspace, plan)
        torch.cuda.synchronize()
        candidate_leaves = workspace.index_select(0, leaf_indices)
        leaf_raw_bytes_equal = _same_raw_bytes(candidate_leaves, reference_leaves)

        _launch_tree(case, config, a, b, workspace, output, plan)
        torch.cuda.synchronize()
        root_raw_bytes_equal = _same_raw_bytes(output, reference_output)
        if not leaf_raw_bytes_equal or not root_raw_bytes_equal:
            raise RuntimeError(
                f"{case.name}/{config.slug} changed strict GEMM raw bytes: "
                f"leaf={leaf_raw_bytes_equal}, root={root_raw_bytes_equal}"
            )

        leaf_timing = _measure(
            lambda: _launch_leaf(case, config, a, b, workspace, plan),
            warmup=warmup,
            samples=samples,
        )
        tree_timing = _measure(
            lambda: _launch_tree(case, config, a, b, workspace, output, plan),
            warmup=warmup,
            samples=samples,
        )
        result = {
            "config": asdict(config),
            "slug": config.slug,
            "leaf_raw_bytes_equal": leaf_raw_bytes_equal,
            "root_raw_bytes_equal": root_raw_bytes_equal,
            "leaf_timing": leaf_timing,
            "tree_timing": tree_timing,
        }
        results.append(result)
        print(
            f"  {config.slug}: leaf={leaf_timing['median_ms']:.4f} ms, "
            f"tree={tree_timing['median_ms']:.4f} ms",
            flush=True,
        )
        del candidate_leaves, workspace, output

    baseline_result = next(result for result in results if result["slug"] == _BASELINE.slug)
    baseline_leaf = float(baseline_result["leaf_timing"]["median_ms"])
    baseline_tree = float(baseline_result["tree_timing"]["median_ms"])
    for result in results:
        leaf_median = float(result["leaf_timing"]["median_ms"])
        tree_median = float(result["tree_timing"]["median_ms"])
        result["leaf_speedup_vs_baseline"] = baseline_leaf / leaf_median
        result["tree_speedup_vs_baseline"] = baseline_tree / tree_median
    results.sort(key=lambda result: float(result["leaf_timing"]["median_ms"]))
    return {
        "case": asdict(case),
        "tree": {
            "leaf_count": len(plan.host.leaf_nodes),
            "node_count": plan.host.node_count,
            "reduction_levels": len(plan.host.reduction_levels),
        },
        "baseline_fingerprints": reference_fingerprints,
        "results": results,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this benchmark requires a ROCm PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("no ROCm GPU is available")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"--device must be in [0, {torch.cuda.device_count() - 1}]")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("--warmup must be non-negative and --samples must be positive")
    unknown = sorted(set(args.cases) - _CASES.keys())
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    if _BASELINE not in args.configs:
        args.configs.append(_BASELINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cases",
        type=_parse_csv,
        default=["fwd_gate_m32", "fwd_down_m32"],
        help=f"comma-separated case names; choices: {','.join(_CASES)}",
    )
    parser.add_argument(
        "--configs",
        type=lambda value: [_parse_config(item) for item in _parse_csv(value)],
        default=[
            LeafConfig(16, 64, 2),
            LeafConfig(16, 128, 4),
            LeafConfig(32, 64, 2, "n"),
            LeafConfig(32, 128, 4, "n"),
            _BASELINE,
        ],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(args.device)
    tracked_diff = _git_output("diff", "--binary")
    benchmark_source = Path(__file__).read_bytes()
    payload = {
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "gpu_index": args.device,
            "gpu": properties.name,
            "architecture": getattr(properties, "gcnArchName", ""),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_status": _git_output("status", "--short").splitlines(),
            "tracked_diff_sha256": _sha256(tracked_diff.encode()),
            "kernel_source_sha256": _sha256(_KERNEL_SOURCE.read_bytes()),
            "benchmark_source_sha256": _sha256(benchmark_source),
            "rocm_smi_snapshot": _command_output(
                "rocm-smi",
                "--showuse",
                "--showmemuse",
                "--showtemp",
                "--showclocks",
            ),
        },
        "methodology": {
            "timing": "GPU events around preallocated direct kernel launches",
            "correctness": "raw bytes of every leaf node and final BF16 root",
            "baseline": asdict(_BASELINE),
            "warmup": args.warmup,
            "samples": args.samples,
        },
        "cases": [],
    }
    for case_name in args.cases:
        payload["cases"].append(
            _run_case(
                _CASES[case_name],
                args.configs,
                device=device,
                warmup=args.warmup,
                samples=args.samples,
            )
        )
        torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"results: {args.output.resolve()}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
