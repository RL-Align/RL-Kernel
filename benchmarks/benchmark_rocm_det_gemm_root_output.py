# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A/B the gfx942 deterministic-GEMM root copy against direct output.

Both arms launch the same leaf kernels and canonical BF16 reduction tree with
preallocated storage.  The legacy arm stores the final node in the workspace
and copies it to the output; the candidate stores that same final node directly
in the output.  Blocks are run in alternating AB/BA order to limit clock drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

from rl_engine.kernels.ops.triton.matmul import det_gemm


@dataclass(frozen=True)
class Case:
    name: str
    m_size: int
    k_size: int
    n_size: int


_CASES = {
    case.name: case
    for case in (
        Case("qwen3_tp4_decode_qkv", 1, 4096, 1536),
        Case("qwen3_tp4_decode_o_proj", 1, 1024, 4096),
        Case("qwen3_tp4_decode_gate", 1, 4096, 3072),
        Case("qwen3_tp4_decode_down", 1, 3072, 4096),
    )
}


@dataclass
class CaseState:
    case: Case
    a: torch.Tensor
    b: torch.Tensor
    workspace: torch.Tensor
    output: torch.Tensor
    plan: object
    leaf_config: object


def _raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _launch(state: CaseState, *, direct_root_output: bool) -> None:
    case = state.case
    config = state.leaf_config
    plan = state.plan
    tiles_m = triton.cdiv(case.m_size, config.block_m)
    tiles_n = triton.cdiv(case.n_size, config.block_n)
    leaf_grid = (
        (tiles_n, tiles_m, len(plan.host.leaf_nodes))
        if config.n_fastest
        else (len(plan.host.leaf_nodes), tiles_m, tiles_n)
    )
    det_gemm._det_gemm_kernel[leaf_grid](
        state.a,
        state.b,
        state.workspace,
        plan.leaf_starts,
        plan.leaf_lengths,
        plan.leaf_nodes,
        M=case.m_size,
        N=case.n_size,
        K=case.k_size,
        stride_am=state.a.stride(0),
        stride_ak=state.a.stride(1),
        stride_bk=state.b.stride(0),
        stride_bn=state.b.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=det_gemm._BLOCK_K,
        N_FASTEST=config.n_fastest,
        num_warps=config.num_warps,
    )

    reduction_block = 256
    for level_index, (operations, (lower, upper, output)) in enumerate(
        zip(
            plan.host.reduction_levels,
            plan.reduction_levels,
            strict=True,
        )
    ):
        final_level = level_index == len(plan.host.reduction_levels) - 1
        if direct_root_output and final_level:
            if len(operations) != 1:
                raise RuntimeError("final deterministic GEMM tree level must have one root")
            grid = (triton.cdiv(case.m_size * case.n_size, reduction_block),)
            det_gemm._det_gemm_tree_reduce_to_output_kernel[grid](
                state.workspace,
                state.output,
                lower,
                upper,
                M=case.m_size,
                N=case.n_size,
                BLOCK=reduction_block,
            )
        else:
            grid = (
                len(operations),
                triton.cdiv(case.m_size * case.n_size, reduction_block),
            )
            det_gemm._det_gemm_tree_reduce_kernel[grid](
                state.workspace,
                lower,
                upper,
                output,
                M=case.m_size,
                N=case.n_size,
                BLOCK=reduction_block,
            )

    if not direct_root_output:
        det_gemm._copy_tree_root_kernel[(triton.cdiv(state.output.numel(), reduction_block),)](
            state.workspace,
            state.output,
            plan.host.root,
            state.output.numel(),
            BLOCK=reduction_block,
        )


def _block_median_ms(state: CaseState, direct: bool, samples: int) -> float:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        _launch(state, direct_root_output=direct)
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def _run_case(case: Case, *, warmup: int, samples: int, blocks: int) -> dict[str, object]:
    a = torch.randn((case.m_size, case.k_size), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((case.k_size, case.n_size), device="cuda", dtype=torch.bfloat16)
    plan = det_gemm._device_tree_plan(case.k_size, a.device)
    config = det_gemm._tree_leaf_config(
        a.device,
        case.m_size,
        case.k_size,
        case.n_size,
        transpose_output=False,
        preserve_a_strides=False,
    )
    state = CaseState(
        case=case,
        a=a,
        b=b,
        workspace=torch.empty(
            (plan.host.node_count, case.m_size, case.n_size),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        output=torch.empty((case.m_size, case.n_size), device="cuda", dtype=torch.bfloat16),
        plan=plan,
        leaf_config=config,
    )

    for direct in (False, True):
        for _ in range(warmup):
            _launch(state, direct_root_output=direct)
    torch.cuda.synchronize()
    _launch(state, direct_root_output=False)
    legacy = state.output.clone()
    _launch(state, direct_root_output=True)
    candidate = state.output.clone()
    torch.cuda.synchronize()
    raw_bytes_equal = torch.equal(legacy.view(torch.uint8), candidate.view(torch.uint8))
    if not raw_bytes_equal:
        raise RuntimeError(f"root-output raw-byte mismatch for {case.name}")

    series = {"legacy_copy_ms": [], "direct_output_ms": []}
    for block_index in range(blocks):
        arms = ((False, "legacy_copy_ms"), (True, "direct_output_ms"))
        if block_index % 2:
            arms = tuple(reversed(arms))
        for direct, label in arms:
            series[label].append(_block_median_ms(state, direct, samples))
    legacy_ms = statistics.median(series["legacy_copy_ms"])
    direct_ms = statistics.median(series["direct_output_ms"])
    return {
        "case": asdict(case),
        "tree": {
            "leaf_count": len(plan.host.leaf_nodes),
            "reduction_levels": len(plan.host.reduction_levels),
        },
        "leaf_config": {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "num_warps": config.num_warps,
            "n_fastest": config.n_fastest,
        },
        "raw_bytes_equal": raw_bytes_equal,
        "sha256": _raw_sha256(candidate),
        "series": series,
        "median_ms": {"legacy_copy": legacy_ms, "direct_output": direct_ms},
        "speedup_percent": (legacy_ms / direct_ms - 1.0) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cases", default="qwen3_tp4_decode_qkv,qwen3_tp4_decode_o_proj")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    arch = str(getattr(properties, "gcnArchName", "")).partition(":")[0]
    if arch != "gfx942":
        raise RuntimeError(f"root-output promotion is qualified only on gfx942, got {arch!r}")
    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(_CASES))
    if unknown:
        raise ValueError(f"unknown cases {unknown}; choices are {sorted(_CASES)}")
    if args.warmup < 0 or args.samples <= 0 or args.blocks <= 0:
        raise ValueError("warmup must be non-negative; samples and blocks must be positive")

    torch.manual_seed(args.seed)
    results = [
        _run_case(_CASES[name], warmup=args.warmup, samples=args.samples, blocks=args.blocks)
        for name in selected
    ]
    legacy_sum = sum(item["median_ms"]["legacy_copy"] for item in results)
    direct_sum = sum(item["median_ms"]["direct_output"] for item in results)
    payload = {
        "environment": {
            "device": properties.name,
            "architecture": str(getattr(properties, "gcnArchName", "")),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": sys.argv,
        },
        "methodology": {
            "timing": "GPU events around preallocated complete tree launches",
            "order": "AB/BA alternating blocks",
            "warmup": args.warmup,
            "samples_per_block": args.samples,
            "blocks": args.blocks,
            "seed": args.seed,
        },
        "results": results,
        "combined_projection_medians_ms": {
            "legacy_copy": legacy_sum,
            "direct_output": direct_sum,
            "speedup_percent": (legacy_sum / direct_sum - 1.0) * 100.0,
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
