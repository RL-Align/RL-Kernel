#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the expert-parallel MoE prior art at EP4 or EP1; merge launches.

    torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \\
        --checkpoint <ckpt> --output <dir>/ep4
    torchrun --nproc-per-node 1 benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \\
        --checkpoint <ckpt> --output <dir>/ep1
    python benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \\
        --merge <dir>/ep4 <dir>/ep4-<other NCCL setting> <dir>/ep1 --out report.json

The first ``--merge`` directory is the EP4 reference, the last the EP1 run; any
directories in between are EP4 runs under other NCCL settings.
"""

import argparse
import json
import os
import sys
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from benchmarks.models.qwen3_next.moe_ep_candidates import (  # noqa: E402
    BATCHES,
    EXPERTS,
    HIDDEN,
    LATENCY_TOKENS,
    LOCAL,
    candidate_report,
    digest,
    expert_weights,
    fp64_reference,
)
from tools.validation.models.qwen3_next.qwen3_next_gate_common import (  # noqa: E402
    CheckpointWeights,
    provenance,
)

NCCL_SETTINGS = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS")


def _init_vllm(ep: int, rank: int) -> None:
    """vLLM's parallel state over the same ranks, so its all-reduce dispatch is real."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as vllm_state

    with set_current_vllm_config(VllmConfig()):
        vllm_state.init_distributed_environment(ep, rank, "env://", rank, "nccl")
        vllm_state.initialize_model_parallel(tensor_model_parallel_size=ep)


def measure(checkpoint: Path, output: Path) -> None:
    # Required by torch.use_deterministic_algorithms; set for every candidate alike.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ep = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    if ep not in (1, 4):
        raise SystemExit("Run with one or four ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", rank)
    )
    try:
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=ep,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(1234)
        _init_vllm(ep, rank)
        from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import moe_hf_shapes

        generator = torch.Generator(device="cuda").manual_seed(4280)
        rows = max(LATENCY_TOKENS)
        x_all = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
        x_all = x_all * 0.1
        dy_all = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
        with ExitStack() as stack:
            weights = CheckpointWeights(checkpoint, "model.layers.0.mlp.", moe_hf_shapes(), stack)
            router = weights["gate.weight"].cuda()
            all_gate_up, all_down = expert_weights(weights, range(EXPERTS))
        owned = slice(rank * LOCAL, (rank + 1) * LOCAL) if ep > 1 else slice(0, EXPERTS)
        gate_up = all_gate_up[owned].cuda()
        down = all_down[owned].cuda()
        with torch.no_grad():
            reference = fp64_reference(x_all[:1024], router, all_gate_up, all_down)
        del all_gate_up, all_down
        report = {
            "ep": ep,
            "rank": rank,
            "tokens": 1024,
            "nccl_settings": {name: os.environ.get(name) for name in NCCL_SETTINGS},
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "input_sha256": digest(x_all),
            **provenance(),
            "candidates": candidate_report(
                x_all,
                dy_all,
                router,
                gate_up,
                down,
                ep=ep,
                group=dist.group.WORLD,
                reference=reference,
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _all(launch, key, field, sub=None):
    rows = [r["candidates"][key] for r in launch]
    rows = [row[sub] for row in rows] if sub else rows
    return [row[field] for row in rows]


def merge(runs: list[Path]) -> dict:
    def load(run):
        return [json.loads(path.read_text()) for path in sorted(run.glob("rank-*.json"))]

    ep4, others, ep1 = load(runs[0]), [load(r) for r in runs[1:-1]], load(runs[-1])
    if len(ep4) != 4 or len(ep1) != 1 or any(len(o) != 4 for o in others):
        raise SystemExit("Expected four-rank EP4 runs first and a one-rank EP1 run last")
    launches = [ep4, *others, ep1]
    ep4_launches = [ep4, *others]
    if len({r["input_sha256"] for launch in launches for r in launch}) != 1:
        raise SystemExit("Launches did not see the same input")
    commits = {r["rl_kernel_commit"] for launch in launches for r in launch}
    table = {}
    for key, cand in ep4[0]["candidates"].items():
        if "unavailable" in cand:
            table[key] = cand
            continue
        one = ep1[0]["candidates"][key]
        row = {
            "ranks_agree": all(
                len(set(_all(launch, key, "output_sha256"))) == 1 for launch in ep4_launches
            ),
            "ep4_equals_ep1": cand["output_sha256"] == one["output_sha256"],
            "nccl_setting_invariant": len(
                {launch[0]["candidates"][key]["output_sha256"] for launch in ep4_launches}
            )
            == 1,
            "distinct_ep4_results": len(
                {launch[0]["candidates"][key]["output_sha256"] for launch in ep4_launches}
            ),
            "repeat_bitwise": all(all(_all(launch, key, "repeat_bitwise")) for launch in launches),
            "batch_bitwise": all(all(_all(launch, key, "batch_bitwise")) for launch in launches),
            "batch_cells_failed": sorted(
                {
                    cell
                    for launch in launches
                    for r in launch
                    for cell, ok in r["candidates"][key]["batch_grid"].items()
                    if not ok
                }
            ),
            "max_abs_vs_fp64": cand["max_abs_vs_fp64"],
            "mean_abs_vs_fp64": cand["mean_abs_vs_fp64"],
            "max_abs_vs_fp64_ep1": one["max_abs_vs_fp64"],
            "max_abs_vs": cand["max_abs_vs"],
            "latency_us_ep4": {
                n: max(r["candidates"][key]["latency_us"][n] for r in ep4)
                for n in cand["latency_us"]
            },
            "latency_us_ep1": one["latency_us"],
        }
        if "backward" in cand:
            b4 = [r["candidates"][key]["backward"] for r in ep4]
            b1 = one["backward"]
            row["backward"] = {
                "ranks_agree_dx": all(
                    len(set(_all(launch, key, "dx_sha256", "backward"))) == 1
                    for launch in ep4_launches
                ),
                "dx_ep4_equals_ep1": b4[0]["dx_sha256"] == b1["dx_sha256"],
                "dw_ep4_equals_ep1": all(
                    b4[r]["d_gate_up_sha256"] == b1["d_gate_up_slice_sha256"][r]
                    and b4[r]["d_down_sha256"] == b1["d_down_slice_sha256"][r]
                    for r in range(4)
                ),
                "nccl_setting_invariant": all(
                    len({launch[r]["candidates"][key]["backward"][f] for launch in ep4_launches})
                    == 1
                    for r in range(4)
                    for f in ("dx_sha256", "d_gate_up_sha256", "d_down_sha256")
                ),
                "repeat_bitwise": all(
                    all(_all(launch, key, "repeat_bitwise", "backward")) for launch in launches
                ),
                "batch_bitwise": all(
                    all(_all(launch, key, "batch_bitwise", "backward")) for launch in launches
                ),
                "latency_us_ep4": max(b["latency_us"] for b in b4),
                "latency_us_ep1": b1["latency_us"],
            }
        table[key] = row
    return {
        "kind": "qwen3_next_moe_ep_prior_art",
        "schema_version": 2,
        "rfc": 428,
        "rl_kernel_commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "tracked_tree_dirty": any(r["tracked_tree_dirty"] for launch in launches for r in launch),
        "environment": ep4[0]["environment"],
        "tokens": ep4[0]["tokens"],
        "batches": list(BATCHES),
        "latency_tokens": list(LATENCY_TOKENS),
        "nccl_settings": [launch[0]["nccl_settings"] for launch in ep4_launches],
        "candidates": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.merge:
        report = merge(args.merge)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        for key, row in report["candidates"].items():
            print(key, {k: v for k, v in row.items() if k not in ("max_abs_vs", "backward")})
            if "backward" in row:
                print("  backward", row["backward"])
    else:
        measure(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
