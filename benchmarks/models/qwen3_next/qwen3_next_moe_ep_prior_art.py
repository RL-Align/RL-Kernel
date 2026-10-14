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
    EXPERTS,
    HIDDEN,
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

TOKENS = 1024
NCCL_SETTINGS = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS")


def measure(checkpoint: Path, output: Path) -> None:
    # Required by torch.use_deterministic_algorithms; set for every candidate alike.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ep = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    if ep not in (1, 4):
        raise SystemExit("Run with one or four ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=15), device_id=torch.device("cuda", rank)
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
        from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import moe_hf_shapes

        generator = torch.Generator(device="cuda").manual_seed(4280)
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
        x = x * 0.1
        with ExitStack() as stack:
            weights = CheckpointWeights(checkpoint, "model.layers.0.mlp.", moe_hf_shapes(), stack)
            router = weights["gate.weight"].cuda()
            all_gate_up, all_down = expert_weights(weights, range(EXPERTS))
        owned = slice(rank * LOCAL, (rank + 1) * LOCAL) if ep > 1 else slice(0, EXPERTS)
        gate_up = all_gate_up[owned].cuda()
        down = all_down[owned].cuda()
        with torch.no_grad():
            reference = fp64_reference(x, router, all_gate_up, all_down)
        del all_gate_up, all_down
        report = {
            "ep": ep,
            "rank": rank,
            "tokens": TOKENS,
            "nccl_settings": {name: os.environ.get(name) for name in NCCL_SETTINGS},
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "input_sha256": digest(x),
            **provenance(),
            "candidates": candidate_report(
                x, router, gate_up, down, ep=ep, group=dist.group.WORLD, reference=reference
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def merge(runs: list[Path]) -> dict:
    def load(run):
        return [json.loads(path.read_text()) for path in sorted(run.glob("rank-*.json"))]

    ep4, others, ep1 = load(runs[0]), [load(r) for r in runs[1:-1]], load(runs[-1])
    if len(ep4) != 4 or len(ep1) != 1 or any(len(o) != 4 for o in others):
        raise SystemExit("Expected four-rank EP4 runs first and a one-rank EP1 run last")
    launches = [ep4, *others, ep1]
    if len({r["input_sha256"] for launch in launches for r in launch}) != 1:
        raise SystemExit("Launches did not see the same input")
    commits = {r["rl_kernel_commit"] for launch in launches for r in launch}
    table = {}
    for key, cand in ep4[0]["candidates"].items():
        if "unavailable" in cand:
            table[key] = cand
            continue
        rows = [r["candidates"][key] for launch in launches for r in launch]
        ep4_digests = [r["candidates"][key]["output_sha256"] for r in ep4]
        table[key] = {
            "ranks_agree": all(
                len({r["candidates"][key]["output_sha256"] for r in launch}) == 1
                for launch in (ep4, *others)
            ),
            "ep4_equals_ep1": ep4_digests[0] == ep1[0]["candidates"][key]["output_sha256"],
            "nccl_setting_invariant": all(
                o[0]["candidates"][key]["output_sha256"] == ep4_digests[0] for o in others
            ),
            "distinct_ep4_results": len(
                {o[0]["candidates"][key]["output_sha256"] for o in (ep4, *others)}
            ),
            "repeat_bitwise": all(row["repeat_bitwise"] for row in rows),
            "batch_bitwise": all(row["batch_bitwise"] for row in rows),
            "max_abs_vs_fp64": cand["max_abs_vs_fp64"],
            "max_abs_vs_fp64_ep1": ep1[0]["candidates"][key]["max_abs_vs_fp64"],
            "max_abs_vs": cand["max_abs_vs"],
            "latency_us_ep4": max(r["candidates"][key]["latency_us"] for r in ep4),
            "latency_us_ep1": ep1[0]["candidates"][key]["latency_us"],
        }
    return {
        "kind": "qwen3_next_moe_ep_prior_art",
        "rfc": 428,
        "rl_kernel_commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "tracked_tree_dirty": any(r["tracked_tree_dirty"] for launch in launches for r in launch),
        "environment": ep4[0]["environment"],
        "tokens": ep4[0]["tokens"],
        "nccl_settings": [launch[0]["nccl_settings"] for launch in (ep4, *others)],
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
            print(key, {k: v for k, v in row.items() if k != "max_abs_vs"})
    else:
        measure(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
