#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the vocabulary-parallel logprob prior art at one TP degree; merge launches.

    torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \\
        --checkpoint <ckpt> --output <dir>/tp4
    torchrun --nproc-per-node 1 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \\
        --checkpoint <ckpt> --output <dir>/tp1
    python benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \\
        --merge <dir>/tp4 <dir>/tp4-<other NCCL setting> <dir>/tp1 --out report.json

The first ``--merge`` directory is the TP4 reference, the last the TP1 run;
any directories in between are TP4 runs under other NCCL settings.
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

from benchmarks.models.qwen3_next import thirdparty_sources  # noqa: E402
from benchmarks.models.qwen3_next.logprob_candidates import (  # noqa: E402
    VOCAB,
    Context,
    candidate_report,
    digest,
)
from tools.validation.models.qwen3_next.qwen3_next_gate_common import (  # noqa: E402
    CheckpointWeights,
    provenance,
)

TOKENS = 1024
NCCL_SETTINGS = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS")


def measure(checkpoint: Path, output: Path, miles_src: Path | None, verl_src: Path | None) -> None:
    third_party = []
    if miles_src is not None:
        third_party.append(thirdparty_sources.load_miles(miles_src))
    if verl_src is not None:
        third_party.append(thirdparty_sources.load_verl(verl_src))
    tp = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    if tp not in (1, 4):
        raise SystemExit("Run with one or four ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=15), device_id=torch.device("cuda", rank)
    )
    try:
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
        group = parallel_state.get_tensor_model_parallel_group()
        from rl_engine.models.qwen3_next.qwen3_next_forward import shared_linear

        generator = torch.Generator(device="cuda").manual_seed(4280)
        with ExitStack() as stack:
            head = CheckpointWeights(checkpoint, "", ("lm_head.weight",), stack)["lm_head.weight"]
            hidden = torch.randn(
                TOKENS, 2048, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            targets = torch.randint(0, VOCAB, (TOKENS,), device="cuda", generator=generator)
            with torch.no_grad():
                full = shared_linear(hidden, head.cuda())
        report = {
            "tp": tp,
            "rank": rank,
            "tokens": TOKENS,
            "nccl_settings": {name: os.environ.get(name) for name in NCCL_SETTINGS},
            "full_logits_sha256": digest(full),
            "third_party": third_party,
            **provenance(),
            "candidates": candidate_report(
                full,
                targets,
                tp=tp,
                group=group,
                context=Context(hidden=hidden, head=head.cuda()),
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def merge(runs: list[Path]) -> dict:
    load = lambda run: json.loads((run / "rank-0.json").read_text())  # noqa: E731
    tp4, others, tp1 = load(runs[0]), [load(r) for r in runs[1:-1]], load(runs[-1])
    if tp4["tp"] != 4 or tp1["tp"] != 1 or any(o["tp"] != 4 for o in others):
        raise SystemExit("Expected TP4 runs first and a TP1 run last")
    if len({r["full_logits_sha256"] for r in (tp4, tp1, *others)}) != 1:
        raise SystemExit("Launches did not see the same logits")
    commits = {r["rl_kernel_commit"] for r in (tp4, tp1, *others)}
    table = {}
    for key, cand in tp4["candidates"].items():
        if "unavailable" in cand:
            table[key] = cand
            continue
        table[key] = {
            "tp4_equals_tp1": cand["selected_sha256"]
            == tp1["candidates"].get(key, {}).get("selected_sha256"),
            "nccl_setting_invariant": all(
                o["candidates"][key].get("selected_sha256") == cand["selected_sha256"]
                for o in others
            ),
            "batch_bitwise": cand["batch_bitwise"],
            "distinct_results_in_repeats": max(
                r["candidates"][key]["distinct_results_in_repeats"] for r in (tp4, tp1, *others)
            ),
            "repeats": cand["repeats"],
            "reference": cand["reference"],
            "max_abs_vs_fp64": cand["max_abs_vs_fp64"],
            "max_abs_vs": cand["max_abs_vs"],
            "latency_us_tp4": cand["latency_us"],
        }
    return {
        "kind": "qwen3_next_logprob_prior_art",
        "rfc": 428,
        "rl_kernel_commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "tracked_tree_dirty": any(r["tracked_tree_dirty"] for r in (tp4, tp1, *others)),
        "environment": tp4["environment"],
        "third_party": tp4.get("third_party", []),
        "tokens": tp4["tokens"],
        "nccl_settings": [r["nccl_settings"] for r in (tp4, *others)],
        "candidates": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--miles-src", type=Path, help="checkout of radixark/miles")
    parser.add_argument("--verl-src", type=Path, help="checkout of volcengine/verl")
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.merge:
        report = merge(args.merge)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        for key, row in report["candidates"].items():
            print(key, {k: v for k, v in row.items() if k != "max_abs_vs"})
    else:
        measure(args.checkpoint, args.output, args.miles_src, args.verl_src)


if __name__ == "__main__":
    main()
