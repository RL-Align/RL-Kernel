#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Is the selected-token logprob backward reproducible? verl, Miles, RL-Kernel on the real LM head.

    TRITON_F32_DEFAULT=ieee python \\
        benchmarks/models/qwen3_next/qwen3_next_logprob_backward.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --miles-src <miles> \\
        --verl-src <verl> --output <dir>/tp1
    torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_logprob_backward.py \\
        --checkpoint <ckpt> --miles-src <miles> --verl-src <verl> --output <dir>/tp4-<setting>
    python benchmarks/models/qwen3_next/qwen3_next_logprob_backward.py \\
        --merge <dir>/tp4-default <dir>/tp4-<other setting> <dir>/tp1 --out report.json

Seeded hidden states, targets and output gradient on the real 151936-row LM head,
whole on one GPU (TP1) or split over four ranks (TP4). Candidates:

The logits of the logits-based candidates come from RL-Kernel's batch-invariant GEMM;
verl fuses its own GEMM into the kernel and is measured as it is.

* ``verl_fused`` -- verl's ``linear_cross_entropy``: its backward accumulates
  ``d_hidden`` and ``d_weight`` with ``tl.atomic_add`` over vocabulary blocks.
* ``miles_fused`` -- Miles' ``fused_log_probs_and_entropy``: store-only Triton
  kernels (no atomics); the gradient is the logits gradient.
* ``rl_kernel_ws2`` -- RL-Kernel's ``shared_logprob`` (the control); gradient is
  the logits gradient.

A launch records, per gradient, its SHA-256 (and, at TP1, the SHA-256 of each of
the four slices a TP4 rank would own), how many of ``--repeats`` identical calls
equal the first bit for bit, and whether 8 probe rows keep their gradient alone or
inside batches of 8-1024 rows at the first, middle and last position. ``--merge``
compares the TP4 launches (NCCL settings) with each other and with TP1.
(The forward of verl is the other place its source uses ``tl.atomic_add``, only
for ``reduction`` sum / mean.)
"""

from __future__ import annotations

import argparse
import hashlib
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
from benchmarks.models.qwen3_next.logprob_candidates import VOCAB, Context, _rl_kernel  # noqa: E402
from tools.validation.models.qwen3_next.qwen3_next_gate_common import (  # noqa: E402
    CheckpointWeights,
    provenance,
)

PROBE = 8
BATCHES = (8, 64, 256, 1024)
TOKENS = 1024
NCCL_SETTINGS = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS")
SHARDED = {"d_logits": 1, "d_weight": 0}  # gradient -> sharded dimension (TP4 rank r owns slice r)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def _compare(first, other):
    if torch.equal(first, other):
        return True, 0.0, 0.0
    difference = first.float() - other.float()
    return False, float(difference.abs().max()), float((difference != 0).float().mean())


def make_runners(head, miles, verl, tp, group):
    """Each runner maps (hidden rows, targets, dlogprob) to the gradients it produces.

    ``head`` is this rank's shard of the LM head (the whole head at TP1).
    """
    from rl_engine.models.qwen3_next.qwen3_next_forward import shared_linear

    runners = {}
    tp_group = group if tp > 1 else None

    def verl_run(hidden, targets, dlp):
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        h = hidden.detach().clone().requires_grad_(True)
        w = head.detach().clone().requires_grad_(True)
        logprob, _ = linear_cross_entropy(h, w, targets, 1.0, "none", tp_group)
        logprob.backward(dlp)
        return {"d_hidden": h.grad, "d_weight": w.grad}

    def logits_runner(factory):
        def run(hidden, targets, dlp):
            # A plain matmul picks a different cuBLAS algorithm per row count, which
            # would make a probe row's logits depend on the batch; the batch-invariant
            # GEMM keeps the comparison about the logprob kernels.
            logits = shared_linear(hidden, head).detach().requires_grad_(True)
            logprob = factory(logits, targets)
            logprob.backward(dlp)
            return {"d_logits": logits.grad}

        return run

    def miles_logprob(logits, targets):
        from miles.backends.training_utils.loss.hub.fused_log_probs import (
            fused_log_probs_and_entropy,
        )

        rows = torch.arange(logits.shape[0], device=logits.device)
        return fused_log_probs_and_entropy(
            logits, rows, targets, tp_group=tp_group, vocab_size=VOCAB
        )[0]

    def ws2_logprob(logits, targets):
        return _rl_kernel(tp, group, Context())(logits, targets)

    if verl is not None:
        runners["verl_fused"] = (verl_run, "d_hidden")
    if miles is not None:
        runners["miles_fused"] = (logits_runner(miles_logprob), "d_logits")
    runners["rl_kernel_ws2"] = (logits_runner(ws2_logprob), "d_logits")
    return runners


def launch_report(run, row_key, hidden_all, targets_all, dlp_all, repeats, tp):
    hidden, targets, dlp = hidden_all[:TOKENS], targets_all[:TOKENS], dlp_all[:TOKENS]
    first = run(hidden, targets, dlp)
    identical = {key: 1 for key in first}
    worst = {key: (0.0, 0.0) for key in first}
    for _ in range(repeats - 1):
        again = run(hidden, targets, dlp)
        for key in first:
            same, largest, share = _compare(first[key], again[key])
            identical[key] += same
            worst[key] = (max(worst[key][0], largest), max(worst[key][1], share))
        del again
    probe = (hidden_all[:PROBE], targets_all[:PROBE], dlp_all[:PROBE])
    alone = run(*probe)[row_key]
    grid = {}
    for size in BATCHES:
        for position in sorted({0, (size - PROBE) // 2, size - PROBE}):
            rows = []
            for index, source in enumerate((hidden_all, targets_all, dlp_all)):
                filler = source[TOKENS:]
                rows.append(
                    torch.cat((filler[:position], probe[index], filler[position : size - PROBE]))
                )
            got = run(*rows)[row_key][position : position + PROBE]
            grid[f"{size}@{position}"] = bool(torch.equal(got, alone))
    gradients = {}
    for key, value in first.items():
        entry = {
            "repeats_equal_to_first": identical[key],
            "repeats": repeats,
            "max_abs_diff_between_repeats": worst[key][0],
            "max_share_of_elements_differing": worst[key][1],
            "dtype": str(value.dtype).removeprefix("torch."),
            "sha256": digest(value),
        }
        if tp == 1 and key in SHARDED:
            width = value.shape[SHARDED[key]] // 4
            entry["slice_sha256"] = [
                digest(value.narrow(SHARDED[key], r * width, width)) for r in range(4)
            ]
        gradients[key] = entry
    return {
        "gradients": gradients,
        "row_gradient": row_key,
        "batch_grid": grid,
        "batch_invariant": all(grid.values()),
    }


def measure(args):
    third_party = []
    miles = verl = None
    if args.miles_src is not None:
        third_party.append(thirdparty_sources.load_miles(args.miles_src))
        miles = True
    if args.verl_src is not None:
        third_party.append(thirdparty_sources.load_verl(args.verl_src))
        verl = True
    tp = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if tp not in (1, 4):
        raise SystemExit("Run with one or four ranks")
    torch.cuda.set_device(rank)
    group = None
    if tp > 1:
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", rank)
        )
        group = dist.group.WORLD
    try:
        generator = torch.Generator(device="cuda").manual_seed(4282)
        rows = 2 * TOKENS
        hidden_all = torch.randn(
            rows, 2048, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        targets_all = torch.randint(0, VOCAB, (rows,), device="cuda", generator=generator)
        dlp_all = torch.randn(rows, device="cuda", dtype=torch.float32, generator=generator)
        with ExitStack() as stack:
            head = CheckpointWeights(args.checkpoint, "", ("lm_head.weight",), stack)[
                "lm_head.weight"
            ]
        width = VOCAB // tp
        head = head[rank * width : (rank + 1) * width].contiguous().cuda()
        report = {}
        for name, (run, row_key) in make_runners(head, miles, verl, tp, group).items():
            try:
                report[name] = launch_report(
                    run, row_key, hidden_all, targets_all, dlp_all, args.repeats, tp
                )
            except Exception as error:  # noqa: BLE001 - recorded per candidate
                report[name] = {"unavailable": f"{type(error).__name__}: {str(error)[:300]}"}
        meta = provenance()
        out = {
            "kind": "qwen3_next_logprob_backward",
            "tp": tp,
            "rank": rank,
            "tokens": TOKENS,
            "repeats": args.repeats,
            "nccl_settings": {name: os.environ.get(name) for name in NCCL_SETTINGS},
            "third_party": third_party,
            "environment": meta["environment"],
            "rl_kernel_commit": meta["rl_kernel_commit"],
            "tracked_tree_dirty": meta["tracked_tree_dirty"],
            "candidates": report,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        name = "report.json" if tp == 1 else f"rank-{rank}.json"
        (args.output / name).write_text(json.dumps(out, indent=2) + "\n")
        for cand, row in report.items():
            if "unavailable" in row:
                print(cand, "UNAVAILABLE", row["unavailable"])
                continue
            print(
                cand,
                {
                    k: (v["repeats_equal_to_first"], v["repeats"])
                    for k, v in row["gradients"].items()
                },
            )
            print("   batch invariant:", row["batch_invariant"])
        if tp > 1:
            dist.barrier()
    finally:
        if tp > 1:
            dist.destroy_process_group()


def merge(runs: list[Path]) -> dict:
    """TP4 launches first (NCCL defaults, then other settings), the TP1 launch last."""

    def load_tp4(run):
        return [json.loads((run / f"rank-{r}.json").read_text()) for r in range(4)]

    tp1 = json.loads((runs[-1] / "report.json").read_text())
    launches = [load_tp4(run) for run in runs[:-1]]
    table = {}
    for name, base in launches[0][0]["candidates"].items():
        if "unavailable" in base:
            table[name] = base
            continue
        entry = {"gradients": {}}
        for key in base["gradients"]:
            per_setting = [
                [r["candidates"][name]["gradients"][key]["sha256"] for r in ranks]
                for ranks in launches
            ]
            reference = tp1["candidates"][name]["gradients"][key]
            if key in SHARDED:
                equal_tp1 = all(
                    ranks[r]["candidates"][name]["gradients"][key]["sha256"]
                    == reference["slice_sha256"][r]
                    for ranks in launches
                    for r in range(4)
                )
            else:
                equal_tp1 = all(
                    ranks[r]["candidates"][name]["gradients"][key]["sha256"] == reference["sha256"]
                    for ranks in launches
                    for r in range(4)
                )
            entry["gradients"][key] = {
                "distinct_results_over_settings": len({tuple(s) for s in per_setting}),
                "settings": len(launches),
                "tp4_equals_tp1": equal_tp1,
                "repeat_bitwise_all_launches": all(
                    r["candidates"][name]["gradients"][key]["repeats_equal_to_first"]
                    == r["candidates"][name]["gradients"][key]["repeats"]
                    for ranks in launches
                    for r in ranks
                ),
            }
        entry["batch_invariant"] = all(
            r["candidates"][name]["batch_invariant"] for ranks in launches for r in ranks
        )
        table[name] = entry
    return {
        "kind": "qwen3_next_logprob_backward_tp4",
        "rl_kernel_commit": launches[0][0]["rl_kernel_commit"],
        "tracked_tree_dirty": any(r["tracked_tree_dirty"] for ranks in launches for r in ranks),
        "environment": launches[0][0]["environment"],
        "third_party": launches[0][0]["third_party"],
        "nccl_settings": [ranks[0]["nccl_settings"] for ranks in launches],
        "candidates": table,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--miles-src", type=Path)
    parser.add_argument("--verl-src", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.merge:
        report = merge(args.merge)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        for name, row in report["candidates"].items():
            print(name, json.dumps(row)[:400])
    else:
        measure(args)


if __name__ == "__main__":
    main()
