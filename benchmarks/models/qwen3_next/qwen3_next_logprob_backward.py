#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Is the selected-token logprob backward reproducible? verl, Miles, RL-Kernel on the real LM head.

    TRITON_F32_DEFAULT=ieee python \\
        benchmarks/models/qwen3_next/qwen3_next_logprob_backward.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --miles-src <miles> \\
        --verl-src <verl> --output <dir>

One GPU, the whole 151936-row LM head (TP1), seeded hidden states, targets and
output gradient. Candidates:

* ``verl_fused`` -- verl's ``linear_cross_entropy``: its backward accumulates
  ``d_hidden`` and ``d_weight`` with ``tl.atomic_add`` over vocabulary blocks.
* ``miles_fused`` -- Miles' ``fused_log_probs_and_entropy``: store-only Triton
  kernels (no atomics); the gradient is the logits gradient.
* ``rl_kernel_ws2`` -- RL-Kernel's ``shared_logprob`` (the control); gradient is
  the logits gradient.

For ``--repeats`` identical launches it records how many equal the first bit
for bit, the largest difference and the share of elements that differ, then
whether 8 probe rows keep their gradient alone or inside batches of 8-1024
rows at the first, middle and last position. (The forward of verl is the other
place its source uses ``tl.atomic_add``, only for ``reduction`` sum / mean.)
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from benchmarks.models.qwen3_next import thirdparty_sources  # noqa: E402
from benchmarks.models.qwen3_next.logprob_candidates import VOCAB, Context, _rl_kernel  # noqa: E402
from tools.validation.models.qwen3_next.qwen3_next_gate_common import (  # noqa: E402
    CheckpointWeights,
    provenance,
)

PROBE = 8
BATCHES = (8, 64, 256, 1024)
TOKENS = 1024


def _compare(first, other):
    if torch.equal(first, other):
        return True, 0.0, 0.0
    difference = first.float() - other.float()
    return False, float(difference.abs().max()), float((difference != 0).float().mean())


def make_runners(head, miles, verl):
    """Each runner maps (hidden rows, targets, dlogprob) to the gradients it produces."""
    runners = {}

    def verl_run(hidden, targets, dlp):
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        h = hidden.detach().clone().requires_grad_(True)
        w = head.detach().clone().requires_grad_(True)
        logprob, _ = linear_cross_entropy(h, w, targets, 1.0, "none", None)
        logprob.backward(dlp)
        return {"d_hidden": h.grad, "d_weight": w.grad}

    def logits_runner(factory):
        def run(hidden, targets, dlp):
            logits = (hidden @ head.t()).detach().requires_grad_(True)
            logprob = factory(logits, targets)
            logprob.backward(dlp)
            return {"d_logits": logits.grad}

        return run

    def miles_logprob(logits, targets):
        from miles.backends.training_utils.loss.hub.fused_log_probs import (
            fused_log_probs_and_entropy,
        )

        rows = torch.arange(logits.shape[0], device=logits.device)
        return fused_log_probs_and_entropy(logits, rows, targets, tp_group=None, vocab_size=VOCAB)[
            0
        ]

    def ws2_logprob(logits, targets):
        return _rl_kernel(1, None, Context())(logits, targets)

    if verl is not None:
        runners["verl_fused"] = (verl_run, "d_hidden")
    if miles is not None:
        runners["miles_fused"] = (logits_runner(miles_logprob), "d_logits")
    runners["rl_kernel_ws2"] = (logits_runner(ws2_logprob), "d_logits")
    return runners


def measure(args):
    third_party = []
    miles = verl = None
    if args.miles_src is not None:
        third_party.append(thirdparty_sources.load_miles(args.miles_src))
        miles = True
    if args.verl_src is not None:
        third_party.append(thirdparty_sources.load_verl(args.verl_src))
        verl = True
    generator = torch.Generator(device="cuda").manual_seed(4282)
    rows = 2 * TOKENS
    hidden_all = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
    targets_all = torch.randint(0, VOCAB, (rows,), device="cuda", generator=generator)
    dlp_all = torch.randn(rows, device="cuda", dtype=torch.float32, generator=generator)
    with ExitStack() as stack:
        head = CheckpointWeights(args.checkpoint, "", ("lm_head.weight",), stack)["lm_head.weight"]
    head = head.cuda()
    hidden, targets, dlp = hidden_all[:TOKENS], targets_all[:TOKENS], dlp_all[:TOKENS]
    report = {}
    for name, (run, row_key) in make_runners(head, miles, verl).items():
        try:
            first = run(hidden, targets, dlp)
            identical = {key: 1 for key in first}
            worst = {key: (0.0, 0.0) for key in first}
            for _ in range(args.repeats - 1):
                again = run(hidden, targets, dlp)
                for key in first:
                    same, largest, share = _compare(first[key], again[key])
                    identical[key] += same
                    worst[key] = (max(worst[key][0], largest), max(worst[key][1], share))
                del again
            grid = {}
            probe = (hidden_all[:PROBE], targets_all[:PROBE], dlp_all[:PROBE])
            alone = run(*probe)[row_key]
            for size in BATCHES:
                for position in sorted({0, (size - PROBE) // 2, size - PROBE}):
                    rows_slice = []
                    for index, source in enumerate((hidden_all, targets_all, dlp_all)):
                        filler = source[TOKENS:]
                        rows_slice.append(
                            torch.cat(
                                (filler[:position], probe[index], filler[position : size - PROBE])
                            )
                        )
                    got = run(*rows_slice)[row_key][position : position + PROBE]
                    grid[f"{size}@{position}"] = bool(torch.equal(got, alone))
            report[name] = {
                "gradients": {
                    key: {
                        "repeats_equal_to_first": identical[key],
                        "repeats": args.repeats,
                        "max_abs_diff_between_repeats": worst[key][0],
                        "max_share_of_elements_differing": worst[key][1],
                        "dtype": str(first[key].dtype).removeprefix("torch."),
                    }
                    for key in first
                },
                "row_gradient": row_key,
                "batch_grid": grid,
                "batch_invariant": all(grid.values()),
            }
        except Exception as error:  # noqa: BLE001 - recorded per candidate
            report[name] = {"unavailable": f"{type(error).__name__}: {str(error)[:300]}"}
    meta = provenance()
    out = {
        "kind": "qwen3_next_logprob_backward",
        "tokens": TOKENS,
        "repeats": args.repeats,
        "third_party": third_party,
        "environment": meta["environment"],
        "rl_kernel_commit": meta["rl_kernel_commit"],
        "tracked_tree_dirty": meta["tracked_tree_dirty"],
        "candidates": report,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, row in report.items():
        if "unavailable" in row:
            print(name, "UNAVAILABLE", row["unavailable"])
            continue
        print(
            name,
            {k: (v["repeats_equal_to_first"], v["repeats"]) for k, v in row["gradients"].items()},
        )
        print("   batch invariant:", row["batch_invariant"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--miles-src", type=Path)
    parser.add_argument("--verl-src", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
