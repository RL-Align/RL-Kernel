#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write an RFC #420 WS2 evidence report (byte equality vs WS1, timing, readback).

Each TP size runs as real NCCL processes, one GPU per rank, with the
deterministic CUDA collective. Run it on a clean tree on one 8-GPU node;
``tools/validation/models/plot_h3_evidence.py`` turns the report into a figure.

    export RL_KERNEL_H3_WEIGHTS=<dir written by tools/weights/prepare_h3_weights.py>
    python tools/validation/models/h3_ws2_evidence.py --op tp_adaln_3mod --worlds 1,2,4,8 \\
        --out reports/experiments/h3-tp-adaln-3mod-b200/report.json
    python tools/validation/models/h3_ws2_evidence.py --op sp_norm_adaln --worlds 1,2,4,8 \\
        --out reports/experiments/h3-sp-norm-adaln-b200/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from rl_engine.validation.models.h3_chain import environment, git_state  # noqa: E402
from rl_engine.validation.models.h3_weights import (  # noqa: E402
    load_h3_conditioning_weights,
    load_h3_manifest,
)
from rl_engine.validation.models.h3_ws2 import (  # noqa: E402
    naive_sp_mismatch,
    run_world,
    sp_case,
    sp_region_sweep,
    tp_projection_sweep,
)

WEIGHT = "transformer_blocks.0.adaln_proj.linear.weight"
BIAS = "transformer_blocks.0.adaln_proj.linear.bias"


def _tp_adaln(worlds: list[int], max_t: int) -> dict:
    weights = load_h3_conditioning_weights("cpu")
    weight, bias = weights[WEIGHT], weights[BIAS]
    g = torch.Generator(device="cpu").manual_seed(0)
    inputs = {
        "temb": torch.randn(max_t, weight.shape[1], generator=g) * 2,
        "weight": weight,
        "bias": bias,
        "grad": torch.randn(max_t, weight.shape[0], generator=g).bfloat16(),
    }
    runs = []
    for world in worlds:
        ranks = run_world(world, tp_projection_sweep, inputs)
        runs.append({"tp": world, "ranks": ranks})
        equal = all(
            v for r in ranks for e in r["equality"] for k, v in e.items() if k != "num_timesteps"
        )
        print(f"tp={world}: byte-equal to WS1 on every rank and T: {equal}")
    return {"weights": [WEIGHT, BIAS], "runs": runs}


SP_CASES = [
    {"batch": 1, "seq": 4097, "layout": "block", "seed": 1},
    {"batch": 1, "seq": 4097, "layout": "interleaved", "seed": 2},
    {"batch": 2, "seq": 4097, "layout": "block", "seed": 3},
    {"batch": 1, "seq": 32768, "layout": "block", "seed": 4},
    {"batch": 1, "seq": 32768, "layout": "interleaved", "seed": 5},
    {"batch": 1, "seq": 131072, "layout": "block", "seed": 6},
]


def _sp_norm(worlds: list[int], _max_t: int) -> dict:
    runs = []
    for world in worlds:
        ranks = run_world(world, sp_region_sweep, {}, cases=SP_CASES)
        runs.append({"sp": world, "ranks": ranks})
        equal = all(all(c["equal"].values()) for r in ranks for c in r["cases"])
        print(f"sp={world}: byte-equal to WS1 on every rank and case: {equal}")
    naive = {
        layout: naive_sp_mismatch(sp_case(1, 32768, layout=layout, seed=4), max(worlds))
        for layout in ("block", "interleaved")
    }
    region = "norm2(residual + gate_msa[row] * y) * (1 + scale_mlp[row]) + shift_mlp[row]"
    return {
        "region": region,
        "hidden": 5376,
        "runs": runs,
        "naive_sp": {"sp": max(worlds), **naive},
    }


OPS = {"tp_adaln_3mod": _tp_adaln, "sp_norm_adaln": _sp_norm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, choices=sorted(OPS))
    parser.add_argument("--worlds", default="1,2,4,8")
    parser.add_argument("--max-timesteps", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    worlds = [int(w) for w in args.worlds.split(",")]
    if torch.cuda.device_count() < max(worlds):
        raise SystemExit(f"needs {max(worlds)} GPUs, found {torch.cuda.device_count()}")

    manifest = load_h3_manifest()
    report = {
        "kind": "h3_ws2_report",
        "op": args.op,
        "rfc": manifest["rfc"],
        "model_revision": manifest["model_identity"]["revision"],
        **git_state(),
        "environment": {**environment(), "gpus": torch.cuda.device_count()},
        **OPS[args.op](worlds, args.max_timesteps),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    commit, dirty = report["rl_kernel_commit"][:7], report["tracked_tree_dirty"]
    print(f"wrote {args.out} (commit {commit}, dirty={dirty})")


if __name__ == "__main__":
    main()
