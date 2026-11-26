# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``sglang_runtime_adapter_h3`` on SGLang's own H3 modules: JSON report.

Block 0, the final layer and the token refiner's final norm of the pinned checkpoint
(``$RL_KERNEL_H3_WEIGHTS``), loaded through SGLang's loader at each ``--tp`` size, one GPU per
rank (``rl_engine.validation.models.h3_sglang_rollout``). For each bound output (the four
projections, the block's AdaLN table, both modulated norms and its output, the final layer's table
and output, and the refiner norms) it records whether SGLang's native forward and the bound forward
equal the training arithmetic bitwise, whether each gives the same rows when it sees only a prefix
of them, and whether each TP size equals TP1 (the reference). At each TP size it also runs
SGLang's whole DiT (``full_model_target``: 2 blocks at H3's widths, seeded weights) and records the
adapter's readback: whether every bound forward ran once per model forward. And it runs the
SGLang plugin through SGLang's own plugin lifecycle (``plugin_target``): entry point, hook,
install, guarded forwards, the Spectrum refusal and a caught bypass.

    python tools/validation/models/h3_sglang_rollout_report.py --out report.json \
        [--rows 261] [--prefix-rows 37] [--tp 1 2 4]
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from rl_engine.validation.distributed.h3_ws2 import run_world
from rl_engine.validation.models import h3_sglang_rollout as hs
from rl_engine.validation.models.h3_weights import load_h3_block_weights

HIDDEN, HEADS, HEAD_DIM, FFN = 5376, 56, 128, 14336


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def full_columns(name: str, shards: list[torch.Tensor]) -> torch.Tensor:
    """TP column shards (``[q_r; k_r; v_r]`` or ``[gate_r; up_r]``) back to the TP1 layout."""

    parts = 3 if name == "qkv" else 2
    split = [s.chunk(parts, dim=1) for s in shards]
    return torch.cat([torch.cat([s[i] for s in split], dim=1) for i in range(parts)], dim=1)


def summarize(results: list[dict], prefix_rows: int) -> dict:
    out: dict = {"loaded": results[0]["loaded"], "report": results[0]["report"], "rows": {}}
    for name in hs.NAMES + hs.NORM_NAMES:
        row: dict = {}
        for rank, r in enumerate(results):
            canon, native, bound = r[f"canonical_{name}"], r[f"native_{name}"], r[f"bound_{name}"]
            row[f"rank{rank}"] = {
                "bound_equals_canonical": bool(torch.equal(bound, canon)),
                "native_equals_canonical": bool(torch.equal(native, canon)),
                "native_max_abs_diff": max_abs(native, canon),
                "restored_equals_native": r[f"restored_equals_native_{name}"],
            }
            if prefix_rows and name not in hs.TABLES:
                row[f"rank{rank}"]["bound_prefix_equals_rows"] = bool(
                    torch.equal(r[f"bound_prefix_{name}"], bound[:prefix_rows])
                )
                row[f"rank{rank}"]["native_prefix_equals_rows"] = bool(
                    torch.equal(r[f"native_prefix_{name}"], native[:prefix_rows])
                )
                row[f"rank{rank}"]["native_prefix_max_abs_diff"] = max_abs(
                    r[f"native_prefix_{name}"], native[:prefix_rows]
                )
        out["rows"][name] = row
    return out


def full_model(ranks: list[dict]) -> dict:
    """The whole-DiT run of one TP size: readback and checks per rank."""

    keys = ("bound_finite", "bound_repeatable", "restored_equals_native", "batching_restored")
    out: dict = {"layers": ranks[0]["layers"], "forwards": ranks[0]["forwards"]}
    for rank, r in enumerate(ranks):
        report = r["report"]
        out[f"rank{rank}"] = {
            "patched": len(report["patched"]),
            "skipped": report["skipped"],
            "readback": r["readback"],
            "calls": report["calls"],
            "overrides": report["overrides"],
            **{k: r[k] for k in keys},
        }
    out["implementations"] = ranks[0]["report"]["implementations"]
    return out


PLUGIN_KEYS = (
    "hooked",
    "installed",
    "forwards",
    "readback",
    "repeatable",
    "readback_file_forwards",
    "readback_file_calls",
    "overrides",
    "spectrum_refused",
    "forwards_after_refusal",
    "bypass_caught",
    "forward_restored",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=261)
    ap.add_argument("--prefix-rows", type=int, default=37)
    ap.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sglang

    weights = load_h3_block_weights("cpu")
    gen = torch.Generator().manual_seed(args.seed)
    inputs = {
        "x": (torch.randn(args.rows, HIDDEN, generator=gen) * 0.5).bfloat16(),
        "a": (torch.randn(args.rows, HEADS * HEAD_DIM, generator=gen) * 0.5).bfloat16(),
        "h": (torch.randn(args.rows, FFN, generator=gen) * 0.5).bfloat16(),
        **hs.block_weight_inputs(weights),
        **hs.norm_inputs(args.rows, seed=args.seed),
    }
    report: dict = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sglang": sglang.__version__,
            "python": platform.python_version(),
        },
        "rows": args.rows,
        "prefix_rows": args.prefix_rows,
        "seed": args.seed,
        "tp": {},
    }
    results = {}
    for tp in args.tp:
        results[tp] = run_world(tp, hs.rollout_target, inputs, prefix_rows=args.prefix_rows)
        report["tp"][str(tp)] = summarize(results[tp], args.prefix_rows)
        report["tp"][str(tp)]["full_model"] = full_model(
            run_world(tp, hs.full_model_target, {}, layers=2, forwards=2, seed=args.seed)
        )
        plugin_dir = hs.write_plugin_dist(str(args.out.parent / "plugin-dist"))
        ranks = run_world(tp, hs.plugin_target, {}, plugin_dir=plugin_dir, seed=args.seed)
        report["tp"][str(tp)]["plugin"] = {
            f"rank{rank}": {k: r.get(k) for k in PLUGIN_KEYS} for rank, r in enumerate(ranks)
        }
    if 1 in results:
        base = results[1][0]
        for tp in (t for t in results if t > 1):
            ranks = results[tp]
            for path in ("bound", "native"):
                invariance = {}
                for name in hs.NAMES + hs.NORM_NAMES:
                    key = f"{path}_{name}"
                    if name in ("qkv", "fc1"):
                        whole = full_columns(name, [r[key] for r in ranks])
                        same = torch.equal(whole, base[key])
                    else:
                        same = all(torch.equal(r[key], base[key]) for r in ranks)
                    invariance[name] = bool(same)
                report["tp"][str(tp)][f"{path}_equals_tp1"] = invariance
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
