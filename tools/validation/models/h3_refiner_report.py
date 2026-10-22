# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Accuracy and timing evidence (JSON + PNG) of the H3 token refiner on real weights (H100)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.validation.models import h3_block as hb  # noqa: E402
from rl_engine.validation.models import h3_refiner as hr  # noqa: E402
from rl_engine.validation.models.h3_weights import load_h3_tensors  # noqa: E402


def _median_ms(fn, repeats=5):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - start) * 1e3)
    return statistics.median(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="*", default=[12, 257, 925])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    names = [f"{hr.PREFIX}.0.{leaf}" for leaf in hr.PARAM_LEAVES] + [hr.FINAL_NORM]
    weights = load_h3_tensors(names, "cuda")
    ops = hb.CandidateOps()
    report = {"device": torch.cuda.get_device_name(0), "cases": []}
    for tokens in args.tokens:
        case = hr.run_refiner_case(weights, block=0, tokens=tokens, seed=3, ops=ops)
        ctx = hr.make_refiner_context(weights, block=0, tokens=tokens, seed=3)
        with torch.no_grad():
            cand = _median_ms(lambda: hr.run_graph("candidate", ctx, ops))
            prov = _median_ms(lambda: hr.run_graph("provider", ctx))
        case["candidate_ms"], case["provider_ms"] = cand, prov
        case["nodes"] = [
            {k: v for k, v in n.items() if k != "candidate_sha256"} for n in case["nodes"]
        ]
        report["cases"].append(case)
        print(f"T={tokens}: candidate {cand:.2f} ms provider {prov:.2f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        big = report["cases"][-1]
        xs = range(len(big["nodes"]))
        ax[0].semilogy(xs, [max(n["candidate_rel_err_vs_golden"], 1e-9) for n in big["nodes"]], "o-", label="RL-Kernel")
        ax[0].semilogy(xs, [max(n["provider_rel_err_vs_golden"], 1e-9) for n in big["nodes"]], "x--", label="provider")
        ax[0].set_xticks(list(xs))
        ax[0].set_xticklabels([n["node"] for n in big["nodes"]], rotation=70, ha="right", fontsize=6)
        ax[0].set(ylabel=f"relative error vs FP64 ({big['tokens']} tokens)", title="Accuracy per node")
        ax[0].legend()
        toks = [c["tokens"] for c in report["cases"]]
        ax[1].plot(toks, [c["candidate_ms"] for c in report["cases"]], "o-", label="RL-Kernel")
        ax[1].plot(toks, [c["provider_ms"] for c in report["cases"]], "x--", label="provider")
        ax[1].set(xlabel="tokens", ylabel="forward ms, one refiner block (H100)", title="Performance")
        ax[1].legend()
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)


if __name__ == "__main__":
    main()
