# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Charts of the full-transformer and dual-trajectory evidence (rollout vs train drift, accuracy, time)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOOR = 1e-12  # plotted value for a bitwise-equal step


def _per_step(pair: dict) -> dict[int, float]:
    steps = {}
    for entry in pair.get("entries", []):
        steps[entry["step"]] = max(steps.get(entry["step"], 0.0), entry["max_abs_diff"])
    return steps


def main(evidence: Path, out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    series = (
        ("provider", "", "provider", "tab:red"),
        ("RL-Kernel strict", "", "rlkernel_strict", "tab:blue"),
        ("strict + video proj. diagnostic", "_diag", "rlkernel_strict", "tab:green"),
    )
    for label, suffix, key, color in series:
        for i, seed in enumerate((11, 22, 33)):
            path = evidence / f"h3_dual_trajectory_seed{seed}{suffix}.json"
            pair = json.loads(path.read_text())["pairs"].get(key)
            if pair is None:
                continue
            steps = _per_step(pair)
            xs = range(4)
            ax[0].semilogy(
                xs,
                [max(steps.get(s, 0.0), FLOOR) for s in xs],
                "o-",
                color=color,
                alpha=0.8,
                label=label if i == 0 else None,
            )
    ax[0].set(
        xlabel="denoising step",
        ylabel=f"max |single - batched| (floor {FLOOR:g} = bitwise equal)",
        title="Rollout vs train path drift, 3 seeds",
    )
    ax[0].legend(fontsize=7)
    full = json.loads((evidence / "h3_full_transformer_h100.json").read_text())
    cmp_ = full["candidate_vs_provider"]
    names = ["video_rel", "audio_rel"] + [f"d{k}" for k in cmp_["input_grad_rel"]]
    values = [cmp_["video_rel"], cmp_["audio_rel"], *cmp_["input_grad_rel"].values()]
    ax[1].bar(names, values)
    ax[1].set(yscale="log", ylabel="relative difference to provider", title="Strict vs provider (33B)")
    ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    sec = full["seconds"]
    ax[2].bar(
        ["provider", "strict", "provider fwd", "strict fwd"],
        [sec["provider"], sec["candidate"], sec["provider_forward"], sec["candidate_forward"]],
    )
    ax[2].set(ylabel="seconds (H100)", title="Wall time incl. warm-up, tiny request")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
