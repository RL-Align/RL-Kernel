# SPDX-License-Identifier: Apache-2.0
"""Plot observed three-update metrics; unsupported native data stay visibly N/A."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- choose the noninteractive backend first.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.summary.read_text())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        x
        for x in d["consistency"]
        if x["group"] == "R/R" and x["comparison_kind"] == "actor_vs_rollout"
    ]
    versions = [x["version"] for x in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key, title in zip(
        axes.flat,
        ["mismatch_count", "mismatch_rate", "mean_abs_diff", "max_abs_diff"],
        [
            "Bitwise mismatch count",
            "Bitwise mismatch rate",
            "Mean absolute raw FP32 logprob difference",
            "Maximum absolute raw FP32 logprob difference",
        ],
        strict=True,
    ):
        ax.plot(versions, [x[key] for x in rows], marker="o", label="R/R: rocm_triton")
        ax.set_xticks(versions)
        ax.set_xlabel("Weight version")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend()
        ax.text(0.03, 0.9, "P/P: BLOCKED (no valid scores)", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Four-layer PP2/DP1: within-run same-version train/rollout consistency")
    fig.tight_layout()
    fig.savefig(a.output_dir / "consistency.png", dpi=170)
    plt.close(fig)
    perf = d["performance"]["R/R"]["versions"]
    steps = [x for x in perf if x["version"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in [
        ("rollout_generate_seconds", "Generate new responses"),
        ("rollout_teacher_seconds", "Rollout teacher scoring"),
        ("actor_logprob_pp_max_rank_seconds", "Actor scoring (max rank)"),
        ("forward_backward_optimizer_pp_max_rank_seconds", "Training F/B + optimizer"),
        ("weight_sync_seconds", "Weight sync"),
    ]:
        axes[0].plot(
            [x["version"] for x in steps], [x[key] for x in steps], marker="o", label=label
        )
    axes[1].bar(
        [x["version"] for x in steps],
        [x["full_diagnostic_acceptance_cycle_seconds"] for x in steps],
        color="#2F67D8",
    )
    for ax in axes:
        ax.set_xlabel("Continuous update")
        ax.set_ylabel("Seconds")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.2)
    axes[0].set_title("Observed phase times (not an additive decomposition)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Full cycle with trace/hash/audit/I/O overhead")
    fig.suptitle("R/R only; P/P blocked — no speedup comparison")
    fig.tight_layout()
    fig.savefig(a.output_dir / "phase-times.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        [x["version"] for x in steps],
        [x["rollout_response_tokens_per_gpu_second"] for x in steps],
        marker="o",
        label="Generated response tokens / (s × 2 rollout GPUs)",
    )
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xlabel("Continuous update")
    axes[0].set_ylabel("Response tokens/GPU/s")
    axes[0].legend(fontsize=8)
    peaks = d["performance"]["R/R"]["sampled_peak_vram_bytes"]
    axes[1].bar(
        list(peaks),
        [x / 1024**3 for x in peaks.values()],
        color=["#2F67D8", "#2F67D8", "#E28A2B", "#E28A2B"],
    )
    axes[1].set_xlabel("Physical GPU (actor 4/5; rollout 6/7)")
    axes[1].set_ylabel("GiB")
    axes[1].set_title("Observed resident VRAM peak (~1 s polling)")
    fig.suptitle("Diagnostic workload only; P/P N/A; no reward or performance-benefit claim")
    fig.tight_layout()
    fig.savefig(a.output_dir / "throughput-memory.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
