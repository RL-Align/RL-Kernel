#!/usr/bin/env python3
# isort: skip_file
# SPDX-License-Identifier: Apache-2.0
"""Export review figures and CSV from raw audit reports (no recomputation)."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.models.qwenimage.streams import STREAMS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
modes = [
    "native",
    "cublas_policy",
    "fixed",
    "vllm_local",
    "sglang_local",
    "megatron_local",
    "vllm_liger",
    "vllm_flashattn",
]
labels = [
    "Native",
    "cuBLAS policy",
    "Fixed CUDA",
    "vLLM GEMM adapter",
    "SGLang GEMM adapter",
    "Megatron GEMM adapter",
    "vLLM + Liger adapter",
    "vLLM + flash-attn LN adapter",
]
keys = [(s, "torch." + d) for d in ["bfloat16", "float32"] for s, _ in STREAMS]
reports = {}
for mode in modes:
    for phase in ["reuse", "bi", "accuracy", "workload"]:
        path = a.directory / f"{mode}-{phase}.json"
        if path.exists():
            reports[mode, phase] = {
                (c["stream"], c["dtype"]): c for c in json.loads(path.read_text())["cases"]
            }
rows = []
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), layout="constrained")
for ax, component, title in zip(
    axes,
    ["output", "temb"],
    ["Forward: singleton vs full batch", "d_temb: singleton vs full batch"],
):
    data = np.full((6, len(modes)), np.nan)
    annotations = {}
    for j, mode in enumerate(modes):
        for i, key in enumerate(keys):
            case = reports.get((mode, "bi"), {}).get(key)
            if case is None:
                continue
            checks = [c for c in case["checks"] if c["kind"] == "every_row_singleton"]
            n = sum(c["rows"] for c in checks)
            bad = sum(c["mismatched_rows"][component] for c in checks)
            data[i, j] = bad / n if n else np.nan
            annotations[i, j] = f"{bad:,}/{n:,}"
            rows.append(
                dict(
                    mode=mode,
                    stream=key[0],
                    dtype=key[1],
                    metric=component + "_BI_rows",
                    bad=bad,
                    total=n,
                )
            )
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(modes)), labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(keys)), [f'{s} / {d.removeprefix("torch.")}' for s, d in keys])
    ax.set_title(title)
    for i in range(len(keys)):
        for j in range(len(modes)):
            ax.text(
                j,
                i,
                annotations.get((i, j), "not tested"),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if np.isfinite(data[i, j]) and (data[i, j] < 0.1 or data[i, j] > 0.9)
                    else "black"
                ),
            )
fig.suptitle(
    "Qwen-Image / B200 — raw-byte BI mismatches (lower is better)\n"
    "3 seeds; every row at B=64,257,2048. This panel alone is not the full BI verdict.",
    fontsize=13,
)
fig.savefig(a.directory / "bi.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
colors = [
    "#64748b",
    "#eab308",
    "#2563eb",
    "#9333ea",
    "#059669",
    "#ea580c",
    "#db2777",
    "#0891b2",
]
width = 0.8 / len(modes)
for ax, batch, title in zip(
    axes[0],
    [1, 2048],
    ["Small batch: full forward + backward", "Large batch: full forward + backward"],
):
    for j, mode in enumerate(modes):
        vals = []
        lows = []
        highs = []
        for key in keys:
            case = reports.get((mode, "reuse"), {}).get(key)
            match = (
                next((x for x in case["performance"] if x["batch"] == batch), None)
                if case
                else None
            )
            samples = match["measurements"]["forward_backward"]["samples_ms"][mode] if match else []
            vals.append(np.median(samples) if samples else np.nan)
            lows.append(np.percentile(samples, 10) if samples else np.nan)
            highs.append(np.percentile(samples, 90) if samples else np.nan)
        vals = np.array(vals)
        x = np.arange(len(keys)) + (j - (len(modes) - 1) / 2) * width
        ax.bar(x, vals, width=0.95 * width, label=labels[j], color=colors[j])
        ax.errorbar(
            x,
            vals,
            yerr=np.array([vals - np.array(lows), np.array(highs) - vals]),
            fmt="none",
            ecolor="black",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_ylabel("ms (median; whiskers p10–p90)")
    ax.set_title(title)
    ax.set_xticks(
        range(len(keys)), [f'{s}\n{d.removeprefix("torch.")}' for s, d in keys], fontsize=9
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0, 0].legend(fontsize=8)
ax = axes[1, 0]
for j, mode in enumerate(modes):
    vals = []
    for key in keys:
        c = reports.get((mode, "reuse"), {}).get(key)
        vals.append(
            max(v for k, v in c["fp64_relative_max_error"].items() if k != "output")
            if c
            else np.nan
        )
    ax.bar(
        np.arange(len(keys)) + (j - (len(modes) - 1) / 2) * width,
        vals,
        width=0.95 * width,
        color=colors[j],
    )
ax.set_yscale("log")
ax.set_ylabel("max |error| / max |FP64 reference|")
ax.set_title("Worst gradient normalized error at B=3, S=3")
ax.set_xticks(range(len(keys)), [f'{s}\n{d.removeprefix("torch.")}' for s, d in keys], fontsize=9)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax = axes[1, 1]
ax.axis("off")
lines = ["Full-BI checks beyond the heatmap:"]
for mode, label in zip(modes, labels):
    cs = list(reports.get((mode, "bi"), {}).values())
    n = sum(len(c["checks"]) for c in cs)
    bad = sum(
        any(x.get("mismatched_rows", {}).values()) or not all(x.get("all_equal", {}).values())
        for c in cs
        for x in c["checks"]
    )
    lines.append(f"{label}: {bad}/{n} check records fail")
lines += [
    "",
    "Latency includes Python launch gaps; separate processes.",
    "Small-batch final: S=33; large-batch final: S=3.",
    "Raw samples include fwd, bwd, total and peak memory.",
    "FP64 uses explicit storage casts and straight-through VJP.",
    "Strict elementwise tolerance results: *-accuracy.json.",
    "Source commit and file hashes are recorded in each report.",
]
ax.text(0, 1, "\n".join(lines), va="top", fontsize=11, linespacing=1.6)
fig.suptitle("Reuse audit: speed does not replace accuracy and BI acceptance", fontsize=15)
fig.savefig(a.directory / "reuse.png", dpi=180)
plt.close(fig)
with (a.directory / "bi-summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Strict acceptance is deliberately separate from normalized error at small B.
fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
for ax, dtype in zip(axes, ["torch.bfloat16", "torch.float32"]):
    axis_keys = [(s, b) for s, _ in STREAMS for b in [256, 2048]]
    for j, mode in enumerate(["fixed", "vllm_local"]):
        path = a.directory / f"{mode}-accuracy_large.json"
        if not path.exists():
            continue
        cases = json.loads(path.read_text())["cases"]
        vals, bads = [], []
        for stream, batch in axis_keys:
            checks = [
                x
                for c in cases
                if c["stream"] == stream and c["dtype"] == dtype
                for x in c["checks"]
                if x["batch"] == batch
            ]
            vals.append(
                max(
                    (v["max_tolerance_ratio"] for x in checks for v in x["contract"].values()),
                    default=np.nan,
                )
            )
            bads.append(sum(v["violations"] for x in checks for v in x["contract"].values()))
        xs = np.arange(len(axis_keys)) + (j - 0.5) * 0.35
        ax.bar(
            xs,
            vals,
            0.33,
            label=["Fixed CUDA", "vLLM GEMM adapter"][j],
            color=["#2563eb", "#9333ea"][j],
        )
        for x, val, bad in zip(xs, vals, bads):
            if np.isfinite(val):
                ax.text(
                    x,
                    val + 0.025,
                    f"{val:.2f}\n{bad} fail",
                    ha="center",
                    fontsize=8,
                    color="#b91c1c" if bad else "#1f2937",
                )
    ax.axhline(1, color="#b91c1c", linestyle="--", label="Acceptance limit")
    ax.set_xticks(range(len(axis_keys)), [f"{s}\nB={b}" for s, b in axis_keys])
    ax.set_title(dtype.removeprefix("torch."))
    ax.set_ylabel("max error / (atol + rtol × |reference|)")
    ax.set_ylim(top=max(1.25, ax.get_ylim()[1] * 1.2))
    ax.legend(fontsize=9)
fig.suptitle("Large-batch FP64 check: BI passes do not imply numerical acceptance", fontsize=14)
fig.savefig(a.directory / "accuracy.png", dpi=180)
plt.close(fig)
