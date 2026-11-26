# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render index_select benchmark JSON as a Markdown table and one PNG per workload."""

import argparse
import json
from pathlib import Path

BACKENDS = {
    "torch": "PyTorch",
    "torch_deterministic": "PyTorch deterministic",
    "triton": "Triton",
}
PANELS = {
    "forward": "Forward",
    "backward": "Backward",
    "forward_backward": "Forward + backward",
    "incremental_peak_bytes": "Forward + backward peak",
}


def panel_rows(results, phase):
    """Group backend measurements by the layouts relevant to this phase."""
    rows = {}
    for result in results:
        if phase not in result["timings"]:
            continue
        # Reports predating the layout sweep only measured contiguous inputs.
        x = "C" if result.get("x_contiguous", True) else "S"
        grad = "C" if result.get("grad_y_contiguous", True) else "S"
        layout = (
            "—" if phase == "backward" else x,
            "—" if phase == "forward" else grad,
        )
        timing = result["timings"][phase]
        if phase == "incremental_peak_bytes":
            values = (timing / 2**20,)
        else:
            values = (timing["median_wall_ms"] * 1000, timing["median_event_ms"] * 1000)
        rows.setdefault(layout, {})[result["implementation"]] = values
    return rows


def render_workload(results, title, image_path):
    """Use identical rows for the table and the corresponding grouped bars."""
    from matplotlib import pyplot as plt

    table = [
        "| Phase | x | grad_y | " + " | ".join(BACKENDS.values()) + " |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (phase, label) in zip(axes.flat, PANELS.items(), strict=True):
        rows = panel_rows(results, phase)
        memory = phase == "incremental_peak_bytes"
        for (x, grad), backends in rows.items():
            cells = [" / ".join(f"{value:.3f}" for value in backends[key]) for key in BACKENDS]
            table.append(f"| {label} | {x} | {grad} | " + " | ".join(cells) + " |")
        ticks = range(len(rows))
        colors = ("#326c99", "#be762c", "#18836b")
        for i, (backend, color) in enumerate(zip(BACKENDS, colors, strict=True)):
            ax.bar(
                [tick + (i - 1) * 0.25 for tick in ticks],
                [backends[backend][0] for backends in rows.values()],
                width=0.23,
                color=color,
                label=BACKENDS[backend],
            )
        ax.set_title(label)
        ax.set_xticks(list(ticks), [" / ".join(v for v in layout if v != "—") for layout in rows])
        ax.set_xlabel({"forward": "x", "backward": "grad_y"}.get(phase, "x / grad_y"))
        ax.set_ylabel("Extra allocated MiB" if memory else "Median wall time (µs)")
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{title}\nC = contiguous     S = stride-two view", y=0.98)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.79, hspace=0.55, wspace=0.27)
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    return table


def write_report(paths, output_dir):
    """Render completed runs without requiring PyTorch, Triton or a GPU."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Index selection by tensor layout",
        "",
        "C means contiguous. S is a view with storage stride two on its innermost axis. "
        "The index vector is contiguous throughout.",
        "",
        "Forward varies x. The isolated backward uses a retained graph built with contiguous x "
        "and varies grad_y. Combined measurements vary both layouts.",
        "",
        "Timing cells give median synchronized wall / device-event time in µs in eager mode. "
        "Peak cells give extra allocated MiB for a combined call above the live inputs and graph.",
    ]
    for run, path in enumerate(paths, 1):
        report = json.loads(path.read_text())
        if not report.get("complete") or not report["results"]:
            raise ValueError(f"Benchmark has incomplete results: {path}")
        groups = {}
        for result in report["results"]:
            groups.setdefault((tuple(result["shape"]), result["dim"]), []).append(result)
        env, args = report["environment"], report["arguments"]
        lines.extend(
            [
                "",
                f"Measured on {env['gpu']} with PyTorch {env['pytorch']} "
                f"and Triton {env['triton']}. "
                f"{args['warmup']} warmups and {args['samples']} samples per timing.",
            ]
        )
        for (shape, dim), results in groups.items():
            size = " × ".join(map(str, shape))
            title = f"{env['gpu']} · {size} · dim {dim} · {args['dtype']}"
            name = f"run{run}-{'x'.join(map(str, shape))}-dim{dim}-{args['dtype']}.png"
            table = render_workload(results, title, output_dir / name)
            lines.extend(["", f"## {title}", "", *table, "", f"![{title}]({name})"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/index-select"))
    args = parser.parse_args()
    write_report(args.reports, args.output_dir)
