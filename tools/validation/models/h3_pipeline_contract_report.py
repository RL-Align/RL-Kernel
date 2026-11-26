# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Evidence for RFC #420 Phase B ``fl2va_pipeline_contract`` / ``ref2va_pipeline_contract``.

    python tools/validation/models/h3_pipeline_contract_report.py --row fl2va \
        --out fl2va-report.json --png fl2va-report.png [--transformer-dir <.../transformer>]

One JSON per row, with:

* ``repeatability`` (H-C7): every case is preprocessed several times in this process (default
  threads and one thread) and in fresh processes with other thread counts; the sealed
  packed-layout artifacts must be identical;
* ``reference``: case counts and bitwise results against the pinned Diffusers builders and
  SGLang's packed-sequence builders, when they are importable. A Diffusers comparison counts
  for the verdict only when the imported Diffusers source is the pinned one
  (``environment.provider_source_sha256`` equals ``PROVIDER_SOURCE_SHA256``);
* ``identity`` (H16): checkpoint component hashes of both variants and, with
  ``--transformer-dir``, which variant a local transformer directory holds;
* ``timing``: layout construction time of ours, Diffusers and SGLang at the released size;
* row-specific findings.

The row logic lives in ``h3_fl2va_report.py`` / ``h3_ref2va_report.py`` next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rl_engine.models.minimax_h3 import packed_layout as pl  # noqa: E402

# Each row's module ships with its own PR (FL2VA #724, Ref2VA #725), so only the rows whose module
# is next to this file are offered: ``--row`` refuses the others instead of failing on import.
ROW_MODULES = {"fl2va": "h3_fl2va_report", "ref2va": "h3_ref2va_report"}
ROWS = {
    row: module
    for row, module in ROW_MODULES.items()
    if (Path(__file__).resolve().parent / f"{module}.py").is_file()
}

# The Diffusers files the references run, at the pin (pl.PROVIDER_REVISION). Hashing the files
# works for a git checkout and a source archive alike; an installed release (0.43.0, say) has
# other files and does not match.
PROVIDER_SOURCE_FILES = ("modular_pipelines/minimax_h3/*.py", "image_processor.py")
PROVIDER_SOURCE_SHA256 = "b959c0ce2290be529e09ea379365786ffc1946dd950e82db96f5b938a48fd376"


def provider_source_sha256() -> str | None:
    """sha256 over the name and bytes of the imported Diffusers' ``PROVIDER_SOURCE_FILES``."""

    try:
        import diffusers
        from diffusers.modular_pipelines import minimax_h3  # noqa: F401
    except Exception:
        return None
    root = Path(diffusers.__file__).resolve().parent
    digest = hashlib.sha256()
    for pattern in PROVIDER_SOURCE_FILES:
        for path in sorted(root.glob(pattern)):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _version(module: str) -> str | None:
    try:
        return str(importlib.import_module(module).__version__)
    except Exception:
        return None


def environment() -> dict[str, Any]:
    import numpy
    import PIL
    import torch

    git = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    gpus = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return {
        "commit": git,
        "tracked_tree_dirty": dirty,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "torchaudio": _version("torchaudio"),
        "diffusers": _version("diffusers"),
        "sglang": _version("sglang"),
        "gpus": gpus,
        "provider_pin": f"{pl.PROVIDER}@{pl.PROVIDER_REVISION}",
        "provider_source_sha256": provider_source_sha256(),
        "checkpoint_pin": "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe",
    }


def repeatability(row: str, cases: list[str], *, repeats: int = 3) -> dict[str, Any]:
    """Seal each case ``repeats`` times here, once on one thread, and in two fresh processes."""

    module = importlib.import_module(ROWS[row])
    import torch

    local: dict[str, list[str]] = {case: [] for case in cases}
    artifacts: dict[str, dict] = {}
    for case in cases:
        for _ in range(repeats):
            artifact = module.preprocess(case)
            local[case].append(artifact["artifact_sha256"])
            artifacts[case] = artifact
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            local[case].append(module.preprocess(case)["artifact_sha256"])
        finally:
            torch.set_num_threads(threads)
    remote: dict[str, list[str]] = {case: [] for case in cases}
    for threads in ("2", "8"):
        code = (
            "import json, sys; sys.path[:0] = sys.argv[1:3];"
            f"import {ROWS[row]} as m;"
            "print(json.dumps({c: m.preprocess(c)['artifact_sha256'] for c in sys.argv[3:]}))"
        )
        env = {**os.environ, "OMP_NUM_THREADS": threads, "PYTHONHASHSEED": threads}
        out = subprocess.run(
            [sys.executable, "-c", code, str(REPO), str(Path(__file__).parent), *cases],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        for case, sha in json.loads(out.stdout.strip().splitlines()[-1]).items():
            remote[case].append(sha)
    result = {}
    for case in cases:
        shas = local[case] + remote[case]
        artifact = artifacts[case]
        result[case] = {
            "runs": len(shas),
            "identical": len(set(shas)) == 1,
            "artifact_sha256": artifact["artifact_sha256"],
            "sequence_length": artifact["sequence_length"],
            "layout_component_sha256": artifact["layout_component_sha256"],
            "input_sha256": artifact["input_sha256"],
            "pipeline_identity": artifact["pipeline_identity"],
        }
    return result


def identity(transformer_dir: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "checkpoint_component_sha256": {
            variant: pl.checkpoint_component_hashes(variant) for variant in pl.PIPELINES
        },
        "preprocessing_recipe_sha256": {
            variant: spec.recipe_sha256() for variant, spec in pl.PIPELINES.items()
        },
        "transformer_shards_differ": sum(
            pl.transformer_shard_hashes("FL2VA")[name] != sha
            for name, sha in pl.transformer_shard_hashes("Ref2VA").items()
        ),
    }
    if transformer_dir:
        try:
            out["local_transformer_dir"] = {
                "identified_as": pl.identify_transformer_dir(transformer_dir),
                "source": "Hub download metadata or cache blob names (no shard read)",
            }
        except Exception as exc:  # recorded, not hidden
            out["local_transformer_dir"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def time_call(fn: Callable[[], Any], *, repeats: int = 15) -> dict[str, float]:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "runs": repeats,
    }


def plot(report: dict[str, Any], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = report["reference"]
    timing = report["timing"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [k for k, v in reference.items() if isinstance(v, dict) and "cases" in v]
    cases = [reference[k]["cases"] for k in names]
    equal = [reference[k]["bitwise_equal"] for k in names]
    y = range(len(names))
    left.barh([i + 0.2 for i in y], cases, height=0.4, color="#9aa5b1", label="cases")
    left.barh([i - 0.2 for i in y], equal, height=0.4, color="#2f6f9f", label="bitwise equal")
    left.set_yticks(list(y), names)
    left.set_xlabel("cases")
    left.set_title(f"{report['row']}: layout vs prior art", fontsize=10)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    impls = list(timing)
    right.bar(impls, [timing[k]["median_ms"] for k in impls], color="#2f6f9f")
    right.set_ylabel("median ms per layout")
    right.set_title(timing[impls[0]].get("label", "layout build, released size"), fontsize=10)
    for side in (left, right):
        side.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# One chart per Results table in the PR body (light surface, validated two-hue palette).
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")
REFERENCE_LABELS = {
    "diffusers_static": "Diffusers static builder",
    "diffusers_media": "Diffusers media steps",
    "sglang_prefix": "SGLang (unpadded prefix)",
}
TIMING_LABELS = {
    "rl_kernel": "RL-Kernel (ours)",
    "diffusers": "Diffusers",
    "sglang": "SGLang",
}


def _axes(width: float, height: float, title: str, subtitle: str) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=8.5, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    import matplotlib.pyplot as plt

    plt.close(fig)


def compared_references(report: dict[str, Any]) -> list[str]:
    """The reference entries that ran a comparison (a skipped one has no ``cases``)."""

    return [k for k, v in report["reference"].items() if isinstance(v, dict) and "cases" in v]


def verdict(report: dict[str, Any]) -> bool:
    """Every case repeatable, at least one reference compared, and every compared case equal.

    With Diffusers and SGLang both missing nothing is compared, which is not a pass. Neither is
    a compared reference that ran zero cases, nor a report with no repeatability case, nor a
    Diffusers comparison against another Diffusers than the pinned one.
    """

    compared = compared_references(report)
    source = report["environment"].get("provider_source_sha256")
    return (
        all(source == PROVIDER_SOURCE_SHA256 for k in compared if k.startswith("diffusers"))
        and bool(report["repeatability"])
        and all(v["identical"] for v in report["repeatability"].values())
        and bool(compared)
        and all(
            report["reference"][k]["cases"] > 0
            and report["reference"][k]["bitwise_equal"] == report["reference"][k]["cases"]
            for k in compared
        )
    )


def charts(report: dict[str, Any], out_dir: str) -> list[str]:
    """Write one PNG per Results table: equality, timing, repeatability, row finding."""

    from matplotlib.ticker import FuncFormatter

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = report["row"].split("_")[0]
    name = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}[row]
    env = report["environment"]
    gpu = env["gpus"][0] if env["gpus"] else "CPU"
    where = f"{gpu}, commit {env['commit'][:8]}, {row}-report.json"
    written = []

    # Byte equality with prior art: cases per reference, all bitwise equal or not.
    reference = report["reference"]
    keys = compared_references(report)
    if keys:  # nothing to draw when every reference was skipped
        fig, ax = _axes(
            7.5,
            0.9 + 0.55 * len(keys),
            f"{name} layout vs prior art: bitwise-equal cases ({gpu})",
            where,
        )
        y = list(range(len(keys)))[::-1]
        equal = [reference[k]["bitwise_equal"] for k in keys]
        differ = [reference[k]["cases"] - reference[k]["bitwise_equal"] for k in keys]
        ax.barh(y, equal, height=0.5, color=SERIES[0], label="bitwise equal")
        ax.barh(y, differ, left=equal, height=0.5, color=SERIES[1], label="differ")
        for i, k in zip(y, keys, strict=True):
            ax.text(
                reference[k]["cases"],
                i,
                f"  {reference[k]['bitwise_equal']} / {reference[k]['cases']} equal, "
                f"{reference[k]['cases'] - reference[k]['bitwise_equal']} differ",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        ax.set_yticks(y, [REFERENCE_LABELS.get(k, k) for k in keys])
        ax.set_xlabel("cases", color=INK_2, fontsize=8.5)
        ax.set_xlim(0, max(reference[k]["cases"] for k in keys) * 1.45)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        path = out / f"{row}-equality.png"
        _save(fig, path)
        written.append(path.name)

    # Layout build time: median bar, minimum marker; only the implementations that ran.
    timing = report["timing"]
    impls = [k for k in TIMING_LABELS if k in timing]
    label = timing["rl_kernel"].get("label", "layout build")
    fig, ax = _axes(
        7.5,
        3.2,
        f"{name} layout build time, released size ({gpu} host, CPU)",
        f"{label}\n{where}",
    )
    x = list(range(len(impls)))
    med = [round(timing[k]["median_ms"], 2) for k in impls]
    low = [round(timing[k]["min_ms"], 2) for k in impls]
    ax.bar(
        x,
        med,
        width=0.5,
        color=SERIES[0],
        label=f"median of {timing['rl_kernel']['runs']} runs",
    )
    ax.scatter(
        x,
        low,
        s=64,
        color=SERIES[1],
        zorder=3,
        label="minimum",
        edgecolors=SURFACE,
        linewidths=2,
    )
    for i, (m, lo) in enumerate(zip(med, low, strict=True)):
        ax.text(i, m, f"median {m:.2f}\n", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.text(i + 0.28, lo, f"min {lo:.2f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, [TIMING_LABELS[k] for k in impls])
    ax.set_ylabel("ms per layout", color=INK_2, fontsize=8.5)
    ax.set_ylim(0, max(med) * 1.3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    path = out / f"{row}-timing.png"
    _save(fig, path)
    written.append(path.name)

    # H-C7: rows per case, every case identical across its runs.
    rep = report["repeatability"]
    cases = list(rep)
    runs = {rep[c]["runs"] for c in cases}
    same = sum(rep[c]["identical"] for c in cases)
    fig, ax = _axes(
        7.5,
        1.0 + 0.3 * len(cases),
        f"{name} H-C7: sealed artifact identical in {same} / {len(cases)} cases ({gpu} host)",
        f"rows per case; {'/'.join(map(str, sorted(runs)))} runs each; {where}",
    )
    y = list(range(len(cases)))[::-1]
    rows = [rep[c]["sequence_length"] for c in cases]
    ax.barh(
        y,
        rows,
        height=0.6,
        color=[SERIES[0] if rep[c]["identical"] else SERIES[1] for c in cases],
    )
    for i, c in zip(y, cases, strict=True):
        verdict = "identical" if rep[c]["identical"] else "DIFFER"
        ax.text(
            rows[cases.index(c)],
            i,
            f"  {rows[cases.index(c)]:,}  ({rep[c]['runs']} runs {verdict})",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(y, cases)
    ax.set_xlabel("packed rows", color=INK_2, fontsize=8.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(0, max(rows) * 1.45)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    path = out / f"{row}-repeatability.png"
    _save(fig, path)
    written.append(path.name)

    # Ref2VA finding: rows whose position differs between the two summations.
    span = report["findings"].get("video_span_summation")
    if span:
        req = span["random_requests"]
        bars = [
            ("rows sampled", req["rows"]),
            ("differ in fp64", req["rows_with_fp64_difference"]),
            ("differ after fp32 cast", req["rows_with_fp32_difference_after_cast"]),
        ]
        fig, ax = _axes(
            7.5,
            2.6,
            f"{name} span summation, neumaier vs sequential, {req['layouts']} layouts ({gpu} host)",
            f"{span['interpreter']}; {where}",
        )
        y = list(range(len(bars)))[::-1]
        ax.barh(y, [v for _, v in bars], height=0.5, color=[SERIES[0], SERIES[1], SERIES[1]])
        for i, (_, v) in zip(y, bars, strict=True):
            text = f"  {v:,}" if v else "  = 0, bitwise equal"
            ax.text(v, i, text, va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [b for b, _ in bars])
        ax.set_xlabel("rows", color=INK_2, fontsize=8.5)
        ax.set_xlim(0, req["rows"] * 1.35)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
        path = out / f"{row}-span-summation.png"
        _save(fig, path)
        written.append(path.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--row", choices=sorted(ROWS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--png")
    parser.add_argument("--transformer-dir")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--charts", help="also write one PNG per Results table into this dir")
    parser.add_argument("--replot", help="only redraw --png / --charts from this report JSON")
    args = parser.parse_args()
    if args.replot:
        measured = json.loads(Path(args.replot).read_text())
        if args.png:
            plot(measured, args.png)
        if args.charts:
            print(json.dumps(charts(measured, args.charts)))
        return
    sys.path.insert(0, str(Path(__file__).parent))
    module = importlib.import_module(ROWS[args.row])
    report: dict[str, Any] = {
        "schema": "rlkernel.h3.pipeline_contract_report.v1",
        "row": f"{args.row}_pipeline_contract",
        "environment": environment(),
        "repeatability": repeatability(args.row, module.CASES, repeats=args.repeats),
        "reference": module.reference(),
        "identity": identity(args.transformer_dir),
        "timing": module.timing(time_call),
        "findings": module.findings(),
    }
    report["passed"] = verdict(report)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.png:
        plot(report, args.png)
    if args.charts:
        charts(report, args.charts)
    print(json.dumps({"row": report["row"], "passed": report["passed"]}))


if __name__ == "__main__":
    main()
