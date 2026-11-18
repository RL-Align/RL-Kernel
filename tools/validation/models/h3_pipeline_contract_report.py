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
  SGLang's packed-sequence builders, when they are importable;
* ``identity`` (H16): checkpoint component hashes of both variants and, with
  ``--transformer-dir``, which variant a local transformer directory holds;
* ``timing``: layout construction time of ours, Diffusers and SGLang at the released size;
* row-specific findings.

The row logic lives in ``h3_fl2va_report.py`` / ``h3_ref2va_report.py`` next to this file.
"""

from __future__ import annotations

import argparse
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

ROWS = {"fl2va": "h3_fl2va_report", "ref2va": "h3_ref2va_report"}


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
            variant: identity.recipe_sha256() for variant, identity in pl.PIPELINES.items()
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
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "runs": repeats}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--row", choices=sorted(ROWS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--png")
    parser.add_argument("--transformer-dir")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--replot", help="only redraw --png from this existing report JSON")
    args = parser.parse_args()
    if args.replot:
        plot(json.loads(Path(args.replot).read_text()), args.png)
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
    report["passed"] = all(v["identical"] for v in report["repeatability"].values()) and all(
        v["bitwise_equal"] == v["cases"]
        for v in report["reference"].values()
        if isinstance(v, dict) and "cases" in v
    )
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.png:
        plot(report, args.png)
    print(json.dumps({"row": report["row"], "passed": report["passed"]}))


if __name__ == "__main__":
    main()
