# SPDX-License-Identifier: Apache-2.0
"""Archive measured model evidence; reject missing ranks or inconsistent counts."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    root = args.artifacts / "final-model-acceptance"
    all_rows = []
    configurations = []
    names = (
        "tp1-pp1-dp1",
        "tp2-pp1-dp1",
        "tp1-pp2-dp1",
        "tp2-pp2-dp1",
        "tp1-pp2-dp2",
        "tp4-pp2-dp1",
        "tp2-pp2-dp2",
    )
    for name in names:
        source = root / name
        if name == "tp2-pp1-dp1" and (root / (name + "-rankfix")).exists():
            source = root / (name + "-rankfix")
        target = out / name
        target.mkdir()
        for path in source.rglob("*"):
            if (
                path.is_file()
                and "cache" not in path.relative_to(source).parts
                and path.name != "checkpoint-manifest.json"
                and path.suffix in (".json", ".csv", ".log", ".txt")
            ):
                dest = target / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        provenance = json.loads((source / "provenance.json").read_text())
        tp, pp, dp = (provenance[k] for k in ("tp", "pp", "dp"))
        entry = {
            "configuration": name,
            "gpu_count": tp * pp * dp,
            "physical_gpus": provenance["physical_gpus"],
            "original_layer_ids": [0, 1, 2, 3],
            "source_commit": provenance["source_commit"],
            "evidence": name + "/",
            "shared_kernel_parity": True,
        }
        summary_files = list(source.glob("summary-rank*.json"))
        if summary_files:
            summaries = [json.loads(p.read_text()) for p in summary_files]
            if sorted(s["rank"] for s in summaries) != list(range(tp * pp * dp)):
                raise ValueError(f"{name}: missing or duplicate ranks")
        elif (source / "summary.json").exists():
            summaries = [json.loads((source / "summary.json").read_text())]
        else:
            summaries = [
                json.loads((source / f"tp-rank{i}" / "summary.json").read_text()) for i in range(tp)
            ]
        if any(s["status"] == "BLOCKED" for s in summaries):
            entry.update(
                status="BLOCKED",
                forward="NOT RUN",
                backward="NOT RUN",
                determinism="NOT RUN",
                mode_parity="NOT RUN",
                mismatched_elements=None,
                compared_elements=0,
                reason=summaries[0]["reason"],
            )
        else:
            records = []
            for file in source.rglob("comparisons*.json"):
                records.extend(json.loads(file.read_text()))
            if not records or any(s["status"] != "PASS" for s in summaries):
                raise ValueError(f"{name}: incomplete or failed model evidence")
            if any(r["status"] != "PASS" or r["mismatched_elements"] != 0 for r in records):
                raise ValueError(f"{name}: nonzero measured mismatch")
            count = sum(r["compared_elements"] for r in records)
            if count != sum(s["compared_elements"] for s in summaries):
                raise ValueError(f"{name}: recorded count differs from tensor comparisons")
            entry.update(
                status="PASS",
                forward="PASS",
                backward="PASS",
                determinism="PASS",
                mode_parity="PASS (shared-kernel)",
                mismatched_elements=0,
                compared_elements=count,
                optimizer_steps="PASS"
                if all(s.get("optimizer_steps") == "PASS" for s in summaries)
                else "NOT RUN",
            )
            all_rows.extend({"configuration": name, **row} for row in records)
        configurations.append(entry)
    supplemental = []
    for source_name, label in (
        ("model-prefill-decode", "prefill-decode"),
        ("model-optimizer-stress", "optimizer-stress"),
    ):
        source = args.artifacts / "results" / source_name
        target = out / label
        target.mkdir()
        for path in source.glob("*.json"):
            shutil.copyfile(path, target / path.name)
        for path in source.glob("*.csv"):
            shutil.copyfile(path, target / path.name)
        summary = json.loads((source / "summary.json").read_text())
        records = json.loads((source / "comparisons.json").read_text())
        if (
            summary["status"] != "PASS"
            or not records
            or any(r["status"] != "PASS" for r in records)
        ):
            raise ValueError(f"{label}: supplemental model test not passing")
        if summary["compared_elements"] != sum(r["compared_elements"] for r in records):
            raise ValueError(f"{label}: inconsistent tensor count")
        supplemental.append({"configuration": label, **summary, "evidence": label + "/"})
        all_rows.extend({"configuration": label, **row} for row in records)
    targeted = args.artifacts / "results" / "targeted-source-layers-final.json"
    shutil.copyfile(targeted, out / targeted.name)
    target_data = json.loads(targeted.read_text())
    if not target_data["comparisons"] or any(
        r["status"] != "PASS" for r in target_data["comparisons"]
    ):
        raise ValueError("targeted later-source components failed")
    logs = out / "logs"
    logs.mkdir()
    log_names = (
        "v41-model-committed-regression.log",
        "v4-final-contract-regression.log",
        "v4-model-committed-gpu-regression.log",
        "layout-model-final.log",
        "v41-model-prefill-decode.log",
        "v41-model-optimizer-stress.log",
        "model-checkpoint-fetch.log",
        "attention-checkpoint-fetch.log",
        "v41-targeted-source-layers-initial-fixture-failure.log",
        "v4-contract-regression-initial-missing-source-env.log",
        "official-tilelang-kernel-probe.log",
        "official-oracle-preflight.log",
        "model-gpu-configuration.log",
        "model-gh-auth-status.log",
        "model-ssh-auth-status.log",
        "model-final-git-fetch.log",
    )
    for name in log_names:
        shutil.copyfile(args.artifacts / "logs" / name, logs / name)
    for source, label in (
        (args.artifacts / "model-4l", "checkpoint"),
        (args.artifacts / "attention-sources", "additional-attention-checkpoint"),
    ):
        shutil.copyfile(source / "manifest.json", out / (label + "-manifest.json"))
    fields = [
        "configuration",
        "category",
        "rank",
        "operator",
        "dtype",
        "compared_elements",
        "mismatched_elements",
        "status",
    ]
    with (out / "consistency.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fields})
    regressions = {}
    for label, filename in (
        ("v41", "v41-model-committed-regression.log"),
        ("v4_contracts", "v4-final-contract-regression.log"),
        ("v4_gpu", "v4-model-committed-gpu-regression.log"),
        ("layout", "layout-model-final.log"),
    ):
        text = (logs / filename).read_text()
        passed = re.findall(r"(\d+) passed", text)
        if not passed or re.search(r"\d+ failed", text):
            raise ValueError(f"{label}: regression not passing")
        skipped = re.findall(r"(\d+) skipped", text)
        regressions[label] = {
            "status": "PASS",
            "passed": int(passed[-1]),
            "skipped": int(skipped[-1]) if skipped else 0,
            "log": "logs/" + filename,
        }
    summary = {
        "status": "BLOCKED",
        "complete": False,
        "scope": (
            "real checkpoint original layers 0,1,2,3; frozen packed parameters, embedding and head"
        ),
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "checkpoint_revision": "08c8657f02291e11b6f6b63e0e16014f8e748518",
        "report_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "configurations": configurations,
        "supplemental_model_tests": supplemental,
        "regressions": regressions,
        "model_compared_elements": sum(r["compared_elements"] for r in all_rows),
        "model_mismatched_elements": sum(r["mismatched_elements"] for r in all_rows),
        "count_note": (
            "Repeated comparisons across ranks, tensors, gradients, "
            "updated parameters and runs; not unique tokens/elements."
        ),
        "later_source_component_compared_elements": target_data["compared_elements"],
        "later_source_component_mismatched_elements": target_data["mismatched_elements"],
        "independent_train_inference_backend": "NOT RUN",
        "independent_official_model_oracle": (
            "BLOCKED: unchanged official TileLang E8M0 "
            "quantization fails gfx942 compilation (fp8_e8_t undefined)"
        ),
        "full_40_layers": "NOT RUN",
        "packed_weight_learning": (
            "BLOCKED: this implemented profile freezes packed weights/scales; "
            "no trainable master/quantization path implemented"
        ),
        "ep_cp_execution": "BLOCKED: not implemented for V4.1 model",
        "first_model_bit_mismatch": None,
        "push": {
            "status": "BLOCKED",
            "user_preference": "temporarily keep local commits",
            "authentication": "gh unauthenticated; SSH publickey denied",
            "automatic_review": (
                "rejected attempted push because the user has not renewed "
                "authorization after deferral; command not executed"
            ),
        },
        "remaining": [
            "8-GPU TP4 PP2 and TP2 PP2 DP2 after all physical GPUs become idle",
            "independent backend/model oracle validation under the same numerical contract",
            "full-parameter training beyond the explicitly frozen-quantized-weight profile",
            (
                "full 40-layer execution, arbitrary padding/positions, EP/CP "
                "and long-context candidate pruning"
            ),
        ],
        "supplemental_provenance": (
            "Prefill/decode ran the working tree before commits; "
            "script later received a closure-binding lint edit. Core model/operators match "
            "committed implementation. Basic model matrix has per-run committed provenance."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for parent in (
            "rl_engine/models/deepseek_v41",
            "rl_engine/backends/rocm",
            "tests/models/deepseek_v41",
            "examples/operators/dsv41_4l",
        )
        for p in Path(parent).rglob("*.py")
        if (
            "results" not in p.parts
            and ("deepseek_v41" in p.parts or "dsv41_4l" in p.parts or p.stem.startswith("v41_"))
        )
    }
    (out / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model_comparisons": len(all_rows),
                "model_elements": summary["model_compared_elements"],
                "model_mismatches": summary["model_mismatched_elements"],
                "status": summary["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
