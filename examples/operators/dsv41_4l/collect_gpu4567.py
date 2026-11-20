# SPDX-License-Identifier: Apache-2.0
"""Archive the requested 4/5/6/7 model matrix with verified per-rank GPU identity."""

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from examples.operators.dsv41_4l.gpu_visibility import validate_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    all_rows = []
    matrix = []
    configurations = (
        "tp1-pp1-dp1",
        "tp2-pp1-dp1",
        "tp1-pp2-dp1",
        "tp2-pp2-dp1",
        "tp1-pp2-dp2",
        "tp1-pp1-dp1-case1",
    )
    for name in configurations:
        source = args.artifacts / "runs" / name
        target = out / name
        target.mkdir()
        for file in source.rglob("*"):
            if (
                file.is_file()
                and "cache" not in file.relative_to(source).parts
                and file.name != "checkpoint-manifest.json"
                and file.suffix in (".json", ".csv", ".log", ".txt")
            ):
                dest = target / file.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, dest)
        provenance = json.loads((source / "provenance.json").read_text())
        mapping = json.loads((source / "gpu-mapping.json").read_text())
        env = {"ROCR_VISIBLE_DEVICES": mapping["rocr_visible_devices"]}
        validate_mapping(mapping, mapping["observed"], env)
        assert mapping["status"] == "PASS" and mapping["allocation_roundtrip"]
        assert all(v is None for v in mapping["secondary_filters"].values())
        tp, pp, dp = (provenance[key] for key in ("tp", "pp", "dp"))
        world = tp * pp * dp
        summaries = []
        for logical in range(world):
            if pp == 2:
                summary_path = source / f"summary-rank{logical}.json"
                rows_path = source / f"comparisons-rank{logical}.json"
            elif tp > 1:
                summary_path = source / f"tp-rank{logical}" / "summary.json"
                rows_path = summary_path.with_name("comparisons.json")
            else:
                summary_path = source / "summary.json"
                rows_path = source / "comparisons.json"
            summary = json.loads(summary_path.read_text())
            records = json.loads(rows_path.read_text())
            assignment = summary["gpu_assignment"]
            assert summary["status"] == "PASS" and records
            assert assignment["assigned_logical_gpu"] == logical
            assert assignment["assigned_physical_gpu"] == mapping["physical_gpus"][logical]
            assert assignment["physical_gpus"] == mapping["physical_gpus"]
            assert all(v is None for v in assignment["secondary_filters"].values())
            validate_mapping(assignment, assignment["observed"], env)
            assert sum(r["compared_elements"] for r in records) == summary["compared_elements"]
            assert all(r["status"] == "PASS" and r["mismatched_elements"] == 0 for r in records)
            for record in records:
                all_rows.append(
                    {
                        "configuration": name,
                        "physical_gpu_ids": ",".join(map(str, mapping["physical_gpus"])),
                        "logical_gpu_ids": ",".join(map(str, mapping["logical_gpus"])),
                        "assigned_physical_gpu": assignment["assigned_physical_gpu"],
                        "assigned_logical_gpu": logical,
                        **record,
                    }
                )
            summaries.append(summary)
        counts = {}
        for category in ("deterministic", "gradient", "logprob", "mode_parity"):
            matched = []
            for row in (r for r in all_rows if r["configuration"] == name):
                c = row["category"]
                include = {
                    "deterministic": "deterministic" in c or "replay" in c,
                    "gradient": "backward" in c or "gradient" in c,
                    "logprob": row["operator"] == "raw_token_logprobs",
                    "mode_parity": "train_inference" in c or "updated_weight_modes" in c,
                }[category]
                if include:
                    matched.append(row)
            assert matched, f"{name}: missing {category} evidence"
            counts[category] = {
                "compared_elements": sum(r["compared_elements"] for r in matched),
                "mismatched_elements": sum(r["mismatched_elements"] for r in matched),
            }
        matrix.append(
            {
                "configuration": name,
                "status": "PASS",
                "physical_gpu_ids": mapping["physical_gpus"],
                "logical_gpu_ids": mapping["logical_gpus"],
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "original_layers": [0, 1, 2, 3],
                "source_commit": provenance["source_commit"],
                "forward": "PASS",
                "backward": "PASS",
                "mapping": "PASS",
                "checks": counts,
                "shared_kernel_parity": True,
                "compared_elements": sum(s["compared_elements"] for s in summaries),
                "mismatched_elements": sum(s["mismatched_elements"] for s in summaries),
                "dp_inputs": summaries[0].get("dp_batch"),
                "optimizer_steps": all(s.get("optimizer_steps") == "PASS" for s in summaries)
                if pp == 2
                else False,
            }
        )
    supplemental = []
    for name in ("prefill-decode", "optimizer-stress"):
        source = args.artifacts / name
        target = out / name
        target.mkdir()
        for path in source.glob("*"):
            if path.is_file() and path.suffix in (".json", ".csv", ".log"):
                shutil.copyfile(path, target / path.name)
        summary = json.loads((source / "summary.json").read_text())
        rows = json.loads((source / "comparisons.json").read_text())
        assert summary["status"] == "PASS" and rows and all(r["status"] == "PASS" for r in rows)
        assignment = summary["gpu_assignment"]
        assert assignment["physical_gpus"] == [4] and assignment["assigned_logical_gpu"] == 0
        assert summary["compared_elements"] == sum(r["compared_elements"] for r in rows)
        for row in rows:
            all_rows.append(
                {
                    "configuration": name,
                    "physical_gpu_ids": "4",
                    "logical_gpu_ids": "0",
                    "assigned_physical_gpu": 4,
                    "assigned_logical_gpu": 0,
                    **row,
                }
            )
        supplemental.append({"configuration": name, **summary})
    logs = out / "logs"
    logs.mkdir()
    for path in args.artifacts.glob("*"):
        if path.is_file() and path.suffix in (".json", ".log"):
            shutil.copyfile(path, logs / path.name)
    regressions = {}
    for name in (
        "visibility-policy-tests",
        "v41-regression",
        "v4-gpu-regression",
        "v4-contract-regression",
        "layout-regression",
    ):
        text = (args.artifacts / (name + ".log")).read_text()
        passed = re.findall(r"(\d+) passed", text)
        skipped = re.findall(r"(\d+) skipped", text)
        assert passed and not re.search(r"\d+ failed", text), name
        regressions[name] = {
            "status": "PASS",
            "passed": int(passed[-1]),
            "skipped": int(skipped[-1]) if skipped else 0,
            "physical_gpu_ids": [6] if name != "visibility-policy-tests" else [],
            "logical_gpu_ids": [0] if name != "visibility-policy-tests" else [],
            "log": "logs/" + name + ".log",
        }
    fields = [
        "configuration",
        "physical_gpu_ids",
        "logical_gpu_ids",
        "assigned_physical_gpu",
        "assigned_logical_gpu",
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
    totals = defaultdict(lambda: {"compared_elements": 0, "mismatched_elements": 0})
    for row in all_rows:
        category = row["category"]
        if (
            "train_inference" in category
            or "shared_kernel_modes" in category
            or "updated_weight_modes" in category
        ):
            group = "train_inference_shared_kernel"
        elif "replay" in category or "deterministic" in category:
            group = "deterministic_replay"
        else:
            group = "partition_gradient_or_weight_consistency"
        for key in ("compared_elements", "mismatched_elements"):
            totals[group][key] += row[key]
    summary = {
        "status": "PASS",
        "requested_gpu_matrix_complete": True,
        "project_complete": False,
        "allowed_physical_gpu_ids": [4, 5, 6, 7],
        "forbidden_physical_gpu_ids": [0, 1, 2, 3],
        "eight_gpu_tests": {
            "requested": False,
            "status": "NOT RUN",
            "reason": "canceled by user; not a current blocker",
        },
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "revision": "08c8657f02291e11b6f6b63e0e16014f8e748518",
        "training_profile": (
            "frozen packed weights/scales, embedding and head; activation identity STE"
        ),
        "parity": "shared-kernel parity; no independent backend/model oracle claim",
        "matrix": matrix,
        "supplemental_model_tests": supplemental,
        "regressions": regressions,
        "category_totals": dict(totals),
        "comparison_rows": len(all_rows),
        "compared_elements": sum(r["compared_elements"] for r in all_rows),
        "mismatched_elements": sum(r["mismatched_elements"] for r in all_rows),
        "count_note": (
            "Repeated storage-element comparisons across ranks/gradients/updates, "
            "not unique tokens."
        ),
        "push": {
            "status": "BLOCKED",
            "reason": (
                "gh unauthenticated and SSH publickey denied; "
                "normal push authorized when authentication recovers"
            ),
        },
        "remaining": [
            "independent backend/model oracle",
            "full-parameter quantized training",
            "full 40-layer execution",
            "EP/CP and long-context pruning",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": len(all_rows),
                "compared_elements": summary["compared_elements"],
                "mismatched_elements": summary["mismatched_elements"],
            }
        )
    )


if __name__ == "__main__":
    main()
