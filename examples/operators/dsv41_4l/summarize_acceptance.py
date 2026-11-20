# SPDX-License-Identifier: Apache-2.0
"""Aggregate retained component evidence without promoting it to model evidence."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


def summarize_run(world, physical, path, destination):
    if world != len(physical.split(",")):
        raise ValueError("physical GPU list differs from world size")
    records = []
    comparisons = []
    dest = destination / f"{world}gpu"
    dest.mkdir()
    pins = set()
    fingerprints = []
    for rank in range(world):
        source = path / f"operator-rank-{rank}.json"
        r = json.loads(source.read_text())
        if (
            r["status"] != "PASS"
            or r["rank"] != rank
            or r["world_size"] != world
            or len(r["cases"]) != 12
        ):
            raise ValueError(f"incomplete or failed rank evidence: {source}")
        pins.add(r["rlk_commit"])
        fingerprints.append(r["source_files_sha256"])
        for c in r["cases"]:
            if (
                c["status"] != "PASS"
                or c["forward_correctness"] != "PASS"
                or c["backward_correctness"] != "PASS"
            ):
                raise ValueError("operator correctness did not pass")
            comparisons.extend(c["deterministic_replay"])
            comparisons.append(c["operator_train_inference_parity"])
        stress = r["stress"]
        if (
            stress["status"] != "PASS"
            or stress["steps"] != 3
            or stress["microbatches_per_step"] != 2
        ):
            raise ValueError("missing optimizer stress")
        comparisons.extend(stress["snapshots"])
        comparisons.extend(stress["trajectory_replay"])
        records.append(r)
        shutil.copy2(source, dest / source.name)
    if len(pins) != 1 or any(fp != fingerprints[0] for fp in fingerprints):
        raise ValueError("rank source provenance differs")
    for c in comparisons:
        if c["status"] != "PASS" or c["compared_elements"] <= 0 or c["mismatched_elements"] != 0:
            raise ValueError("nonzero or empty bit comparison cannot be a pass")
    shutil.copy2(path / "container.log", dest / "container.log")
    image = json.loads((path / "image-inspect.json").read_text())[0]
    (dest / "image.json").write_text(
        json.dumps(
            {
                k: image.get(k)
                for k in ("Id", "RepoTags", "RepoDigests", "Created", "Architecture", "Os")
            },
            indent=2,
        )
        + "\n"
    )
    replay = [b for r in records for c in r["cases"] for b in c["deterministic_replay"]]
    parity = [c["operator_train_inference_parity"] for r in records for c in r["cases"]]
    return {
        "configuration": f"isolated components, {world} ranks",
        "gpu_count": world,
        "physical_gpus": physical,
        "model_layers": None,
        "original_operator_layer_ids": [1, 2],
        "forward": "PASS",
        "backward": "PASS",
        "operator_correctness": "PASS",
        "determinism": "PASS",
        "operator_train_inference_parity": "PASS",
        "shared_kernels_for_parity": True,
        "model_train_inference_parity": "NOT RUN",
        "model_mismatch_count": None,
        "rlk_commit": pins.pop(),
        "cases": world * 12,
        "compared_elements": sum(c["compared_elements"] for c in comparisons),
        "mismatched_elements": sum(c["mismatched_elements"] for c in comparisons),
        "operator_replay_compared_elements": sum(c["compared_elements"] for c in replay),
        "operator_parity_compared_elements": sum(c["compared_elements"] for c in parity),
        "token_comparisons": sum(c["total_tokens"] or 0 for c in comparisons),
        "mismatched_tokens": sum(c["mismatched_tokens"] or 0 for c in comparisons),
        "comparison_count": len(comparisons),
        "status": "PASS",
        "evidence": f"{world}gpu/",
        "note": (
            "Counts include repeated tensor/token comparisons, not unique prompt tokens. "
            "No model logits/logprobs were compared."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=3, action="append", metavar=("WORLD", "GPU_CSV", "PATH"), required=True
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [
        summarize_run(int(world), physical, Path(path), args.output)
        for world, physical, path in args.run
    ]
    pins = {r["rlk_commit"] for r in runs}
    if len(pins) != 1:
        raise ValueError("report requires one tested implementation SHA across GPU counts")
    summary = {
        "status": "BLOCKED",
        "complete": False,
        "scope": "Implemented and tested V4.1 components; full model integration is incomplete",
        "merge_commit": "4b41ae76d2b4cd3f87ba71c7b56868d888ee15bd",
        "merged_v4_head": "71c7cb2d033806632cef1a3678939bf73dc581dd",
        "main_base": "e9d52e95226b62d826f0a224bafad2cab3ed7538",
        "tested_rlk_commit": pins.pop(),
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "checkpoint_revision": "08c8657f02291e11b6f6b63e0e16014f8e748518",
        "model_original_layer_ids": [0, 1, 2, 3],
        "real_four_layer_execution": "NOT RUN",
        "model_forward": "NOT RUN",
        "model_backward": "NOT RUN",
        "model_deterministic_mismatches": None,
        "model_train_inference_mismatches": None,
        "model_logits": "NOT RUN",
        "model_raw_fp32_logprobs": "NOT RUN",
        "runs": runs,
        "total_component_compared_elements": sum(r["compared_elements"] for r in runs),
        "total_component_mismatched_elements": sum(r["mismatched_elements"] for r in runs),
        "eight_gpu": {
            "status": "NOT RUN",
            "reason": (
                "External PID 3131740 owns physical GPU 0; idle guard rejects full-node allocation"
            ),
        },
        "missing_implementation": [
            "V4.1 model construction and complete checkpoint binding",
            "attention/indexer/candidate selection and differentiable cross-layer KV/index state",
            "V4.1 delayed mHC pre-mix semantics and model backward integration",
            "384-expert router, deterministic dispatch/combine and shared-expert model binding",
            "TP/PP/DP/EP/CP model execution, weight synchronization and model-level acceptance",
            "four-layer logits/raw FP32 logprobs and full-model training/inference parity",
        ],
        "model_topologies": [],
        "regressions": {},
        "historical_v4_four_layer_failures_preserved": True,
        "numerical_contract": (
            "Reference tolerances are separate from exact byte replay/parity; "
            "frozen quantized weights use explicit input identity STE."
        ),
    }
    for tp, pp, dp in ((1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (1, 2, 2), (4, 2, 1), (2, 2, 2)):
        summary["model_topologies"].append(
            {
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "gpu_count": tp * pp * dp,
                "original_layer_ids": [0, 1, 2, 3],
                "forward": "NOT RUN",
                "backward": "NOT RUN",
                "determinism": "NOT RUN",
                "train_inference_parity": "NOT RUN",
                "mismatch_count": None,
                "status": "BLOCKED",
                "reason": (
                    "V4.1 model execution is not implemented; "
                    "component rank tests do not provide these topologies"
                ),
            }
        )
    logs = args.output / "logs"
    logs.mkdir()
    for name in (
        "v4-cpu-baseline.log",
        "v4-gpu-baseline.log",
        "v41-official-tests.log",
        "v41-official-tests-fix1.log",
        "v4-cpu-final.log",
        "v4-contracts-final-rocm-runtime.log",
        "v4-gpu-final.log",
        "v41-tests-final.log",
        "layout-final.log",
        "layout-final-with-reasons.log",
        "8gpu-required-gpu0-preflight.log",
        "launch-4gpu-final.log",
    ):
        path = args.artifacts / "logs" / name
        shutil.copy2(path, logs / name)
        if name in (
            "v4-contracts-final-rocm-runtime.log",
            "v4-gpu-final.log",
            "v41-tests-final.log",
            "layout-final.log",
        ):
            match = re.search(r"(\d+) passed", path.read_text())
            summary["regressions"][name] = {
                "status": "PASS" if match and "FAILED " not in path.read_text() else "FAIL",
                "passed": int(match[1]) if match else None,
                "skipped": int(re.search(r"(\d+) skipped", path.read_text())[1])
                if re.search(r"(\d+) skipped", path.read_text())
                else 0,
                "evidence": "logs/" + name,
            }
    # Keep launch failures as failures, without zero-valued numerical claims.
    failures = args.output / "resolved-failures"
    failures.mkdir()
    for run in ("acceptance-1gpu", "acceptance-2gpu-committed", "acceptance-2gpu-final"):
        target = failures / run
        target.mkdir()
        for path in (args.artifacts / run).glob("*.json"):
            if path.name.startswith("operator-rank-"):
                shutil.copy2(path, target / path.name)
        log = args.artifacts / run / "container.log"
        if not log.exists():
            log = args.artifacts / "logs" / "acceptance-1gpu.log"
        shutil.copy2(log, target / "container.log")
    for folder in ("checkpoint-slice", "tokenizer"):
        shutil.copy2(
            args.artifacts / folder / "manifest.json", args.output / (folder + "-manifest.json")
        )
    summary["cpu_only_image_attempt"] = {
        "status": "FAIL",
        "reason": (
            "Existing V4 verl/vLLM rollout-config test imports ROCm platform at collection "
            "and requires a GPU in this installed runtime"
        ),
        "evidence": "logs/v4-cpu-final.log",
        "working_contract_runtime": (
            "Same 264 contracts pass with GPU visibility; no tests removed or skipped"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "consistency.csv").open("w", newline="") as stream:
        fields = [
            "configuration",
            "gpu_count",
            "scope",
            "forward",
            "backward",
            "determinism",
            "train_inference_parity",
            "compared_elements",
            "mismatch_count",
            "status",
            "evidence",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in runs:
            writer.writerow(
                {
                    "configuration": r["configuration"],
                    "gpu_count": r["gpu_count"],
                    "scope": "operators only; no model layers",
                    "forward": "PASS",
                    "backward": "PASS",
                    "determinism": "PASS",
                    "train_inference_parity": "PASS (shared operator kernels)",
                    "compared_elements": r["compared_elements"],
                    "mismatch_count": r["mismatched_elements"],
                    "status": r["status"],
                    "evidence": r["evidence"],
                }
            )
        for t in summary["model_topologies"]:
            writer.writerow(
                {
                    "configuration": f"model TP{t['tp']} PP{t['pp']} DP{t['dp']}",
                    "gpu_count": t["gpu_count"],
                    "scope": "original model layers 0,1,2,3",
                    "forward": "NOT RUN",
                    "backward": "NOT RUN",
                    "determinism": "NOT RUN",
                    "train_inference_parity": "NOT RUN",
                    "status": "BLOCKED",
                    "evidence": "summary.json#model_topologies",
                }
            )
    index = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    (args.output / "evidence-sha256.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "component_elements": summary["total_component_compared_elements"],
                "component_mismatches": summary["total_component_mismatched_elements"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
