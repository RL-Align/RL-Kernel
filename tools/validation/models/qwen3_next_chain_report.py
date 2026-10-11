#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write the WS1 chain-gate report for a finished Qwen3-Next chain.

    python tools/validation/models/qwen3_next_chain_report.py \
        --last <train run dir> --out chain-report.json

Follows ``previous`` from the given run back to the ``l2`` stage, reads every
passed stage's L2 matrix (``baseline`` from ``l2``, ``update1``/``update2``
from ``train1``/``train2``, ``reload`` from ``train``) and the hash-checked case
artifacts it names, and writes ``rl_engine.validation.models.qwen3_next_chain_report``'s
report. Exits 1 when the chain did not pass.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rl_engine.contracts.numerical import load_contract  # noqa: E402
from rl_engine.validation.models.qwen3_next_chain_report import build_report  # noqa: E402
from rl_engine.validation.models.qwen3_next_l2 import (  # noqa: E402
    checked_artifact_path,
    load_tensor_artifact,
)

VERSION_OF_STAGE = {"l2": "baseline", "train1": "update1", "train2": "update2", "train": "reload"}


def chain_runs(last):
    runs, run = [], Path(last).resolve()
    while run is not None:
        manifest = json.loads((run / "run.json").read_text())
        if manifest["stage"] in VERSION_OF_STAGE:
            runs.append((manifest, run))
        if manifest["stage"] == "l2":
            break
        run = Path(manifest["previous"]) if manifest.get("previous") else None
    if not runs or runs[-1][0]["stage"] != "l2":
        raise SystemExit("The chain does not lead back to an l2 run")
    commits = {manifest["source_commit"] for manifest, _ in runs}
    if len(commits) != 1:
        raise SystemExit(f"The chain mixes source commits: {sorted(commits)}")
    return list(reversed(runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=Path, required=True, help="last passed stage run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bi", type=Path, help="passed batch-invariance (bi) stage run")
    args = parser.parse_args()
    matrices, artifacts = {}, {}
    gpu_name = None
    for manifest, run in chain_runs(args.last):
        result = json.loads((run / "result.json").read_text())
        if result.get("status") != "passed":
            raise SystemExit(f"{manifest['stage']} did not pass")
        attempt = run / "attempts" / result["attempt"]
        matrix = json.loads((run / result["evidence"]["l2_matrix"]["path"]).read_text())
        version = VERSION_OF_STAGE[manifest["stage"]]
        matrices[version] = matrix
        artifacts[version] = {
            case_id: {
                role: load_tensor_artifact(checked_artifact_path(record, artifact_root=attempt))
                for role, record in matrix["cases"][case_id]["artifacts"].items()
            }
            for case_id in matrix["case_ids"]
        }
        environment = attempt / "environment.json"
        if gpu_name is None and environment.is_file():
            gpu_name = json.loads(environment.read_text()).get("gpu_name")
    identity = manifest["checkpoint_identity"]
    bi_matrix = None
    if args.bi is not None:
        bi_manifest = json.loads((args.bi / "run.json").read_text())
        bi_result = json.loads((args.bi / "result.json").read_text())
        if (
            bi_manifest["stage"] != "bi"
            or bi_manifest["source_commit"] != manifest["source_commit"]
        ):
            raise SystemExit("--bi must name a bi run on the same commit as the chain")
        bi_matrix = json.loads((args.bi / bi_result["evidence"]["bi_matrix"]["path"]).read_text())
    report = build_report(
        matrices,
        artifacts,
        contract=load_contract(),
        measured_commit=manifest["source_commit"],
        weight_source=f"Qwen/Qwen3-Next-80B-A3B-Instruct@{identity['revision']}",
        weight_hash=identity["expected_content_hash"],
        device="cuda",
        gpu_name=gpu_name,
        compute_capability=None,
        bi_matrix=bi_matrix,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"chain passed={report['passed']} first_drift={report['first_drift']} "
        f"family={report['first_drift_family']} cells={len(report['cells'])}"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
