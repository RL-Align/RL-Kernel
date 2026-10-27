# SPDX-License-Identifier: Apache-2.0
"""Create a reviewable acceptance manifest from completed independent HF controls."""
import argparse
import json
from pathlib import Path

from examples.post_training.tp_ppo.fixed_contract import REVISION, load_packet, load_targets, sha


def main():
    """Write a frozen acceptance manifest of SHA-256s for the fixed HF reference inputs and bounds.

    Verifies the rollout and targets, then hashes them, the actor/critic FP32/BF16 reports of
    repeats 0 and 1, and the bounds files; tensor artifact hashes are copied from the reports,
    not recomputed. Also pins the behavior log-prob atol (actor cross-precision ``0/scores``)
    and the test matrix. Raises ``ValueError`` if ``--output`` exists or a report is incomplete.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("acceptance manifest already exists")
    load_packet(a.reference_root / "rollout")
    load_targets(a.reference_root / "targets")
    files = {}
    for name in [
        "rollout/manifest.json",
        "rollout/rollout.pt",
        "targets/targets.json",
        "targets/targets.pt",
    ]:
        files[name] = sha(a.reference_root / name)
    for role in ["actor", "critic"]:
        for precision in ["fp32", "bf16"]:
            for repeat in [0, 1]:
                directory = f"{role}-{precision}-{repeat}"
                path = a.reference_root / directory / "report.json"
                report = json.loads(path.read_text())
                if report["status"] != "fixed_reference_complete":
                    raise ValueError("reference incomplete")
                files[f"{directory}/report.json"] = sha(path)
                for row in report["artifacts"].values():
                    files[f'{directory}/{row["path"]}'] = row["sha256"]
            name = f"{role}-{precision}-repeat-bounds.json"
            files[name] = sha(a.reference_root / name)
        name = f"{role}-cross-precision-bounds.json"
        files[name] = sha(a.reference_root / name)
    bounds = json.loads((a.reference_root / "actor-cross-precision-bounds.json").read_text())
    result = dict(
        status="frozen",
        model_revision=REVISION,
        files=files,
        rollout_manifest_sha256=files["rollout/manifest.json"],
        targets_manifest_sha256=files["targets/targets.json"],
        fp32_reference_sha256=files["actor-fp32-0/report.json"],
        behavior_logprob={"atol": bounds["bounds"]["0/scores"]["max_abs"], "rtol": 0.0},
        matrix={
            "frameworks": ["megatron", "deepspeed"],
            "precisions": ["fp32", "bf16"],
            "tp": [1, 2],
            "roles": ["actor", "critic"],
            "rounds": 2,
            "topologies": [[1, 1, 2], [2, 1, 1]],
        },
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
