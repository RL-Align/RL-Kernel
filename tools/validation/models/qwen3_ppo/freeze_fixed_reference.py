# SPDX-License-Identifier: Apache-2.0
"""Create a reviewable acceptance manifest from completed independent HF controls."""
import argparse
import json
from pathlib import Path

from tools.validation.models.qwen3_ppo.fixed_contract import (
    REVISION,
    load_packet,
    load_targets,
    sha,
)
from tools.validation.models.qwen3_ppo.hf_ppo_reference import IMPLEMENTATION_CONTROL


def check_report_inputs(report, *, role, precision, rollout_sha256, targets_sha256):
    """Raise ``ValueError`` unless a control report was made for this role, precision and inputs."""
    if (report.get("role"), report.get("precision")) != (role, precision):
        raise ValueError(f"control report is not {role} {precision}")
    if (report.get("rollout_manifest_sha256"), report.get("targets_manifest_sha256")) != (
        rollout_sha256,
        targets_sha256,
    ):
        raise ValueError(f"{role} {precision} control used another rollout or targets")


def check_bounds_source(bounds, *, reference, other, controls, name):
    """Raise ``ValueError`` unless a bounds file was frozen from exactly these report digests."""
    recorded = [c.get("report_sha256") for c in bounds.get("extra_controls", [])]
    if (
        bounds.get("status") != "frozen_reference_bounds"
        or bounds.get("reference_report_sha256") != reference
        or bounds.get("other_report_sha256") != other
        or recorded != list(controls)
    ):
        raise ValueError(f"{name} was not frozen from the selected control reports")


def main():
    """Write a frozen acceptance manifest of SHA-256s for the fixed HF reference inputs and bounds.

    Verifies the rollout and targets, then hashes them, the actor/critic FP32/BF16 reports of
    repeats 0 and 1 and of the implementation control (``-impl``), and the bounds files;
    tensor artifact hashes are copied from the reports, not recomputed. Also pins the
    behavior log-prob atol (actor cross-precision ``0/scores``), the control rule and the
    test matrix. Raises ``ValueError`` if ``--output`` exists or a report is incomplete.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("acceptance manifest already exists")
    load_packet(a.reference_root / "rollout")
    load_targets(a.reference_root / "targets")
    rollout_sha256 = sha(a.reference_root / "rollout/manifest.json")
    targets_manifest = json.loads((a.reference_root / "targets/targets.json").read_text())
    if targets_manifest.get("rollout_manifest_sha256") != rollout_sha256:
        raise ValueError("frozen targets were computed from another rollout")
    files = {}
    states = set()
    for name in [
        "rollout/manifest.json",
        "rollout/rollout.pt",
        "targets/targets.json",
        "targets/targets.pt",
    ]:
        files[name] = sha(a.reference_root / name)
    for role in ["actor", "critic"]:
        for precision in ["fp32", "bf16"]:
            for run in ["0", "1", "impl"]:
                directory = f"{role}-{precision}-{run}"
                path = a.reference_root / directory / "report.json"
                report = json.loads(path.read_text())
                if report["status"] != "fixed_reference_complete":
                    raise ValueError("reference incomplete")
                if (report.get("implementation_control") is None) != (run != "impl"):
                    raise ValueError(f"wrong implementation for control {directory}")
                check_report_inputs(
                    report,
                    role=role,
                    precision=precision,
                    rollout_sha256=rollout_sha256,
                    targets_sha256=files["targets/targets.json"],
                )
                states.add(report.get("optimizer_state"))
                files[f"{directory}/report.json"] = sha(path)
                for row in report["artifacts"].values():
                    files[f'{directory}/{row["path"]}'] = row["sha256"]
            name = f"{role}-{precision}-repeat-bounds.json"
            check_bounds_source(
                json.loads((a.reference_root / name).read_text()),
                reference=files[f"{role}-{precision}-0/report.json"],
                other=files[f"{role}-{precision}-1/report.json"],
                controls=[files[f"{role}-{precision}-impl/report.json"]],
                name=name,
            )
            files[name] = sha(a.reference_root / name)
        name = f"{role}-cross-precision-bounds.json"
        check_bounds_source(
            json.loads((a.reference_root / name).read_text()),
            reference=files[f"{role}-fp32-0/report.json"],
            other=files[f"{role}-bf16-0/report.json"],
            controls=[files[f"{role}-bf16-impl/report.json"]],
            name=name,
        )
        files[name] = sha(a.reference_root / name)
    if len(states) != 1:
        raise ValueError(f"reference runs disagree on optimizer state placement: {states}")
    bounds = json.loads((a.reference_root / "actor-cross-precision-bounds.json").read_text())
    result = dict(
        status="frozen",
        model_revision=REVISION,
        files=files,
        rollout_manifest_sha256=files["rollout/manifest.json"],
        targets_manifest_sha256=files["targets/targets.json"],
        fp32_reference_sha256=files["actor-fp32-0/report.json"],
        behavior_logprob={"atol": bounds["bounds"]["0/scores"]["max_abs"], "rtol": 0.0},
        reference_optimizer_state=states.pop(),
        controls=dict(
            repeat="the same HF run repeated (repeat 1 against repeat 0)",
            implementation=IMPLEMENTATION_CONTROL,
            rule=(
                "same-precision bounds: 4 * max(repeat, implementation control) + floor; "
                "cross-precision bounds: 4 * max(BF16 repeat 0, BF16 implementation control "
                "against FP32 repeat 0) + floor"
            ),
        ),
        matrix={
            "frameworks": ["megatron", "deepspeed"],
            "precisions": ["fp32", "bf16"],
            "tp": [2, 4],
            "roles": ["actor", "critic"],
        },
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
