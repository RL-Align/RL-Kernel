from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from examples.vime_rocm_attention_ablation.run import (
    MatrixConfig,
    _prepare_run_dir,
    build_arm_environment,
    frozen_input_manifest,
    public_arm_environment,
)
from examples.vime_rocm_attention_ablation.validate_artifacts import (
    CASE_IMPLEMENTATIONS,
    validate_arm,
    write_report,
)


RUN_DIR = Path("/app/model/vime-runs/pr393-200round-p-p")


def main() -> int:
    root = Path("/workspace/RL-Kernel-pr390")
    config = MatrixConfig(
        vime_root=Path("/workspace/vime"),
        rl_kernel_root=root,
        megatron_root=Path("/workspace/Megatron-LM-vime"),
        model_root=Path("/app/model/Qwen3-8B"),
        reference_checkpoint=Path("/app/model/Qwen3-8B_torch_dist"),
        prompt_data=Path("/app/model/dapo-math-17k/dapo-math-17k.jsonl"),
        run_dir=RUN_DIR,
        launcher=root / "examples/vime_rocm_attention_ablation/launch_arm.sh",
        num_rollout=200,
    )
    config.validate(require_paths=True)
    _prepare_run_dir(RUN_DIR)
    frozen_before = frozen_input_manifest(config)
    write_report(RUN_DIR / "frozen-inputs.before.json", frozen_before)

    case_id = "P/P"
    arm_dir = RUN_DIR / "arms/p-p"
    for directory in (
        arm_dir / "readbacks",
        arm_dir / "dump",
        arm_dir / "checkpoint",
        arm_dir / "mismatch_sidecars",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    environment = build_arm_environment(config, case_id, arm_dir, arm_index=0)
    environment["VLLM_GPU_MEMORY_UTILIZATION"] = "0.38"
    launch = {
        "schema_version": "rlkernel.vime_rocm_attention_arm_launch.v1",
        "case_id": case_id,
        "expected_implementations": CASE_IMPLEMENTATIONS[case_id],
        "frozen_input_fingerprint": frozen_before["fingerprint"],
        "command": ["bash", str(config.launcher.resolve())],
        "environment": public_arm_environment(environment),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_report(arm_dir / "launch.json", launch)
    with (arm_dir / "launcher.log").open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["bash", str(config.launcher.resolve())],
            cwd=config.rl_kernel_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report = validate_arm(arm_dir, case_id, launcher_returncode=process.returncode)
    write_report(arm_dir / "validation.json", report)
    frozen_after = frozen_input_manifest(config)
    write_report(RUN_DIR / "frozen-inputs.after.json", frozen_after)
    summary = {
        "run_dir": str(RUN_DIR),
        "num_rollout": 200,
        "launcher_returncode": process.returncode,
        "passed": report["passed"],
        "errors": report["errors"],
        "metrics": report["metrics"],
        "frozen_sources_match": frozen_before["fingerprint"] == frozen_after["fingerprint"],
    }
    write_report(RUN_DIR / "single-arm-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
