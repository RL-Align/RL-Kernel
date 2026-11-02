"""Run the full-native P/P arm with the PR377 comparison workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.run import (
    MatrixConfig,
    _canonical_fingerprint,
    _prepare_run_dir,
    build_arm_environment,
    frozen_input_manifest,
    public_arm_environment,
)
from rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.validate_artifacts import (  # noqa: E501
    compare_train_rollout_logps,
    load_readbacks,
    load_rollout_identity,
    write_report,
)

CASE_ID = "P/P"


class FullStackConfig(MatrixConfig):
    def frozen_parameters(self):
        value = super().frozen_parameters()
        value["ffn_case"] = CASE_ID
        value["logp_case"] = CASE_ID
        value["framework_consistency"] = {
            "use_rollout_logprobs": True,
            "get_mismatch_metrics": True,
            "custom_tis_function": (
                (
                    "rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.tis_me"
                    "trics.metrics_only_tis"
                )
            ),
        }
        return value


def sealed_manifest(config: MatrixConfig):
    value = frozen_input_manifest(config)
    value["fingerprint"] = _canonical_fingerprint(
        {key: item for key, item in value.items() if key != "fingerprint"}
    )
    return value


def validate_native_readbacks(readback_dir: Path):
    errors = []
    frameworks = set()
    paths = []
    for record in load_readbacks(readback_dir):
        paths.append(record["_path"])
        framework = record.get("framework")
        if framework in {"megatron", "vllm"}:
            frameworks.add(framework)
        if record.get("fallbacks"):
            errors.append(f"{record['_path']}: unexpected adapter fallback")
        operators = record.get("operators", {})
        for module in ("attention", "ffn", "logp"):
            operator = operators.get(module)
            if not isinstance(operator, dict):
                errors.append(f"{record['_path']}: missing {module} readback")
                continue
            if operator.get("case_id") != CASE_ID:
                errors.append(
                    f"{record['_path']}: {module} case is " f"{operator.get('case_id')!r}"
                )
            if operator.get("implementation") != "production":
                errors.append(
                    f"{record['_path']}: {module} implementation is "
                    f"{operator.get('implementation')!r}"
                )
    if frameworks != {"megatron", "vllm"}:
        errors.append(f"readback frameworks are {sorted(frameworks)!r}")
    return {"passed": not errors, "errors": errors, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-rollout", type=int, default=3)
    args = parser.parse_args()

    root = Path("/workspace/RL-Kernel-pr390")
    config = FullStackConfig(
        vime_root=Path("/workspace/vime"),
        rl_kernel_root=root,
        megatron_root=Path("/workspace/Megatron-LM-vime"),
        model_root=Path("/app/model/Qwen3-8B"),
        reference_checkpoint=Path("/app/model/Qwen3-8B_torch_dist"),
        prompt_data=Path("/app/model/dapo-math-17k/dapo-math-17k.jsonl"),
        run_dir=args.run_dir,
        launcher=root
        / "rl_engine/integrations/orchestrators/vime/experiments/rocm_attention/launch_arm.sh",
        num_rollout=args.num_rollout,
        rollout_batch_size=1,
        samples_per_prompt=8,
        global_batch_size=8,
        max_response_length=7168,
        max_tokens_per_gpu=4096,
        seed=1234,
        rollout_seed=1234,
    )
    config.validate(require_paths=True)
    _prepare_run_dir(args.run_dir)
    frozen_before = sealed_manifest(config)
    write_report(args.run_dir / "frozen-inputs.before.json", frozen_before)

    arm_dir = args.run_dir / "arms/p-p"
    for directory in (
        arm_dir / "readbacks",
        arm_dir / "dump",
        arm_dir / "checkpoint",
        arm_dir / "mismatch_sidecars",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    environment = build_arm_environment(config, CASE_ID, arm_dir, arm_index=0)
    environment.update(
        {
            "RL_KERNEL_ATTENTION_CASE": CASE_ID,
            "RL_KERNEL_FFN_CASE": CASE_ID,
            "RL_KERNEL_LOGP_CASE": CASE_ID,
            "RL_KERNEL_ROCM_FIXED_PAGED_TILE": "128",
            "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS": "8192",
            "RLK_ABLATION_USE_ROLLOUT_LOGPROBS": "1",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.38",
        }
    )
    launch = {
        "schema_version": "rlkernel.vime_rocm_full_stack_arm_launch.v1",
        "case_id": CASE_ID,
        "expected_implementations": {
            "attention": "production",
            "ffn": "production",
            "logp": "production",
        },
        "framework_consistency": {
            "use_rollout_logprobs": True,
            "mismatch_metrics_recompute": True,
        },
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

    errors = []
    if process.returncode:
        errors.append(f"Vime launcher exited with status {process.returncode}")
    try:
        readbacks = validate_native_readbacks(arm_dir / "readbacks")
    except Exception as exc:
        readbacks = {"passed": False, "errors": [str(exc)], "paths": []}
    errors.extend(readbacks["errors"])
    rollout_identity = load_rollout_identity(arm_dir / "dump" / "rollout_data")
    errors.extend(rollout_identity["errors"])
    try:
        metrics = compare_train_rollout_logps(
            arm_dir / "mismatch_sidecars",
            require_exact=False,
            tensor_parallel_size=4,
            context_parallel_size=2,
        )
    except Exception as exc:
        metrics = {"passed": False, "errors": [str(exc)]}
    errors.extend(metrics["errors"])

    frozen_after = sealed_manifest(config)
    write_report(args.run_dir / "frozen-inputs.after.json", frozen_after)
    frozen_match = frozen_before["fingerprint"] == frozen_after["fingerprint"]
    if not frozen_match:
        errors.append("frozen source fingerprint changed during the run")
    report = {
        "run_dir": str(args.run_dir),
        "num_rollout": args.num_rollout,
        "case_id": CASE_ID,
        "framework_consistency": "use_rollout_logprobs",
        "launcher_returncode": process.returncode,
        "passed": not errors,
        "errors": errors,
        "readbacks": readbacks,
        "rollout_identity": rollout_identity,
        "metrics": metrics,
        "frozen_sources_match": frozen_match,
    }
    write_report(arm_dir / "validation.json", report)
    write_report(args.run_dir / "single-arm-summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
