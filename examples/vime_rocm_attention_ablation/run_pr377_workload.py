"""Run one native or consistency arm of the Qwen3-8B workload on ROCm.

The native arm selects production attention, FFN and logp on both frameworks.
The consistency arm selects the strict RL-Kernel route on both frameworks and
validates bitwise train/rollout agreement. Both arms recompute training logp;
neither enables rollout-logprob reuse. Paths, topology, and round count are CLI
arguments so the same script serves different machine layouts.

Example::

    python -m examples.vime_rocm_attention_ablation.run_qwen3_8b \
      --mode consistency --rollouts 8 \
      --run-dir /app/model/vime-runs/mfma-rr-3round \
      --rl-kernel-root /work/RL-Kernel --vime-root /work/vime \
      --megatron-root /work/Megatron-LM-vime
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from examples.vime_rocm_attention_ablation.run import (
    MatrixConfig,
    _canonical_fingerprint,
    _git_identity,
    _prepare_run_dir,
    build_arm_environment,
    frozen_input_manifest,
    public_arm_environment,
)
from examples.vime_rocm_attention_ablation.validate_artifacts import (
    CASE_IMPLEMENTATIONS,
    compare_train_rollout_logps,
    load_readbacks,
    load_rollout_identity,
    validate_arm,
    write_report,
)
from examples.vime_rocm_attention_ablation.validate_module_artifacts import (
    validate_module_readbacks,
)

# Strict-path knobs forwarded into the Ray runtime environment when set.
FORWARDED_STRICT_ENVIRONMENT = (
    "RL_KERNEL_DET_GEMM_BACKEND",
    "RL_KERNEL_ROCM_FIXED_PAGED_TILE",
    "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS",
    "RL_KERNEL_ROCM_ATTENTION_BACKEND",
    "RL_KERNEL_SPARSE_TOP_P_REPLAY",
)


class WorkloadConfig(MatrixConfig):
    case_id: str = "R/R"

    def frozen_parameters(self):
        value = super().frozen_parameters()
        value["grpo_std_normalization"] = getattr(self, "grpo_std_normalization", "enabled")
        value["ffn_case"] = self.case_id
        value["logp_case"] = self.case_id
        value["framework_consistency"] = {
            "use_rollout_logprobs": False,
            "get_mismatch_metrics": True,
            "custom_tis_function": ("vime_rocm_attention_ablation.tis_metrics.metrics_only_tis"),
        }
        return value


def sealed_manifest(config: MatrixConfig):
    value = frozen_input_manifest(config)
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise ValueError("vLLM source cannot be resolved from the active Python environment")
    vllm_root = Path(spec.origin).resolve().parents[1]
    value["sources"]["vllm"] = _git_identity(vllm_root)
    value["fingerprint"] = _canonical_fingerprint(
        {key: item for key, item in value.items() if key != "fingerprint"}
    )
    return value


def validate_native_readbacks(readback_dir: Path, *, log_text: str):
    readbacks = load_readbacks(readback_dir)
    report = validate_module_readbacks(
        readbacks,
        {module: "P/P" for module in ("attention", "ffn", "logp")},
        log_text=log_text,
    )
    report["paths"] = [record["_path"] for record in readbacks]
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mode", choices=["native", "consistency"])
    mode_group.add_argument("--case", choices=["P/P", "R/R"], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--rollouts",
        "--num-rollout",
        dest="num_rollout",
        type=int,
        default=200,
        help="training/rollout steps; --num-rollout remains a compatibility alias",
    )
    parser.add_argument("--rl-kernel-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--vime-root", type=Path, default=Path(os.environ.get("VIME_ROOT", "/work/vime"))
    )
    parser.add_argument(
        "--megatron-root",
        type=Path,
        default=Path(os.environ.get("MEGATRON_ROOT", "/work/Megatron-LM-vime")),
    )
    parser.add_argument("--model-root", type=Path, default=Path("/app/model/Qwen3-8B"))
    parser.add_argument(
        "--reference-checkpoint", type=Path, default=Path("/app/model/Qwen3-8B_torch_dist")
    )
    parser.add_argument(
        "--prompt-data", type=Path, default=Path("/app/model/dapo-math-17k/dapo-math-17k.jsonl")
    )
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--max-response-length", type=int, default=7168)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=1234)
    parser.add_argument("--visible-gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--rollout-tp-size", type=int, default=4)
    parser.add_argument("--rollout-cp-size", type=int, default=1)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=-1)
    parser.add_argument(
        "--grpo-std-normalization", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--fixed-paged-tile", default="128")
    parser.add_argument("--paged-kv-max-tokens", default="8192")
    parser.add_argument("--vllm-gpu-memory-utilization", default="0.38")
    parser.add_argument("--ray-port", type=int, default=6385)
    parser.add_argument("--ray-dashboard-port", type=int, default=28265)
    args = parser.parse_args(argv)
    args.case = args.case or {"native": "P/P", "consistency": "R/R"}[args.mode]
    args.mode = args.mode or {"P/P": "native", "R/R": "consistency"}[args.case]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    case_id = args.case
    root = args.rl_kernel_root.resolve()
    config = WorkloadConfig(
        vime_root=args.vime_root,
        rl_kernel_root=root,
        megatron_root=args.megatron_root,
        model_root=args.model_root,
        reference_checkpoint=args.reference_checkpoint,
        prompt_data=args.prompt_data,
        run_dir=args.run_dir,
        launcher=root / "examples/vime_rocm_attention_ablation/launch_arm.sh",
        visible_gpus=args.visible_gpus,
        num_gpus=args.num_gpus,
        tensor_parallel_size=args.tp_size,
        context_parallel_size=args.cp_size,
        rollout_tensor_parallel_size=args.rollout_tp_size,
        rollout_context_parallel_size=args.rollout_cp_size,
        num_rollout=args.num_rollout,
        rollout_batch_size=args.rollout_batch_size,
        samples_per_prompt=args.samples_per_prompt,
        global_batch_size=args.global_batch_size,
        max_response_length=args.max_response_length,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        seed=args.seed,
        rollout_seed=args.rollout_seed,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        rollout_top_k=args.rollout_top_k,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        kl_coef=args.kl_coef,
        ray_port=args.ray_port,
        ray_dashboard_port=args.ray_dashboard_port,
    )
    config.case_id = case_id
    config.grpo_std_normalization = args.grpo_std_normalization
    config.validate(require_paths=True)
    _prepare_run_dir(args.run_dir)
    frozen_before = sealed_manifest(config)
    write_report(args.run_dir / "frozen-inputs.before.json", frozen_before)

    arm_slug = case_id.lower().replace("/", "-")
    arm_dir = args.run_dir / "arms" / arm_slug
    for directory in (
        arm_dir / "readbacks",
        arm_dir / "dump",
        arm_dir / "checkpoint",
        arm_dir / "mismatch_sidecars",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    environment = build_arm_environment(
        config, case_id, arm_dir, arm_index=0 if case_id == "P/P" else 3
    )
    environment.update(
        {
            "RLK_ABLATION_USE_ROLLOUT_LOGPROBS": "0",
            "RLK_ABLATION_DISABLE_GRPO_STD_NORMALIZATION": (
                "1" if args.grpo_std_normalization == "disabled" else "0"
            ),
            "RL_KERNEL_ROCM_ATTENTION_BACKEND": "triton",
            "RL_KERNEL_SPARSE_TOP_P_REPLAY": "1" if case_id == "R/R" else "0",
            "RL_KERNEL_ATTENTION_CASE": case_id,
            "RL_KERNEL_FFN_CASE": case_id,
            "RL_KERNEL_LOGP_CASE": case_id,
            "RL_KERNEL_ROCM_FIXED_PAGED_TILE": args.fixed_paged_tile,
            "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS": args.paged_kv_max_tokens,
            "VLLM_GPU_MEMORY_UTILIZATION": args.vllm_gpu_memory_utilization,
        }
    )
    launch = {
        "schema_version": "rlkernel.vime_rocm_attention_arm_launch.v1",
        "mode": args.mode,
        "case_id": case_id,
        "topology_evidence": (
            "reference"
            if (
                args.num_gpus,
                args.tp_size,
                args.cp_size,
                args.rollout_tp_size,
                args.rollout_cp_size,
            )
            == (8, 4, 2, 4, 1)
            else "experimental"
        ),
        "expected_implementations": CASE_IMPLEMENTATIONS[case_id],
        "framework_consistency": {
            "use_rollout_logprobs": False,
            "mismatch_metrics_recompute": True,
            "strict_linear_logp": case_id == "R/R",
        },
        "strict_environment": {
            name: environment[name] for name in FORWARDED_STRICT_ENVIRONMENT if name in environment
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

    if case_id == "R/R":
        report = validate_arm(arm_dir, case_id, launcher_returncode=process.returncode)
    else:
        errors = []
        if process.returncode:
            errors.append(f"Vime launcher exited with status {process.returncode}")
        log_text = (arm_dir / "launcher.log").read_text(encoding="utf-8", errors="replace")
        try:
            readbacks = validate_native_readbacks(arm_dir / "readbacks", log_text=log_text)
        except Exception as exc:  # pragma: no cover - runtime evidence failure
            readbacks = {"passed": False, "errors": [str(exc)], "paths": []}
        errors.extend(readbacks["errors"])
        rollout_identity = load_rollout_identity(arm_dir / "dump" / "rollout_data")
        errors.extend(rollout_identity["errors"])
        try:
            metrics = compare_train_rollout_logps(
                arm_dir / "mismatch_sidecars",
                require_exact=False,
                tensor_parallel_size=config.tensor_parallel_size,
                context_parallel_size=config.context_parallel_size,
            )
        except Exception as exc:  # pragma: no cover - runtime evidence failure
            metrics = {"passed": False, "errors": [str(exc)]}
        errors.extend(metrics["errors"])
        report = {
            "case_id": case_id,
            "launcher_returncode": process.returncode,
            "passed": not errors,
            "errors": errors,
            "readbacks": readbacks,
            "rollout_identity": rollout_identity,
            "metrics": metrics,
        }
    expected_samples = config.num_rollout * config.rollout_batch_size * config.samples_per_prompt
    actual_samples = report.get("metrics", {}).get("sample_count", 0)
    if actual_samples != expected_samples:
        report["errors"].append(
            f"expected {expected_samples} compared samples, got {actual_samples}"
        )
        report["passed"] = False
    report["expected_sample_count"] = expected_samples
    frozen_after = sealed_manifest(config)
    write_report(args.run_dir / "frozen-inputs.after.json", frozen_after)
    frozen_match = frozen_before["fingerprint"] == frozen_after["fingerprint"]
    if not frozen_match:
        report["errors"] = list(report.get("errors", [])) + [
            "frozen source fingerprint changed during the run"
        ]
        report["passed"] = False
    report["frozen_sources_match"] = frozen_match
    report["run_dir"] = str(args.run_dir)
    report["num_rollout"] = args.num_rollout
    write_report(arm_dir / "validation.json", report)
    summary = {
        "run_dir": str(args.run_dir),
        "mode": args.mode,
        "case_id": case_id,
        "topology_evidence": launch["topology_evidence"],
        "num_rollout": args.num_rollout,
        "launcher_returncode": process.returncode,
        "passed": report["passed"],
        "errors": report["errors"],
        "metrics": report.get("metrics"),
        "strict_environment": launch["strict_environment"],
        "frozen_sources_match": frozen_match,
    }
    write_report(args.run_dir / "single-arm-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
