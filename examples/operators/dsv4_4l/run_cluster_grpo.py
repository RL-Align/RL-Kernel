# SPDX-License-Identifier: Apache-2.0
"""Submit the four-layer PP2 GRPO recipe to an existing, allocated Ray cluster."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def build_command(args):
    """Keep the measured numerical layout; vary only physical node placement."""
    actor_ips, rollout_ips = args.actor_nodes.split(","), args.rollout_nodes.split(",")
    import ipaddress

    for ips in (actor_ips, rollout_ips):
        if len(ips) not in (1, 2) or len(set(ips)) != len(ips):
            raise ValueError("each role requires one or two distinct node IPs")
        for ip in ips:
            ipaddress.ip_address(ip)
    if len(set(actor_ips + rollout_ips)) < 2:
        raise ValueError("use run_grpo.py for the measured single-node recipe")
    if not args.ray_address or args.ray_address == "local":
        raise ValueError("an existing Ray cluster address is required")
    if not 0 <= args.stress_final_repeats <= 10000:
        raise ValueError("stress repeats must be between 0 and 10000")
    for path in (args.checkpoint, args.data, args.output):
        if not path.is_absolute():
            raise ValueError("model, data and shared output paths must be absolute")
    prompts = json.loads((args.data / "extended-audit-prompts.json").read_text())
    if (
        not isinstance(prompts, list)
        or len(prompts) != 3
        or not all(isinstance(p, str) and p for p in prompts)
    ):
        raise ValueError("audit prompts must contain three nonempty strings")
    repo = Path(__file__).resolve().parents[3]
    overrides = {
        "actor_rollout_ref.model.path": str(args.checkpoint),
        "data.train_files": str(args.data / "train.parquet"),
        "data.val_files": str(args.data / "val.parquet"),
        "data.train_batch_size": 2,
        "data.max_prompt_length": 128,
        "data.max_response_length": 16,
        "data.dataloader_num_workers": 0,
        "data.shuffle": False,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
        "actor_rollout_ref.rollout.n": 2,
        "actor_rollout_ref.rollout.max_model_len": 256,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 512,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "trainer.nnodes": len(actor_ips),
        "trainer.n_gpus_per_node": 2 // len(actor_ips),
        "actor_rollout_ref.rollout.nnodes": len(rollout_ips),
        "actor_rollout_ref.rollout.n_gpus_per_node": 2 // len(rollout_ips),
        "trainer.total_training_steps": 3,
        "trainer.total_epochs": 1,
        "trainer.save_freq": -1,
        "trainer.default_local_dir": str(args.output / "checkpoints"),
        "reward.num_workers": 1,
        "reward.custom_reward_function.path": str(repo / "examples/operators/dsv4_4l/copy_task.py"),
        "reward.custom_reward_function.name": "compute_score",
        "rlk.multinode": True,
        "rlk.actor_node_ips": actor_ips,
        "actor_rollout_ref.rollout.rlk_node_ips": rollout_ips,
        "rlk.sync_dir": str(args.output / "policy-sync"),
        "rlk.audit_response_length": 16,
        "rlk.audit_prompts": prompts,
        "+rlk.stress_final_repeats": args.stress_final_repeats,
        "rlk.preflight_files": [
            str(args.data / name)
            for name in (
                "replay-fixed-sequences.json",
                "checkpoint-inventory.json",
                "extended-audit-prompts.json",
            )
        ],
        "+ray_kwargs.ray_init.address": args.ray_address,
        "ray_kwargs.ray_init.num_cpus": None,
        "hydra.run.dir": str(args.output / "hydra"),
    }
    env = {
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "VLLM_PP_LAYER_PARTITION": os.environ.get("RLK_DSV4_PP_SPLIT", "3,1"),
        "RLK_DSV4_PP_SPLIT": os.environ.get("RLK_DSV4_PP_SPLIT", "3,1"),
        "DSV4_TRACE_FORMAT": "cas-v1",
        "RLK_REPLAY_FIXED_CASES": str(args.data / "replay-fixed-sequences.json"),
        "HYDRA_FULL_ERROR": "1",
    }
    for key, value in env.items():
        overrides["++ray_kwargs.ray_init.runtime_env.env_vars." + key] = value
    return [
        sys.executable,
        "-m",
        "verl.experimental.rl_kernel.main_ppo",
        *[f"{k}={json.dumps(v)}" for k, v in overrides.items()],
    ], env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-nodes", required=True, help="comma-separated Ray node IPs")
    parser.add_argument("--rollout-nodes", required=True, help="comma-separated Ray node IPs")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="NEW shared output directory")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--stress-final-repeats", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, env = build_command(args)
    print(json.dumps(dict(command=command, environment=env), indent=2), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "launch.json").write_text(
        json.dumps(dict(command=command, environment=env), indent=2)
    )
    start = time.monotonic()
    with (args.output / "driver.log").open("w") as log:
        result = subprocess.run(
            command, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT
        )
    (args.output / "job-result.json").write_text(
        json.dumps(dict(exit_code=result.returncode, wall_seconds=time.monotonic() - start))
    )
    (args.output / "exit-code.json").write_text(json.dumps(dict(exit_code=result.returncode)))
    print(f"Exit {result.returncode}; logs: {args.output / 'driver.log'}", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
