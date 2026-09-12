# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the TP4/CP2 supplementary VIME experiment suites serially.

The controller is deliberately fail-closed: it waits for all eight GPUs to be
idle, refuses existing run IDs, stops on the first failed Ray job, captures the
authoritative Ray log, and seals only runs accepted by ``validate_run.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

MODULE_GROUPS = ("M000", "M100", "M010", "M001", "M110", "M101", "M011", "M111")
PRECISION_GROUPS = ("G00", "G10", "G01", "G11")
PRECISION_SEEDS = (1234, 2345, 3456)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("module", "precision"), required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rl-kernel-root", type=Path, required=True)
    parser.add_argument("--vime-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--ref-load", type=Path, required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ray-bin", type=Path, required=True)
    parser.add_argument("--extra-pythonpath", action="append", default=[])
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--rollout-tp-size", type=int, default=4)
    parser.add_argument("--rollout-cp-size", type=int, default=1)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--idle-poll-seconds", type=int, default=60)
    return parser.parse_args()


def run_checked(command: list[str], *, stdout=None) -> None:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
    )
    if result.returncode:
        raise RuntimeError(f"command failed with return code {result.returncode}: {command[0]}")


def gpu_memory() -> list[int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def wait_for_idle_gpus(*, threshold: int, poll_seconds: int) -> None:
    while True:
        memory = gpu_memory()
        if len(memory) != 8:
            raise RuntimeError(f"expected exactly 8 GPUs, found {len(memory)}")
        if all(value <= threshold for value in memory):
            return
        print(
            json.dumps(
                {
                    "event": "waiting_for_idle_gpus",
                    "memory_used_mib": memory,
                    "threshold_mib": threshold,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def specs(phase: str) -> list[tuple[str, int, int]]:
    if phase == "module":
        return [(group, 8, 1234) for group in MODULE_GROUPS]
    return [(group, 8, seed) for seed in PRECISION_SEEDS for group in PRECISION_GROUPS]


def run_one(args: argparse.Namespace, group: str, rounds: int, seed: int) -> None:
    run_id = f"{args.suite_id}-{group.lower()}-n{rounds}-b8-s16-" f"refkl001-s{seed}"
    run_dir = args.output_root / run_id
    submission_id = f"vime200-{run_id}"
    if run_dir.exists():
        raise FileExistsError(f"refusing existing run directory: {run_dir}")
    status = subprocess.run(
        [str(args.ray_bin), "job", "status", submission_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode == 0:
        raise RuntimeError(f"refusing existing Ray submission: {submission_id}")

    wait_for_idle_gpus(
        threshold=args.idle_memory_mib,
        poll_seconds=args.idle_poll_seconds,
    )
    command = [
        str(args.python),
        str(args.rl_kernel_root / "examples/vime_qwen3_8b_tp4_cp2_200/run_arm.py"),
        "--group",
        group,
        "--run-id",
        run_id,
        "--num-rollout",
        str(rounds),
        "--seed",
        str(seed),
        "--rollout-seed",
        str(seed),
        "--output-root",
        str(args.output_root),
        "--rl-kernel-root",
        str(args.rl_kernel_root),
        "--vime-root",
        str(args.vime_root),
        "--megatron-root",
        str(args.megatron_root),
        "--model-root",
        str(args.model_root),
        "--ref-load",
        str(args.ref_load),
        "--prompt-data",
        str(args.prompt_data),
        "--python",
        str(args.python),
        "--ray-bin",
        str(args.ray_bin),
        "--ld-library-path",
        args.ld_library_path,
        "--rollout-batch-size",
        "8",
        "--n-samples-per-prompt",
        "16",
        "--global-batch-size",
        "128",
        "--rollout-tp-size",
        str(args.rollout_tp_size),
        "--rollout-cp-size",
        str(args.rollout_cp_size),
        "--max-response-len",
        "7168",
        "--max-tokens-per-gpu",
        "4096",
        "--vllm-gpu-memory-utilization",
        "0.4",
        "--router-policy",
        "round_robin",
        "--use-kl-loss",
        "--kl-loss-coef",
        "0.001",
        "--wait",
    ]
    for path in args.extra_pythonpath:
        command.extend(["--extra-pythonpath", path])

    args.output_root.mkdir(parents=True, exist_ok=True)
    controller_log = args.output_root / f"{run_id}.controller.log"
    started = time.time()
    print(json.dumps({"event": "start", "group": group, "run_id": run_id}), flush=True)
    with controller_log.open("w", encoding="utf-8") as handle:
        run_checked(command, stdout=handle)

    with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
        run_checked([str(args.ray_bin), "job", "logs", submission_id], stdout=handle)
    run_checked(
        [
            str(args.python),
            str(args.rl_kernel_root / "examples/vime_qwen3_8b_tp4_cp2_200/validate_run.py"),
            "--run-dir",
            str(run_dir),
            "--seal",
        ]
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "group": group,
                "run_id": run_id,
                "elapsed_seconds": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    lock = args.output_root / f".{args.suite_id}.lock"
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"suite lock already exists: {lock}") from exc
    try:
        for group, rounds, seed in specs(args.phase):
            run_one(args, group, rounds, seed)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
