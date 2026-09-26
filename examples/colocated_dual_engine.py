# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Colocated dual-engine: vLLM(TP=N) + persistent DeepSpeed workers on shared GPUs.

Architecture (single-node, N GPUs):
  Main process:  owns vLLM LLM(TP=N), orchestrates phases
  Worker procs:  torchrun N-rank DeepSpeed, launched ONCE, stays alive

  Phase 1 (rollout):  vLLM wake → N-GPU generate → sleep (free GPU mem)
  Phase 2 (training):  signal workers → workers load model to GPU → train →
                       offload to CPU → signal done
  Repeat

vLLM sleep(level=2) releases all GPU memory. Workers offload model to CPU
between steps. Neither side holds GPU memory while the other is active.
Worker cold-start happens only once; subsequent steps are signal-only.

Usage:
  python examples/colocated_dual_engine.py \
    --model /path/to/model --num-gpus 8 --steps 10

  python examples/colocated_dual_engine.py \
    --model /path/to/model --num-gpus 8 --steps 10 \
    --output-log benchmark_results/colocated.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StepResult:
    step: int
    phase: str
    duration_ms: float
    metrics: dict = field(default_factory=dict)


WORKER_SCRIPT = Path(__file__).parent / "_colocated_train_worker.py"


class ColocatedOrchestrator:
    """Orchestrates vLLM and persistent training workers on shared GPUs."""

    def __init__(self, args):
        self.args = args
        self.llm = None
        self.worker_proc = None
        self.results: list[StepResult] = []
        self.work_dir = tempfile.mkdtemp(prefix="colocated_")

    def setup(self):
        a = self.args
        print("=" * 60)
        print(" Colocated Dual-Engine (Persistent Worker)")
        print(f" Model:  {a.model}")
        print(f" GPUs:   {a.num_gpus} (shared)")
        print(f" Steps:  {a.steps}")
        print(f" Work:   {self.work_dir}")
        print("=" * 60)

        # --- 1. Start persistent training workers FIRST ---
        # Workers init DeepSpeed, then offload to CPU and stabilize.
        # Must complete before vLLM starts, so GPU memory is stable during
        # vLLM's memory profiling.
        print("\n[Setup] Starting persistent training workers...")
        t0 = time.time()
        self._start_workers()
        print(f"[Setup] Workers ready ({time.time() - t0:.1f}s)")

        # --- 2. Initialize vLLM (GPU memory is now stable) ---
        print(f"[Setup] Initializing vLLM (TP={a.num_gpus})...")
        t0 = time.time()
        from vllm import LLM

        self.llm = LLM(
            model=a.model,
            tensor_parallel_size=a.num_gpus,
            gpu_memory_utilization=a.vllm_gpu_memory_utilization,
            enforce_eager=True,
            dtype="bfloat16",
            max_model_len=a.max_prompt_len + a.max_completion_len,
            enable_sleep_mode=True,
            trust_remote_code=True,
        )
        print(f"[Setup] vLLM ready ({time.time() - t0:.1f}s)")
        print("[Setup] Complete\n")

    def _start_workers(self):
        """Launch torchrun workers once. They stay alive for all steps."""
        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            f"--nproc_per_node={self.args.num_gpus}",
            "--standalone",
            str(WORKER_SCRIPT),
            "--model", self.args.model,
            "--work-dir", self.work_dir,
            "--max-steps", str(self.args.steps),
            "--lr", str(self.args.lr),
            "--lora-rank", str(self.args.lora_rank),
            "--max-len", str(self.args.max_prompt_len + self.args.max_completion_len),
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        self.worker_proc = subprocess.Popen(
            cmd, env=env, stdout=sys.stdout, stderr=sys.stderr
        )

        # Wait for workers to initialize and offload to CPU
        ready_path = os.path.join(self.work_dir, "worker_ready.signal")
        for _ in range(600):
            if os.path.exists(ready_path):
                time.sleep(1)  # extra buffer for GPU memory release
                return
            if self.worker_proc.poll() is not None:
                raise RuntimeError(
                    f"Training worker died during init (rc={self.worker_proc.returncode})"
                )
            time.sleep(0.5)
        raise RuntimeError("Training workers did not become ready within 300s")

    def rollout_phase(self, step: int) -> str:
        """vLLM generate. Returns path to completions file."""
        t0 = time.time()
        from vllm import SamplingParams

        prompts = [
            f"Solve: {step * self.args.prompts_per_step + i} + {i * 3} = "
            for i in range(self.args.prompts_per_step)
        ]
        params = SamplingParams(
            n=self.args.samples_per_prompt,
            max_tokens=self.args.max_completion_len,
            temperature=0.7,
        )
        outputs = self.llm.generate(prompts, params)

        completions = []
        for output in outputs:
            for candidate in output.outputs:
                completions.append({"prompt": output.prompt, "completion": candidate.text})

        path = os.path.join(self.work_dir, f"completions_step{step}.json")
        with open(path, "w") as f:
            json.dump(completions, f)

        ms = (time.time() - t0) * 1000
        self.results.append(
            StepResult(step=step, phase="rollout", duration_ms=ms,
                       metrics={"num_completions": len(completions)})
        )
        print(f"  [Rollout]  {len(completions)} completions in {ms:.0f}ms")
        return path

    def sleep_phase(self):
        t0 = time.time()
        self.llm.sleep(level=self.args.vllm_sleep_level)
        ms = (time.time() - t0) * 1000
        print(f"  [Sleep]    vLLM → CPU ({ms:.0f}ms)")
        return ms

    def wake_phase(self):
        t0 = time.time()
        self.llm.wake_up()
        ms = (time.time() - t0) * 1000
        print(f"  [Wake]     vLLM → GPU ({ms:.0f}ms)")
        return ms

    def training_phase(self, step: int) -> dict:
        """Signal persistent workers to train, wait for completion."""
        t0 = time.time()

        # Signal workers
        signal_path = os.path.join(self.work_dir, f"train_step_{step}.signal")
        with open(signal_path, "w") as f:
            f.write(f"train step={step}")

        # Wait for done
        done_path = os.path.join(self.work_dir, f"train_done_{step}.signal")
        for _ in range(6000):
            if os.path.exists(done_path):
                break
            if self.worker_proc.poll() is not None:
                raise RuntimeError(f"Worker died during step {step}")
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Worker timed out on step {step}")

        ms = (time.time() - t0) * 1000

        metrics = {}
        metrics_path = os.path.join(self.work_dir, f"train_metrics_step{step}.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

        self.results.append(
            StepResult(step=step, phase="training", duration_ms=ms, metrics=metrics)
        )

        loss_str = f"loss={metrics['loss']:.4f} " if "loss" in metrics else ""
        load_str = f"load={metrics.get('load_ms', 0):.0f}ms " if metrics.get("load_ms") else ""
        train_str = f"train={metrics.get('train_ms', 0):.0f}ms" if metrics.get("train_ms") else ""
        print(f"  [Train]    {loss_str}{load_str}{train_str} total={ms:.0f}ms")
        return metrics

    def train_loop(self):
        print(f"\n{'=' * 60}")
        print(f" Colocated Training Loop: {self.args.steps} steps")
        print(f"{'=' * 60}\n")

        for step in range(self.args.steps):
            step_t0 = time.time()
            print(f"[Step {step}/{self.args.steps}]")

            completions_path = self.rollout_phase(step)
            sleep_ms = self.sleep_phase()
            metrics = self.training_phase(step)
            wake_ms = self.wake_phase()

            self.results.append(
                StepResult(step=step, phase="overhead", duration_ms=sleep_ms + wake_ms,
                           metrics={"sleep_ms": sleep_ms, "wake_ms": wake_ms})
            )

            total_ms = (time.time() - step_t0) * 1000
            print(f"  [Total]    {total_ms:.0f}ms\n")

        self._print_summary()

    def _print_summary(self):
        rollout_ms = sum(r.duration_ms for r in self.results if r.phase == "rollout")
        training_ms = sum(r.duration_ms for r in self.results if r.phase == "training")
        overhead_ms = sum(r.duration_ms for r in self.results if r.phase == "overhead")
        total_ms = rollout_ms + training_ms + overhead_ms

        print("=" * 60)
        print("COLOCATED TRAINING SUMMARY (Persistent Worker)")
        print("=" * 60)
        print(f"  Steps:          {self.args.steps}")
        print(f"  GPUs:           {self.args.num_gpus} (all shared)")
        print(f"  Architecture:   vLLM(TP={self.args.num_gpus}) + persistent torchrun")
        print(f"  Rollout:        {rollout_ms / 1000:.1f}s ({rollout_ms / max(total_ms, 1) * 100:.1f}%)")
        print(f"  Training:       {training_ms / 1000:.1f}s ({training_ms / max(total_ms, 1) * 100:.1f}%)")
        print(f"  Sleep/Wake:     {overhead_ms / 1000:.1f}s ({overhead_ms / max(total_ms, 1) * 100:.1f}%)")
        print(f"  Total:          {total_ms / 1000:.1f}s")
        print(f"  Avg step:       {total_ms / max(self.args.steps, 1) / 1000:.1f}s")
        print("=" * 60)

        if self.args.output_log:
            with open(self.args.output_log, "w") as f:
                for r in self.results:
                    f.write(json.dumps(asdict(r)) + "\n")
                f.write(json.dumps({
                    "type": "summary",
                    "steps": self.args.steps,
                    "num_gpus": self.args.num_gpus,
                    "rollout_ms": round(rollout_ms, 1),
                    "training_ms": round(training_ms, 1),
                    "overhead_ms": round(overhead_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "avg_step_ms": round(total_ms / max(self.args.steps, 1), 1),
                }) + "\n")
            print(f"\nResults: {self.args.output_log}")

    def cleanup(self):
        # Shutdown workers
        shutdown_path = os.path.join(self.work_dir, "shutdown.signal")
        with open(shutdown_path, "w") as f:
            f.write("shutdown")
        if self.worker_proc and self.worker_proc.poll() is None:
            self.worker_proc.wait(timeout=30)
        if self.llm is not None:
            del self.llm
        shutil.rmtree(self.work_dir, ignore_errors=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--samples-per-prompt", type=int, default=4)
    p.add_argument("--max-prompt-len", type=int, default=256)
    p.add_argument("--max-completion-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--vllm-sleep-level", type=int, default=2, choices=[1, 2])
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.40)
    p.add_argument("--output-log", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    orch = ColocatedOrchestrator(args)
    try:
        orch.setup()
        orch.train_loop()
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
