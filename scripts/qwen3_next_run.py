#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the Qwen3-Next full-model chain inside any allocation with four B200 GPUs.

    python scripts/qwen3_next_run.py --stage all \\
        --runs-root <dir> --checkpoint <Qwen3-Next-80B-A3B-Instruct> --cuda-home <cuda-13.0>

Stages run in order, each in its own run directory and its own clean clone of
the current commit; ``--from``/``--until`` select a contiguous part. The stages
are ``preflight`` (environment, operator gates, device guards, TP4 probe,
checkpoint hashes, rollout smoke), ``layers`` (per-layer and
scheduler gates, which certify the shared provider), ``l2`` (full-model
train/prefill/decode raw-bit matrix), then ``train1``, ``train2`` and ``train``
(two optimizer updates, export, L2 on the updated weights, reload and
resumption). A single stage takes ``--previous <run dir>`` of its predecessor;
``bi`` (the batch-invariance matrix) runs after a passed ``l2``.

Every run directory holds ``run.json`` (the immutable manifest, identified by a
fresh ``execution_id``), ``attempts/<n>/`` (the stage's output) and, once the
stage passes, ``result.json``. The launcher refuses a dirty tree and never
talks to a batch scheduler; wrap it in whatever allocation command the cluster
uses.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qwen3_next_contract import (
    CONTRACT_VERSION,
    checkpoint_identity,
    operator_requirements,
    validate_result,
)
from qwen3_next_layer_contract import layer_requirements
from qwen3_next_runtime import (
    STAGES,
    atomic_json,
    failure_fields,
    file_digest,
    interpreter_path,
    probe_environment,
    require_predecessor,
    verify_environment,
    wait_owned_child,
)

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("preflight", "layers", "l2", "train1", "train2", "train")
PROVIDER_SOURCES = (
    "rl_engine/integrations/qwen3_next_provider.py",
    "rl_engine/integrations/qwen3_next_conv.py",
    "rl_engine/integrations/qwen3_next_gdn.py",
    "rl_engine/integrations/qwen3_next_engine_weights.py",
    "rl_engine/integrations/qwen3_next_profile.py",
    "rl_engine/integrations/qwen3_next_tp.py",
    "rl_engine/integrations/qwen3_next_tp_gdn.py",
    "rl_engine/integrations/qwen3_next_tp_blocks.py",
    "rl_engine/integrations/qwen3_next_forward.py",
    "rl_engine/integrations/qwen3_next_embedding.py",
    "rl_engine/integrations/qwen3_next_logprob.py",
    "rl_engine/integrations/qwen3_next_vime.py",
    "rl_engine/integrations/qwen3_next_vllm.py",
    "rl_engine/integrations/qwen3_next_vllm_attention.py",
    "rl_engine/integrations/qwen3_next_vllm_model.py",
    "rl_engine/kernels/ops/cuda/norm/rmsnorm.py",
    "csrc/cuda/rmsnorm.cu",
)


def _git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def provider_config():
    return {
        "profile": "qwen3_next_tp4_shared_eager_v1",
        "scope": "shared_full_forward_implementation",
        "scheduler_batch_invariance_certified": False,
        "sources": {name: file_digest(ROOT / name) for name in PROVIDER_SOURCES},
        "weights": "bfloat16",
        "recurrent_state": "float32",
        "mtp": False,
        "prefix_reuse": False,
        "dropout": False,
        "dynamic_tuning": False,
        "cuda_graph": False,
        "sequence_parallel": False,
        "fp32_residual_connection": False,
        "vocab_size": 151936,
        "padded_vocab_size": 151936,
        "expert_tensor_parallel_size": 4,
        # Large enough that the batch-invariance matrix (up to 64 concurrent
        # requests) runs under the same certified provider identity.
        "native_rollout_kv_cache_memory_bytes": 4 << 30,
        "runtime_env": {"TRITON_F32_DEFAULT": "ieee", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    }


def prepare(stage, previous, args):
    """Create the run directory, its clean source clone and its immutable manifest."""
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Commit the source before starting an immutable run")
    commit = _git("rev-parse", "HEAD")
    checkpoint = args.checkpoint.resolve(strict=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    execution_id = uuid4().hex
    run_id = f"{stamp}-{stage}-{commit[:8]}-{execution_id[:8]}"
    run_dir = args.runs_root.resolve() / run_id
    run_dir.mkdir(parents=True)
    source = run_dir / "source"
    subprocess.run(
        ["git", "clone", "-q", "--shared", "--no-checkout", str(ROOT), str(source)], check=True
    )
    subprocess.run(["git", "checkout", "-q", "--detach", commit], cwd=source, check=True)
    python = interpreter_path(args.python)
    manifest = {
        "run_id": run_id,
        "execution_id": execution_id,
        "contract_version": CONTRACT_VERSION,
        "suite": "all",
        "execution_gpus": 4,
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "stage": stage,
        "source_commit": commit,
        "source": str(source),
        "python": python,
        "cuda_home": str(args.cuda_home.resolve(strict=True)),
        "checkpoint": str(checkpoint),
        "memory": args.memory or STAGES[stage][0],
        "previous": str(previous) if previous else None,
        "status": "prepared",
        "full_model_l2": "not_evaluated",
    }
    if stage == "bi":
        manifest["bi_suite"] = args.bi_suite
    if args.training_storage_root and stage == "train1":
        manifest["training_storage"] = str(args.training_storage_root.resolve(strict=True) / run_id)
    environment = probe_environment(python)
    manifest["environment"] = environment
    manifest["environment_sha256"] = environment["sha256"]
    manifest["cuda_identity"] = {
        "home": manifest["cuda_home"],
        "nvcc_sha256": file_digest(Path(manifest["cuda_home"]) / "bin/nvcc"),
    }
    manifest["checkpoint_identity"] = checkpoint_identity(checkpoint)
    manifest["provider_config"] = provider_config()
    manifest["operator_requirements"] = operator_requirements(ROOT, "all")
    manifest["layer_requirements"] = layer_requirements(ROOT)
    manifest["topology"] = {"tp": 4, "pp": 1, "cp": 1, "ep": 1}
    require_predecessor(stage, previous, checkpoint, manifest)
    if stage == "l2":
        certificate = previous / "result.json"
        manifest["scheduler_profile_certificate"] = {
            "path": str(certificate),
            "sha256": file_digest(certificate),
        }
    elif stage in ("train1", "train2", "train", "bi"):
        manifest["scheduler_profile_certificate"] = json.loads((previous / "run.json").read_text())[
            "scheduler_profile_certificate"
        ]
    atomic_json(run_dir / "run.json", manifest)
    return run_dir, manifest


def stage_environment(manifest, run_dir, attempt):
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(
        CUDA_HOME=manifest["cuda_home"],
        PYTHONUNBUFFERED="1",
        VLLM_PLUGINS="",
        MAX_JOBS="4",
        KERNEL_ALIGN_DEV_RPATH="1",
        QWEN3_NEXT_ATTEMPT=attempt,
        QWEN3_NEXT_ALLOCATED_MEMORY=manifest["memory"],
        # Four native ranks share the host. Left unpinned, each rank spins a
        # thread per core on the ~50k small per-tensor loads of the HF import.
        OMP_NUM_THREADS="16",
        MKL_NUM_THREADS="16",
    )
    # Layer gates qualify the providers without vLLM's aten overrides; the
    # certified full-model stages run with them.
    if manifest["stage"] in ("preflight", "layers"):
        env["VLLM_BATCH_INVARIANT"] = "0"
    else:
        env["VLLM_BATCH_INVARIANT"] = "1"
        env["QWEN_SHARED_PROFILE_RESULT"] = manifest["scheduler_profile_certificate"]["path"]
    env.update(manifest["provider_config"]["runtime_env"])
    base_bin = Path(manifest["environment"]["identity"]["base_prefix"]) / "bin"
    env["PATH"] = (
        f"{Path(manifest['python']).parent}:{manifest['cuda_home']}/bin:{base_bin}:{env['PATH']}"
    )
    env["PYTHONPATH"] = manifest["source"]
    env["TRITON_CACHE_DIR"] = str(run_dir / "cache/triton")
    env["TORCH_EXTENSIONS_DIR"] = str(run_dir / "cache/torch")
    return env


def execute(run_dir, manifest):
    """Run one stage in its clone; on success write and validate ``result.json``."""
    attempt = "0"
    output = run_dir / "attempts" / attempt
    output.mkdir(parents=True)
    status = {
        "run_id": manifest["run_id"],
        "execution_id": manifest["execution_id"],
        "stage": manifest["stage"],
        "attempt": attempt,
        "status": "running",
        "full_model_l2": "not_evaluated",
    }
    atomic_json(output / "status.json", status)
    (run_dir / "RUNNING").touch()
    received = None

    def stop(signum, frame):
        nonlocal received
        received = received or signum

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    child = None
    try:
        verify_environment(manifest["environment"], require_venv=manifest["stage"] != "preflight")
        if manifest["environment"]["identity"]["build_tools"]["ninja"] is None:
            raise RuntimeError("environment_mismatch: locked Ninja build tool is missing")
        source = manifest["source"]
        if _git("rev-parse", "HEAD^{tree}", cwd=source) != manifest["source_tree"] or _git(
            "status", "--porcelain", "--untracked-files=no", cwd=source
        ):
            raise RuntimeError("source_mismatch: frozen tracked source changed")
        if (
            file_digest(Path(manifest["cuda_home"]) / "bin/nvcc")
            != manifest["cuda_identity"]["nvcc_sha256"]
        ):
            raise RuntimeError("environment_mismatch: CUDA toolkit changed")
        atomic_json(output / "runtime-identity.json", manifest["environment"])
        command = [
            manifest["python"],
            str(Path(source) / STAGES[manifest["stage"]][3]),
            "--manifest",
            str(run_dir / "run.json"),
            "--output",
            str(output),
        ]
        child = subprocess.Popen(
            command,
            cwd=source,
            env=stage_environment(manifest, run_dir, attempt),
            start_new_session=True,
        )
        code = wait_owned_child(child, lambda: received is not None)
        if received:
            status.update(status="interrupted", signal=received, failure_kind="unknown")
        elif code:
            status.update(
                status="failed",
                exit_code=code,
                **failure_fields(RuntimeError(f"Stage exited with code {code}")),
            )
            failure = output / "result.json"
            if failure.is_file():
                record = json.loads(failure.read_text())
                status.update({k: record[k] for k in ("failure_kind", "error") if k in record})
        else:
            result = json.loads((output / "result.json").read_text())
            validate_result(manifest, result, run_dir=run_dir, attempt=attempt)
            status.update(result)
            atomic_json(run_dir / "result.json", result)
    except Exception as exc:
        status.update(status="interrupted" if received else "failed", **failure_fields(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            wait_owned_child(child, lambda: True)
        atomic_json(output / "status.json", status)
        (run_dir / "RUNNING").unlink(missing_ok=True)
    return status


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=(*ORDER, "bi", "all"), required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--previous", type=Path, help="passed run directory of the predecessor")
    parser.add_argument(
        "--from",
        dest="first",
        choices=ORDER,
        default="preflight",
        help="with --stage all: first stage to run (later than preflight needs --previous)",
    )
    parser.add_argument(
        "--until", choices=ORDER, default="train", help="with --stage all: last stage to run"
    )
    parser.add_argument(
        "--memory",
        help="host memory this allocation grants (e.g. 1280G); default: the stage's budget",
    )
    parser.add_argument(
        "--bi-suite",
        choices=("full", "quick"),
        default="full",
        help="with --stage bi: the whole batch-invariance matrix or the CI subset",
    )
    parser.add_argument(
        "--training-storage-root",
        type=Path,
        help="shared directory for training checkpoints and exports (default: in the run)",
    )
    args = parser.parse_args()
    if args.stage == "all":
        stages = ORDER[ORDER.index(args.first) : ORDER.index(args.until) + 1]
        if not stages:
            raise SystemExit("--from must not come after --until")
    else:
        stages = (args.stage,)
    if (stages[0] == "preflight") != (args.previous is None):
        raise SystemExit("Every stage after preflight needs --previous <passed predecessor run>")
    previous = args.previous.resolve(strict=True) if args.previous else None
    for stage in stages:
        run_dir, manifest = prepare(stage, previous, args)
        print(f"[qwen3-next] {stage}: {run_dir}", flush=True)
        status = execute(run_dir, manifest)
        print(f"[qwen3-next] {stage}: {status['status']}", flush=True)
        if status["status"] != "passed":
            raise SystemExit(1)
        previous = run_dir


if __name__ == "__main__":
    main()
