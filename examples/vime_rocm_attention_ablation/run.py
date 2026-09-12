# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plan or run the four-arm ROCm Vime Attention implementation matrix.

Every arm starts from the same model and Megatron checkpoint, uses the same
prompt data and seeds, and changes only ``RL_KERNEL_ATTENTION_CASE``.  FFN and
Logp remain on the strict ``R/R`` path. Runtime JSON is written below ``--run-dir``; this example
does not require or ship a checked-in JSON configuration or result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

try:
    from .validate_artifacts import (
        CASE_IMPLEMENTATIONS,
        validate_arm,
        validate_matrix,
        write_report,
    )
except ImportError:  # Direct ``python examples/.../run.py`` execution.
    from validate_artifacts import (  # type: ignore[no-redef]
        CASE_IMPLEMENTATIONS,
        validate_arm,
        validate_matrix,
        write_report,
    )

PLAN_SCHEMA_VERSION = "rlkernel.vime_rocm_attention_operator_plan.v1"
FROZEN_SCHEMA_VERSION = "rlkernel.vime_rocm_attention_frozen_inputs.v1"
CASE_ORDER = ("P/P", "P/R", "R/P", "R/R")
RL_KERNEL_PLUGIN_ENTRY_POINT = "rl_engine.integrations.vllm_runtime:register_vllm_plugin"

# Proxy settings inherited from an operator shell break long rollout requests
# (httpx and the router honor them; a proxy's upstream timeout yields 502s).
_PROXY_ENVIRONMENT = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
)
_CUDA_ONLY_ENVIRONMENT = (
    "CUBLASLT_WORKSPACE_SIZE",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_ALGO",
    "RL_KERNEL_CUDA_ONLY",
    "RL_KERNEL_DET_GEMM_SM90_ONLY",
    "RL_KERNEL_PRECOMPILE_FA4",
    "VLLM_BATCH_INVARIANT",
)


@dataclass(frozen=True)
class MatrixConfig:
    vime_root: Path
    rl_kernel_root: Path
    megatron_root: Path
    model_root: Path
    reference_checkpoint: Path
    prompt_data: Path
    run_dir: Path
    launcher: Path
    visible_gpus: str = "0,1,2,3,4,5,6,7"
    num_gpus: int = 8
    tensor_parallel_size: int = 4
    context_parallel_size: int = 2
    rollout_tensor_parallel_size: int = 4
    colocate: bool = True
    offload_train: bool = False
    offload_rollout: bool = True
    router_policy: str = "round_robin"
    num_rollout: int = 1
    rollout_batch_size: int = 2
    samples_per_prompt: int = 1
    global_batch_size: int = 2
    real_vocab_size: int = 151936
    padded_vocab_size: int = 152064
    max_response_length: int = 32
    max_tokens_per_gpu: int = 256
    seed: int = 1234
    rollout_seed: int = 42
    ray_port: int = 6385
    ray_dashboard_port: int = 28265

    @property
    def training_gpus(self) -> int:
        return self.tensor_parallel_size * self.context_parallel_size

    @property
    def rollout_gpus(self) -> int:
        return self.num_gpus if self.colocate else self.num_gpus - self.training_gpus

    @property
    def rollout_engines(self) -> int:
        return self.rollout_gpus // self.rollout_tensor_parallel_size

    def validate(self, *, require_paths: bool) -> None:
        visible = [item.strip() for item in self.visible_gpus.split(",") if item.strip()]
        if len(visible) != self.num_gpus or len(set(visible)) != len(visible):
            raise ValueError("visible_gpus must contain exactly num_gpus unique device IDs")
        for name in (
            "num_gpus",
            "tensor_parallel_size",
            "context_parallel_size",
            "rollout_tensor_parallel_size",
            "num_rollout",
            "rollout_batch_size",
            "samples_per_prompt",
            "global_batch_size",
            "real_vocab_size",
            "padded_vocab_size",
            "max_response_length",
            "max_tokens_per_gpu",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.training_gpus > self.num_gpus:
            raise ValueError("training TP*CP cannot exceed num_gpus")
        if self.colocate and self.training_gpus != self.num_gpus:
            raise ValueError("colocated training TP*CP must use all visible GPUs")
        if self.rollout_gpus <= 0:
            raise ValueError("non-colocated training TP*CP must leave GPUs for rollout")
        if self.rollout_gpus % self.rollout_tensor_parallel_size:
            raise ValueError("rollout GPU count must be divisible by rollout TP")
        if self.router_policy != "round_robin":
            raise ValueError("the two-engine strict matrix requires round_robin routing")
        generated = self.rollout_batch_size * self.samples_per_prompt
        if generated < self.rollout_engines:
            raise ValueError(
                "rollout_batch_size*samples_per_prompt must issue at least one "
                "request per rollout engine"
            )
        if self.global_batch_size != generated:
            raise ValueError(
                "global_batch_size must equal rollout_batch_size*samples_per_prompt "
                "for the one-step frozen matrix"
            )
        if self.real_vocab_size > self.padded_vocab_size:
            raise ValueError("real_vocab_size cannot exceed padded_vocab_size")
        if self.padded_vocab_size % self.tensor_parallel_size:
            raise ValueError("padded_vocab_size must be divisible by training TP")
        if self.padded_vocab_size % 64:
            raise ValueError("padded_vocab_size must be divisible by 64 vocab tiles")
        if not 1024 <= self.ray_port <= 65535 or not 1024 <= self.ray_dashboard_port <= 65535:
            raise ValueError("Ray ports must be between 1024 and 65535")
        if abs(self.ray_port - self.ray_dashboard_port) < len(CASE_ORDER):
            raise ValueError("Ray GCS and dashboard port ranges overlap")
        dashboard_ports = range(self.ray_dashboard_port, self.ray_dashboard_port + len(CASE_ORDER))
        if any(10001 <= port <= 19999 for port in dashboard_ports):
            raise ValueError("Ray dashboard ports overlap the default client/worker range")
        if not require_paths:
            return
        required = {
            "vime_root": self.vime_root,
            "rl_kernel_root": self.rl_kernel_root,
            "megatron_root": self.megatron_root,
            "model_root": self.model_root,
            "reference_checkpoint": self.reference_checkpoint,
            "prompt_data": self.prompt_data,
            "launcher": self.launcher,
        }
        for name, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} does not exist: {path}")
        if not (self.vime_root / "train.py").is_file():
            raise FileNotFoundError(f"Vime train.py is missing below {self.vime_root}")
        if not (self.vime_root / "scripts" / "models" / "qwen3-8B.sh").is_file():
            raise FileNotFoundError("Vime Qwen3-8B model argument script is missing")
        if not (self.rl_kernel_root / "rl_engine").is_dir():
            raise FileNotFoundError("rl_kernel_root does not contain rl_engine")
        metrics_hook = (
            self.rl_kernel_root / "examples" / "vime_rocm_attention_ablation" / "tis_metrics.py"
        )
        if not metrics_hook.is_file():
            raise FileNotFoundError(f"Attention mismatch metrics hook is missing: {metrics_hook}")
        _validate_checkpoint_marker(self.reference_checkpoint)
        _validate_rl_kernel_plugin_installation()

    def frozen_parameters(self) -> dict[str, Any]:
        return {
            "model": "Qwen/Qwen3-8B",
            "visible_gpus": self.visible_gpus,
            "num_gpus": self.num_gpus,
            "training": {
                "num_gpus": self.training_gpus,
                "tensor_parallel_size": self.tensor_parallel_size,
                "context_parallel_size": self.context_parallel_size,
                "pipeline_parallel_size": 1,
                "sequence_parallel": False,
                "dtype": "bf16",
                "attention_backend": "flash",
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
            },
            "rollout": {
                "num_gpus": self.rollout_gpus,
                "engine_count": self.rollout_engines,
                "tensor_parallel_size": self.rollout_tensor_parallel_size,
                "router_policy": self.router_policy,
                "temperature": 1.0,
                "top_p": 1.0,
                # Vime's deterministic-inference flag exports
                # VLLM_BATCH_INVARIANT=1, which native ROCM_AITER_FA correctly
                # declares unsupported.  The R route owns its deterministic
                # per-row schedule; the P route remains the native baseline.
                "vllm_batch_invariant": False,
                "enforce_eager": False,
                "execution_mode": "compiled_hip_graph",
                "custom_all_reduce": False,
                "attention_backend": "ROCM_AITER_FA",
                "shuffle_kv_cache_layout": False,
            },
            "placement": {
                "colocate": self.colocate,
                "offload_train": self.offload_train,
                "offload_rollout": self.offload_rollout,
            },
            "batch": {
                "start_rollout_id": 0,
                "num_rollout": self.num_rollout,
                "rollout_batch_size": self.rollout_batch_size,
                "samples_per_prompt": self.samples_per_prompt,
                "global_batch_size": self.global_batch_size,
                "max_response_length": self.max_response_length,
                "max_tokens_per_gpu": self.max_tokens_per_gpu,
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-6,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.98,
            },
            "seed": self.seed,
            "rollout_seed": self.rollout_seed,
            "mismatch_metrics_hook": ("vime_rocm_attention_ablation.tis_metrics.metrics_only_tis"),
            "ffn_case": "R/R",
            "logp_case": "R/R",
            "real_vocab_size": self.real_vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "platform": "rocm",
        }

    def paths(self) -> dict[str, str]:
        return {
            "vime_root": str(self.vime_root.resolve()),
            "rl_kernel_root": str(self.rl_kernel_root.resolve()),
            "megatron_root": str(self.megatron_root.resolve()),
            "model_root": str(self.model_root.resolve()),
            "reference_checkpoint": str(self.reference_checkpoint.resolve()),
            "prompt_data": str(self.prompt_data.resolve()),
            "launcher": str(self.launcher.resolve()),
            "run_dir": str(self.run_dir.resolve()),
        }


def _validate_checkpoint_marker(checkpoint: Path) -> None:
    marker = checkpoint / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(
            "reference_checkpoint is not a Megatron checkpoint: missing " f"{marker}"
        )
    value = marker.read_text(encoding="utf-8").strip()
    if value == "release":
        return
    try:
        iteration = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid Megatron checkpoint marker {marker}: {value!r}") from exc
    if iteration < 0:
        raise ValueError(f"invalid Megatron checkpoint iteration in {marker}: {iteration}")


def _normalize_distribution_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _validate_rl_kernel_plugin_installation() -> None:
    """Fail before Ray starts unless vLLM can discover the installed plugin."""

    try:
        distribution = importlib_metadata.distribution("RL-Kernel")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "RL-Kernel must be installed (for example, `pip install -e <rl-kernel-root>`) "
            "so vLLM can discover its plugin entry point"
        ) from exc

    candidates = [
        entry_point
        for entry_point in importlib_metadata.entry_points(group="vllm.general_plugins")
        if entry_point.name == "rl_kernel"
    ]
    if not candidates:
        raise RuntimeError(
            "installed RL-Kernel does not expose the `rl_kernel` entry point in "
            "vllm.general_plugins"
        )
    matching = [
        entry_point
        for entry_point in candidates
        if entry_point.value == RL_KERNEL_PLUGIN_ENTRY_POINT
    ]
    if not matching:
        values = sorted({entry_point.value for entry_point in candidates})
        raise RuntimeError(
            "the visible rl_kernel vLLM plugin entry point has an unexpected target: " f"{values!r}"
        )

    installed_name = distribution.metadata.get("Name", "RL-Kernel")
    owners = {
        entry_point.dist.metadata.get("Name", entry_point.dist.name)
        for entry_point in matching
        if getattr(entry_point, "dist", None) is not None
    }
    if owners and _normalize_distribution_name(installed_name) not in {
        _normalize_distribution_name(owner) for owner in owners
    }:
        raise RuntimeError(
            "the visible rl_kernel vLLM plugin entry point is not owned by the "
            "installed RL-Kernel distribution"
        )


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fingerprint_path(path: Path) -> dict[str, Any]:
    """Seal a file or tree without rereading every multi-GB weight shard.

    Individual files (the prompt dataset and launcher) are content-hashed. For
    checkpoint trees, every relative path, size, mtime, and symlink target is
    sealed, while small checkpoint/config manifests receive an additional
    content hash.
    """

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    single_file = resolved.is_file()
    files = (
        [resolved]
        if single_file
        else sorted(item for item in resolved.rglob("*") if item.is_file())
    )
    aggregate = hashlib.sha256()
    byte_count = 0
    content_hashed = 0
    for item in files:
        relative = item.name if single_file else item.relative_to(resolved).as_posix()
        stat = item.stat()
        size = stat.st_size
        link_target = os.readlink(item) if item.is_symlink() else None
        manifest = (
            single_file
            or item.name == "latest_checkpointed_iteration.txt"
            or item.name.endswith(".index.json")
            or item.name
            in {
                "config.json",
                "generation_config.json",
                "metadata.json",
                "tokenizer.json",
                "tokenizer_config.json",
            }
        )
        record = {
            "path": relative,
            "size": size,
            "mtime_ns": stat.st_mtime_ns,
            "symlink": link_target,
            "content_sha256": _hash_file(item)[0] if manifest else None,
        }
        content_hashed += int(manifest)
        aggregate.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        aggregate.update(b"\n")
        byte_count += size
    return {
        "path": str(resolved),
        "kind": "file" if single_file else "directory",
        "seal_mode": "content" if single_file else "metadata_plus_checkpoint_manifests",
        "file_count": len(files),
        "content_hashed_file_count": content_hashed,
        "byte_count": byte_count,
        "sha256": aggregate.hexdigest(),
    }


def _git_identity(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    try:
        revision = run("rev-parse", "HEAD").strip()
        status = run("status", "--porcelain=v1", "--untracked-files=no")
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"source root is not a readable Git checkout: {path}") from exc
    return {
        "path": str(path.resolve()),
        "revision": revision,
        "tracked_dirty": bool(status.strip()),
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def frozen_input_manifest(config: MatrixConfig) -> dict[str, Any]:
    """Seal inputs and tracked source state before/after the four executions."""

    payload = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "parameters": config.frozen_parameters(),
        "inputs": {
            "model_root": fingerprint_path(config.model_root),
            "reference_checkpoint": fingerprint_path(config.reference_checkpoint),
            "prompt_data": fingerprint_path(config.prompt_data),
        },
        "sources": {
            "vime": _git_identity(config.vime_root),
            "rl_kernel": _git_identity(config.rl_kernel_root),
            "megatron": _git_identity(config.megatron_root),
        },
        "launcher": fingerprint_path(config.launcher),
    }
    payload["fingerprint"] = _canonical_fingerprint(payload)
    return payload


def case_slug(case_id: str) -> str:
    if case_id not in CASE_IMPLEMENTATIONS:
        raise ValueError(f"unknown case {case_id!r}")
    return case_id.lower().replace("/", "-")


def build_arm_environment(
    config: MatrixConfig,
    case_id: str,
    arm_dir: Path,
    *,
    arm_index: int,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the exact inherited environment for one Vime/Ray arm."""

    if case_id not in CASE_IMPLEMENTATIONS:
        raise ValueError(f"unknown case {case_id!r}")
    env = dict(os.environ if base_environment is None else base_environment)
    for name in _CUDA_ONLY_ENVIRONMENT:
        env.pop(name, None)
    for name in _PROXY_ENVIRONMENT:
        env.pop(name, None)
    existing_pythonpath = env.get("PYTHONPATH", "")
    python_paths = [
        str((config.rl_kernel_root / "examples").resolve()),
        str(config.rl_kernel_root.resolve()),
        str(config.vime_root.resolve()),
        str(config.megatron_root.resolve()),
    ]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONUNBUFFERED": "1",
            "HIP_VISIBLE_DEVICES": config.visible_gpus,
            # PyTorch on ROCm and Ray still consume CUDA_VISIBLE_DEVICES.
            "CUDA_VISIBLE_DEVICES": config.visible_gpus,
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "0",
            "VLLM_ATTENTION_BACKEND": "ROCM_AITER_FA",
            "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE": "32",
            "RL_KERNEL_ATTENTION_CASE": case_id,
            "RL_KERNEL_FFN_CASE": "R/R",
            "RL_KERNEL_LOGP_CASE": "R/R",
            "RL_KERNEL_VLLM_REAL_VOCAB_SIZE": str(config.real_vocab_size),
            "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE": str(config.padded_vocab_size),
            "RL_KERNEL_VLLM_INTEGRATION": "1",
            "RL_KERNEL_READBACK_DIR": str((arm_dir / "readbacks").resolve()),
            "RL_KERNEL_MISMATCH_SIDECAR_DIR": str((arm_dir / "mismatch_sidecars").resolve()),
            "RLK_ABLATION_CASE_ID": case_id,
            "RLK_ABLATION_ARM_DIR": str(arm_dir.resolve()),
            "RLK_ABLATION_VIME_ROOT": str(config.vime_root.resolve()),
            "RLK_ABLATION_RL_KERNEL_ROOT": str(config.rl_kernel_root.resolve()),
            "RLK_ABLATION_MEGATRON_ROOT": str(config.megatron_root.resolve()),
            "RLK_ABLATION_MODEL_ROOT": str(config.model_root.resolve()),
            "RLK_ABLATION_REFERENCE_CHECKPOINT": str(config.reference_checkpoint.resolve()),
            "RLK_ABLATION_PROMPT_DATA": str(config.prompt_data.resolve()),
            "RLK_ABLATION_NUM_GPUS": str(config.num_gpus),
            "RLK_ABLATION_TP_SIZE": str(config.tensor_parallel_size),
            "RLK_ABLATION_CP_SIZE": str(config.context_parallel_size),
            "RLK_ABLATION_ROLLOUT_TP_SIZE": str(config.rollout_tensor_parallel_size),
            "RLK_ABLATION_COLOCATE": "1" if config.colocate else "0",
            "RLK_ABLATION_OFFLOAD_TRAIN": "1" if config.offload_train else "0",
            "RLK_ABLATION_OFFLOAD_ROLLOUT": "1" if config.offload_rollout else "0",
            "RLK_ABLATION_ROUTER_POLICY": config.router_policy,
            "RLK_ABLATION_NUM_ROLLOUT": str(config.num_rollout),
            "RLK_ABLATION_ROLLOUT_BATCH_SIZE": str(config.rollout_batch_size),
            "RLK_ABLATION_SAMPLES_PER_PROMPT": str(config.samples_per_prompt),
            "RLK_ABLATION_GLOBAL_BATCH_SIZE": str(config.global_batch_size),
            "RLK_ABLATION_MAX_RESPONSE_LENGTH": str(config.max_response_length),
            "RLK_ABLATION_MAX_TOKENS_PER_GPU": str(config.max_tokens_per_gpu),
            "RLK_ABLATION_SEED": str(config.seed),
            "RLK_ABLATION_ROLLOUT_SEED": str(config.rollout_seed),
            "RLK_ABLATION_RAY_PORT": str(config.ray_port + arm_index),
            "RLK_ABLATION_RAY_DASHBOARD_PORT": str(config.ray_dashboard_port + arm_index),
            "RLK_ABLATION_RAY_DASHBOARD_AGENT_PORT": str(
                config.ray_dashboard_port + arm_index + 10_000
            ),
        }
    )
    return env


def public_arm_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return only experiment variables; never serialize arbitrary host secrets."""

    names = {
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "RL_KERNEL_ATTENTION_CASE",
        "RL_KERNEL_ROCM_FIXED_PAGED_TILE",
        "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS",
        "RL_KERNEL_DET_GEMM_BACKEND",
        "RL_KERNEL_ROCM_ATTENTION_BACKEND",
        "RL_KERNEL_FFN_CASE",
        "RL_KERNEL_LOGP_CASE",
        "RL_KERNEL_VLLM_REAL_VOCAB_SIZE",
        "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE",
        "RL_KERNEL_MISMATCH_SIDECAR_DIR",
        "RL_KERNEL_READBACK_DIR",
        "RL_KERNEL_VLLM_INTEGRATION",
        "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE",
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT",
        "VLLM_ATTENTION_BACKEND",
    }
    return {
        name: value
        for name, value in sorted(environment.items())
        if name in names or name.startswith("RLK_ABLATION_")
    }


def build_plan(config: MatrixConfig) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for index, case_id in enumerate(CASE_ORDER):
        arm_dir = config.run_dir / "arms" / case_slug(case_id)
        environment = build_arm_environment(
            config,
            case_id,
            arm_dir,
            arm_index=index,
            base_environment={},
        )
        arms.append(
            {
                "case_id": case_id,
                "training_implementation": CASE_IMPLEMENTATIONS[case_id]["training"],
                "rollout_implementation": CASE_IMPLEMENTATIONS[case_id]["rollout"],
                "arm_dir": str(arm_dir.resolve()),
                "command": ["bash", str(config.launcher.resolve())],
                "environment": public_arm_environment(environment),
            }
        )
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "matrix_kind": "attention_operator_implementation_cross_config",
        "parameters": config.frozen_parameters(),
        "paths": config.paths(),
        "arms": arms,
        "claim_boundary": {
            "implemented": ["P/P", "P/R", "R/P", "R/R"],
            "not_implemented": "A0-A7 runtime mutation matrix",
        },
    }
    payload["plan_fingerprint"] = _canonical_fingerprint(payload)
    return payload


def _prepare_run_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def execute_matrix(config: MatrixConfig, *, fail_fast: bool = False) -> dict[str, Any]:
    config.validate(require_paths=True)
    _prepare_run_dir(config.run_dir)
    plan = build_plan(config)
    write_report(config.run_dir / "matrix-plan.json", plan)

    frozen_before = frozen_input_manifest(config)
    write_report(config.run_dir / "frozen-inputs.before.json", frozen_before)
    reports: dict[str, Mapping[str, Any]] = {}

    for index, case_id in enumerate(CASE_ORDER):
        arm_dir = config.run_dir / "arms" / case_slug(case_id)
        readback_dir = arm_dir / "readbacks"
        dump_dir = arm_dir / "dump"
        checkpoint_dir = arm_dir / "checkpoint"
        sidecar_dir = arm_dir / "mismatch_sidecars"
        for directory in (readback_dir, dump_dir, checkpoint_dir, sidecar_dir):
            directory.mkdir(parents=True, exist_ok=False)
        environment = build_arm_environment(
            config,
            case_id,
            arm_dir,
            arm_index=index,
        )
        arm_manifest = {
            "schema_version": "rlkernel.vime_rocm_attention_arm_launch.v1",
            "case_id": case_id,
            "expected_implementations": CASE_IMPLEMENTATIONS[case_id],
            "frozen_input_fingerprint": frozen_before["fingerprint"],
            "command": ["bash", str(config.launcher.resolve())],
            "environment": public_arm_environment(environment),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        write_report(arm_dir / "launch.json", arm_manifest)

        log_path = arm_dir / "launcher.log"
        returncode = 127
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(
                    ["bash", str(config.launcher.resolve())],
                    cwd=config.rl_kernel_root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            returncode = process.returncode
        except OSError as exc:
            log_path.write_text(f"failed to start launcher: {exc}\n", encoding="utf-8")

        report = validate_arm(arm_dir, case_id, launcher_returncode=returncode)
        write_report(arm_dir / "validation.json", report)
        reports[case_id] = report
        if fail_fast and report["passed"] is not True:
            break

    frozen_after = frozen_input_manifest(config)
    write_report(config.run_dir / "frozen-inputs.after.json", frozen_after)
    summary = validate_matrix(
        reports,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
    )
    summary = {
        **summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(config.run_dir.resolve()),
        "plan_fingerprint": plan["plan_fingerprint"],
    }
    write_report(config.run_dir / "matrix-validation.json", summary)
    return summary


def _path_argument(value: str | None, environment_name: str) -> Path | None:
    selected = value or os.getenv(environment_name)
    return None if not selected else Path(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vime-root", default=None)
    parser.add_argument("--rl-kernel-root", default=None)
    parser.add_argument("--megatron-root", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--prompt-data", default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs")
        / "vime_rocm_attention_ablation"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path(__file__).with_name("launch_arm.sh"),
    )
    parser.add_argument("--visible-gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--rollout-tp-size", type=int, default=4)
    parser.add_argument("--num-rollout", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--max-response-length", type=int, default=32)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--ray-port", type=int, default=6385)
    parser.add_argument("--ray-dashboard-port", type=int, default=28265)
    parser.add_argument("--run", action="store_true", help="execute all four Vime arms")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> MatrixConfig:
    values = {
        "vime_root": _path_argument(args.vime_root, "VIME_ROOT"),
        "rl_kernel_root": _path_argument(args.rl_kernel_root, "RL_KERNEL_ROOT"),
        "megatron_root": _path_argument(args.megatron_root, "MEGATRON_ROOT"),
        "model_root": _path_argument(args.model_root, "MODEL_ROOT"),
        "reference_checkpoint": _path_argument(args.reference_checkpoint, "TORCH_DIST_ROOT"),
        "prompt_data": _path_argument(args.prompt_data, "PROMPT_DATA"),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"missing required paths: {options}")
    return MatrixConfig(
        **values,  # type: ignore[arg-type]
        run_dir=args.run_dir,
        launcher=args.launcher,
        visible_gpus=args.visible_gpus,
        num_gpus=args.num_gpus,
        tensor_parallel_size=args.tp_size,
        context_parallel_size=args.cp_size,
        rollout_tensor_parallel_size=args.rollout_tp_size,
        num_rollout=args.num_rollout,
        rollout_batch_size=args.rollout_batch_size,
        samples_per_prompt=args.samples_per_prompt,
        global_batch_size=args.global_batch_size,
        max_response_length=args.max_response_length,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        seed=args.seed,
        rollout_seed=args.rollout_seed,
        ray_port=args.ray_port,
        ray_dashboard_port=args.ray_dashboard_port,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        config.validate(require_paths=args.run)
        if args.run:
            report = execute_matrix(config, fail_fast=args.fail_fast)
        else:
            report = {"status": "planned", **build_plan(config)}
    except Exception as exc:
        report = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "status": "error",
            "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "error": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
