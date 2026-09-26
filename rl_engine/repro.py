# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""User-facing launcher for the Qwen3 VIME TP4/CP2 reproduction.

The experiment itself remains implemented by the example runners.  This module
only resolves the profile, performs actionable preflight checks, and delegates
to those runners so the long-form reproduction runbook stays an audit escape
hatch rather than the normal user interface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXAMPLE_RELATIVE = Path("examples/vime_qwen3_8b_tp4_cp2_200")
DEFAULT_PROFILE = EXAMPLE_RELATIVE / "profiles/qwen3-8b-tp4-cp2.json"
ARM_ALIASES = {
    "g00": "native",
    "g01": "consistency",
    "native": "native",
    "consistency": "consistency",
}


class ReproError(RuntimeError):
    """An actionable user-facing reproduction error."""


def canonical_arm(value: str) -> str:
    """Map a user mode to the descriptive run name."""
    canonical = ARM_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise ReproError(f"unknown mode {value!r}; choose native or consistency")
    return canonical


@dataclass(frozen=True)
class Paths:
    workspace: Path
    rl_kernel_root: Path
    vime_root: Path
    megatron_root: Path
    runtime_root: Path
    cuda_runtime_root: Path
    runtime_site: Path
    cuda_python_site: Path
    te_root: Path
    data_root: Path
    model_root: Path
    ref_load: Path
    prompt_data: Path
    output_root: Path
    python: Path
    ray: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_profile(value: str | None) -> Path:
    value = value or os.environ.get("RLK_REPRO_PROFILE")
    local_profile = _repo_root() / ".rlk-profile.json"
    if not value and local_profile.is_file():
        value = str(local_profile)
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        repo_candidate = _repo_root() / candidate
        if repo_candidate.is_file():
            return repo_candidate.resolve()
        raise ReproError(f"profile does not exist: {candidate}")
    candidate = _repo_root() / DEFAULT_PROFILE
    if not candidate.is_file():
        raise ReproError(f"default profile is missing: {candidate}")
    return candidate


def _load_profile(value: str | None) -> tuple[dict[str, Any], Path]:
    path = _find_profile(value)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReproError(f"invalid JSON profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise ReproError(f"profile must be a JSON object: {path}")
    if profile.get("schema_version") != "rlkernel.repro.profile.v1":
        raise ReproError(f"unsupported profile schema in {path}")
    return profile, path


def _path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser().resolve()


def _profile_path(profile: dict[str, Any], key: str) -> str | None:
    paths = profile.get("paths", {})
    value = paths.get(key) if isinstance(paths, dict) else None
    return str(value) if value else None


def _override_or_env(value: str | None, env_name: str) -> str | None:
    return value or os.environ.get(env_name)


def _required_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise ReproError(
            f"{label} is not configured; pass the corresponding --{label.replace('_', '-')} "
            "option or set the documented RLK_REPRO_* environment variable"
        )
    return value


def resolve_paths(
    profile: dict[str, Any],
    *,
    workspace: str | None,
    arm: str,
    rl_kernel_root: str | None = None,
    vime_root: str | None = None,
    megatron_root: str | None = None,
    runtime_root: str | None = None,
    cuda_runtime_root: str | None = None,
    runtime_site: str | None = None,
    cuda_python_site: str | None = None,
    te_root: str | None = None,
    data_root: str | None = None,
    model_root: str | None = None,
    ref_load: str | None = None,
    prompt_data: str | None = None,
    output_root: str | None = None,
    python: str | None = None,
    ray: str | None = None,
) -> Paths:
    arm = canonical_arm(arm)
    workspace_value = _override_or_env(
        workspace or _profile_path(profile, "workspace"), "RLK_REPRO_WORKSPACE"
    )
    workspace_path = _required_path(_path(workspace_value), "workspace")

    def choose(value: str | None, profile_key: str, env_name: str, default: Path | None) -> Path:
        selected = _override_or_env(value or _profile_path(profile, profile_key), env_name)
        if selected and profile_key in {"python", "ray"}:
            # A venv Python is often a symlink to the system executable.
            # Resolving that symlink discards pyvenv.cfg and its packages.
            return Path(selected).expanduser().absolute()
        return _path(selected) or _required_path(default, profile_key)

    active_runtime = Path(sys.prefix).resolve()
    runtime_path = choose(runtime_root, "runtime_root", "RLK_REPRO_RUNTIME_ROOT", active_runtime)
    required_python = str(profile.get("requirements", {}).get("python", "3.11"))
    python_major_minor = ".".join(required_python.split(".")[:2])
    runtime_site_packages = runtime_path / f"lib/python{python_major_minor}/site-packages"
    runtime_python = (
        Path(sys.executable).absolute()
        if runtime_path == active_runtime
        else runtime_path / f"bin/python{python_major_minor}"
    )
    data_path = choose(data_root, "data_root", "RLK_REPRO_DATA_ROOT", workspace_path / "data")
    output_path = choose(
        output_root,
        "output_root",
        "RLK_REPRO_OUTPUT_ROOT",
        data_path / "runs" / "convergence",
    )
    paths = Paths(
        workspace=workspace_path,
        rl_kernel_root=choose(
            rl_kernel_root,
            "rl_kernel_root",
            "RLK_REPRO_RL_KERNEL_ROOT",
            _repo_root(),
        ),
        vime_root=choose(vime_root, "vime_root", "RLK_REPRO_VIME_ROOT", workspace_path / "vime"),
        megatron_root=choose(
            megatron_root,
            "megatron_root",
            "RLK_REPRO_MEGATRON_ROOT",
            workspace_path / "Megatron-LM",
        ),
        runtime_root=runtime_path,
        cuda_runtime_root=choose(
            cuda_runtime_root,
            "cuda_runtime_root",
            "RLK_REPRO_CUDA_RUNTIME_ROOT",
            runtime_path / "lib/python3.11/site-packages/nvidia/cuda_runtime",
        ),
        runtime_site=choose(
            runtime_site,
            "runtime_site",
            "RLK_REPRO_RUNTIME_SITE",
            runtime_site_packages,
        ),
        cuda_python_site=choose(
            cuda_python_site,
            "cuda_python_site",
            "RLK_REPRO_CUDA_PYTHON_SITE",
            runtime_site_packages,
        ),
        te_root=(
            _path(_override_or_env(te_root, "RLK_REPRO_TE_ROOT"))
            or _path(_profile_path(profile, f"te_root_{arm.lower()}"))
            or _path(_profile_path(profile, "te_root"))
            or _path(
                os.environ.get(str(profile.get("modes", {}).get(arm, {}).get("te_root_env", "")))
            )
            or runtime_site_packages
        ),
        data_root=data_path,
        model_root=choose(
            model_root,
            "model_root",
            "RLK_REPRO_MODEL_ROOT",
            workspace_path / "checkpoints/Qwen3-8B_vime_rlkernel_tp2_cp2",
        ),
        ref_load=choose(
            ref_load,
            "ref_load",
            "RLK_REPRO_REF_LOAD",
            workspace_path / "checkpoints/Qwen3-8B_torch_dist",
        ),
        prompt_data=choose(
            prompt_data,
            "prompt_data",
            "RLK_REPRO_PROMPT_DATA",
            data_path / "datasets/dapo-math-17k.vime.jsonl",
        ),
        output_root=output_path,
        python=choose(python, "python", "RLK_REPRO_PYTHON", runtime_python),
        ray=choose(ray, "ray", "RLK_REPRO_RAY", runtime_python.parent / "ray"),
    )
    return paths


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, env=env)


def _git_clean(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _gpu_names(backend: str = "cuda") -> list[str] | None:
    command = "rocm-smi" if backend == "rocm" else "nvidia-smi"
    executable = shutil.which(command)
    if not executable:
        return None
    if backend == "rocm":
        command_line = [executable, "--showproductname", "--json"]
    else:
        command_line = [executable, "--query-gpu=name", "--format=csv,noheader"]
    result = subprocess.run(command_line, check=False, capture_output=True, text=True)
    if result.returncode:
        return None
    if backend == "rocm":
        try:
            value = json.loads(result.stdout[result.stdout.index("{") :])
        except (ValueError, json.JSONDecodeError):
            return None
        names = []
        for key, card in value.items():
            if key.startswith("card") and isinstance(card, dict):
                names.append(str(card.get("Card series") or card.get("Card SKU") or key))
        return names or None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _require_idle_gpus() -> None:
    """Avoid submitting a colocated eight-GPU job over an existing workload."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.stdout.strip():
        raise ReproError(
            "GPUs have active compute processes; wait for them to finish before retrying. "
            "PID and memory: " + result.stdout.strip().replace("\n", "; ")
        )


def _python_versions(python: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata as m, platform, json; "
        "names=('torch','vllm','ray','transformer_engine','sympy','pylatexenc'); "
        "out={'python': platform.python_version()}; "
        "out.update({n: next((d.version for d in m.distributions() "
        "if (d.metadata.get('Name') or '').lower().replace('-','_') == n), 'missing') "
        "for n in names}); "
        "print(json.dumps(out, sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script], check=False, capture_output=True, text=True
        )
    except OSError as exc:
        return {"error": str(exc)}
    if result.returncode:
        return {"error": result.stderr.strip() or f"exit {result.returncode}"}
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stdout.strip()}
    return value if isinstance(value, dict) else {"error": "invalid version output"}


def _version_matches(actual: str, expected: str) -> bool:
    """Accept an exact frozen version with an optional local wheel suffix."""
    normalized = actual.split("+", 1)[0]
    return (
        actual == expected
        or normalized == expected
        or (expected.count(".") == 1 and normalized.startswith(expected + "."))
    )


def doctor(
    paths: Paths,
    profile: dict[str, Any],
    *,
    ray_address: str,
    as_json: bool = False,
    backend: str = "cuda",
) -> int:
    requirements = profile.get("requirements", {})
    expected = int(requirements.get("gpus", 8))
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    try:
        gpu_names = _gpu_names(backend)
    except TypeError:
        # Keep the small test seam and third-party callers that monkeypatch the
        # historical zero-argument probe working for the default CUDA path.
        if backend != "cuda":
            raise
        gpu_names = _gpu_names()
    gpu_count = len(gpu_names) if gpu_names is not None else None
    add(
        "gpu",
        gpu_count == expected,
        f"found {gpu_count if gpu_count is not None else 'unavailable'}, expected {expected}",
    )
    expected_gpu_model = str(requirements.get("gpu_model", "")).strip()
    if expected_gpu_model:
        add(
            "gpu_model",
            bool(gpu_names) and all(expected_gpu_model in name for name in gpu_names),
            f"found {gpu_names or 'unavailable'}, expected {expected_gpu_model}",
        )
    for name, path in asdict(paths).items():
        if name in {"workspace", "data_root", "output_root"}:
            continue
        if backend == "rocm" and name == "cuda_runtime_root":
            continue
        exists = path.exists()
        add(name, exists, str(path) if exists else f"missing: {path}")
    for name, path in (
        ("rl_kernel_git", paths.rl_kernel_root),
        ("vime_git", paths.vime_root),
        ("megatron_git", paths.megatron_root),
    ):
        add(name, (path / ".git").exists(), f"{path}/.git")
    for name, path in (
        ("model", paths.model_root),
        ("reference_checkpoint", paths.ref_load),
        ("prompt_data", paths.prompt_data),
    ):
        add(name, path.exists(), str(path) if path.exists() else f"missing: {path}")
    versions = _python_versions(paths.python)
    add("runtime", "error" not in versions, json.dumps(versions, sort_keys=True))
    for package in ("python", "torch", "vllm", "ray", "transformer_engine"):
        expected_version = requirements.get(package)
        if expected_version is None:
            continue
        actual_version = versions.get(package, "missing")
        add(
            f"{package}_version",
            _version_matches(actual_version, str(expected_version)),
            f"found {actual_version}, expected {expected_version}",
        )
    try:
        ray_status = subprocess.run(
            [str(paths.ray), "job", "list", f"--address={ray_address}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        ray_detail = (
            f"{ray_address} reachable"
            if ray_status.returncode == 0
            else (ray_status.stderr.strip() or ray_status.stdout.strip())[-1000:]
        )
        add("ray_jobs_api", ray_status.returncode == 0, ray_detail or ray_address)
    except (OSError, subprocess.TimeoutExpired) as exc:
        add("ray_jobs_api", False, f"{ray_address}: {exc}")

    payload = {
        "profile": profile.get("name"),
        "backend": backend,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Profile: {payload['profile']}")
        for item in checks:
            marker = "PASS" if item["passed"] else "FAIL"
            print(f"[{marker}] {item['name']}: {item['detail']}")
        print("Doctor: PASS" if payload["passed"] else "Doctor: FAIL")
    return 0 if payload["passed"] else 1


def _arm_config(profile: dict[str, Any], arm: str) -> dict[str, Any]:
    arm = canonical_arm(arm)
    arms = profile.get("modes", {})
    config = arms.get(arm)
    if not isinstance(config, dict):
        raise ReproError(f"unsupported mode {arm!r}; choose one of {', '.join(sorted(arms))}")
    return config


def _validate_topology_args(args: argparse.Namespace) -> None:
    tp_size = int(args.tp_size)
    if args.cp_size is None:
        args.cp_size = 8 // tp_size if tp_size > 0 else 0
    cp_size = int(args.cp_size)
    rollout_tp_size = int(args.rollout_tp_size)
    rollout_cp_size = int(args.rollout_cp_size)
    if min(tp_size, cp_size, rollout_tp_size, rollout_cp_size) <= 0:
        raise ReproError("TP/CP sizes must be positive")
    if tp_size * cp_size != 8:
        raise ReproError("colocated Qwen3-8B training requires --tp-size * --cp-size = 8")
    if any(size % tp_size for size in (32, 8, 152064)):
        raise ReproError("--tp-size must divide Qwen3-8B heads, query groups, and vocabulary")
    if 8 % (rollout_tp_size * rollout_cp_size):
        raise ReproError("--rollout-tp-size * --rollout-cp-size must divide 8 GPUs")
    # vLLM exposes prefill context parallelism separately from decode TP.
    # The VIME companion adapter forwards this as
    # ParallelConfig.prefill_context_parallel_size.


def _example_root(rl_kernel_root: Path | None = None) -> Path:
    root = (rl_kernel_root or _repo_root()) / EXAMPLE_RELATIVE
    if not root.is_dir():
        raise ReproError(f"reproduction example is missing: {root}")
    return root


def _example_root_for_run(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root_value = manifest.get("paths", {}).get("rl_kernel_root")
            if root_value:
                return _example_root(Path(root_value))
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return _example_root()


def _runner_command(paths: Paths, profile: dict[str, Any], args: argparse.Namespace) -> list[str]:
    arm = canonical_arm(args.arm)
    for name in ("lr", "weight_decay", "kl_coef"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ReproError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.max_response_len is not None and args.max_response_len <= 0:
        raise ReproError("--max-response-len must be positive")
    if args.max_tokens_per_gpu is not None and args.max_tokens_per_gpu <= 0:
        raise ReproError("--max-tokens-per-gpu must be positive")
    if (
        args.vllm_gpu_memory_utilization is not None
        and not 0 < args.vllm_gpu_memory_utilization < 1
    ):
        raise ReproError("--vllm-gpu-memory-utilization must be in (0, 1)")
    if getattr(args, "backend", "cuda") == "rocm":
        _validate_topology_args(args)
        # ROCm and CUDA use the same VIME prefill-context-parallel contract.
        # The rollout engine count is derived from rollout TP * rollout CP.
        if not math.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0:
            raise ReproError("--temperature must be finite and positive")
        if not 0 < args.rollout_top_p <= 1:
            raise ReproError("--top-p must be in (0, 1]")
        if args.rollouts <= 0:
            raise ReproError("--rollouts must be positive")
        run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{arm}-%Y%m%dT%H%M%S%fZ")
        if Path(run_id).name != run_id or run_id in (".", ".."):
            raise ReproError("--run-id must be a single directory name")
        command = [
            str(paths.python),
            "-m",
            "examples.vime_rocm_attention_ablation.run_qwen3_8b",
            "--mode",
            arm,
            "--run-dir",
            str(paths.output_root / run_id),
        ]
        for flag, value in {
            "rollouts": args.rollouts,
            "seed": args.seed,
            "rollout-seed": args.rollout_seed,
            "tp-size": args.tp_size,
            "cp-size": args.cp_size,
            "rollout-tp-size": args.rollout_tp_size,
            "rollout-cp-size": args.rollout_cp_size,
            "rollout-temperature": args.rollout_temperature,
            "rollout-top-p": args.rollout_top_p,
            "rollout-top-k": args.rollout_top_k,
            "grpo-std-normalization": args.grpo_std_normalization,
            "lr": args.lr,
            "weight-decay": args.weight_decay,
            "kl-coef": args.kl_coef,
            "rl-kernel-root": paths.rl_kernel_root,
            "vime-root": paths.vime_root,
            "megatron-root": paths.megatron_root,
            "model-root": paths.model_root,
            "reference-checkpoint": paths.ref_load,
            "prompt-data": paths.prompt_data,
        }.items():
            command.extend([f"--{flag}", str(value)])
        if args.max_response_len is not None:
            command.extend(["--max-response-length", str(args.max_response_len)])
        if args.command == "run" and not args.wait:
            raise ReproError("ROCm run currently waits for validation; --detach is not supported")
        if args.allow_dirty:
            raise ReproError("ROCm requires frozen source checks; --allow-dirty is not supported")
        if "--ray-address" in getattr(args, "explicit_flags", set()):
            raise ReproError(
                "ROCm manages its Ray instance via --ray-port and --ray-dashboard-port"
            )
        if args.command == "verify" or args.require_updates:
            raise ReproError(
                "ROCm does not yet implement the weight-update verify contract; use run "
                "for train/rollout logprob validation"
            )
        if args.rollout_top_k != -1:
            raise ReproError("strict ROCm top-k replay is not supported; use --top-k -1")
        if args.vllm_gpu_memory_utilization is not None:
            command.extend(["--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization)])
        for name in (
            "ray_port",
            "ray_dashboard_port",
            "samples_per_prompt",
            "global_batch_size",
            "rollout_batch_size",
            "max_tokens_per_gpu",
        ):
            value = getattr(args, name, None)
            if value is not None:
                command.extend([f"--{name.replace('_', '-')}", str(value)])
        return command
    if args.grpo_std_normalization != "enabled":
        raise ReproError("--grpo-std-normalization disabled currently requires --backend rocm")
    rocm_only = [
        name
        for name in (
            "ray_port",
            "ray_dashboard_port",
            "samples_per_prompt",
            "global_batch_size",
            "rollout_batch_size",
        )
        if getattr(args, name, None) is not None
    ]
    if rocm_only:
        raise ReproError("these workload options require --backend rocm: " + ", ".join(rocm_only))
    _arm_config(profile, arm)
    _validate_topology_args(args)
    # The launcher and runner are one interface and must come from the same
    # checkout.  The target RL-Kernel checkout may intentionally be pinned to
    # an older runtime revision whose runner predates the current CLI modes.
    example_root = _example_root()
    command = [
        str(paths.python),
        str(example_root / "run_arm.py"),
        "--group",
        arm,
        "--num-rollout",
        str(args.rollouts),
        "--seed",
        str(args.seed),
        "--rollout-seed",
        str(args.rollout_seed),
        "--tp-size",
        str(args.tp_size),
        "--cp-size",
        str(args.cp_size),
        "--rollout-tp-size",
        str(args.rollout_tp_size),
        "--rollout-cp-size",
        str(args.rollout_cp_size),
        "--output-root",
        str(paths.output_root),
        "--rl-kernel-root",
        str(paths.rl_kernel_root),
        "--vime-root",
        str(paths.vime_root),
        "--megatron-root",
        str(paths.megatron_root),
        "--model-root",
        str(paths.model_root),
        "--ref-load",
        str(paths.ref_load),
        "--prompt-data",
        str(paths.prompt_data),
        "--python",
        str(paths.python),
        "--ray-bin",
        str(paths.ray),
        "--ld-library-path",
        f"{paths.cuda_runtime_root / 'lib'}:{paths.te_root / 'transformer_engine/wheel_lib'}",
    ]
    for extra_pythonpath in dict.fromkeys(
        (paths.runtime_site, paths.cuda_python_site, paths.te_root)
    ):
        command.extend(["--extra-pythonpath", str(extra_pythonpath)])
    for item in profile.get("runner_args", []):
        command.extend([str(part) for part in item])
    max_tokens_per_gpu = args.max_tokens_per_gpu
    if max_tokens_per_gpu is None:
        max_tokens_per_gpu = 1024 // int(args.cp_size)
    if max_tokens_per_gpu <= 0:
        raise ReproError("--max-tokens-per-gpu must be positive")
    command.extend(["--max-tokens-per-gpu", str(max_tokens_per_gpu)])
    vllm_gpu_memory_utilization = (
        float(args.vllm_gpu_memory_utilization)
        if args.vllm_gpu_memory_utilization is not None
        else (0.2 if int(args.tp_size) == 1 and int(args.rollout_tp_size) != 1 else 0.4)
    )
    command.extend(
        [
            "--vllm-gpu-memory-utilization",
            str(vllm_gpu_memory_utilization),
            "--rollout-temperature",
            str(args.rollout_temperature),
            "--rollout-top-p",
            str(args.rollout_top_p),
        ]
    )
    command.extend(
        [
            "--rollout-top-k",
            str(args.rollout_top_k),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
        ]
    )
    if not math.isfinite(args.kl_coef) or args.kl_coef < 0:
        raise ReproError("--kl-coef must be finite and nonnegative")
    if args.kl_coef > 0:
        command.extend(["--use-kl-loss", "--kl-loss-coef", str(args.kl_coef)])
    if args.max_response_len is not None:
        command.extend(["--max-response-len", str(args.max_response_len)])
    if args.require_updates:
        command.append("--require-updates")
    if args.ray_address:
        command.extend(["--ray-address", args.ray_address])
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.wait:
        command.append("--wait")
    if args.allow_dirty:
        command.append("--allow-dirty")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _print_plan(paths: Paths, profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = _runner_command(paths, profile, args)
    plan = {
        "schema_version": "rlkernel.repro.plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.get("name"),
        "mode": canonical_arm(args.arm),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "runner_command": command,
    }
    print(json.dumps(plan, indent=2))
    return plan


def _prepare_repositories(paths: Paths, profile: dict[str, Any], arm: str) -> None:
    repos = profile.get("repositories", {})
    mode = canonical_arm(arm)
    for key, target in (
        ("rl_kernel", paths.rl_kernel_root),
        ("vime", paths.vime_root),
        ("megatron", paths.megatron_root),
    ):
        config = repos.get(key, {})
        url = config.get("url")
        if not target.exists():
            if not url:
                raise ReproError(f"cannot clone {key}: profile has no repository URL")
            target.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", str(url), str(target)])
        if not (target / ".git").exists():
            raise ReproError(f"{key} is not a git checkout: {target}")
        if not _git_clean(target):
            raise ReproError(
                f"{key} has local changes; commit or clean it before prepare: {target}"
            )
        revision = config.get("revisions", {}).get(mode)
        if revision:
            _run(["git", "-C", str(target), "fetch", "origin"])
            _run(["git", "-C", str(target), "checkout", "--detach", str(revision)])


def _convert_checkpoint(paths: Paths) -> None:
    if paths.ref_load.exists():
        print(f"Reference checkpoint already exists: {paths.ref_load}")
        return
    model_script = paths.vime_root / "scripts/models/qwen3-8B.sh"
    converter = paths.vime_root / "tools/convert_hf_to_torch_dist.py"
    if not model_script.is_file():
        raise ReproError(f"VIME model argument script is missing: {model_script}")
    if not converter.is_file():
        raise ReproError(f"checkpoint converter is missing: {converter}")
    if not paths.model_root.is_dir():
        raise ReproError(f"HF model root is missing: {paths.model_root}")
    paths.ref_load.parent.mkdir(parents=True, exist_ok=True)
    bash = shutil.which("bash")
    if not bash:
        raise ReproError("bash is required for checkpoint conversion on the training host")
    command = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(model_script))}; "
        f"PYTHONPATH={shlex.quote(str(paths.megatron_root))} "
        f"{shlex.quote(str(paths.python))} {shlex.quote(str(converter))} "
        '"${MODEL_ARGS[@]}" '
        f"--hf-checkpoint {shlex.quote(str(paths.model_root))} "
        f"--save {shlex.quote(str(paths.ref_load))}"
    )
    _run([bash, "-lc", command])


def prepare(paths: Paths, profile: dict[str, Any], args: argparse.Namespace) -> int:
    paths.data_root.mkdir(parents=True, exist_ok=True)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    _prepare_repositories(paths, profile, canonical_arm(args.arm))
    if args.download_data:
        source = paths.data_root / "datasets/dapo-math-17k.parquet"
        paths.prompt_data.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(paths.python),
                str(_example_root() / "prepare_dapo_data.py"),
                "--download",
                "--source",
                str(source),
                "--output",
                str(paths.prompt_data),
            ]
        )
    if args.convert_checkpoint:
        _convert_checkpoint(paths)
    print(f"Prepared workspace: {paths.workspace}")
    print("Next: start Ray, then run:")
    print(f"  rlk-repro doctor --workspace {paths.workspace}")
    return 0


def _add_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="profile JSON path")
    parser.add_argument("--workspace", default=None, help="reproduction workspace root")
    for name, label in (
        ("runtime-root", "runtime environment root"),
        ("cuda-runtime-root", "CUDA runtime package root"),
        ("runtime-site", "additional runtime site-packages"),
        ("cuda-python-site", "CUDA Python site-packages"),
        ("te-root", "Transformer Engine site-packages root"),
        ("data-root", "data root"),
        ("model-root", "HF model root"),
        ("ref-load", "Megatron torch-dist reference checkpoint"),
        ("prompt-data", "prepared prompt JSONL"),
        ("output-root", "append-only run output root"),
        ("python", "experiment Python executable"),
        ("ray", "Ray executable"),
        ("rl-kernel-root", "RL-Kernel checkout"),
        ("vime-root", "VIME checkout"),
        ("megatron-root", "Megatron-LM checkout"),
    ):
        parser.add_argument(f"--{name}", default=None, help=label)


def _resolved_paths(profile: dict[str, Any], args: argparse.Namespace) -> Paths:
    values = vars(args)
    return resolve_paths(
        profile,
        workspace=values.get("workspace"),
        arm=canonical_arm(str(values.get("arm", "native"))),
        **{
            key.replace("-", "_"): values.get(key.replace("-", "_"))
            for key in (
                "rl-kernel-root",
                "vime-root",
                "megatron-root",
                "runtime-root",
                "cuda-runtime-root",
                "runtime-site",
                "cuda-python-site",
                "te-root",
                "data-root",
                "model-root",
                "ref-load",
                "prompt-data",
                "output-root",
                "python",
                "ray",
            )
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlk-repro", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check host, runtime, assets, and topology"
    )
    _add_path_options(doctor_parser)
    doctor_parser.add_argument("--backend", choices=("cuda", "rocm"), default=None)
    doctor_parser.add_argument(
        "--mode",
        default="native",
        dest="arm",
        metavar="MODE",
        help="native (no rollout-logprob reuse) or consistency (RL-Kernel operators)",
    )
    doctor_parser.add_argument(
        "--arm", dest="arm", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    doctor_parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_API_SERVER_ADDRESS", "http://127.0.0.1:8265"),
        help="Ray Jobs API address",
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="clone/pin sources and optionally prepare data/checkpoint",
        description=(
            "Prepare source, data, and checkpoint assets. This command does not "
            "create the Python environment or download model weights."
        ),
    )
    _add_path_options(prepare_parser)
    prepare_parser.add_argument(
        "--mode",
        default="native",
        dest="arm",
        metavar="MODE",
        help="native (no rollout-logprob reuse) or consistency (RL-Kernel operators)",
    )
    prepare_parser.add_argument(
        "--arm", dest="arm", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    prepare_parser.add_argument(
        "--download-data", action="store_true", help="download and convert DAPO-Math-17k"
    )
    prepare_parser.add_argument(
        "--convert-checkpoint",
        action="store_true",
        help="convert the configured local Qwen3-8B model to Megatron torch-dist",
    )

    for command in ("plan", "run", "verify", "debug"):
        command_parser = subparsers.add_parser(command, help=f"{command} one reproduction mode")
        _add_path_options(command_parser)
        command_parser.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
        command_parser.add_argument(
            "--mode",
            default="consistency",
            dest="arm",
            metavar="MODE",
            help="native (no rollout-logprob reuse) or consistency (RL-Kernel operators)",
        )
        command_parser.add_argument(
            "--arm", dest="arm", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
        command_parser.add_argument(
            "--rollouts",
            "--steps",
            type=int,
            default=2 if command == "verify" else 200,
            help="training/rollout steps (use 8 for smoke)",
        )
        command_parser.add_argument("--seed", type=int, default=1234)
        command_parser.add_argument("--rollout-seed", type=int, default=1234)
        command_parser.add_argument("--tp-size", "--tp", type=int, default=4)
        command_parser.add_argument(
            "--cp-size", "--cp", type=int, default=None, help="training CP; defaults to 8 / TP"
        )
        command_parser.add_argument("--rollout-tp-size", "--rollout-tp", type=int, default=4)
        command_parser.add_argument("--rollout-cp-size", "--rollout-cp", type=int, default=1)
        command_parser.add_argument(
            "--rollout-temperature", "--temperature", type=float, default=1.0
        )
        command_parser.add_argument("--rollout-top-p", "--top-p", type=float, default=1.0)
        for name in (
            "ray-port",
            "ray-dashboard-port",
            "samples-per-prompt",
            "global-batch-size",
            "rollout-batch-size",
        ):
            command_parser.add_argument(
                f"--{name}", type=int, default=None, help="ROCm workload option"
            )
        command_parser.add_argument(
            "--vllm-gpu-memory-utilization",
            type=float,
            default=None,
            help=("vLLM memory fraction; defaults to 0.2 for training TP1 and 0.4 otherwise"),
        )
        command_parser.add_argument("--rollout-top-k", "--top-k", type=int, default=-1)
        command_parser.add_argument(
            "--max-tokens-per-gpu",
            type=int,
            default=None,
            help=(
                "training microbatch token budget per CP rank; defaults to 1024 / CP, "
                "preserving the logical microbatch budget"
            ),
        )
        command_parser.add_argument(
            "--grpo-std-normalization",
            choices=("enabled", "disabled"),
            default="enabled",
            help="ROCm GRPO advantage normalization",
        )
        command_parser.add_argument("--lr", type=float, default=5e-7)
        command_parser.add_argument("--weight-decay", type=float, default=0.1)
        command_parser.add_argument(
            "--kl-coef", type=float, default=0.01 if command == "verify" else 0.0
        )
        command_parser.add_argument(
            "--max-response-len",
            "--max-response-length",
            type=int,
            default=512 if command == "verify" else None,
        )
        command_parser.add_argument(
            "--require-updates", action="store_true", default=command == "verify"
        )
        command_parser.add_argument("--run-id", default=None)
        command_parser.add_argument(
            "--wait",
            default=(command in {"run", "verify"}),
            action="store_true",
            help="stream until completion and save run.log plus ray-status.txt",
        )
        command_parser.add_argument("--detach", dest="wait", action="store_false")
        command_parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="allow non-publishable development runs from dirty repositories",
        )
        command_parser.add_argument(
            "--dry-run", action="store_true", help="write a manifest without submitting to Ray"
        )
        command_parser.add_argument(
            "--ray-address",
            default=os.environ.get("RAY_API_SERVER_ADDRESS", "http://127.0.0.1:8265"),
            help="Ray Jobs API address",
        )
        if command == "debug":
            command_parser.add_argument(
                "source", type=Path, help="Saved run, rollout .pt, or frozen replay JSON"
            )
            command_parser.add_argument(
                "--sample", type=int, default=0, help="Sample to replay (default: 0)"
            )
            command_parser.add_argument("--step", type=int, default=0, help="Saved rollout step")
            command_parser.add_argument(
                "--matrix",
                default="auto",
                help="auto, attribute, full, or comma-separated Mxxx groups",
            )
            command_parser.add_argument(
                "--baseline",
                help="Original Mxxx configuration; inferred from source or native M000",
            )
            command_parser.add_argument(
                "--report-only",
                action="store_true",
                help="Analyze existing snapshots without GPU execution",
            )

            command_parser.add_argument(
                "--resume",
                type=Path,
                default=None,
                help="Continue an incomplete diagnostic directory, revalidating completed arms",
            )
            command_parser.add_argument("--json", action="store_true", dest="as_json")

    validate_parser = subparsers.add_parser("validate", help="validate one completed run")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.add_argument("--seal", action="store_true")
    validate_parser.add_argument("--profile", default=None)

    report_parser = subparsers.add_parser("report", help="collect and plot validated runs")
    _add_path_options(report_parser)
    report_parser.add_argument("--phase", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "debug" and argv[1] == "doctor":
        parser.error("'debug doctor' was removed; use 'doctor' instead")
    args = parser.parse_args(argv)
    try:
        if args.command == "debug" and (
            args.report_only or (args.source / "capture-manifest.json").is_file()
        ):
            from rl_engine.alignment.debug.entry import main as diagnostic_main

            return diagnostic_main([str(args.source)])
        profile, _profile_path_value = _load_profile(getattr(args, "profile", None))
        if args.command in ("plan", "run", "verify", "debug"):
            defaults = profile.get("defaults", {})
            allowed = {
                "backend",
                "mode",
                "rollouts",
                "tp",
                "cp",
                "rollout-tp",
                "rollout-cp",
                "temperature",
                "top-p",
                "top-k",
                "grpo-std-normalization",
                "lr",
                "weight-decay",
                "kl-coef",
                "steps",
                "max-response-len",
                "seed",
                "rollout-seed",
                "ray-port",
                "ray-dashboard-port",
                "samples-per-prompt",
                "global-batch-size",
                "rollout-batch-size",
                "max-response-length",
                "max-tokens-per-gpu",
                "vllm-gpu-memory-utilization",
            }
            if not isinstance(defaults, dict) or set(defaults) - allowed:
                raise ReproError("profile defaults contains unsupported run options")
            defaults = dict(defaults)
            defaults.setdefault("backend", profile.get("requirements", {}).get("backend", "cuda"))
            explicit = {item.split("=", 1)[0] for item in argv[1:] if item.startswith("--")}
            # Changing TP alone must infer CP from that TP, not retain a stale
            # machine-profile CP. An explicit --cp always takes precedence.
            if explicit & {"--tp", "--tp-size"} and not explicit & {"--cp", "--cp-size"}:
                defaults.pop("cp", None)
            if args.command in {"verify", "debug"}:
                for key in ("rollouts", "steps", "max-response-length", "max-response-len"):
                    defaults.pop(key, None)
            flags = [part for key, value in defaults.items() for part in (f"--{key}", str(value))]
            args = parser.parse_args([argv[0], *flags, *argv[1:]])
            args.explicit_flags = explicit
        elif (
            profile.get("requirements", {}).get("backend") == "rocm"
            and args.command not in {"doctor"}
        ):
            raise ReproError(
                "This ROCm profile supports plan and run (including automatic validation); "
                f"the {args.command} command is currently CUDA-only"
            )
        if args.command == "validate":
            run_dir = args.run_dir.expanduser().resolve()
            command = [
                sys.executable,
                str(_example_root_for_run(run_dir) / "validate_run.py"),
                "--run-dir",
                str(run_dir),
            ]
            if args.seal:
                command.append("--seal")
            return _run(command).returncode

        paths = _resolved_paths(profile, args)
        if args.command == "debug":
            from rl_engine.alignment.debug.adapters import run_debug

            return run_debug(paths, profile, args)
        if args.command == "doctor":
            backend = args.backend or profile.get("requirements", {}).get("backend", "cuda")
            return doctor(
                paths,
                profile,
                ray_address=args.ray_address,
                as_json=args.as_json,
                backend=backend,
            )
        if args.command == "prepare":
            return prepare(paths, profile, args)
        if args.command == "plan":
            _print_plan(paths, profile, args)
            return 0
        if args.command in {"run", "verify"}:
            if args.dry_run:
                _print_plan(paths, profile, args)
                return 0
            if args.backend == "rocm":
                if args.dry_run:
                    _print_plan(paths, profile, args)
                    return 0
                command = _runner_command(paths, profile, args)
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(
                    filter(
                        None,
                        (
                            str(paths.rl_kernel_root),
                            str(paths.vime_root),
                            str(paths.megatron_root),
                            _profile_path(profile, "vllm_root"),
                            env.get("PYTHONPATH", ""),
                        ),
                    )
                )
                # The ROCm launcher checks paths, plugin installation, HIP devices,
                # and Ray ownership before starting; CUDA doctor is not applicable.
                return _run(command, env=env).returncode
            command = _runner_command(paths, profile, args)
            preflight = doctor(paths, profile, ray_address=args.ray_address)
            if preflight:
                print("Run aborted: fix the failed doctor checks first.", file=sys.stderr)
                return preflight
            if not args.dry_run:
                _require_idle_gpus()
            print(
                f"Starting {canonical_arm(args.arm)}: training TP{args.tp_size}/CP{args.cp_size}, "
                f"rollout TP{args.rollout_tp_size}, {args.rollouts} steps; "
                f"temperature={args.rollout_temperature}, top-p={args.rollout_top_p}, "
                f"top-k={args.rollout_top_k}",
                flush=True,
            )
            env = os.environ.copy()
            env["CUDNN_FRONTEND_CUDART_LIB_NAME"] = str(
                paths.cuda_runtime_root / "lib/libcudart.so.12"
            )
            if args.run_id is None:
                args.run_id = (
                    f"{canonical_arm(args.arm)}-tp{args.tp_size}cp{args.cp_size}-"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                )
                command = _runner_command(paths, profile, args)
            result = _run(command, env=env)
            if result.returncode or not args.wait or args.dry_run:
                return result.returncode
            return _run(
                [
                    str(paths.python),
                    str(_example_root() / "validate_run.py"),
                    "--run-dir",
                    str(paths.output_root / args.run_id),
                ],
                env=env,
            ).returncode
        if args.command == "report":
            results_root = paths.data_root / "results"
            _run(
                [
                    str(paths.python),
                    str(_example_root() / "collect_results.py"),
                    "--runs-root",
                    str(paths.data_root / "runs"),
                    "--output-dir",
                    str(results_root),
                ]
            )
            plot_command = [
                str(paths.python),
                str(_example_root() / "plot_results.py"),
                "--rounds-csv",
                str(results_root / "rounds.csv"),
                "--output-dir",
                str(results_root / "figures"),
            ]
            if args.phase:
                plot_command.extend(["--phase", args.phase])
            _run(plot_command)
            print(f"Report written to {results_root}")
            return 0
    except (OSError, ReproError, subprocess.CalledProcessError) as exc:
        print(f"rlk-repro: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
