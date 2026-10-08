# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Launcher preparation and pre-engine worker verification for BI plans."""

from __future__ import annotations

import hashlib
import json
import os
import platform as python_platform
import subprocess
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, MutableMapping
from urllib.parse import unquote, urlparse

from .adapters import get_adapter
from .builtin import builtin_catalog
from .catalog import RuntimeContext, canonical, fingerprint

PLAN_ENV = "RL_KERNEL_BI_PLAN"
_ACTIVE: dict[str, Any] | None = None
_LOCK = RLock()
_PACKAGES = (
    "torch",
    "vllm",
    "transformer-engine",
    "triton",
    "aiter",
    "transformers",
    "ray",
    "flash-attn",
    "flash-attn-4",
    "megatron-core",
    "nvidia-cublas-cu12",
    "nvidia-cublas-cu13",
)


def _repository_revision(name: str, root: str) -> str:
    head = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", root, "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"BI requires a clean {name} checkout: {root}")
    return head


def _package_identity(name: str) -> dict[str, Any] | None:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    result = {
        "version": distribution.version,
        "build_record": hashlib.sha256(
            (distribution.read_text("RECORD") or "").encode()
        ).hexdigest(),
    }
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        url = urlparse(direct_url.get("url", ""))
        if url.scheme != "file":
            raise RuntimeError(f"BI cannot identify editable source for {name}")
        result["source_revision"] = _repository_revision(name, unquote(url.path))
    elif direct_url.get("vcs_info"):
        result["source_revision"] = direct_url["vcs_info"].get("commit_id")
    return result


def enabled(environment: Mapping[str, str] | None = None) -> bool:
    environment = os.environ if environment is None else environment
    raw = environment.get("RL_KERNEL_BI", "0").strip().lower()
    if raw not in {"0", "1", "false", "true", "off", "on"}:
        raise ValueError("RL_KERNEL_BI must be 0 or 1 (also accepts true/false, on/off)")
    return raw in {"1", "true", "on"}


def runtime_identity(repositories: Mapping[str, str]) -> dict[str, Any]:
    import torch

    packages = {name: _package_identity(name) for name in _PACKAGES}
    revisions = {
        name: _repository_revision(name, root) for name, root in sorted(repositories.items())
    }
    # Include numerical implementation content, not catalog/benchmark records:
    # hashing the latter would make benchmark context IDs self-referential.
    root = Path(__file__).resolve().parents[1]
    sources = hashlib.sha256()
    source_paths = [path for path in root.rglob("*") if path != root / "bi" / "builtin.py"]
    for directory in ("vime_qwen3_8b_tp4_cp2_200", "vime_rocm_attention_ablation"):
        source_paths.extend((root.parent / "examples" / directory).glob("*.py"))
        source_paths.extend((root.parent / "examples" / directory).glob("*.sh"))
    for path in sorted(source_paths):
        if path.is_file() and path.suffix in {".py", ".sh", ".cpp", ".cu", ".cuh", ".h", ".so"}:
            sources.update(str(path.relative_to(root.parent)).encode())
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    sources.update(chunk)
    driver_paths = (Path("/proc/driver/nvidia/version"), Path("/sys/module/amdgpu/version"))
    return {
        "python": python_platform.python_version(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "host_kernel": python_platform.release(),
        "drivers": {str(path): path.read_text() for path in driver_paths if path.is_file()},
        "repositories": revisions,
        "implementation_sha256": sources.hexdigest(),
    }


def hardware_identity() -> tuple[str, tuple[str, ...]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RL_KERNEL_BI requires a supported CUDA or ROCm GPU runtime")
    platform = "rocm" if torch.version.hip is not None else "cuda"
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", None) or f"sm{props.major}{props.minor}"
        devices.append(f"{props.name}:{arch}:{props.total_memory}")
    return platform, tuple(devices)


def _route_environment(environment: Mapping[str, str]) -> dict[str, str]:
    # Preserve routing/graph/numerical settings but omit per-run output paths.
    return {
        key: value
        for key, value in environment.items()
        if key.startswith(("RL_KERNEL_", "VLLM_", "NVTE_", "CUBLAS", "NCCL_"))
        and key not in {"RL_KERNEL_BI", PLAN_ENV, "RL_KERNEL_RUN_ID"}
        and not key.endswith(("_DIR", "_ROOT", "_PYTHON"))
    }


def prepare_environment(
    environment: MutableMapping[str, str],
    *,
    model_root: Path,
    framework: str,
    platform: str,
    topology: tuple[int, int, int, int, int],
    workload: Mapping[str, Any],
    repositories: Mapping[str, str],
    caller_environment: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Resolve once in the launcher and export an immutable worker envelope.

    Called after framework launcher defaults, before worker submission. Model
    and hardware support are checked by the selected, shipped plan adapter.
    """
    caller = os.environ if caller_environment is None else caller_environment
    if not enabled(caller):
        return None
    catalog = builtin_catalog()
    model_path = model_root.resolve() / "config.json"
    config = json.loads(model_path.read_text(encoding="utf-8"))
    model_id = catalog.identify(config)
    actual_platform, hardware = hardware_identity()
    if actual_platform != platform or len(hardware) != topology[0]:
        raise ValueError("BI launcher platform/device count disagrees with the local GPU runtime")
    for source in (caller, environment):
        if source.get("RL_KERNEL_MODE", "strict") != "strict":
            raise ValueError("RL_KERNEL_BI=1 requires RL_KERNEL_MODE=strict")
        for module in ("ATTENTION", "FFN", "LOGP"):
            if source.get(f"RL_KERNEL_{module}_CASE", "R/R") != "R/R":
                raise ValueError("RL_KERNEL_BI=1 conflicts with a native/mixed operator arm")
    context = RuntimeContext(
        model_id=model_id,
        framework=framework,
        platform=platform,
        hardware=hardware,
        topology=topology,
        model_config=canonical(config),
        runtime=canonical(runtime_identity(repositories)),
        workload=canonical(dict(workload)),
        route_environment=canonical(_route_environment(environment)),
    )
    plan, reason = catalog.select(context)
    get_adapter(plan.adapter).validate(context)
    selected_env = dict(environment)
    selected_env.update(plan.environment)
    # Strictness is a contract, even if a future catalog entry is malformed.
    if any(
        selected_env.get(f"RL_KERNEL_{key}_CASE") != "R/R" for key in ("ATTENTION", "FFN", "LOGP")
    ):
        raise ValueError("BI execution plans must preserve all R/R routes")
    if selected_env.get("RL_KERNEL_MODE") != "strict":
        raise ValueError("BI execution plans must preserve strict mode")
    if selected_env.get("RL_KERNEL_VLLM_INTEGRATION") != "1":
        raise ValueError("BI execution plans must preserve the rollout integration")
    record = {
        "schema": "rlkernel.bi.plan.v1",
        "plan_id": plan.plan_id,
        "adapter": plan.adapter,
        "plan_digest": plan.digest,
        "context": asdict(context),
        "context_digest": context.digest,
        "reason": reason,
        "model_config_path": str(model_path),
        "repositories": dict(repositories),
        "environment": _route_environment(selected_env),
    }
    record["digest"] = fingerprint(record)
    environment.update(plan.environment)
    environment["RL_KERNEL_BI"] = "1"
    environment[PLAN_ENV] = canonical(record)
    print(f"[RL-Kernel BI] {plan.plan_id}: {reason}; context={context.digest}", flush=True)
    return record


def prepare_vime_environment(
    environment: MutableMapping[str, str], **kwargs: Any
) -> dict[str, Any] | None:
    return prepare_environment(environment, framework="vime", **kwargs)


def verify_worker_plan(*, check_hardware: bool = False) -> dict[str, Any] | None:
    """Fail before patch installation/graph capture if the launch contract changed."""
    global _ACTIVE
    if not enabled():
        if _ACTIVE is not None or os.getenv(PLAN_ENV):
            raise RuntimeError("BI cannot be disabled after plan preparation")
        return None
    raw = os.getenv(PLAN_ENV)
    if not raw:
        raise RuntimeError(
            "RL_KERNEL_BI=1 needs a prepared plan; launch with rlk run / the Vime runners"
        )
    record = json.loads(raw)
    with _LOCK:
        if _ACTIVE is not None:
            if canonical(_ACTIVE) != canonical(record):
                raise RuntimeError("BI plan changed after worker initialization; restart the job")
            _verify_environment(record)
            if check_hardware:
                _verify_hardware(record)
            return record
        unsigned = {key: value for key, value in record.items() if key != "digest"}
        if record.get("schema") != "rlkernel.bi.plan.v1" or fingerprint(unsigned) != record.get(
            "digest"
        ):
            raise RuntimeError("invalid BI plan envelope")
        context_data = dict(record["context"])
        context_data["hardware"] = tuple(context_data["hardware"])
        context_data["topology"] = tuple(context_data["topology"])
        context = RuntimeContext(**context_data)
        catalog = builtin_catalog()
        plan, _ = catalog.select(context)
        if record.get("adapter") != plan.adapter:
            raise RuntimeError("worker adapter differs from the selected BI plan")
        get_adapter(plan.adapter).validate(context)
        if (plan.plan_id, plan.digest, context.digest) != (
            record["plan_id"],
            record["plan_digest"],
            record["context_digest"],
        ):
            raise RuntimeError("worker catalog does not match the launcher's selected BI plan")
        config = json.loads(Path(record["model_config_path"]).read_text(encoding="utf-8"))
        if (
            canonical(config) != context.model_config
            or catalog.identify(config) != context.model_id
        ):
            raise RuntimeError("model config changed after BI plan preparation")
        expected_environment = json.loads(context.route_environment)
        expected_environment.update(plan.environment)
        if expected_environment != record["environment"]:
            raise RuntimeError("BI environment does not implement the selected catalog plan")
        if canonical(runtime_identity(record["repositories"])) != context.runtime:
            raise RuntimeError(
                "worker dependency/source versions differ from the BI launch context"
            )
        if check_hardware:
            _verify_hardware(record)
        _verify_environment(record)
        _ACTIVE = record
        return record


def _verify_hardware(record: Mapping[str, Any]) -> None:
    platform, hardware = hardware_identity()
    context = record["context"]
    # Ray can expose a subset of devices. Frontend plugin discovery must not
    # initialize CUDA; check hardware only when an actual model is constructed.
    if (
        platform != context["platform"]
        or not hardware
        or not set(hardware) <= set(context["hardware"])
    ):
        raise RuntimeError("worker hardware differs from the BI launch context")


def _verify_environment(record: Mapping[str, Any]) -> None:
    for key, value in record["environment"].items():
        if os.getenv(key) != value:
            raise RuntimeError(f"BI plan environment changed: {key}")


def active_plan_readback() -> dict[str, Any] | None:
    if _ACTIVE is None:
        return None
    return {
        key: _ACTIVE[key]
        for key in ("plan_id", "plan_digest", "context_digest", "reason", "digest")
    }
