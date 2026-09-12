# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run and archive the Vime Qwen3-8B TP=4/CP=2 validation entry point.

This is an integration example, not a synthetic pass generator.  A dry run
only records the exact launch contract.  ``--run`` executes Vime and records
whether the strict RL-Kernel provider was actually observed in the log.  The
report deliberately leaves attention/FFN unclaimed until both framework
readbacks are supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG = Path(__file__).with_name("qwen3_8b_tp4_cp2.json")
PROVIDER_MARKER = "linear_logp provider active"
FALLBACK_MARKERS = ("using native path", "fallback=True", "fallback=true")
RUNTIME_EVIDENCE_SCHEMA = "rlkernel.operator_runtime_evidence.v1"
_OPERATOR_METRICS = {
    "attention": (
        "out_max_abs",
        "lse_max_abs",
        "dq_max_abs",
        "dk_max_abs",
        "dv_max_abs",
    ),
    "ffn": ("out_max_abs", "dx_max_abs", "dw_max_abs"),
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("example config must contain a JSON object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    training = config.get("training")
    rollout = config.get("rollout")
    provider = config.get("linear_logp_provider")
    if (
        not isinstance(training, Mapping)
        or not isinstance(rollout, Mapping)
        or not isinstance(provider, Mapping)
    ):
        raise ValueError("training, rollout, and linear_logp_provider sections are required")
    expected = {
        "tensor_model_parallel_size": 4,
        "context_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "world_size": 8,
    }
    for name, value in expected.items():
        if training.get(name) != value:
            raise ValueError(f"training.{name} must be {value!r}")
    if rollout.get("top_p") != 1.0:
        raise ValueError("rollout.top_p must remain 1.0 for the strict provider contract")
    if provider.get("mode") != "strict":
        raise ValueError("linear_logp_provider.mode must be strict")
    if provider.get("path") != "rl_engine.integrations.vime.linear_logp_provider.provider":
        raise ValueError("example must use the RL-Kernel Vime provider")
    if provider.get("backend_id") != "rlkernel.linear_logp.bitwise.v1":
        raise ValueError("example must pin the deterministic vocab-parallel backend")


def load_runtime_evidence(path: Path | None) -> dict[str, Any] | None:
    """Load post-execution readback without treating configuration as evidence."""

    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != RUNTIME_EVIDENCE_SCHEMA:
        raise ValueError(f"runtime evidence must use schema {RUNTIME_EVIDENCE_SCHEMA!r}")
    return value


def _operator_evidence_status(evidence: Mapping[str, Any] | None, operator: str) -> str:
    if evidence is None:
        return "unclaimed"
    operators = evidence.get("operators")
    item = operators.get(operator) if isinstance(operators, Mapping) else None
    if not isinstance(item, Mapping):
        return "unclaimed"
    training = item.get("training")
    rollout = item.get("rollout")
    comparison = item.get("comparison")
    if not isinstance(training, Mapping) or not isinstance(rollout, Mapping):
        return "unclaimed"
    if not isinstance(comparison, Mapping) or comparison.get("passed") is not True:
        return "failed"
    required_identity = ("implementation_id", "backend_id", "contract_id")
    if any(not training.get(name) or not rollout.get(name) for name in required_identity):
        return "failed"
    if training["implementation_id"] != rollout["implementation_id"]:
        return "failed"
    for metric in _OPERATOR_METRICS[operator]:
        value = comparison.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != 0.0:
            return "failed"
    return "passed"


def validate_runtime_evidence(evidence: Mapping[str, Any] | None) -> None:
    """Reject malformed evidence before it can affect a report."""

    if evidence is None:
        return
    for operator in _OPERATOR_METRICS:
        status = _operator_evidence_status(evidence, operator)
        if status == "failed":
            raise ValueError(f"runtime evidence for {operator} is incomplete or non-zero")


def _revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_environment(vime_root: Path, rl_kernel_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = [str(vime_root), str(rl_kernel_root), "/root/Megatron-LM"]
    if env.get("PYTHONPATH"):
        existing.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(existing)
    env["RL_KERNEL_ROOT"] = str(rl_kernel_root)
    env["TP_SIZE"] = "4"
    env["CP_SIZE"] = "2"
    env["ROLLOUT_TOP_P"] = "1.0"
    return env


def build_command(config: Mapping[str, Any], vime_root: Path) -> list[str]:
    script = vime_root / str(config.get("vime_script", ""))
    if not script.is_file():
        raise FileNotFoundError(f"Vime entry script does not exist: {script}")
    return ["bash", str(script)]


def build_report(
    config: Mapping[str, Any],
    *,
    vime_root: Path,
    rl_kernel_root: Path,
    command: list[str],
    status: str,
    returncode: int | None,
    log_text: str,
    log_path: Path | None,
    runtime_evidence: Mapping[str, Any] | None = None,
    runtime_evidence_path: Path | None = None,
) -> dict[str, Any]:
    provider_active = PROVIDER_MARKER in log_text
    fallback_observed = any(marker in log_text for marker in FALLBACK_MARKERS)
    strict_provider_passed = status == "passed" and provider_active and not fallback_observed
    effective_status = (
        "passed" if strict_provider_passed else ("failed" if status == "passed" else status)
    )
    attention_status = _operator_evidence_status(runtime_evidence, "attention")
    ffn_status = _operator_evidence_status(runtime_evidence, "ffn")
    return {
        "schema_version": "rlkernel.vime_validation_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": effective_status,
        "claim_boundary": {
            "qwen3_8b_tp4_cp2_vime_training": strict_provider_passed,
            "attention_train_infer_consistency": attention_status,
            "ffn_train_infer_consistency": ffn_status,
            "reason": (
                "attention and FFN require executed Megatron/vLLM runtime readbacks; "
                "the evidence contract accepts only exact-zero comparison metrics"
            ),
        },
        "config": dict(config),
        "topology": config["training"],
        "provider": {
            "configured_path": config["linear_logp_provider"]["path"],
            "configured_mode": config["linear_logp_provider"]["mode"],
            "backend_id": config["linear_logp_provider"]["backend_id"],
            "active_observed": provider_active,
            "fallback_observed": fallback_observed,
        },
        "command": command,
        "returncode": returncode,
        "artifacts": {
            "log": None if log_path is None else str(log_path),
            "runtime_evidence": (
                None if runtime_evidence_path is None else str(runtime_evidence_path)
            ),
        },
        "runtime_evidence": (None if runtime_evidence is None else dict(runtime_evidence)),
        "revisions": {
            "vime": _revision(vime_root),
            "rl_kernel": _revision(rl_kernel_root),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vime-root", type=Path, default=Path(os.environ.get("VIME_ROOT", ".")))
    parser.add_argument(
        "--rl-kernel-root",
        type=Path,
        default=Path(os.environ.get("RL_KERNEL_ROOT", ".")),
    )
    parser.add_argument("--output", type=Path, default=Path("qwen3_8b_tp4_cp2.validation.json"))
    parser.add_argument(
        "--runtime-evidence",
        type=Path,
        default=None,
        help="post-execution Megatron/vLLM operator readback JSON (strict exact-zero contract)",
    )
    parser.add_argument("--run", action="store_true", help="execute the Vime script")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config)
    runtime_evidence = load_runtime_evidence(args.runtime_evidence)
    validate_runtime_evidence(runtime_evidence)
    vime_root = args.vime_root.resolve()
    rl_kernel_root = args.rl_kernel_root.resolve()
    command = build_command(config, vime_root)

    status = "not_run"
    returncode: int | None = None
    log_text = ""
    log_path: Path | None = None
    if args.run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.output.with_suffix(".log")
        env = build_environment(vime_root, rl_kernel_root)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                cwd=vime_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        returncode = process.returncode
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        status = "passed" if returncode == 0 else "failed"

    report = build_report(
        config,
        vime_root=vime_root,
        rl_kernel_root=rl_kernel_root,
        command=command,
        status=status,
        returncode=returncode,
        log_text=log_text,
        log_path=log_path,
        runtime_evidence=runtime_evidence,
        runtime_evidence_path=args.runtime_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"passed", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
