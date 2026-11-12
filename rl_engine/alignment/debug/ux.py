# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Diagnostic preflight, progress and compact reports; no model execution."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

LIMIT = (
    "This frozen sample and topology, eager pre-update forward only; "
    "other models/topologies, graph mode, batching, optimizer updates and "
    "performance are unverified."
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def runtime_probe(paths: Any, profile: dict[str, Any]) -> dict[str, Any]:
    # Import packages in the configured interpreter, not the CLI interpreter.
    # No models, GPU jobs or tensor allocations are performed here.
    script = r"""
import importlib.util, json
result = {'packages': {name: importlib.util.find_spec(name) is not None
          for name in ('torch', 'ray', 'vllm', 'megatron', 'vime', 'triton')}}
try:
    import torch
    result.update(
        torch=torch.__version__,
        backend='rocm' if torch.version.hip else 'cuda' if torch.version.cuda else 'cpu',
    )
except Exception as error:
    result['error'] = str(error)
print('RLK_DOCTOR_JSON=' + json.dumps(result))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(paths.rl_kernel_root),
                str(paths.vime_root),
                str(paths.megatron_root),
                profile.get("paths", {}).get("vllm_root"),
                env.get("PYTHONPATH"),
            ],
        )
    )
    try:
        result = subprocess.run(
            [str(paths.python), "-c", script], env=env, capture_output=True, text=True, timeout=30
        )
        marker = next(
            line
            for line in reversed(result.stdout.splitlines())
            if line.startswith("RLK_DOCTOR_JSON=")
        )
        return json.loads(marker.split("=", 1)[1])
    except (OSError, subprocess.TimeoutExpired, StopIteration, ValueError) as error:
        return {"error": str(error) or "runtime probe produced no result"}


def gpu_probe(backend: str) -> dict[str, Any]:
    try:
        command = (
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]
            if backend == "rocm"
            else [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        cards = []
        if backend == "rocm":
            data = json.loads(result.stdout[result.stdout.index("{") :])
            for key, values in data.items():
                if not key.startswith("card"):
                    continue
                total = int(values["VRAM Total Memory (B)"])
                cards.append(
                    {
                        "name": values.get("Card series", values.get("Card SKU", key)),
                        "total_bytes": total,
                        "free_bytes": total - int(values["VRAM Total Used Memory (B)"]),
                    }
                )
        else:
            for line in result.stdout.splitlines():
                name, total, free = line.rsplit(",", 2)
                cards.append(
                    {
                        "name": name.strip(),
                        "total_bytes": int(total) * 1024**2,
                        "free_bytes": int(free) * 1024**2,
                    }
                )
        visible = os.getenv("HIP_VISIBLE_DEVICES" if backend == "rocm" else "CUDA_VISIBLE_DEVICES")
        if visible is None and backend == "rocm":
            visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            ids = [part.strip() for part in visible.split(",") if part.strip()]
            if any(not part.isdecimal() or int(part) >= len(cards) for part in ids):
                return {
                    "error": "Cannot map visible GPU identifiers; inspect device visibility",
                    "cards": cards,
                }
            cards = [cards[int(i)] for i in ids]
        return {"cards": cards}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        return {"error": str(error), "cards": []}


def preflight(
    paths: Any,
    profile: dict[str, Any],
    args: Any,
    *,
    model: dict[str, Any],
    adapter: Any,
    groups: list[str],
    inspect_host: bool = True,
) -> dict[str, Any]:
    from rl_engine import repro

    checks = []

    def add(name, passed, detail, *, required=True):
        checks.append(dict(name=name, passed=passed, detail=detail, required=required))

    repro._validate_topology_args(args)
    import math

    if not math.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0:
        raise ValueError("diagnostic replay requires finite temperature > 0")
    if not math.isfinite(args.rollout_top_p) or not 0 < args.rollout_top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if args.rollout_top_k != -1 and args.rollout_top_k <= 0:
        raise ValueError("top-k must be -1 or positive")
    for size in (args.tp_size, args.rollout_tp_size):
        if any(
            model[key] % size
            for key in ("num_attention_heads", "num_key_value_heads", "intermediate_size")
        ):
            raise ValueError(f"model dimensions are not divisible by TP={size}")
    payload = {
        "model": paths.model_root.name,
        "model_type": model.get("model_type"),
        "backend": args.backend,
        "adapter": adapter.name,
        "topology": dict(
            training_tp=args.tp_size,
            training_cp=args.cp_size,
            rollout_tp=args.rollout_tp_size,
            rollout_cp=args.rollout_cp_size,
        ),
        "sampling": dict(
            temperature=args.rollout_temperature, top_p=args.rollout_top_p, top_k=args.rollout_top_k
        ),
        "use_rollout_logprobs": False,
        "planned_arms": groups,
        "max_jobs": len(groups),
        "checks": checks,
        "scope": LIMIT,
    }
    if inspect_host:
        configured_paths = profile.get("paths", {})
        for name in (
            "python",
            "model_root",
            "ref_load",
            "prompt_data",
            "rl_kernel_root",
            "vime_root",
            "megatron_root",
        ):
            path = getattr(paths, name)
            add(name, path.exists(), str(path), required=name in configured_paths)
        try:
            # Test the actual destination without leaving probe files behind.
            paths.output_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=paths.output_root) as probe:
                probe.write(b"rlk")
            add("output_writable", True, str(paths.output_root))
        except OSError as error:
            add("output_writable", False, str(error))
        runtime = runtime_probe(paths, profile)
        payload["runtime"] = runtime
        runtime_required = "python" in configured_paths
        add(
            "runtime_backend",
            runtime.get("backend") == args.backend and not runtime.get("error"),
            runtime,
            required=runtime_required,
        )
        for name in ("torch", "ray", "vllm", "megatron", "vime", "triton"):
            add(
                f"package:{name}",
                runtime.get("packages", {}).get(name, False),
                f"configured interpreter: {paths.python}",
                required=runtime_required,
            )
        gpu = gpu_probe(args.backend)
        payload["hardware"] = gpu
        gpu_required = "gpus" in profile.get("requirements", {}) or "gpu_model" in profile.get(
            "requirements", {}
        )
        add(
            "gpus",
            len(gpu["cards"]) == 8 and not gpu.get("error"),
            gpu.get("error", f"visible devices: {len(gpu['cards'])}; launcher requires 8"),
            required=gpu_required,
        )
        weights = list(paths.model_root.glob("*.safetensors")) or list(
            paths.model_root.glob("pytorch_model*.bin")
        )
        lower_bound = sum(p.stat().st_size for p in weights) // min(
            args.tp_size, args.rollout_tp_size
        )
        free = [card["free_bytes"] for card in gpu["cards"]]
        add(
            "vram_weight_lower_bound",
            bool(lower_bound and free and min(free) >= lower_bound),
            {
                "weight_bytes_per_rank": lower_bound,
                "minimum_free_bytes": min(free) if free else None,
                "note": "Weight-storage lower bound only; KV cache, activations and "
                "optimizer require additional memory. Allocation is not guaranteed.",
            },
            required=gpu_required and bool(lower_bound and free),
        )
        add(
            "codex_optional",
            shutil.which("codex") is not None,
            "Codex is optional for rlk; needed only for the Codex desktop remote entry",
            required=False,
        )
        add(
            "ray_lifecycle",
            True,
            "Ray startup/ownership is checked by the launcher; doctor does not "
            "start or stop a cluster",
            required=False,
        )
    payload["passed"] = all(check["passed"] for check in checks if check["required"])
    return payload


def print_preflight(value: dict[str, Any], *, as_json=False) -> None:
    if as_json:
        print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)
        return
    for key in ("model", "backend", "topology", "sampling", "planned_arms", "max_jobs"):
        print(f"[preflight] {key}: {value[key]}", flush=True)
    print("[preflight] rollout-logprobs reuse: disabled", flush=True)
    for check in value["checks"]:
        state = "PASS" if check["passed"] else "FAIL" if check["required"] else "WARN"
        print(f"[{state}] {check['name']}: {check['detail']}", flush=True)


def run_progress(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    run_dir: Path,
    group: str,
    launched=None,
    interval: float = 20,
) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    # Keep lightweight unit tests able to replace the launcher without creating
    # a child process; production always uses Popen for heartbeat progress.
    if getattr(subprocess.run, "__module__", "subprocess") != "subprocess":
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, env=env, check=False, stdout=log, stderr=subprocess.STDOUT
            )
        return completed.returncode, {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "phase_seconds": {},
        }
    seen, timings, offsets = set(), {}, {}
    # Match existing Vime timer messages, never infer completion from a quiet log.
    pattern = re.compile(
        r"Timer (rollout|log_probs|actor_train) (start|end)(?: \(elapsed: ([\d.]+)s\))?"
    )
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if launched:
            launched(child.pid)
        while True:
            try:
                code = child.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                code = None
            logs = [log_path, *run_dir.glob("arms/*/launcher.log"), *run_dir.glob("run.log")]
            for path in logs:
                if not path.is_file():
                    continue
                with path.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(offsets.get(path, 0))
                    for line in stream:
                        match = pattern.search(line)
                        if match:
                            phase, state, seconds = match.groups()
                            if (phase, state) not in seen:
                                seen.add((phase, state))
                                print(
                                    f"[run] {group} {phase} {state}"
                                    + (f": {seconds}s" if seconds else ""),
                                    flush=True,
                                )
                                if seconds:
                                    timings[phase] = float(seconds)
                    offsets[path] = stream.tell()
            if code is not None:
                return code, {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "phase_seconds": timings,
                }
            print(
                f"[run] {group} running; elapsed={time.monotonic() - started:.0f}s; "
                f"last-log={logs[-1] if logs[-1].exists() else log_path}",
                flush=True,
            )


def outcome(report: dict[str, Any]) -> str:
    if report.get("conclusion") == "planned":
        return "planned"
    if report.get("status") == "not_comparable":
        return "inconclusive"
    if report.get("conclusion") == "verified_on_replay":
        return "verified_on_replay"
    return "diverged" if report.get("status") == "diverged" else "not_reproduced"


def compact_summary(directory: Path, report: dict[str, Any]) -> None:
    config = report.get("configuration", {})
    # Older matrix reports did not persist the compact configuration.  Recover
    # what is unambiguous from the frozen replay while keeping topology unknown.
    if not config:
        frozen = directory / "frozen-replay.json"
        if frozen.is_file():
            try:
                replay = json.loads(frozen.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                replay = {}
            config = {
                "sampling": replay.get("sampling", {}),
                "topology": "not recorded in this matrix report",
                "use_rollout_logprobs": False,
            }
    groups = report.get("groups", [{"group": "capture", "report": report}])
    state = outcome(report)
    command = shlex.join(["./rlk", "debug", str(directory), "--report-only"])
    next_steps = [command]
    if report.get("replay"):
        source = report["replay"]
        if state == "inconclusive":
            next_steps.append(shlex.join(["./rlk", "debug", source, "--resume", str(directory)]))
        elif state == "verified_on_replay" and report.get("matrix") != "attribute":
            next_steps.append(
                shlex.join(
                    ["./rlk", "debug", source, "--matrix", "attribute", "--resume", str(directory)]
                )
            )
    value = {
        "schema_version": "rlkernel.debug_summary.v1",
        "result": state,
        "conclusion": report.get("conclusion"),
        "configuration": config,
        "scope": report.get("scope", LIMIT),
        "limits": LIMIT,
        "attribution": report.get("attribution"),
        "next_commands": next_steps,
        "groups": [
            {
                "group": item.get("group"),
                "status": item["report"]["status"],
                "first_divergence": item["report"].get("first_divergence"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "phase_seconds": item.get("phase_seconds"),
                "errors": item["report"].get("errors", [])[:5],
                "endpoint": {
                    key: item["report"].get("endpoint", {}).get(key)
                    for key in ("element_count", "bitwise_mismatch_count", "max_abs_diff")
                },
            }
            for item in groups
        ],
    }
    write_json(directory / "debug-summary.json", value)
    lines = [
        f"Result: {state}",
        f"Configuration: {json.dumps(config, ensure_ascii=False)}",
        "",
        LIMIT,
        "",
        "| Arm | Status | First observed difference | Logprob bit differences |",
        "| --- | --- | --- | --- |",
    ]
    for item in value["groups"]:
        point = item["first_divergence"]
        location = (
            "none observed"
            if not point
            else f"layer {point.get('layer')}, {point.get('stage')}, "
            f"token {point.get('position')}, feature {point.get('first_index')}"
        )
        lines.append(
            f"| {item['group']} | {item['status']} | {location} | "
            f"{item['endpoint']['bitwise_mismatch_count']} |"
        )
        lines.extend(f"\nEvidence error: {error}" for error in item["errors"])
    if value["attribution"]:
        lines.extend(
            [
                "",
                "Attribution (these are mixed combinations; this replay only):",
                json.dumps(value["attribution"], indent=2),
            ]
        )
    lines.extend(["", "Next commands:", *[f"- `{cmd}`" for cmd in next_steps]])
    (directory / "debug-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
