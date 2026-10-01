# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

import copy
import json
import os
import subprocess  # noqa: F401 - retained as a test seam for launcher substitution
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from .replay import load_sample, source_configuration, source_baseline
from .evidence import diagnosis


def matrix_groups(value: str, baseline: str = "M000") -> list[str]:
    if value == "auto":
        return list(dict.fromkeys([baseline, "M111", "M011", "M101", "M110"]))
    if value == "full":
        return [f"M{number:03b}" for number in range(8)]
    if value == "attribute":
        return list(dict.fromkeys([baseline, "M111", "M011", "M101", "M110"]))
    groups = value.upper().split(",")
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(len(g) != 4 or g[0] != "M" or set(g[1:]) - {"0", "1"} for g in groups)
    ):
        raise ValueError(
            "--matrix must be auto, attribute, full, or unique comma-separated Mxxx groups"
        )
    return groups


def run_debug(paths: Any, profile: dict[str, Any], args: Any, *, adapter: Any) -> int:
    """Reuse production launchers, changing only replay/capture and module selection."""
    from rl_engine import repro
    from .entry import analyze, save_report, print_report
    from .ux import preflight, print_preflight, run_progress

    json_stdout = sys.stdout if getattr(args, "as_json", False) else None
    if json_stdout is not None:
        # Keep stdout a single machine-readable document. Human progress stays
        # visible on stderr, so `rlk debug ... --json > result.json` is safe.
        sys.stdout = sys.stderr
    try:
        batch = load_sample(args.source, args.sample, args.step)
        baseline, origin = source_baseline(args.source, getattr(args, "baseline", None))
        groups = matrix_groups(args.matrix, baseline)
        source_options = source_configuration(args.source)
        for key, value in source_options.get("options", {}).items():
            if (
                key == "cp_size"
                and args.explicit_flags & {"--tp", "--tp-size"}
                and not args.explicit_flags & {"--cp", "--cp-size"}
            ):
                continue
            aliases = {
                "tp_size": ("tp", "tp-size"),
                "cp_size": ("cp", "cp-size"),
                "rollout_tp_size": ("rollout-tp", "rollout-tp-size"),
                "rollout_cp_size": ("rollout-cp", "rollout-cp-size"),
            }
            flags = aliases.get(
                key, (key.replace("_", "-"), key.removeprefix("rollout_").replace("_", "-"))
            )
            if not args.explicit_flags & {f"--{flag}" for flag in flags}:
                setattr(args, key, value)
        if source_options.get("model_root") and "--model-root" not in args.explicit_flags:
            paths = replace(paths, model_root=Path(source_options["model_root"]))
        if batch["source_step"] and not (args.explicit_flags & {"--model-root"}):
            raise ValueError(
                "For a post-update failure, pass --model-root with the "
                "exported HF checkpoint for that step; the initial "
                "checkpoint is not equivalent"
            )
        for key, value in batch.get("sampling", {}).items():
            flag = key.replace("_", "-")
            if not args.explicit_flags & {f"--{flag}", f"--rollout-{flag}"}:
                setattr(args, f"rollout_{key}", value)
        batch["baseline"] = baseline
        batch["sampling"] = {
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "top_k": args.rollout_top_k,
        }
        replacements = adapter.replacements(args.debug_model, args)
        eligible = [candidate for candidate in replacements if candidate.eligible]
        if not eligible and any(group != "M000" for group in groups):
            if args.matrix != "auto":
                raise ValueError(
                    "Unsupported replacement: "
                    + "; ".join(
                        reason for candidate in replacements for reason in candidate.reasons
                    )
                )
            if baseline != "M000":
                raise ValueError(
                    "Original replacement configuration is unsupported by this adapter"
                )
            groups = [baseline]
        print(f"[baseline] {baseline} ({origin})", flush=True)
        print("[scope] one frozen sample; eager pre-update replay; independent scoring", flush=True)
        capability = "candidate requires verification" if eligible else "no eligible replacement"
        print(f"[capability] {capability}", flush=True)
        for candidate in replacements:
            for reason in candidate.reasons:
                print(f"  {reason}", flush=True)
        run_id = args.run_id or datetime.now(timezone.utc).strftime("debug-%Y%m%dT%H%M%S%fZ")
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("--run-id must be a single directory name")
        root = (
            Path(args.resume).expanduser().resolve() if args.resume else paths.output_root / run_id
        )
        resumed = args.resume is not None
        if resumed:
            replay = root / "frozen-replay.json"
            if not root.is_dir() or not replay.is_file():
                raise ValueError(f"resume directory lacks frozen-replay.json: {root}")
            previous_batch = json.loads(replay.read_text(encoding="utf-8"))
            for key in ("tokens", "prompt_length", "mask", "sampling"):
                if previous_batch.get(key) != batch.get(key):
                    raise ValueError(f"resume replay differs in {key}; start a new run instead")
            existing = (
                json.loads((root / "matrix-report.json").read_text())
                if (root / "matrix-report.json").is_file()
                else {}
            )
            completed_groups = {
                item["group"]
                for item in existing.get("groups", [])
                if item.get("process_returncode", 0) == 0
                and item.get("report", {}).get("status") != "not_comparable"
            }
            run_id = root.name
            print(
                f"[resume] {root}; completed={','.join(sorted(completed_groups)) or 'none'}",
                flush=True,
            )
        else:
            if root.exists():
                raise ValueError(
                    f"diagnostic output already exists: {root}; use --report-only or "
                    "--resume to continue"
                )
            root.mkdir(parents=True)
            replay = root / "frozen-replay.json"
            replay.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
            existing = {}
            completed_groups = set()
        debug_profile = copy.deepcopy(profile)
        repro._validate_topology_args(args)
        planned_groups = list(groups)
        preflight_report = preflight(
            paths,
            profile,
            args,
            model=args.debug_model,
            adapter=adapter,
            groups=planned_groups,
            inspect_host=not args.dry_run,
        )
        print_preflight(preflight_report, as_json=False)
        if not preflight_report["passed"]:
            raise ValueError(
                "preflight failed; fix the required checks above before launching GPU jobs"
            )
        replicas = 8 // (args.tp_size * args.cp_size)
        results = list(existing.get("groups", []))
        if results:
            print(f"[resume] skipping {','.join(sorted(completed_groups))}", flush=True)
        groups = [group for group in groups if group not in completed_groups]
        for group in groups:
            run_args = copy.copy(args)
            run_args.command = "run"
            run_args.arm = "consistency"
            run_args.rollouts, run_args.require_updates = 1, False
            run_args.kl_coef = 0.0  # no reference forward mixed into the policy trace
            run_args.wait = True
            run_args.run_id = f"{run_id}-{group}"
            run_args.max_response_len = len(batch["tokens"]) - batch["prompt_length"]
            env = os.environ.copy()
            env.update(
                RL_KERNEL_ALIGNMENT_DIAGNOSTICS="1",
                RL_KERNEL_ALIGNMENT_REPLAY=str(replay),
                RL_KERNEL_ALIGNMENT_GROUP=group,
            )
            env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        str(paths.rl_kernel_root),
                        str(paths.vime_root),
                        str(paths.megatron_root),
                        repro._profile_path(profile, "vllm_root"),
                        env.get("PYTHONPATH", ""),
                    ),
                )
            )
            if args.backend == "rocm":
                run_args.samples_per_prompt = run_args.global_batch_size = replicas
                run_args.rollout_batch_size = 1
                run_args.grpo_std_normalization = "disabled"
            else:
                # Last occurrence wins in argparse; preserve all other profile options.
                debug_profile["runner_args"] = list(profile.get("runner_args", [])) + [
                    ["--n-samples-per-prompt", str(replicas)],
                    ["--global-batch-size", str(replicas)],
                    ["--rollout-batch-size", "1"],
                ]
            command = repro._runner_command(paths, debug_profile, run_args)
            if args.backend == "cuda":
                command[command.index("--group") + 1] = group
            run_dir = paths.output_root / run_args.run_id
            plan = {
                "group": group,
                "command": command,
                "run_dir": str(run_dir),
                "replay": str(replay),
                "capture_mode": "eager_observational",
            }
            (root / f"{group}.plan.json").write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8"
            )
            if args.dry_run:
                print(f"{group}: {run_dir}")
                continue
            log_path = root / f"{group}.log"
            print(f"[run] {group} started; log={log_path}", flush=True)
            # Keep framework startup noise in a file, while the terminal shows
            # semantic progress and actionable results.
            returncode, progress = run_progress(
                command, env=env, log_path=log_path, run_dir=run_dir, group=group
            )
            report = analyze(run_dir, output=root / f"{group}.report.json")
            result = {
                "group": group,
                "process_returncode": returncode,
                "run_dir": str(run_dir),
                "log": str(log_path),
                "report": report,
                **progress,
            }
            apply_evidence_gates(result, results[0]["report"] if results else None)
            results.append(result)
            save_report(root / f"{group}.report.json", report)
            print_report(report, label=group)
            print(f"[run] {group} finished in {result['elapsed_seconds']:.1f}s", flush=True)
            if args.matrix in {"auto", "attribute"}:
                if report["status"] == "not_comparable":
                    print(
                        "[stop] incomplete evidence; inspect job log before further trials.",
                        flush=True,
                    )
                    break
                if group == baseline and report["status"] == "equal":
                    print("[stop] original mismatch not reproduced; no repair claim.", flush=True)
                    break
                if args.matrix == "auto" and group == "M111" and report["status"] == "equal":
                    print(
                        "[stop] replacement verified on this replay; production is unchanged.",
                        flush=True,
                    )
                    break
                # Prioritize the module containing the first observed mismatch.
                if args.matrix == "auto" and group == "M111":
                    module = (report.get("first_divergence") or {}).get("module")
                    target = {"attention": "M011", "ffn": "M101", "logp": "M110"}.get(module)
                    index = groups.index(group) + 1
                    if target in groups[index:]:
                        groups.remove(target)
                        groups.insert(index, target)
        summary = {
            "schema_version": "rlkernel.debug_matrix_report.v2",
            "adapter": adapter.name,
            "baseline": baseline,
            "baseline_origin": origin,
            "replacement": [asdict(candidate) for candidate in replacements],
            "replay": str(replay),
            "groups": results,
            "matrix": args.matrix,
            "configuration": {
                "sampling": batch.get("sampling", {}),
                "topology": {
                    "training_tp": args.tp_size,
                    "training_cp": args.cp_size,
                    "rollout_tp": args.rollout_tp_size,
                    "rollout_cp": args.rollout_cp_size,
                },
                "use_rollout_logprobs": False,
            },
            "planned": args.dry_run,
            "scope": (
                "module attribution on one frozen sample; "
                "eager diagnostic replay is not a performance benchmark"
            ),
        }
        finish_summary(root, summary)
        # One canonical compact report is used by live runs and --report-only.
        # Keeping this in ux.py prevents the two paths from drifting in schema.
        from .ux import compact_summary

        compact_summary(root, summary)
        print(f"[result] {summary['conclusion']}", flush=True)
        print(f"[report] {root / 'matrix-report.md'}", flush=True)
        print(f"[summary] {root / 'debug-summary.md'}", flush=True)
        print(
            "[scope] bitwise result is for this frozen replay and topology only; "
            "graph mode, other batches, optimizer updates and performance require "
            "separate validation",
            flush=True,
        )
        if getattr(args, "as_json", False):
            sys.stdout = json_stdout
            print((root / "debug-summary.json").read_text(encoding="utf-8"), flush=True)
        return (
            0
            if args.dry_run
            else {"equal": 0, "diverged": 1, "not_comparable": 2}[summary["status"]]
        )
    except (OSError, ValueError, KeyError) as exc:
        if json_stdout is not None:
            sys.stdout = json_stdout
        raise repro.ReproError(str(exc)) from exc


def apply_evidence_gates(result: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    """Identical live/offline gates, including process failures after capture."""
    report = result["report"]
    if baseline:
        for key in ("runtime_identity", "model_weights"):
            if report.get(key) != baseline.get(key):
                report["errors"].append(f"cross-arm {key} changed")
        if report.get("output_head", {}).get("weight_fingerprints") != baseline.get(
            "output_head", {}
        ).get("weight_fingerprints"):
            report["errors"].append("cross-arm output head weights changed")
    code = result.get("process_returncode")
    report["execution"] = {"process_returncode": code, "log": result.get("log")}
    summary_path = Path(result["run_dir"]) / "single-arm-summary.json"
    arm_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if arm_summary.get("frozen_sources_match") is False:
        report["errors"].append("frozen source fingerprint changed during capture")
    if code:
        completed = arm_summary.get("launcher_returncode") == 0
        if not completed:
            report["errors"].insert(0, f"capture process exited with {code}")
    if report["errors"]:
        report["status"] = "not_comparable"
    report["diagnosis"] = diagnosis(report)


def finish_summary(directory: Path, summary: dict[str, Any]) -> None:
    reports = {r["group"]: r["report"] for r in summary["groups"]}
    original = reports.get(summary.get("baseline"))
    replacement = reports.get("M111")
    if summary.get("planned"):
        conclusion, status = "planned", "not_comparable"
    elif not reports or any(r["status"] == "not_comparable" for r in reports.values()):
        conclusion, status = "inconclusive", "not_comparable"
    elif original and original["status"] == "equal":
        # Manual matrices may still reveal another failing configuration.
        conclusion = "not_reproduced"
        status = "diverged" if any(r["status"] == "diverged" for r in reports.values()) else "equal"
    elif (
        original
        and original["status"] == "diverged"
        and replacement
        and replacement["status"] == "equal"
    ):
        conclusion, status = "verified_on_replay", "equal"
    elif replacement:
        conclusion = (
            "unresolved" if replacement["status"] == "diverged" else "checked_without_baseline"
        )
        status = replacement["status"]
    else:
        conclusion = (
            "localized"
            if any(r["status"] == "diverged" for r in reports.values())
            else "checked_without_baseline"
        )
        status = "diverged" if conclusion == "localized" else "equal"
    summary.update(
        status=status,
        conclusion=conclusion,
        first_divergence=(original or {}).get("first_divergence"),
    )
    if summary.get("matrix") == "attribute" or {
        item.get("group") for item in summary.get("groups", [])
    } & {"M011", "M101", "M110"}:
        reports = {item["group"]: item["report"] for item in summary.get("groups", [])}
        summary["attribution"] = {
            module: {
                "group": group,
                "combination": (
                    "native attention + RL-Kernel FFN + RL-Kernel logprob"
                    if module == "attention"
                    else "RL-Kernel attention + native FFN + RL-Kernel logprob"
                    if module == "ffn"
                    else "RL-Kernel attention + RL-Kernel FFN + native logprob"
                ),
                "passed_on_replay": reports.get(group, {}).get("status") == "equal",
                "status": reports.get(group, {}).get("status", "not_run"),
            }
            for module, group in (("attention", "M011"), ("ffn", "M101"), ("logp", "M110"))
        }
    (directory / "matrix-report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_matrix_summary(directory, summary)


def write_matrix_summary(directory: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Module diagnostic matrix",
        "",
        summary["scope"],
        f"Conclusion: {summary.get('conclusion', 'inconclusive')}",
        f"Original configuration: {summary.get('baseline', 'unknown')}",
        "",
        "| Group | Replay result | First observed divergence |",
        "| --- | --- | --- |",
    ]
    for group in summary["groups"]:
        report = group["report"]
        point = report.get("first_divergence")
        label = (
            f"layer {point['layer']}, token {point.get('position')}, {point.get('stage')}"
            if point
            else "none in captured stages"
        )
        lines.append(f"| {group['group']} | {report['status']} | {label} |")
    lines += [
        "",
        "Mixed/native arms may differ by design. A changed mismatch is "
        "attribution evidence, not proof of a kernel bug.",
        "Missing identity, layers, shards or endpoint evidence is never a pass.",
        "Inspect each group's report and .pt snapshot, then rerun M111 with "
        "the same frozen replay after fixing the implementation.",
    ]
    if summary.get("attribution"):
        lines += [
            "",
            "## Module attribution",
            "",
            "| Module | Arm | Result |",
            "| --- | --- | --- |",
        ]
        for module, value in summary["attribution"].items():
            result = "passed on replay" if value["passed_on_replay"] else value["status"]
            lines.append(f"| {module} | {value['group']} | {result} |")
    (directory / "matrix-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
