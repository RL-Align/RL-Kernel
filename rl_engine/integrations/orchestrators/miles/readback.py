# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Cross-process audit of the per-worker integration readback files.

Each Megatron worker writes ``<framework>-<target>-<pid>.json`` when the
integration is installed; this module reads a directory of those files and
answers the two questions a single-process check cannot: did every worker
install, and did every RL-Kernel route it selected actually execute.

Usage:
    python -m rl_engine.integrations.orchestrators.miles.readback DIR [--expect-workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RL_KERNEL_BACKEND_PREFIXES = ("rlkernel.", "pytorch-vocab-parallel-logp")
_MODULES = ("attention", "ffn", "logp")


@dataclass(frozen=True)
class WorkerRecord:
    path: Path
    framework: str
    target: str
    pid: int | None
    cases: dict[str, str]
    hooks: tuple[str, ...]
    operators: dict[str, str]
    fallbacks: tuple[Any, ...]
    problems: tuple[str, ...]


@dataclass(frozen=True)
class AuditReport:
    directory: Path
    workers: tuple[WorkerRecord, ...]
    expected_workers: int | None
    problems: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.problems and all(not worker.problems for worker in self.workers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "healthy": self.healthy,
            "expected_workers": self.expected_workers,
            "problems": list(self.problems),
            "workers": [
                {
                    "path": str(worker.path),
                    "framework": worker.framework,
                    "target": worker.target,
                    "pid": worker.pid,
                    "cases": worker.cases,
                    "hooks": list(worker.hooks),
                    "operators": worker.operators,
                    "fallbacks": list(worker.fallbacks),
                    "problems": list(worker.problems),
                }
                for worker in self.workers
            ],
        }


def _training_route(case_id: str) -> str:
    return case_id.split("/")[0].strip().upper()


def _parse_name(path: Path) -> tuple[str, str, int | None, list[str]]:
    stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
    parts = stem.split("-")
    if len(parts) != 3 or not parts[2].isdigit():
        return "", "", None, [f"unexpected readback file name {path.name!r}"]
    return parts[0], parts[1], int(parts[2]), []


def _worker_record(
    path: Path,
    payload: dict[str, Any],
    expected_target: str,
    require_execution: bool,
) -> WorkerRecord:
    framework, target, pid, problems = _parse_name(path)
    if target and target != expected_target:
        problems.append(f"file name target {target!r} is not {expected_target!r}")
    payload_target = str(payload.get("target", ""))
    if payload_target and payload_target != expected_target:
        problems.append(f"payload target {payload_target!r} is not {expected_target!r}")

    cases = {
        str(module): str(case.get("case_id", ""))
        for module, case in ((payload.get("plan") or {}).get("cases") or {}).items()
    }
    if set(cases) != set(_MODULES):
        problems.append(f"plan modules are {sorted(cases)}, expected {list(_MODULES)}")

    hooks_payload = payload.get("installed_hooks") or {}
    hooks = tuple(sorted(str(module) for module in hooks_payload))
    operators_payload = payload.get("operators") or {}
    operators = {
        str(module): str(record.get("backend_id", ""))
        for module, record in operators_payload.items()
    }
    fallbacks = tuple(payload.get("fallbacks") or [])
    if fallbacks:
        problems.append(f"fallbacks recorded: {list(fallbacks)}")

    for module, case_id in cases.items():
        if not case_id or _training_route(case_id) != "R":
            continue
        if module not in hooks_payload:
            problems.append(f"{module}: RL-Kernel route selected but no installed hook")
        if not require_execution:
            continue
        record = operators_payload.get(module)
        if not isinstance(record, dict):
            problems.append(f"{module}: RL-Kernel route selected but never executed")
            continue
        implementation = str(record.get("implementation", ""))
        if implementation != "rl_kernel":
            problems.append(f"{module}: executed as {implementation!r}, not 'rl_kernel'")
        backend_id = str(record.get("backend_id", ""))
        if not backend_id.startswith(_RL_KERNEL_BACKEND_PREFIXES):
            problems.append(f"{module}: backend {backend_id!r} is not an RL-Kernel backend")
        if int(record.get("call_count", 0)) <= 0:
            problems.append(f"{module}: execution record has call_count 0")

    return WorkerRecord(
        path=path,
        framework=framework,
        target=target,
        pid=pid,
        cases=cases,
        hooks=hooks,
        operators=operators,
        fallbacks=fallbacks,
        problems=tuple(problems),
    )


def _load(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"unreadable readback file: {exc}"]
    if not isinstance(payload, dict):
        return None, ["readback file is not a JSON object"]
    return payload, []


def audit_readbacks(
    directory: str | Path,
    *,
    expected_workers: int | None = None,
    target: str = "training",
    require_execution: bool = False,
) -> AuditReport:
    """Audit every readback file in ``directory`` for one training target.

    By default this checks the installation: every expected worker left a file, the
    workers agree on one plan, no fallback was recorded, and each RL-Kernel route has
    an installed hook. ``require_execution`` additionally demands an execution record
    per RL-Kernel route, which is only meaningful once training has run.
    """
    root = Path(directory)
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    problems: list[str] = []
    if not root.is_dir():
        problems.append(f"readback directory {root} does not exist")
    elif not paths:
        problems.append(f"no readback files in {root}")
    if expected_workers is not None and len(paths) != expected_workers:
        problems.append(f"expected {expected_workers} worker readback files, found {len(paths)}")

    workers: list[WorkerRecord] = []
    for path in paths:
        payload, load_problems = _load(path)
        if payload is None:
            workers.append(
                WorkerRecord(
                    path=path,
                    framework="",
                    target="",
                    pid=None,
                    cases={},
                    hooks=(),
                    operators={},
                    fallbacks=(),
                    problems=tuple(load_problems),
                )
            )
            continue
        workers.append(_worker_record(path, payload, target, require_execution))

    plans = {tuple(sorted(worker.cases.items())) for worker in workers if worker.cases}
    if len(plans) > 1:
        problems.append("workers disagree on the integration plan")

    return AuditReport(
        directory=root,
        workers=tuple(workers),
        expected_workers=expected_workers,
        problems=tuple(problems),
    )


def _format_table(report: AuditReport) -> str:
    header = f"{'file':<44} {'pid':>7} {'target':<9} {'plan':<22} {'hooks':>5} {'ops':>4}  status"
    lines = [header, "-" * len(header)]
    for worker in report.workers:
        plan = " ".join(f"{module}={worker.cases.get(module, '?')}" for module in _MODULES)
        status = "ok" if not worker.problems else "; ".join(worker.problems)
        lines.append(
            f"{worker.path.name:<44} {worker.pid if worker.pid is not None else '-':>7} "
            f"{worker.target:<9} {plan:<22} {len(worker.hooks):>5} "
            f"{len(worker.operators):>4}  {status}"
        )
    lines.append("")
    healthy = sum(1 for worker in report.workers if not worker.problems)
    lines.append(f"{healthy}/{len(report.workers)} workers healthy")
    if report.problems:
        lines.append("")
        lines.extend(f"PROBLEM: {problem}" for problem in report.problems)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Miles worker readback files for one integration plan."
    )
    parser.add_argument("directory", help="directory holding <framework>-<target>-<pid>.json")
    parser.add_argument("--expect-workers", type=int, default=None)
    parser.add_argument("--target", default="training")
    parser.add_argument(
        "--require-execution",
        action="store_true",
        help="also require an execution record for every RL-Kernel route",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    report = audit_readbacks(
        args.directory,
        expected_workers=args.expect_workers,
        target=args.target,
        require_execution=args.require_execution,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_table(report))
    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
