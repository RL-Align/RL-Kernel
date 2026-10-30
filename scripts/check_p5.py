#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5 start-kit acceptance command (issue #8, ``P5-S0``).

Runs a provider's operators through the frozen routed/shared pipelines and
compares every operator boundary byte-for-byte against the FP32 oracle
executed on the same device. Any mismatching strict boundary fails the run.

Examples:
    python scripts/check_p5.py
    python scripts/check_p5.py --provider mypkg.p5:CudaP5Provider --device cuda
    python scripts/check_p5.py --cases base_plus_lora,uneven_experts --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.moe import fixtures, oracle  # noqa: E402
from rl_engine.moe.contract import tensor_sha256  # noqa: E402
from rl_engine.moe.provider import ExpertProvider, resolve_provider  # noqa: E402
from rl_engine.moe.trace import ExpertTrace  # noqa: E402


def _compare(golden: dict[str, str], candidate: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for name, want in golden.items():
        got = candidate.get(name)
        rows.append(
            {
                "boundary": name,
                "ok": got == want,
                "golden": want[:12],
                "got": (got or "<missing>")[:12],
            }
        )
    return rows


def _run_e2e(provider: ExpertProvider, name: str, device: str) -> list[dict[str, Any]]:
    batch = fixtures.make_expert_batch(name).to(device)
    gold_trace = ExpertTrace(numeric_profile="oracle")
    y_gold, saved_gold = oracle.routed_expert_forward(batch, gold_trace)
    dy = fixtures.make_grad_output(name, tuple(y_gold.shape)).to(device)
    grads_gold = oracle.routed_expert_backward(batch, saved_gold, dy, gold_trace)

    cand_trace = ExpertTrace(numeric_profile=provider.numeric_profile)
    y_cand, saved_cand = oracle.routed_expert_forward(batch, cand_trace, ops=provider)
    grads_cand = oracle.routed_expert_backward(batch, saved_cand, dy, cand_trace, ops=provider)

    golden = gold_trace.hashes()
    candidate = cand_trace.hashes()
    for key, grad in grads_gold.items():
        if grad is not None:
            golden[f"grad.{key}"] = tensor_sha256(grad)
    for key, grad in grads_cand.items():
        if grad is not None:
            candidate[f"grad.{key}"] = tensor_sha256(grad)
    return _compare(golden, candidate)


def _run_shared(provider: ExpertProvider, name: str, device: str) -> list[dict[str, Any]]:
    batch = fixtures.make_shared_batch(name).to(device)
    y_gold, saved_gold = oracle.shared_expert_mlp_fwd(batch)
    dy = fixtures.make_grad_output(name, tuple(y_gold.shape)).to(device)
    dx_gold = oracle.shared_expert_mlp_bwd(dy, batch, saved_gold)
    y_cand, saved_cand = provider.shared_expert_mlp_fwd(batch)
    dx_cand = provider.shared_expert_mlp_bwd(dy, batch, saved_cand)
    golden = {"shared_out": tensor_sha256(y_gold), "grad.dx": tensor_sha256(dx_gold)}
    candidate = {"shared_out": tensor_sha256(y_cand), "grad.dx": tensor_sha256(dx_cand)}
    return _compare(golden, candidate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider", default="reference", help="'reference', 'stub', or module.path:ClassName"
    )
    parser.add_argument("--cases", default=None, help="comma-separated case names (default: all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", dest="json_path", default=None, help="write full report as JSON")
    args = parser.parse_args()

    provider = resolve_provider(args.provider)
    e2e_names = list(fixtures.E2E_CASES)
    shared_names = list(fixtures.SHARED_CASES)
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - set(e2e_names) - set(shared_names)
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        e2e_names = [n for n in e2e_names if n in wanted]
        shared_names = [n for n in shared_names if n in wanted]

    report: dict[str, Any] = {
        "provider": provider.name,
        "device": args.device,
        "provenance": provider.provenance(),
        "cases": {},
    }
    failed = False
    for name in e2e_names + shared_names:
        runner = _run_e2e if name in fixtures.E2E_CASES else _run_shared
        try:
            rows = runner(provider, name, args.device)
        except NotImplementedError as exc:
            rows = [
                {"boundary": "<all>", "ok": False, "golden": "", "got": f"NotImplemented: {exc}"}
            ]
        report["cases"][name] = rows
        case_ok = all(r["ok"] for r in rows)
        failed = failed or not case_ok
        status = "PASS" if case_ok else "FAIL"
        print(f"[{status}] {name}")
        for r in rows:
            mark = "  ok " if r["ok"] else "  XX "
            print(f"{mark} {r['boundary']:<24} golden={r['golden']} got={r['got']}")

    print(f"\nprovider={provider.name} profile={provider.numeric_profile} device={args.device}")
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"report written to {args.json_path}")
    print(
        "RESULT:",
        "FAIL (strict boundaries diverged)" if failed else "PASS (all boundaries byte-equal)",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
