#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1 start-kit acceptance command (issue #2, ``P1-S0``).

Runs a provider's operators through the frozen mHC + RMSNorm block and
compares every operator boundary byte-for-byte against the FP32 oracle
executed on the same device. Any mismatching strict boundary fails the run.

Examples:
    python scripts/check_p1.py
    python scripts/check_p1.py --provider mypkg.p1:CudaMHCProvider --device cuda
    python scripts/check_p1.py --cases packed_t16,fused_pre_norm --json out.json
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

from rl_engine.mhc import fixtures, oracle  # noqa: E402
from rl_engine.mhc.contract import tensor_sha256  # noqa: E402
from rl_engine.mhc.provider import (  # noqa: E402
    MHCProvider,
    check_capability,
    resolve_provider,
)
from rl_engine.mhc.trace import MHCTrace  # noqa: E402


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


def _hashes(trace: MHCTrace, grads: dict[str, Any], r_new: Any) -> dict[str, str]:
    out = trace.hashes()
    for key, grad in grads.items():
        if grad is not None:
            out[f"grad.{key}"] = tensor_sha256(grad)
    out["r_new"] = tensor_sha256(r_new)
    return out


def _run_block(provider: MHCProvider, name: str, device: str) -> list[dict[str, Any]]:
    batch = fixtures.make_batch(name).to(device)
    check_capability(provider, batch.contract)
    grads = fixtures.make_grads(name, fixtures.make_batch(name)).to(device)

    gold_trace = MHCTrace(numeric_profile="oracle")
    r_gold, saved_gold = oracle.mhc_block_forward(batch, gold_trace)
    grads_gold = oracle.mhc_block_backward(batch, saved_gold, grads, gold_trace)

    cand_trace = MHCTrace(numeric_profile=provider.numeric_profile)
    r_cand, saved_cand = oracle.mhc_block_forward(batch, cand_trace, ops=provider)
    grads_cand = oracle.mhc_block_backward(batch, saved_cand, grads, cand_trace, ops=provider)

    return _compare(
        _hashes(gold_trace, grads_gold, r_gold), _hashes(cand_trace, grads_cand, r_cand)
    )


def _run_batch_invariance(provider: MHCProvider, name: str, device: str) -> list[dict[str, Any]]:
    """Same row, different batch: bytes must not move (issue #2 acceptance)."""
    batch = fixtures.make_batch(name).to(device)
    full, _ = oracle.mhc_block_forward(batch, ops=provider)
    rows = []
    for row in range(batch.tokens):
        single = fixtures.slice_batch(batch, row, row + 1)
        one, _ = oracle.mhc_block_forward(single, ops=provider)
        ok = bool((one[0] == full[row]).all())
        rows.append(
            {
                "boundary": f"batch_invariance.row{row}",
                "ok": ok,
                "golden": "packed",
                "got": "one-row" if ok else "DIVERGED",
            }
        )
    return rows


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
    names = list(fixtures.BLOCK_CASES)
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - set(names)
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        names = [n for n in names if n in wanted]

    report: dict[str, Any] = {
        "provider": provider.name,
        "device": args.device,
        "provenance": provider.provenance(),
        "capabilities": provider.capabilities(),
        "cases": {},
    }
    failed = False
    for name in names:
        try:
            rows = _run_block(provider, name, args.device)
            rows += _run_batch_invariance(provider, name, args.device)
        except NotImplementedError as exc:
            rows = [
                {"boundary": "<all>", "ok": False, "golden": "", "got": f"NotImplemented: {exc}"}
            ]
        report["cases"][name] = rows
        case_ok = all(r["ok"] for r in rows)
        failed = failed or not case_ok
        print(f"[{'PASS' if case_ok else 'FAIL'}] {name}")
        for r in rows:
            print(
                f"{'  ok ' if r['ok'] else '  XX '} {r['boundary']:<28} "
                f"golden={r['golden']} got={r['got']}"
            )

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
