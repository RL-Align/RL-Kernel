"""python -m rl_engine.p3: T01 manifest/recordings/check_p3/verify entry points."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .artifact import create, verify
from .checker import check_anchor
from .contract import P3Error, P3OpCtxHost, P3Verdict, manifest
from .fixtures import catalog, fixture_manifest
from .recordings import record_case
from .serialization import encode


def cuda_slice(*, state_dir=None, run_id="p3-gpu", engine_id="cuda", rank=0):
    import torch

    from .provider import InvocationAllocator
    from .stable_topk6 import CudaTopKProvider

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise P3Error(P3Verdict.UNSUPPORTED_CAPABILITY, "CUDA/H100 unavailable; no fallback")
    records = []
    from .serialization import fingerprint

    directory = Path(
        state_dir or os.environ.get("P3_STATE_DIR", str(Path.home() / ".local/state/rl-kernel/p3"))
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    scope = [run_id, engine_id, rank]
    journal = directory / (fingerprint(scope) + ".json")
    # The execution lock prevents another runner superseding this attempt mid-slice.
    import fcntl

    with journal.with_suffix(".execution.lock").open("a+") as execution_lock:
        fcntl.flock(execution_lock, fcntl.LOCK_EX)
        allocator = InvocationAllocator.new_attempt(journal, *scope)
        for case in catalog():
            recording = record_case(case)
            if recording["producer_verdict"] != "PASS":
                continue
            q = recording["operators"]["stable_topk6_fwd"]["inputs"][0]
            ctx = P3OpCtxHost(
                run_id,
                engine_id,
                rank,
                case.row_active.copy(),
                backend_tag="cuda",
                allocator=allocator,
            )
            tensor = torch.as_tensor(q, device="cuda:0")
            baseline = None
            for block in (1, 32, 64, 128):
                result = CudaTopKProvider().stable_topk6_fwd(ctx, tensor, block_size=block)
                if result.verdict != P3Verdict.PASS:
                    raise P3Error(result.verdict, "CUDA Stable Top-6 slice did not pass")
                ids = result.payload["ids"].cpu().numpy()
                expected = recording["operators"]["stable_topk6_fwd"]["payload"]["ids"]
                if not np.array_equal(ids[case.row_active], expected[case.row_active]):
                    raise P3Error(P3Verdict.TOPK_ORDER_MISMATCH, "CUDA ids disagree with oracle")
                if baseline is not None and not np.array_equal(ids, baseline):
                    raise P3Error(P3Verdict.TOPK_ORDER_MISMATCH, "launch invariance failed")
                baseline = ids.copy()
                records.append(
                    {
                        "case_id": case.case_id,
                        "verdict": "PASS",
                        "ids": ids,
                        "provenance": result.provenance,
                    }
                )
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("manifest", "catalog", "recordings", "negative-fixtures", "check_p3", "verify"),
    )
    parser.add_argument("path", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--operator")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-ws1", action="store_true")
    parser.add_argument(
        "--state-dir", help="persistent CUDA invocation journals (never put inside sealed artifact)"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            result = manifest()
        elif args.command == "catalog":
            result = fixture_manifest()
        elif args.command == "recordings":
            result = [
                {
                    "case_id": c.case_id,
                    "operators": {
                        name: op
                        for name, op in record_case(c)["operators"].items()
                        if args.operator is None or args.operator == name
                    },
                }
                for c in catalog()
            ]
        elif args.command == "negative-fixtures":
            from .negative import run_negative_fixtures

            result = run_negative_fixtures()
        elif args.command == "verify":
            if not args.path:
                parser.error("verify requires an artifact directory")
            artifact = verify(args.path)
            result = {
                "integrity": "PASS",
                "cpu_replay": "PASS",
                "cases": len(artifact["recordings"]),
                "evidence_matrix": artifact["evidence_matrix"],
            }
        else:
            if not args.output:
                parser.error("check_p3 requires --output")
            if not args.resume and Path(args.output).exists():
                raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "new attempts require a fresh directory")
            hardware = (
                cuda_slice(state_dir=args.state_dir)
                if args.backend == "cuda" and not args.resume
                else None
            )
            artifact = create(
                args.output, backend=args.backend, resume=args.resume, hardware=hardware
            )
            result = {
                "scope": "T01_START_KIT",
                "verdict": "CASE_PASS",
                "cases": len(artifact["recordings"]),
                "operator_recordings": sum(len(r["operators"]) for r in artifact["recordings"]),
                "evidence_matrix": artifact["evidence_matrix"],
                "artifact": args.output,
                "resume": args.resume,
                "first_mismatch": None,
                "ResultCube": [
                    {
                        "case_id": r["case_id"],
                        "scope": "T01_START_KIT",
                        "verdict": (
                            "CASE_PASS"
                            if r["producer_verdict"] == "PASS"
                            else r["producer_verdict"]
                        ),
                    }
                    for r in artifact["recordings"]
                ],
            }
            if args.require_ws1:
                check_anchor()
                raise P3Error(
                    P3Verdict.UPSTREAM_EVIDENCE_MISSING,
                    "other owners' CUDA evidence required for WS1",
                )
        print(json.dumps(encode(result), indent=2))
        return 0
    except P3Error as exc:
        print(json.dumps({"verdict": exc.verdict.name, "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
