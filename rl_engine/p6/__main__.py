# SPDX-License-Identifier: Apache-2.0
"""Run P6 contracts, independent recordings and sealed reference conformance."""

import argparse
import json
from pathlib import Path
import platform
import sys

from .artifact import build_payload, publish, resume, verify
from .contract import CombinePlan, ContractError, manifest, require
from .fixtures import evaluate, make_case
from .mocks import ep_return
from .negative import negative_fixtures
from .oracle import forward
from .recordings import boundary_recordings, catalog, golden_records, verify_source_cases


def conformance(records, device, require_h100=False, graph=False):
    verify_source_cases(records)
    checks = []
    for record in records:
        c = record["input"]
        plan = CombinePlan.from_dict(c["plan"])
        for ep in (1, 2, 4, 8):
            for placement in range(min(ep, 2)):
                result = ep_return(plan, c["rows"], plan.context, ep=ep, placement=placement)
                actual = forward(plan, result["rows"], c["shared"], c["residual"], plan.context)
                require(
                    actual["stages"] == record["expected"]["forward"],
                    "BYTE_MISMATCH",
                    "mock EP return",
                )
        checks.append(
            {
                "case": c["name"],
                "status": "SCALAR_ORACLE_AND_MOCK_RETURN_PASS",
                "ep_layouts": [1, 2, 4, 8],
                "transport_executed": False,
            }
        )
    if device == "cpu":
        require(not graph and not require_h100, "UNSUPPORTED_CAPABILITY", "CPU is not GPU")
        provenance = {
            "provider": "stdlib-scalar-oracle",
            "python": platform.python_version(),
            "device": "cpu",
            "hardware_execution": False,
        }
    else:
        from .torch_reference import conformance as tensor_conformance

        provenance, device_checks = tensor_conformance(
            records, "cpu" if device == "torch-cpu" else device, require_h100, graph
        )
        checks += device_checks
    return provenance, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest")
    commands.add_parser("catalog")
    commands.add_parser("negative-fixtures")
    rec = commands.add_parser("recordings")
    rec.add_argument("--operator", choices=[op["name"] for op in manifest()["operators"]])
    check = commands.add_parser("verify")
    check.add_argument("directory")
    conf = commands.add_parser("conformance")
    conf.add_argument("--device", default="cpu", help="cpu, torch-cpu or cuda:N")
    conf.add_argument("--hidden-size", type=int, default=0)
    conf.add_argument("--require-h100", action="store_true")
    conf.add_argument("--graph", action="store_true")
    conf.add_argument(
        "--resume", action="store_true", help="reuse only matching sealed complete attempt"
    )
    conf.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            result = manifest()
        elif args.command == "catalog":
            result = catalog()
        elif args.command == "negative-fixtures":
            result = negative_fixtures()
        elif args.command == "recordings":
            result = [
                r
                for r in boundary_recordings()
                if not args.operator or r["operator"] == args.operator
            ]
        elif args.command == "verify":
            result = verify(args.directory)
        else:
            require(args.hidden_size >= 0, "UNSUPPORTED_GEOMETRY", "hidden-size")
            records = golden_records()
            if args.hidden_size:
                case = make_case(
                    f"width-{args.hidden_size}", n=2, h=args.hidden_size, mode="mixed", seed=31
                )
                records.append({"input": case, "expected": evaluate(case)})
            request = {
                "device": args.device,
                "hidden_size": args.hidden_size,
                "graph": args.graph,
                "require_h100": args.require_h100,
            }
            if args.resume:
                result = resume(args.output, records, request)
            else:
                require(
                    not Path(args.output).exists(), "OUTPUT_EXISTS", "use a fresh attempt directory"
                )
                provenance, checks = conformance(
                    records, args.device, args.require_h100, args.graph
                )
                payload = build_payload(records, provenance, checks, request)
                result = publish(args.output, payload)
                result.update(
                    cases=len(records),
                    operator_recordings=len(payload["operator_recordings"]),
                    provenance=provenance,
                )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ContractError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "code": getattr(exc, "code", type(exc).__name__),
                    "detail": str(exc),
                    "production_certified": False,
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
