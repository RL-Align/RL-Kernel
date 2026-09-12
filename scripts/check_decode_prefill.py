#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C6 (#272): direct decode–prefill GPU gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest.accelerator import (  # noqa: E402
    disable_tf32,
    resolve_device,
)
from rl_engine.kernels.gtest.kv_consistency import (  # noqa: E402
    assert_decode_prefill_consistent,
    build_decode_prefill_cases,
)
from rl_engine.kernels.gtest.tolerance import load_contract  # noqa: E402
from rl_engine.testing.ws1_workload import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WS1 C6 direct decode-prefill gate")
    parser.add_argument(
        "--backend-profile",
        choices=("cuda_bf16", "triton_cuda_bf16", "ascend_bf16"),
        required=True,
    )
    parser.add_argument("--candidate", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to the backend profile's own accelerator (cuda or npu).",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device, profile=args.backend_profile)
    except RuntimeError as exc:
        print(f"ERROR: C6 declared-candidate gate needs a real device: {exc}", file=sys.stderr)
        return 2
    disable_tf32(device.type)
    contract = load_contract()
    manifest = load_manifest()
    report = assert_decode_prefill_consistent(
        backend_profile=args.backend_profile,
        candidate=args.candidate,
        contract=contract,
        manifest=manifest,
        device=device,
        require_declared_candidate=True,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"profile={report.backend_profile} candidate={report.candidate_id} "
            f"passed={report.passed} device={report.device} cc={report.compute_capability}"
        )
        for cell in report.cells:
            print(
                f"  {cell.case_id} attn_pass={cell.attention_compare.passed} "
                f"max_abs={cell.attention_compare.max_abs_error:.8e} "
                f"logp_pass={cell.logprob_verdict.passed}"
            )
        print(f"  cases={len(build_decode_prefill_cases(manifest))} (all include direct decode)")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
