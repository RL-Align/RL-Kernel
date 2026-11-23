#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Print, write or check the MiniMax-H3 architecture fingerprint.

    python tools/validation/models/h3_fingerprint.py print
    python tools/validation/models/h3_fingerprint.py write          # refresh the golden file
    python tools/validation/models/h3_fingerprint.py check          # golden == recomputed
    python tools/validation/models/h3_fingerprint.py checkpoint DIR # local shard headers

``check`` exits non-zero on any difference, so it is usable as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import fingerprint as fp  # noqa: E402

GOLDEN = REPO_ROOT / "rl_engine/models/minimax_h3/arch_fingerprint.json"


def _dump(fingerprint: dict) -> str:
    return json.dumps(fingerprint, indent=1, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("print")
    sub.add_parser("write")
    sub.add_parser("check")
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("directory", type=Path)
    checkpoint.add_argument("--variant", default="FL2VA", choices=sorted(fp.VARIANTS))
    args = parser.parse_args()

    if args.command == "checkpoint":
        count = fp.validate_checkpoint_headers(args.directory, args.variant)
        print(f"ok: {count} tensors in {args.directory} match the {args.variant} fingerprint")
        return 0
    fingerprint = fp.build_fingerprint()
    if args.command == "print":
        sys.stdout.write(_dump(fingerprint))
    elif args.command == "write":
        GOLDEN.write_text(_dump(fingerprint))
        print(f"wrote {GOLDEN} ({fingerprint['fingerprint_sha256']})")
    else:
        if not GOLDEN.exists() or json.loads(GOLDEN.read_text()) != fingerprint:
            print("golden fingerprint differs from the recomputed one", file=sys.stderr)
            return 1
        print(f"ok: {fingerprint['fingerprint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
