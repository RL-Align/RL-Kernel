# SPDX-License-Identifier: Apache-2.0
"""INK-68: command-line gate for rocm recorded BF16 GRPO evidence."""

import argparse
import json
from pathlib import Path

from rl_engine.validation.models.inkling2.e2e_acceptance import validate_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    p.add_argument("strict", type=Path)
    p.add_argument("--steps", type=int, choices=(1, 10, 100, 200), required=True)
    args = p.parse_args()
    result = validate_pair(json.loads(args.native.read_text()),
                           json.loads(args.strict.read_text()),
                           platform="rocm", steps=args.steps)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["strict_zero_mismatch"] else 1)


if __name__ == "__main__":
    main()
