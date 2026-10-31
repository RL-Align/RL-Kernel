#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicitly install one unmodified MIT-licensed upstream source, never at runtime."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

COMMIT = "326cffa38c9eda16bef19375ba55334e4d3acd59"
SHA256 = "56db3b46c58f8fb30912d4564189b214e2c9a37fa2138a155b45a4ce51b5c844"
SOURCE = f"https://raw.githubusercontent.com/ROCm/aiter/{COMMIT}/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        data = args.output.read_bytes()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise SystemExit("existing source has the wrong hash; refusing to overwrite it")
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise SystemExit("upstream hash mismatch; nothing was written")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(data)
    print(f"Verified unmodified aiter source: {args.output.resolve()}")
    print(f"RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE={args.output.resolve()}")


if __name__ == "__main__":
    main()
