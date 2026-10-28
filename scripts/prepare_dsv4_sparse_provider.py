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
WRAPPER_SHA256 = "ede08814720b9c783c7375cab9f95ee38bf6212ca77ba23149e33309b72842ff"
WRAPPER_SOURCE = f"https://raw.githubusercontent.com/ROCm/aiter/{COMMIT}/aiter/ops/triton/attention/sparse_mla_dsv4_train.py"


def provision(output, source, expected_hash, environment_name):
    if output.exists():
        data = output.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise SystemExit("existing source has the wrong hash; refusing to overwrite it")
    else:
        with urllib.request.urlopen(source, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise SystemExit("upstream hash mismatch; nothing was written")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(data)
    print(f"Verified unmodified aiter source: {output.resolve()}")
    print(f"{environment_name}={output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backward-wrapper-output", type=Path)
    args = parser.parse_args()
    if (
        args.backward_wrapper_output is not None
        and args.output.resolve() == args.backward_wrapper_output.resolve()
    ):
        raise SystemExit("kernel and wrapper outputs must be different files")
    provision(args.output, SOURCE, SHA256, "RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE")
    if args.backward_wrapper_output is not None:
        provision(
            args.backward_wrapper_output,
            WRAPPER_SOURCE,
            WRAPPER_SHA256,
            "RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE",
        )


if __name__ == "__main__":
    main()
