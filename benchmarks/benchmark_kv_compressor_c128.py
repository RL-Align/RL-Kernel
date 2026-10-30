# SPDX-License-Identifier: Apache-2.0
"""MI300X benchmark for the fixed aiter C128 forward boundary."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_engine.backends.rocm.attention.kv_compressor_c128 import AiterC128Backend
from rl_engine.contracts.operators.kv_compressor_c128 import C128Contract
from rl_engine.models.deepseek_v4.kv_compressor_c128 import C128Compressor
from tests.models.deepseek_v4.c128_helpers import fixture


def timed(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = AiterC128Backend()
    operator = C128Compressor(C128Contract(), backend=backend)
    cases = []
    for groups in (1, 8, 32, 128):
        x, ids, ape, norm, cos, sin = fixture(groups * 128, device="cuda")
        bound = operator.bind(x, ids, version="v1")

        def forward(bound=bound, ape=ape, norm=norm, cos=cos, sin=sin):
            operator(bound, ape, norm, cos, sin, version="v1")

        cases.append(
            {
                "groups": groups,
                "tokens": groups * 128,
                "input_bytes": x.numel() * x.element_size(),
                "output_bytes": groups * 512 * 2,
                "bind_us": timed(
                    lambda x=x, ids=ids: operator.bind(x, ids, version="v1"), args.repeats
                ),
                "forward_us": timed(forward, args.repeats),
                "pool_us": timed(
                    lambda bound=bound, ape=ape, norm=norm, cos=cos, sin=sin: backend.forward(
                        bound, ape, norm, cos, sin, operator.contract
                    ),
                    args.repeats,
                ),
            }
        )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": __import__("triton").__version__,
        "backend": backend.provenance,
        "contract": operator.contract.fingerprint,
        "implementation": operator.implementation_fingerprint,
        "metric": (
            "median unavailable; synchronized average over warmup+repeated loops, Python included"
        ),
        "scope": "complete C128 forward; aiter pool and norm/RoPE; no cache commit, no full model",
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
