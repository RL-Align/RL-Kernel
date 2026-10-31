# SPDX-License-Identifier: Apache-2.0
"""MI300X benchmark for the fixed aiter C4 overlap forward boundary."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_engine.backends.rocm.attention.kv_compressor_c4 import AiterC4Backend
from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor
from tests.models.deepseek_v4.c4_helpers import fixture


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
    backend = AiterC4Backend()
    operator = C4Compressor(C4Contract(), backend=backend)
    cases = []
    for groups in (1, 8, 64, 128):
        values, ids, ape, norm, cos, sin = fixture(groups, device="cuda")
        bound = operator.bind(values, ids, version="v1")

        def forward(bound=bound, ape=ape, norm=norm, cos=cos, sin=sin):
            operator(bound, ape, norm, cos, sin, version="v1")

        cases.append(
            {
                "groups": groups,
                "tokens": groups * 4,
                "input_bytes": values.numel() * values.element_size(),
                "output_bytes": groups * 512 * 2,
                "bind_us": timed(
                    lambda values=values, ids=ids: operator.bind(values, ids, version="v1"),
                    args.repeats,
                ),
                "forward_us": timed(forward, args.repeats),
                "provider_forward_us": timed(
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
            "synchronized average over warmup+repeated loops; Python validation and "
            "paged-cache gather included"
        ),
        "scope": (
            "complete C4 forward; aiter overlap pool and norm/RoPE; no live cache "
            "commit, no full model"
        ),
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
