# SPDX-License-Identifier: Apache-2.0
"""MI300X benchmark for the fixed aiter C4 overlap forward boundary."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

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
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return {
        "event_us": start.elapsed_time(end) * 1e3 / repeats,
        "wall_us": (time.perf_counter() - wall_start) * 1e6 / repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("positive repeats and an MI300X ROCm device are required")
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
        "date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "time_zone": "Asia/Shanghai",
        "gpu": torch.cuda.get_device_name(),
        "gpu_name_note": (
            "This ROCm container returns an empty torch.cuda.get_device_name(); "
            "gcnArchName identifies the observed gfx942 device."
        ),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": __import__("triton").__version__,
        "backend": backend.provenance,
        "contract": operator.contract.fingerprint,
        "implementation": operator.implementation_fingerprint,
        "metric": (
            "Warmed repetitions; same-pass HIP event and synchronized wall averages. "
            "Event intervals include host/launch gaps, not isolated kernel time."
        ),
        "repeats": args.repeats,
        "benchmark_validity": (
            "Bind, checked forward, and raw provider calls have distinct scopes. "
            "Do not subtract provider from checked forward as dispatch overhead; "
            "no baseline engine speedup is claimed."
        ),
        "scope": (
            "complete C4 forward; aiter overlap pool and norm/RoPE; no live cache "
            "commit, no full model"
        ),
        "cases": cases,
    }
    sources = (
        "benchmarks/benchmark_kv_compressor_c4.py",
        "rl_engine/backends/rocm/attention/kv_compressor_c4.py",
        "rl_engine/contracts/operators/kv_compressor_c4.py",
        "rl_engine/models/deepseek_v4/kv_compressor_c4.py",
    )
    root = Path(__file__).resolve().parents[1]
    result["tested_sources"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
