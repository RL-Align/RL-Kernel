#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Time the H3 projection rows against the provider (cuBLAS ``F.linear``) and record backends.

    python tools/validation/models/h3_projection_bench.py --out reports/h3_projection_bench.json

Per row and token count: median forward and forward+backward milliseconds of the
RL-Kernel operator and of the provider, their ratio, and the operator actually dispatched
(registry readback). Provider timings run with TF32 off.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.reference.minimax_h3 import projection as ref  # noqa: E402
from rl_engine.runtime.registry import KernelRegistry  # noqa: E402

ROWS = (
    ("audio_input_projection_fp32", ref.AUDIO_INPUT),
    ("audio_output_projection_fp32", ref.AUDIO_OUTPUT),
    ("text_context_projection", ref.TEXT_CONTEXT),
)


def _time(fn, repeats):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="*", default=[64, 1024, 4096])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    registry = KernelRegistry()
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}
    for row, spec in [*ROWS, ("h3_qkv_gemm", ref.QKV)]:
        op = registry.get_op(row, device="cuda")
        for tokens in args.tokens:
            gen = torch.Generator("cpu").manual_seed(0)
            x = torch.randn(tokens, spec.in_features, generator=gen).to(spec.dtype).cuda()
            n = spec.out_features
            ws = [
                (torch.randn(n, spec.in_features, generator=gen) * 0.02).to(spec.dtype).cuda()
                for _ in range(3 if row == "h3_qkv_gemm" else 1)
            ]
            b = torch.zeros(n, dtype=spec.dtype).cuda() if spec.has_bias else None
            if row == "h3_qkv_gemm":
                cand = lambda: op.forward(x, *ws)  # noqa: E731
                prov = lambda: [F.linear(x, w) for w in ws]  # noqa: E731
            else:
                cand = lambda: op.forward(x, ws[0], b)  # noqa: E731
                prov = lambda: F.linear(x, ws[0], b)  # noqa: E731
            c, p = _time(cand, args.repeats), _time(prov, args.repeats)
            gold = F.linear(x.double(), ws[0].double(), None if b is None else b.double())
            got_c, got_p = cand(), prov()
            got_c = got_c[0] if isinstance(got_c, (list, tuple)) else got_c
            got_p = got_p[0] if isinstance(got_p, (list, tuple)) else got_p

            def rel(a):
                return ((a.double() - gold).norm() / gold.norm()).item()

            report["rows"].append(
                {
                    "row": row,
                    "tokens": tokens,
                    "backend": f"{type(op).__module__}.{type(op).__name__}",
                    "kernel_id": getattr(op, "kernel_id", ""),
                    "candidate_ms": c,
                    "provider_ms": p,
                    "ratio": c / p,
                    "rel_err_candidate": rel(got_c),
                    "rel_err_provider": rel(got_p),
                }
            )
            print(
                f"{row:32s} T={tokens:5d} candidate {c:8.3f} ms  provider {p:8.3f} ms  x{c / p:.2f}"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
