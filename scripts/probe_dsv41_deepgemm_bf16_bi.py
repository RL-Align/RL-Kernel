# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM's BF16 GEMM for batch invariance at the DSv4.1 shapes (SM90).

Under design decision D12 (docs/architecture/dsv41-design.md 4.2) DSv4.1's MXFP8 dense
weights are dequantized once to BF16 and every dense projection runs on DeepGEMM
``bf16_gemm_nt``, so the DSv4 probe (``probe_dsv4_deepgemm_bf16_bi.py``: rows of an
M = 4096 batch against prefixes, single rows and a window, and run to run) is repeated at
H = 5120. ``wo_a`` / ``wo_b`` assume DSv4's ``o_groups = 8`` and ``o_lora_rank = 1024``
until the checkpoint's config says otherwise.

Usage:
    python scripts/probe_dsv41_deepgemm_bf16_bi.py [--output report.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch

_base_path = pathlib.Path(__file__).with_name("probe_dsv4_deepgemm_bf16_bi.py")
_spec = importlib.util.spec_from_file_location("probe_dsv4_deepgemm_bf16_bi", _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

F32, BF16 = torch.float32, torch.bfloat16
SHAPES = (  # (name, N, K, output dtype)
    ("wq_a", 1280, 5120, F32),
    ("wkv", 512, 5120, F32),
    ("wq_b", 32768, 1280, F32),
    ("wo_a_group", 1024, 4096, F32),
    ("wo_b", 5120, 8192, F32),
    ("shared_fc1", 4608, 5120, F32),
    ("shared_fc2", 5120, 2304, F32),
    ("indexer_wq_b", 4096, 1280, F32),
    ("indexer_weights_proj", 64, 5120, BF16),
    ("compressor_r2_wkv_wgate", 1024, 5120, F32),
    ("compressor_r1_wkv", 512, 5120, F32),
    ("indexer_wk", 128, 512, F32),
    ("moe_router_gate", 384, 5120, F32),
    ("engram_wkv", 25600, 6144, F32),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = []
    for name, n, k, out in SHAPES:
        for seed in (0, 1):
            r = base.probe(base._deepgemm, name, n, k, out, seed)
            rows.append(r)
            print(
                f"{name:26s} N={n:6d} K={k:5d} {str(out):14s} seed={seed} "
                f"{'PASS' if r['passed'] else 'FAIL ' + ','.join(r['failed'])} "
                f"rel={r['rel_err_vs_fp32']:.2e}",
                flush=True,
            )
    ok = all(r["passed"] for r in rows)
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps({"passed": ok, "rows": rows}, indent=1))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
