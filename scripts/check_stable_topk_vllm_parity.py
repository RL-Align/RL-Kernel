# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Byte-compare stable_topk (golden and CUDA) with vLLM's DSv4 BI selector.

vLLM's ``csrc/ds4_bi_topk_extension.cu`` (``ISEEKYAN/vllm@1a6542c``) is a
standalone torch library, so it is JIT-compiled here from a checkout of that
file and run on the same inputs: random, few-valued (large tie groups),
ReLU-like (mostly zeros) and raw random bit patterns (every NaN payload, +-0,
denormals, infinities), over several widths, top-k sizes and ragged row ranges.

Usage (SM90, rl_engine._C built):
    python scripts/check_stable_topk_vllm_parity.py --vllm-ref /path/to/ds4_bi_topk_extension.cu
"""

from __future__ import annotations

import argparse
import sys

import torch


def _case(kind: str, rows: int, width: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    if kind == "randn":
        x = torch.randn(rows, width, generator=gen)
    elif kind == "few":
        x = torch.randint(-3, 4, (rows, width), generator=gen).float()
    elif kind == "relu":
        x = torch.relu(torch.randn(rows, width, generator=gen) - 1.5)
    else:
        bits = torch.randint(-(2**31), 2**31 - 1, (rows, width), generator=gen, dtype=torch.int64)
        x = bits.to(torch.int32).view(torch.float32)
    lengths = torch.randint(0, width + 1, (rows,), generator=gen)
    starts = (torch.rand(rows, generator=gen) * (width - lengths + 1)).long().clamp(min=0)
    ends = starts + lengths
    return x.cuda(), starts.int().cuda(), ends.int().cuda()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-ref", required=True, help="path to ds4_bi_topk_extension.cu")
    args = parser.parse_args()

    from torch.utils.cpp_extension import load

    load(
        name="ds4_bi_topk_ref",
        sources=[args.vllm_ref],
        is_python_module=False,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )
    from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda
    from rl_engine.kernels.ops.pytorch.topk import stable_topk_per_row

    bad = total = 0
    for kind in ("randn", "few", "relu", "bits"):
        for k in (1, 64, 512, 1024, 2048):
            for width in (100, 2049, 9000, 70000):
                x, starts, ends = _case(kind, 24, width, seed=k * 7 + width)
                ref = torch.empty(x.shape[0], k, dtype=torch.int32, device="cuda")
                torch.ops.ds4_bi.top_k_per_row_prefill(
                    x, starts, ends, ref, x.shape[0], x.stride(0), x.stride(1), k
                )
                golden = stable_topk_per_row(x, starts, ends, k)
                results = {"golden": golden}
                for threads in (0, 256, 512, 1024):
                    results[f"cuda{threads or ''}"] = stable_topk_per_row_cuda(
                        x, starts, ends, k, threads=threads
                    )
                total += 1
                failed = [name for name, out in results.items() if not torch.equal(out, ref)]
                if failed:
                    bad += 1
                    print(f"MISMATCH kind={kind} k={k} width={width}: {failed}")
    print(f"vLLM ds4_bi parity: {total - bad}/{total} cases byte-equal")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
