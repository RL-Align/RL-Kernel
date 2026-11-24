# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Byte-compare indexer_q_rope_fp8 (golden and CUDA) with vLLM's CuTe-DSL kernel.

vLLM's CUDA rollout quantizes the DSv4 indexer query with
``fused_indexer_q_rope_quant_fp8_cutedsl`` (``IndexerQFp8Kernel``) whenever
``nvidia-cutlass-dsl`` is installed, which the vLLM CUDA build requires. Run
this in an environment where
``vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl`` imports (a vLLM
install from ``ISEEKYAN/vllm@1a6542c``, or that file plus ``vllm.cute_utils`` on
``PYTHONPATH`` with ``nvidia-cutlass-dsl==4.6.2`` and ``quack-kernels==0.6.4``).

It compares q_fp8 and weights_out bytes on random queries at several
magnitudes (including rows under the 1e-4 amax floor), both rotary-cache
dtypes, decode-sized and prefill-sized batches, and positions up to 64K.

Usage:
    python scripts/check_indexer_q_vllm_parity.py
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    try:
        from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
            fused_indexer_q_rope_quant_fp8_cutedsl as vllm_kernel,
        )
    except ImportError:
        from vllm.models_ref.fused_indexer_q_cutedsl import (  # local shim layout
            fused_indexer_q_rope_quant_fp8_cutedsl as vllm_kernel,
        )
    from rl_engine.kernels.ops.cuda.indexer import indexer_q_rope_fp8_cuda
    from rl_engine.kernels.ops.pytorch.indexer import indexer_q_rope_fp8

    heads, dim, rope, max_pos = 64, 128, 64, 1 << 16
    softmax_scale, head_scale = dim**-0.5, heads**-0.5
    inv = 1.0 / (10000 ** (torch.arange(0, rope, 2).double() / rope))
    angles = torch.arange(max_pos).double()[:, None] * inv[None]
    table = torch.cat([angles.cos(), angles.sin()], -1)

    bad = total = 0
    for cs_dtype in (torch.float32, torch.bfloat16):
        cache = table.to(cs_dtype).cuda()
        for tokens in (1, 7, 511, 512, 3000):
            for scale in (1.0, 1e-3, 1e-7, 40.0, 5e3):
                gen = torch.Generator().manual_seed(tokens * 31 + int(scale * 1e3) % 1000)
                q = (
                    (torch.randn(tokens, heads, dim, generator=gen) * scale)
                    .to(torch.bfloat16)
                    .cuda()
                )
                pos = torch.randint(0, max_pos, (tokens,), generator=gen).cuda()
                w = torch.randn(tokens, heads, generator=gen).to(torch.bfloat16).cuda()
                ref_q = torch.empty(tokens, heads, dim, dtype=torch.float8_e4m3fn, device="cuda")
                ref_w = torch.empty(tokens, heads, dtype=torch.float32, device="cuda")
                vllm_kernel(pos, q, cache, w, softmax_scale, head_scale, ref_q, ref_w)
                outs = {
                    "golden": indexer_q_rope_fp8(q, pos, cache, w, softmax_scale, head_scale),
                    "cuda": indexer_q_rope_fp8_cuda(q, pos, cache, w, softmax_scale, head_scale),
                }
                total += 1
                failed = [
                    name
                    for name, (oq, ow) in outs.items()
                    if not (
                        torch.equal(oq.view(torch.uint8), ref_q.view(torch.uint8))
                        and torch.equal(ow.view(torch.int32), ref_w.view(torch.int32))
                    )
                ]
                if failed:
                    bad += 1
                    print(f"MISMATCH cache={cs_dtype} tokens={tokens} scale={scale}: {failed}")
    print(f"vLLM CuTe indexer-Q parity: {total - bad}/{total} cases byte-equal")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
