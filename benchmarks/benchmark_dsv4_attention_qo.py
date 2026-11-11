# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention q / o-side kernels against vLLM's default (non-invariant) path.

- q side: our ``q_norm_rope_kv_insert`` against vLLM's
  ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` (same work: per-head q RMSNorm + RoPE,
  kv RoPE + fp8_ds_mla insert).
- o side: our ``OutputProjection`` (inverse RoPE + FP8, 8 per-group DeepGEMM ``fp8_gemm_nt``
  for ``wo_a``, then ``wo_b``) against vLLM's ``fused_inv_rope_fp8_quant`` + batched DeepGEMM
  ``fp8_einsum`` + the same ``wo_b``. vLLM's batch-invariant mode on SM90 runs ``wo_a`` per
  group too (Triton); its default runs the batched einsum.

CUDA-graph replay, median, microseconds. Needs the vLLM fork in the environment.

    PYTHONPATH=. python benchmarks/benchmark_dsv4_attention_qo.py
"""

from __future__ import annotations

import statistics

import torch

HEADS, HEAD_DIM, HIDDEN, GROUPS, RANK, BLOCK = 64, 512, 4096, 8, 1024, 64
MAX_POS = 32768


def _graph_us(fn, iters: int = 20) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def main() -> None:
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
    from vllm.utils.deep_gemm import fp8_einsum

    from rl_engine.kernels.ops.cuda import dsv4_attention as core
    from rl_engine.kernels.ops.cuda.fp8_linear import Fp8Linear, quantize_weight

    g = torch.Generator(device="cuda").manual_seed(0)
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    ang = torch.arange(MAX_POS, device="cuda").float()[:, None] * inv[None]
    cos_sin = torch.cat([ang.cos(), ang.sin()], -1).contiguous()
    wo_a = quantize_weight(
        (
            torch.randn(GROUPS * RANK, HEADS // GROUPS * HEAD_DIM, generator=g, device="cuda")
            * 0.02
        ).bfloat16()
    )
    wo_b = Fp8Linear.from_bf16(
        (torch.randn(HIDDEN, GROUPS * RANK, generator=g, device="cuda") * 0.01).bfloat16()
    )
    ours_o = core.OutputProjection(wo_a, wo_b, GROUPS)
    w3 = wo_a.q.view(GROUPS, RANK, -1)
    s3 = wo_a.scale.view(GROUPS, RANK // 128, -1)

    def vllm_o(o, pos):
        o8, sc = fused_inv_rope_fp8_quant(
            o,
            pos,
            cos_sin,
            n_groups=GROUPS,
            heads_per_group=HEADS // GROUPS,
            nope_dim=HEAD_DIM - 64,
            rope_dim=64,
            tma_aligned_scales=False,
        )
        z = torch.empty(o.shape[0], GROUPS, RANK, dtype=torch.bfloat16, device="cuda")
        fp8_einsum("bhr,hdr->bhd", (o8, sc), (w3, s3), z, recipe=(1, 128, 128))
        return wo_b(z.flatten(1))

    print("H200, microseconds (CUDA-graph replay, median); ratio = ours / vLLM default")
    print("     T |  q ours  q vLLM ratio |   o ours   o vLLM ratio | o rel diff")
    for t in (1, 64, 1024, 4096, 16384):
        pos = torch.randint(MAX_POS, (t,), generator=g, device="cuda")
        q = torch.randn(t, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
        kv = torch.randn(t, HEAD_DIM, generator=g, device="cuda").bfloat16()
        blocks = t // BLOCK + 2
        cache_a = torch.zeros(blocks, BLOCK * 584, dtype=torch.uint8, device="cuda")
        cache_b = torch.zeros_like(cache_a)
        slots = torch.randperm(blocks * BLOCK, generator=g, device="cuda")[:t]
        o = torch.randn(t, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()

        q_ours = _graph_us(
            lambda: core.q_norm_rope_kv_insert(
                q, kv, pos, cos_sin, cache_a, BLOCK, slots, HEADS, 1e-6
            )
        )
        q_vllm = _graph_us(
            lambda: torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q, kv, cache_b, slots, pos, cos_sin, HEADS, 1e-6, BLOCK
            )
        )
        o_ours = _graph_us(lambda: ours_o(o, pos, cos_sin))
        o_vllm = _graph_us(lambda: vllm_o(o, pos))
        a, b = ours_o(o, pos, cos_sin).float(), vllm_o(o, pos).float()
        rel = float((a - b).norm() / b.norm())
        print(
            f"{t:6d} | {q_ours:7.1f} {q_vllm:7.1f} {q_ours / q_vllm:5.2f} | "
            f"{o_ours:8.1f} {o_vllm:8.1f} {o_ours / o_vllm:5.2f} | {rel:.1e}"
        )


if __name__ == "__main__":
    main()
