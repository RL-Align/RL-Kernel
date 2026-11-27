# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare RL-Kernel's DSv4 attention ops with the vLLM fork's (``ISEEKYAN/vllm@1a6542c``) byte
for byte, at the real head shapes:

- ``q_norm_rope_kv_insert`` vs ``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``
  (weightless per-head RMSNorm + GPT-J RoPE of Q, padded heads; RoPE + UE8M0 FP8 quant + paged
  ``fp8_ds_mla`` insert of KV): the Q output and the cache bytes.

    PYTHONPATH=. python scripts/probe_dsv4_vllm_attention_ops.py
"""

from __future__ import annotations

import torch


def main() -> int:
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    from rl_engine.kernels.ops.cuda.attention import dsv4_attention as core
    from rl_engine.models.deepseek_v4.config import Dsv4Config
    from rl_engine.models.deepseek_v4.rope import rope_cos_sin

    torch.manual_seed(0)
    dev = "cuda"
    heads, padded, dim, block, eps = 64, 64, 512, 64, 1e-6
    cfg = Dsv4Config(
        rope_scaling={
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
            "type": "yarn",
        },
        compress_ratios=(0,) * 43,
    )
    ok = True
    for ratio in (1, 4):
        cos_sin = rope_cos_sin(cfg, ratio, 8192, dev)
        for t in (1, 7, 64, 333, 1024):
            q = (torch.randn(t, heads, dim, device=dev) * 3).to(torch.bfloat16)
            kv = (torch.randn(t, dim, device=dev) * 2).to(torch.bfloat16)
            pos = torch.randint(0, 8000, (t,), device=dev, dtype=torch.int64)
            nblocks = t // block + 4
            slots = torch.randperm(nblocks * block, device=dev)[:t].to(torch.int64)
            ours_cache = torch.zeros(nblocks, block * 584, dtype=torch.uint8, device=dev)
            vllm_cache = torch.zeros_like(ours_cache)
            q_ours = core.q_norm_rope_kv_insert(
                q, kv, pos, cos_sin, ours_cache, block, slots, padded, eps
            )
            q_vllm = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q, kv, vllm_cache, slots, pos, cos_sin, padded, eps, block
            )
            q_same = torch.equal(q_ours.view(torch.int16), q_vllm.view(torch.int16))
            c_same = torch.equal(ours_cache, vllm_cache)
            qd = (q_ours.float() - q_vllm.float()).abs().max().item()
            rows_bad = int(
                (q_ours.view(torch.int16) != q_vllm.view(torch.int16)).flatten(1).any(1).sum()
            )
            print(
                f"ratio {ratio} T {t:5d}: q {'same' if q_same else 'DIFF'} "
                f"({rows_bad} rows, max {qd:.3e}), cache {'same' if c_same else 'DIFF'}"
            )
            ok &= q_same and c_same
    print("PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
