# Qwen3-Next TP4 mixers: real-weight gates and attention prior art (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-tp4-mixers-b200.zip).

ZIP SHA-256: `0976d8e99ce454a242c9c4cfc88dc979b8679a47441ed5fb2cb1fa4ddf1e8d4b`.

The archive holds the TP4 gate reports `tp4-{gdn,attention,moe}/rank-{0..3}.json`,
the attention prior-art `attention/report.json` and `attention/figure.png` and
the original README, byte for byte as measured, plus `SHA256SUMS`. Every file
records its commit and `tracked_tree_dirty: false`; each run used a clean clone
with `rl_engine._C` built in place, on B200 (torch 2.13.0+cu130, vLLM 0.30.0,
NCCL 2.29.7) and the official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` (GDN and MoE gates: layer 0;
attention gate: layer 3).

| Files | Measured at | Command (current layout) |
| --- | --- | --- |
| `tp4-gdn/` | `fce346d` | `torchrun --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_tp_gdn_check.py --checkpoint <ckpt> --output tp4-gdn` |
| `tp4-attention/` | `fce346d` | `torchrun --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_tp_attention_check.py --checkpoint <ckpt> --output tp4-attention` |
| `tp4-moe/` | `fce346d` | `TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_tp_moe_check.py --checkpoint <ckpt> --output tp4-moe` |
| `attention/` | `137d0fb` | `python benchmarks/models/qwen3_next/qwen3_next_attention_prior_art.py --only <all but te_fused> --out report-main.json`, then `--only te_fused --merge report-main.json --out report.json`; `python benchmarks/models/qwen3_next/plot_qwen3_next_attention_prior_art.py report.json` |

At those commits the runners lived in `tools/validation/models/`; the
attention prior-art runner is now part of the Qwen3-Next benchmark PR.

## TP4 gates

GDN (layer 0, 7 cases), full attention (layer 3, 10 cases) and MoE (layer 0,
12 cases) pass on all four ranks: HF load/export round trip, TP4 equal to the
TP1 reference where the contract requires it, chunked == full prefill, token
reorder, training forward == no-grad forward, finite non-zero gradients.

## Attention prior art (one TP4 rank: 4 query heads, 1 KV head, D=256, causal)

| Candidate | Batch | Prefill == decode | Backward batch | Max abs vs FP64 | Prefill 2048 / 8192 | Fwd+bwd 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| **RL-Kernel deterministic attention** | **yes** | **yes** | **yes** | 6.7e-3 | 10.1 / 159.8 ms | 25.0 ms |
| torch SDPA | yes | no | no (dq) | 6.7e-3 | 0.12 / 0.61 ms | 0.74 ms |
| vLLM FlashAttention-2, `num_splits=0` | yes | no (last row) | n/a | 6.7e-3 | 0.15 / 0.66 ms | n/a |
| vLLM FlashAttention-2, `num_splits=1` | yes | yes | n/a | 6.7e-3 | 0.16 / 0.66 ms | n/a |
| vLLM Triton unified attention | yes | no (last row) | n/a | 6.7e-3 | 0.32 / 1.13 ms | n/a |
| FlashInfer ragged prefill | no | no | n/a | 6.7e-3 | 0.29 / 0.76 ms | n/a |
| FlashAttention-4 (CuTe DSL, vendored by vLLM) | yes | yes | yes | 6.7e-3 | 0.21 / 0.31 ms | 0.91 ms |
| Transformer Engine, training (unfused fallback) | no | fails | no | 1.3e-2 | 5.5 / 22.7 ms | 10.3 ms |
| Transformer Engine, inference (cuDNN fused) | yes | yes | n/a (no D=256 backward) | 6.7e-3 | 0.32 / 0.61 ms | n/a |
| Megatron-core `DotProductAttention` (local) | yes | yes | yes | 1.3e-2 | 0.32 / 3.2 ms | 1.34 ms |

- Batch: a 1000-token target alone vs packed first and last with 777-, 1500-
  and 64-token sequences. Prefill == decode: the last 64 rows and the last row
  computed against the full KV vs the same rows of the full prefill.
  Backward batch: the target's dq/dk/dv alone vs packed.
- FlashAttention-4's `deterministic=True` backward asserts for head dim 256 on
  SM100; the row above is its default backward, measured as batch invariant.
- The ZIP's README records the two-process setup (the fused TE row needs its
  own process) and the training-environment versions.

Site paths, host names and the scheduler do not appear in any file.
