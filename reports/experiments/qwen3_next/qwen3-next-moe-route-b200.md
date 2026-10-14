# Qwen3-Next routed MoE: prior art and TP4 gate (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-moe-route-b200.zip).

ZIP SHA-256: `9fcc704291bf697274edfe16d6958ca799134008acd9d5f452767b9a92758cb0`.

The archive holds `report.json`, `figure.png`, the TP4 gate's
`tp4-moe/rank-{0..3}.json` and the original README, byte for byte as measured,
plus `SHA256SUMS`. The prior art was measured at commit `c58816b` from a clean
clone on one B200 (torch 2.13.0+cu130, Triton 3.7.1); the TP4 gate at
`2e950f3`, whose MoE code is identical, on four B200. The prior art uses
random weights (scale 0.02) at the TP1 shape (H=2048, 512 experts, top-10,
width 512); the TP4 gate reads layer 0 of the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`.

```bash
# prior art (benchmark PR, benchmarks/models/qwen3_next/), training environment
TRITON_F32_DEFAULT=ieee python benchmarks/models/qwen3_next/qwen3_next_moe_prior_art.py \
    --only rl_kernel_cuda,hf_transformers,vllm_bi0,vllm_bi1,flashinfer_cutlass,megatron_te --out main.json
# SGLang environment
TRITON_F32_DEFAULT=ieee python benchmarks/models/qwen3_next/qwen3_next_moe_prior_art.py \
    --only sglang_triton,sglang_deterministic --merge main.json --out report.json
python benchmarks/models/qwen3_next/plot_qwen3_next_moe_prior_art.py report.json
# TP4 gate on real layer-0 weights (this PR)
TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_tp_moe_check.py \
    --checkpoint <checkpoint> --output tp4-moe
```

At `c58816b` / `2e950f3` the runners lived in `tools/validation/models/` and
`rl_engine/validation/models/`; the commands above use the current layout.

| Candidate | Forward batch invariant | Backward batch invariant | Different expert set (of 256 tokens) | Max abs vs FP64 | Forward, 1024 tokens |
| --- | --- | --- | --- | --- | --- |
| **RL-Kernel `shared_moe`** | **yes** | **yes** | **0** | 2.1e-3 | 2.04 ms |
| HF transformers `Qwen3NextExperts` (eager) | yes | no (1024) | 11 | 1.1e-1 | 89.3 ms |
| vLLM `fused_experts`, `VLLM_BATCH_INVARIANT=0` | no (256, 1024) | n/a (inference) | 11 | 1.1e-1 | 0.87 ms |
| vLLM `fused_experts`, `VLLM_BATCH_INVARIANT=1` | yes | n/a (inference) | 11 | 1.1e-1 | 0.87 ms |
| FlashInfer `cutlass_fused_moe` | no (256, 1024) | n/a (inference) | 11 | 1.1e-1 | 0.90 ms |
| Megatron-core `MoELayer` + TE grouped GEMM | no (1024) | no (1024) | 0 | 2.2e-3 | 11.7 ms |
| SGLang `fused_moe` (Triton), default | no (256, 1024) | n/a (inference) | 11 | 1.1e-1 | 0.83 ms |
| SGLang `fused_moe`, deterministic inference | yes | n/a (inference) | 11 | 1.1e-1 | 0.84 ms |

- Batch invariance: 8 probe tokens computed alone and placed first and last in
  batches of 16, 64, 256 and 1024 tokens; the token counts in parentheses are
  the batches that differ.
- Accuracy: 256 tokens against the HF formula in FP64 with FP64 routing. The
  candidates with a BF16 router pick a different expert set for 11 of 256
  tokens, which dominates their error.
- SGLang's compiled `sgl_kernel` 0.3.21 cannot load next to torch 2.13, so its
  two kernels on this path are SGLang's own replacements (Triton
  `moe_sum_reduce`, JIT `moe_align_block_size`); `environments` in
  `report.json` records both Python environments.
- TP4 gate (`tp4-moe/rank-*.json`): HF round trip, replicated routes, chunk and
  reorder invariance, training forward and gradients pass on every rank.

Site paths, host names and the scheduler do not appear in any file.
