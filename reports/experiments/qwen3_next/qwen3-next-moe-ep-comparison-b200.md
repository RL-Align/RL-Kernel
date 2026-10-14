# Qwen3-Next expert-parallel MoE comparison (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-ep4-evidence/qwen3-next-moe-ep-comparison-b200.zip).

ZIP SHA-256: `3741db60471dbc7586ff64e32b0492039a8af509f29583fd6307955868622c68`.

Measured at commit `225c5a4b` (`measured-commit.txt`; every rank report has
`tracked_tree_dirty: false`) on one node with four B200 GPUs, torch
2.13.0+cu130, vLLM 0.30.0, Megatron-core 0.16.0rc0, NCCL 2.29.7, the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. Every candidate gets the same
1024 seeded BF16 tokens (replicated on every rank) and the layer-0 router and
512 experts; the shared expert, identical for all, is left out. At EP4 rank r
owns experts `[128r, 128r + 128)` at full width.

```bash
torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \
    --checkpoint <ckpt> --output runs/ep4-default
NCCL_ALGO=<Ring|allreduce:tree> NCCL_PROTO=<LL|LL128|Simple> [NCCL_MAX_NCHANNELS=1] \
    torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \
    --checkpoint <ckpt> --output runs/ep4-<setting>
torchrun --nproc-per-node 1 benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \
    --checkpoint <ckpt> --output runs/ep1
python benchmarks/models/qwen3_next/qwen3_next_moe_ep_prior_art.py \
    --merge runs/ep4-default runs/ep4-<each other setting> runs/ep1 --out report.json
python benchmarks/models/qwen3_next/plot_qwen3_next_moe_ep.py <extracted ZIP dir>
```

`TRITON_F32_DEFAULT=ieee` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` were set for
every launch. NCCL 2.29 has no Tree all-gather, so the Tree settings restrict
all-reduce only.

| Candidate | EP4 == EP1 | Distinct EP4 results over 13 NCCL settings | Repeat bitwise | Batch invariant | Ranks agree | Max abs vs FP64 | EP4 latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RL-Kernel EP4** (`qwen3_next_ep`) | **yes** | **1** | **yes** | **yes** | **yes** | 1.8e-5 | 2.26 ms |
| vLLM `fused_experts` + `expert_map`, NCCL `all_reduce` | no | 5 | yes | no | yes | 8.7e-4 | 0.52 ms |
| same, `VLLM_BATCH_INVARIANT=1` | no | 5 | yes | no | yes | 8.7e-4 | 0.55 ms |
| Megatron-core `MoELayer`, alltoall dispatcher | no | 13 | no | no | no | 4.5e-5 | 13.7 ms |
| Megatron-core `MoELayer`, allgather dispatcher | no | 13 | no | no | no | 2.5e-5 | 12.6 ms |
| Megatron alltoall, `torch.use_deterministic_algorithms(True)` | yes | 1 | yes | yes | yes | 4.2e-5 | 14.8 ms |
| Megatron allgather, `torch.use_deterministic_algorithms(True)` | no | 1 | yes | yes | yes | 2.5e-5 | 13.7 ms |

- "Batch invariant": the first 16 tokens computed alone equal the same tokens
  inside the 1024-token batch, in every launch. vLLM's expert kernel alone
  (EP1) is batch invariant; at EP4 the NCCL all-reduce after it is not under
  Ring and NCCL defaults, while five of six all-reduce Tree settings are.
- `VLLM_BATCH_INVARIANT=1` selects a different tile config (checked) but gives
  the same bits as the default on this layer; the NCCL sum after it is unchanged.
- Megatron's token unpermute sums the routed outputs with `scatter_add_`, so
  by default every call and every rank differs. Under deterministic algorithms
  it uses `index_add_`: the alltoall dispatcher is then invariant on every axis
  measured here, and the allgather dispatcher on every axis except EP4 == EP1.
- No two implementations give the same bits. Largest difference: RL-Kernel vs
  any Megatron variant 3.1e-5, any of them vs vLLM 8.7e-4.
- Latency is the median of 20 calls on each rank, the slowest rank reported.
  EP1 (one GPU): RL-Kernel 1.94 ms, vLLM 0.77 / 0.81 ms, Megatron 53.8-56.2 ms
  (`SequentialMLP` evaluates the 512 experts one after another).

Every result file is JSON with the commit, environment and NCCL setting it was
measured under; `report.json` is the merge, `figure.png` its rendering,
`SHA256SUMS` covers every file.
