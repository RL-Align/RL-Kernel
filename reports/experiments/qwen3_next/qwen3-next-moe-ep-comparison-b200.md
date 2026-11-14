# Qwen3-Next expert-parallel MoE comparison (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-moe-ep-comparison-b200.zip).

ZIP SHA-256: `2375ec36485eb09a8acc81586268584b80ef3914f69a4724d6acbac5bf97c6cf`.

Measured at commit `7db4a14b` (`measured-commit.txt`; every rank report has
`tracked_tree_dirty: false`) on one node with four B200 GPUs, torch
2.13.0+cu130, vLLM 0.30.0, FlashInfer 0.6.18.post1, Megatron-core 0.16.0rc0,
NCCL 2.29.7, the official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. Every candidate gets the same
seeded BF16 tokens (replicated on every rank) and the layer-0 router and 512
experts; the shared expert, identical for all, is left out. At EP4 rank r owns
experts `[128r, 128r + 128)` at full width. The figure was rendered at
`b75f4422` (plot code only).

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

Every launch set `TRITON_F32_DEFAULT=ieee`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
and a short node-local `TMPDIR` (vLLM binds IPC sockets there; a long path
crashes its parallel-state setup). vLLM's own all-reduce dispatched FlashInfer,
custom, symmetric-memory and PyNCCL in that order. NCCL 2.29 has no Tree
all-gather, so the Tree settings restrict all-reduce only.

## Forward (1024 tokens)

| Candidate | EP4 == EP1 | Distinct EP4 results, 13 NCCL settings | Repeat | Batch invariant | Ranks agree | Max / mean abs vs FP64 | EP4 latency 64 / 1024 / 4096 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RL-Kernel EP4** (`qwen3_next_ep`) | **yes** | **1** | **yes** | **yes** | **yes** | 1.8e-5 / 6.9e-7 | 1.85 / 2.27 / 4.08 ms |
| vLLM `fused_experts` + vLLM's all-reduce | no | 1 | yes | yes | yes | 8.7e-4 / 5.3e-6 | 0.35 / 0.48 / 0.68 ms |
| same, `VLLM_BATCH_INVARIANT=1` | no | 1 | yes | yes | yes | 8.7e-4 / 5.3e-6 | 0.41 / 0.51 / 0.81 ms |
| vLLM `fused_experts` + NCCL `all_reduce` | no | 5 | yes | no (64-1024) | yes | 8.7e-4 / 5.4e-6 | 0.39 / 0.52 / 0.74 ms |
| FlashInfer `cutlass_fused_moe` (CUTLASS) + vLLM's all-reduce | no | 1 | yes | yes | yes | 8.7e-4 / 5.2e-6 | 0.33 / 0.39 / 0.48 ms |
| Megatron-core `MoELayer`, alltoall | no | 13 | no | no | no | 4.2e-5 / 1.1e-6 | 11.8 / 14.6 / 14.9 ms |
| Megatron-core `MoELayer`, allgather | no | 13 | no | no | no | 2.5e-5 / 9.2e-7 | 10.8 / 13.2 / 13.6 ms |
| Megatron alltoall, deterministic algorithms | yes | 1 | yes | yes | yes | 4.2e-5 / 1.1e-6 | 12.7 / 15.4 / 15.9 ms |
| Megatron allgather, deterministic algorithms | no | 1 | yes | yes | yes | 2.5e-5 / 9.2e-7 | 11.6 / 14.3 / 15.0 ms |

## Backward (candidates that train; 1024 tokens)

| Candidate | dx EP4 == EP1 | dW EP4 == EP1 (owned experts) | NCCL-setting invariant | Repeat | Batch invariant (dx) | Ranks agree (dx) | EP4 / EP1 latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RL-Kernel EP4** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** | 51 / 163 ms |
| Megatron alltoall | no | no | no | no | no | no | 41 / 154 ms |
| Megatron allgather | no | no | no | no | no | no | 37 / 146 ms |
| Megatron alltoall, deterministic algorithms | no | no | yes | yes | **no** | yes | 42 / 153 ms |
| Megatron allgather, deterministic algorithms | no | no | yes | yes | **no** | no | 40 / 151 ms |

vLLM and FlashInfer have no backward.

- Batch invariance is a grid: 8 probe tokens alone vs inside batches of 8,
  64, 256 and 1024 tokens at the first, middle and last position, in every
  launch, on the output and (backward) the input gradient.
- vLLM's expert kernel and its own all-reduce dispatch are batch- and
  NCCL-setting-invariant here; replacing that all-reduce with NCCL makes the
  result depend on the NCCL algorithm and, under Ring or the defaults, on the
  batch. `VLLM_BATCH_INVARIANT=1` selects a different tile config (checked) but
  gives the same bits on this layer.
- Megatron's token unpermute sums routed outputs with `scatter_add_`, so by
  default every call differs. Under deterministic algorithms it uses
  `index_add_`: the alltoall forward then passes every forward check, but its
  backward input gradient still depends on the batch and differs from EP1.
- No two implementations give the same bits. RL-Kernel vs Megatron differ by
  up to 3.1e-5; vLLM and FlashInfer, which compute the router logits in BF16,
  differ from both by up to 8.7e-4.
- Latency is the median of 20 forward calls (5 backward calls) on each rank,
  the slowest rank reported. EP1 forward at 1024 tokens: RL-Kernel 1.89 ms,
  vLLM 0.75 ms, FlashInfer 0.74 ms, Megatron 53-57 ms (`SequentialMLP` runs the
  512 experts one after another).
- Not measured: SGLang's EP MoE (its compiled `sgl_kernel` does not load next to
  torch 2.13; by source, sglang 0.5.21 combines EP ranks with
  `tensor_model_parallel_all_reduce` like vLLM, `fused_moe_triton/layer.py:1597`)
  and Miles' MoE backward (`radixark/miles@23d3d38`
  `miles/kernels/moe/fused_experts_bwd.py`, whose input, weight and route-weight
  gradients use `tl.atomic_add`).

Every result file is JSON with the commit, environment and NCCL setting it was
measured under; `report.json` is the merge, `figure.png` its rendering,
`SHA256SUMS` covers every file. Site paths, host names and the scheduler do not
appear in any file.
