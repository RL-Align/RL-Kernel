# WS2 strict ROCm Attention — bitwise parity and performance

> Operator-only benchmark. No model checkpoint or serving engine was used;
> the shapes are Qwen3-8B's attention shapes.

## Environment

| Item | cpu |
|---|---|
| architecture | n/a |
| cpu_count | 192 |
| cuda | None |
| extension_attention_symbols | none |
| gpu | n/a (host execution) |
| gpu_count | 0 |
| hip | None |
| native_collective | n/a (single-process host run) |
| python | 3.12.3 |
| torch | 2.12.0+rocm7.14.0a20260608 |
| torch_threads | 192 |
| triton | n/a (host execution) |

## Methodology

- Operator shape: `Hq=32`, `Hkv=8`, `D=128`, `B=1`, causal; sequence sweep 512, 1024, 2048.
- Measured paths:
  - `sdpa`: `torch.nn.functional.scaled_dot_product_attention`. **Speed baseline only** — as in PR #325, no accuracy comparison is mixed into the speed table.
  - `strict-aiter`: `StrictRocmAiterCKAttentionCore` called **once for all heads**. This is the core, not the production schedule: the Vime provider launches it once per (batch row, KV group). See the per-KV-group schedule table for that cost.
  - `reference-native`: `_C.deterministic_attention_forward/backward`, the materializing FP32 reference core hipified from the shared `.cu`.
  - `triton-bitwise`: `TritonDeterministicAttentionOp`, whose contract is bit-identity with `reference-native`.
- Timing: CUDA events, median and p95. Peak memory is the per-call increase in `torch.cuda.max_memory_allocated` above what was live before the call.
- Accuracy is against an FP64 oracle over the same BF16/FP16-rounded inputs. Repeat = two identical calls are bitwise equal; batch-invariant = a row computed alone is bitwise equal to the same row inside a batch.
- 2 warmups, 5 measured forward samples, 3 measured forward+backward samples. Raw medians, p95, min and max are in `results.json`.

Reproduce from the repository root:

```bash
python benchmarks/benchmark_ws2_rocm_attention.py \
  --seq-lens 512,1024,2048 \
  --dtypes bf16,fp16 \
  --warmup 2 --samples 5 \
  --training-samples 3 \
  --output-dir benchmarks/results/ws2_rocm_mi300x
```

### Unavailable paths

- `strict-aiter`: GPU-only path; not available on the host
- `strict-fa4`: GPU-only path; not available on the host
- `reference-native`: GPU-only path; not available on the host
- `triton-bitwise`: GPU-only path; not available on the host

## Bitwise parity: Triton port vs the native reference core

Acceptance is 0 mismatched elements. This is the contract the Triton core exists to hold.

| dtype | S | out mismatched | lse mismatched | dQ | dK | dV | bitwise |
|---|---:|---:|---:|---:|---:|---:|:---:|

`dQ/dK/dV` are measured on the BF16 sweep only; `n/a` marks the FP16 rows.

## Single-device Attention (bf16)

### Forward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB | out max-abs vs FP64 | lse max-abs vs FP64 | Repeat |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 512 | sdpa | 38.1068 | 43.2369 | 1.00x | 0.1 | 8.606e-03 | n/a | yes |
| 512 | pytorch-native | 7.0910 | 9.0480 | 0.19x | 0.0 | 1.947e-02 | n/a | yes |
| 1024 | sdpa | 174.7517 | 252.6524 | 1.00x | 109.8 | 8.191e-03 | n/a | yes |
| 1024 | pytorch-native | 58.5536 | 87.5402 | 0.34x | 130.2 | 1.610e-02 | n/a | yes |
| 2048 | sdpa | 192.7010 | 232.4717 | 1.00x | 111.9 | 9.314e-03 | n/a | yes |
| 2048 | pytorch-native | 227.7564 | 307.3500 | 1.18x | 548.6 | 1.790e-02 | n/a | yes |

### Forward+backward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB |
|---:|---|---:|---:|---:|---:|
| 512 | sdpa | 170.4205 | 175.4842 | 1.00x | 7.5 |
| 512 | pytorch-native | 48.0205 | 156.2514 | 0.28x | 64.8 |
| 1024 | sdpa | 297.3551 | 358.7597 | 1.00x | 135.3 |
| 1024 | pytorch-native | 267.1181 | 313.6076 | 0.90x | 190.8 |
| 2048 | sdpa | 555.5966 | 569.2160 | 1.00x | 112.9 |
| 2048 | pytorch-native | 489.2399 | 550.9883 | 0.88x | 815.6 |

## Single-device Attention (fp16)

### Forward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB | out max-abs vs FP64 | lse max-abs vs FP64 | Repeat |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 512 | sdpa | 31.7511 | 33.4260 | 1.00x | 0.0 | 1.045e-03 | n/a | yes |
| 512 | pytorch-native | 697.2643 | 707.5163 | 21.96x | 0.0 | 2.811e-03 | n/a | yes |
| 1024 | sdpa | 96.1593 | 113.6039 | 1.00x | 79.9 | 1.075e-03 | n/a | yes |
| 1024 | pytorch-native | 2726.1935 | 2804.8509 | 28.35x | 65.7 | 2.117e-03 | n/a | yes |
| 2048 | sdpa | 250.5798 | 286.6925 | 1.00x | 81.5 | 1.065e-03 | n/a | yes |
| 2048 | pytorch-native | 21112.2733 | 21152.1648 | 84.25x | 515.2 | 2.143e-03 | n/a | yes |

### Forward+backward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB |
|---:|---|---:|---:|---:|---:|
| 512 | sdpa | 204.9800 | 242.0397 | 1.00x | 0.0 |
| 512 | pytorch-native | 2154.7387 | 2248.4103 | 10.51x | 0.0 |
| 1024 | sdpa | 386.7709 | 387.0739 | 1.00x | 79.7 |
| 1024 | pytorch-native | 8579.2534 | 12587.9295 | 22.18x | 189.2 |
| 2048 | sdpa | 1122.4169 | 1212.3130 | 1.00x | 79.7 |
| 2048 | pytorch-native | 53718.0483 | 54692.0301 | 47.86x | 764.6 |

## Production core versus the reference core

These are two different kernels, so this is a tolerance comparison, not a parity claim. It is here to size the gap, not to assert equality.

| dtype | S | out max-abs | out relative-L2 | lse max-abs |
|---|---:|---:|---:|---:|

## Batch-composition invariance

A row computed alone must be bitwise equal to the same row inside a batch. The strict ROCm core rejects `B > 1` outright, so for that path the property is structural rather than measured.

| S | Path | Bitwise | Mismatched | Note |
|---:|---|:---:|---:|---|
| 512 | sdpa | yes | 0 | measured |
| 512 | pytorch-native | yes | 0 | measured |
| 1024 | sdpa | yes | 0 | measured |
| 1024 | pytorch-native | yes | 0 | measured |
| 2048 | sdpa | yes | 0 | measured |
| 2048 | pytorch-native | yes | 0 | measured |

## TP-degree invariance of the strict ROCm core

A head shard computed under TP=N versus the same slice of an unsharded run. TP performs no cross-rank reduction in attention, so any nonzero value means the kernel's result depends on how many heads shared the launch. `raw_launch` is one launch for all heads; `one_kv_group_per_launch` is the schedule the Vime provider actually uses.

| S | Schedule | TP | Local Hq | Local Hkv | out max-abs | lse max-abs | Invariant |
|---:|---|---:|---:|---:|---:|---:|:---:|

## Figures

`reference-native` and `triton-bitwise` allocate exactly the same buffers, so their memory curves coincide and the later-drawn series hides the earlier one.

![Single-device latency and memory grid](single_gpu_grid.png)

![Single-device latency](single_gpu_latency.png)

![Single-device peak memory](single_gpu_memory.png)

![Bitwise exactness matrix](exactness_matrix.png)

![TP-degree invariance](tp_degree_invariance.png)
