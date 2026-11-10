# DSv4 Selected-Token Log-Probability

The DSv4 LM head as selected-token log-probabilities (P7), profile `dsv4-logprob-v1`. It
takes the model's final hidden state BF16 `[T, 4096]` and the frozen head BF16
`[129280, 4096]`. It returns FP32 `logp` and `lse` per token, with a backward to the hidden
state.

| Piece | Where |
|---|---|
| Golden (contract) | `rl_engine/kernels/ops/pytorch/logprob_dsv4/dsv4_logprob.py` |
| CUDA kernels | `csrc/cuda/logprob_dsv4/dsv4_logprob.cu` |
| Python wrapper + autograd | `rl_engine/kernels/ops/cuda/logprob_dsv4/dsv4_logprob.py` (`selected_logprob`) |
| Model entry | `Dsv4Model.selected_logprob(hidden, targets, temperature)` |
| GEMM probe | `scripts/probe_dsv4_head_gemm.py` |
| Tests | `tests/ops/logprob_dsv4/test_dsv4_logprob.py`, `tests/models/deepseek_v4/test_dsv4_model.py` |
| Benchmark | `benchmarks/benchmark_dsv4_logprob.py` |

## Contract

**Logits.** `z = hidden @ W.T` runs on DeepGEMM `bf16_gemm_nt` with FP32 output; the
accumulator is never rounded to BF16. The probe checked, bytewise:
- every row is the same for M from 1 to 4096 (batch-invariant);
- a vocabulary shard (TP 2, 4, 8) gives the same columns as the full head;
- reruns are identical.

**Log-sum-exp.** Each FP32 step is correctly rounded, with no FMA; `expf` and `logf` are
libdevice:

```text
z'   = z, or z / T (a division)
64 fixed tiles of V / 64 columns (2020 for DSv4)
per (row, tile):  m_t = max;  lane l of 32 sums expf(z'_c - m_t) over columns l + 32k, in
                  order;  s_t = halving over the lanes
per row:          M = max_t m_t;  e_t = s_t * expf(m_t - M)
                  S = (e_i + e_{i+32}), then halving;  lse = M + logf(S)
                  logp = z'_target - lse
```

The 64-tile split is the #241 vocab-TP structure. With the head sharded over TP ranks on
tile boundaries (TP 1, 2, 4 or 8), each rank computes its tiles' `(m_t, s_t)`
(`tile_stats_cuda`), the partials are gathered, and `merge_cuda` gives the same bytes as a
single device. The test checks this. #241's own merge is defined by torch's reduction
order, so this profile writes its own arithmetic but keeps the structure.

**Backward (dhidden only; the head is frozen).** `dL/dlogp` is one scalar `g` per row:

```text
P  = (onehot(target) - expf(z' - lse)) [/ T]      BF16 [n, V]
Q  = P @ W                                        DeepGEMM bf16_gemm_nn, FP32 [n, H]
dh = BF16(g * Q)
```

- When the hidden state requires grad, the forward computes `Q` while the chunk's logits are
  still in memory, and keeps only `Q`.
- The backward is one multiply. It is deterministic and batch-invariant.
- Without grad (rollout, actor recompute), nothing extra runs.
- `lse` carries no gradient.

**Chunking.** Tokens run in chunks of `chunk_rows` (default 4096) so that the FP32 logits
never exist for the whole batch. Each row is computed alone, so the chunking does not change
any byte.

## Semantics and the sampling gate

- **Temperature.** With `T = 1`, `logp` is the raw model log-probability, which the T02
  sampling gate (`docs/contracts/p7-t02-m10-sampling-contract.md`,
  `rl_engine/validation/sampling_params/gate.py`) requires in consistency mode. Other
  temperatures divide the FP32 logits, as vLLM's sampler does (`logits.div_(T)` in FP32).
- **Not implemented:** top-k / top-p / min-p replay. The gate rejects those settings in
  consistency mode.
- **Logit precision differs from the rollout default.** vLLM's default head produces BF16
  logits; this profile keeps FP32. The rollout patch must run this head (or at least these
  logits) for rollout and actor log-probs to agree bitwise.

## Results (H200, H = 4096, V = 129280)

- The CUDA output is byte-equal to the golden for logp, lse, the tile partials and dL/dz.
  This holds at V = 2560, 2112 and 129280, with logits scaled by 1 and by 30, and at T = 1
  and T = 0.7.
- The golden matches float64 `log_softmax` within 2e-5.
- `selected_logprob` is batch-invariant: a sub-batch equals the full batch, and so do other
  chunk sizes.
- dhidden is deterministic and batch-invariant. It is within 1e-2 relative L2 of a float64
  autograd reference.
- The model-level test checks that the next-token logp is the same for batch, alone and
  decode.

Microseconds, median, eager. "Naive" is cuBLAS BF16 logits + FP32 `log_softmax` + gather,
with PyTorch autograd for the backward (not batch-invariant).

| T | fwd | naive fwd | ratio | fwd + bwd | naive fwd + bwd | ratio |
|---|---|---|---|---|---|---|
| 1 | 309 | 294 | 1.05 | 611 | 589 | 1.04 |
| 64 | 328 | 375 | 0.88 | 677 | 734 | 0.92 |
| 1024 | 1916 | 2417 | 0.79 | 4080 | 4710 | 0.87 |
| 4096 | 7888 | 9714 | 0.81 | 16371 | 18540 | 0.88 |
| 16384 | 35559 | 37991 | 0.94 | 64557 | 77107 | 0.84 |

The head GEMM dominates. DeepGEMM's FP32-output GEMM takes 266 µs at T = 1 (reading the
1 GB head) and 6.9 ms per 4096 rows; cuBLAS BF16 takes 249 µs and 6.2 ms.

A first version recomputed the logits in the backward (three GEMMs). It ran at 1.16-1.43x
for forward + backward and failed the gate at T = 1. Computing `Q` in the forward brought it
back to two GEMMs.

## Limitations

- No entropy output. verl's fused path computes entropy; add it to the tile stats if needed.
- The TP path exists as `tile_stats_cuda` + gather + `merge_cuda`, but no
  `torch.distributed` wrapper is wired yet.
- No CUDA graph capture of the chunk loop.
