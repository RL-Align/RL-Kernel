# PR 396: ROCm strict R/R with MFMA GEMM and chunked Triton attention

Measurements behind https://github.com/RL-Align/RL-Kernel/pull/396 on one node
of 8 × AMD Instinct MI300X VF (gfx942), torch 2.12.0+rocm7.14, HIP 7.14.60850,
Triton 3.7.0, vLLM 0.26.1rc1, AITER with CK headers, Vime `c80200e`.
Vime rounds were produced with `examples/vime_rocm_attention_ablation/run_pr377_workload.py`
and tabulated with `summarize_pr377_runs.py`; kernel numbers are medians of
`triton.testing.do_bench` over the shapes listed in each table.

## 1. Where the strict R/R time went

PR 394's own trace showed the strict route ~3.9x slower than native end to end
(step 295 s vs 77 s).  Two structural costs explain most of it:

1. The gfx942 deterministic GEMM (`triton/matmul/det_gemm.py`) evaluates
   scalar-FMA leaves of 32 K values and a BF16 midpoint tree over them.  It
   never touches the MFMA units, and every K/32 leaf writes a full `M x N`
   BF16 partial to HBM.  On Qwen3-8B TP4 shapes it is 3-30x slower than
   hipBLASLt (table below); with 4 GEMMs per layer this alone put the decode
   step near 12 ms of GEMM work and the 4096-token training microbatch near
   1 s per forward.
2. Attention decode ran the AITER/CK fixed-M128 prefill template one program
   per (request, KV head) over the whole cache: ~316 us for four 7168-token
   requests, 36 layers -> ~11 ms per decode step.

## 2. MFMA batch-invariant GEMM (`rlkernel.det_gemm.triton_mfma_rocm.v1`)

Contract: `v_mfma_f32_16x16x16_bf16` with `matrix_instr_nonkdim=16` and
`kpack=2` pinned, `BLOCK_K=64` tiles inside `CHUNK_K=1024` chunks, FP32 chunk
partials combined ascending, one BF16 rounding.  Experimentally, `kpack`
changes the result bits (it changes the in-tile K order); `BLOCK_K` (32/64/128),
`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`, `waves_per_eu`, weight
layout, row count and the split schedule do not.  `tests/test_rocm_mfma_gemm.py`
pins all of this.

Median latency in us (k-tree = previous contract with its inference schedule):

| shape (K x N) | M | hipBLASLt | k-tree | MFMA |
|---|---:|---:|---:|---:|
| qkv 4096x1536 | 8 | 10.2 | 81.7 | 26.0 |
| qkv 4096x1536 | 4096 | 103.3 | 3271.1 | 125.0 |
| o_proj 1024x4096 | 8 | 8.1 | 86.0 | 10.9 |
| o_proj 1024x4096 | 4096 | 72.7 | 2287.8 | 90.4 |
| gate_up 4096x6144 | 8 | 18.8 | 81.7 | 22.9 |
| gate_up 4096x6144 | 4096 | 340.6 | 15494.2 | 482.4 |
| down 3072x4096 | 8 | 11.6 | 83.4 | 27.4 |
| down 3072x4096 | 4096 | 189.1 | 6806.1 | 242.5 |
| lm_head 4096x37984 | 8 | 130.3 | 285.8 | 123.0 |
| lm_head 4096x37984 | 4096 | 2251.2 | 99358.0 | 3056.9 |

Per decode layer (TP4, M=8) the four projections drop from ~333 us to ~87 us;
per 4096-token training forward layer from ~28 ms to ~0.94 ms.  Large-M
forward stays 1.2-1.4x behind hipBLASLt; the weight gradient uses a
reduction-major copy of `dY` because a column-major A operand loads 3-4x
slower on gfx942 (offline sweep: gate_up wgrad 1463 us as a strided view vs
~480 us after the copy).

Decode chunk sweep (M=8, sum over the four projections): `CHUNK_K=1024/BLOCK_K=64`
79.3 us; 512/64 84.5 us; 256/64 84.4 us; 1024/32 87.7 us.

## 3. Chunked Triton attention (`rlkernel.rocm.triton_chunked_flash_attention.v1`)

Contract: 64-key blocks inside 512-token KV chunks, each chunk's online
softmax from an empty state, chunks merged ascending with the exact FA2
rescale, fully masked blocks/chunks leave a row untouched, `P` rounded to the
input dtype before `P.V`, hardware `exp2`, FP contraction off.  The
monolithic schedule (query tiles) and the split schedule (one program per
sequence/KV head/chunk plus an ascending merge) are bit-identical, so a
Megatron full-sequence forward, a vLLM prefill, a prefix-cached extend and a
single-token decode agree exactly (`tests/test_rocm_triton_chunked_attention.py`).

Eager single-call latency in us (HQ=8, HKV=2, D=128, 16-token pages;
Triton with the unmasked fast path for fully visible key blocks):

| case | Triton chunked | CK fixed M128 |
|---|---:|---:|
| prefill 1 x 4096 | 270 | 179 |
| prefill 1 x 1024 | 56 | 53 |
| prefill 4 x 1024 | 93 | 59 |
| decode 4 x 7168 | 68 | 316 |
| decode 4 x 2048 | 68 | 95 |
| decode 8 x 4096 | 69 | 184 |

Training core (`StrictRocmAiterCKAttentionCore.forward_with_lse`) forward /
forward+backward in ms, B=1, AITER deterministic backward in both cases:

| S | CK fwd | CK fwd+bwd | Triton fwd | Triton fwd+bwd | peak |
|---:|---:|---:|---:|---:|---:|
| 4096 | 0.67 | 3.15 | 0.84 | 2.51 | 4.1 GiB |
| 8192 | 0.98 | 7.62 | 1.51 | 6.46 | 16.2 GiB |

The backward dominates training attention and is unchanged; the deterministic
AITER backward is O(S^2) in workspace (16 GiB at S=8192).

## 4. End-to-end, PR377 workload, one round

Qwen3-8B, actor TP4/CP2/PP1, two TP4 vLLM engines, 8 samples, 7168-token
response limit, 4096 training tokens/GPU, seeds 1234, HIP Graph
FULL_AND_PIECEWISE (capture 32), `RL_KERNEL_ROCM_FIXED_PAGED_TILE=128`.

### 4a. Healthy node after host GPU reset (vLLM memory utilization 0.38)

Primary numbers.  Health check before the pair: 8-GPU copy 3.82-3.90 TB/s,
decode GEMM 23-28 us, prefill GEMM 335-360 us, TP4 all-reduce 1 MB 43-53 us.
Runs `mxs-pair-v6-p-p` and `mxs-pair-v6-r-r-triton`.

| Config | Mismatch Count | Max \|dlogp\| | torch.equal |
|---|---:|---:|:---:|
| P/P native | 23294 / 42042 | 2.342051 | false |
| R/R strict (MFMA GEMM + Triton attention) | **0 / 28652** | **0** | **true** |

| Metric | P/P native | R/R (Triton attn) | R/R vs P/P |
|---|---:|---:|---:|
| rollout time | **56.31 s** | 67.17 s | 19.3% slower |
| effective tokens/GPU/s | **93.32** | 53.32 | 42.9% lower |
| update weights | 2.59 s | **1.18 s** | **54.6% faster** |
| log probs | 8.71 s | **7.19 s** | **17.4% faster** |
| actor train | 14.92 s | **10.67 s** | **28.5% faster** |
| train time | 24.31 s | **18.36 s** | **24.5% faster** |
| actor train tok/s | 2886.6 | 2779.7 | 3.7% lower |
| end-to-end step | **83.69 s** | 87.71 s | 4.8% slower |

Mean sampled response length was 5255 tokens (P/P) vs 3581 (R/R), so the per-token rollout throughput (93 vs 53 tokens/GPU/s, 1.75x) is the honest measure of the remaining gap, not the 19% rollout-time difference. Against the #394 R/R baseline on a healthy node (round 0: 39 tokens/GPU/s, log probs 25.5 s, actor train 50.1 s, step 183.0 s) this branch is 1.36x faster in rollout throughput and 4.7x faster in actor train.

### 4b. Same degraded node, same conditions (vLLM memory utilization 0.30)

| Config | Mismatch Count | Max \|dlogp\| | torch.equal |
|---|---:|---:|:---:|
| P/P native | 20305 / 37034 | 2.840120 | false |
| R/R strict, MFMA GEMM + CK attention | 0 / 33320 | 0 | true |
| R/R strict, MFMA GEMM + Triton attention | 0 / 28652 | 0 | true |

| Metric | P/P native | R/R (CK attn) | R/R (Triton attn) | Triton R/R vs P/P |
|---|---:|---:|---:|---:|
| rollout time | 172.13 s | 336.20 s | 312.16 s | 81.4% slower |
| effective tokens/GPU/s | 26.89 | 12.39 | 11.47 | 57.3% lower |
| update weights | 7.39 s | 3.53 s | 8.35 s | 13.0% slower |
| log probs | 31.32 s | 44.33 s | 37.63 s | 20.1% slower |
| actor train | 63.25 s | 54.56 s | 48.15 s | 23.9% faster |
| train time | 95.58 s | 99.45 s | 86.38 s | 9.6% faster |
| actor train tok/s | 601.6 | 629.3 | 616.2 | 2.4% higher |
| end-to-end step | 276.24 s | 441.30 s | 409.79 s | 48.3% slower |

### 4c. Healthy-node points measured before the degradation

| Run | rollout | log probs | actor train | step | consistency |
|---|---:|---:|---:|---:|---|
| P/P native (v167 round 0, morning) | 42.14 s | 9.44 s | 14.07 s | 69.02 s | 94330/133216 over 3 rounds, false |
| R/R PR 394 baseline (v165 round 0) | 104.87 s | 25.46 s | 50.05 s | 182.98 s | 0/130954, true |
| R/R this branch, MFMA + CK attn | 132.01 s | 21.00 s | 11.59 s | 169.08 s | 0/33320, true |

Response lengths differ between arms (the sampled tokens differ because the
arithmetic differs), which is why rollout time is best read together with
tokens/GPU/s: 98.4 (P/P), 39.3 (PR 394 R/R), 31.6 (this branch, CK attn).

## 5. Node degradation and the tools added for it

During this work the node degraded: a native P/P round went from 69 s to
276 s with every phase ~4x slower, while single-GPU copy bandwidth
(3.85 TB/s) and GEMM latencies stayed unchanged.  The
container's PID 1 is `sleep infinity`, so killed Ray/vLLM workers become
permanent zombies; reaping them (a ptrace-injected `wait4` on PID 1) removed
2285 zombies but the driver still lists the GPU contexts of eight killed vLLM
workers (73 GB and 11 GB per GPU, hardware queues still mapped) under
`/sys/class/kfd/kfd/proc/`.  A host-side GPU reset restored the node (section 4a was measured after
it); the absolute numbers in 4b are inflated for every arm, the relative
ones are same-condition.

Two harness pitfalls fixed on the way: operator-shell `http_proxy` variables
inherited by Ray turned every generation longer than 30 s into a 502 retry
loop, and AITER's JIT resolved `GPU_ARCHS=native` to an empty offload list
inside vLLM workers (compiled for gfx906 and failed); `launch_arm.sh` now
strips the proxy variables and pins `GPU_ARCHS`.

## 6. What is still missing for a net win

- Rollout decode is ~1.75x native per token even though the GEMMs are
  within 1.3x of hipBLASLt and the attention core is faster than CK.  The
  decode step runs inside a full HIP graph, so the gap has to be attributed
  kernel by kernel.  `profile_rocm_rollout_decode.py` + `summarize_rollout_trace.py`
  capture per-rank kernel traces of one decode workload for this analysis.
- Training attention at CP2 still all-gathers Q/K/V and computes the whole
  sequence on every CP rank; the per-row-invariant kernel allows computing
  only the local zigzag rows against the gathered K/V.
