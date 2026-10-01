# A100/SM80 native `linear_logp` — Stage 3 architecture experiments

- Date: 2026-09-21
- Branch: `sm80-linear-logp`
- GPU: NVIDIA A100 80GB PCIe, SM80, GPU 1
- Shape: BF16, D=4096, V=128256, N=128..4096
- Scope: forward-only, single GPU, explicit PoC call; no registry/backward/TP/deterministic/FP8 changes

## 1. Baseline

Stage 2 uses BM=32, BN=128, BK=32, 8 warps/CTA, WMMA BF16-to-FP32 and a two-stage `cp.async` pipeline. It does not materialize `[N,V]`.

- N=4096: 246.12 ms
- Triton: about 83.6 ms
- cuBLAS/materialized: about 25.5 ms
- Resources: 72 registers/thread, 31,744 B shared/CTA, 256 threads
- Register-limited theoretical residency: 3 CTA/SM = 24 warps/SM = 37.5% warp occupancy

## 2. Experiment A — `ldmatrix + mma.sync`

### Mapping

- Kept Stage 2's BM/BN/BK=32/128/32 and double-buffered `cp.async` pipeline.
- A is staged row-major `[BM,BK]`; `ldmatrix.m8n8.x4` supplies an `m16k16` fragment.
- Weight remains physically `[BN,BK]` row-major, which is logical B `[K,BN]` column-major. Consequently B uses non-transposed `ldmatrix.m8n8.x2`; using `.trans` was experimentally shown to be incorrect for this physical layout.
- Each warp owns four `m16n8k16` FP32 accumulator sets. Accumulator ownership follows the PTX mapping: rows `lane/4` and `lane/4+8`, columns `2*(lane%4)+{0,1}`.
- No explicit XOR swizzle was added: the 64-byte row stride and native ldmatrix submatrix access were retained so that this experiment isolates WMMA abstraction removal. The shared operands remain double buffered.
- Existing online max/sumexp/target-logit logic was unchanged; its partition width changed from 16 to 8 only to match `m16n8` output fragments.

An intermediate diagnostic using `.trans` for B produced incorrect results and was rejected. Direct PTX-fragment loads proved `mma.sync`, accumulator ownership and online softmax correct; switching B to non-transposed `ldmatrix` restored full correctness.

### Resources and performance

| Metric | Stage 2 WMMA | Experiment A |
|---|---:|---:|
| registers/thread | 72 | 80 |
| shared/CTA | 31,744 B | 30,720 B |
| threads/CTA | 256 | 256 |
| theoretical occupancy | 37.5% | 37.5% |
| N<=2048 fixed region | about 167.5 ms | about 207.8 ms |
| N=4096 | 246.12 ms | 336.87 ms |

The hand-written path is 24% slower in the single-wave/fixed-latency region and 37% slower at N=4096. Registers increased rather than decreased because `m16n8` requires more independently managed output fragments and online-softmax partitions. Therefore the experiment does not show a material WMMA abstraction bottleneck.

## 3. Experiment B — split-V

Experiment B was derived independently from the Stage 2 WMMA commit; it does not contain Experiment A's hand-written MMA path.

### Design

- Primary grid: `ceil(N/32) * split_v` CTAs.
- The 1002 BN=128 vocabulary tiles are divided by integer tile boundaries, so all splits are aligned and together cover V exactly even when 1002 is not divisible by `split_v`.
- Each CTA emits `local_max`, `local_sumexp`, and `local_target_logit` for 32 tokens.
- Workspace: three FP32 arrays of logical shape `[split_v,N]`, exactly `12*N*split_v` bytes.
- Combine kernel first computes global max, then accumulates `local_sumexp * exp(local_max-global_max)`, max-propagates the unique target logit, and emits final logp.
- No `[N,V]` tensor is created.

### Grid and workspace

For split=16:

| N | primary CTA | combine CTA | workspace |
|---:|---:|---:|---:|
| 128 | 64 | 1 | 24 KiB |
| 1024 | 512 | 4 | 192 KiB |
| 4096 | 2048 | 16 | 768 KiB |

Primary-kernel resources remain 72 registers/thread, 31,744 B shared/CTA and 256 threads/CTA; the theoretical residency ceiling remains 37.5%. The speedup therefore comes from shorter per-CTA vocabulary scans and increased grid parallelism.

### Split sweep

Times below use the same warmup=2, iterations=5 sweep protocol.

| split_v | N=128 | N=1024 | N=4096 | workspace at N=4096 |
|---:|---:|---:|---:|---:|
| 2 | 83.40 ms | 83.24 ms | 174.25 ms | 96 KiB |
| 4 | 41.85 ms | 61.66 ms | 146.02 ms | 192 KiB |
| 8 | 21.01 ms | 43.81 ms | 147.92 ms | 384 KiB |
| **16** | **10.50 ms** | **36.71 ms** | **142.34 ms** | **768 KiB** |

For N=128, latency scales almost inversely with split count, directly demonstrating removal of the full-V single-CTA latency wall. At N=4096 the curve flattens after split=4: the grid is already large and resource/bandwidth contention dominates, so extra splitting gives only a small benefit.

## 4. Full performance table

Final split=16 numbers use warmup=5, iterations=20. Stage 2 and Experiment A use their corresponding final 20-iteration runs.

| N | cuBLAS materialized | Triton | Stage 2 WMMA | Experiment A mma.sync | best split-V (16) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.980 ms | 61.515 ms | 167.373 ms | 207.862 ms | **10.491 ms** |
| 256 | 1.639 ms | 61.313 ms | 167.455 ms | 207.841 ms | **15.472 ms** |
| 512 | 3.174 ms | 61.397 ms | 167.489 ms | 207.857 ms | **21.929 ms** |
| 1024 | 6.419 ms | 61.495 ms | 167.512 ms | 207.674 ms | **36.712 ms** |
| 2048 | 12.818 ms | 62.557 ms | 167.608 ms | 207.642 ms | 74.716 ms |
| 4096 | 25.706 ms | 83.655 ms | 246.116 ms | 336.874 ms | 143.696 ms |

Split=16 beats Triton for N<=1024, is 19% slower at N=2048, and 1.72x slower at N=4096. Relative to Stage 2 it is 15.95x faster at N=128 and 1.71x faster at N=4096.

Final split=16 throughput:

| N | tokens/s |
|---:|---:|
| 128 | 12,201 |
| 256 | 16,546 |
| 512 | 23,349 |
| 1024 | 27,893 |
| 2048 | 27,411 |
| 4096 | 28,505 |

## 5. Correctness and safety

| Path | max abs error range | mean abs error range | NaN/Inf | peak activation at N=4096 | sanitizer |
|---|---:|---:|---|---:|---|
| Experiment A | 1.62e-5 to 2.77e-5 | 5.63e-6 to 5.93e-6 | none | 0.03 MB | 0 errors |
| split=16 | 1.72e-5 to 2.77e-5 | 5.61e-6 to 5.92e-6 | none | 0.78 MB | 0 errors |

The split workspace is O(N*split_v), not logits materialization. The merge correctly rescales each local sum around the global maximum. Target ownership is represented by `-inf` in non-owning splits and max-propagated by the combine kernel.

## 6. Final judgement

### A. Is hand-written `mma.sync` worth continuing?

Not as the primary direction based on this experiment. It increased registers from 72 to 80 and increased single-CTA latency by about 24%. Removing WMMA abstraction did not expose a hidden speedup. A different, substantially redesigned register/shared layout might behave differently, but incremental work on this exact m16n8 mapping is not justified by the measurements.

### B. Should split-V be part of the final architecture?

Yes. It is the first change that removes the small-N fixed latency wall and it improves every measured N. The nearly inverse split scaling at N=128 proves the gain is grid-level parallelism/shorter CTA scans, not an instruction-level effect. A shape-dependent split policy is likely preferable: split=16 wins this sweep, while N=4096 is already close to a plateau and may benefit from a more refined split/occupancy tradeoff.

### C. Stage 4 recommendation

Continue split-V architecture, not the current hand-written MMA experiment. If Stage 4 is authorized, first make split count shape-dependent and reduce the large-N bandwidth/resource plateau. Do not merge the current Experiment A implementation into split-V: A was independently negative. Reconsider `mma.sync + swizzle` only as a new layout design after split-V is stable, rather than mechanically combining the measured A and B kernels.

Stage 4 was not started.
