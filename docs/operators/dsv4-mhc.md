# DSv4 mHC

Batch-invariant manifold-constrained hyper-connections for DeepSeek-V4 (SM90), profile
`dsv4-mhc-v1`:

- `mhc_pre`: four residual streams to one sublayer input, with the layer's attn_norm /
  ffn_norm optionally fused.
- `mhc_post`: the sublayer output and the streams back to four streams.
- `hc_head`: four streams to one before the final norm, which can be fused.
- The dX backward of all three.

| Piece | Where |
|---|---|
| Golden (contract) | `rl_engine/kernels/ops/pytorch/mhc_dsv4/dsv4_mhc.py` |
| CUDA kernels | `csrc/cuda/mhc_dsv4/dsv4_mhc.cu` |
| Python wrapper + autograd | `rl_engine/kernels/ops/cuda/mhc_dsv4/dsv4_mhc.py` (`mhc_pre`, `mhc_post`, `hc_head`) |
| Tests | `tests/ops/mhc_dsv4/test_dsv4_mhc_golden.py` (CPU), `tests/ops/mhc_dsv4/test_dsv4_mhc_cuda.py` |
| Benchmark | `benchmarks/benchmark_dsv4_mhc.py` |

## Contract

Shapes: residual BF16 `[T, 4, H]` with `H % 128 == 0` and `H <= 4096`, `K = 4H`.
- **mhc_pre**: `fn` FP32 `[24, K]`, `scale [3]`, `base [24]`.
- **hc_head**: `fn [4, K]`, `scale [1]`, `base [4]`.

The norm weight is BF16 `[H]`, as in vLLM.

**Controller GEMM.** DeepGEMM `tf32_hc_prenorm_gemm` runs with a fixed 16 splits, so the
result is batch-invariant. It produces the partials `pmix [16, T, Np]` and `pss [16, T]`
(`Np` is the row count of `fn` rounded up to a multiple of 8; `hc_head`'s padded FP32 `fn`
is built once per tensor and cached, which took 2-3 µs off each call, 16.8 to 14.1 µs at T = 1). Each value is reduced by
halving: `p[0:8] + p[8:16]`, then 4, 2, 1.

**Per token.** Every step is correctly rounded with no FMA contraction, except the
reductions marked "slice", which use `fmaf`. "Sum of four" means `(a0 + a1) + (a2 + a3)`.

```text
r      = 1 / sqrt(ss / K + 1e-6)
logit  = (raw * r) * scale_seg + base          seg: pre 0-3, post 4-7, comb 8-23
pre    = sigmoid(logit) + 1e-6                 sigmoid = 1 / (1 + expf(-x))
post   = 2 * sigmoid(logit)
comb   = softmax(4x4 logits, rows) + 1e-6, then column-normalize, then 19 x (row, column);
         each step M / (sum of four + 1e-6)
y      = BF16(sum of four of pre_i * res_i)
out    = y, or BF16((y * rn) * w) with rn = 1 / sqrt(slice(y, y) / H + 1e-6)
new_j  = BF16(post_j * x + sum of four of comb[i, j] * res_i)          (mhc_post)
```

`slice(a, b)` is an H-long dot product with a fixed partition:
- It uses 128 slices. Slice `i` owns the runs `[8i + 1024j, 8i + 1024j + 8)`.
- Each slice accumulates an `fmaf` chain, `j` ascending, then element ascending.
- The 128 partials are reduced by halving within each group of 32, then summed as a sum of
  four.

The runs of 8 match the kernels' 16-byte vector loads. The golden emulates the FMA exactly:
float64 product, TwoSum, then a fix-up for ties.

## Backward (dX only)

`fn`, `scale`, `base` and the norm weight are frozen (the target is LoRA training). With
`d_j = dL/dnew_j` and `dz = dL/out`:

```text
mhc_post:  dx      = sum of four of post_j * d_j
           dres_i  = sum of four of comb[i, j] * d_j
           dpost_j = slice(d_j, x)        dcomb[i, j] = slice(res_i, d_j)
mhc_pre:   u = dz * w;  c = ((slice(u, y) * rn) * (rn * rn)) / H;  dy = u * rn - c * y
           (dy = dz without a norm)
           dpre_i  = slice(res_i, dy);  controller backward -> draw [N], dss
           dres[k] = pre_i * dy + ((fmaf chain over n of draw_n * fn[n, k]) + (2 x_k) * dss)
```

The controller backward is the golden `_controller_backward`. It goes through the Sinkhorn
steps in reverse, applying `g / d - (sum of four of g * m) / (d * d)` at each step. Then
come the softmax and sigmoid derivatives. Last comes `dr`, a left-to-right sum over the 24
logits.

Gradient dtypes:
- **Input**: BF16 or FP32. BF16 to FP32 is exact.
- **Output**: dx / d_residual are BF16 or FP32. BF16 is the FP32 result rounded once.

Autograd uses BF16 in and out.

## Kernels

| Kernel | Grid / block | Notes |
|---|---|---|
| `pre_epilogue_kernel<24 or 4>` | one block per token, 160 threads | See below |
| `post_kernel` | (token, H / 1024), 128 | 8 columns per thread, 16-byte loads |
| `post_bwd_kernel` | one block per token, 128 | dx, d_residual and the 20 slice dot products in one pass |
| `pre_bwd_kernel` | one block per token, 160 | Same split as the epilogue |
| `gemm_bwd_kernel` | (K / 1024, T / 16 or T / 64), 256 | Writes d_residual once |

**`pre_epilogue_kernel`.** Warp 4 runs the split sums and the controller.
- It passes `pre` to warps 0-3 with a named barrier *before* the Sinkhorn iterations. The
  stream mix and the fused RMSNorm therefore run while the iterations run.
- The kernel is launched with programmatic dependent launch (PDL) after the DeepGEMM kernel.

**`pre_bwd_kernel`.** Warp 4 recomputes the controller and saves the Sinkhorn steps in
shared memory. Meanwhile warps 0-3 compute dy through the norm and `dpre`. Warp 4 then runs
the controller backward in shuffles.

**`gemm_bwd_kernel`.** Each thread holds 4 `fn` columns in registers. It reads `draw` as
float4 from shared memory and computes `pre_i * dy + g` directly into d_residual. A block
covers 16 tokens below T = 1024 (so small batches fill the GPU) and 64 above.

The residual and `dy` of the next 5 tokens are staged in shared memory with `cp.async`. Each
thread fetches and reads only its own elements, so no block barrier is needed. This keeps
enough bytes in flight without the extra registers that a register prefetch costs.

### Differences from vLLM

The rollout kernel (`mhc_pre_tilelang`, `ISEEKYAN/vllm@1a6542c`) chooses DeepGEMM's split
count from T: 64 for small batches, 2 at T = 4096. That makes it not batch-invariant, and
its TileLang epilogue has its own rounding. The rollout patch therefore has to run these
kernels too, as with the router. The model semantics are DeepSeek's: the RMS uses
`rsqrt(mean + eps)`.

## Results (H200, torch 2.13 / CUDA 13, H = 4096)

- CUDA output is byte-equal to the golden:
  - forward and backward;
  - with and without the norm;
  - `H` of 256, 1152 and 4096;
  - inputs scaled by 1e-3 and by 30;
  - BF16 and FP32 gradients;
  - `T = 0`.
- Batch-invariant: a sub-batch equals the full batch row for row. Runs are identical.

CUDA-graph replay, median, microseconds:

| T | pre + norm | vLLM pre + norm | ratio | post | vLLM post | ratio | block bwd |
|---|---|---|---|---|---|---|---|
| 1 | 14.2 | 12.4 | 1.15 | 7.9 | 8.0 | 0.99 | 24.6 |
| 64 | 15.1 | 12.8 | 1.18 | 8.1 | 8.1 | 1.00 | 32.0 |
| 1024 | 32.4 | 28.9 | 1.12 | 25.0 | 24.4 | 1.02 | 120.5 |
| 4096 | 106.4 | 95.1 | 1.12 | 79.9 | 78.6 | 1.02 | 402.2 |
| 16384 | 365.0 | 334.5 | 1.09 | 298.4 | 293.1 | 1.02 | 1419.3 |

- **The remaining forward gap is the GEMM**, not the epilogue. The fixed 16 splits are the
  price of batch invariance:
  - At T = 1, our GEMM takes 5.5 µs and vLLM's 64-split GEMM 3.0 µs. The epilogues take
    4.7 µs and 5.0 µs.
  - At T = 4096, the GEMMs take 51 µs and 39 µs. The epilogues are equal at 51 µs.
- **Train step** (`mhc_pre` with the norm, then `mhc_post`, forward + backward through
  autograd): 0.07-0.09x of a naive PyTorch implementation of the same math at T = 1024 to
  16384. vLLM has no backward to compare against.
- **Backward at T = 16384:**
  - `post_bwd` takes 449 µs, close to the HBM bound for its 1.8 GB.
  - `pre_bwd` takes 453 µs.
  - `gemm_bwd` takes 517 µs. It was 746 µs, and the gain came from `cp.async` staging:
    - With 124 registers per thread holding `fn`, only 16 warps fit per SM, each with one
      token's loads in flight. That is about half the bytes in flight that HBM needs.
    - Register prefetch pushed the kernel past 128 registers and doubled its time.
    - Two columns per thread (more occupancy) were slower: 996 µs.
    - Larger token blocks alone gained only 11%.
    - Staging 5 tokens ahead through shared memory with `cp.async` gave 516 µs; depths 3-6
      gave the same time.

### History

These changes took `pre + norm` from 1.5-1.8x of vLLM to 1.09-1.18x:
1. The controller moved from a single thread to a warp.
2. The RMSNorm was fused into the epilogue.
3. The Sinkhorn iterations now overlap the mix (warp 4 hands `pre` over before them).
4. The epilogue is launched with PDL.

The block backward at T = 16384 went from 4.1 ms to 1.42 ms. The changes:
- The controller backward moved to a warp.
- BF16 gradients are accepted and written.
- d_residual is written once.
- `gemm_bwd` was vectorized (LDS.128 for `draw`).
- `gemm_bwd` stages its residual / dy rows with `cp.async` (1.63 ms to 1.42 ms).

Two attempts were measured and reverted:
- Staging the residual in registers or shared memory before the controller (forward
  epilogue).
- Register prefetching in `gemm_bwd`.
