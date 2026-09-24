# SM90 Fused Routed-Expert MLP (MXFP8 × MXFP4)

Inference forward for the DSv4 routed experts on Hopper:

```
z   = fc1(x_q, W1[e])              gate | up, FP32 accumulator, FP8 WGMMA
h   = BF16(SiLU(min(z_g,10)) * clamp(z_u,-10,10) * p_s)
h_q = mx_quant(h)                  E4M3 + E8M0 per 32 columns
y   = BF16(fc3(h_q, W2[e]))
```

Numeric profile `p5-sm90-fused-mlp-v1`. **Deterministic and batch-invariant, not
byte-aligned to `oracle-fp32-serial-v1`**: the 32 products inside one MX block are
summed by the tensor core; everything else follows the P5 recipe exactly.

| Item | Location |
|---|---|
| Kernel | `csrc/cuda/moe/sm90_fused_moe_mlp.cu` |
| Provider | `rl_engine.moe.backends.sm90_fused_mlp:Sm90FusedMoeMlp` |
| Tests | `tests/test_sm90_fused_moe_mlp.py` |
| Benchmark | `benchmarks/benchmark_sm90_fused_moe_mlp.py` |
| Build | `KERNEL_ALIGN_MOE_SM90=1` (adds `sm_90a`, `-lcuda`, `RL_KERNEL_ENABLE_SM90`) |

Scope (v1): base weights only (no LoRA), no backward, no combine, no TP, H and F
multiples of 128, F ≤ 2048 for the single-launch path. Fail-closed on a batch that
declares another profile, on CPU tensors, on non-SM90 devices, and on weights whose
block scales cannot be folded exactly (see below).

The provider's per-launch checks are host-side only. Full contract validation
(`ExpertBatch.validate()`) is the caller's job at batch construction: it walks the offsets
on device and SHA-256s every weight byte, which costs ~185 ms at these shapes, far more
than the kernels it would guard.

## Design

**Tiles and threads.** BM = 64 tokens, BN = 128 weight rows, BK = 128 K elements
(four 32-wide MX blocks) per pipeline stage, 4 stages. 384 threads: one producer
warpgroup and two consumer warpgroups that each own 64 output columns. One CTA per
SM. Grid = (upper bound on token blocks over all experts) × (N tiles); a CTA maps
`blockIdx.x` to `(expert, block)` by scanning `expert_offsets` on device, so no host
sync and no descriptor kernel.

**Producer.** TMA loads the E4M3 activation tile (64 × 128 B, 128B swizzle) and the
packed E2M1 weight tile (128 rows × 64 B) for stage g+1 before stage g is converted.
The 128 producer threads convert the packed nibbles from the raw staging buffer into
E4M3 bytes in the swizzled B buffer, fold the residual block scale (below), fence the
generic-proxy writes for the async proxy, and arrive on the stage's `full` barrier.
Activation scale codes and weight residuals are prefetched one stage ahead.

**Consumer.** Each MX block is one `wgmma.m64n64k32.e4m3.e4m3.f32` into a fresh block
accumulator. Three rotating block accumulators keep two WGMMAs in flight while a third
block is promoted into the FP32 running accumulator. The promote is 32 FMAs per thread:
`acc += blk * 2^(sa - 127)`, the activation scale only.

**Exact MX scaling with one FMA per element.** Applying both block scales per element
(`acc += blk * sa * sw`) costs a multiply and an FMA per element per block and caps the
kernel at roughly a quarter of FP8 peak on issue slots. Instead, for every weight
column `c` a reference exponent `ref_c = max_j sw[c][j] - 6` is computed once (weights
are frozen), and the converter writes `e2m1 * 2^(sw[c][j] - ref_c)` as the E4M3 value.
Every E2M1 magnitude times a residual in [-8, 6] is an exact E4M3 number, and every
term of a column is scaled by the same power of two, so the FP32 sum is bit-identical
to promoting both scales per block; `2^(ref_c - 127)` is applied once in the epilogue.
`sm90_moe_prepare_weight_ref` verifies the residual range and fails closed otherwise.

**Epilogues.**
- `fc1_z`: FP32 `z` (validation).
- `fc1_swiglu_quant`: the tile is laid out `[gate 0-31 | up 0-31 | gate 32-63 | up 32-63]`
  so each consumer warpgroup holds matching gate/up columns in the same thread. Clamp,
  SiLU, `p_s`, one BF16 round, 32-column amax via a quad shuffle, E8M0 code via `frexpf`,
  `cvt.rn.satfinite.e4m3`; writes `h` codes and scales.
- `fc3`: BF16 `y`.
- `fused`: one CTA per token block runs every fc1 tile with `h_q` written into a
  shared-memory-resident buffer in the WGMMA A layout, then every fc3 tile reading it.
  Bit-identical to the two-launch path.

**Batch invariance.** Tile constants are compile-time and never chosen by shape; every
output element is reduced over the full K inside one CTA in ascending block order; no
split-K, no atomics; token blocks are padded with rows whose results are masked. A
row's bytes depend only on its own row, its expert's weights, and the constants:
`fwd(x)[t] == fwd(x[t:t+1])` (tested across expert boundaries, fused and two-launch).

## Results (H100 SXM, H=4096, F=2048, E=8)

Measured on an idle GPU, 30 iterations after 5 warmups (end-to-end FLOPs = 6·M·H·F):

| rows | fc1+SwiGLU+quant | fc3 | two-launch | fused (2 stages) | BF16 cuBLAS loop | two-launch TFLOPS |
|---|---|---|---|---|---|---|
| 512 | 0.08 ms | 0.04 ms | **0.12 ms** | 2.68 ms | 0.60 ms | 208 |
| 2048 | 0.30 ms | 0.15 ms | **0.45 ms** | 2.70 ms | 0.60 ms | 227 |
| 8192 | 1.18 ms | 0.56 ms | **1.71 ms** | 2.52 ms | 0.85 ms | 242 |
| 32768 | 4.49 ms | 2.22 ms | **6.74 ms** | 9.75 ms | 3.07 ms | 245 |

The two-launch path beats the BF16 cuBLAS loop by 4.8x at 512 rows and 1.3x at 2048,
where cuBLAS pays per-expert launch overhead on small GEMMs; above that cuBLAS wins by
about 2x. FP8 dense peak is ≈ 1979 TFLOPS, so the kernel sits near 12% of peak, capped by
the per-block activation-scale promote.

The fused path is flat at ~2.6 ms from 512 to 8192 rows because each CTA walks the whole
chain serially: 32 fc1 tiles × 32 stages plus 32 fc3 tiles × 16 stages = 1536 stages. Only
the CTA *count* grows with M, so wall time stays constant until CTAs exceed the 132 SMs;
at 32768 rows that is 512 CTAs ≈ 4 waves ≈ 4 × 2.5 ms.

Accuracy against the FP32 reference: `z` max relative error 3e-5; `y` max relative error
1e-2 (0.2% of `h` codes move by one FP8 ulp because of accumulation order, then the BF16
output round).

## Known limits and next steps

1. **Fused path is slower than two launches.** The 128 KB resident `h_q` leaves room for
   only two 33 KB stages, which exposes TMA latency, and each CTA walks 1536 stages
   serially. A BK=64 variant (16.5 KB stages, 3–4 deep) is the fix.
2. **Exact activation-scale promote is the remaining ceiling** (~45% of FP8 peak in this
   structure). Options: fold activation scales when a row's blocks share exponents (row
   local, still batch-invariant), or accept a coarser scale granularity as a separate
   profile.
3. Weights whose block scales span more than 8 binades within a column are rejected; an
   unfolded fallback kernel could serve them.
4. Not implemented: LoRA deltas, backward (`z` is never written), Blackwell block-scaled
   MMA (would remove the FP4 conversion and the promote entirely).
