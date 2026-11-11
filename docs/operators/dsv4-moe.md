# DSv4 MoE

> Part of the DSv4 SM90 batch-invariant kernel set. Design, module map, how to run the
> tests and benchmarks: [DSv4 design](../architecture/dsv4-design.md).

The DSv4 MoE block has two halves that run independently and are summed by the
fixed-order combine:

```
routed expert   x -> mx_quant -> fc1 -> clamp-SwiGLU * p_s -> mx_quant -> fc3     (MXFP8 x MXFP4)
shared expert   x -> fp8_group_quant -> fc1 -> clamp-SwiGLU -> fp8 quant -> fc2   (DeepGEMM FP8)
```

The routed half runs on the SM90 kernels in `csrc/cuda/moe/sm90_fused_moe_mlp.cu`
(fc1 with the activation and re-quantization in its epilogue, then fc2). The shared half
runs on DeepGEMM (`rl_engine/moe/backends/deepgemm_shared_expert.py`). The block that
ties router, plan, both halves and the combine together is `Dsv4MoE`
(`rl_engine/models/deepseek_v4/moe.py`). Other modules under `rl_engine/moe/backends/`
are legacy (see the end of this page).

## Numeric contract

Everything here is driven by one requirement: **train and infer must produce
the same bytes** on the same numeric profile. A tensor that quantizes
differently in the training engine and the rollout engine makes the two paths
drift, and the drift shows up as a logprob gap rather than an error.

Two consequences run through every kernel below.

**Reductions are row-local and order-fixed.** A block amax never crosses a row,
so it never crosses a rank. GEMM reductions walk the full K inside one CTA in a
fixed order, with no split-K and no atomics, so a row's bytes do not depend on
how many rows are in flight: `fwd(x)[t] == fwd(x[t:t+1])` byte for byte.

**Subnormals must survive.** `KERNEL_ALIGN_USE_FAST_MATH=1` compiles
`fmaxf` / `fabsf` / `__fdiv_rn` to their `.ftz` forms, which would flush the
subnormal E8M0 scale `2^-127` and subnormal inputs. The MX path therefore does
its amax as an **integer max over `|x|` bit patterns** (for non-negative floats
the unsigned pattern is monotone in the value), reads `floor(log2(·))` off the
FP32 exponent field, assembles the scale from its bit pattern, and divides
through an explicit non-`.ftz` `div.rn.f32`. A fast-math build emits identical
bytes, which is a regression test rather than a build that refuses to compile.

### MX activation quantization

For each 32-element block along the last dimension (OCP Microscaling: E4M3
elements, one E8M0 shared scale per 32):

```
amax  = max(|x|)                                      # row-local, order-independent
code  = clamp(floor(log2(max(amax, FLT_MIN))) - 8, -127, 127) + 127
code  = 127 if amax == 0                              # all-zero block -> scale 1.0
scale = 2**(code - 127)                               # code 0 is the subnormal 2**-127
elem  = cvt.rn.satfinite.e4m3(clamp(x / scale, -448, 448))
```

E4M3 encoding is clamp-then-RNE (start-kit decision D1), which is what
`cvt.rn.satfinite.e4m3` does in hardware. Saturation is not a corner case:
`x / scale` reaches just under 512 whenever the block amax has a mantissa above
1.75. Non-finite input is fail-closed — the kernels raise rather than emit
garbage codes. Backward is the straight-through estimator `dX = dY`.

The fused kernel implements exactly this recipe in its fc1 epilogue, so `h_q`
is bit-identical to quantizing `h` with the standalone operator.

### Profiles

Profiles on the DSv4 path:

| Profile | Meaning |
|---|---|
| `oracle-fp32-serial-v1` | Byte-equal with the FP32 oracle: serial ascending-k, mul-then-add, no FMA contraction. The MX quantizer holds this. |
| `p5-sm90-fused-mlp-v1` | The routed forward. Deterministic and batch-invariant, **not** byte-equal to the oracle: the 32 products inside one MX block are summed by the tensor core. Everything else follows the recipe above. |
| `p5-sm90-bwd-bf16-v1` | The routed dX backward (see "Backward"). |
| `dsv4-deepgemm-fp8-v1` | The shared expert on DeepGEMM FP8, byte-equal to the vLLM rollout's composition. |
| `dsv4-lora-v1` | The MoE LoRA adapters (see "LoRA"). |

Legacy profiles (not used by DSv4):

| Profile | Meaning |
|---|---|
| `p5-det-gemm-v1` | The det_gemm-backed shared expert. Batch-invariant and TP-equivalent, ~5e-3 from the oracle (its K-tree rounds every 32-wide leaf to BF16). |
| `p5-triton-dot-v1` | The Triton `tl.dot` shared expert, unfused. Batch-invariant; tile reduction order differs from every CUDA backend. |
| `p5-triton-fused-v1` | The portable Triton routed lane (activation fused into fc1). Batch-invariant, byte-equal to nothing else: no Triton exponential reproduces nvcc's `expf`, and the accumulator is flat FP32 rather than a K-tree. |
| `p5-triton-fused-fp8-v1` | The opt-in native FP8 routed lane (gfx94x): e4m3fnuz MFMA, one dot per 32-block, weight scale folded per column. Batch-invariant; different rounding from `p5-triton-fused-v1`. |
| `p5-triton-fused-tp8-v1` | The portable Triton shared expert: fused fc1+SwiGLU, then fc2 over a fixed 8-leaf BF16 K-tree. Batch-invariant and TP-invariant (TP 1/2/4/8 byte-equal with the deterministic all-reduce). |

A backend that cannot hold a profile declares its own rather than relaxing the
tolerance of an existing one.

## Routed expert: the fused SM90 kernel

Inference forward for the routed experts on Hopper:

```
z   = fc1(x_q, W1[e])              gate | up, FP32 accumulator, FP8 WGMMA
h   = BF16(SiLU(min(z_g,10)) * clamp(z_u,-10,10) * p_s)
h_q = mx_quant(h)                  E4M3 + E8M0 per 32 columns
y   = BF16(fc3(h_q, W2[e]))
```

| Item | Location |
|---|---|
| Kernel | `csrc/cuda/moe/sm90_fused_moe_mlp.cu` |
| Provider | `rl_engine.moe.backends.sm90_fused_mlp:Sm90FusedMoeMlp` |
| Tests | `tests/test_sm90_fused_moe_mlp.py` |
| Benchmark | `benchmarks/benchmark_sm90_fused_moe_mlp.py` |
| Build | `KERNEL_ALIGN_MOE_SM90=1` on an SM90 host (compiles the kernel, binds `sm90_moe_*` under `RL_KERNEL_ENABLE_MOE_SM90`, builds for `sm_90a` only, adds `-lcuda` and `RL_KERNEL_ENABLE_SM90`). Other builds do not bind `sm90_moe_*`, so the provider fails closed at construction |

Scope (v1): base weights only (no LoRA), no combine, no TP, H and F
multiples of 128. Fail-closed on a batch that
declares another profile, on CPU tensors, on non-SM90 devices, and on weights whose
block scales cannot be folded exactly (see below).

The provider's per-launch checks are host-side only. Full contract validation
(`ExpertBatch.validate()`) is the caller's job at batch construction: it walks the offsets
on device and SHA-256s every weight byte, which costs ~185 ms at these shapes, far more
than the kernels it would guard.

### Design

**Two modes, same bytes.** The two-launch GEMMs run in one of two modes. They differ
only in where the E4M3 weight operand is built and how the CTAs are tiled. The
provider picks the mode per call. `Sm90FusedMoeMlp(preexpand=True/False)` forces one;
the same switch drives the backward.

| | Inline | Pre-expanded |
|---|---|---|
| Used when | fewer rows than 256 per expert on average | 256 or more rows per expert, and the scratch is at most 4 GiB |
| Weight operand | FP4 converted into FP8 in shared memory, once per token tile | `sm90_moe_expand_w8` writes the folded E4M3 `W` (`E x N x K` bytes, freed after the call) for experts that have rows, by the same conversion code |
| Tile | 64 tokens | 128 tokens |
| Producer | TMA plus FP4 conversion | one thread issues the A and B TMA loads, 6 stages deep |
| Consumers | share the 64 rows, each takes 64 B columns (`m64n64k32`) | each takes 64 rows and all 128 B columns (`m64n128k32`); `setmaxnreg` gives 40 registers to the producer and 232 to the consumers |
| Scheduling | one tile per CTA | persistent |

Both modes promote the same blocks in the same order with the same scales, so they
give the same bytes. Tests force each mode and compare all three GEMMs and the full
forward; outputs also match the pre-change build byte for byte.

**Tiles and threads.** BN = 128 weight rows, BK = 128 K elements (four 32-wide MX
blocks) per pipeline stage. 384 threads: one producer warpgroup and two consumer
warpgroups. Each CTA maps a tile index to `(expert, token block, n-tile)` by scanning
`expert_offsets` on device, so there is no host sync and no descriptor kernel.

Tiles are rasterized in groups of 16 token blocks. Inside a group the token block
varies fastest, then the n-tile, so a wave reuses its A and B tiles from L2.

**Producer (inline).** TMA loads the E4M3 activation tile (64 × 128 B, 128B swizzle) and the
packed E2M1 weight tile (128 rows × 64 B) for stage g+1 before stage g is converted.
The 128 producer threads convert the packed nibbles from the raw staging buffer into
E4M3 bytes in the swizzled B buffer, fold the residual block scale (below), fence the
generic-proxy writes for the async proxy, and arrive on the stage's `full` barrier.
Activation scale codes and weight residuals are prefetched one stage ahead.

**Consumer.** Each MX block is one WGMMA (`k32`) into a fresh block accumulator.

- Two rotating block accumulators keep one WGMMA in flight while the previous block is
  promoted into the FP32 running accumulator: `acc += blk * 2^(sa - 127)`, the
  activation scale only.
- The loop runs stage by stage, four blocks per stage, with the last stage peeled. No
  WGMMA sits behind a branch that depends on the loop index.
- That structure matters. An earlier loop issued WGMMAs inside such branches. ptxas then
  serialized every WGMMA (warning C7518), and this was the main reason the forward was
  slow.

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

A single-launch variant (`h_q` resident in shared memory, fc1 then fc3 in one CTA)
was bit-identical but 3x slower (10.1 ms against 3.2 ms at 32768 rows: only two
pipeline stages fit next to the 128 KB `h_q`). It was removed on 2026-11-11.

**Batch invariance.** Tile constants are compile-time and never chosen by shape (the
mode may follow the batch size because both modes give the same bytes); every
output element is reduced over the full K inside one CTA in ascending block order; no
split-K, no atomics; token blocks are padded with rows whose results are masked. A
row's bytes depend only on its own row, its expert's weights, and the constants:
`fwd(x)[t] == fwd(x[t:t+1])` (tested across expert boundaries).

### Results (H200, H=4096, F=2048, 32 local experts = EP 8)

Times are CUDA-graph replay medians from `benchmarks/benchmark_sm90_fused_moe_mlp.py`; the end-to-end FLOPs are 6·M·H·F.
The baseline is a per-expert BF16 cuBLAS loop on weights dequantized ahead of time
(not timed), with the SwiGLU in PyTorch. It is not batch-invariant. The two-launch
times include the weight expansion when the provider picks it (about 0.36 ms for
`w1` + `w2` at 32 experts, near HBM bandwidth).

| rows | forward | BF16 cuBLAS loop | ratio | TFLOPS |
|---|---|---|---|---|
| 256 | 0.41 ms | 1.06 ms | 0.39 | 31 |
| 2048 | 0.45 ms | 1.05 ms | 0.43 | 230 |
| 8192 | 1.12 ms | 1.31 ms | 0.86 | 369 |
| 32768 | 3.17 ms | 3.11 ms | 1.02 | 521 |

How 32768 rows went from 6.2 ms (inline, serialized WGMMAs) to 3.17 ms:

| Change | Two-launch time | Ratio |
|---|---|---|
| Inline mode, serialized WGMMAs | 6.2 ms | — |
| Pre-expanded weights, operands by TMA | 4.36 ms | 1.38 |
| Grouped rasterization | 4.22 ms | 1.34 |
| Persistent CTAs | 4.28 ms | 1.37 (no gain on its own) |
| 128-token tiles with `m64n128k32` | 3.17 ms | 1.02 |

Things that did not help:

- Deeper TMA prefetch in inline mode was slower (4.83 to 5.96 ms for fc1 alone). The
  TMA thread also converts, so it ended up waiting on the consumers.
- Grouped rasterization made no difference while the WGMMAs were serialized.

Accuracy against the FP32 reference: `z` max relative error 3e-5; `y` max relative error
1e-2 (0.2% of `h` codes move by one FP8 ulp because of accumulation order, then the BF16
output round).

### Known limits and next steps

1. **The exact activation-scale promote is the remaining ceiling.** It costs one FMA per
   output element per 32-wide block.
   - Option 1: fold the activation scales when a row's blocks share exponents. This is
     row-local, so it stays batch-invariant.
   - Option 2: accept a coarser scale granularity, as a separate profile.
2. Weights whose block scales span more than 8 binades within a column are rejected; an
   unfolded fallback kernel could serve them.
3. Not implemented in the kernel: LoRA deltas (the model adds them around it, see
   "LoRA"), Blackwell block-scaled MMA (would remove the FP4
   conversion and the promote entirely). Backward: see below.

### Backward (SM90, frozen base)

Profile `p5-sm90-bwd-bf16-v1` (`csrc/cuda/moe/sm90_moe_mlp_backward.cu`,
`Sm90FusedMoeMlp.backward`). No `dW`: the routed experts are frozen. Three steps:

1. Recompute `z = fc1(x_q)` (FP32) with the forward's fc1 kernel, so `z` is the
   forward's own pre-activation.
2. `dh = BF16(dy @ W2[e])` on a BF16 WGMMA grouped GEMM. The clamp-SwiGLU backward runs
   in the epilogue (the oracle's formulas), giving `dz = [dgate | dup]` in BF16 and the
   per-row `dp_s`. `dp_s` is summed per 128-column tile in a fixed order, then a second
   kernel adds the tiles in ascending order.
3. `dx = dz @ W1[e]` (FP32 out) on the same GEMM.

Tiles: 128 tokens x 128 output columns x 64 reduced rows per stage, 5 stages. The
whole reduction stays inside one CTA (no split-K, no atomics), so a row's bytes depend
only on that row and its expert: batch-invariant and run-to-run identical (tested on a
sub-batch across an expert boundary). MXFP4 is expanded to BF16 exactly; scale codes
must lie in [2, 252] (fail-closed). There are two ways to get the BF16 weight tile,
and they give the same bytes (tested):

- **Inline**: one producer warpgroup converts the raw E2M1 tile into shared memory
  every stage. Each 128-row token tile repeats this conversion.
- **Pre-expanded**: `sm90_moe_dequant_t` (the same conversion code) writes `W^T` in
  BF16 once per call, only for experts that have rows, and the GEMM loads it by TMA.
  The provider uses this when the average expert has at least 256 rows and the scratch
  (`E x N x K x 2` bytes, freed after the call) is at most 4 GiB.

CTAs are rasterized in groups of 8 token tiles, with the output tile varying fastest,
so the A tiles stay in L2.

Results (H200, H 4096, F 2048, 32 local experts = EP 8, tokens spread evenly,
CUDA-graph replay; `benchmarks/benchmark_sm90_moe_backward.py`). Baseline (not
batch-invariant): weights dequantized to BF16 ahead of time (not timed), a per-expert
cuBLAS loop for the fc1 recompute, `dh` and `dx`, and the SwiGLU backward in PyTorch.

| rows | ours | baseline | ratio |
|---|---|---|---|
| 256 | 1.14 ms | 2.58 ms | 0.44 |
| 2048 | 1.19 ms | 2.70 ms | 0.44 |
| 8192 | 2.19 ms | 3.37 ms | 0.65 |
| 32768 | 5.71 ms | 7.72 ms | 0.74 |

These numbers include the faster forward fc1 (the recompute). Before the forward
work, 32768 rows took 8.22 ms (1.07x).

Breakdown at 32768 rows:

| Step | Inline | Pre-expanded | cuBLAS |
|---|---|---|---|
| fc1 recompute | 4.46 ms (247 TFLOPS) | 4.46 ms | 1.52 ms |
| Weight expansion (`w1` + `w2`) | — | 0.65 ms | — |
| `dh` + SwiGLU backward | 2.84 ms | 1.41 ms (389 TFLOPS) | 0.77 ms |
| `dx` | 4.39 ms | 1.76 ms (626 TFLOPS) | 1.46 ms |

The breakdown was measured before the forward work. The fc1 recompute is now about
1.6 ms, not 4.46 ms (see the forward results above).

Accuracy is measured against an FP32 chain that follows the oracle on the same `z`:
`dx` rel < 2e-2, `dp_s` rel < 1e-2. For the whole block (`tests/models/test_dsv4_moe_layer.py`),
the routed part has rel-L2 < 2e-2. With the shared expert's FP8 dgrad included, rel-L2 is
< 8e-2.

### LoRA (dsv4-lora-v1)

`Dsv4MoE.lora` is a `MoeLora` (`rl_engine/models/deepseek_v4/lora.py`):
- routed `fc1` (H to 2F, gate rows then up rows) and `fc2` (F to H) adapters, **shared by
  every routed expert**, as Megatron-Bridge's default `share_expert_adapters=True` with
  EP = 1;
- optional shared-expert `fc1` / `fc2` adapters.

The forward is unmerged, so the rollout runs the same bytes. The base weights stay MXFP4.

**Routed forward.** The fused fc1 epilogue cannot add a delta, so with adapters the block
takes four steps:
1. `fc1_z` (FP32 z).
2. `moe_swiglu_quant` (`csrc/cuda/moe/moe_swiglu_quant.cu`). It is the fused epilogue as a
   standalone kernel, byte-equal to it (tested). It adds the per-token delta
   `z_lora = scale * (BF16(x A1^T) B1^T)` in FP32 as it reads z, computed once per token
   rather than once per routed row.
3. `fc3`.
4. `y = BF16(FP32(y) + scale * (BF16(h A2^T) B2^T))`, with `h` the BF16 SwiGLU output
   (`p_s` already in it).

Zero adapters (B = 0) give exactly the fast path's bytes, and the block stays
batch-invariant.

**Routed backward.**
- `sm90_moe_bwd_dh_swiglu` takes an optional `dh_add` (the fc2 adapter's dX), added to
  `dy @ W2` in FP32 before the BF16 round. The recomputed z includes the fc1 delta.
- The fc1 adapter's dX is summed per token (slot order) before `@ A1`, and so is `dA1`.
- Shared expert: `lora_apply` after `fc1` and after `fc2`. The BF16 `h` comes from the fused
  SwiGLU + FP8 quant kernel (`return_h`).
- dX and every dA / dB match an FP32 autograd reference
  (`tests/models/test_dsv4_moe_lora.py`).

**Cost.** H200, E 256, H 4096, F 2048, rank 32, against the same block without adapters:

| T | forward | forward + backward |
|---|---|---|
| 64 | 1.02x | 1.05x |
| 1024 | 1.09x | 1.11x |
| 4096 | 1.17x | 1.21x |

Most of the remaining cost is the FP32 `z` round trip, which the fused epilogue avoids.

**Expert parallelism.** Megatron-Bridge broadcasts the shared expert adapter from EP rank 0
and all-reduces its gradients over the EP group, so it is one global adapter. Here:
- The token's rank computes `u1 = BF16(x A1^T)` and sends it with the MXFP8 row
  (`ep_send`, r values per row). The expert's rank applies `B1`. The GEMMs are row-invariant,
  so the forward bytes equal EP = 1.
- In the backward, `du = dz B1` returns along the forward's splits for the per-token dX / dA1.
- The routed adapters' gradients are all-reduced over the EP group. The shared-expert adapter
  is an ordinary data-parallel parameter.
- `scripts/check_dsv4_moe_ep.py --lora` checks EP = R against EP = 1: outputs and dx
  bytewise, adapter gradients to 1e-4. **It has not run yet: no 2-GPU allocation was
  available.**

**Not implemented:** per-expert adapters (`share_expert_adapters=False`).

## Route plan and combine (EP = 1)

The block glue between the router (`docs/operators/dsv4-router.md`) and the two
expert halves. Golden: `rl_engine/kernels/ops/pytorch/moe/route_plan.py`; CUDA
slot sum: `csrc/cuda/moe/moe_slot_sum.cu`; CUDA plan: `csrc/cuda/moe/moe_route_plan.cu`
(`build_plan_cuda`); block: `rl_engine/models/deepseek_v4/moe.py`
(`Dsv4MoE`, forward).

- **Plan.** The `T * 6` (token, slot) pairs are sorted by `(expert, token, slot)`
  (golden: stable sort on the expert id, then `searchsorted` for `expert_offsets`);
  `token` is the global token id, so with expert parallelism an expert's rows keep this
  order whichever rank sends them. Integer work only, no host sync. The block builds it
  with a CUDA counting sort: per-chunk histograms, a column scan, then a stable scatter
  ranked within each warp by `__match_any_sync`. A stable sort's result is unique, so this
  equals the golden exactly (tested for T = 0..32768, 256 / 384 experts, int32 / int64 ids,
  skewed routing). It also returns each row's token id.
- **Dispatch.** Rows, `p_s` and `output_slot` are gathered in plan order. The MXFP8
  activation is quantized once per token and its codes / scales gathered (row-local,
  so equal to quantizing the rows). The routed kernel applies `p_s` once.
- **Combine** (fixed slot order). Per token in FP32: copy the slot-0 row, add slots 1..5 in
  order, add the shared expert once and the residual once, round to BF16.
- **Backward.** `dy` goes to every row of its token (exact); the expert-input gradient
  fans back in with the same slot-sum kernel (FP32 out, + the shared expert's `dx`);
  `dL/dweights = dp_s[inv]` (exact). The routed-expert backward is the SM90 kernel
  above.

Checks (H200, `tests/ops/moe/test_route_plan.py`, `tests/models/test_dsv4_moe_layer.py`):
CUDA slot sum byte-equal to the golden for every dtype combination (signed zero
included); the whole block (router -> plan -> routed experts -> combine + DeepGEMM shared
expert + residual) is batch-invariant across splits and run to run, and within 3% of an
FP32 reference on the same MX operands.

Graph-replay time, microseconds (H 4096, 256 experts). Baseline: `index_add_`
(atomic FP32 scatter, not BI) plus the merge:

| T | plan (golden) | plan (CUDA) | combine | index_add | ratio | fan-in | index_add | ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 18.8 | 9.1 | 8.1 | 19.6 | 0.41 | 8.1 | 8.9 | 0.91 |
| 64 | 44.4 | 9.5 | 8.0 | 30.5 | 0.26 | 8.0 | 16.0 | 0.50 |
| 256 | 47.8 | 10.5 | 8.2 | 80.2 | 0.10 | 8.9 | 43.1 | 0.21 |
| 1024 | 97.6 | 12.1 | 24.6 | 301.7 | 0.08 | 34.1 | 173.7 | 0.20 |
| 4096 | 86.0 | 16.3 | 76.9 | 1201.5 | 0.06 | 116.1 | 699.8 | 0.17 |
| 16384 | 92.4 | 35.5 | 283.4 | 4757.5 | 0.06 | 439.9 | 2807.9 | 0.16 |

The CUDA plan (three small kernels) is 2-8x faster than the golden's PyTorch sort +
searchsorted + scatter.

## Shared expert

Profile `dsv4-deepgemm-fp8-v1` (`rl_engine/moe/backends/deepgemm_shared_expert.py`). The
checkpoint stores the shared expert as FP8 with 128x128 weight blocks, and the vLLM
rollout runs it through DeepGEMM; this provider makes the same calls:

```
x_q, s_x = fp8_group_quant(x)                    per-128 group, UE8M0 scales
z        = fp8_gemm_nt(x_q, W1)                  DeepGEMM, BF16 out
h_q, s_h = silu_and_mul_clamp_quant(z, 10)       vLLM's clamped SwiGLU fused with the FP8 quant
y        = fp8_gemm_nt(h_q, W2)                  DeepGEMM, BF16 out
```

- `csrc/cuda/activation/silu_and_mul_clamp.cu` fuses the clamped SwiGLU with the FP8 group
  quant. Its bytes equal vLLM's two separate kernels for every BF16 gate value.
- Backward: dX only (weights frozen). The default is the FP8 DeepGEMM dgrad through the
  precomputed `W.T`; `backward="bf16_det_gemm"` uses det_gemm in BF16 as a reference
  (fast only with a `KERNEL_ALIGN_DET_GEMM_SM90=1` build).
- Forward and dX are batch-invariant and run-to-run identical
  (`tests/test_deepgemm_shared_expert.py`). Relative error against FP32 on the dequantized
  weights (H200, T = 512 / 2048): y 4.7%, FP8 dX 3.8%, BF16 dX 0.74%.
- H200, H 4096, F 2048, T = 16..4096: forward + backward 0.71-0.78x of eager BF16
  (`benchmarks/benchmark_shared_expert_mlp.py`).

## Legacy backends (not used by DSv4)

These stay in the tree but are not on the DSv4 path and have no DSv4 tests:

- The FP32-oracle provider harness (`rl_engine/moe/provider.py`) and its per-operator
  backends: `backends.mxfp8_act_quant` (provider wrapper; DSv4 calls
  `rl_engine/kernels/ops/cuda/moe/mxfp8_act_quant.py` directly), `backends.grouped_gemm`,
  `backends.lora_delta`, `backends.clamp_swiglu`, `backends.shared_expert` (strict
  serial and det_gemm BF16 shared experts, profiles `oracle-fp32-serial-v1` /
  `p5-det-gemm-v1`).
- The portable Triton lanes (`backends.triton_fused_mlp`, `rl_engine/moe/triton_*`;
  profiles `p5-triton-*`), written for ROCm (MI300X).
- `csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm*.cu`, earlier routed-expert GEMMs superseded by
  `sm90_fused_moe_mlp.cu`.

The FP32 oracle (`rl_engine/moe/oracle.py`), the batch contract (`rl_engine/moe/contract.py`)
and the MX format (`rl_engine/moe/mx_format.py`) are still used as references by the DSv4
MoE tests. The MX quantizer reproduces the committed golden manifest
(`tests/fixtures/moe/golden_hashes.json`).
