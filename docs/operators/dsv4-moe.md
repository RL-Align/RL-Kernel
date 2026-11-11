# DSv4 MoE

The DSv4 MoE block has two halves that run independently and are summed by the
P6 combine:

```
routed expert   x -> mx_quant -> fc1 -> clamp-SwiGLU * p_s -> mx_quant -> fc3
shared expert   x -> fc1 -> one-round SwiGLU -> fc2
```

The routed half is the production path and runs on **one SM90 GEMM core**
(`csrc/cuda/moe/sm90_fused_moe_mlp.cu`): fc1 with the activation and re-quantization
fused into its epilogue, then fc2. The shared half and the individual
routed operators also exist as standalone per-operator kernels; those are
reference artifacts that pin the numerics operator by operator, not the path
production takes. See `rl_engine/moe/backends/` for the split.

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

| Profile | Meaning |
|---|---|
| `oracle-fp32-serial-v1` | Byte-equal with the FP32 oracle: serial ascending-k, mul-then-add, no FMA contraction. The MX quantizer and the strict shared-expert kernels hold this. |
| `p5-sm90-fused-mlp-v1` | The fused routed kernel. Deterministic and batch-invariant, **not** byte-equal: the 32 products inside one MX block are summed by the tensor core. Everything else follows the recipe above. |
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
  so equal to quantizing the rows). P5 applies `p_s` once.
- **Combine** (P6 policy). Per token in FP32: copy the slot-0 row, add slots 1..5 in
  order, add the shared expert once and the residual once, round to BF16.
- **Backward.** `dy` goes to every row of its token (exact); the expert-input gradient
  fans back in with the same slot-sum kernel (FP32 out, + the shared expert's `dx`);
  `dL/dweights = dp_s[inv]` (exact). The routed-expert backward is the SM90 kernel
  above.

Checks (H200, `tests/ops/moe/test_route_plan.py`, `tests/models/test_dsv4_moe_layer.py`):
CUDA slot sum byte-equal to the golden for every dtype combination (signed zero
included); the whole block (router -> plan -> P5 -> combine + DeepGEMM shared expert +
residual) is batch-invariant across splits and run to run, fused == two-launch, and
within 3% of an FP32 reference on the same MX operands.

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

Every valid token runs `fc1 -> one-round SwiGLU -> fc2` once, on BF16 frozen
weights. Backward returns only `dX` (no `dW`). The shared output stays
independent of the routed path; the combine belongs to P6.

```
z  = x @ w_fc1.T            # BF16 operands, FP32 serial ascending-k, mul-then-add
h  = BF16(SiLU(gate) * up)  # FP32 math, single round; no clamp, no p_s
y  = BF16(h @ w_fc2.T)      # FP32 accumulate, one round
dX = FP32(dz @ w_fc1)       # dh, dz round BF16 at operator edges
```

Note the difference from the routed path: **no clamp and no route weight**. The
one-round SwiGLU core runs in shared mode (`p_s = None`) and is the same core
P5-2 extends with clamp, `p_s`, and `dp_s`.

The strict kernels hold `oracle-fp32-serial-v1` byte for byte: one lane owns one
output element and reduces serially in ascending k with `__fmul_rn` / `__fadd_rn`
(CUDA) or uncontracted IEEE FP32 arithmetic (Triton), so there is no cross-lane
floating-point reduction.

**Sigmoid is transcendental**, so its bits follow the libm implementation. nvcc's
`expf` — which is what `torch.sigmoid` uses — bit-matches, while `tl.exp` and
libdevice `__nv_expf` do not (10–45% of values differ by one ulp). The Triton
path therefore takes the sigmoid tensor from `torch.sigmoid` as a kernel input
and fuses only the remaining SwiGLU math.

A second backend composes the shared expert from `det_gemm` (profile
`p5-det-gemm-v1`): batch-invariant and TP-equivalent because det_gemm's
mid-split K-tree makes a contiguous half-K shard one child of the tree, at the
cost of byte-equality.

## The portable Triton lane

Everything above is Hopper-specific: TMA, `wgmma`, warp specialization, an
explicit shared-memory pipeline. `rl_engine/kernels/ops/triton/moe/fused_mlp.py`
is the MoE block written to run on CDNA, and it is the lane that gets tuned on
a ROCm host. Maintainability comes first there, performance second, so it is
not an attempt to reproduce the CUDA kernels.

| # | Kernel | Math | Invariance |
|---|---|---|---|
| 1 | `fused_shared_fc1_swiglu` | BF16 `x @ W1[gate\|up]ᵀ` → `BF16(SiLU(g)·u)` | batch; TP by layout (column-parallel) |
| 2 | `shared_fc3` | BF16 `h @ W2ᵀ` over a fixed 8-leaf K-tree | batch; TP 1/2/4/8 byte-equal |
| 3 | `routed_fc1_swiglu_quant` | MXFP8×MXFP4 grouped fc1 → clamp-SwiGLU·p_s → MX quant | batch |
| 4 | `routed_fc3` | MXFP8×MXFP4 grouped fc3 → BF16 | batch |

| Item | Location |
|---|---|
| Kernels | `rl_engine/kernels/ops/triton/moe/fused_mlp.py` |
| Providers | `backends.triton_fused_mlp:TritonFusedMoeMlp`, `backends.shared_expert:TritonFusedSharedExpertProvider` |
| Tests | `tests/test_triton_fused_moe_mlp.py` |
| Benchmark | `benchmarks/benchmark_triton_fused_moe_mlp.py` |

**No FP8 `tl.dot`.** `tl.float8e4nv` is NVIDIA's e4m3**fn**; CDNA3's native FP8
is e4m3**fnuz** (`tl.float8e4b8`), a different bias with no infinities. Rather
than branch the numerics on the target, both MX operands are decoded to BF16
and fed to a plain BF16 `tl.dot`. The decode is exact — E4M3 keeps 4
significand bits, E2M1 keeps 2, BF16 has 8 — so nothing is lost, and folding the
E8M0 scale out of the inner sum means no dequantized tensor is materialized and
no fresh accumulator is needed per 32-wide block. On gfx94x the activation
decode uses the hardware fnuz convert: an e4m3fn byte read as fnuz is exactly
half its value, so `×2` restores it, and the one code that differs (0x80: −0 in
fn, NaN in fnuz) is mapped to +0. That changes speed, not bytes.

**The loaders, not the math, set the speed.** Gathering the packed E2M1 bytes
along k (each byte loaded twice, one nibble per load) held routed fc1 at
~65 TFLOPS on MI300X. Loading the bytes as a contiguous `[BN, BK/2]` tile,
decoding both nibbles, and restoring k order with `tl.join` + `reshape` before
the transpose is 2.5–3.5× faster and bit-identical.

**Native FP8 is opt-in, and slower on MI300X.** `fp8=True` runs the routed
GEMMs on e4m3fnuz MFMA: the activation bytes are fed as fnuz unchanged (×2 in
the epilogue), the E2M1 weight is converted with its block scale folded against
a per-column reference exponent (`prepare_mxfp4_fp8`, exact for residuals in
[−9, 5], fail-closed otherwise), and the activation scale is promoted per
32-block. Profile `p5-triton-fused-fp8-v1`. It loses to the BF16 lane: on one
8192×4096×4096 GEMM a whole-tile FP8 dot takes ~0.35 ms, but exact MX semantics
force one dot per 32-block because the activation scale changes every 32 k, and
that structure costs ~1.0 ms even with pre-converted weights (~1.4 ms with the
in-kernel conversion) against ~1.16 ms for the BF16 lane. 16×16×32 MFMA,
`waves_per_eu` and `kpack` do not help. End to end, routed fc1 is ~2× slower
(E=8 M=8192: 3.3 vs 1.7 ms; E=256 M=8192: 13.1 vs 6.6 ms). What would change
this: a coarser activation scale (a different numeric contract; measured ~1.8×
faster than the BF16 lane), or gfx950's MX-scaled MFMA via `tl.dot_scaled`.

**No single-launch variant.** Triton has no user-managed shared memory, so `h_q`
cannot be kept resident across the two GEMMs. The routed lane is two launches
only — which is the faster CUDA path anyway.

**TP invariance of the shared expert.** fc1 is column-parallel: every output
column reduces its whole K inside one program at any TP size, so a rank's
`[gate_shard; up_shard]` launch is byte-equal to the matching slice of the TP=1
output. fc2 is row-parallel, so its K *is* the split axis. `shared_fc3` fixes
the reduction to the deterministic all-reduce's tree
(`fixed_tree_reduce` in `csrc/cuda/distributed/deterministic_collective.cu`):

```
F = [ s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 ]          8 leaves of F/8
leaf  = BF16(FP32 ascending-BK sum over the leaf)
y     = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))   each + is one BF16 add
```

At TP=t a rank owns 8/t adjacent leaves — a whole subtree — and sends that
subtree's BF16 value; the collective's tree over t ranks is the top of the same
tree. TP=1, 2, 4 and 8 give identical bytes, and the partials travel as BF16.
Leaves are F/8 wide whatever BK is (a ragged last tile is zero-masked), so F
only has to be a multiple of 8. The price is seven BF16 rounds more than a flat
FP32 sum, still well inside the shared expert's oracle tolerance. The K-tree
does not apply to fc1: `SiLU(g1+g2)*(u1+u2) != SiLU(g1)*u1 + SiLU(g2)*u2`.

**No byte-equality with CUDA.** Sigmoid: measured over 262144 elements,
`tl.sigmoid` differs from nvcc's `expf` in 6 elements and libdevice `exp` in 2,
so the strict Triton shared expert has to import `torch.sigmoid` as a tensor to
stay byte-equal, and a *fused* kernel cannot — the activation is inside the
epilogue.

**Batch invariance** is what the tests assert: tiles are pinned per
architecture (`_TILES` in `fused_mlp.py`) and never chosen from M, there is no
split-K and no atomic, and each output tile reduces the full K inside one
program. Byte-equality holds across sub-batch slicing, single-row launches on
both sides of every expert boundary, run to run, and — for 37 experts with
empty and one-row experts — against per-expert launches. The program → expert
map is a binary search over a device-side block prefix (one tiny launch, no
host sync), so it stays cheap at DSv4's expert count.

On gfx942 every BM/BN/BK/warps combination tried produced identical bytes for
all four kernels (the FP32 accumulator walks K in ascending MFMA steps however
it is tiled). That is measured, not guaranteed by Triton, so the pin stays; it
does mean a shape-bucketed tile choice could be added later behind a
cross-config equality test.

### Results (MI300X, H=4096, F=2048)

Routed, E=8 (`benchmarks/benchmark_triton_fused_moe_mlp.py`):

| rows | fc1 | fc3 | total | before the loader rewrite |
|---|---|---|---|---|
| 512 | 0.23 ms | 0.18 ms | 0.41 ms | 0.80 ms |
| 2048 | 0.96 ms | 0.49 ms | 1.45 ms | 1.73 ms |
| 8192 | 1.74 ms | 0.78 ms | 2.53 ms | 6.30 ms |

Max relative error against FP64 is unchanged (4.1e-2 at M=512, 3.7e-2 at
M=2048) — the MX quantization floor.

With E=256 and 32 rows per expert (M=8192) routed fc1 takes 6.5 ms and fc3
2.8 ms: each weight element is decoded for only a handful of rows, so the BF16
decode, not the MFMA, is the ceiling in that regime.

Shared expert, TP=1:

| rows | fc1+SwiGLU | fc3 | total | torch BF16 chain (unfused) |
|---|---|---|---|---|
| 512 | 0.08 ms | 0.06 ms | 0.14 ms | 0.18 ms |
| 2048 | 0.22 ms | 0.21 ms | 0.43 ms | 0.49 ms |
| 8192 | 0.74 ms | 0.74 ms | 1.48 ms | 1.30 ms |

### Versus torch (MI300X, H=4096, F=2048)

GPU device time per forward (sum of kernel durations from `torch.profiler`, so
host launch overhead is excluded). "Performance" is torch time / Triton time:
above 100% the Triton lane is faster.

**Shared expert (fc1 + SwiGLU + fc3), per rank.** Baseline: the unfused torch
BF16 chain, `x @ w1ᵀ` → SiLU·mul → `h @ w2ᵀ` on hipBLASLt, on the same
`[gate_shard; up_shard]` / `w2[:, F/tp]` shards. The all-reduce is excluded;
both sides send the same BF16 `[T, H]` payload.

| TP | T | Triton | torch | performance |
|---|---|---|---|---|
| 1 | 512 | 0.134 ms | 0.109 ms | 82% |
| 1 | 2048 | 0.425 ms | 0.295 ms | 69% |
| 1 | 8192 | 1.481 ms | 0.985 ms | 66% |
| 2 | 512 | 0.097 ms | 0.073 ms | 76% |
| 2 | 2048 | 0.208 ms | 0.178 ms | 85% |
| 2 | 8192 | 0.716 ms | 0.506 ms | 71% |
| 4 | 512 | 0.085 ms | 0.050 ms | 59% |
| 4 | 2048 | 0.123 ms | 0.114 ms | 93% |
| 4 | 8192 | 0.371 ms | 0.314 ms | 85% |

The shared lane runs at 59–93% of hipBLASLt. The gap is the price of
invariance: hipBLASLt picks its kernel per shape, while this lane has
fixed tiles, no split-K, and fc3's fixed 8-leaf BF16 tree. In eager wall-clock time the
order flips at small batches (TP=2/4, T ≤ 2048: 1.5–1.6× faster than torch)
because the lane is two launches against torch's ~7.

**Routed experts (fc1 + clamp-SwiGLU·p_s + MX quant + fc3), one GPU.**
Baseline: the same math in torch with the MXFP4 weights dequantized to BF16 on
every call (the torch equivalent of keeping weights in MXFP4), MX activations
dequantized, `h` MX-quantized between the GEMMs, and `torch._grouped_mm` for
both GEMMs. E=256 uses top-8 routing (M = 8T).

| experts | T | rows / expert | Triton | torch | performance |
|---|---|---|---|---|---|
| 8 | 512 | 64 | 0.40 ms | 4.62 ms | 1140% |
| 8 | 2048 | 256 | 0.71 ms | 5.04 ms | 710% |
| 8 | 8192 | 1024 | 2.54 ms | 6.37 ms | 250% |
| 256 | 512 | 16 | 9.20 ms | 138.3 ms | 1500% |
| 256 | 2048 | 64 | 9.72 ms | 141.5 ms | 1460% |
| 256 | 8192 | 256 | 20.7 ms | 154.1 ms | 740% |

Against torch holding pre-dequantized BF16 weights instead (4× the weight
memory, ~13 GB per layer at E=256) the routed lane is roughly even, 70–140%:
it falls behind only at E=256 with 16 rows per expert, where the in-kernel E2M1
decode is the bound.

Max relative difference from torch: 5e-3 to 1e-2 for the shared expert (BF16
rounding), 3e-2 to 4e-2 for the routed experts (the MX quantization floor).

### Results (H100 SXM, H=4096, F=2048, E=8; before the loader rewrite)

| rows | Triton fc1 | Triton fc3 | Triton total | CUDA two-launch | ratio |
|---|---|---|---|---|---|
| 512 | 0.52 ms | 0.23 ms | 0.74 ms | 0.12 ms | 6.2x |
| 2048 | 1.91 ms | 0.85 ms | 2.76 ms | 0.45 ms | 6.1x |
| 8192 | 7.41 ms | 3.31 ms | 10.72 ms | 1.71 ms | 6.3x |

The gap is the three things given up: FP8 tensor cores (BF16 `tl.dot` is half
the throughput before the decode is counted), the pre-folded weight scale, and
warp specialization. Accuracy is not part of the trade — against an FP64
reference both lanes land on the MX quantization floor and agree to four
digits. At fixture width (T=64, H=128, F=64, E=2) the whole Triton chain is
byte-equal to the FP32 oracle, `h_q` codes and scales included, which is what
pins the recipe.

Forward only: `z` is not written, so there is no backward — use
`TritonDetSharedExpertProvider` for training.

## Reference backends

Per-operator P5 artifacts. Each overrides only the operators its kernel
delivers and leaves the rest on the FP32 oracle, so `scripts/check_p5.py` runs
end to end against any one of them.

| Module | Operator | Backends |
|---|---|---|
| `backends.mxfp8_act_quant` | P5-1 activation quantization | CUDA, Triton |
| `backends.grouped_gemm` | P5-4 base GEMM + P5-3 LoRA delta | CUDA |
| `backends.lora_delta` | P5-3 LoRA delta | torch-native, CUDA, Triton |
| `backends.clamp_swiglu` | P5-2 clamp-SwiGLU with route weight | CUDA |
| `backends.shared_expert` | P5-5 shared expert MLP | CUDA, Triton |

`backends.routed_checks` is not a backend: it holds the per-launch batch
check the two fused routed providers share, so the contract is read once
rather than once per backend.

Only `csrc/cuda/moe/sm90_fused_moe_mlp.cu` is compiled into `rl_engine._C` by
default; the other MoE sources are present but not wired, so their providers
raise until they are added to `setup.py` and `csrc/ops.cpp`.

Acceptance for a wired reference backend:

```bash
python scripts/check_p5.py --device cuda \
    --provider rl_engine.moe.backends.<module>:<Provider>
```

The MX quantizer additionally reproduces the committed golden manifest
(`tests/fixtures/p5/golden_hashes.json`, anchored on the CPU x86 oracle) and,
independently, torchao's OCP-MX cast in FLOOR mode and Triton's
`downcast_to_mxfp` in ROUND_DOWN mode — two outside implementations agreeing
byte for byte is a stronger signal than the local tests alone.
