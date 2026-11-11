# DSv4 SM90 Batch-Invariant Execution Profile

The frozen execution profile for DeepSeek-V4 on Hopper (SM90): what runs where, which
reduction order every module uses, and what a change must keep. Plan and history live in
[`dsv4-cuda-design.md`](dsv4-cuda-design.md); arithmetic details live on the operator
pages under `docs/operators/`. This page collects the contracts in one place and is the
checklist for anyone touching the kernel set.

- Scope: CUDA SM90 (H100 / H200), torch 2.13 / CUDA 13, DeepGEMM `vllm-project/DeepGEMM@e21c821f`,
  FlashMLA `b70aff3` (SM90 sources identical to vLLM's pin `a8f794d1`).
- Reference architecture: vLLM #54955 (`ISEEKYAN/vllm@1a6542c`), batch-invariant mode.
- Last updated: 2026-11-11.

## 1. Guarantees

| Path | Guarantee |
|---|---|
| Forward (rollout, actor recompute, training forward) | **Bitwise batch-invariant.** A token's bytes do not depend on the other tokens in the step, on the batch order, or on whether its request runs as one prefill, as chunked prefill or as decode steps. `forward_train` is byte-equal to `forward_batch`. |
| Selected-token logprob | Bitwise batch-invariant and vocab-TP invariant (TP 1/2/4/8 on tile boundaries). |
| Backward (LoRA training, dX path) | Correct against FP32 autograd per component. **Not** required to be deterministic (user decision 2026-11-11); cuDNN's sparse-attention dKV uses FP32 atomics. Most other backward kernels are deterministic anyway. |
| Weight gradients (LoRA `dA` / `dB`) | Deterministic for fixed shapes, not batch-invariant (a reduction over tokens). |

"Batch-invariant" is always checked bytewise: `fwd(x)[t] == fwd(x[t:t+1])`, prefixes,
windows, permuted batches, and run to run. A numeric tolerance never stands in for it.

## 2. Decisions this profile rests on (summary of D1-D7)

| | Decision | Consequence for the profile |
|---|---|---|
| D1 | DP-attention + EP-MoE, TP = 1 inside attention | No cross-rank reduction on the attention path. The only cross-rank steps: EP dispatch / combine (fixed order), the vocab-TP logprob merge, the mHC broadcast |
| D2 | Routed experts on our own P5 SM90 kernel (DeepGEMM runs FP4 only on SM100) | Routed experts have their own profile `p5-sm90-fused-mlp-v1` |
| D3 | Dense FP8 GEMMs on unmodified pinned DeepGEMM | `fp8_gemm_nt` / `fp8_einsum` are BI as shipped; re-probe on any pin, toolkit or SM-count change |
| D4 | No bare `torch.mm` / cuBLAS on the forward path | Router gate, compressor scores, logits all use DeepGEMM `bf16_gemm_nt` with FP32 output |
| D5 | Backward is relaxed | dX correctness only; see section 1 |
| D6 | The formal anchor is verl | The rollout must be patched to run this kernel set (section 7) |
| D7 | Each operator within 1.3x of its non-invariant path | Budget in section 6 |

## 3. Pinning rules

Adopted from vLLM #54955. Every rule is enforced in code and covered by a test.

1. **No batch-local plan in attention.** Each request owns a workspace slab of fixed size
   (`N + G` rows, `N = max_model_len // ratio`); prefill and decode call the same
   `flash_mla_sparse_fwd` with `num_splits` / workspace fixed from the config.
2. **One selector, canonical order.** Indexer top-k uses the composite key
   `(ordered_bits(score), local_index)`, ties to the larger index, the same kernel for
   prefill and decode. The ids are sorted descending before FlashMLA (its accumulation
   follows index order; a permuted index set changes the bytes).
3. **Capacities come from the config.** Top-k width, gather capacity, workspaces and MoE
   capacities are fixed, never derived from the batch (`expert_num_tokens.max()` and the
   like are banned).
4. **Fixed launch geometry for quantization** with 64-bit offsets (tested at >= 2^31
   elements).
5. **mHC**: explicit pre / post, a fixed 16-way split in the controller GEMM, stable token
   order.
6. **Hopper keeps FP32 block scales** (UE8M0-valued); SM90 and SM100 are not treated as
   numerically interchangeable.
7. **A missing kernel is a hard error.** `require_deepgemm`, the FlashMLA version check and
   the `_C` symbol checks fail closed; there is no silent fallback to a non-BI path.

Additional rules learned on this branch:

8. **No split-K, no atomics on the forward path.** Every GEMM reduces its full K inside one
   CTA in a fixed order; scatter-adds are replaced by fixed-order slot sums.
9. **Correctly rounded FP32 in our kernels.** `__fadd_rn` / `__fmul_rn` / `div.rn`, no FMA
   contraction unless the contract names an `fmaf` chain; no `.ftz`; `--use_fast_math` is
   refused (the MX quantizer is fast-math-proof by construction).
10. **Transcendentals are libdevice** (`expf`, `logf`, `log1pf`), so goldens are defined on
    CUDA, where `torch.exp` is the same function.
11. **Goldens divide by tensors, not Python scalars** (CUDA PyTorch multiplies by the
    reciprocal for a scalar divisor).
12. **Host syncs are explicit.** The decode step reads nothing back: sizes travel as host
    integers in `Dsv4Batch`, and the MoE finite check can be deferred into a device flag
    (`check_finite="defer"`, read by `Dsv4Model.raise_if_nonfinite()`), so a whole decode
    step is CUDA-graph capturable at EP = 1.

## 4. Reduction-order contract per module

The table names the order that fixes each module's bytes. "Row-local" means a row's output
reads only that row (and frozen weights), which makes the module batch-invariant by
construction. Full formulas are on the linked pages.

### 4.1 Model skeleton

| Module | Kernel | Reduction order | Page |
|---|---|---|---|
| Embedding | PyTorch indexing | Exact copy into 4 identical streams (vLLM pre-sums `hc_attn_fn` over the streams instead; equal only in exact arithmetic) | [dsv4-model](../operators/dsv4-model.md) |
| `mhc_pre` / `hc_head` controller | DeepGEMM `tf32_hc_prenorm_gemm`, 16 splits | Partials reduced by halving (`p[0:8] + p[8:16]`, then 4, 2, 1); `hc_head`'s padded `fn` cached per tensor | [dsv4-mhc](../operators/dsv4-mhc.md) |
| mHC per-token math | `dsv4-mhc-v1` CUDA | Correctly rounded steps; "sum of four" = `(a0 + a1) + (a2 + a3)`; Sinkhorn 19 x (row, column); H-long dots use a fixed 128-slice partition (runs of 8, `fmaf` chain, halving within 32, then sum of four) | [dsv4-mhc](../operators/dsv4-mhc.md) |
| Fused RMSNorm | same kernel | `slice(y, y)` as above; `rsqrt(mean + eps)` spelled `1 / sqrt(...)` | [dsv4-mhc](../operators/dsv4-mhc.md) |
| RoPE tables | PyTorch, built once | YaRN on compressed layers (bit-equal to vLLM), plain RoPE on window-only layers (bit-equal to DeepSeek's `precompute_freqs_cis`) | [dsv4-model](../operators/dsv4-model.md) |

### 4.2 Attention

| Module | Kernel | Reduction order | Page |
|---|---|---|---|
| q / kv / o projections (`wq_a`, `wkv`, `wq_b`, `wo_b`) | `Fp8Linear`: `fp8_group_quant` + DeepGEMM `fp8_gemm_nt` | Activation: per-128 group absmax (row-local), UE8M0 scale from bits. GEMM: full K in one CTA, one FP32 promotion per 128-K block, tile from N / K only | [fp8-group-quant](../operators/fp8-group-quant.md) |
| q norm + RoPE, KV RoPE + `fp8_ds_mla` insert | `csrc/cuda/attention/dsv4_qo.cu` | Row-local; fixed-order sum of squares (4 channels per thread, 32-lane butterfly, warps ascending) | design doc T07 |
| KV compressor (C4 / C128, indexer) | `csrc/cuda/compressor/compress_fp8.cu` | Per channel over the window: max, `expf`, ascending-j sum, `kv * e` sum without FMA; RMSNorm as above; reads only its own request's state through the block table | [compressor](../operators/compressor.md) |
| Compressor score GEMMs, `weights_proj` | DeepGEMM `bf16_gemm_nt`, FP32 out | As `fp8_gemm_nt` (no split-K) | design doc T03 |
| Indexer Q | `indexer_q_rope_fp8` | Row-local, byte-equal to vLLM's CuTe-DSL kernel (FMA where it uses FMA) | [indexer-q](../operators/indexer-q.md) |
| Indexer logits | DeepGEMM `fp8_fp4_mqa_logits` (prefill), `fp8_fp4_paged_mqa_logits` (decode) | One WGMMA over K = 128 per logit plus a fixed-order head sum; no split across KV positions; paged decode logits equal prefill logits | design doc T03 |
| Indexer top-k | `stable_topk_per_row` | Exact (integer keys); ties to the larger request-local index | [stable-topk](../operators/stable-topk.md) |
| Index canonicalization | PyTorch | Top-k ids sorted descending, then the window ascending, `-1` padded to a multiple of 128 | design doc T05 |
| Sparse attention core | FlashMLA `flash_mla_sparse_fwd` | `num_splits` / workspace from the config, per-request slabs, accumulation in canonical index order | design doc T05 |
| Output projection | inverse RoPE + FP8 quant, then `wo_a` as DeepGEMM `fp8_einsum("bhr,hdr->bhd")` below 8192 tokens, per-group `fp8_gemm_nt` from 8192 up; then `wo_b` | Both `wo_a` paths give the same bytes at every T = 1..16384 (`scripts/probe_dsv4_wo_a_einsum.py`), so the switch point is not a batch dependence | [dsv4-performance](../operators/dsv4-performance.md) |

### 4.3 MoE

| Module | Kernel | Reduction order | Page |
|---|---|---|---|
| Router gate | DeepGEMM `bf16_gemm_nt`, FP32 out | No split-K | [dsv4-router](../operators/dsv4-router.md) |
| Router top-6 / hash | own CUDA | `sqrt(softplus)` with libdevice `expf` / `log1pf`; top-6 ties to the **smallest** expert id (unlike the indexer); weight sum left to right over the 6 slots | [dsv4-router](../operators/dsv4-router.md) |
| Route plan | `moe_route_plan.cu` counting sort | Integer, stable: `(expert, global token, slot)` order; equal to the golden sort | [dsv4-moe](../operators/dsv4-moe.md) |
| MXFP8 activation quant | own CUDA | Per-32 block, integer max over `|x|` bits, scale from the exponent field; once per token, gathered to its rows | [dsv4-moe](../operators/dsv4-moe.md) |
| Routed experts | P5 `sm90_fused_moe_mlp.cu` two-launch (`p5-sm90-fused-mlp-v1`) | Fixed K walk per CTA, no split-K; the 32 products of an MX block summed by the tensor core (deterministic, not oracle-equal); SwiGLU clamp 10, `p_s` applied once | [dsv4-moe](../operators/dsv4-moe.md) |
| Shared expert | DeepGEMM `fp8_gemm_nt` + own clamped SwiGLU + FP8 quant (`dsv4-deepgemm-fp8-v1`) | As the dense projections | [dsv4-moe](../operators/dsv4-moe.md) |
| Combine | `moe_slot_sum.cu` | Per token in FP32: slot 0, then slots 1..5 in order, then the shared expert, then the residual; one BF16 round | [dsv4-moe](../operators/dsv4-moe.md) |

### 4.4 Head

| Module | Kernel | Reduction order | Page |
|---|---|---|---|
| Logits | DeepGEMM `bf16_gemm_nt`, FP32 out (never rounded to BF16) | No split-K; vocab shards give the same columns | [dsv4-logprob](../operators/dsv4-logprob.md) |
| Log-sum-exp | `dsv4-logprob-v1` CUDA | 64 fixed vocab tiles; per tile 32 lanes over strided columns, then halving; tile merge `M = max`, rescale, `(e_i + e_{i+32})` then halving; `lse = M + logf(S)` | [dsv4-logprob](../operators/dsv4-logprob.md) |
| Chunking | `chunk_rows` (default 4096) | Each row is computed alone, so chunking changes no byte | [dsv4-logprob](../operators/dsv4-logprob.md) |

### 4.5 LoRA (`dsv4-lora-v1`)

Unmerged adapters on attention `wq_a`, `wkv`, `wq_b`, `wo_b`, and on the MoE (one routed
`fc1` / `fc2` pair shared by all experts, one shared-expert pair):

```text
h = BF16(x @ A^T)                     DeepGEMM bf16_gemm_nt, FP32 accumulate, one round
d = h @ B^T                           DeepGEMM bf16_gemm_nt, FP32
y = BF16(FP32(y_base) + scale * d)    two correctly rounded FP32 steps, one BF16 round
```

The rank is zero-padded to a multiple of 64 (exact). The routed `fc1` delta depends only on
the token, so it is computed once per token and added to the FP32 `z` inside
`moe_swiglu_quant` (byte-equal to the fused epilogue). Zero adapters give the no-adapter
bytes. Under EP the token's rank computes `u1 = BF16(x A1^T)` and sends it with the MXFP8
row, which gives the EP = 1 bytes.

## 5. Numeric profiles in use

| Profile | Module | Byte-equal to |
|---|---|---|
| `dsv4-mhc-v1` | mHC pre / post / `hc_head` + fused norms | its golden (not vLLM's TileLang kernels; differences on the mHC page) |
| `dsv4-logprob-v1` | logits + LSE + logprob | its golden; TP 1/2/4/8 merge |
| `dsv4-deepgemm-fp8-v1` | shared expert | itself across batches (DeepGEMM FP8; the rollout runs the same GEMMs) |
| `p5-sm90-fused-mlp-v1` | routed experts (forward) | itself across batches (deterministic, BI) |
| `p5-sm90-bwd-bf16-v1` | routed experts (dX) | FP32 autograd within tolerance |
| `dsv4-lora-v1` | all adapters | its golden; no adapters = base bytes |
| (no profile name) | `fp8_group_quant`, `indexer_q_rope_fp8`, `stable_topk_per_row`, compressor | **vLLM's kernels**, byte for byte |

A backend that cannot hold a profile declares a new one rather than relaxing an existing
one's tolerance.

## 6. Performance budget

Every operator is within the D7 gate of 1.3x (full table:
[dsv4-performance](../operators/dsv4-performance.md), H200). The closest to the limit:

| Operator | Ratio | Cost driver |
|---|---|---|
| Router (gate + top-6) | 1.03-1.22x | DeepGEMM gate against cuBLAS |
| `mhc_pre` + fused norm | 1.09-1.18x | TF32 GEMM with a fixed 16 splits |
| Indexer Q | 1.00-1.14x | vLLM's kernel is the baseline |
| Attention q side | 0.80-1.14x | vLLM's fused kernel is the baseline |
| Output projection | 0.77-1.06x | was 1.45-1.53x before `fp8_einsum` |

Not measured yet: the whole attention layer and the decoder layer against vLLM's
modules on the same shapes (needs vLLM's forward context; comes with the layerwise gate).

## 7. What the rollout must change

The vLLM fork is not byte-equal to this profile in these places; the verl rollout patch
must run our kernels (or our tables) there:

- **RoPE on window-only layers**: vLLM uses YaRN (factor 16) everywhere; this profile uses
  plain RoPE there. Pass our RoPE table to every layer.
- **Router**: vLLM's Triton / CUDA softplus uses approximate math and a cuBLAS gate; run our
  router.
- **Logits**: vLLM's default head produces BF16 logits; this profile keeps FP32.
- **mHC**: our `dsv4-mhc-v1` (fixed slices, explicit pre / post) instead of the TileLang
  kernels; the first layer's four identical streams instead of the pre-summed broadcast.
- **Routed experts**: confirm which kernel vLLM runs on Hopper for FP4 experts; if it is not
  P5, vLLM must call P5.

Matching already (no patch): `fp8_group_quant`, indexer Q, stable top-k, the compressor,
FlashMLA sparse forward, DeepGEMM projections and `fp8_einsum`.

## 8. Re-validation triggers

Re-run the probes (`scripts/probe_dsv4_*`) and the full DSv4 regression when any of these
changes: the DeepGEMM or FlashMLA pin, the CUDA toolkit major version, the GPU model or SM
count (the probes ran at 132 SMs), the torch version (libdevice / `torch.exp`), or a
prefill batch above the probed M (4096 for dense GEMMs; 16384 for `wo_a`).

Regression command (H200, `KERNEL_ALIGN_MOE_SM90=1` build):

```bash
export DSV4_META=$HOME/dsv4-work/dsv4-meta DSV4_REF=$HOME/dsv4-work/ref/dsv4-inference
python -m pytest -q -p no:cacheprovider tests/ops/logprob_dsv4 tests/ops/mhc_dsv4 tests/models \
    tests/ops/router tests/ops/moe tests/ops/compressor tests/ops/indexer tests/ops/topk \
    tests/ops/attention_dsv4 tests/ops/quant tests/ops/activation tests/build
```

## 9. Open items

- Layerwise exact-zero hidden-state gate and the #230 cross-config M0-M10 logprob gate
  (Phase 4).
- EP > 1 on real hardware (DeepEP dispatch / combine, fixed order); multi-GPU end-to-end
  runs move to another machine.
- DCP attention needs one comparator across local and global selection (out of scope for v1).
