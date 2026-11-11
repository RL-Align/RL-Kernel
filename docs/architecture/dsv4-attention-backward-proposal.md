# DSv4 attention backward for LoRA training on SM90: design proposal

Status: proposal awaiting the user's decisions (Section 5), 2026-11-11. It comes from source
reading only; no GPU runs. Items marked **[unverified]** are inferences from source reading or
arithmetic, not measurements.

Scope: backward of `rl_engine/models/deepseek_v4/attention.py::Dsv4Attention` (our BI forward)
producing dL/dhidden plus the inputs to the LoRA adapter gradients. Base weights, norms, `ape`
and the attention sink are frozen. Indexer selection is a constant. Requirements (design doc D5,
D7): run-to-run deterministic (not bitwise BI), within about 1.3x of an efficient
non-deterministic baseline on the same shapes, SM90.

Why the backward must attach to *our* forward and not to Megatron's `dsv4_hybrid` module: the
PPO / GRPO loss uses the log-probs of the training forward. If the training forward with grad
runs a different kernel set from rollout and recompute, the ratio at step 0 is not exactly 1,
which is the mismatch this project removes. So the training forward is `Dsv4Attention` (same
bytes as recompute), and the backward below differentiates it, with straight-through estimators
(STE) at the FP8 quantization points.

---

## 1. Sparse MLA attention backward kernels available for SM90

### 1a. FlashMLA: no sparse backward on any arch

- Our pinned build (`~/dsv4-work/flashmla-src`, `vllm-project/FlashMLA@a8f794d`): forward only.
  `flash_mla_sparse_fwd` returns `(out, max_logits, lse)`
  (`flash_mla/flash_mla_interface.py:182-219`). With a sink, `lse` is the **KV-only** LSE
  ("This argument has no effect on lse"); empty rows get `lse = +inf` (`tests/ref.py:40-52`).
- Upstream `deepseek-ai/FlashMLA`: the trees of branches `main`, `nv_dev` and
  `v41-open-source` (GitHub API, 2026-11-11; latest main commit `395cb3af`, 2026-10-27) hold
  backward code only under `sm100/prefill/dense/` (the dense MHA forward/backward from NVIDIA's
  PR #76). There is **no sparse backward for SM90 or SM100**.
- **Repo issue (fixed):** `rl_engine/kernels/ops/cuda/dsv4_attention.py` took `[:2]`, so the
  value our code called `lse` was in fact `max_logits`. Only byte comparisons used it. It now
  returns element `[2]`, and `tests/ops/attention_dsv4/test_sparse_core.py` checks it against
  an eager `logsumexp`.

### 1b. TileLang: forward only for DSv4; the V3.2 backward does not fit as is

- The installed `tilelang 0.1.12` (`~/.conda/envs/dsv4-t213/.../site-packages/tilelang`) ships
  **no examples**. They exist only in the GitHub repo (`tile-ai/tilelang@824c5c42`, main,
  2026-11-10):
  - `examples/deepseek_v4/`: `sparse_attn_fwd_sm90.py` (forward with sink, partial RoPE,
    "simply an illustration and maybe not optimal", TODO window, `TL_ENABLE_FAST_MATH`), plus
    `act_quant.py` and an SM100 GEMM. **No DSv4 backward.**
  - `examples/deepseek_v32/sparse_mla_bwd.py` and `examples/dsa_sparse_finetune/sparse_mla_bwd.py`:
    V3.2 shapes, `D = 512` value dims plus `D_tail = 64` (d_qk 576, d_v 512). Restrictions:
    - Causal mask `Indices <= s_i` (query and KV share one index space), with an assert that
      non-causal is unsupported.
    - No sink.
    - BF16 KV only; `topk % 32`.
    - **dKV uses `T.atomic_addx4` into an FP32 buffer, so it is nondeterministic.**
- Megatron-LM v0.19.2 vendors a copy
  (`~/dsv4-work/ref/megatron-lm/.../experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py`,
  adapted from tilelang `4ff81c7`). It replaces the causal mask with `idx != -1` and keeps the
  atomics. Its config passes disable TMA lowering and warp specialization; `block_H <= 32`.
- Fit for DSv4: DSv4 has d_qk = d_v = 512 with RoPE inside the 512, so `D_tail = 0`
  (zero-size shared buffers; **[unverified]** whether it compiles). A sink can be emulated by
  passing `lse' = logaddexp(lse_kv, sink)`; this is exactly what cuDNN does (below).
- **Megatron main rejects TileLang for DSv4:** `transformer_config.py:2153` raises
  "dsv4_hybrid does not support dsa_kernel_backend='tilelang'; use 'cudnn' ... or 'none'".

### 1c. What Megatron uses for DSv4, and the cuDNN-frontend DSA backward (the real candidate)

Megatron main (`~/dsv4-work/megatron-main@d2b0e35cc`; `ref/mcore-main` holds only 3 files):

- `experimental_attention_variant/csa_utils/fused_sparse_attention.py`: `CSASparseAttnFunc`
  runs forward with **FlashMLA `flash_mla_sparse_fwd`** and backward with **cuDNN frontend
  `cudnn.DSA.sparse_attention_backward_wrapper`** (CuTe-DSL). It saves `q, kv, sink, idx, out,
  lse`. The packed THD path is the same (`csa_utils/packed_sparse_attention.py`).
- `dsa_kernel_backend='none'` falls back to eager PyTorch (`csa.py:227`
  `unfused_compressed_sparse_attn`: `index_select` gather + einsum; its backward is
  `index_add`, so it uses atomics on CUDA).
- Determinism:
  - `csa.py:52` labels the CSA ops "unknown".
  - `csa_utils/cp_utils.py:321` raises `NotImplementedError` for `deterministic_mode` on the packed
    path ("tracked in #7921").
  - The compressor is plain eager autograd (FP32 softmax pooling, `csa.py:436-612`).
- verl's DSv4 Megatron recipe (`ref/verl/examples/sft/gsm8k/run_deepseek_v4_megatron_lite.sh`)
  uses these "fused DSA kernels on Hopper and Blackwell" (cudnn-frontend + cutlass-dsl). So
  **the current verl actor backward on H200 is FlashMLA forward + cuDNN SM90 backward, and it is
  not deterministic.**

The installed cuDNN-frontend DSA backward is `nvidia-cudnn-frontend 1.31.0`
(Apache-2.0 AND MIT), in both `dsv4-t213` and `dsv4-verl`, with `nvidia-cutlass-dsl 4.6.2`.
Path: `cudnn/deepseek_sparse_attention/sparse_attention_backward/`.

- `api.py`: dispatches to SM90 (`_interface_sm90.py`, `dsa_bwd_sm90.py`, 2394 lines) or to
  SM100.
- Contract:
  - Tensors: `q [T, H, D]`, `kv [S_kv, D]` (K = V, MQA), `out`, `dout [T, H, D_v]`,
    `lse [T, H]` FP32 (KV-only LSE, natural log), `attn_sink [H]` FP32, `topk_idxs [T, W]`
    int32 (global; entries outside `[0, S_kv)` are ignored), optional `topk_length [T]`.
  - `head_dim in {512, 576}`; `D = 512` gives `D_v = 512`, which is **exactly DSv4**.
  - `num_head > 1`; 64 heads is one 64-row head tile.
- Sink: the preprocess folds the sink into the LSE (`lse_log2 = log2(2^lse + 2^sink)`) and
  computes `d_sink = -sum p_sink * Delta`. This matches our oracle
  `rl_engine/p2/oracle.py::joint_attention_backward`.
- **Work decomposition (SM90):** one work tile = (one query token, 64-head tile). It loops over
  all of that query's top-k blocks of 64 in a fixed (reverse) order (`dsa_bwd_sm90.py:1716-1740`).
  It computes S, P, dP, dS, `dQ += dS K` in registers and stores dQ with TMA, so **dQ has a
  single writer and is deterministic**. The dKV tile `dS^T Q + P^T dO` (64 keys x 512) is
  scattered with **FP32 float4 `atomicAdd`** into a padded FP32 `dkv_accum`
  (`scatter_dkv_atomic`, line 1379). A postprocess kernel then casts it to BF16. `d_sink` uses
  one FP32 atomic per warp. **dKV and d_sink are nondeterministic.**
- `deterministic=True` is accepted **only on SM100** (`api.py`: "deterministic DSA backward
  requires SM100 and heads in (16, 32, 64, 96, 128)"). The SM100 scheme
  (`dsa_bwd_sm100_deterministic.py`) is a bounded wave: 128 FP32 dKV shards, CTA `i` is the
  only writer of shard `i` in a wave, waves are serialized by launch order, then a fixed fold.
  There is no SM90 port.
- `aligned_hca_backward` (fixed-geometry C128 backward) requires GB300 or Rubin
  (capability 10.3 / 10.7), so it does not apply.

### 1d. Other options

- FlashAttention-3: dense or varlen only, no per-query index lists. With the window and C128
  layers it could cover the "dense band" parts, but not C4 top-k.
- NSA kernels (`tilelang/examples/deepseek_nsa/*_bwd.py`, Triton NSA): block-sparse, with a
  different selection granularity and atomics. Not a fit.
- MoBA: block-sparse attention, same problem.
- Our `csrc/cuda/attention/deterministic_attention.cu`: deterministic dense attention
  (forward + backward, head_dim 128, scalar kernels, materialized P). It is useful as a pattern
  but not as a kernel for D = 512.

**Conclusion for Q1:** the only SM90 sparse-MLA backward that matches the DSv4 contract (D 512,
K = V, MQA 64 heads, sink, per-query index lists with `-1` / `topk_length`, FlashMLA's LSE
convention) is cuDNN-frontend's `dsa_bwd_sm90`. It is the production path in Megatron / verl
and is nondeterministic only in dKV (and d_sink, which we do not need).

---

## 2. Determinism of dKV

dKV is a scatter-add. Each workspace row receives contributions from every (query, slot) that
selected it. A window row gets up to 128 consecutive queries; a C128 entry gets every later
query; a C4 entry gets every query whose top-512 contains it. dQ is row-local and already
deterministic in cuDNN's kernel. d_sink is not needed (sink frozen).

Scale for one C4 request of 4096 tokens [arithmetic]:
- About 510 valid keys per query, so 2.09 M (query, key) pairs.
- Backward is about 0.68 TFLOP; forward is about 0.27.
- dKV atomic traffic is 4.3 GB of FP32 reductions into an accumulator of only 10.5 MB (5120
  rows x 2 KB), which stays **L2-resident** (H200 L2 50 MB). That L2 locality is what makes
  cuDNN's atomics affordable. Any deterministic scheme that breaks it moves gigabytes to HBM.

Candidate schemes, in recommended prototyping order:

**S1. Fixed-point integer accumulation (order-independent atomics).**
- Change only `scatter_dkv_atomic` in a fork of `dsa_bwd_sm90.py`:
  - Convert each FP32 contribution to `int64` as `round(x * 2^s)`. This is deterministic per
    contribution because scaling by `2^s` is exact.
  - Accumulate with `red.global.add.u64`. Integer addition is associative, so the result does
    not depend on order.
  - The postprocess converts back with `* 2^-s`.
- Choosing the scale: a per-layer bound
  `|dKV| <= sum_q,h (|dS||Q| + P|dO|) <= T * H * (2 max|dP| max|Q| + max|dO|)` (a cheap amax
  pre-pass), or a fixed constant. A bound that is too loose by 2^20 still leaves about 40 bits
  of resolution below the true maximum, which is more than FP32's 24.
- With a fixed per-layer scale the sum is even batch-invariant.
- Costs [unverified]: 2x atomic bytes (8.6 GB) and no vector form of u64 `red` (4x the
  instructions of `red.v4.f32`). The accumulator is 21 MB and still fits L2 at 4K tokens. At
  32K+ tokens neither the FP32 nor the int64 accumulator fits.
- This is the smallest change and the best first prototype. Measure it against stock cuDNN.

**S2. Ordered ("ticket") accumulation into one shared FP32 accumulator.**
- Build the inverse index (for each KV row, its referencing (query, slot) pairs in ascending
  query order) with a counting sort, as in `csrc/cuda/moe/moe_route_plan.cu` (`ace607e9`).
  Ranks are analytic for window and C128 rows; only the C4 top-k part needs the sort.
- Each CTA waits until `sem[row] == rank` before adding, then fences and releases. A persistent
  grid with ascending query order per CTA makes this deadlock-free.
- Risk [arithmetic]: consecutive queries share almost all window rows, so the adds serialize
  into a chain. A wave of 132 adjacent queries waits row by row; a rough estimate is several ms
  per 4K request. Use it only with an interleaved query schedule, and expect trouble at the
  1.3x gate.

**S3. Sharded single-writer accumulation and a fixed fold (the SM100 cuDNN scheme).**
- Each shard keeps the full workspace rows (5,120 at 4K tokens) as one private FP32 accumulator.
  Each shard is 10.5 MB; 128 shards are 1.34 GB, which is not L2-resident.
- Each (query, key) pair becomes a 4 KB HBM read-modify-write, about 8.6 GB, roughly 2 ms
  [estimate]. That fails the gate for C4.
- Query-range shards can cut the window part to about two shards per row, but the compressed
  rows are still touched by every shard.

**Rejected:**
- Materializing per-(query, slot) dKV partials and then a segmented reduction: 5.4 GB per 4K
  request in FP32.
- KV-stationary dense backward for the C4 top-k part: density is about 50% at 4K and about 3%
  at 64K, so most of the work is wasted.
- KV-stationary dense backward does fit the **window band** and **C128** parts (FA2-style,
  dKV in-CTA, no atomics), but it needs a separate dQ pass: about 1.4x FLOPs. Keep it as a
  fallback for ratio-1 and C128 layers if S1 fails.

Golden for all schemes: an FP32 PyTorch backward with a fixed-order segmented sum (sort the
(row, query) pairs, then `fixed_sum`), extended from `rl_engine/p2/oracle.py::joint_attention_backward`.

---

## 3. The rest of the backward chain

Forward reference: `Dsv4Attention.forward_batch`. In the table, "det" means a single writer or a
fixed-order reduction.

| Step (backward order) | Operation | Kind | Existing piece | det |
|---|---|---|---|---|
| `wo_b` dX | FP8 DeepGEMM dgrad | big GEMM | `Fp8Linear.backward_dx` | yes (no split-K, BI probe) |
| `wo_a` grouped dX | 8 groups, `[T,1024] x [1024,4096]` | big GEMM | new: per-group `fp8_gemm_nt` with transposed group weights (`Fp8BlockWeight.t()`) | yes |
| FP8 quant of `o` (`inv_rope_fp8`) | STE (identity) | none | — | — |
| inverse RoPE on o[448:512] | apply forward rotation (transpose) | elementwise | `csrc/cuda/rope/rope_sm90.cu` has `sin_sign` for backward; or a small kernel | yes |
| **sparse attention** | dQ, dKV(workspace) from `q_pad, ws, idx, lens, out, lse_kv, dO` | **big kernel** | cuDNN `flash_attn_bwd_sm90` (call it directly with `need_d_sink=False`) | dQ yes, **dKV no**: see Section 2 |
| workspace → sources | slab rows: window rows → per-token KV, compressed rows → entries | gather (each row has one source in a full-sequence prefill slab) | index math from `build_workspace` / `build_indices` | yes |
| FP8 `fp8_ds_mla` quant of KV (window and compressed) | STE: gradient w.r.t. the dequantized BF16 value | none | — | — |
| q: weightless RMSNorm per (token, head) + RoPE | RoPE^T, then `dx = r (dy - x̂ mean(dy x̂))` over 512 | elementwise + row reduction | new small kernel (mirror of `dsv4_qo.cu`) | yes |
| `wq_b` dX | FP8 dgrad, `[T,32768] -> [T,1024]` (0.27 TFLOP at 4K) | big GEMM | `Fp8Linear.backward_dx` | yes |
| `qr` RMSNorm (q_norm, frozen) | dX only | row reduction | `rmsnorm_backward_dx_cuda` (`csrc/cuda/norm/rmsnorm.cu:279`) | yes |
| kv: RoPE + RMSNorm (kv_norm, frozen) | RoPE^T, RMSNorm dX | elementwise / row reduction | rope + `rmsnorm_backward_dx_cuda` | yes |
| `fused_wqa_wkv` dX | one FP8 dgrad on `[d qr_raw \| d kv_raw]` `[T,1536] -> [T,4096]` | GEMM | `Fp8Linear.backward_dx` | yes |
| **compressor** (main KV, C4 / C128) | g = dKV rows of the compressed entries → RoPE^T → RMSNorm dX (comp_norm frozen) → softmax-pooling backward: `dkv_j = p_j g`, `dscore_j = p_j g (kv_j - pooled)` per channel (APE frozen: its grad is not needed) | elementwise / gather | new kernel next to `csrc/cuda/compressor/compress_fp8.cu`; recompute `p_j` from the saved FP32 `[kv \| score+ape]` (or recompute the score GEMM) | yes: in C4 overlap each (token, half, channel) feeds exactly one entry (half 1 → entry `t//4`, half 0 → `t//4+1`), so the backward needs no cross-entry reduction |
| compressor `wkv_wgate` dX | `dscore_kv [T, 2*coff*512] @ W [.,4096]` | GEMM | DeepGEMM `bf16_gemm_nn` (already used on SM90 in `logprob_dsv4`) or cuBLAS FP32; decide FP32→BF16 rounding of dY | yes (no split-K) |
| indexer (Q path, `weights_proj`, indexer compressor, top-k) | constant | none | — | — |
| d hidden | `fused_wqa_wkv dX + compressor dX` (+ LoRA dX), fixed order | add | — | yes |
| sink, `ape`, norm weights | frozen | none | — | — |
| LoRA adapters | `dB = s dyᵀ(Ax)`, `dA = s (Bᵀdy)ᵀ x`, `dx += s Aᵀ(Bᵀdy)` | small GEMMs | `Fp8Linear.backward_dw` pattern (DeepGEMM wgrad, padded `T % 16`); BF16 rank-r: cuBLAS is run-to-run deterministic on a fixed GPU / stream config, or DeepGEMM `bf16_gemm_tn` / `k_grouped_bf16_gemm_tn_contiguous` (**[unverified]** on SM90) | yes |

Summary: only the **sparse attention backward** is a big new kernel. The **compressor backward**
and the q / kv / o **norm + RoPE backwards** are small memory-bound kernels (each a few hundred
lines of CUDA plus golden plus tests, in the existing fixed-order style). The projections reuse
`Fp8Linear.backward_dx`; the grouped `wo_a` needs a thin wrapper.

Notes:
- The cuDNN backward needs `out` as FlashMLA produced it (before the inverse RoPE in
  `OutputProjection`). That is what our forward has; Megatron has to undo its in-place output
  RoPE (`undo_output_rope_for_backward`).
- cuDNN's postprocess writes BF16 dKV from a padded "fake-col" FP32 accumulator. For S1 (and to
  keep FP32 into the compressor / kv backward) we write our own postprocess.
- The backward does not need BI (D5). It may use a compact KV workspace (only existing rows,
  remapped indices) instead of the forward's fixed `max_model_len // ratio` slab. The probe
  showed FlashMLA's output bytes do not depend on the slab offset, so `out` and `lse` stay valid.
- The `build_compact` decode layout duplicates rows per token. Use the training backward only
  with the full-sequence prefill (slab) layout.

---

## 4. Phased plan

Estimates are for one engineer familiar with the repo [rough].

| Phase | Work | Estimate | Exit criterion |
|---|---|---|---|
| A0 | FP32 golden of the whole layer backward: an FP32 eager forward with explicit STE points and autograd, plus `fixed_sum` segmented dKV | 3-4 d | golden matches finite differences on tiny shapes |
| A1 | cuDNN SM90 DSA backward wrapper (`rl_engine/kernels/ops/cuda/`, fail-closed version pin like `flashmla.py`); probe: values vs golden at 64x512, W 640/1152, sink, `-1` / `topk_length`; **measure run-to-run dKV variance and time** (this is the baseline for D7) | 2-3 d | numbers in a `docs/validation/` json |
| A2 | Small kernels with goldens: q-norm/RoPE backward, kv RoPE + RMSNorm backward, output inverse-RoPE backward, workspace→token / entry gather, compressor backward (C4 overlap, C128) | 1-1.5 w | byte-equal golden/CUDA, deterministic |
| A3 | Projections: grouped `wo_a` dgrad, compressor BF16 dgrad, LoRA module (forward delta + dA / dB / dX, deterministic) | 1 w | LoRA grads vs FP32 reference |
| A4 | `torch.autograd.Function` for the layer (save / recompute policy, packed varlen); end-to-end layer gradient vs golden; benchmark vs Megatron cuDNN path | 1 w | full layer backward works (dKV still nondeterministic) |
| B1 | Fork `dsa_bwd_sm90.py` (Apache-2.0) with S1 int64 fixed-point dKV, own postprocess, no `d_sink` | 1-2 w | bitwise run-to-run; <= 1.3x of A1 |
| B2 | If B1 misses the gate: S2 ordered tickets, or KV-stationary kernels for window / C128 plus S1 for C4 top-k only | 2-4 w | same |
| C | verl / Megatron integration: training forward = `Dsv4Attention` with grad; LoRA wiring | 2+ w (outside this proposal) | — |

## 5. Decisions needed from the user

1. **Backward kernel source:**
   - (a) Fork cuDNN-frontend `dsa_bwd_sm90` (CuTe-DSL Python; adds a runtime dependency on
     `cudnn-frontend` + `cutlass-dsl`, which verl's DSv4 recipe already needs). **Recommended.**
   - (b) Adapt TileLang V3.2 backward (D_tail 0, sink via LSE, slower config, atomics too;
     Megatron rejects it for DSv4).
   - (c) Own CUDA / CuTe C++ kernel (4-8 w).
   - Note: "CUDA first" so far meant no Triton. CuTe-DSL is what vLLM's CUDA build already
     runs (indexer Q kernel parity), so it seems acceptable. Please confirm.
2. **dKV determinism scheme:** S1 int64 fixed point (recommended first) vs S2 ordered vs S3
   sharded. For S1, choose a fixed per-layer scale (gives batch invariance too) or a per-call
   amax bound.
3. **Recompute vs store:**
   - Never store P (335 MB per layer at 4K in BF16).
   - Saving `q_pad` + `out` costs 0.54 GB per layer at 4K tokens. Options: save `out`, `lse`,
     `hidden` and recompute the q path, KV workspace and compressor scores in backward; or
     rely on Megatron-style full-layer recompute.
   - Recommendation: per-layer recompute, keeping `out` and `lse` from the recompute.
4. **KV precision in training:** keep the forward's FP8 `fp8_ds_mla` round trip (needed for
   recompute = rollout bytes) and use STE. BF16 KV in the training forward would break D6
   parity. Please confirm STE at every FP8 point (KV cache, `o` before `wo_a`, activations of
   every `Fp8Linear`).
5. **dgrad precision of the frozen FP8 projections:** DeepGEMM FP8 dgrad (quantizes dY per
   1x128, as in the DeepSeek-V3 recipe; fastest) vs BF16 dgrad on dequantized W (more precise,
   about 2x GEMM time).
6. **LoRA placement:**
   - Which of `wq_a` / `wkv` (fused), `wq_b`, `wo_a` (grouped), `wo_b`, compressor
     `wkv_wgate`. Indexer projections get no gradient by construction.
   - Rollout must run the same unmerged LoRA forward (or merge and re-quantize identically on
     both sides) to keep recompute = rollout.
7. **Index-width / context range to optimize for:** this decides whether the dKV accumulator
   stays L2-resident (4K: 10-21 MB; 32K+: no) and therefore how much S1 costs.

## References

- Design doc: `docs/architecture/dsv4-cuda-design.md` (D5, D7, Phase 2 T05 / T07).
- Ours:
  - `rl_engine/models/deepseek_v4/attention.py`
  - `rl_engine/kernels/ops/cuda/dsv4_attention.py`
  - `rl_engine/kernels/ops/cuda/fp8_linear.py`
  - `rl_engine/kernels/ops/pytorch/compressor/compressor.py`
  - `rl_engine/p2/oracle.py:278`
  - `csrc/cuda/norm/rmsnorm.cu`
  - `csrc/cuda/rope/rope_sm90.cu`
  - `csrc/cuda/moe/moe_route_plan.cu`
- FlashMLA: `~/dsv4-work/flashmla-src` (`flash_mla/flash_mla_interface.py`, `tests/ref.py`);
  https://github.com/deepseek-ai/FlashMLA (branches main / nv_dev / v41-open-source, trees
  checked via the GitHub API).
- cuDNN frontend 1.31.0:
  `~/.conda/envs/dsv4-t213/lib/python3.12/site-packages/cudnn/deepseek_sparse_attention/sparse_attention_backward/{api.py,_interface_sm90.py,dsa_bwd_sm90.py,dsa_bwd_sm100_deterministic.py}`;
  docs https://docs.nvidia.com/deeplearning/cudnn/v1.27.0/fe-oss-apis/dsa.html.
- Megatron main: `~/dsv4-work/megatron-main/megatron/core/transformer/experimental_attention_variant/{csa.py,csa_utils/fused_sparse_attention.py,csa_utils/cp_utils.py}`,
  `megatron/core/transformer/transformer_config.py:2153`.
- Megatron-LM 0.19.2 TileLang copy: `~/dsv4-work/ref/megatron-lm/megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py`.
- TileLang: https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v4,
  `.../examples/deepseek_v32/sparse_mla_bwd.py`, `.../examples/dsa_sparse_finetune/sparse_mla_bwd.py`.
- verl: `~/dsv4-work/ref/verl/examples/sft/gsm8k/run_deepseek_v4_megatron_lite.sh`.
