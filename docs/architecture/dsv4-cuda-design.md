# DSv4 SM90 Batch-Invariant Plan and Todo

- Control PR: RL-Align/RL-Kernel #635 (`test-dsv4-cuda`)
- Reference architecture: vLLM #54955 (`ISEEKYAN/vllm:ds4-v9-rc1 @ 1a6542c`)
- Integration target: verl (rollout and actor training must call the same kernel set)
- Scope: CUDA SM90 only. Forward (selected-token logprob) must be bitwise; backward only
  needs to be run-to-run deterministic. CUDA kernels come first; Triton paths are deferred.
- Last updated: 2026-11-09

## 0. Strategy in one sentence

Do not write every kernel from scratch. Freeze an execution profile from the pinning rules
that vLLM #54955 has already validated end to end, borrow kernel bodies wherever possible
(FlashMLA sparse, DeepGEMM, the DeepEP fork, our own P1/P5/det_gemm), and have RL-Kernel
supply only what decides the reduction order: contracts, stable top-k, per-request
chunking, and test oracles. Get every component batch-invariant (BI) and aligned end to end
first, then come back for any operator that misses the performance gate.

## 1. Current state

| Package | Contents | Status | Disposition in this plan |
|---|---|---|---|
| P1 | mHC + RMSNorm + fp32_gemm_rms | Done (unfused canonical) | Keep as is; DSv4 uses its own fused `dsv4-mhc-v1` (Phase 3) |
| P5 | MXFP8 act quant, MXFP4 routed-expert SM90 fused MLP (`sm90_fused_moe_mlp.cu`), LoRA delta, shared expert, clamp SwiGLU | Done | Routed-expert kernel: keep (optimized and BI). Shared expert: move from the det_gemm K-tree to DeepGEMM FP8 (D2, Phase 3) |
| P3 | Router: gate GEMM, stable_topk6, routing weights, RoutePlan | Early T01 + T05 were merged but carry almost no usable progress; no `rl_engine/p3` on this branch; T08 (#639) draft | **Redo** on this branch: gate / stable top-k / weights / combine-plan |
| P2 | Compressed attention: KV compressor C4/C128, FP4 index, paged cache | T01 start kit only | **Critical path**, T02-T07 all open |
| P6 | Fixed-order MoE combine | T01 + T05 fwd (#462) | T06 + multi-rank |
| P4 | EP transport | Empty | DeepEP fork adapter |
| P7 | Sampling gate + M10 logprob | T02 (#648) was merged but carries almost no usable progress | **Redo** on this branch; wire to the #241 vocab-TP deterministic reduce |

Development happens on `test-dsv4-cuda` directly. The original P-dev branches are far behind
and are not rebased or tracked any more.

## 2. Design decisions (settled)

### D1. Parallel layout: DP-attention + EP-MoE, TP=1 inside attention

Each rank owns whole requests, so there is no cross-rank reduction anywhere on the attention
path and batch invariance follows directly from per-request chunking. The only cross-rank
operations left are EP dispatch/combine (fixed order), the LM-head vocab-TP reduce (#241
already provides a deterministic one), and the mHC broadcast.

**DCP/CP attention is out of scope for v1.** The review on #54955 already shows that the
local selector (prefers larger local indices) and the DCP merge (prefers smaller global ids)
use inconsistent tie-breaks and can keep different KV sets. Supporting DCP requires a single
comparator carried through local and global selection; track it as a separate issue.

### D2. Expert path stays on our own P5

- Routed experts: `csrc/cuda/moe/sm90_fused_moe_mlp.cu` (MXFP8 x MXFP4 fused MLP, FP8
  WGMMA + TMA, built with `KERNEL_ALIGN_MOE_SM90=1`; optimized and BI). Not switching to
  DeepGEMM: the checkpoint's routed experts are FP4, and DeepGEMM accepts FP4 operands only
  on SM100 (`check_ab_fp8_fp4` asserts `arch_major == 10`). Its Mega MoE is SM100-only too,
  and assigns token slots with `atomicAdd`.
- Shared expert: moves from `p5-det-gemm-v1` (det_gemm mid-split K-tree, BF16, well above
  the D7 gate) to DeepGEMM `fp8_gemm_nt`. The checkpoint stores the shared expert as FP8
  with 128x128 weight blocks, the vLLM fork runs it through DeepGEMM, and the Phase 1 probe
  found its `gate_up` / `down` shapes batch-invariant at 0.6-0.93x of eager BF16. This
  also matches the rollout's FP8 arithmetic. The SwiGLU between the two GEMMs stays ours.
  The shared expert gets a new numeric profile, so its P5 tests and golden bytes change.
- det_gemm's TP-equivalence property (a contiguous half-K shard is one child of the tree) is
  not needed under the D1 layout, so giving it up costs nothing.

### D3. Dense FP8 block-scaled GEMM (q/kv/indexer projections): three-step DeepGEMM probe

det_gemm is a BF16 fixed K-tree with no 1x128 / 128x128 FP8 scaling; the P5 routed kernel
is MXFP8xMXFP4 with block-32 scales, also the wrong format. Dense FP8 therefore goes through
DeepGEMM:

1. Pin one DeepGEMM commit, change nothing, and run the P5-3 invariance sweep against dense
   `fp8_gemm_nt` (M in {1, 8, 64, 256, 4096}). DeepGEMM's BLOCK_K=128 with one FP32
   promotion per 128-K block does not depend on M, so SM90 dense will likely pass as is.
2. If it fails: drop M from config selection in the C++ heuristic (pick a fixed tile from
   N/K only). Carry it as a patch under `third_party/patches/` applied at build time.
3. If that is still not enough: take `zhuzilin/DeepGEMM@batch_invariant` directly, with a
   capability probe that fails closed when the installed build does not support the
   contract (the same pattern vLLM uses).

**Outcome (2026-11-09): step 1 passes; steps 2 and 3 are not needed.** Dense FP8 uses
unmodified DeepGEMM pinned to `vllm-project/DeepGEMM@e21c821f` (the vLLM fork's pin), with
the default `compiled_dims="nk"` and FP32 UE8M0-valued scales. On an H200 with CUDA 12.8,
`scripts/probe_dsv4_deepgemm_bi.py` found every check byte-identical over 3 seeds
(`docs/validation/dsv4-deepgemm-bi-2026-11-09.json`):

- Dense `fp8_gemm_nt` at all seven DSv4 shapes (`fused_wqa_wkv`, `wq_b`, `wo_a` per group,
  `wo_b`, indexer `wq_b`, shared-expert `gate_up` / `down`): rows of an M=4096 batch equal
  the same rows at M in {1, 3, 8, 64, 129, 256, 1000}, alone at seven row offsets, in a
  mid-batch window, and on a second run.
- Grouped GEMM at the `wo_a` shape (8 groups): token outputs do not change when other
  groups gain tokens, between the contiguous and masked layouts, or with the masked
  `expected_m` hint (1, mean, max), and they equal per-group dense `fp8_gemm_nt`.
- A negative control (eager BF16 cuBLAS) differs with M on 4 of 7 shapes, so the
  comparisons do detect variance. This also backs D4.
- Performance (CUDA-graph replay, scales already in DeepGEMM's TMA-aligned layout):
  0.43-1.09x of eager BF16 cuBLAS on the same shapes, within D7. Converting the scale
  layout per call adds about 1.5 us at small M, so the activation-quant kernel should emit
  that layout directly.

Backward (same probe, same shapes, 3 seeds):

- dgrad (`dX = dY @ W`, NT kernel with `W.T`) passes the dense row checks. Block-quantizing
  `W.T` gives the same bytes as transposing the block-quantized `W`, so `W.T` can be
  precomputed once per weight.
- wgrad (`dW = dY.T @ X`, 1D1D kernel, `recipe=(1, 1, 128)`) is run-to-run identical over
  10 runs at 128, 1008 and 4096 tokens, including FP32 accumulation into an existing
  gradient. On SM90 it requires FP32 output with `c` (no BF16 output) and a token count
  that is a multiple of 16 (TMA stride); callers pad with zero tokens, which add nothing.
- Speed: wgrad 0.87-1.06x of BF16 at 4096 tokens; dgrad 0.41-1.0x except at small M for
  the long-K shapes (`wq_b` dgrad, K=32768: 1.56x at M=1, 1.29x at M=64; indexer `wq_b`
  dgrad: 1.23x at M=1). Training backward runs with thousands of tokens, where these are
  0.56-0.67x.
- Precision on Gaussian data: FP8 forward, dgrad and wgrad all have 3.8% relative L2
  error against FP32, versus 0.17% for BF16. That is the E4M3 rounding of the operands,
  not a kernel defect; whether FP8 gradients are acceptable has to be judged on training
  curves.

Limits: the result holds for this commit, this toolkit (CUDA 12.8; DeepGEMM recommends
12.9 for speed) and the default SM count (132). Re-run the probe when any of them changes,
and extend it if prefill batches exceed M=4096.

The verl training side must call the same kernel as rollout; otherwise we get BI without
train/inference parity. slime #2262's patch-set approach is the reference for patching the
training backend's kernels.

### D4. The router gate GEMM must not use bare `torch.mm`

vLLM can use `torch.mm` only because `VLLM_BATCH_INVARIANT=1` makes `batch_invariant.py`
override `aten.mm/addmm` with a fixed-tile Triton matmul. A bare cuBLAS FP32 sgemm chooses
split-K on its own at small M and is not BI. RL-Kernel runs the gate (`[T, 4096] x [256,
4096]`, FP32 out) on pinned DeepGEMM `bf16_gemm_nt`, like the compressor score GEMMs: BI
probe passed at N=256 (prefixes, single rows, window, 3 seeds; rel. error 4.4e-6; the
cuBLAS control varies at M = 1/3/8/64), 0.96-1.13x of cuBLAS. det_gemm is not used: its
`det_gemm_fwd_fp32` returns BF16 and is not bound in `rl_engine._C`.

### D5. Backward is relaxed

FlashMLA sparse has no backward. Training-side backward uses the verl training backend's
existing kernels with a fixed reduction; it must be run-to-run deterministic but is not
required to be bitwise against inference.

### D6. The formal anchor is verl

The formal WS1 anchor for #230 is verl: rollout and actor recompute on the same kernel set.
Until a verl anchor run is recorded, the slot keeps `MISSING_PROVENANCE`. The vLLM fork is
not written into #230's formal provenance. Its golden outputs are used only as a
development fixture under `tests/`, with the artifact explicitly labelled
`reference: vllm-ds4-preview@1a6542c (development only)`.

### D7. Performance gate

Each operator must stay within 1.3x of the non-invariant path to merge; anything above goes
on the Phase 5 list. The non-invariant path is the naive PyTorch implementation of the same
operator, or another efficient kernel for it that is not batch-invariant (for example cuBLAS
or DeepGEMM with default heuristics), measured on the same shapes. The det_gemm P5 shared
expert is above the gate today and stays merged until its DeepGEMM replacement lands
(Phase 3). BI does not mean giving up tensor cores: pinned tiles, no split-K, and
`allow_tf32=False` are sufficient (demonstrated in P5-3).

## 3. Pinning rules adopted from vLLM #54955 (to be written into the execution profile)

1. Attention never uses a batch-local grouped plan: prefill chunks each request
   independently; decode and prefill call the same `flash_mla_sparse_fwd`.
2. Top-k uses a composite (score, index) key with a stable tie-break, the same selector for
   prefill and decode, and canonicalizes the selected indices before FlashMLA.
3. Capacities come from the model config: top-k width, gather capacity, workspaces, and MoE
   expert capacity are all fixed, never derived from neighbours or `expert_num_tokens.max()`.
4. Activation quantization uses fixed launch geometry and 64-bit offsets (P5 already has the
   >= 2^31-element test).
5. mHC uses explicit pre/post operations, a fixed split count, and stable token order
   (consistent with P1).
6. Hopper keeps FP32 block scales; SM90 and SM100+ are not treated as numerically
   interchangeable.
7. A missing kernel is a hard error, never a silent fallback.

## 4. Reuse map

| Need | Reuse from | What RL-Kernel adds |
|---|---|---|
| Sparse MLA core (fwd) | FlashMLA `flash_mla_sparse_fwd` | Wrapper: num_splits/workspace pinned from config; canonical sorted indices; per-request chunking; stable addresses under CUDA graphs |
| Indexer top-k | Design of vLLM `csrc/ds4_bi_topk_extension.cu` | Own implementation (done, T04), byte-equal to it. The comparator is **not** shared with the router: vLLM's DSv4 router top-6 (`fused_moe/router/dsv4_topk.py`) breaks ties toward the **smallest** expert id, the indexer toward the largest local index; P3 follows the router's own order |
| Indexer logits (FP4/FP8 MQA scoring) | DeepGEMM `fp8_mqa_logits` / FP4 indexer | Verify head-sum and K reduction do not depend on M; fall back to own kernel if not |
| Dense FP8 GEMM | DeepGEMM (D3 three-step) | Sweep plus possible heuristic patch |
| Routed expert, MX act quant, SwiGLU, LoRA | Own P5 (routed: `sm90_fused_moe_mlp.cu`) | Unchanged |
| Shared expert GEMMs | DeepGEMM `fp8_gemm_nt` (D2, D3) | Provider with a new numeric profile; SwiGLU stays ours |
| EP dispatch/combine | `zhuzilin/DeepEP@align_fp8_quantization` | P4 adapter + route-metadata capture + fixed-capacity staging |
| Router gate | DeepGEMM `bf16_gemm_nt`, FP32 output (D4) | Done in the P3 redo (`rl_engine/models/deepseek_v4/router.py`) |
| mHC / RMSNorm / logprob | Own `dsv4-mhc-v1` (DeepGEMM TF32 GEMM + own epilogue) / linear_logp / #241 | mHC: fused norm, fixed 16 splits |
| Sliding-window dense attention | Own #240 or FA3 | Verify split-KV is pinned |
| Training integration | verl; slime #2262 for methodology | Rollout and actor recompute on the same kernel set; layerwise exact-zero diff gate; logprob CI threshold |

## 5. Phases

| Phase | Weeks | Goal | Parallelism |
|---|---|---|---|
| 0 | this week | Unblock #635 on this branch; merge main last | - |
| 1 | 1-2 | DeepGEMM three-step probe (BI + performance); then freeze the execution profile | overlaps the tail of Phase 0 |
| 2 | 2-5 | P2 attention T02-T07 (critical path) | parallel with Phase 3 |
| 3 | 3-6 | Redo P3 and P7; P6 T06; P4 DeepEP adapter | parallel with Phase 2 |
| 4 | 5-7 | Layerwise 0-diff gate -> #230 cross_config M0-M10 bitwise -> verl integration -> performance table | - |
| 5 | 7+ | Performance debt: operators above 1.3x, DCP (if required) | - |

## 6. Todo

### Phase 0 - #635 blockers (CUDA path)

Order: build gating first (everything else needs a clean build), then correctness fixes,
benchmark fixes, formatting, and merging main last.

- [x] SM90 MoE build gating. The two halves must change together, or `_C` import breaks:
  - `build_tools/extensions.py`: `sm90_fused_moe_mlp.cu` is compiled only on CUDA with
    `KERNEL_ALIGN_MOE_SM90=1`, under a dedicated `-DRL_KERNEL_ENABLE_MOE_SM90`. It does not
    reuse `RL_KERNEL_ENABLE_SM90`: det_gemm (`KERNEL_ALIGN_DET_GEMM_SM90=1`) also defines
    that macro, so an `ops.cpp` guard on it would bind `sm90_moe_*` while the source is not
    compiled (undefined symbol at import). The MoE build emits `sm_90a` only (a plain
    `compute_90` pass also compiles the WGMMA PTX and fails ptxas) and fails fast on a
    non-SM90 host.
  - `csrc/cuda/moe/sm90_fused_moe_mlp.cu`: kernel-body guards use
    `RL_KERNEL_ENABLE_MOE_SM90`.
  - `csrc/bindings/ops.cpp`: the `sm90_moe_*` declarations and `m.def` calls sit under
    `#if defined(RL_KERNEL_ENABLE_MOE_SM90)`. Before, the stubs were always bound, so the
    `hasattr` check in `rl_engine/moe/backends/sm90_fused_mlp.py` always passed and the
    failure only showed up at call time; now it fails closed at resolve time.
  - Tests: `tests/build/test_build_platform_collectives.py` covers the default, det_gemm-only,
    MoE, non-Hopper and ROCm configurations.
- [x] Fast-math is a hard error on ROCm too. hipcc defines `__FAST_MATH__`, while the
      `#error` guards in `fused_shared_expert_mlp.cu` and `clamp_swiglu_weighted.cu` check
      `__USE_FAST_MATH__`, so they did not fire. The error now names every affected kernel.
- [x] `fp32_gemm_rms.cu`: the G8 dispatch also requires `4*n_i + 8 <= 128`, so N=31 /
      T>288 takes the G2 fallback (the G8 kernel has 128 compute threads and needs 4N + 8
      slots); `test_eight_token_regime_edge_n_matches_oracle` covers N=30 and N=31.
- [x] Benchmark fixes: per-provider numeric profile in `benchmark_shared_expert_mlp.py`,
      TF32 flag save/restore in `benchmark_lora_delta.py`, external MX baselines as asserts
      in `benchmark_mxfp8_act_quant.py`, `.mul_(3.0)` in place in
      `test_p5_mxfp8_act_quant.py`
- [x] Keep `csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm_wgmma.cu` (not referenced by any build
      script or Python module yet)
- [x] `pre-commit run --all-files` passes (Black, isort, flake8)
- [x] Merge main last, after the fixes above pass (`9df841a1`)

Known failures outside Phase 0 (present before and after it, default build):
`test_p1_provider::test_golden_manifest_anchor` and four Triton `test_p5_lora_delta` cases
(Triton path, deferred).

### Phase 1 - DeepGEMM probe

Current goal: every operator runs, its performance is acceptable (D7), and it is
batch-invariant. The vLLM-fork golden is deferred (see "Deferred - vLLM golden").

- [x] Pin DeepGEMM to the commit the vLLM fork uses
      (`vllm-project/DeepGEMM@e21c821f39a2056d68067a466c64ddc942200106`; requires CUDA 12.8+)
- [x] DeepGEMM probe step 1: sweep dense `fp8_gemm_nt` (M in {1, 8, 64, 256, 4096}) at the
      DSv4 projection shapes, contiguous grouped and masked grouped, including "same tokens
      via contiguous vs masked"; check row bytes against M = 1 and run-to-run identity
- [x] Measure each probed GEMM against a non-invariant baseline on the same shapes (D7)
- [x] Only on failure: step 2 (heuristic patch) / step 3 (zhuzilin branch + capability
      probe): not needed
- [x] Decide the dense FP8 GEMM path and record it in D3 (step 1 passed)
- [ ] Afterwards: write `docs/architecture/dsv4-sm90-batch-invariant-profile.md` (D1-D7, the
      seven rules in section 3, the reuse map in section 4, the reduction-order contract for
      each M-module, the performance budget)

### Phase 2 - P2 attention (critical path)

- [ ] T02 KV compressor C4/C128 CUDA: block boundaries are per request, never crossing
      requests
  - What vLLM runs (2026-11-10 reading of `compressor.py`, `save_partial_states.py`,
    `fused_compress_quant_cache.py`): the score GEMM output is split into `kv` and `score`
    (`coff * head_dim` each, `coff = 2` for C4 overlap); `save_partial_states` writes
    `kv` and `score + ape[pos % ratio]` per token into a paged FP32 state cache; at each
    boundary token (`(pos + 1) % ratio == 0`) the kernel gathers the `coff * ratio` window
    (C4: the older 4 tokens use channel half 0, the newer 4 half 1; `pos < 0` masked), takes
    a per-channel softmax over the window, sums `kv * p`, RMSNorm (FP32, `rsqrt`), GPT-J
    RoPE at `compressed_pos = (pos // ratio) * ratio`, rounds to BF16 and quantizes.
    Indexer (head 128) on CUDA: Triton `_fused_kv_compress_norm_rope_insert_indexer_attn`,
    FP8 row + one FP32 power-of-two scale per entry. Main KV (head 512): CuTe-DSL
    `compress_norm_rope_store_cutedsl` writing the `fp8_ds_mla` layout (448 FP8 nope values
    with UE8M0 per 64, 64 BF16 rope values) that FlashMLA reads.
  - Parity measurement (`~/dsv4-work/cute_ref/probe_compressor.py`, not in the repo): a plain
    FP32 PyTorch version of the indexer compressor matches vLLM's Triton kernel on
    655359/655360 FP8 bytes and all scales (C4, 5 seeds, magnitudes 3e-5 to 50). FP8 rounding
    hides most last-bit differences of Triton's approximate exp / divide / rsqrt and its
    reduction trees, but not all: about 2e-4 mismatching entries, so a full model run would
    almost surely differ somewhere. **Decision (2026-11-10): (a)**, an own CUDA compressor
    with a defined fixed-order arithmetic (golden and CUDA byte-equal to each other;
    vLLM byte agreement is measured, not required). The options were: (a) own CUDA compressor with a
    defined fixed-order arithmetic (BI and deterministic; rollout parity only if the rollout
    also runs it), (b) emulate Triton's exact arithmetic (fragile across Triton versions),
    or (c) reuse vLLM's Triton/CuTe compressor kernels on the training side (bitwise with
    the vLLM rollout by construction, but a Triton dependency, against "CUDA first").
    The same question applies to every vLLM Triton/CuTe kernel on the DSv4 path whose
    output is not hidden behind an exact step (top-k, indexer Q and the probed DeepGEMM
    kernels are already exact matches)
  - [x] Indexer compressor (head 128, C4 overlap): golden
        `rl_engine/kernels/ops/pytorch/compressor/`, CUDA `csrc/cuda/compressor/compress_fp8.cu`
        (`docs/operators/compressor.md`). Byte-equal golden/CUDA (5 magnitude regimes, both
        rotary dtypes), batch-invariant, chunked prefill equals one pass; writes DeepGEMM's
        paged FP8 indexer cache. 1.05-1.14x of vLLM's Triton kernel, 99.999% of FP8 bytes
        equal to it. vLLM stores `rotate=True` but applies no Hadamard; neither do we
  - [x] Main compressor (head 512, C4 with overlap and C128): same kernel, writes vLLM's
        `fp8_ds_mla` cache layout (448 FP8 in 7 UE8M0 blocks of 64 + 64 BF16 RoPE values,
        576-byte rows, 8 scale bytes per entry) that the sparse attention dequantizes;
        `dequant_ds_mla` is the exact inverse. Byte-equal golden/CUDA, batch-invariant
        (`tests/ops/compressor/test_compress_main.py`); 0.98-1.07x of vLLM's Triton head-512
        kernel on C4, 0.38-0.41x on C128
- [ ] T03 indexer projection + FP4/FP8 logits: use the GEMM decided in Phase 1; run the
      invariance probe against DeepGEMM `fp8_mqa_logits` (5 seeds x 5 sub-batch sizes x
      production shape)
  - On SM90 the indexer is FP8 end to end: DeepGEMM's FP4 MQA-logits paths assert
    `arch_major == 10`, so vLLM's `use_fp4_indexer_cache` cannot be on for Hopper; the P2
    start kit's MXFP4 index cache is not the Hopper rollout format. FP8 cache row: 128 E4M3
    bytes + one FP32 scale per token (vLLM's V3.2 layout).
  - [x] Logits probe (`scripts/probe_dsv4_indexer_logits_bi.py`,
        `docs/validation/dsv4-indexer-logits-bi-2026-11-09.json`): unmodified pinned DeepGEMM,
        64 heads x 128, FP32 logits, 3 seeds. `fp8_fp4_mqa_logits` (prefill) is byte-identical
        run to run, for prefixes, single queries, windows, a request packed at another KV
        offset next to other requests, and the `max_seqlen_k` layout;
        `fp8_fp4_paged_mqa_logits` (decode) is byte-identical for a request alone vs in
        batches of 8/64 (different SM schedules) and under a permuted block table; and **paged
        decode logits equal prefill logits** for the same query and KV bytes. Each logit is
        one WGMMA over K = 128 plus a fixed-order head sum; no split across KV positions.
        Values match `sum_h w_h relu(q_h . k) k_scale` to 1.6e-4 relative (FP8 WGMMA
        accumulation)
  - [x] Indexer Q: RoPE (GPT-J interleaved, last 64 dims, BF16 round) + per-(token, head)
        power-of-two FP8 scale (amax floor 1e-4) folded into the per-head weights.
        `indexer_q_rope_fp8` (golden `rl_engine/kernels/ops/pytorch/indexer/`, CUDA
        `csrc/cuda/indexer/indexer_q_rope_fp8.cu`; `docs/operators/indexer-q.md`) is
        byte-equal to vLLM's **CuTe-DSL** `IndexerQFp8Kernel`, which the CUDA rollout runs
        (`nvidia-cutlass-dsl` is a hard dependency of the vLLM CUDA build); vLLM's Triton
        fallback rounds differently in three places and is not the target. FMA contraction
        found by running the compiled CuTe kernel (`scripts/check_indexer_q_vllm_parity.py`,
        50/50 cases). 0.011-0.015x of the eager golden; 1.0-1.14x of vLLM's kernel
  - [x] `weights_proj` (BF16 [hidden -> 64], unquantized) and the two compressor
        `fused_wkv_wgate` score GEMMs (FP32 output; N = 512 indexer, 1024 C128, 2048 C4) use
        unmodified DeepGEMM `bf16_gemm_nt` (`scripts/probe_dsv4_deepgemm_bf16_bi.py`,
        `docs/validation/dsv4-deepgemm-bf16-bi-2026-11-10.json`): batch-invariant at all four
        shapes (prefixes, single rows, window, run to run; 3 seeds; BF16 and FP32 output),
        cuBLAS negative control varies on all four; 0.90-1.28x of cuBLAS; FP32-output error
        4.4e-6 (FP32 accumulation). det_gemm is not used here: its K-tree rounds partial sums
        to BF16. Backward (dX / dW determinism) not probed yet
  - [x] KV selection: `rl_engine/kernels/ops/cuda/indexer/select.py`
        (`indexer_select_prefill` / `indexer_select_decode`: DeepGEMM FP8 MQA logits fed
        row-strided into `stable_topk_per_row`, as vLLM's BI path does). Ids equal the golden
        selector on the logits, are batch-invariant, decode selects exactly what prefill
        selects, and a decode request alone selects what it selects in a batch
        (`tests/ops/indexer/test_indexer_select.py`). The FP8 indexer K cache it reads is
        written by the compressor (T02)
  - [ ] Indexer op composition: Q path + logits + `stable_topk_per_row`. vLLM's
        short-context shortcut (`max_seq_len // ratio <= topk`, eager only) fills ascending
        ids `0..n-1` where the top-k path emits score order: same set, different order,
        chosen from the batch's longest request. FlashMLA's accumulation follows index order,
        so the T05 wrapper **must** canonicalize the indices (rule 2); then both orders give
        the same bytes
- [x] T04 stable top-k: composite key, fixed output slots, one selector for prefill and
      decode; regression covers "all ties + true cutoff". `stable_topk_per_row` (golden
      `rl_engine/kernels/ops/pytorch/topk/`, CUDA `csrc/cuda/topk/stable_topk.cu`;
      `docs/operators/stable-topk.md`) reproduces vLLM's BI selector
      `ds4_bi.top_k_per_row_prefill` byte for byte
      (`scripts/check_stable_topk_vllm_parity.py`, 80/80 cases incl. random bit patterns):
      key `(ordered_bits(score), local_index)` descending, so ties keep the **larger**
      request-local index; int32 ids in key order, `-1` padding. Decision (2026-11-09):
      follow vLLM; the P2 v1 start-kit contract (`finite_score_desc_global_id_asc`,
      `p2/oracle.topk512`) stays unchanged as a separate contract. 0.31-0.96x of
      `torch.topk` on H200 (decode and causal prefill, k = 512 and 2048), 1.4-3.7x
      faster than vLLM's own BI kernel
- [x] T05 FlashMLA sparse wrapper: canonicalize indices, pin num_splits/workspace from
      config, stable addresses under CUDA graphs; test `fwd(x)[t] == fwd(x[t:t+1])`
      byte-equal. Confirmed (2026-11-10): index canonicalization and config-fixed
      num_splits / workspace are hard requirements, not tuning choices
  - [x] Dependency and probe. vLLM pins `vllm-project/FlashMLA@a8f794d1`, which builds only
        against torch's stable ABI (>= 2.10; the vLLM fork runs torch 2.13, we run 2.9). We
        build `b70aff3` instead: `csrc/sm90` and `csrc/smxx` are identical to the pin (the five
        commits in between touch only bindings and SM100). `rl_engine/kernels/ops/cuda/flashmla.py`
        is the fail-closed check (SM90, `1.0.0+b70aff3`; loads the interface directly because
        that commit's package `__init__` re-exports a missing name).
        `scripts/probe_dsv4_flashmla_bi.py` (`docs/validation/dsv4-flashmla-bi-2026-11-10.json`):
        `flash_mla_sparse_fwd` at 64 heads x 512, index widths 640 and 1152, 3 seeds: output
        and LSE byte-identical run to run, for query prefixes / single queries / windows, with
        the KV workspace at another offset, and `topk_length` vs `-1` padding. Permuting the
        same index set **does** change the bytes, so canonicalization is required (vLLM's BI
        path sorts the top-k ids descending). 1.9e-3 relative to an eager reference with the
        sink; 9-24x faster than the eager gather + softmax
  - [x] Attention core: `rl_engine/kernels/ops/cuda/dsv4_attention.py` (vLLM's BI sparse path):
        per-request workspace slabs of `N + G` rows (`N = max_model_len // ratio`, fixed;
        dequantized compressed entries, then the dequantized window), canonical index lists
        (top-k ids sorted descending, then the window ascending, `-1` padded to a multiple of
        128), `flash_mla_sparse_fwd` with the sink. Gather / dequantize / indices are exact
        PyTorch indexing. For C4, C128 and sliding-window-only layers: values match eager
        attention, a request gives the same bytes alone or in another batch order, decode
        equals prefill and a chunked-prefill suffix, and a permuted top-k gives the same bytes
        (`tests/ops/attention_dsv4/test_sparse_core.py`)
  - [x] Query / output side (`csrc/cuda/attention/dsv4_qo.cu`, golden
        `rl_engine/kernels/ops/pytorch/attention_dsv4/`, same fixed-order contract as the
        compressor): q weightless RMSNorm + RoPE (padded heads zeroed) and KV RoPE + fp8_ds_mla
        insert into the sliding-window cache (vLLM's `fused_deepseek_v4_qnorm_rope_kv_insert`);
        attention output inverse RoPE + per-128 FP8 into per-group DeepGEMM A operands with
        MN-major scales (vLLM's `fused_inv_rope_fp8_quant`); `OutputProjection` = that + `wo_a`
        as one `fp8_gemm_nt` per group + `wo_b` (`Fp8Linear`). Byte-equal to the goldens,
        batch-invariant, 8% of FP32 at FP8 (`tests/ops/attention_dsv4/test_qo.py`). The `qr` /
        `kv` RMSNorms use the existing batch-invariant `rmsnorm_cuda` (P1)
- [x] T06 sliding window: confirm split-KV is pinned on the #240 / FA3 path. Not needed:
      as in vLLM's FlashMLA backend, the window runs through the same FlashMLA sparse kernel
      (window positions appended to each query's index list; sliding-window-only layers have
      no compressed part), covered by the T05 / T07 tests for ratio 1
- [x] T07 full-layer attention byte comparison against the golden + benchmark (<= 1.3x).
      `rl_engine/models/deepseek_v4/attention.py` (`Dsv4Attention`, forward only) composes the
      byte-verified operators (FP8 / BF16 DeepGEMM projections, P1 RMSNorm, q / KV kernel,
      compressors, indexer, FlashMLA core, output projection). For C4, C128 and
      sliding-window-only layers a request's output bytes are the same in a batch, alone, as
      chunked prefill and token by token in decode next to another request
      (`tests/models/test_dsv4_attention_layer.py`). H200, eager
      (`benchmarks/benchmark_dsv4_attention_layer.py`): prefill 1 x 4096 tokens 4.8 / 3.7 /
      2.5 ms (C4 / C128 / window-only), decode 64 requests at 4K context 5.2 / 2.7 / 1.6 ms;
      after the glue rewrite (Phase 5) 3.0 / 2.2 / 1.8 ms and 0.47 / 0.52 / 0.22 ms
- [ ] Wire the verl training-side recompute path to the same kernel set (forward only).
      Decisions (2026-11-10): the actor side is verl's **Megatron (mcore)** worker; verl's
      **vLLM rollout is patched too**, so both run this kernel set (D6; needed for bitwise
      rollout = recompute under decision (a)); and the work moves to the vLLM fork's
      environment: **torch 2.13.0 + CUDA 13.0, Python 3.12** (`_C`, DeepGEMM and FlashMLA
      rebuilt there, FlashMLA at vLLM's exact pin `a8f794d1`; every probe re-run). The layer
      itself already gives rollout-style prefill + decode the same bytes as full-sequence
      recompute (T07 tests)
  - [x] Environment moved (conda env `dsv4-t213`): torch 2.13.0+cu130, CUDA 13.0, Python
        3.12; FlashMLA at vLLM's exact pin `a8f794d1` (host compile needs
        `CPLUS_INCLUDE_PATH=$CUDA_HOME/include/cccl`: CUDA 13 moved libcu++ under `cccl/`).
        Full regression 1149 passed, 0 failed; every probe passes again
        (`docs/validation/*-torch213-cuda13-2026-11-10.json`: DeepGEMM FP8 87/87 cases,
        BF16, MQA logits, FlashMLA). The torch 2.9 env (`dsv4`, FlashMLA `b70aff3`) stays as a
        fallback; `require_flashmla` accepts both builds
  - Integration plan and findings (2026-11-10), to settle on the installed stack:
    - Actor: verl's DSv4 Megatron path uses `experimental_attention_variant="dsv4_hybrid"`,
      which exists only on Megatron-LM `main`
      (`megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py`),
      not in the `core_v0.19.2` verl pins: a BF16 training module (TE linears, TP = 1, no
      inference path). The patch replaces its forward on the recompute path (no grad) with
      `Dsv4Attention` over packed sequences (`cu_seqlens`) and temporary caches.
    - Rollout: replace the vLLM fork's `DeepseekV4Attention.forward` with the same operator
      sequence, reusing vLLM's caches, block tables and slot mappings (our cache formats are
      vLLM's: `fp8_ds_mla`, 132-byte FP8 indexer rows, `[kv | score + ape]` FP32 state).
    - Both patches take one RoPE table built by us, so the two sides cannot diverge there.
      This also replaces vLLM's YaRN on the window-only layers with plain RoPE, as in
      DeepSeek's inference code and Megatron-Core (`rl_engine/models/deepseek_v4/rope.py`).
    - Weights: both sides must run the same FP8 bytes. verl's FP8 sync quantizes with plain
      FP32 block scales (`scaled_fp8_blockwise`, Triton) but only for parameters named like
      `q_proj` / `o_proj` / `gate_proj`; DSv4's `wq_a` / `wkv` / `wq_b` / `wo_a` / `wo_b` do not
      match, and vLLM may re-quantize block scales to UE8M0 for DeepGEMM
      (`requant_weight_ue8m0_inplace`). Trace the real chain on the installed stack, then
      have the actor patch call the same functions on the same BF16 master weights

### Phase 3 - MoE and sampling

- [ ] P3 redo: gate GEMM, top-6 / hash routing, routing weights, combine-plan live
      provider; re-evaluate the T08 draft (#639) against the redo
  - [x] Gate GEMM on DeepGEMM `bf16_gemm_nt` FP32 output (D4), batch-invariant
  - [x] Routing (`docs/operators/dsv4-router.md`): own fixed arithmetic (vLLM writes
        softplus differently in its hash and scored kernels, with fast-math exp/log/sqrt);
        one formula `sqrt(max(x,0) + log1pf(expf(-|x|)))` for both; top-6 of `s + bias`,
        ties to the smallest expert id (the router's order, not the indexer's), NaN last;
        hash layers take `tid2eid[input_ids]`; weights `s / sum * 1.5` summed in slot order.
        Golden `rl_engine/kernels/ops/pytorch/router/`, CUDA `csrc/cuda/router/dsv4_router.cu`,
        byte-equal; the rollout patch runs the same kernel
  - [x] Backward with the gate and bias frozen (user decision): dX only, as a six-row
        FP32 gather over the gate (exact in slot order; byte-equal CUDA/golden)
  - [x] Performance (H200, graph replay, T = 1..16384): forward 1.03-1.22x of cuBLAS +
        vLLM's Triton `dsv4_topk`; dX 0.48-1.03x of the dense dgrad
  - [x] RoutePlan, EP = 1 (`docs/operators/dsv4-moe.md`, "Route plan and combine"):
        dispatch by (expert, global token, slot), combine / dX fan-in in slot order (P6
        policy) on a CUDA slot-sum kernel, byte-equal to the golden; the whole MoE block
        (router -> plan -> P5 -> combine + DeepGEMM shared expert + residual) is
        batch-invariant on one H200. Combine 0.06-0.42x and fan-in 0.16-0.91x of an
        `index_add_` scatter. P5-1 MXFP8 activation quant is now in the AOT build
  - [x] Expert parallelism over NCCL `all_to_all_single` (`Dsv4MoE(ep_group=...)`):
        MXFP8 codes / scales and `p_s` travel in plan order, the receiver stable-sorts by
        local expert (= (expert, global token, slot)), results return along the reverse
        splits; empty ranks join every collective. `scripts/check_dsv4_moe_ep.py`: EP = 2
        output bitwise equal to EP = 1 (scored and hash layers; uneven, decode, one-rank,
        empty-first splits; run to run). EP = 4/8 not run: this account is capped at 2
        GPUs per job (Slurm association `gres/gpu=2`)
  - [ ] DeepEP transport (P4) with the same bytes; EP = 4/8 once more GPUs are allowed;
        compare with #639
  - [x] Routed-expert backward on SM90 (profile `p5-sm90-bwd-bf16-v1`,
        `csrc/cuda/moe/sm90_moe_mlp_backward.cu`):
        - Steps: fc1 recompute, then `dh` (BF16 WGMMA, MXFP4 expanded exactly) fused with the
          clamp-SwiGLU backward, then `dx` in FP32. Frozen base, `dX` only.
        - Batch-invariant and run-to-run identical. The weight pre-expansion used for large M
          gives the same bytes as the inline path.
        - H200, EP 8 shapes: 0.44 / 0.44 / 0.65 / 0.74x of a per-expert cuBLAS BF16 loop at
          256 / 2048 / 8192 / 32768 rows (with the faster forward fc1).
        - The EP = 2 block backward is bitwise equal to EP = 1.
  - [ ] Real checkpoint: DSv4 FP4 experts are MXFP4 `[E, 2F, H/2]` + E8M0 `[E, 2F, H/32]`,
        gate rows then up rows (vLLM loader), the P5 layout; check the nibble order and
        P5's per-column scale-residual range ([-8, 6], fail-closed) on real shards
- [x] DSv4 mHC (`docs/operators/dsv4-mhc.md`, profile `dsv4-mhc-v1`): `mhc_pre` /
      `mhc_post` / `hc_head` with the attn_norm / ffn_norm / final norm fused, plus the dX
      backward. Parameters and norm weight are frozen (user decision); RMS is
      `rsqrt(mean + eps)`. Controller GEMM is DeepGEMM `tf32_hc_prenorm_gemm` with a fixed
      16 splits; everything after it is our CUDA. H-long dot products use a fixed 128-slice,
      runs-of-8 FMA partition. Byte-equal to the golden and batch-invariant. H200:
      pre + norm is 1.09-1.18x of vLLM's `mhc_pre_tilelang` (the gap is the fixed split
      count; the epilogues are equal), post 0.92-1.02x. A train step (fwd + bwd) is
      0.07-0.10x of naive PyTorch. vLLM's split count depends on T, so the rollout must run
      these kernels
- [x] P7 redo, kernels (`docs/operators/dsv4-logprob.md`, profile `dsv4-logprob-v1`): FP32
      head logits on DeepGEMM `bf16_gemm_nt` (probed BI and vocab-shard invariant), a
      64-tile fixed-order log-sum-exp in the #241 vocab-TP structure (TP 1/2/4/8 give the
      same bytes), dhidden backward via DeepGEMM `bf16_gemm_nn` computed in the forward
      when grad is needed. Byte-equal to the golden; BI. H200: fwd 0.79-1.05x, fwd + bwd
      0.84-1.04x of naive PyTorch. `Dsv4Model.selected_logprob`. Temperature divides the
      FP32 logits; with T = 1 it is the raw logprob the T02 sampling gate requires
- [ ] P7 integration: rollout head patch (vLLM's default head makes BF16 logits; ours are
      FP32), verl `linear_cross_entropy` hook, entropy output if verl needs it, a
      `torch.distributed` wrapper for a TP-sharded head, M10 wiring
- [ ] P6 T06 + real EP=2/4/8 collectives (replacing the values-only mock)
- [ ] P4: DeepEP `align_fp8_quantization` adapter, route-metadata capture (top-k slot
      order), fixed-capacity staging; empty ranks still join the collective
- [ ] DeepGEMM integration shared by the attention projections and the shared expert:
      optional dependency pinned to `vllm-project/DeepGEMM@e21c821f`, a fail-closed
      capability check (rule 7; DeepGEMM JIT-compiles, so `nvcc` from CUDA 12.8+ is needed at
      runtime), and a per-token 1x128 FP8 activation-quant kernel that emits scales directly
      in DeepGEMM's TMA-aligned layout (per-call conversion costs about 1.5 us)
  - [x] Activation quantization: `fp8_group_quant` (golden + CUDA, byte-equal; matches
        vLLM's `per_token_group_fp8_quant` with UE8M0 scales; token-direction variant for
        wgrad). See `docs/operators/fp8-group-quant.md`
  - [x] DeepGEMM dependency pin and fail-closed capability check: `dsv4` extra in
        `pyproject.toml`, `rl_engine/kernels/ops/cuda/deepgemm.py` (SM90, pinned
        `2.6.1+e21c821`, JIT `nvcc` >= 12.8)
  - [x] General dense FP8 linear for the attention/indexer projections and the shared
        expert: `rl_engine/kernels/ops/cuda/fp8_linear.py` (`Fp8Linear`: checkpoint or
        BF16 weights; forward = `fp8_group_quant` + `fp8_gemm_nt`, byte-equal to the rollout
        composition; FP8 dX through the precomputed `W.T`; FP32 dW on the 1D1D wgrad kernel,
        optionally accumulated). Forward and dX batch-invariant at the five DSv4 projection
        shapes, dW run-to-run identical (`tests/ops/gemm/test_dsv4_fp8_linear.py`). The
        shared-expert provider is built on it
- [ ] Shared expert on DeepGEMM: FP8 `gate_up` -> our SwiGLU -> FP8 `down`, as a P5 provider
      with a new numeric profile; batch-invariance and run-to-run tests, benchmark against
      eager BF16 and the det_gemm provider; update the P5 shared-expert tests and golden
      bytes; retire `p5-det-gemm-v1` from the DSv4 path
  - [x] `rl_engine/moe/backends/deepgemm_shared_expert.py`, profile `dsv4-deepgemm-fp8-v1`:
        forward byte-equal to the vLLM rollout composition (`fp8_group_quant`, DeepGEMM,
        vLLM's `silu_and_mul_with_clamp`, limit 10); dX only (weights frozen), FP8 DeepGEMM
        dgrad by default or det_gemm BF16 (`backward="bf16_det_gemm"`) as the reference.
        Forward and dX are batch-invariant and run-to-run identical. Relative error against
        FP32 on the dequantized weights (H200, T=512/2048): y 4.7%, FP8 dX 3.8%, BF16 dX 0.74%
  - [x] CUDA kernels for the clamped SwiGLU (`csrc/cuda/activation/silu_and_mul_clamp.cu`):
        forward fused with the FP8 group quant (same bytes as vLLM's two kernels; DeepGEMM
        has no SM90 epilogue fusion, and vLLM does not fuse it into the GEMM either) and the
        backward, both byte-equal to the golden over every BF16 gate value. Provider fwd+bwd
        is now 0.71-0.78x of eager BF16 (H=4096, F=2048, T=16..4096; was 2.1x)
  - [x] det_gemm launches an empty grid for M=0 and leaves a pending CUDA error that the
        next checked call reports: `gemm_dispatch` (and the short-token `db_transposed`
        path) now return empty outputs for empty M/N and zeros for K=0 without launching
        (`test_empty_shapes_launch_nothing_and_leave_no_pending_error`)
  - [ ] The det_gemm BF16 fallback needs a `KERNEL_ALIGN_DET_GEMM_SM90=1` build to be fast
- [x] The P5 oracle's shared-expert SwiGLU does not clamp and rounds once from FP32, unlike
      the rollout (`SiluAndMulWithClamp(10)` on BF16, two roundings). Decision: record the
      difference, do not change P5. `p5-det-gemm-v1` (oracle, `fused_shared_expert_mlp.cu`,
      its golden bytes) stays a separate profile off the DSv4 path; DSv4 uses
      `dsv4-deepgemm-fp8-v1`, which matches the rollout
- [ ] Re-run `scripts/probe_dsv4_deepgemm_bi.py` whenever the DeepGEMM commit, CUDA toolkit
      or SM count changes

### Phase 4 - integration and gates

- [x] Decoder layer + model forward (`docs/operators/dsv4-model.md`): embedding -> 4
      streams -> per layer mHC pre/post around attention and MoE -> `hc_head` + final norm;
      loader for the DeepSeek-V4-Flash checkpoint layout (spec checked against the real
      headers). Batch / alone / chunked / decode byte-identical. Against DeepSeek's
      inference code on a toy checkpoint: no wiring error (layer-0 attention 4.2%, MoE 2.5%
      rel L2, from vLLM-style FP8 choices). RoPE follows DeepSeek and Megatron-Core: plain
      RoPE on the window-only layers, YaRN on the compressed ones. vLLM applies YaRN to
      the window-only layers too, so the rollout patch must use our table on every layer.
      Forward only (no SM90 sparse attention backward yet)
- [ ] Layerwise exact-zero hidden-state diff gate, advancing one layer at a time from layer 0
- [ ] #230 cross_config M0-M10 selected-token logprob bitwise; add a logprob threshold to CI
      (target 0; slime's 1.9e-7 MAE is the reference upper bound)
- [ ] verl integration: rollout and actor recompute produce the same selected-token logprob
      on the same kernel set. Find which routed-expert kernel vLLM uses on Hopper (DeepGEMM
      cannot run FP4 there); if it is not our P5 kernel, either vLLM uses ours or we match
      its arithmetic
- [x] DeepGEMM backward (dgrad / wgrad) determinism, if the training side uses it (D5):
      dgrad is batch-invariant and wgrad run-to-run identical at the DSv4 shapes (Phase 1
      probe, `4d6e033e`; see D3 "Backward")
- [ ] Performance table: per-operator overhead against the non-invariant path; anything above
      1.3x goes to Phase 5

### Phase 5 - performance debt

- [ ] Operators that exceeded the Phase 4 performance table
- [x] P5 routed forward at large M, previously 2.2x the BF16 cuBLAS loop at 32768 rows:
      - Root cause: the consumer loop issued WGMMAs inside index-dependent branches, so
        ptxas serialized every WGMMA (C7518). The loop is now straight-line.
      - Changes: pre-expanded folded FP8 weights loaded by TMA, 128-token tiles with
        `m64n128k32` and `setmaxnreg`, persistent CTAs, grouped rasterization. Output bytes
        are unchanged and tested against the earlier build.
      - H200, EP 8 shapes, 256 / 2048 / 8192 / 32768 rows: forward 0.39 / 0.43 / 0.86 /
        1.02x, backward 0.44 / 0.44 / 0.65 / 0.74x
- [ ] P5 `fused` single-launch path: still has serialized WGMMAs and only 2 stages; apply
      the same loop and a BK=64 variant, or drop it
- [ ] MoE route plan as one counting-sort kernel (now PyTorch sort + searchsorted +
      scatter: 19-98 us, the slow part of the MoE glue at decode sizes)
- [x] DSv4 attention glue (was: C4 decode 64 x 4K at 5.2 ms, 2.1 ms of it GPU time). CUDA
      `ds_mla_gather` (exact fp8_ds_mla dequantize by slot) and `save_states`
      (`csrc/cuda/attention/dsv4_cache.cu`); index lists by scatter; sizes passed as host
      integers in `Dsv4Batch`, so a decode step reads nothing back and is CUDA-graph
      capturable (tested, replay = eager bytes); decode reads the paged indexer cache and
      builds a per-token workspace of only its selected rows (byte-equal to the slab path,
      tested). H200: decode 64 x 4K 0.47 / 0.52 / 0.22 ms (C4 / C128 / window-only, graph
      replay), prefill 1 x 4096 3.0 / 2.2 / 1.8 ms (eager)
- [ ] DCP attention (single comparator across local and global selection; separate issue)

### Deferred - vLLM golden

- [ ] Bring up the `ISEEKYAN/vllm:ds4-v9-rc1` image (`VLLM_BATCH_INVARIANT=1`,
      `VLLM_DS4_DECODE_KERNEL=sparse`) and produce selected-token logprob goldens for a fixed
      prompt set: batch 1/8/64, chunked prefill, decode
- [ ] Land the goldens under `tests/fixtures/dsv4/`, labelled development only; do not
      write them into #230's verl anchor provenance

### Deferred - Triton path

CUDA comes first. Revisit after the CUDA path is aligned:

- [ ] `rl_engine/kernels/ops/triton/mhc.py`: replace libdevice `mul_rn/add_rn` (FTZ flushes
      subnormals) with plain operators, and launch with `enable_fp_fusion=False` so the
      plain mul + add is not contracted into an FMA (the oracle requires mul-then-add, no
      fusion); add a subnormal byte-equality test that fails before the fix

### Explicitly out of scope (v1)

- DCP / CP attention
- Bitwise FlashMLA sparse backward
- ROCm (as stated on #635)
- Writing the vLLM fork into the formal anchor

## 7. References

- RL-Kernel #635, #447 (P2 T01), #498 (P3 T01), #448 (P6 T01), #462 (P6 T05), #648 (P7
  T02), #639 (P3 T08), #230 (cross_config), #241 (TP logprob)
- vLLM #54955 (DSv4 batch-invariant preview)
- verl (integration target)
- slime #2262 (Megatron/SGLang DSA alignment)
- `zhuzilin/DeepGEMM@batch_invariant`, `zhuzilin/DeepEP@align_fp8_quantization`
