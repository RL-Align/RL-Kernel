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
| P1 | mHC + RMSNorm + fp32_gemm_rms | Done (unfused canonical) | Keep as is |
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
split-K on its own at small M and is not BI. RL-Kernel uses det_gemm's FP32-output path
(`det_gemm_fwd_fp32`).

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
| Router gate | det_gemm FP32 output (`det_gemm_fwd_fp32`) | Wire into the redone P3 provider |
| mHC / RMSNorm / logprob | Own P1 / linear_logp / #241 | Unchanged |
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
  - [ ] Indexer Q: RoPE (GPT-J interleaved, last 64 dims, BF16 round) + per-(token, head)
        power-of-two FP8 scale (amax floor 1e-4) and the weight rescale
        `w * q_scale * softmax_scale * n_head**-0.5`, byte-equal to vLLM's
        `fused_indexer_q_rope_quant`
  - [ ] `weights_proj` (BF16 [hidden -> 64], unquantized): needs a batch-invariant BF16 GEMM
  - [ ] Indexer op composition: Q path + logits + `stable_topk_per_row`; vLLM's
        short-context shortcut (`max_seq_len // ratio <= topk` fills indices directly, eager
        only) decides per batch and must be checked against the BI contract
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
- [ ] T05 FlashMLA sparse wrapper: canonicalize indices, pin num_splits/workspace from
      config, stable addresses under CUDA graphs; test `fwd(x)[t] == fwd(x[t:t+1])`
      byte-equal
- [ ] T06 sliding window: confirm split-KV is pinned on the #240 / FA3 path
- [ ] T07 full-layer attention byte comparison against the golden + benchmark (<= 1.3x)
- [ ] Wire the verl training-side recompute path to the same kernel set (forward only)

### Phase 3 - MoE and sampling

- [ ] P3 redo: gate GEMM on det_gemm FP32 output (never bare `torch.mm`), `stable_topk6`
      with the comparator shared with the indexer top-k, routing-weights CUDA, combine-plan
      live provider; re-evaluate the T08 draft (#639) against the redo
- [ ] P7 redo: sampling gate + M10 selected-token logprob, wired to the #241 vocab-TP
      deterministic reduce
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
