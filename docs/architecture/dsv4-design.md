# DeepSeek-V4 on SM90: Batch-Invariant Kernel Set

The design of record for running DeepSeek-V4 (DSv4) rollout and LoRA training on Hopper
with **bitwise batch-invariant** forward numerics. Read this page first. Each operator has
its own page under `docs/operators/` with the full arithmetic, test list and benchmark
table; this page links to them.

- PR: RL-Align/RL-Kernel #635 (`test-dsv4-cuda`)
- Hardware: NVIDIA H100 / H200 (SM90). ROCm and SM100 are out of scope.
- Reference architecture: vLLM #54955 (`ISEEKYAN/vllm@1a6542c`, batch-invariant mode)
- Integration target: verl. Rollout and actor recompute must run this kernel set.
- Last updated: 2026-11-20

Contents:
1. Goal and guarantees
2. Getting started (environment, build, tests, benchmarks)
3. Model structure and module map
4. Decisions
5. Execution profile: pinning rules and the reduction order of every module
6. Numeric profiles
7. Training (LoRA, backward)
8. Validation
9. Performance
10. What the rollout must change
11. Legacy code not used by DSv4
12. Status and open work
13. Numerics lessons
14. References

## 1. Goal and guarantees

The RL problem: the logprob the trainer computes for a token must equal the logprob the
rollout engine computed when it sampled that token. Batch composition changes between
the two (different neighbours, packing, prefill vs decode), so every kernel must give a
token the same bytes whatever else is in the batch.

| Path | Guarantee |
|---|---|
| Forward (rollout, actor recompute, training forward) | **Bitwise batch-invariant.** A token's bytes do not depend on the other tokens in the step, the batch order, or whether its request runs as one prefill, chunked prefill or decode. `forward_train` is byte-equal to `forward_batch`. |
| Selected-token logprob | Bitwise batch-invariant, and invariant to a vocab-TP split (TP 1/2/4/8). |
| Expert parallelism | EP = 2 forward bitwise equal to EP = 1, with and without LoRA (tested on 2 GPUs). dX bitwise equal without LoRA; with LoRA, dX is the same run to run and within 1e-4 relative of EP = 1 (next row). |
| Backward (LoRA, dX path) | Correct against FP32 autograd per component. **Not required to be deterministic** (user decision 2026-11-11): cuDNN's sparse-attention dKV uses FP32 atomics. Not batch-invariant either: `lora_backward` runs FP32 `torch.mm`, and cuBLAS picks its kernel from the row count, so a row's dX depends on how many rows share the call (seen on the routed `fc2` adapter, ~1e-4 relative). |
| LoRA weight gradients | Deterministic for fixed shapes; not batch-invariant (a reduction over tokens). |
| Performance | Every forward operator within 1.3x of its non-invariant baseline (section 9). |

"Batch-invariant" is always tested bytewise: `fwd(x)[t] == fwd(x[t:t+1])`, prefixes,
windows, permuted batches, chunked prefill vs one pass, decode vs prefill, and run to run.
A tolerance never stands in for it.

## 2. Getting started

### 2.1 Environment

| Component | Version | Notes |
|---|---|---|
| GPU | H100 / H200 (SM90, 132 SMs) | The BI probes ran at 132 SMs; re-run them on another SM count |
| PyTorch | 2.13 with CUDA 13.0 | Earlier work ran torch 2.9 / CUDA 12.8; reports for both exist |
| CUDA toolkit | 13.0 (`nvcc` on `PATH`, `CUDA_HOME` set) | DeepGEMM JIT-compiles at runtime and needs `nvcc` >= 12.8 |
| DeepGEMM | `vllm-project/DeepGEMM@e21c821f` (`2.6.1+e21c821`) | `pip install -e ".[dsv4]"` or build from a clean checkout with `--no-build-isolation`; `rl_engine/kernels/ops/cuda/matmul/deepgemm.py` fails closed on another version |
| FlashMLA | `1.0.0+b70aff3` or `1.0.0+a8f794d` | SM90 sources identical to vLLM's pin; `FLASH_MLA_DISABLE_SM100=1 pip install --no-build-isolation --no-deps ./FlashMLA`; checked by `rl_engine/kernels/ops/cuda/attention/flashmla.py` |
| cuDNN frontend | `nvidia-cudnn-frontend==1.31.0` | Training only (sparse attention backward, `dsa_backward.py`) |

Every dependency check fails closed (rule 7 below). `RL_KERNEL_FLASHMLA_ALLOW_UNPINNED=1`
and `RL_KERNEL_CUDNN_DSA_ALLOW_UNPINNED=1` accept other builds for development only.

### 2.2 Build

The routed-expert kernels are behind a build flag:

```bash
export CUDA_HOME=/path/to/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0 RL_KERNEL_REQUIRE_EXT=1 MAX_JOBS=24
KERNEL_ALIGN_MOE_SM90=1 python setup.py build_ext --inplace
python -c 'import rl_engine._C as c; print(c.sm90_moe_fc1_swiglu_quant_forward)'
```

`KERNEL_ALIGN_MOE_SM90=1` builds `csrc/cuda/moe/sm90_*` for `sm_90a` only. Without it
the MoE and model tests skip. Never build with `--use_fast_math` (section 13).

### 2.3 Tests

The full DSv4 regression on one GPU (H200: 1217 passed, 72 skipped, 80 s):

```bash
export DSV4_META=/path/to/dsv4-meta DSV4_REF=/path/to/dsv4-inference   # optional, see below
python -m pytest -q -p no:cacheprovider \
    tests/ops/logprob_dsv4 tests/ops/mhc tests/models tests/ops/router tests/ops/moe \
    tests/ops/compressor tests/ops/indexer tests/ops/topk tests/ops/attention_dsv4 \
    tests/ops/quant tests/ops/activation tests/ops/gemm/test_dsv4_fp8_linear.py \
    tests/ops/gemm/test_deepgemm_capability.py tests/test_deepgemm_shared_expert.py \
    tests/test_sm90_fused_moe_mlp.py tests/test_sm90_moe_backward.py tests/build
```

- `DSV4_META`: a directory with the real checkpoint's `config.json` and `headers.json`
  (`scripts/make_dsv4_toy_checkpoint.py --fetch` downloads them; no weights). Enables the
  loader-spec test.
- `DSV4_REF`: DeepSeek-V4's inference directory (`model.py`, `kernel.py`). Enables the
  check against DeepSeek's own forward.
- CPU-only tests (goldens, sampling gate, M10) run in the GitHub `CI-Pipeline` workflow.

The model tests use a random checkpoint at the real dimensions (H 4096, 64 heads,
256 experts) with 4 layers, one of each attention type.

### 2.4 Layerwise gate

```bash
PYTHONPATH=. python scripts/check_dsv4_layerwise_gate.py --out report.json           # random weights
PYTHONPATH=. python scripts/check_dsv4_layerwise_gate.py --ckpt DIR --layers 3 --no-train
```

It runs a continuous-batching rollout and the actor recompute (and the training forward)
on the same sequences and reports, per layer, how many tokens differ (section 8.3).

### 2.5 Benchmarks

Every benchmark times CUDA-graph replay (eager timing measures host overhead at small
shapes) and prints the ratio to its non-invariant baseline.

| Benchmark | Operator |
|---|---|
| `benchmarks/benchmark_dsv4_attention_qo.py` | q-norm / RoPE / KV insert, output projection |
| `benchmarks/benchmark_dsv4_attention_layer.py` | whole attention layer (prefill, decode) |
| `benchmarks/benchmark_dsv4_attention_backward.py` | attention backward |
| `benchmarks/benchmark_compressor.py` | KV compressors |
| `benchmarks/benchmark_indexer_q.py`, `benchmark_stable_topk.py` | indexer query, top-k |
| `benchmarks/benchmark_dsv4_router.py` | router forward / dX |
| `benchmarks/benchmark_dsv4_moe_combine.py` | route plan, combine, dX fan-in |
| `benchmarks/benchmark_sm90_fused_moe_mlp.py`, `benchmark_sm90_moe_backward.py` | routed experts |
| `benchmarks/benchmark_shared_expert_mlp.py` | shared expert |
| `benchmarks/benchmark_mxfp8_act_quant.py` | MXFP8 activation quant |
| `benchmarks/benchmark_dsv4_mhc.py` | mHC pre / post / train step |
| `benchmarks/benchmark_dsv4_logprob.py` | logprob head |

Run them with `PYTHONPATH=. python benchmarks/<name>.py`.

### 2.6 Probes

The `scripts/probe_dsv4_*.py` scripts establish that a third-party kernel is
batch-invariant before we depend on it. Their reports are in `docs/validation/`.

| Probe | Checks | Report |
|---|---|---|
| `probe_dsv4_deepgemm_bi.py` | DeepGEMM FP8 GEMM forward / dgrad / wgrad at the DSv4 shapes | `dsv4-deepgemm-bi-torch213-cuda13-2026-11-10.json` |
| `probe_dsv4_deepgemm_bf16_bi.py` | DeepGEMM BF16 GEMM (router, compressor scores, `weights_proj`) | `dsv4-deepgemm-bf16-bi-torch213-cuda13-2026-11-10.json` |
| `probe_dsv4_indexer_logits_bi.py` | DeepGEMM MQA logits, prefill and paged decode | `dsv4-indexer-logits-bi-torch213-cuda13-2026-11-10.json` |
| `probe_dsv4_flashmla_bi.py` | FlashMLA sparse forward | `dsv4-flashmla-bi-torch213-cuda13-2026-11-10.json` |
| `probe_dsv4_head_gemm.py` | logits GEMM, vocab shards | printed |
| `probe_dsv4_wo_a_einsum.py`, `probe_dsv4_wo_a_grouped.py` | `fp8_einsum` vs per-group GEMMs; masked grouped GEMM (rejected) | printed |
| `probe_dsv4_grouped_lora_gemm.py` | grouped BF16 GEMM for future per-expert LoRA | printed |

Parity scripts against vLLM's compiled kernels: `check_indexer_q_vllm_parity.py`,
`check_stable_topk_vllm_parity.py`. Against DeepSeek's inference code:
`check_dsv4_model_ref.py`. Real FP4 checkpoint scale range: `check_dsv4_fp4_scales.py`.
EP = 2: `PYTHONPATH=. torchrun --nproc-per-node 2 scripts/check_dsv4_moe_ep.py [--lora]`
(needs 2 GPUs).

## 3. Model structure and module map

The forward follows vLLM's `DeepseekV4Model` in its batch-invariant mode (unfused pre /
block / post):

```text
streams = embed[input_ids] copied to 4 streams                         BF16 [T, 4, H]
per layer i:
    x, post, comb = mhc_pre(streams, hc_attn_*, attn_norm)              mHC + fused RMSNorm
    streams       = mhc_post(attention_i(x), streams, post, comb)
    x, post, comb = mhc_pre(streams, hc_ffn_*, ffn_norm)
    streams       = mhc_post(moe_i(x, input_ids), streams, post, comb)
hidden = hc_head(streams, hc_head_*, norm)                              BF16 [T, H]
logp   = selected_logprob(hidden, head, targets)                       FP32 [T]
```

Layer `i` takes its type from the config: attention `compress_ratios[i]` is 0 (128-token
sliding window only), 4 (C4: compressed KV with the indexer's top-512) or 128 (C128: every
compressed entry); routing is hash (`tid2eid[input_ids]`) for `i < num_hash_layers`,
scored top-6 otherwise.

### 3.1 Where things are

| Module | Python entry | CUDA kernels | Operator page | Tests |
|---|---|---|---|---|
| Model, layer, loader, RoPE | `rl_engine/models/deepseek_v4/{model,loader,config,rope}.py` | (composes the rest) | [dsv4-model](../operators/dsv4-model.md) | `tests/models/deepseek_v4/` |
| mHC + fused norms | `rl_engine/kernels/ops/cuda/mhc/` | `csrc/cuda/mhc/dsv4_mhc.cu` + DeepGEMM `tf32_hc_prenorm_gemm` | [dsv4-mhc](../operators/dsv4-mhc.md) | `tests/ops/mhc/` |
| Attention layer | `rl_engine/models/deepseek_v4/attention.py` | see below | this page, 5.2 | `tests/models/test_dsv4_attention_layer.py` |
| Dense FP8 linear | `rl_engine/kernels/ops/cuda/matmul/fp8_linear.py` | DeepGEMM `fp8_gemm_nt` / `fp8_einsum` | [fp8-group-quant](../operators/fp8-group-quant.md) | `tests/ops/gemm/test_dsv4_fp8_linear.py` |
| FP8 activation quant | `rl_engine/kernels/ops/cuda/quant/` | `csrc/cuda/quant/fp8_group_quant.cu` | [fp8-group-quant](../operators/fp8-group-quant.md) | `tests/ops/quant/` |
| q / KV side, output projection, caches | `rl_engine/kernels/ops/cuda/attention/dsv4_attention.py` | `csrc/cuda/attention/dsv4_qo.cu`, `dsv4_cache.cu` | this page, 5.2 | `tests/ops/attention_dsv4/` |
| KV compressors | `rl_engine/kernels/ops/cuda/compressor/` | `csrc/cuda/compressor/compress_fp8.cu` | [compressor](../operators/compressor.md) | `tests/ops/compressor/` |
| Indexer query | `rl_engine/kernels/ops/cuda/indexer/indexer_q.py` | `csrc/cuda/indexer/indexer_q_rope_fp8.cu` | [indexer-q](../operators/indexer-q.md) | `tests/ops/indexer/` |
| Indexer logits + selection | `rl_engine/kernels/ops/cuda/indexer/select.py` | DeepGEMM MQA logits + `csrc/cuda/topk/stable_topk.cu` | [stable-topk](../operators/stable-topk.md) | `tests/ops/indexer/test_indexer_select.py`, `tests/ops/topk/` |
| Sparse attention core | `rl_engine/kernels/ops/cuda/attention/flashmla.py`, `dsv4_attention.py` | FlashMLA `flash_mla_sparse_fwd` | this page, 5.2 | `tests/ops/attention_dsv4/test_sparse_core.py` |
| Attention backward | `rl_engine/models/deepseek_v4/attention_backward.py`, `kernels/ops/cuda/attention/dsa_backward.py` | cuDNN `dsa_bwd_sm90`, `csrc/cuda/attention/dsv4_attn_bwd.cu` | this page, 7 | `tests/models/test_dsv4_attention_backward.py`, `tests/ops/attention_dsv4/test_sparse_backward.py` |
| Router | `rl_engine/models/deepseek_v4/router.py`, `kernels/ops/cuda/router/` | `csrc/cuda/router/dsv4_router.cu` + DeepGEMM gate | [dsv4-router](../operators/dsv4-router.md) | `tests/ops/router/`, `tests/models/test_dsv4_router_layer.py` |
| MoE block (plan, dispatch, combine, EP) | `rl_engine/models/deepseek_v4/moe.py`, `kernels/ops/cuda/moe/` | `csrc/cuda/moe/moe_route_plan.cu`, `moe_slot_sum.cu` | [dsv4-moe](../operators/dsv4-moe.md) | `tests/models/test_dsv4_moe_layer.py`, `tests/ops/moe/` |
| MXFP8 activation quant | `rl_engine/kernels/ops/cuda/moe/mxfp8_act_quant.py` | `csrc/cuda/moe/mxfp8_act_quant.cu` | [dsv4-moe](../operators/dsv4-moe.md) | `tests/ops/moe/test_mxfp8_act_quant.py` |
| Routed experts | `rl_engine/moe/backends/sm90_fused_mlp.py` | `csrc/cuda/moe/sm90_fused_moe_mlp.cu`, `sm90_moe_mlp_backward.cu`, `moe_swiglu_quant.cu` | [dsv4-moe](../operators/dsv4-moe.md) | `tests/test_sm90_fused_moe_mlp.py`, `tests/test_sm90_moe_backward.py` |
| Shared expert | `rl_engine/moe/backends/deepgemm_shared_expert.py` | DeepGEMM + `csrc/cuda/activation/silu_and_mul_clamp.cu` | [dsv4-moe](../operators/dsv4-moe.md) | `tests/test_deepgemm_shared_expert.py`, `tests/ops/activation/` |
| LoRA | `rl_engine/models/deepseek_v4/lora.py` | DeepGEMM BF16 GEMMs | this page, 7.1 | `tests/models/test_dsv4_lora.py`, `test_dsv4_moe_lora.py` |
| Logprob head | `rl_engine/kernels/ops/cuda/logprob_dsv4/` | `csrc/cuda/logprob/dsv4_logprob.cu` + DeepGEMM | [dsv4-logprob](../operators/dsv4-logprob.md) | `tests/ops/logprob_dsv4/` |
| Sampling gate, M10 | `rl_engine/validation/sampling_params/`, `rl_engine/validation/m10/` | none | [dsv4-sampling-contract](../contracts/dsv4-sampling-contract.md) | `tests/validation/sampling_params/`, `tests/validation/m10/` |
| Layerwise gate | `rl_engine/models/deepseek_v4/layerwise_gate.py` | none | section 8.3 | `tests/models/deepseek_v4/test_dsv4_layerwise_gate.py` |

Every CUDA operator has a PyTorch golden under `rl_engine/kernels/ops/pytorch/<op>/`
defined on CUDA (section 13), and its test checks the CUDA kernel byte-equal to it.

## 4. Decisions

### 4.1 Architecture (D1-D7)

| | Decision | Why |
|---|---|---|
| D1 | DP-attention + EP-MoE, TP = 1 inside attention | Each rank owns whole requests, so nothing on the attention path reduces across ranks. The only cross-rank steps are EP dispatch / combine (fixed order), the vocab-TP logprob merge and the mHC broadcast. DCP / CP attention is out of scope (vLLM's local and global selectors break ties differently) |
| D2 | Routed experts on our own SM90 kernel (`sm90_fused_moe_mlp.cu`, MXFP8 x MXFP4) | The checkpoint's routed experts are FP4; DeepGEMM accepts FP4 only on SM100, and its Mega MoE assigns slots with `atomicAdd` |
| D3 | Dense FP8 GEMMs on **unmodified** DeepGEMM pinned at `e21c821f` | The probe found every DSv4 shape batch-invariant as shipped (one FP32 promotion per 128-K block, tile choice from N / K). Shared expert moved there too (0.71-0.78x of eager BF16) |
| D4 | No bare `torch.mm` / cuBLAS on the forward path | cuBLAS picks split-K at small M (the probes' negative control varies). Router gate, compressor scores, `weights_proj` and logits use DeepGEMM `bf16_gemm_nt` with FP32 output |
| D5 | Backward is relaxed | FlashMLA has no backward; training needs correct gradients, not bitwise ones |
| D6 | The formal anchor is verl | Rollout and actor recompute on this kernel set. The vLLM fork is a development reference only |
| D7 | Each operator within 1.3x of its non-invariant path | Baseline = naive PyTorch or an efficient non-BI kernel on the same shapes |

### 4.2 Product decisions (user, 2026-11-09 .. 11)

- **LoRA targets:** attention `wq_a`, `wkv`, `wq_b`, `wo_b`; MoE routed `fc1` / `fc2`
  (one adapter shared by all experts, Megatron-Bridge's default
  `share_expert_adapters=True`); shared-expert `fc1` / `fc2`. Base weights, embedding,
  mHC parameters, router gate and bias are frozen (dX only).
- **LoRA is unmerged** so the rollout runs the same forward with FP8 base weights.
- **Per-expert LoRA adapters are deferred** until verl needs them. When built they use
  DeepGEMM's grouped BF16 GEMM (probed byte-equal to per-expert GEMMs) under a new
  profile `dsv4-lora-expert-v1` with the delta rounded to BF16.
- **Backward determinism is not required.**
- **RoPE:** plain RoPE on sliding-window-only layers (DeepSeek's code, Megatron-Core),
  YaRN on compressed layers. vLLM uses YaRN everywhere; the rollout patch must pass our
  tables (section 10).
- **mHC:** RMS is `rsqrt(mean + eps)`; own fused kernels with a new profile.
- **Indexer top-k** follows vLLM's order (ties to the larger index); the **router** top-6
  breaks ties toward the smaller expert id (vLLM's router order).
- **The P5 fused single-launch MoE path was dropped**; the two-launch path is the only
  routed forward.
- **Chunked training** (gradient carried across chunks of one sequence) is not planned;
  packed multi-sequence batches work.

## 5. Execution profile

### 5.1 Pinning rules

From vLLM #54955 (1-7) and from this branch (8-12). Each is enforced in code and covered
by tests.

1. **No batch-local plan in attention.** Each request owns a workspace slab of fixed size
   (`N + G` rows, `N = max_model_len // ratio`); prefill and decode call the same
   `flash_mla_sparse_fwd` with `num_splits` / workspace fixed from the config.
2. **One selector, canonical order.** Indexer top-k uses the composite key
   `(ordered_bits(score), local_index)`, the same kernel for prefill and decode. Ids are
   sorted descending before FlashMLA: its accumulation follows index order, and a permuted
   index set changes the bytes.
3. **Capacities come from the config**, never from the batch (`expert_num_tokens.max()`
   and the like are banned).
4. **Fixed launch geometry for quantization** with 64-bit offsets.
5. **mHC**: explicit pre / post, a fixed 16-way split in the controller GEMM, stable token
   order.
6. **Hopper keeps FP32 block scales** (UE8M0-valued); SM90 and SM100 are not numerically
   interchangeable.
7. **A missing kernel is a hard error.** DeepGEMM, FlashMLA, cuDNN and `_C` symbol checks
   fail closed; there is no silent fallback to a non-BI path.
8. **No split-K and no atomics on the forward path.** Every GEMM reduces its full K inside
   one CTA in a fixed order; scatter-adds are replaced by fixed-order slot sums.
9. **Correctly rounded FP32 in our kernels**: `__fadd_rn` / `__fmul_rn` / `div.rn`, no FMA
   contraction unless the contract names an `fmaf` chain, no `.ftz`, `--use_fast_math`
   refused.
10. **Transcendentals are libdevice** (`expf`, `logf`, `log1pf`), so goldens are defined on
    CUDA, where `torch.exp` is the same function.
11. **Goldens divide by tensors, not Python scalars** (CUDA PyTorch multiplies by the
    reciprocal for a scalar divisor).
12. **Host syncs are explicit.** A decode step reads nothing back: sizes travel as host
    integers in `Dsv4Batch`, and with `model.set_finite_check("defer")` the MoE finite
    check writes a device flag (`model.raise_if_nonfinite()` reads it), so a whole decode
    step is CUDA-graph capturable at EP = 1.

### 5.2 Reduction order per module

"Row-local" means a row's output reads only that row and frozen weights, which makes the
module batch-invariant by construction. Full formulas are on the linked pages.

**Model skeleton**

| Module | Kernel | Reduction order |
|---|---|---|
| Embedding | indexing | Exact copy into 4 identical streams (vLLM pre-sums `hc_attn_fn` over the streams instead: equal only in exact arithmetic) |
| `mhc_pre` / `hc_head` controller | DeepGEMM `tf32_hc_prenorm_gemm`, 16 splits | Partials reduced by halving (`p[0:8] + p[8:16]`, then 4, 2, 1); `hc_head`'s padded `fn` cached per tensor |
| mHC per-token math + fused RMSNorm | `dsv4_mhc.cu` | Correctly rounded steps; "sum of four" = `(a0 + a1) + (a2 + a3)`; Sinkhorn 19 x (row, column); H-long dots use a fixed 128-slice partition (runs of 8, `fmaf` chain, halving within 32, then sum of four) |
| RoPE tables | built once | YaRN tables bit-equal to vLLM's; plain tables bit-equal to DeepSeek's `precompute_freqs_cis` |

**Attention**

| Module | Kernel | Reduction order |
|---|---|---|
| Projections `wq_a`, `wkv`, `wq_b`, `wo_b` | `Fp8Linear`: `fp8_group_quant` + DeepGEMM `fp8_gemm_nt` | Per-128 group absmax (row-local), UE8M0 scale from the FP32 bits; full K per CTA |
| q norm + RoPE, KV RoPE + `fp8_ds_mla` insert | `dsv4_qo.cu` | Row-local; sum of squares 4 channels per thread, 32-lane butterfly, warps ascending |
| KV compressors (C4, C128, indexer head 128) | `compress_fp8.cu` | Per channel over the window: max, `expf`, ascending sum, `kv * e` sum without FMA; reads only its own request's state through the block table |
| Compressor score GEMMs, `weights_proj` | DeepGEMM `bf16_gemm_nt`, FP32 out | No split-K |
| Indexer query | `indexer_q_rope_fp8.cu` | Row-local; byte-equal to vLLM's CuTe-DSL kernel (FMA where it uses FMA) |
| Indexer logits | DeepGEMM `fp8_fp4_mqa_logits` (prefill) / `fp8_fp4_paged_mqa_logits` (decode) | One WGMMA over K = 128 per logit plus a fixed-order head sum; paged decode logits equal prefill logits |
| Indexer top-k | `stable_topk.cu` | Exact integer keys; ties to the larger request-local index |
| Index canonicalization | indexing | Top-k ids descending, then the window ascending, `-1` padded to a multiple of 128 |
| Sparse attention | FlashMLA `flash_mla_sparse_fwd` with the sink | Config-fixed `num_splits` / workspace, per-request slabs, canonical index order |
| Output projection | inverse RoPE + FP8, `wo_a` as DeepGEMM `fp8_einsum("bhr,hdr->bhd")` below 8192 tokens and per-group `fp8_gemm_nt` above, then `wo_b` | Both `wo_a` paths give the same bytes at every T = 1..16384, so the switch is not a batch dependence |

The sliding window runs through the same FlashMLA kernel (window positions appended to
each query's index list), so there is no separate window kernel to pin. Decode reads the
paged indexer cache and builds a per-token workspace of only its selected rows
(byte-equal to the slab path).

**MoE**

| Module | Kernel | Reduction order |
|---|---|---|
| Router gate | DeepGEMM `bf16_gemm_nt`, FP32 out | No split-K |
| Router top-6 / hash | `dsv4_router.cu` | `sqrt(softplus)` with libdevice `expf` / `log1pf`; ties to the smaller expert id; weights summed left to right over the 6 slots, times 1.5 / sum |
| Route plan | `moe_route_plan.cu` | Integer counting sort, stable on `(expert, global token, slot)` |
| MXFP8 activation quant | `mxfp8_act_quant.cu` | Per-32 block, integer max over `|x|` bits, scale from the exponent field; once per token, gathered to its rows |
| Routed experts | `sm90_fused_moe_mlp.cu` (`p5-sm90-fused-mlp-v1`) | Fixed K walk per CTA, no split-K; the 32 products of an MX block summed by the tensor core; clamped SwiGLU (limit 10), `p_s` applied once |
| Shared expert | DeepGEMM `fp8_gemm_nt` + `silu_and_mul_clamp.cu` (`dsv4-deepgemm-fp8-v1`) | As the dense projections; SwiGLU byte-equal to vLLM's two kernels |
| Combine | `moe_slot_sum.cu` | Per token in FP32: slot 0, slots 1..5 in order, shared expert, residual; one BF16 round |
| Expert parallelism | NCCL `all_to_all_single` | Rows travel in plan order; the receiver stable-sorts by local expert, which is the same `(expert, global token, slot)` order |

**Head**

| Module | Kernel | Reduction order |
|---|---|---|
| Logits | DeepGEMM `bf16_gemm_nt`, FP32 out | No split-K; vocab shards give the same columns |
| Log-sum-exp | `dsv4_logprob.cu` (`dsv4-logprob-v1`) | 64 fixed vocab tiles, 32 lanes per tile over strided columns then halving; tile merge by max, rescale, `(e_i + e_{i+32})` then halving; `lse = M + logf(S)` |
| Chunking | `chunk_rows` (default 4096) | Each row is computed alone, so chunking changes no byte |

## 6. Numeric profiles

A numeric profile names a fixed arithmetic. A backend that cannot hold one declares a new
profile rather than relaxing a tolerance.

| Profile | Module | Byte-equal to |
|---|---|---|
| `dsv4-mhc-v1` | mHC pre / post / `hc_head` with the fused norms | its golden (differences from vLLM's TileLang kernels on the mHC page) |
| `dsv4-logprob-v1` | logits + LSE + logprob | its golden; TP 1/2/4/8 merge |
| `dsv4-deepgemm-fp8-v1` | shared expert | the vLLM rollout composition (same DeepGEMM calls and SwiGLU) |
| `p5-sm90-fused-mlp-v1` | routed experts, forward | itself across batches (deterministic, BI; not oracle-equal) |
| `p5-sm90-bwd-bf16-v1` | routed experts, dX | FP32 autograd within tolerance |
| `dsv4-lora-v1` | every adapter | its golden; zero adapters = base bytes |
| (vLLM's) | `fp8_group_quant`, indexer query, stable top-k, compressors | vLLM's kernels, byte for byte |

## 7. Training (LoRA)

`Dsv4Model.forward_train(input_ids, batch, caches)` runs the forward under autograd and is
byte-equal to `forward_batch`. `attach_lora(model, rank, alpha)` creates the adapters and
`model.lora_parameters()` returns them. A loss on `model.selected_logprob(hidden,
targets)` back-propagates to every adapter. Full-prefill (packed) batches only.

### 7.1 LoRA arithmetic (`dsv4-lora-v1`)

```text
h = BF16(x @ A^T)                     DeepGEMM bf16_gemm_nt, FP32 accumulate, one round
d = h @ B^T                           DeepGEMM bf16_gemm_nt, FP32
y = BF16(FP32(y_base) + scale * d)    two correctly rounded FP32 steps, one BF16 round
```

The rank is zero-padded to a multiple of 64 (exact). The routed `fc1` delta depends only on
the token, so it is computed once per token and added to the FP32 `z` inside
`moe_swiglu_quant.cu` (byte-equal to the fused fc1 epilogue). Under EP the token's rank
computes `u1 = BF16(x A1^T)` and sends it with the MXFP8 row, so EP > 1 gives the EP = 1
bytes; the routed adapter is one global adapter whose gradients are summed over the EP
group. Cost against the same MoE block without adapters: forward 1.02-1.17x, forward +
backward 1.05-1.21x (E 256, rank 32).

### 7.2 Backward per module

| Module | Backward | Accuracy / cost |
|---|---|---|
| Logprob head | `dh = BF16(g * (P @ W))`, `P @ W` on DeepGEMM `bf16_gemm_nn` computed in the forward when grad is needed | deterministic, BI; fwd + bwd 0.84-1.04x of naive PyTorch |
| mHC | dX only, golden-defined order (frozen `fn` / scale / base / norm) | byte-equal to golden; train step 0.07-0.09x of naive PyTorch |
| Attention | cuDNN-frontend `dsa_bwd_sm90` for the sparse core (FlashMLA has no backward); FP8 dgrads through precomputed `W.T`; one fused kernel for the RoPE transposes and RMSNorm backwards; one kernel for the main compressor's pooling / norm / RoPE backward. The indexer gets no gradient (its selection is a constant) | dL/dhidden rel L2 7.3-7.8%, cosine 0.997 against an FP32 rebuild (the forward itself is 6.7-6.9% from FP32). One 4096-token request: 2.7 / 4.8 / 4.0 ms (window / C4 / C128) vs forward 1.7 / 3.1 / 2.4 ms. dKV not deterministic (cuDNN atomics) |
| Router | dX via a six-row FP32 gather over the gate, slot order | exact, BI; 0.48-1.03x of the dense dgrad |
| Routed experts | fc1 recompute, `dh` (BF16 WGMMA, MXFP4 expanded exactly) fused with the clamped-SwiGLU backward, `dx` in FP32 (`sm90_moe_mlp_backward.cu`) | BI, run-to-run identical; 0.44-0.74x of a per-expert cuBLAS loop |
| Shared expert | FP8 DeepGEMM dgrad (default) or det_gemm BF16 (`backward="bf16_det_gemm"`, reference) | BI; FP8 dX 3.8% rel from FP32 (operand rounding) |
| MoE glue | `dy` to every row of its token (exact), dX fan-in by the slot-sum kernel | exact order |

FP8 gradients carry the E4M3 rounding of their operands (3.8% relative L2 on Gaussian
data, against 0.17% for BF16). Whether that is acceptable has to be judged on training
curves.

### 7.3 Training checks

`tests/models/deepseek_v4/test_dsv4_model_train.py`: every adapter gets a finite, nonzero
gradient; on an all-hash-routed copy, predicted and observed loss changes agree over 12
parameter groups (correlation 0.93). Each component is checked exactly against FP32
autograd in its own tests.

## 8. Validation

### 8.1 Per operator

Every operator test checks the CUDA kernel byte-equal to its PyTorch golden and
batch-invariant (prefixes, single rows, windows, permutations, run to run). Kernels taken
from vLLM are also checked byte-equal to vLLM's compiled kernels (parity scripts in 2.6).

### 8.2 Per layer and model

- Attention layer (C4, C128, window-only): alone = in a batch = chunked prefill = decode,
  bytewise; a permuted top-k gives the same bytes; a decode step captured in a CUDA graph
  replays the eager bytes.
- MoE block: batch-invariant across splits, run to run, within 3% of an FP32 reference on
  the same MX operands; EP = 2 bitwise equal to EP = 1 (`scripts/check_dsv4_moe_ep.py`;
  with LoRA the dX path matches to rounding, section 1).
- Model: batch = alone = chunked = decode for hidden states and logprobs; a whole decode
  step captures in a CUDA graph; against DeepSeek's inference code on a toy checkpoint
  there is no wiring error (layer-0 attention 4.2%, MoE 2.5% rel L2, from the FP8 choices).

### 8.3 Layerwise gate

`rl_engine/models/deepseek_v4/layerwise_gate.py` runs the same token sequences three ways:

- **rollout**: continuous batching. Requests arrive staggered, prompts are prefilled in
  chunks under a token budget, and decode tokens share steps with prefill chunks.
- **actor**: one packed prefill of every sequence, requests in reverse order.
- **train**: the same packed prefill through `forward_train`.

It compares every layer's output streams per token and reports the first layer that
differs. H200, random checkpoint at the real dimensions, nonzero LoRA, window / C4 / C128
with hash and scored routing, 2108 tokens: 0 differing tokens at every layer, final
hidden and logprob identical, for 4 layers (with training) and 8 layers. Negative control:
a one-element change in layer 2 is reported at layer 2.

## 9. Performance

H200, ratio = ours / non-invariant baseline, range over the token counts on each page
(typically T = 1 to 16384). Lower is better; the gate is 1.3x.

| Area | Operator | Baseline (not BI) | Ratio |
|---|---|---|---|
| GEMM | `Fp8Linear` (dense projections) | cuBLAS BF16 | 0.43-1.09x |
| GEMM | `bf16_gemm_nt` (compressor / indexer score projections) | cuBLAS | 0.90-1.28x |
| GEMM | `fp8_group_quant` | DeepGEMM helper + layout conversion | ~0.08x |
| Attention | Indexer stable top-k | `torch.topk` | 0.31-0.96x |
| Attention | Indexer query RoPE + FP8 | vLLM CuTe kernel | 1.00-1.14x |
| Attention | Indexer compressor | vLLM Triton | 1.05-1.14x |
| Attention | Main compressor C4 / C128 | vLLM Triton | 0.98-1.07x / 0.38-0.41x |
| Attention | FlashMLA sparse forward (pinned) | eager gather + softmax | 0.04-0.11x |
| Attention | q-norm + RoPE + KV insert | vLLM fused kernel | 0.80-1.14x |
| Attention | Output projection | vLLM inverse RoPE + `fp8_einsum` | 0.77-1.06x |
| Attention | Layer, C4 decode 64 x 4K (graph) | earlier glue: 5.2 ms | 0.47 ms |
| MoE | Router (gate + top-6 / hash) | cuBLAS + vLLM Triton top-k | 1.03-1.22x |
| MoE | Router dX | scatter + `torch.mm` | 0.48-1.03x |
| MoE | Route plan (counting sort) | PyTorch sort + searchsorted | 0.12-0.48x |
| MoE | Combine (slot sum) / dX fan-in | `index_add_` | 0.06-0.41x / 0.16-0.91x |
| MoE | Routed forward / backward (dX) | per-expert cuBLAS BF16 loop | 0.39-1.02x / 0.44-0.74x |
| MoE | Shared expert (FP8) | eager BF16 | 0.71-0.78x |
| mHC | `mhc_pre` + fused norm / `mhc_post` | vLLM TileLang kernels | 1.09-1.18x / 0.92-1.02x |
| mHC | Train step (pre + post, fwd + bwd) | naive PyTorch autograd | 0.07-0.09x |
| Head | Selected-token logprob fwd / fwd + bwd | cuBLAS BF16 logits + FP32 `log_softmax` / autograd | 0.79-1.05x / 0.84-1.04x |

Closest to the gate: the router (DeepGEMM gate against cuBLAS) and `mhc_pre` (TF32 GEMM
with a fixed split count). Not measured yet: the whole attention layer and the decoder
layer against vLLM's modules on the same shapes (needs vLLM's forward context).

Whole-layer numbers (H200): decode 64 requests x 4K context 0.47 / 0.52 / 0.22 ms (C4 /
C128 / window-only, graph replay); prefill one 4096-token request 3.0 / 2.2 / 1.8 ms.

## 10. The vLLM rollout (verl integration)

`rl_engine/integrations/dsv4/vllm_model.py` registers `RlkDeepseekV4ForCausalLM` under
vLLM's `DeepseekV4ForCausalLM` name (the `RL_KERNEL_DSV4_ROLLOUT=1` general plugin; no
vLLM source changes). It keeps vLLM's scheduler, paged caches, FlashMLA layer (gathers,
indices, sink), indexer logits and top-k, `save_partial_states` and the FP8 input
projections, and runs this kernel set wherever the fork rounds differently. Each item
below broke the gate (`scripts/check_dsv4_vllm_gate.py`) until replaced:

| vLLM fork (BI mode, Hopper) | Replaced by |
|---|---|
| TileLang mHC (split count depends on T), pre-summed first-layer broadcast | `mhc_pre` / `mhc_post` / `hc_head`, four identical first-layer streams |
| `DeepseekV4MoE` (router, grouped experts, shared expert) | `Dsv4MoE`, loaded from the checkpoint; vLLM allocates no experts |
| YaRN RoPE on window-only layers | the plain table (`rope.py`) |
| `fused_q_kv_rmsnorm` (a rare row rounds differently and stays in the KV cache a window long) | `rmsnorm_cuda` |
| `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (about 1 query row in 10) | `q_norm_rope_kv_insert` into vLLM's `fp8_ds_mla` cache |
| FlashMLA paged FP8 decode kernel | `VLLM_DS4_DECODE_KERNEL=sparse` (decode on the prefill kernel) |
| vLLM's FlashMLA build (same commit `a8f794d`, other flags: about 6 % of rows) | our pinned build serves vLLM's interface (both register `torch.ops._flashmla_C`, so ours loads first) |
| `fused_indexer_q_rope_quant` | `indexer_q_rope_fp8_cuda` |
| compress / norm / RoPE / quant / store (cutedsl main KV, Triton indexer K; a rare entry) | `compress_main_cuda` / `compress_indexer_cuda` |
| BI-mode `o_proj` (Triton block-scaled GEMM per group instead of `fp8_einsum`) | `OutputProjection`, `wo_a` / `wo_b` loaded from the checkpoint |
| BF16 logits, Triton / torch log-softmax | FP32 head; the sampled token's log-prob from the 64-tile LSE (`processed_logprobs`, V1 and V2 runners) |
| no LoRA | unmerged adapters on `wq_a` / `wkv` / `wq_b` / `wo_b` and in `Dsv4MoE` |

Also: the attention runs on one stream (vLLM overlaps the compressor / indexer on aux
streams); `VLLM_BATCH_INVARIANT=1` starts on the pinned DeepGEMM (vLLM's
`set_batch_invariant` check guards MoE kernels this model does not run). Supported: TP 1,
PP 1, no sequence parallel, MTP or EPLB; DP attention with EP MoE.

Gate, toy checkpoint (4 layers: window, window, C4, C128; 256 experts), 6 requests under a
256-token budget (chunked prefill, decode next to prefill), T = 0.8: every layer, sub-layer
and sampled-token log-prob byte-identical to one packed `Dsv4Model` prefill, without and
with rank-16 LoRA. Real checkpoint (DeepSeek-V4-Flash, first 4 layers, 2 x H100, 2026-11-17,
`run.sh gate` of `scripts/verl_e2e_lora`): 0 of 54 sampled-token log-probs and 0 tokens per
layer differ, without and with rank-16 LoRA; the layerwise gate 0 of 2108 tokens; the verl
engine check passes. Not yet: EP > 1, CUDA graphs.

Ablation (`RL_KERNEL_DSV4_NON_BI=1` with `VLLM_BATCH_INVARIANT=0`; the launcher's `BI=0`):
the rollout without batch invariance, to price it.
- vLLM's own paths come back: non-invariant attention planning, FlashMLA's paged FP8 decode,
  the fused q / kv RMSNorm, query norm / RoPE / KV insert, indexer query and compressor store
  kernels, and the aux-stream overlap.
- RL-Kernel's mHC, MoE, output projection, LoRA, RoPE tables and FP32 head stay, so the
  model is the one the actor trains. There is no actor-side counterpart.

On the toy gate with LoRA (`--non-bi`), 302 of 2102 tokens differ after layer 0 (the decode
tokens and some prefill chunks); the rest are byte-identical, so it is the same model. The
differences then grow through the random-weight layers (sampled-token log-probs: max 0.55).
The toy GRPO on 1 x H200 (8 prompts x 4 x 128 tokens, eager) took `timing_s/gen` 2.39 s
against 3.04 s with BI (0.79x) and 0.85x per step. Every response token mismatched (mean
|diff| 0.14 on random weights).

### 10.1 verl

verl and vLLM run unmodified; RL-Kernel plugs in through their extension points
(`scripts/verl_e2e_lora/run_dsv4_rlkernel_grpo.sh` sets all of them):

| Piece | Where | Selected by |
|---|---|---|
| Actor engine `rlkernel` (`Dsv4Model`, LoRA only, data parallel) | `integrations/dsv4/verl_engine.py` | `model.external_lib`, `model_engine=rlkernel`, `hydra.searchpath=[pkg://rl_engine.integrations.dsv4.verl_config]` |
| Rollout model | `integrations/dsv4/vllm_model.py` | RL-Kernel's `vllm.general_plugins` entry point with `RL_KERNEL_DSV4_ROLLOUT=1` |
| Adapter sync, level-1 sleep | `integrations/dsv4/vllm_worker.py` | `rollout.engine_kwargs.vllm.worker_cls` / `worker_extension_cls` |

- The engine computes next-token log-probs for packed full-prefill micro-batches with the
  rollout's kernels; `scripts/check_dsv4_verl_engine.py` (2 ranks) checks them byte for
  byte across micro-batch layouts, against each sequence alone, with per-sample
  temperatures, before and after training steps.
- The LoRA rank is the engine's (`actor.rlkernel.lora_rank`), and verl's `model.lora_rank`
  stays 0: verl would otherwise turn on vLLM's own LoRA in the rollout. So verl sees no
  LoRA: the reference policy (only with a KL term) is a separate `rlkernel` engine without
  adapters, and its level-2 sleep would discard the rollout's base weights, which are never
  re-sent; `RlkWorker.sleep` keeps level 1.
- Only the BF16 adapters are synced (`get_per_tensor_param`); the rollout loads the base
  weights from the checkpoint (`load_format=auto`; `dummy` is refused). verl's trainer-side
  rollout quantizer (offered for the FP8 / FP4 checkpoint) is accepted and unused.
- Process start: vLLM pickles the engine config into the processes it spawns, and
  unpickling the DeepSeek-V4 quantization config imports vLLM's DeepSeek-V4 package, which
  loads vLLM's FlashMLA build, before any plugin runs. The plugin registers a subclass of
  that config defined in `vllm_model.py`, so those processes import the plugin first.
- The verdict is verl's own `training/rollout_logprobs_mismatch_count` (FP32 `!=` per
  response token; the `probs_diff` metrics miss 1-ulp differences).
- Activation recompute (`actor.rlkernel.activation_recompute`, launcher `RECOMPUTE=1`;
  off by default): `Dsv4Model.forward_train(recompute=True)` keeps one BF16 `[T, 4, H]`
  input per layer and reruns the layer's forward in the backward. The forward is
  deterministic and batch-invariant, so the rerun has the first forward's bytes and the
  gradients are those without recompute (to the attention backward's run-to-run noise).
  Real checkpoint, 4096 tokens, 1 x H100: held for the backward 1.53 GiB per layer without,
  0.18 GiB with (the input streams and the layer's caches); peak over the weights 9.15 to
  4.46 GiB for 4 layers. About 66 vs 8 GiB for 43 layers. verl's
  `model.enable_gradient_checkpointing` is ignored. The first training forward also
  allocates the persisted W8 experts (section 12), about 6 GiB per layer.

Toy GRPO, 2 x H200, toy checkpoint, rank-16 LoRA on every target, 8 prompts x 4 samples x
128 tokens, synthetic reward (the adapters move every step): `rollout_logprobs_mismatch_count`
0 at each of 4 steps. Negative control (`RL_KERNEL_DSV4_SKIP_ADAPTER_SYNC=1`: the rollout
keeps its adapters): 0 at step 1 (zero-B adapters equal the base model), 4096 of 4096 tokens
from step 2. The vLLM plugin is installed in the dev env as an entry-point-only package
(`~/dsv4-work/rlk-plugin`); `pip install` of RL-Kernel provides the same entry point.

### 10.2 Megatron as the actor (investigated 2026-11-20, not planned)

Read from the verl #7483 environment (verl `8c047f4d`, Megatron-Core 0.19.0+fd1121b8f,
Megatron-Bridge 0.6.0+c7774d44, TE 2.20.2); nothing was run. Question: could verl's
Megatron engine be the actor with our operators swapped in?

It can without forking verl or Megatron, but only at **decoder-layer granularity**. Many
of MCore's numerics live in module code, not kernels, and every native MCore operator
left in place gives bytes the rollout does not have:

| Piece | MCore 0.19 | Ours |
|---|---|---|
| mHC RMS | `1/(sqrt(mean x^2) + eps)` (`hyper_connection.py:94`) | `rsqrt(mean + eps)` |
| mHC mixing | weights cast to BF16 before mixing (`:356-361`); fused split-K autotuned per shape | FP32, fixed 16-way split |
| Attention | separate grad / no-grad paths, different index order (`csa.py:1871`); compressor sums in BF16; `torch.topk` | one path, vLLM order |
| Router | BF16 gate via TE GEMM, `s / (sum + 1e-20) * 1.5`, `torch.topk` ties unspecified | DeepGEMM gate, `w * (c / S)`, ties to smaller id |
| Routed experts | BF16, TE GroupedLinear (one cuBLAS GEMM per expert, algorithm per M) | MXFP8 x MXFP4, SM90 kernel |
| LoRA (Bridge PEFT) | `d = BF16(h B^T)`, BF16 scale, BF16 add; adapters BF16 | FP32 `d` and add |
| Weights | Bridge dequantizes everything to BF16 at import | FP8 128x128 blocks, MXFP4 |
| Log-probs | verl's Triton fused linear-CE, `logits * (1 / T)` | `selected_logprob` |

The MoE combine (TE `_unpermute_kernel`, FP32 sum in expert-index order) is batch-invariant;
no batch-invariant GEMM mode exists in MCore. TP must be 1 for DSv4 attention (asserted).

What the swap would take:

- A `DeepSeekV4Bridge` subclass loaded through `model.external_lib` (the bridge registry is
  last-writer-wins) with a custom `transformer_layer_spec` wrapping `Dsv4Layer` under
  MCore's parameter names; it keeps the FP8 / MXFP4 tensors and scales instead of
  dequantizing. Our layers already have autograd Functions (`_MoEBlockFn`,
  `Dsv4AttentionFunction`, `_RecomputedLayer`).
- `selected_logprob` through `GPTModel`'s `output_processor` (or the unfused path, patched).
- A PEFT LoRA class with the `dsv4-lora-v1` arithmetic, and a sync that sends adapters to
  our rollout plugin: with `model.lora.rank > 0` verl turns on vLLM's own LoRA.
- `moe_router_bias_update_rate = 0` (MCore otherwise moves the expert bias every step),
  router replay off, and the `[s, b, h]` / THD packing (fused kernels pad to at least 128
  rows) converted to our packed layout.

What remains of Megatron is the shell: `GPTModel`, embedding, DDP, optimizer and offload,
distributed checkpoints, recompute scheduling. Decision (owner, 2026-11-20): not now; the
`rlkernel` engine already runs this forward. Revisit for PP / optimizer offload on larger
models, or for a same-framework comparison (MCore with our layers vs native MCore).

The baseline's LoRA is not our configuration, which matters when comparing reward curves:
it also targets the compressor, indexer and router; `experts_shared_outer_loras` gives fc1
`B` and fc2 `A` per expert; adapters are BF16 (`lora.dtype=float32` does not reach the
parallel linears); `lora_plus_ratio` is dropped. Its expert bias drifts, but R3 replays
the rollout's routes, so the drift does not select experts.

### 10.3 Rollout decode profile (2026-11-21) and what to do about it

The rollout server alone under nsys (node-level graph tracing), E3's settings (BI, DP 8 /
EP 8, graphs, rank-16 adapters reserved), A3's (BI off) and BL2's (vLLM 0.32, TP 8 / EP 8,
no LoRA adapter, so slightly optimistic for the baseline); full model, 8 x H200. Each
request decodes exactly 256 tokens; ms per decode step:

| Concurrent requests (per DP rank) | BL2 rollout | ours, BI (E3) | ours, BI, NCCL fixed | ours, BI off (A3) |
|---|---|---|---|---|
| 1 (1) | 9.6 | 33.7 | 32.1 | 28.6 |
| 8 (1) | 11.8 | 43.9 | 36.4 | 32.5 |
| 64 (8) | 14.2 | 68.9 | 52.2 | 46.2 |
| 512 (64) | 53.0 | 190.1 | 121.5 | 106.9 |

**Found and fixed (2026-11-21):** `VLLM_BATCH_INVARIANT=1` makes every vLLM worker set one
NCCL channel, the Simple protocol and tree all-reduce before NCCL starts
(`override_envs_for_invariance`), so that NCCL's reductions sum in a fixed order. Our
rollout runs no NCCL reduction on model data (TP 1; the EP exchange only moves bytes), so
this only cut the exchange to about one SM's copy bandwidth: at 64 requests per rank
`SendRecv` took 303 us per call with BI and 38 us without, the same bytes. The plugin now
restores the NCCL variables after vLLM's override (`_keep_nccl_bandwidth`;
`RL_KERNEL_DSV4_NCCL_DETERMINISTIC=1` keeps them): 40 us per call, the 64-per-rank step
136 -> 68.5 ms; the vLLM gate at DP 8 with graphs and LoRA still 0 of 432. Batch invariance
now costs 12-14 % over BI off at 8-64 requests per rank, where the earlier E3 / A3 gap was
mostly this. The bullets below describe the traces before the fix.

- The graphs work: a decode step is one replay of one full graph (attention included),
  0.13 ms between replays, so no host or launch overhead.
- Inside it, at batch 1: 9177 kernels (about 210 per layer, mostly 1-3 us) in 32.3 ms, 27.1
  busy; vLLM 1888 kernels (44 per layer) in 9.2 ms. Our layer (715 us): the MoE's EP
  exchange about 240 us (router, the static dispatch's count / codes / scales / p_s
  all-to-alls, the adapters' `ep_send`, the return, and the index / searchsorted glue
  around them), the 8 unmerged adapters about 215 us (about 10 small kernels each), the
  rest (projections, compressor, indexer, attention, experts, shared expert, mHC) about
  260 us.
- At 8 requests per rank: NCCL `SendRecv` 17 -> 89 us per call on every rank alike (the
  one-channel cap above), 23 ms of the 64 ms step; our expert GEMM 107-114 us against
  vLLM's MXFP4 MoE (`humming`) 60 us for the same rows per rank.
- At 64 requests per rank after the fix (68.5 ms; vLLM's TP 8 step with all 512 tokens
  22.7 ms): expert GEMM 13.0 ms, `SendRecv` 10.4 ms, elementwise and index kernels about
  15 ms, the adapters' BF16 GEMMs 5.2 ms. The static exchange sends every rank pair a fixed
  tokens x top-k rows, about EP times the average rows, so it moves several times the bytes
  it needs; vLLM's two all-reduces per layer took 4.7 ms in all.
- Why vLLM needs no exchange: with TP attention every rank holds every token after `wo_b`'s
  all-reduce. Each rank routes all tokens (replicated gate), runs only its own experts
  (`expert_map`: local id or -1), and one all-reduce sums the partial outputs, the same
  all-reduce TP needs anyway (vLLM all-to-alls only with DP > 1:
  `FusedMoEParallelConfig.use_all2all_kernels`). Two one-shot all-reduces per layer, about
  9 us each. The cost: the unsharded attention parts and the router run on all tokens on
  every rank, and every rank caches every sequence's KV.
- That layout is not open to us as is: TP splits `wo_b`'s sum over ranks and the all-reduce
  sums the experts in rank order, both unlike the actor's order, so the rollout would no
  longer match the actor bitwise (and vLLM picks one- or two-shot all-reduce by size).

Directions after the NCCL fix, in order (none started):

1. **Fewer, packed exchanges on NCCL.** One exchange per direction instead of five: pack
   codes, scales, `p_s`, the adapters' rows and the counts into one buffer; fuse the index
   glue on both sides into one kernel each. Same rows in the same order, so bytes
   unchanged; no new dependency.
2. **Fused unmerged LoRA.** One kernel per adapter site (`x A^T` to BF16, `h B^T` in FP32,
   the scaled add and the cast), or both sites of an expert pair at once, keeping
   `dsv4-lora-v1`'s arithmetic. About 9 ms at batch 1.
3. **The expert GEMM at decode sizes** (P5 against vLLM's MXFP4 kernel, about 1.8x at 48
   rows per rank) and the remaining elementwise glue (fewer, larger kernels).
4. **DeepEP's low-latency mode as the transport** (section 12's open item;
   `zhuzilin/DeepEP@align_fp8_quantization`), if the exchange still matters after 1: two
   kernels per layer, only the valid rows, written once into the peer's buffer,
   graph-capturable. For the same bytes: its in-kernel FP8 cast (128-element scales) must
   give way to our MXFP8 codes and scales sent as payload; its combine sums the top-k
   outputs in-kernel, which must equal the actor's slot-order FP32 sum or become a plain
   return; its masked per-expert layout needs a P5 variant or a compaction; NVSHMEM has
   to build here. Estimated a few ms per step after 1 to 3.
5. **TP-style layout that keeps the actor's order.** All-gather the tokens after attention
   (every rank routes all tokens), each rank runs its experts, then all-gather the expert
   rows (not an all-reduce) and combine per token in the actor's slot order locally. More
   bytes than vLLM's all-reduce, no dispatch at all. Worth costing against 4.

The scripts: `scripts/verl_e2e_lora/profile_rollout.sh bi|nonbi|bl` (with
`prof_client.py`); `nsys export -t sqlite` and the graph-node kernels grouped by the launch's
correlation id give a replay's breakdown.

## 11. Legacy code not used by DSv4

The P1 unfused mHC (`rl_engine/mhc/`, `fp32_gemm_rms`, `rmsnorm_residual`,
`mhc_pre_h_aggregate`) and the P5 MoE provider backends and Triton lanes were removed
(2026-11-24); none was on the DSv4 path. Still in the tree:

| Code | What it is |
|---|---|
| `csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm*.cu`, `clamp_swiglu_weighted.cu`, `shared_expert_mlp.cu`, `fused_shared_expert_mlp.cu` | Earlier MoE kernels, superseded by `sm90_fused_moe_mlp.cu` and the DeepGEMM shared expert (`fused_shared_expert_mlp.cu` is still compiled into `_C`) |
| `rl_engine/moe/oracle.py`, `contract.py`, `mx_format.py`, `fixtures.py` | Still used: the FP32 MoE oracle, the MX format and the fixtures are references for the DSv4 MoE tests (`tests/ops/moe/`) |

## 12. Status and open work

Done on one GPU: every forward operator BI and within the gate; LoRA training end to end;
EP = 2 forward and dX bitwise equal to EP = 1; EP = 2 with LoRA adapters (forward bitwise,
dX and adapter gradients to rounding); graph-capturable decode at EP = 1; the layerwise
gate on one device.

Open, needs the multi-GPU machine:

- [x] EP = 8 with real collectives (engine, rollout plugin and verl; see below).
- [ ] DeepEP (`zhuzilin/DeepEP@align_fp8_quantization`) transport with the same bytes,
      fixed-capacity staging, route-metadata capture.
- [x] vLLM rollout plugin (section 10): bitwise equal to `Dsv4Model` on one GPU, with LoRA.
- [x] verl integration (user decision 2026-11-15: the actor is a verl engine backend on
      `Dsv4Model`, not Megatron; verl and Megatron stay unmodified): engine backend,
      adapter sync into the rollout, toy GRPO on 2 GPUs bitwise (section 10.1).
- [x] 8-GPU run against verl #7483 (Hopper, non-BI, LoRA) on the real checkpoint, all 43
      layers (2026-11-20 / 21): the guide is `scripts/verl_e2e_lora/README.md`, the runs
      `experiments.sh` (README section 8 has the numbers). rlkernel bitwise at every step
      (E2 100 steps at 32 x 8, E3 100 steps at 32 x 16 with responses up to 4096); the
      baseline runs (BL2, 50 steps). Open: our decode step is about 2.5x the baseline's
      (BI about 30 of the 50 ms); the profile and the directions are in section 10.3.
- [x] The rollout's BI-off switch for the ablation (section 10; rollout side only, the
      actor's kernels have no non-invariant variant). In the 8-GPU guide as `run.sh ablation`.
      Full model (A3, 30 steps of E3's workload): decode step 56 against E3's 84 ms, 156k to
      437k mismatched tokens per step (BL2: 452k to 768k).
- [x] `rlkernel` engine at EP > 1 (`expert_parallel_size`; groups of consecutive ranks;
      routed-adapter gradients divided by E before the data-parallel average): EP = 2, 4
      and 2 x 2 groups give the zero-B log-probs of EP = 1 byte for byte and its first
      grad norm to 1e-3 (real checkpoint, first 4 layers, H100, 2026-11-17). EP = 8 and
      2 x 4 groups the same (8 x H200, 2026-11-18).
- [x] Activation recompute in the engine (section 10.1), checked on the real checkpoint.
- [x] verl at EP > 1 (`ACTOR_EP`, `ROLLOUT_DP` in the launcher): actor EP 2 / rollout DP 2
      and actor EP 4 / rollout DP 4 on H100, real checkpoint's first 4 layers, 3 GRPO steps:
      `rollout_logprobs_mismatch_count` and `actor/old_logprobs_mismatch_count` 0 at every
      step (2026-11-17). Actor EP 8 / rollout DP 8 on 8 x H200: the same, 3 steps
      (2026-11-19), with the first 4, 8 and 16 layers and the full 43 (recompute on;
      43 layers with `RL_KERNEL_DSV4_PERSIST_W8=0`, peak 95 GB per GPU, 65 s per step on
      the 8 x 4 workload, 61 s of it the eager rollout).
- [x] verl GRPO on the full model, 100 steps (8 x H200, EP 8 / DP 8, recompute,
      persist_w8 off, 2026-11-20): DAPO-Math-17k with verl's DAPO math reward, 32 prompts
      x 8 responses, prompt 1024 + response 2048 tokens, LoRA rank 16. Both mismatch
      counts 0 at all 100 steps; about 420 s per step; checkpoints every 25 steps; score
      mean 0.27 over steps 1-10 to 0.44 over 91-100. The run continues to 200 steps.
- [x] verl on 8 GPUs of one node: it peaks at about 2730 threads of the user (64 CPUs),
      over a per-user limit of 2048 (`ulimit -u`). On cluster u4478697 that limit reaches
      Slurm job steps from the login node; a shell reached by ssh into the allocated node
      has the node's own limit, and the run goes there.
- [ ] Parameter offload. Persisted W8 expert weights cost
      E x N x K bytes per weight in both the actor and the rollout, about 0.8 GB per layer
      per GPU at EP 8; `RL_KERNEL_DSV4_PERSIST_W8=0` expands them per call instead.
- [x] Rollout plugin at EP > 1 (vLLM DP attention + EP MoE, `--dp N` in
      `scripts/check_dsv4_vllm_gate.py`) on the real checkpoint's first 4 layers: DP 2 and
      DP 4 bitwise equal to `Dsv4Model`, with and without LoRA (2026-11-17); DP 8 the same
      (8 x H200, 2026-11-19).
- [x] Rollout plugin with CUDA graphs (2026-11-20): the fork's breakable graphs; a decode step
      replays one full graph (nsys, 2026-11-21). The MoE uses a static EP exchange for forwards up
      to vLLM's largest capture size (fixed tokens x top-k rows per rank pair, counts on the device,
      same row order, no host sync); the adapters are reserved at load (`RL_KERNEL_DSV4_LORA`) and
      each sync copies into them; launcher `GRAPH=1`. Bytes unchanged: `check_dsv4_moe_ep.py
      --graph` (EP 8, with LoRA), the vLLM gate `--graph` at DP 1 and DP 8 (with LoRA and
      `--lora-sync`, 0 of 2680 log-probs differ), verl GRPO on the full model (mismatch 0). Speed,
      full model, 8 x H200: `timing_s/gen` 61 -> 17-21 s on the 8 x 4 / 256-token workload, 404 ->
      165 s on 32 x 8 / 2048 (about 204 -> 80 ms per decode step); the step 430 -> 194 s.
- Dropped (owner, 2026-11-21): the verl #7688 guide (verl + vLLM batch-invariant,
  full-parameter, `scripts/verl_e2e/`, never run). The comparison is against verl #7483
  (`scripts/verl_e2e_lora/`); the guide stays in git history (`33b65bb4`).
- [ ] Layerwise gate on the real checkpoint
      (`scripts/check_dsv4_layerwise_gate.py --ckpt DIR --layers N`, advancing N). First 4
      layers: 0 differing tokens, with the vLLM gate and the verl engine check (section 10).
- [ ] #230 cross-config M0-M10 selected-token logprob bitwise; a logprob threshold in CI
      (target 0).
- [ ] vLLM golden (development only): `ISEEKYAN/vllm:ds4-v9-rc1` with
      `VLLM_BATCH_INVARIANT=1`, logprobs for a fixed prompt set (batch 1/8/64, chunked
      prefill, decode) under `tests/fixtures/dsv4/`.
- [ ] Whole attention layer and decoder layer against vLLM's modules (performance).
- [ ] Rollout decode latency (section 10.3): packed EP exchanges with fused index glue,
      fused unmerged LoRA, the expert GEMM at decode sizes, then DeepEP as the transport.

Open, single GPU, deprioritized:

- [ ] Per-expert LoRA (section 4.2).
- [ ] Deterministic dKV (fixed-point accumulation), only if backward determinism becomes a
      requirement.
- [ ] The det_gemm BF16 shared-expert reference needs a `KERNEL_ALIGN_DET_GEMM_SM90=1`
      build to be fast.

Out of scope for v1: DCP / CP attention, a bitwise sparse attention backward, ROCm,
writing the vLLM fork into the formal anchor.

Known failures that predate this work: `tests/runtime/test_attention_dispatch.py` (2, from
a main merge). The GitHub `WS1-gtest-GPU` workflow fails on main too (RunPod
orchestration), independent of this PR.

## 13. Numerics lessons

Each of these broke a byte-equality test at some point.

- "Already equal to vLLM" from an op probe on a few inputs is not enough: vLLM's fused
  q / kv RMSNorm, q-norm / RoPE / KV insert, compressor store and its FlashMLA build each
  matched on some data and differed on other data. Only the end-to-end gate on mixed
  batches found them; a rare row in a cache entry spreads to every later query.
- vLLM loads weights under the model dtype as the default dtype: `torch.randn` without
  `dtype=` then draws BF16 directly.

- CUDA PyTorch `tensor / python_scalar` multiplies by the reciprocal (not correctly
  rounded); goldens use `tensor / tensor`. CPU is exact, so CPU-only tests miss it.
- On CUDA, `torch.exp` is libdevice `expf`, the same as vLLM's kernels; CPU `exp` differs
  in the last bit, so rollout-parity goldens are defined on CUDA.
- In CUDA kernels, spell FP32 steps with `__fadd_rn` / `__fmul_rn` / `div.rn` to stop FMA
  contraction; avoid `.ftz`; refuse `--use_fast_math`.
- vLLM's FP32-scale UE8M0 uses `exp2f(ceilf(log2f(s)))`. It equals the bit-math ceiling
  for every BF16 / FP16 input but not for FP32, so the quant ops reject FP32 input.
- DeepGEMM's `per_token_cast_to_fp8` clamps amax to 1e-4 (vLLM uses eps 1e-10): it is not
  a golden.
- DeepGEMM SM90 wgrad: FP32 output with accumulate only, token count % 16 == 0 (pad with
  zeros). Its psum grouped layout starts every group at the previous end aligned up to
  the block M.
- DeepGEMM's masked grouped FP8 GEMM intermittently returned wrong values on the `wo_a`
  operands; it is not used.
- det_gemm with M = 0 used to leave a pending CUDA error that a later, unrelated call
  reported (fixed: empty shapes launch nothing).
- `ptxas` serializes every WGMMA issued inside index-dependent branches (warning C7518);
  keep the consumer loop straight-line.
- Timing: use CUDA-graph replay for small shapes; eager timing measures host overhead.
- FP32 `torch.mm` is not row-invariant: cuBLAS picks its kernel from M, so rows of
  `[M, 1024] @ [1024, 16]` at M = 1, 2, 12 differ from the same rows at M = 97 (H200).
  It looks invariant on one GPU at many sizes and then breaks once EP changes each rank's
  row count (`check_dsv4_moe_ep.py --lora`, routed `fc2` adapter dX).

## 14. References

- RL-Kernel #635 (this PR), #230 (cross-config), #241 (TP logprob)
- vLLM #54955 (DSv4 batch-invariant preview), `ISEEKYAN/vllm@1a6542c`
- DeepGEMM `vllm-project/DeepGEMM@e21c821f`; FlashMLA `vllm-project/FlashMLA@a8f794d1`
- cuDNN frontend 1.31.0 (`dsa_bwd_sm90`)
- verl (integration target); slime #2262 (Megatron / SGLang DSA alignment methodology)
- `zhuzilin/DeepEP@align_fp8_quantization`
