# DeepSeek-V4.1-Flash on SM90: Design Delta from DSv4

A proposal for extending the DSv4 batch-invariant kernel set to DeepSeek-V4.1-Flash
(DSv4.1). It is written as a delta against `docs/architecture/dsv4-design.md` ("the base
doc"); anything not mentioned here is meant to carry over unchanged. Section numbers
written as "base 5.1" refer to that document.

- Status: **proposal, nothing implemented or measured.** Every "reuse" claim below is
  from reading code, not from a passing gate.
- Base: RL-Align/RL-Kernel #635 (`test-dsv4-cuda` at `6d974b9`)
- Branch: `test-dsv41-cuda`, a new branch created from `test-dsv4-cuda`
- Hardware: NVIDIA H100 / H200 (SM90). ROCm and SM100 stay out of scope.
- Model reference: vLLM main `3f86db3`, `vllm/models/deepseek_v41/` (the Hopper path,
  **not** batch-invariant); arXiv 2609.19969 for the architecture
- BI reference: none on Hopper. vLLM #62365 is open, validated on GB200 only
- Last updated: 2026-11-25

Contents:

1. Summary
2. What changes in the model
3. Decisions
4. Module delta
5. Exact sliding-window attention (D8)
6. Execution profile additions
7. Numeric profiles
8. Training
9. Validation
10. The vLLM rollout
11. Memory and performance expectations
12. Work plan
13. To confirm
14. References

## 1. Summary

The guarantees in base 1 are kept as written, including "a token's bytes do not depend
on whether its request runs as one prefill, chunked prefill or decode". DSv4.1 is the
same family of operators with a different wiring, so most kernels carry over.

| | Modules |
|---|---|
| **Reused as is** | Sparse attention core (FlashMLA), `fp8_ds_mla` KV record and insert, sparse attention backward, indexer query, MQA logits, stable top-k, router, route plan, routed experts, MXFP8 activation quant, combine, EP exchange, LoRA arithmetic, logprob head, RoPE tables, sampling gate, verl engine and plugin scaffolding |
| **Small change** | Config and loader; q side (per-head q-norm off); compressor (ratios 1 and 2); mHC (single-pass, H = 5120) |
| **New** | MXFP8 32x32 dense linear path; cross-layer KV / index sharing; indexer K from the compressor latent; two-level candidate selection; Engram; exact layer-staged SWA |
| **Not supported in v1** | DSpark, vision, the approximate SWA bounded replay, FP4 main KV and indexer K, DCP / CP attention |

The one fact that makes this tractable: on every architecture except SM100 (and vLLM's
Ampere kernels), vLLM keeps the **V4 KV record** for DSv4.1, 448 FP8 NoPE + 64 BF16
RoPE + 7 UE8M0 scales, 584 bytes (`attention.py`, `_use_v41_mxfp8_kv_record`). The V4.1
MXFP8 record and the NVFP4 compressed cache exist only in the SM100 kernels. So the
FlashMLA sparse forward, the insert kernel and the cuDNN backward are unchanged on
Hopper.

## 2. What changes in the model

### 2.1 Dimensions

| | DSv4-Flash | DSv4.1-Flash |
|---|---|---|
| Layers | 43 | 40 |
| `hidden_size` | 4096 | 5120 |
| Attention heads x head dim | 64 x 512 | 64 x 512 |
| `q_lora_rank` | 1024 | 1280 |
| Indexer heads x head dim | 64 x 128 | 32 x 128 |
| `index_topk` | 512 | 512 |
| Routed experts, top-k | 256, 6 | 384, 6 |
| `moe_intermediate_size` | 2048 | 2304 |
| Hash-routed layers | 3 | none described |
| `compress_ratios` values | 0, 4, 128 | 0, 1, 2 |
| Dense weight format | FP8, 128x128 block scales | MXFP8, 32x32 UE8M0 block scales |
| Routed expert format | MXFP4 | MXFP4 |
| mHC streams | 4 | 4 |
| Extra modules | MTP | Engram (layers 1, 14), DSpark, ViT |

All values match the release `config.json` (`deepseek-ai/DeepSeek-V4.1-Flash` at
`314113f8`, kept as `rl_engine/models/deepseek_v41/release_config.json`, read by
`Dsv41Config`): `vocab_size` 129280, `o_lora_rank` 1024, `o_groups` 8,
`routed_scaling_factor` 1.5, DSv4's YaRN values, no hash layers (vLLM's DSv4.1 MoE passes
`num_hash_layers=0`). Two values differ from DSv4:

- **`rms_norm_eps` is 1e-20** (DeepSeek's `norm_eps`), used by every RMSNorm, the mHC
  controller's rms and the Engram gate. `hc_eps` stays 1e-6. The DSv4.1 kernels take eps
  as a parameter; the golden defaults are 1e-20.
- `compress_ratios` has 43 entries: the 40 layers and 3 trailing MTP / DSpark zeros.

The release also ships DeepSeek's reference code (`inference/model.py`, `engram.py`,
`kernel.py`), which confirms vLLM's wiring: the delayed pre (`Block.forward`), the
identity pre at the entry, the final `hc_pre(h, pre_mix)` then `norm` (no `hc_head`), and
the Engram gate's operation order, which `dsv41-engram-v1` follows.

### 2.2 Forward

```text
streams = embed[input_ids] copied to 4 streams                          BF16 [T, 4, H]
pre     = identity (selects stream 0)                                   first sublayer only
per layer i:
    if i in engram_layer_ids:
        streams = engram_i(streams, ngram_hashes)                       gated lookup, 4.5
    x, post, comb, pre = mhc_pre(streams, pre, hc_attn_*, attn_norm)    4.3
    streams            = mhc_post(attention_i(x), streams, post, comb)
    x, post, comb, pre = mhc_pre(streams, pre, hc_ffn_*, ffn_norm)
    streams            = mhc_post(moe_i(x), streams, post, comb)
hidden = norm(collapse(streams, pre))       no learned hc_head            BF16 [T, H]
logp   = selected_logprob(hidden, head, targets)                        FP32 [T]
```

`mhc_pre` collapses the streams with the `pre` it is given and returns the `pre` its own
controller computed, for the next sublayer.

### 2.3 Layer map

vLLM does not implement an encoder and a decoder as separate modules. It is one 40-layer
stack, and the "causal encoder-decoder" falls out of three config fields:

- `compress_ratios[i]`: 0 = sliding window only, 1 = full-length compressed cache,
  2 = ratio-2 compressed cache.
- `kv_source_layer_ids`: layers that own a compressor and a compressed KV cache.
- `index_source_layer_ids`: layers that own an indexer and publish top-k indices.

A consumer uses the nearest source at or below itself. The paper's three modes map to:

| Mode | KV source | Index source |
|---|---|---|
| Full | yes | yes |
| Reindex | no | yes |
| Reuse | no | no |

Zero-indexed layer map, derived from the paper's one-indexed description:

| Layers | Ratio | Role |
|---|---|---|
| 0-1 | 0 | Sliding window only |
| 2, 8, 14 | 2 | Full |
| 3-7, 9-13, 15-19 | 2 | Reuse |
| 20 | 1 | Full, and the candidate source |
| 21-23 | 1 | Reuse |
| 24, 28, 32, 36 | 1 | Reindex |
| 25-27, 29-31, 33-35, 37-39 | 1 | Reuse |

So `kv_source_layer_ids = [2, 8, 14, 20]` and `index_source_layer_ids = [2, 8, 14, 20,
24, 28, 32, 36]`. Layer 20's compressor reads layer 20's input, which is the output of
layers 0-19 (the "encoder"); layers 20-39 (the "decoder") all read that one KV. This
map matches the release `config.json`. "Layer
20's input" means its normed attention input, the `x` of that layer's first `mhc_pre`.

Every layer, whatever its mode, still has its own `wq_a` / `wkv` / `wq_b` / `wo_a` /
`wo_b` and its own sliding-window KV.

## 3. Decisions

Base D1-D7 and the product decisions in base 4.2 stand. Additions:

| | Decision | Status | Why |
|---|---|---|---|
| D8 | Sliding-window attention is **exact**. The approximate bounded replay is never used | owner, 2026-11-24 | Bounded replay makes a token's bytes depend on where the replay started, which breaks base 1. Section 5 |
| D9 | **No DSpark.** The loader drops the drafter tensors | owner leans no; confirm | It changes rollout speed only; the log-prob comes from the main model's pass. Supporting it means a second small model (three blocks with their own MoE, a Markov head, a confidence head) plus verification scheduling under BI |
| D10 | **Text only.** The loader drops the ViT; the router uses the text bias | proposed | RL targets are text; the image-token bias path has no user |
| D11 | **Keep the V4 `fp8_ds_mla` KV record** on SM90 | proposed | It is what vLLM's Hopper path does, and it keeps FlashMLA, the insert kernel and the backward unchanged. In vLLM the paper's FP4 main KV exists only in the SM100 kernels |
| D12 | **Dense MXFP8 weights run as exact BF16** on DeepGEMM `bf16_gemm_nt`, **with every input rounded to MXFP8** (`mxfp8_fake_quant`, per 32 values, UE8M0; byte-equal to DeepSeek's `act_quant`), so the GEMM computes the release's MXFP8 x MXFP8 product up to accumulation order | owner, 2026-11-25 (activation rounding) | No new GEMM kernel; the release's training arithmetic. dX is a BF16 dgrad with the rounding passed straight through |
| D13 | **Topology comes from the config** (`kv_source_layer_ids`, `index_source_layer_ids`, `candidate_*`), with vLLM's field names | proposed | One loader for the checkpoint and for toy checkpoints of any shape |
| D14 | **Engram tables are frozen, live in host memory, and are not synced** | proposed | They are 196B parameters; rollout and actor read the release rows |
| D15 | **The indexer K cache stays FP8 on SM90** (DSv4's format, `compress_indexer_fp8`), not the release's FP4 | owner, 2026-11-25 | Same reasoning as D11: the K store tail and the FP8 MQA logits kernel stay unchanged, and nothing on Hopper needs FP4 |

Consequence of D11 and D15 worth stating in the user-facing docs: on Hopper the policy
is trained and sampled with an FP8 compressed KV and an FP8 indexer K, not the FP4
formats the release was QAT'd for. Rollout and actor still agree bitwise; the policy
is simply not the SM100 one.

## 4. Module delta

| Module | DSv4.1 change | Action |
|---|---|---|
| Config, loader | New fields, new checkpoint layout | Change, 4.1 |
| Dense linear | MXFP8 32x32 instead of FP8 128x128 | **New path**, 4.2 |
| mHC | Single-pass; H = 5120 exceeds the kernel's 4096 | Change, 4.3 |
| q side | No per-head q-norm after `wq_b` | Flag, 4.4 |
| Compressor | Ratios 1 and 2, no overlap, no APE | Change, 4.4 |
| Indexer K | `k_norm(wk(latent))`, not an indexer compressor | **New (small)**, 4.4 |
| Indexer query, logits, top-k | 32 heads | Parameter |
| KV / index sharing | Full / Reindex / Reuse | **New**, 4.4 |
| Candidate selection | Block-level preselection at layer 20 | **New**, 4.4 |
| Sparse attention, KV insert | None on SM90 | Reuse |
| Engram | Two gated n-gram lookups | **New**, 4.5 |
| Router | 384 experts, no hash layers expected | Reuse, 4.6 |
| Routed experts, combine, EP | H 5120, F 2304, E 384 | Parameter, 4.6 |
| Shared expert | MXFP8 weights | Follows 4.2 |
| LoRA | Dimensions only | Reuse |
| Logprob head | K = 5120 | Reuse, re-probe |
| RoPE | Same rule as ours | Reuse, 4.4 |

### 4.1 Config and loader

`Dsv4Config.check_supported` rejects the DSv4.1 checkpoint today on three checks:
`hidden_size > 4096`, `weight_block_size != [128, 128]`, and `compress_ratios` (it
accepts 0, 1, 4, 128; `compress_main` itself accepts only 4 and 128). `n_routed_experts
= 384` already passes.

- New fields: `kv_source_layer_ids`, `index_source_layer_ids`,
  `candidate_source_layer_id`, `candidate_topk_blocks`, `candidate_block_size`,
  `engram_layer_ids`, `engram_num_embeddings`, `engram_max_ngram_size`,
  `engram_n_heads`, `engram_head_dim`, `engram_vocab_size`,
  `engram_compressed_vocab_size`, `engram_pad_token_id`.
- A separate `Dsv41Config` (or a `variant` field) so the V4 checks stay strict.
- Loader: MXFP8 linears (E4M3 codes + 32x32 UE8M0 scales), `indexer.wk` and the
  compressor's `wkv` / `wgate` in BF16 without scales, Engram tables as FP8 codes plus
  scales (never upcast), and DSpark / vision tensors dropped (D9, D10).
- A toy-checkpoint generator at the real dimensions with a short layer map that has one
  of each role: window, Full r2, Reuse r2, Full r1 (candidate source), Reuse r1,
  Reindex r1, and one Engram layer.

### 4.2 Dense MXFP8 linear

`Fp8Linear` is `fp8_group_quant` (per-128 activation groups) plus DeepGEMM `fp8_gemm_nt`
with 128x128 weight scales. DSv4.1's dense weights carry one scale per 32x32 block, so
that path cannot represent them. Affected: `wq_a`, `wkv`, `wq_b`, `wo_a`, `wo_b`, the
shared expert, the indexer's `wq_b`, and Engram's `wkv`.

| Option | What | For | Against |
|---|---|---|---|
| **A (proposed)** | Dequantize once at load to BF16 (exact: an E4M3 value times a power of two is representable) and run DeepGEMM `bf16_gemm_nt`, FP32 out | No new kernel; already probed BI at DSv4 shapes; dX becomes a BF16 dgrad | About 1 more byte per dense weight per GPU; a new numeric profile |
| B | Own SM90 MXFP8 x MXFP8 GEMM, derived from `sm90_fused_moe_mlp.cu` with `mxfp8_act_quant.cu` for the activations | Keeps weights at 1 byte; closest to the release's intended arithmetic | A new WGMMA kernel and its backward |
| C | A newer DeepGEMM with a K32 block-scaled GEMM on SM90, if one exists | Third-party kernel | Unknown on SM90; #62365 uses one but was validated on GB200 only |

By default vLLM itself runs these layers as BF16 GEMMs on weights dequantized at load on
H100 when TP > 1, so option A is not exotic. With A, `fp8_group_quant` drops out of the
DSv4.1 forward entirely, and base 7.2's "FP8 gradients carry 3.8% operand rounding"
caveat no longer applies to the dense layers.

Activation rounding (owner, 2026-11-25): every input of an MXFP8 weight is rounded to
MXFP8 before the BF16 GEMM, as DeepSeek's `linear` does: `wq_a` / `wkv`, `wq_b`, `wo_b`,
the indexer's `wq_b`, the shared expert's `w1` / `w3` / `w2`, Engram's `wkv`, and the
attention output before `wo_a` (the reference's in-place `act_quant`). The compressor,
`wk`, `weights_proj` and the router stay unrounded BF16 / FP32, as in the reference.
Rounding is exact arithmetic, so it is batch-invariant; it is byte-equal to the
reference kernel (`tests/models/deepseek_v41/test_dsv41_act_quant.py`), and on equal
MXFP8 operands our GEMM and the reference's `fp8_gemm` are equally far (BF16 rounding)
from FP64.

`wo_a` needs a decision inside A: the grouped einsum is `fp8_einsum` today. In BF16 it
becomes `o_groups` calls of `bf16_gemm_nt` (or DeepGEMM's grouped BF16 GEMM, which base
4.2 already probed for per-expert LoRA).

### 4.3 mHC single-pass (`dsv41-mhc-v1`)

The paper's update is `X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)`: the input mix comes
from the previous block. vLLM applies it per sublayer: attention collapses with the
pre-mix carried in from the previous layer's FFN, the FFN collapses with this layer's
attention pre-mix, and the first attention uses the identity (stream 0).

Checked against vLLM's reference (`mhc_pre_delayed_torch` in
`vllm/model_executor/kernels/mhc/torch.py` and `nvidia/model.py` at `3f86db3`):

- `mhc_pre` takes `pre_in` and returns `pre_out`, its controller's pre. The controller
  (GEMM, sigmoid, softmax, Sinkhorn) is DSv4's. The first sublayer's mix is **a copy of
  stream 0** (`residual[:, 0]`, as vLLM). DeepSeek's reference multiplies by `(1, 0, 0, 0)`
  and sums, which differs from the copy only in the sign of a zero.
- **There is no learned `hc_head`** ("v4.1 has no learned hc_head"): the model ends with
  `norm(collapse(streams, pre))`, the last FFN's pre applied as a plain mix.
- vLLM runs the first layer's controller on the embedding with `fn` summed over the four
  streams (all four streams are equal there). Same value, different rounding; we run the
  controller on the expanded residual, so one GEMM path serves every layer.
- Five chunks need no new reduction rule. The kernel does not add per-chunk partials:
  each of the 128 slices runs one `fmaf` chain over its runs in ascending chunk order,
  then the fixed slice tree. H = 5120 extends the chain; H <= 4096 never enters the
  fifth chunk, so `dsv4-mhc-v1` keeps its bytes. (An earlier draft proposed
  `((c0 + c1) + (c2 + c3)) + c4`; that is not how the kernel reduces.)
- `tf32_hc_prenorm_gemm` runs at K = 4H = 20480; re-probe the 16-split result.
- `POST_MULT` stays 2.0 (vLLM's `hc_post_alpha`).
- Implementation (`dsv41-mhc-v1`): the same `dsv4_mhc.cu` kernels, instantiated for 4 and
  5 chunks, with a block-uniform mix mode (own pre, carried pre, stream 0), plus a
  collapse kernel and its backward. Entry points `dsv41_mhc_pre`, `dsv41_mhc_collapse` and
  their backwards; golden `rl_engine/kernels/ops/pytorch/mhc/dsv41_mhc.py`. In the
  backward the controller's pre gradient is `d_pre_out` (from the next sublayer), and the
  op returns `d_pre_in`.

### 4.4 Attention

**q side.** vLLM calls its fused kernel with `apply_q_norm=False` ("qr is normed before
`wq_b`"). `dsv4_qo.cu`'s q-norm becomes optional; RoPE and the KV insert stay.

**RoPE.** vLLM's `build_deepseek_v4_rope` for DSv4.1 uses plain RoPE when
`compress_ratio == 0` and YaRN otherwise, including ratio 1. That is base 4.2's rule,
so the "YaRN on window-only layers" row of base 10 does not apply to DSv4.1.

**Compressor.** Only on KV-source layers.

| Ratio | Arithmetic | Kernel |
|---|---|---|
| 1 | `wkv` -> RMSNorm -> RoPE -> insert. No gate, no pooling, no state cache | The tail of `compress_main`, window 1 |
| 2 | Softmax gate over 2 consecutive tokens, no overlap, then the same tail | `compress_fp8.cu` instantiated at window 2, `coff = 1` (the C128 no-overlap path) |

No APE term in either (vLLM's `fused_save_compress_norm`: `pooled = sum(kv *
softmax(score))` over the pair at ratio 2, the raw row at ratio 1, then RMSNorm; RoPE at
`(pos // ratio) * ratio`). Implemented (`compress_fp8.cu`, golden `compressor.py`):

- `dsv41_compress_main`: ratio 2 through the existing window code at W = 2, `coff = 1`,
  from the paged FP32 state saved with a zero APE. The C4 / C128 paths are untouched.
- `dsv41_store_rows`: W = 1 reads one FP32 row per token instead of the state cache
  (`x` = the row exactly, which is also what a one-entry softmax gives), so ratio 1
  needs no state cache. D = 512 writes the fp8_ds_mla entry, D = 128 the FP8 indexer
  entry.
- Both can write the BF16 normed row before RoPE (`latent`), the indexer K's input.
- `dsv4_compressor_backward` takes ratio 2 (window 2, gated) and ratio 1 (no gate: the
  softmax weight is 1 and only d kv is written).

**Indexer K.** DSv4 pools hidden states with an indexer-local compressor
(`compress_indexer_cuda`, head 128, C4 with overlap). DSv4.1 derives the key from the
main compressor's pre-RoPE latent: `k = k_norm(wk(latent))` at group-boundary tokens
(`k_norm` an RMSNorm over 128), then RoPE at the group position and a store. The release
stores FP4; on SM90 the store stays DSv4's FP8 (D15), so the MQA logits kernel is
unchanged. vLLM has the same FP8 option (`indexer_k_norm_rope_store(use_fp4_cache=False)`:
per-token FP8 with one FP32 scale). `wk` is a 512 -> 128 BF16 linear without scales,
present only on KV-source layers.

- `wk` on DeepGEMM `bf16_gemm_nt` (FP32 out) over the latent rows.
- Norm, RoPE, FP8 quant and store: `dsv41_store_rows` at D = 128 on the `wk` output, at
  the layer's ratio (1 or 2).
- `compress_indexer_cuda` itself is not on the DSv4.1 path.

**KV and index sharing.** New in `Dsv4Layer`'s attention:

- A KV-source layer writes the compressed cache and the indexer K cache. Reuse and
  Reindex layers read the source's caches and write only their own sliding-window KV.
- An index source publishes top-k ids for its group; consumers read them. The
  canonical order of base 5.1 rule 2 is applied once, at publication.
- The workspace slab of base 5.1 rule 1 becomes one shared compressed slab per KV
  group plus per-layer window rows. With ratio 1, `N = max_model_len`.
- vLLM refuses a PP split inside a KV-sharing group. We are PP 1 (base D1), so nothing
  to do, but it should be an explicit check.

**Two-level candidates.** Layer 20 (the candidate source) scores every causal position,
takes the max over blocks of `candidate_block_size` (8) positions, and publishes the top
`candidate_topk_blocks` (2048) blocks. The four Reindex layers score only those
positions, at most 16,384, before their own top-512.

- Block max is order-independent, so it is exact. Block top-k can run on
  `stable_topk.cu`, whose `kMaxTopK` is already 2048.
- Reindex scoring: gather the candidates' K rows and call the existing MQA logits
  kernel. Each logit is one WGMMA over K = 128 plus a fixed-order head sum and does not
  read any other column, so gathered logits should equal dense-then-masked logits byte
  for byte. **This is an inference from base 5.2 and must be probed.**
- The top-512 tie-break key must use the original compressed position, not the
  position in the gathered list.
- No short-context shortcut (vLLM fills all indices when the context is at most
  `index_topk`; #62365 disables that under BI). One selector for every length.

### 4.5 Engram (`dsv41-engram-v1`)

From vLLM's `common/engram.py`:

```text
hashes[t, c]  = integer n-gram hash, c over (n-gram order 2..max) x n_heads           exact
rows          = table[hashes]                         FP8 codes + UE8M0 scales        exact copy
kv            = wkv(rows flattened)                   [T, (hc_mult + 1) * H]          dense linear, 4.2
per stream h in 0..hc_mult-1:
    key   = kv[:, h*H : (h+1)*H]
    dot   = sum(x_h * q_h * k_h * key) * rsqrt(mean(x_h^2) + eps)
                                       * rsqrt(mean(key^2) + eps) * rsqrt(H)
    g     = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))        0 where the token is masked
    out_h = x_h + g * value                                    value = kv[:, hc_mult*H :]
```

- Injection point: on the full `[T, 4, H]` streams, before the attention `mhc_pre` of
  layers 1 and 14.
- The hash is integer arithmetic, so it is batch-invariant by construction. It needs the
  previous `max_ngram - 1` token ids across chunk and decode boundaries, which is a
  `Dsv4Batch` field, not a numeric concern.
- The three H-long reductions reuse the fixed slice partition of the mHC kernel
  (extended to five chunks in 4.3). `sigmoid` through libdevice `expf` (base 5.1 rule
  10).
- `wkv` is one dense linear, roughly `[T, 6144] -> [T, 25600]` if the paper's
  dimensions hold (3 orders x 8 heads x 256 in, 5 x 5120 out).
- `q_h`, `k_h` are frozen BF16 `[hc_mult, H]` parameters. Nothing in Engram is a LoRA
  target, but both injections are on the dX path (layer 0's adapters sit below layer 1).

The numerics are small. The engineering is the tables: 196B FP8 parameters in host
memory, shared across the ranks of a node, with a gather and host-to-device copy per
step. The rollout plugin can keep vLLM's table storage and prefetch; the `rlkernel`
actor engine needs its own (shared memory across ranks, pinned staging).

### 4.6 MoE

- Router: vLLM's DSv4.1 MoE subclasses the DSv4 one (`sqrtsoftplus`, correction bias,
  hash layers from `mlp_layer_types`). `dsv4_router.cu` is already tested at E = 384.
  Expect zero hash layers; read it from the config.
- Routed experts: `sm90_fused_moe_mlp.cu` at H 5120, F 2304 (both multiples of 128),
  48 local experts at EP 8. Re-run its tests and benchmarks; no arithmetic change is
  expected.
- MXFP8 activation quant: 160 blocks of 32 per row instead of 128.
- Shared expert: follows D12. `silu_and_mul_clamp.cu` is unchanged.
- Persisted W8 expert weights (base 12): about 1.7 GB per layer per GPU at EP 8, 68 GB
  over 40 layers. `RL_KERNEL_DSV4_PERSIST_W8=0` should be the DSv4.1 default.

## 5. Exact sliding-window attention (D8)

### 5.1 The problem

Each layer's sliding-window KV comes from that layer's own hidden states and DSv4.1
does not persist it. The release rebuilds it with **bounded replay**: rerun only the
last `n_win` = 128 tokens, and truncate each replayed query's window at the replay
start `s`, so position `i` attends to `[max(s, i - 127), i]`. A token's bytes then
depend on `s`, which depends on the prefill boundary and the cache state.

D8: no truncation, ever. Two ways to get the exact bytes:

| | Decoder rows computed for an N-token prompt | Use |
|---|---|---|
| Full | N x 20 | v1 |
| Layer-staged | about 24,150 full + 2,540 KV-only, independent of N | v2 |

### 5.2 The layer-staged rule

Let `P` be the first position whose output is needed and layers 20-39 the decoder.

- **Layer 20's compressed KV and indexer K** are written for every token. That is the
  attention `mhc_pre`, `wkv`, the compressor and the stores; no attention.
- **Layer `l` runs in full** on positions `i >= P - 127 * (39 - l)`.
- **Layer `l` writes only its sliding-window KV** (`wq_a` / `wkv`, norm, RoPE, insert;
  no attention, no MoE) on the 127 positions before that.
- **Layers 0-19 run on every token.** Layer 20's KV needs their output everywhere.

Why: the top layer needs window KV for `[P - 127, P - 1]`, which needs layer 38's
output there, which needs layer 38's own window 127 further back, and so on down. The
decoder's compressed KV comes from layer 20's input and never from a decoder hidden
state, so nothing else reaches back.

| Prompt tokens | Full | Layer-staged (full rows) | Share of full |
|---|---|---|---|
| 4,096 | 81,920 | 24,150 | 29.5% |
| 32,768 | 655,360 | 24,150 | 3.7% |
| 131,072 | 2,621,440 | 24,150 | 0.9% |

With `P = N - 1` (5.4), layer `l` runs `127 * (39 - l) + 1` full rows: 127 * 190 + 20.

For comparison the approximate bounded replay computes 2,560 rows, so exact staging
costs about nine times that and is still a small fraction of a long prefill.

Generalized: with a set `R` of positions whose output is needed (every assistant turn
in a multi-turn sequence), layer `l` runs in full on `R` dilated backwards by
`127 * (39 - l)`. Each request carries the intervals as host integers.

**The rule is a derivation, not taken from an implementation.** Section 9 has the gate
that confirms or refutes it. vLLM #62365 goes the same direction (trailing rows kept so
outputs match the full forward; an optional layer-staged replay), on SM100.

### 5.3 Why skipping rows changes no byte

A skipped row is never read. With row-local, batch-invariant kernels, computing fewer
rows cannot change the rows that are computed. The same argument makes the backward
exact: a skipped row has zero gradient because it does not reach the loss.

### 5.4 Where it applies

- **Rollout prefill.** `P = N - 1` for an `N`-token prompt: the last prompt row's
  logits sample the first token, so layer 39 must run in full there. `N` is known when
  the request arrives, so each chunk of a chunked prefill knows which of its rows fall
  in the staged region.
- **Actor recompute and `forward_train`.** Same rule on the packed batch, with `P` one
  before the first response token (log-prob targets are `roll(ids, -1)`). Long-prompt
  agentic workloads save close to half the actor's prompt cost.
- **Decode.** Every new token runs all 40 layers. Unchanged.

### 5.5 Prefix cache

| Phase | Behaviour |
|---|---|
| v1 | The rollout does not reuse a prefix cache; every turn is a full prefill. Exact by construction, nothing to do in the KV manager |
| v2 | Staged replay on a prefix hit. With the cache boundary at `S`, layer `l` of 0-19 reruns from `S - 127 * (19 - l)`, at most 2,413 tokens back. The recomputed compressed KV equals the cached bytes, so it is not rewritten |

v2 must align the replay start to a ratio-2 group boundary, or cache the compressor's
pending state with the prefix.

## 6. Execution profile additions

Base 5.1 rules 1-12 stand. Additions:

13. **Shared attention state is written once and read only.** A KV group's compressed
    cache, indexer K cache, top-k ids and candidate blocks are written by their source
    layer and never by a consumer.
14. **One selector for every context length.** No short-context shortcut, and the same
    candidate path for prefill and decode.
15. **Candidate keys carry the original position.** Any top-k over a gathered subset
    breaks ties on the pre-gather index.
16. **Row sets are host integers.** Which rows a layer computes (section 5) is decided
    from request lengths on the host, never from device data, and never from the batch.

Reduction order for the new and changed modules:

| Module | Kernel | Reduction order |
|---|---|---|
| Dense linear (D12) | DeepGEMM `bf16_gemm_nt`, FP32 out | No split-K; weights exact BF16 from MXFP8 |
| mHC per-token math | `dsv4_mhc.cu`, five chunks | As base 5.2; each slice's FMA chain runs over five chunks (4.3) |
| Compressor r2 | `compress_fp8.cu`, window 2 | Per channel: max, `expf`, ascending sum, `kv * e` sum without FMA |
| Compressor r1 | same tail | Row-local |
| Indexer K | `bf16_gemm_nt` + norm / RoPE / FP8 store (D15) | Row-local |
| Candidate blocks | block max + `stable_topk.cu` | Max is exact; ties to the larger block index |
| Candidate logits | gather + DeepGEMM MQA logits | One WGMMA per logit, fixed head sum |
| Engram hash, lookup | integer kernel, indexing | Exact |
| Engram gate | new kernel | H-long dots on the fixed slice partition; libdevice `expf` |

## 7. Numeric profiles

| Profile | Module | Byte-equal to |
|---|---|---|
| `dsv41-mhc-v1` | Single-pass mHC at H = 5120 | its golden |
| `dsv41-dense-bf16-v1` | Dense linears under D12 | its golden; not equal to any MXFP8 kernel |
| `dsv41-engram-v1` | Engram gate | its golden |
| `dsv4-lora-v1`, `dsv4-logprob-v1`, `p5-sm90-fused-mlp-v1`, `p5-sm90-bwd-bf16-v1` | unchanged | as base 6 |

`dsv4-deepgemm-fp8-v1` (shared expert) is not used by DSv4.1 under D12. The reused
kernels (indexer query, stable top-k) keep the vLLM equalities of base 6. The new
profiles have nothing to be byte-equal to: there is no Hopper BI reference for DSv4.1.

## 8. Training

Base 7 stands. LoRA targets are unchanged and exist on every layer: attention `wq_a`,
`wkv`, `wq_b`, `wo_b`; routed and shared `fc1` / `fc2`. The compressor's `wkv` /
`wgate`, `indexer.wk`, Engram and mHC parameters are frozen.

| Module | Backward change |
|---|---|
| Attention | dKV from every layer of a KV group fans in to the one source, then through the main compressor's backward to the source layer's input. Order is free (base D5) |
| Compressor | Ratio-2 pooling backward (window 2); ratio 1 is norm and RoPE only |
| Indexer, candidates | No gradient (selection is a constant), as in DSv4 |
| mHC | dX through the carried `pre`: the gradient of a sublayer's collapse flows to the previous sublayer's controller |
| Engram | dX through the gate and the add; the tables and `wkv` get none |
| Dense linears | BF16 dgrad under D12 |
| Activation recompute | A Reuse layer's rerun needs its source's compressed KV and top-k ids. Either recompute per KV group or keep both as retained state next to the input streams |

The mHC row is the one real change in gradient structure: in DSv4 a sublayer's `pre` is
used inside the same `mhc_pre`; in DSv4.1 it is used one sublayer later.

## 9. Validation

Everything in base 8 at the new shapes, plus:

- **Probes to re-run at DSv4.1 shapes** (base 2.6; tile choice depends on N and K):
  DeepGEMM BF16 at K = 5120, 1280, 2304 and the Engram `wkv` shape; MQA logits at 32
  heads; FlashMLA sparse forward; the head GEMM at K = 5120;
  `tf32_hc_prenorm_gemm` at K = 20480.
- **Gathered vs dense logits**: candidate logits byte-equal to the same columns of the
  dense logits (4.4).
- **Sharing**: a Reuse layer's output is byte-equal whether its source ran in the same
  step, an earlier chunk, or a decode step.
- **Layerwise gate with staging**: full and staged forwards byte-equal on every row
  each layer is required to compute, for rollout, actor and train.
- **Negative control for staging**: shrink one layer's row set by one position; the
  gate must report that layer.
- **Negative control for Engram**: empty tables must change the bytes after layer 1.
  (Miles reports a log-prob gap jumping from about 0.02 to about 10 when its tables come
  back empty after a sleep / wake, so this failure is silent without a check.)
- **Reference check**: `check_dsv4_model_ref.py` against DeepSeek's DSv4.1 inference
  code on a toy checkpoint, for wiring errors.

There is no vLLM BI golden to compare against on Hopper. The gate is `Dsv4Model`
against itself across batch layouts, plus the wiring check above.

## 10. The vLLM rollout

Same mechanism as base 10: a general plugin registers the model class, no vLLM source
changes. What differs for DSv4.1 is that the plugin starts from vLLM's **non-invariant**
Hopper path, so more of it will need replacing than the base doc's table lists. Expected,
unverified until the gate runs:

| vLLM DSv4.1 (Hopper) | Expected handling |
|---|---|
| MXFP8 linears (stock kernel at TP 1) | D12 path |
| TileLang mHC with stream overlap and fused all-reduce | `dsv41-mhc-v1`, one stream |
| `DeepseekV4MoE` | `Dsv4MoE`, as in DSv4 |
| `fused_q_kv_rmsnorm`, fused q RoPE / KV insert | ours, q-norm off |
| Triton `fused_save_compress_norm`, `rope_quant_insert`, `indexer_k_norm_rope_store` | `compress_main` r1 / r2 and the indexer-K tail |
| `SparseAttnIndexer` or CuTe-DSL `CandidateIndexer`, short-context shortcut | one selector, 4.4 |
| `DecoderReplayLayers` (bounded replay) | never used; `swa_bounded_replay` off; row sets from section 5 |
| Engram Triton gate | `dsv41-engram-v1`; vLLM's table storage kept |
| DSpark, vision | not loaded |
| BF16 logits | FP32 head, as in DSv4 |

Note for the plugin: vLLM turns its decoder replay off when a LoRA config is set and
runs it only in eager prefill steps. Our plugin hides LoRA from vLLM (base 10.1), so
that switch would otherwise be on.

### 10.1 Plugin design (2026-11-25)

**Environment (owner, 2026-11-25): vLLM main `3f86db3` built from source** in a new env
`rlk41` (a clone of `rlk` with torch 2.14, which vLLM main pins; CUDA runtime 13.0, so the
existing forward-compat driver serves it). RL-Kernel's `_C`, its pinned DeepGEMM
(`e21c821f`) and FlashMLA (`a8f794d`) are rebuilt there; vLLM builds its own FlashMLA
(`b18ed744`) and vendored DeepGEMM. Job: `build_rlk41.sbatch` on the cluster.

**Split.** vLLM main has no batch-invariant DSv4.1 path on Hopper (its Hopper kernels are
TileLang / CuTe DSL / Triton, with aux-stream overlap and batch-shaped planning), so unlike
the DSv4 plugin, which kept vLLM's attention ops under the fork's BI mode, the DSv4.1
plugin keeps **only vLLM's scheduler, KV-cache allocation and attention metadata** and runs
every computation as `Dsv41Model` does:

| vLLM main (Hopper) | Plugin |
|---|---|
| `DeepseekV4SWACache` per layer (block size 32, V4 record, 584 B) | our SWA writes and window rows on that tensor and block table (block size from the cache, not 64) |
| Compressed cache of a KV source (`MLAAttentionSpec`, `tokens_per_state = ratio`), read by the group through the forward context | `LayerCaches.kv` / `kv_table` of the source, aliased by the group (as `new_group_caches`) |
| `DeepseekV4IndexerCache` (owned by the KV source) | `idx_k` (FP8 entry, `use_fp4_cache=False`, 132 B) |
| `CompressorStateCache`: a per-request ring of the open ratio-2 group | the state cache of `dsv41_compress_main` with a block table whose every entry is the ring block (row `p % capacity` is what our view computes for block size = capacity) |
| `fused_save_compress_norm`, `rope_quant_insert`, `indexer_k_norm_rope_store` | `save_states` + `dsv41_compress_main` / `dsv41_store_rows` |
| q / kv norm, q RoPE, KV insert | `rmsnorm` + `dsv41_q_rope_kv_insert` |
| `SparseAttnIndexer` / `CandidateIndexer`, short-context shortcut | `Dsv41Attention._index` (one selector) |
| FlashMLA sparse prefill / decode planning | `build_workspace` / `build_compact` + FlashMLA sparse forward (our pinned build) |
| `DecoderReplayLayers` | off (`swa_bounded_replay=False`) |
| mHC (TileLang, overlap, fused all-reduce), Engram Triton gate, MoE, MXFP8 linears, `wo_a` / `wo_b` | `Dsv41DecoderLayer` pieces (`dsv41-mhc-v1`, `dsv41-engram-v1`, `Dsv4MoE` + BF16 shared expert, D12 linears) |
| BF16 logits, sampler | FP32 head and log-prob (`RlkSampler`, as DSv4) |

Order of work once `rlk41` builds: load vLLM's DSv4.1 model on the toy checkpoint and dump
the attention metadata per cache prefix (slot mappings, block tables, ring capacity,
block strides); write the cache adapters; then the model class, registered under
`DeepseekV41ForCausalLM` by the plugin entry point; then the gate (rollout vs `Dsv41Model`,
bitwise), as `scripts/check_dsv4_vllm_gate.py` for DSv4.

verl integration (base 10.1) is unchanged in shape. The actor engine additionally needs
Engram table storage shared across ranks.

## 11. Memory and performance expectations

Estimates from dimensions. Nothing here is measured.

| Item | Estimate | Basis |
|---|---|---|
| Routed experts, MXFP4, per GPU at EP 8 | about 36 GB | 384 x 3 x 5120 x 2304 x 40 = 543.6B weights |
| Persisted W8 experts, per GPU at EP 8 | about 68 GB | 1 byte per weight; default off |
| Dense weights as BF16, per GPU (replicated at TP 1) | about 13 GiB, about 6.5 GiB over FP8 | About 7.0B dense MXFP8 weights: 40 layers x about 167M (with DSv4's `o_groups` / `o_lora_rank`) plus Engram `wkv`. vLLM quotes 12.4 GiB for BF16 |
| Engram tables, host | about 196 GB per node | 196B FP8 parameters, shared by the ranks (SGLang reports 189 GiB) |

- H200 (141 GB) looks feasible at EP 8 with W8 persistence off. H100 (80 GB) is
  doubtful and needs a measured budget before promising it.
- The rollout baseline can run DSpark and ours cannot (D9), so the decode-latency gap
  of base 10.3 will look wider on DSv4.1 than on DSv4.
- Layer 20's indexer scores every position at ratio 1, four times the entries of a
  DSv4 C4 layer per token. The four Reindex layers score at most 16,384.
- Base D7 (within 1.3x per operator) applies to each new kernel against its vLLM
  counterpart.

## 12. Work plan

All DSv4.1 work goes on a new branch, `test-dsv41-cuda`, created from `test-dsv4-cuda`
(#635). It follows `test-dsv4-cuda` by merging from it, and the DSv4 regression of base
2.3 must stay green there: the shared kernels gain parameters and flags, and the DSv4
profiles keep their bytes.

Ordered so each step has a passing gate before the next starts.

| # | Step | Size |
|---|---|---|
| 0 | Create `test-dsv41-cuda` from `test-dsv4-cuda`; add this page as `docs/architecture/dsv41-design.md` | S |
| 1 | Config, loader, toy checkpoint; re-run the probes at DSv4.1 shapes | S |
| 2 | Dense linear under D12, including grouped `wo_a` | S |
| 3 | mHC: five chunks, single-pass, new profile, backward | M |
| 4 | Attention: q-norm flag, compressor r1 / r2 forward and backward, indexer K | M |
| 5 | Attention: KV / index sharing, candidates, backward fan-in, recompute | L |
| 6 | Engram: hash, gate kernel, host tables for the actor | M |
| 7 | Model wiring, layerwise gate (full rows), reference check | M |
| 8 | vLLM plugin and its gate; verl engine | L |
| 9 | Layer-staged SWA (section 5, v2) with its gate and negative control | M |
| 10 | Prefix-cache staged replay (5.5, v2) | M |

Progress (2026-11-25, 1 x H100; `tests/ops/{mhc,compressor,engram,topk}`,
`tests/models/deepseek_v41`):

| Item | State |
|---|---|
| 0 | Branch and this page |
| 1 | `Dsv41Config` on the release `config.json` (`release_config.json`); `checkpoint_spec` equal to the release's tensors (headers of `314113f8`); `random_checkpoint` / `toy_config()`. Probes: DeepGEMM `bf16_gemm_nt` batch-invariant at every DSv4.1 dense shape (`scripts/probe_dsv41_deepgemm_bf16_bi.py`), MQA logits at 32 heads, `tf32_hc_prenorm_gemm` at K = 20480 (reports `docs/validation/dsv41-*`) |
| 2 | `mxfp8_dequant` (exact) and `Bf16Linear`; `wo_a` per group in BF16 (`Bf16OutputProjection`); BF16 shared expert |
| 3 | mHC: carried pre, five chunks, final collapse, dX backward (`dsv41-mhc-v1`, eps = `norm_eps`); DSv4 bytes and speed unchanged |
| 4 | q-norm off (`dsv41_q_rope_kv_insert`); compressor ratio 2 / ratio 1, indexer K, the pre-RoPE latent, backward at both ratios |
| 5 (forward) | KV / index sharing (`new_group_caches`, `shared`), candidates; backward fan-in and recompute not started |
| 6 | Engram gate (`dsv41-engram-v1`), hashing equal to DeepSeek's reference (token map of the release tokenizer, 99092 ids), table lookup |
| 7 (forward) | `Dsv41Model`, `load_model`; the toy model (every role, an Engram layer) is batch-invariant and gives the same bytes in prefill, chunked prefill and decode. Reference check `scripts/check_dsv41_model_ref.py` against DeepSeek's `inference/model.py` on a random toy checkpoint: on equal inputs mHC agrees to 1e-5, the router picks the same experts, every layer's attention (window, Full r2 / r1, Reuse, Reindex with candidates) agrees to 3.3e-2 relative (our FP8 KV against the reference's BF16 with `--bf16-reference`), the shared expert to 4e-3, the MoE to 2e-2; no layer stands out. Through the stack the random toy model amplifies these (0.15 to 0.47 relative over 7 layers in BF16 mode) |
| 5 / training | `Dsv41Model.forward_train` (byte-equal to `forward_batch`) and the attention backward (`attention_backward.py`): no q-norm, BF16 dgrads with the MXFP8 rounding straight through, compressor backward at ratios 1 / 2, and the group handle: a KV source's attention returns a handle that the group's other layers take, so autograd sums their compressed-entry gradients into the source's compressor backward. LoRA on every target including the BF16 shared expert (`attach_lora` from DSv4). Checks: with the loss on a reuse layer only, dL/dx of the KV source matches finite differences (correlation 0.99, magnitude 0.93 to 0.98; steps below 0.2 drown in the FP8 KV rounding); every adapter gets a finite nonzero gradient |
| 5 / recompute | `forward_train(recompute=True)`: per layer, autograd keeps the input streams and the carried pre and reruns the layer in the backward; a KV source's recompute function also returns the group handle, so the rerun backpropagates the group's summed entry gradient. Same forward bytes, gradients within the attention backward's atomics (cosine > 0.999); held activations 1116 to 136 MiB on the toy model at 400 tokens |
| Gate | `scripts/check_dsv41_layerwise_gate.py`: rollout (staggered arrivals, chunked prefill, decode beside prefill) against the actor's packed prefill and the training forward, every layer byte-identical, LoRA rank 16. Toy model PASS; the release, all 40 layers (2026-11-26, 4 x H200, EP 4, Engram tables mapped from the shards, 2108 tokens): PASS, 0 differing tokens at every layer, final hidden and log-probs identical; loads in 20 s |
| In progress | vLLM rollout plugin on vLLM main `3f86db3` (10.1; env `rlk41` building) |
| Not started | verl integration, layer-staged SWA (9), prefix-cache replay |

Step 9 comes after a green gate on purpose: it changes which rows are computed and no
kernel's arithmetic, so a failure there is easy to localize.

## 13. To confirm

- [x] `config.json`: the release's (2.1); every value of 2.3 and of 4.5 confirmed.
- [ ] D9 (no DSpark), D10, D11, D12, D13, D14: owner sign-off. D8 and D15 are decided.
- [x] Which indexer K format vLLM's DSv4.1 Hopper path stores: MXFP4 or per-token FP8
      (`use_fp4_cache`); D15 keeps the FP8 one, which is DSv4's indexer entry.
- [x] Whether the pinned DeepGEMM is batch-invariant at the DSv4.1 shapes: yes for the BF16
      GEMM and the MQA logits (2026-11-25, 12 progress table).
- [x] Gathered-candidate logits: not needed; the indexer gathers request-local rows of
      the dense logits and masks them (4.4, `tests/ops/topk/test_dsv41_candidates.py`).
- [ ] The staging rule of 5.2, by the gate in section 9.
- [x] Where `hc_head` sits: DSv4.1 has none; the output is `norm(collapse(streams, pre))`
      with the last FFN's pre (vLLM `nvidia/model.py`, 4.3).
- [ ] H100 memory budget.
- [x] DeepSeek's DSv4.1 inference code: in the release repository (`inference/`).

## 14. References

- Base: `docs/architecture/dsv4-design.md` (RL-Kernel #635, `6d974b9`)
- vLLM main `3f86db3`: `vllm/models/deepseek_v41/{attention,compressor,sparse_mla,decoder_replay_layers,quant_config}.py`,
  `common/{engram,rope}.py`, `nvidia/{model,quantization}.py`, `nvidia/ops/candidate_indexer.py`
- vLLM #62365: DSv4.1 batch invariance (open, GB200 only)
- DeepSeek-V4.1-Flash technical report, arXiv 2609.19969
- Release: `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` at `314113f8` (`config.json`,
  `inference/`, `DeepSeek_V41_Tech_Report.pdf`)
- Miles DeepSeek-V4.1 Flash docs (Engram host tables, bounded replay behaviour)
- Inferact `zero-mismatch-rl`, DeepSeek-V4.1-Flash case (SM100)
