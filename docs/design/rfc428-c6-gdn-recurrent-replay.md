# RFC #428 C6 — Qwen3-Next Gated DeltaNet recurrent replay

Design notes for RFC #428 work item C6 (GDN recurrent response replay) on the CUDA
track. Measured on 2× B200 (sm_100), torch 2.13.0+cu130, vllm 0.30.0,
transformers 5.17.0. vLLM paths below are relative to the installed `vllm` 0.30.0
package; two files are named `causal_conv1d.py`, and each citation says whether it
means vLLM's (`model_executor/layers/mamba/ops/causal_conv1d.py`) or the golden's
(`rl_engine/kernels/ops/pytorch/linear_attn/causal_conv1d.py`).

Claim level reached: **L0 repeatable, L1 batch-invariant**. L2 is not claimed.

## 1. Which provider a rollout decode actually takes

vLLM 0.30.0 picks the GDN decode path per engine step. The choice depends on env
defaults, on what else the step contains, and on how the model constructs the layer:

| step contains | conv | recurrence |
|---|---|---|
| decodes only, no draft tokens | `causal_conv1d_update` | `fused_recurrent_gated_delta_rule_packed_decode` |
| decodes and at least one prefill, no draft tokens | `causal_conv1d_fn` | `fused_sigmoid_gating_delta_rule_update` for the cached decode rows |
| draft tokens (speculative decode / MTP) | `causal_conv1d_update` with `num_accepted_tokens` | `fused_sigmoid_gating_delta_rule_update`; plain decodes in the step are reclassified as prefills |

- **Decode-only.** With `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` at its default of on
  (`envs.py:1199-1200`), a decode-only step returns early at
  `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1295-1307` into
  `_forward_core_decode_non_spec`, which calls the packed kernel (`:1697-1708`). **The
  goldens target this path.** With the variable off, the same step goes through
  `fused_sigmoid_gating_delta_rule_update` instead (`:1553-1572`).
- **Mixed decode and prefill.** `_forward_core` runs the conv for the whole non-spec
  batch through `causal_conv1d_fn` (branch at `:1373`, call at `:1378-1388`) and the
  cached decode rows through `fused_sigmoid_gating_delta_rule_update`
  (`split_non_spec` defined at `:1408-1412`, branch at `:1493`, call at `:1497-1512`).
  See §5 for what that does to the claim.
- **Draft tokens.** Spec rows go through `causal_conv1d_update` with
  `num_accepted_tokens` (`:1357-1370`) and `fused_sigmoid_gating_delta_rule_update`
  (`:1470-1487`). Plain decodes in the same step are reclassified as prefills
  (`v1/attention/backends/gdn_attn.py:283-289`). A step with speculative decoding
  enabled but zero draft tokens sets `spec_sequence_masks` to `None`
  (`gdn_attn.py:236-243`) and is treated as decode-only.

`VLLM_GDN_DECODE_KERNEL` defaults to `"cuda"`, but that is not what Qwen3-Next runs.
`model_executor/models/qwen3_next.py:495` constructs the layer with
`gqa_interleaved_layout=True`, which makes `_fused_gdn_decode_unsupported_reason`
(`qwen_gdn_linear_attn.py:535-554`) return a reason. The layer then logs a fallback
to `"triton"` (`:520-523`), or raises `ValueError` if `VLLM_GDN_DECODE_KERNEL` was
explicitly set to `cuda` (`:516-519`). As a consequence the fused
`torch.ops._C.fused_gdn_decode_post_conv_mtp` path is unreachable for Qwen3-Next:
`_can_use_fused_gdn_mtp_decode` requires `gdn_decode_kernel == "cuda"` (`:1834`).
This section is read from the source.

`tests/check_qwen3_next_norm_providers.py` asserts both env defaults (`:129-142`). For
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` that is a useful guard: a vLLM bump that
flips it fails loudly. For `VLLM_GDN_DECODE_KERNEL` it guards nothing for Qwen3-Next,
because the interleaved layout, not the default, decides the kernel. Neither
assertion checks which kernel actually runs.

## 2. How the golden relates to the kernel

The golden follows the kernel's arithmetic, not `modeling_qwen3_next.py`'s. Checked
line by line against `third_party/flash_linear_attention/ops/fused_recurrent.py:288-335`,
three places where the kernel and the HF model differ:

1. **The gating is fused, in fp32.** `beta = sigmoid(b)` and
   `g = -exp(A_log) * softplus(a + dt_bias)` are computed inside the Triton kernel,
   with a `softplus` threshold branch at 20. HF computes them as separate PyTorch
   ops — a different rounding path.
2. **No `repeat_interleave`.** The kernel indexes `i_h = i_hv // (HV // H)`, so q/k
   stay at 16 heads while v has 32. HF materializes the repeat.
3. **The QK norm is an L2 norm over a plain sum**, `x / sqrt(sum(x*x) + 1e-6)` — not
   an RMSNorm, not `F.normalize`, and dividing by `sqrt` rather than multiplying by
   `rsqrt`, which differs in the last bit. `scale` is applied to `q` *after* the
   norm; `k` is never scaled.

Where the golden is **not** a transcription:

- softplus: the kernel computes `tl.log(1.0 + tl.exp(x))` (`fused_recurrent.py:327`);
  the golden computes `torch.log1p(torch.exp(safe))`, where `safe` is `x` on the taken
  branch (`gated_delta_rule.py:110-112`). These round differently.
- The kernel's `exp`/`log` become `fast_expf`/`fast_logf` when `FLA_USE_FAST_OPS=1`
  (`third_party/flash_linear_attention/ops/op.py:16-25`). The golden models the
  default.

Checked and found **not** to be a divergence: prefill passes
`use_qk_l2norm_in_kernel=False` (`qwen_gdn_linear_attn.py:1542`) only because
`fused_post_conv_prep(apply_l2norm=True)` (`:1450`) already normalized q/k. Decode
passes `True` (`:1707`). Both paths normalize exactly once.

## 3. State ABI

Mirrored rather than reinvented:

- recurrent state `[num_blocks, HV, V, K]`, V-major, addressed by `ssm_state_indices`
- conv state `[num_blocks, dim, width-1]`, layout chosen by the global
  `is_conv_state_dim_first()`; the golden takes it as an argument and both are tested
- the accumulator is fp32 for the whole step; the store rounds to the state tensor's
  dtype, which `FUSED_GDN_STATE_DTYPES` (`qwen_gdn_linear_attn.py:91`) allows to be
  fp32 **or** bf16
- `causal_conv1d_update` casts `x` to the cache dtype before computing

`NULL_BLOCK_ID` is **not** one contract across the two providers:

| | recurrent provider | conv provider | both goldens |
|---|---|---|---|
| which indices skip | `<= 0` (`fused_recurrent.py:300`) | `== null_block_id` only, default `NULL_BLOCK_ID` = 0 (vLLM `causal_conv1d.py:835`; `v1/attention/backends/utils.py:47`) | `<= 0` |
| output for a skipped row | zeros (`fused_recurrent.py:301-302`) | not written (vLLM `causal_conv1d.py:835-839` returns before any store) | zeros |
| state block | not touched | not touched | not touched |

A negative conv index is therefore inactive in the golden
(`tests/test_gdn_state_contract.py:54-59`) but a real, out-of-range index to the conv
provider.

Contractions use `_chunked_sum` (`gated_delta_rule.py:76-90`): fixed 32-wide chunks,
so the reduction shape per row does not depend on the batch size, rather than
`torch.matmul`, whose reduction order is unspecified. That is the argument for the
L1 claim; L1 itself is established empirically by `test_golden_is_batch_invariant`.
The same choice is why the golden is not bitwise against the kernel's reduction.

## 4. Agreement with the provider

Qwen3-Next dims (H=16, HV=32, K=V=128), bf16 I/O, `use_qk_l2norm_in_kernel=True`,
random inputs, B ∈ {1, 4, 17, 64}, one seed per batch. Bounds asserted by
`tests/check_gdn_recurrent_golden.py` (`_RECURRENT_BOUNDS`). They, and every other
bound in that file, are **regression bounds against the provider, not gate evidence**;
they do not go through `resolve_tolerance`. For scale, the gate contract's
`forward_accuracy/by_op_class/reduction` row in
`rl_engine/kernels/gtest/tolerance_contract.json` is atol = rtol = 1e-4 for float32
and atol = 5e-2, rtol = 2e-2 for bfloat16.

| | max\|diff\| out | max\|diff\| state |
|---|---|---|
| fp32 state | ≤ 1e-3 | ≤ 1e-5 |
| bf16 state | ≤ 1e-3 | ≤ 5e-3 |

Measured values for the same inputs, from `scripts/ws1_gdn_provider_agreement.py` on
B200 at commit `acf38b6` (clean checkout). The last column counts output elements
whose bits differ:

| state | B | max\|diff\| out | max\|diff\| state | out elements differing |
|---|---|---|---|---|
| fp32 | 1 | 1.49e-08 | 1.19e-07 | 1 / 4096 |
| fp32 | 4 | 9.54e-07 | 1.79e-07 | 2 / 16384 |
| fp32 | 17 | 3.81e-06 | 2.38e-07 | 10 / 69632 |
| fp32 | 64 | 6.10e-05 | 2.98e-07 | 40 / 262144 |
| bf16 | 1 | 3.73e-09 | 9.77e-04 | 2 / 4096 |
| bf16 | 4 | 1.53e-05 | 9.77e-04 | 2 / 16384 |
| bf16 | 17 | 3.05e-05 | 1.95e-03 | 14 / 69632 |
| bf16 | 64 | 3.05e-05 | 1.95e-03 | 36 / 262144 |

This reproduces the figures an earlier version of this note quoted without a runner
(out 1.5e-08 .. 6.1e-05 and state ≤ 3.0e-07 for fp32; out 3.7e-09 .. 3.1e-05 and
state ≤ 2.0e-03 for bf16). It is one seed per batch on one device.

Causal conv uses sequential FP32 accumulation **starting from bias**, with
products first rounded to the operand dtype (golden `causal_conv1d.py:167-177`;
the provider initialises from bias at vLLM `causal_conv1d.py:960-967`, `1000`). The
previous BF16 path incorrectly promoted both operands to FP32; its disagreements
were not limited to one BF16 ULP. The CPU tests in `tests/test_gdn_state_contract.py`
cover bias order and bf16 product rounding, and
`test_conv_provider_preserves_bf16_product_cancellation` checks the provider on a
constructed cancellation input. Provider comparisons limit mismatches to 32 elements
on the checked fixtures (a regression bound, not gate evidence). This is not a bitwise claim.

Measured with the same runner and commit: the rolled conv state is bitwise equal in
all 8 (batch, cache dtype) cases. Output elements differing, with an fp32 cache:
0 / 8192 (B=1), 0 / 32768 (B=4), 1 / 139264 (B=17, max|diff| 2.44e-04) and
5 / 524288 (B=64, max|diff| 3.91e-03). With a bf16 cache: 0 at B=1, 4 and 17, and
3 / 524288 at B=64 (max|diff| 1.56e-02).

**Why a few fp32-cache outputs differ.** There are two mechanisms, both on the provider
side, and which one applies depends on the Triton specialization. Measured by
`scripts/ws1_gdn_provider_agreement.py` (`conv_silu`, `conv_noact_bf16`) on B200 at commit
`bb89750`:

*With SiLU -- Qwen3-Next's configuration and the check file's -- the activation's
implementation.*

- With the activation off and the output kept in fp32, provider and golden agree bitwise
  on every element at B = 1, 4, 17 and 64: the cast to the cache dtype, the bias start,
  the product rounding and the tap order all match.
- Both sides compute `acc / (1 + exp(-acc))` (vLLM `causal_conv1d.py:1085`), but Triton
  lowers the exp and the fp32 division to `ex2.approx` and `div.full.f32`, while the
  golden's PyTorch result equals the same expression with `libdevice.exp` and IEEE
  division (`div_rn`). Applying Triton's `x / (1 + tl.exp(-x))` to the golden's
  pre-activation values reproduces the provider's fp32 output bitwise; replacing only the
  exp, or only the division, reproduces neither side.
- In fp32 about 38% of outputs differ: 86% of those by 1 ULP, 12% by 2, 3% by 3 or 4,
  0.2% by more. Only values on opposite sides of a bf16 rounding midpoint survive the bf16
  store: 0, 0, 1 and 5 elements, each one bf16 ULP apart.
- FP contraction plays no part on this path. These specializations compute the tap
  products as packed `mul.f32x2` with separate adds and contain no `FFMA`; recompiling
  with `TRITON_DEFAULT_FP_FUSION=0` leaves every count unchanged.

*Without an activation and with a bf16 output -- FP contraction.*

- With fusion at its default, this specialization contracts each tap's multiply-add
  (`fma.rn.f32x2` in the PTX, `FFMA` in the SASS). Its fp32-output twin computes the same
  values (the provider casts x to the fp32 cache dtype first) but is not contracted, and
  matches the golden bitwise.
- The bf16 outputs differ in 1, 0, 4 and 14 elements (max |diff| 3.9e-3). 18 of the 19
  have |out| < 0.11, where the taps nearly cancel; there the gap can span several bf16
  ULPs (up to 96 at |out| ≈ 2.4e-7) while staying at most 2.4e-7 in absolute terms.
- With `TRITON_DEFAULT_FP_FUSION=0` this specialization compiles to separate multiplies
  and adds, and the mismatches drop to 0. The golden is self-consistent across output
  dtypes. Qwen3-Next's conv applies SiLU, so this specialization is not on its decode
  path.

*Open:* the recurrent provider computes `exp(g)` and `sigmoid(b)` with Triton (`tl.exp`
via the vendored FLA `op.py`), the golden with PyTorch; the same kind of difference may
account for part of the recurrent output mismatches in the table above. Not
investigated.

## 5. Decode versus chunked prefill

The earlier 1024-step drift and prefill tables did not have a checked-in runner;
they are withdrawn as acceptance evidence. The existing 128-step synthetic test
only bounds its fixed seed and gate inputs. It does not establish a universal
plateau, prefill/decode equality, or any bound on model logits.

**Mixed decode-and-prefill steps (provider side).** The goldens target the
decode-only path (§1). In a step that also holds a prefill, vLLM 0.30.0 sends the
cached decode rows through `causal_conv1d_fn` and
`fused_sigmoid_gating_delta_rule_update` instead, so in those steps the recurrent
golden does not target what rollout runs.

*The numbers in the rest of this subsection are a reported observation with no
checked-in runner in this repository. By the same rule that withdrew the tables above,
they are not acceptance evidence.*

A measurement on B200 drove the real `GDNAttentionMetadataBuilder.build()` and
`_forward_core` of **one standalone layer** built from the checkpoint's
`config.json`. Its limits: parameters and cache were synthetic (no weights loaded); it
ran in a single process, with no engine, scheduler or CUDA graph; and the test, not the
scheduler, built the attention metadata (decode rows first). For one target decode
request with an fp32 recurrent state, three prefill-bearing step compositions gave
identical numbers:

| head counts | step's bf16 output | fp32 state |
|---|---|---|
| TP1 (H=16, HV=32) | matched | 75,146 of 524,288 elements differ (8.8e-08 relative) |
| TP4 per-rank (H=4, HV=8) | 1 element differs (4.8e-05 relative) | 32,663 of 131,072 differ (1.7e-07) |

The convolution output and conv state matched bitwise in every composition, so the
difference comes from the recurrent kernels, not from `causal_conv1d_fn`. At TP1 the
state difference reached a bf16 output within the next 16 plain decode steps in 4 of
20 seeds; that was measured at TP1 only, and with synthetic parameters the frequency
does not carry over to real weights. At TP4 per-rank head counts it appears in the
same step's output. This is a provider-side batch-composition dependence: whether a
prefill shares the step changes a decode row's state, and at TP4 per-rank head counts
its output too, which no golden can fix. The measurement was made outside this change
and is not yet published.

vllm-project/vllm#49827, open and unmerged, routes mixed-step *recurrent* decodes
through the packed kernel too. Its two commits, applied to 0.30.0, closed the gap to 0
at both head counts above; its scheduler part was not tested. It does not change the
conv path in mixed steps, which stays `causal_conv1d_fn`, and its own validation is on
Qwen3.5 (non-interleaved), H100, TP1. Disabling
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` instead sends every non-speculative decode
row through `fused_sigmoid_gating_delta_rule_update`; the same measurement found the
decode row's output and state bitwise equal in every step composition in that
configuration, and the recurrent golden's target would then have to change.

The strict profile uses FP32 recurrent state. BF16 state remains a differential
experiment. Full checkpoint prefill, response replay, optimizer updates and
reload all remain required before L2 can pass.

Cache indices must be int32/int64, on the input device, and positive active
indices must be unique and in range. Nonpositive sentinels may repeat. The golden
validates before any cache update. Raw provider calls remain the caller's
responsibility: neither provider checks index values. `causal_conv1d_update`'s
`validate_data=True` adds only shape and stride asserts and a
`null_block_id is not None` assert (vLLM `causal_conv1d.py:1151-1153`, `1188-1201`),
so an out-of-range index is an unchecked memory access in either mode (read from
the source, not exercised).

## 6. Current boundary

Not covered, with reasons:

| deferred | why |
|---|---|
| Speculative decode / MTP | RFC #428 §2.2 excludes speculative decoding from the first claim. Qwen3-Next's MTP head ships in the checkpoint (`mtp.*`, loaded by `model_executor/models/qwen3_next_mtp.py`) and is full attention (`qwen3_next_mtp.py:90-92`). Enabling it changes the target model's GDN path in steps that carry draft tokens (§1); `fused_gdn_decode_post_conv_mtp` is unreachable for Qwen3-Next |
| Backward for the recurrent step | RFC #428 §2.2 item 4: backward need not match rollout, only be correct for the replayed forward. RFC §9.1 makes it a separate work item, **RFC #428 C7** (GDN backward/recompute adapter including prompt-state gradient). `supports_backward=false` here; `_softplus`'s NaN-gradient fix (`08969ac`) adds no backward claim |
| Provider bridge / registry entry | the golden should survive a drift sweep against a real checkpoint first |
| Paged block allocation policy | the ABI is mirrored; the allocator is not modelled |
| TP sharding of `A_log` / `dt_bias` | single card only |
| ROCm / Ascend | CUDA first, per the WS1 order |

`runtime_verified=false` (no checkpoint), `supports_backward=false`,
`checkpoint=absent`. Op-level agreement says nothing about 48 composed layers.

## 7. Accepted execution boundaries

Use a shared, explicitly pinned vLLM-compatible forward provider on both sides,
with independent VIME recomputation. Disable MTP and prefix reuse. Keep FP32
recurrent state for strict acceptance; BF16 is experimental. Require bitwise
logits/logprobs at a common topology, at least two real optimizer updates, weight
synchronization and checkpoint reload. No operator-level test closes these gates.

The 2026-09-30 real-checkpoint startup attempt with vLLM 0.30.0 failed before
inference: `VLLM batch_invariant mode is not supported for GDN_ATTN`. A shared
provider integration must resolve this; disabling the check is not L2 evidence. *This
failure is a reported observation: the attempt's log and launcher are not checked into
this repository, so it is not acceptance evidence either.*
