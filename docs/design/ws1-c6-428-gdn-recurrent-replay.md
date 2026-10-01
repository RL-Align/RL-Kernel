# WS1 C6 — Qwen3-Next Gated DeltaNet recurrent replay

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

`tests/check_qwen3_next_norm_providers.py` asserts both env defaults (`:124-131`). For
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
  branch (`gated_delta_rule.py:107-109`). These round differently.
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
(`tests/test_gdn_state_contract.py:51-56`) but a real, out-of-range index to the conv
provider.

Contractions use `_chunked_sum` (`gated_delta_rule.py:73-87`): fixed 32-wide chunks,
so the reduction shape per row does not depend on the batch size, rather than
`torch.matmul`, whose reduction order is unspecified. That is the argument for the
L1 claim; L1 itself is established empirically by `test_golden_is_batch_invariant`.
The same choice is why the golden is not bitwise against the kernel's reduction.

## 4. Agreement with the provider

Qwen3-Next dims (H=16, HV=32, K=V=128), bf16 I/O, `use_qk_l2norm_in_kernel=True`,
random inputs, B ∈ {1, 4, 17, 64}, one seed per batch. Bounds asserted by
`tests/check_gdn_recurrent_golden.py`:

| | max\|diff\| out | max\|diff\| state |
|---|---|---|
| fp32 state | ≤ 1e-3 | ≤ 1e-5 |
| bf16 state | ≤ 1e-3 | ≤ 5e-3 |

`scripts/ws1_gdn_provider_agreement.py` prints the measured values behind these
bounds for the same inputs. An earlier table here quoted tighter figures from a
single run with no committed runner; it is withdrawn in favour of the runner's
output.

Causal conv uses sequential FP32 accumulation **starting from bias**, with
products first rounded to the operand dtype (golden `causal_conv1d.py:167-177`;
the provider initialises from bias at vLLM `causal_conv1d.py:960-967`, `1000`). The
previous BF16 path incorrectly promoted both operands to FP32; its disagreements
were not limited to one BF16 ULP. The CPU tests in `tests/test_gdn_state_contract.py`
cover bias order and bf16 product rounding, and
`test_conv_provider_preserves_bf16_product_cancellation` checks the provider on a
constructed cancellation input. Provider comparisons limit mismatches to 32 elements
on the checked fixtures. This is not a bitwise claim.

With an fp32 cache a few output elements still differ. The cause is not pinned. One
candidate is FP contraction: Triton's default `enable_fp_fusion=True` may contract the
provider's `acc += matrix_x * matrix_w` (vLLM `causal_conv1d.py:1061`) into an FMA,
whereas the golden rounds the product first. The runner reruns the comparison with
fusion off and counts `fma.rn.f32` in the compiled kernel to test this.

## 5. Decode versus chunked prefill

The earlier 1024-step drift and prefill tables did not have a checked-in runner;
they are withdrawn as acceptance evidence. The existing 128-step synthetic test
only bounds its fixed seed and gate inputs. It does not establish a universal
plateau, prefill/decode equality, or any bound on model logits.

**Mixed decode-and-prefill steps (provider side).** The goldens target the
decode-only path (§1). In a step that also holds a prefill, vLLM 0.30.0 sends the
cached decode rows through `causal_conv1d_fn` and
`fused_sigmoid_gating_delta_rule_update` instead, so in those steps neither golden
matches what rollout runs. The two recurrent kernels do not agree bitwise. A
measurement on B200 with Qwen3-Next TP1 shapes, driving the real
`GDNAttentionMetadataBuilder.build()` and `_forward_core`, found the step's bf16
output equal but the fp32 state different in 75,146 of 524,288 elements (8.8e-08
relative). In 4 of 20 seeds the difference reached a bf16 output within the next 16
decode steps. This was measured outside this change and is not yet published; treat it
as a reported observation.

vllm-project/vllm#49827, open and unmerged, would route mixed-step *recurrent*
decodes through the packed kernel too, according to its description. It was
validated on Qwen3.5 (non-interleaved), H100, TP1, and it does not change the conv
path in mixed steps, which stays `causal_conv1d_fn`. Disabling
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` instead sends every decode row through
`fused_sigmoid_gating_delta_rule_update`, and the recurrent golden's target would
have to change.

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
| Backward for the recurrent step | no upstream backward exists, and a naive BPTT through a sequential recurrence is not batch-invariant — it needs its own design |
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
provider integration must resolve this; disabling the check is not L2 evidence.
