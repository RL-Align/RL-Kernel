# WS1 C6 — Qwen3-Next Gated DeltaNet recurrent replay

Design notes for RFC #428 work item C6 (GDN recurrent response replay) on the CUDA
track. Measured on 2× B200 (sm_100), torch 2.13.0+cu130, vllm 0.30.0,
transformers 5.17.0.

Claim level reached: **L0 repeatable, L1 batch-invariant**. L2 is not claimed.

## 1. Which provider a rollout decode actually takes

vLLM 0.30.0 has three GDN decode paths, selected by env defaults rather than by the
model:

| condition | path |
|---|---|
| `VLLM_GDN_DECODE_KERNEL="cuda"` (default) + MTP | `torch.ops._C.fused_gdn_decode_post_conv_mtp` — conv, recurrence and the gated norm fused |
| `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` (default), decode-only, non-spec | `fused_recurrent_gated_delta_rule_packed_decode` |
| otherwise | `fused_sigmoid_gating_delta_rule_update` |

A pure RL rollout decode — N sequences each emitting one token, no speculative
decoding — returns early at `qwen_gdn_linear_attn.py:1295-1307` into the **packed**
path. The golden targets that one. Aligning against the sigmoid-gating kernel would
validate a path production does not take.

`tests/check_qwen3_next_norm_providers.py` asserts those two env defaults, so a vLLM
bump that flips either fails loudly rather than silently re-pointing the claim.

## 2. Three places the kernel and the HF model disagree

The golden is transcribed from the kernel, not from `modeling_qwen3_next.py`:

1. **The gating is fused.** `beta = sigmoid(b)` and
   `g = -exp(A_log) * softplus(a + dt_bias)` are computed inside the Triton kernel,
   with a `softplus` threshold branch at 20. HF computes them as separate PyTorch
   ops — a different rounding path.
2. **No `repeat_interleave`.** The kernel indexes `i_h = i_hv // (HV // H)`, so q/k
   stay at 16 heads while v has 32. HF materializes the repeat.
3. **The QK norm is an L2 norm over a plain sum**, `x / sqrt(sum(x*x) + 1e-6)` — not
   an RMSNorm, not `F.normalize`, and dividing by `sqrt` rather than multiplying by
   `rsqrt`, which differs in the last bit. `scale` is applied to `q` *after* the
   norm; `k` is never scaled.

A fourth, checked and found **not** to be a divergence: prefill passes
`use_qk_l2norm_in_kernel=False` only because `fused_post_conv_prep(apply_l2norm=True)`
already normalized q/k. Both paths normalize exactly once.

## 3. State ABI

Mirrored rather than reinvented:

- recurrent state `[num_blocks, HV, V, K]`, V-major, addressed by `ssm_state_indices`
- conv state `[num_blocks, dim, width-1]`, layout chosen by the global
  `is_conv_state_dim_first()`; the golden takes it as an argument and both are tested
- `NULL_BLOCK_ID` (index `<= 0`) means skip: zeros out, block untouched
- the accumulator is fp32 for the whole step; the store rounds to the state tensor's
  dtype, which `FUSED_GDN_STATE_DTYPES` allows to be fp32 **or** bf16
- `causal_conv1d_update` casts `x` to the cache dtype before anything else

Contractions run in the repo's fixed 32-wide chunk order rather than `torch.matmul`,
whose reduction order is unspecified. That is what the L1 claim rests on, and also
why the golden is not bitwise against the kernel's tree.

## 4. Agreement with the provider

Qwen3-Next dims (H=16, HV=32, K=V=128), `use_qk_l2norm_in_kernel=True`:

| | max\|diff\| out | max\|diff\| state |
|---|---|---|
| fp32 state, B=1..64 | 1.5e-08 .. 6.1e-05 | ≤ 3.0e-07 |
| bf16 state, B=1..64 | 3.7e-09 .. 3.1e-05 | ≤ 2.0e-03 |

Causal conv uses sequential FP32 accumulation **starting from bias**, with
products first rounded to the operand dtype. The previous BF16 path incorrectly
promoted both operands to FP32; its disagreements were not limited to one BF16
ULP. Cancellation and bias-order CPU tests now cover these errors. Provider
comparisons preserve the existing absolute bounds and additionally limit BF16
mismatches to 32 elements on the checked fixtures. This is not a bitwise claim.

## 5. Decode versus chunked prefill

The earlier 1024-step drift and prefill tables did not have a checked-in runner;
they are withdrawn as acceptance evidence. The existing 128-step synthetic test
only bounds its fixed seed and gate inputs. It does not establish a universal
plateau, prefill/decode equality, or any bound on model logits.

The strict profile uses FP32 recurrent state. BF16 state remains a differential
experiment. Full checkpoint prefill, response replay, optimizer updates and
reload all remain required before L2 can pass.

Cache indices must be int32/int64, on the input device, and positive active
indices must be unique and in range. Nonpositive sentinels may repeat. The golden
validates before any cache update; raw provider calls remain the caller's
responsibility (the provider's default does not bounds-check).

## 6. Current boundary

Not covered, with reasons:

| deferred | why |
|---|---|
| Speculative decode / MTP | `fused_gdn_decode_post_conv_mtp` fuses conv, recurrence and the gated norm; validating it needs a draft model |
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
