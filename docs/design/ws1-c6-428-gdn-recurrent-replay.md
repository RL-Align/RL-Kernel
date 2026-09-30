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

Causal conv (`conv_dim=8192`, `W=4`, B ≤ 64): the **rolled state is bitwise exact in
every configuration**. With an fp32 cache the output is bitwise but for a handful of
elements (15 of 524288 at B=64); with a bf16 cache it agrees on ~63%, each
disagreement exactly one bf16 ULP. The conv taps accumulate **sequentially**,
`acc = acc + win[t] * w[t]` from zero — a tree sum over the same four terms does not
reproduce the provider, an FMA does not either.

Open: where the provider rounds in the bf16-conv-cache case is not reproduced.
Recorded rather than guessed at.

## 5. Decode versus chunked prefill

The quantity RFC #428 §4.2 is about. Single sequence, zero initial state:

| T | max\|diff\|, fp32 state | max\|diff\|, bf16 state |
|---|---|---|
| 8 | 3.66e-04 (7.0e-03 rel) | 5.49e-04 (1.0e-02 rel) |
| 64 | 4.88e-04 (6.6e-03 rel) | 5.49e-04 (7.5e-03 rel) |
| 256 | 3.66e-04 (5.0e-03 rel) | 3.66e-04 (5.0e-03 rel) |

**The gap is ~0.5–1% relative and flat in T** — smaller at T=256 than at T=8 — and the
state dtype barely moves it.

The reason is that the recurrence is **contracting**: per-step decay `exp(g)` averages
~0.47 (max 0.996), so old rounding error is forgotten at roughly the rate old signal
is. Comparing an fp32 state against a bf16 one over 1024 steps shows the same shape:

| step | relative \|d\| state | relative \|d\| out |
|---|---|---|
| 1 | 2.67e-03 | 3.47e-03 |
| 64 | 1.52e-02 | 1.07e-02 |
| 256 | 2.02e-02 | 1.09e-02 |
| 1024 | 2.47e-02 | 1.57e-02 |

A 16× longer run past step 64 grows the drift only 1.6×; it saturates at ~2% relative
on the state and ~1–1.8% on the output.

So the two paths are not bitwise, and ~1% relative on logits is still material for RL
importance ratios, but this is a bounded, characterizable error rather than a
divergence. An fp32 recurrent state remains the right choice for exactness work — the
reason is the 2% plateau, not a blow-up.

Two related constraints: the chunked prefill kernel refuses fp32 q/k/v outright
(`chunk.py:213`), so the prefill side is bf16-only regardless; and
`causal_conv1d_update` does not bounds-check `conv_state_indices` under its default
`validate_data=False` — an index past the cache is an out-of-bounds write, not an
error.

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

## 7. Questions for the maintainers

1. Is a **bf16 recurrent state** a supported configuration? The answer decides whether
   the 2% plateau is a finding or a non-issue.
2. Is MTP / speculative decode in scope for WS1? If so,
   `fused_gdn_decode_post_conv_mtp` becomes the primary provider for both the norm and
   the recurrence, and the largest deferral above reopens.
3. For the CUDA strict profile, is the single source of truth **vLLM's** arithmetic or
   **PyTorch eager**? §1 item 1 asks for one provider on both sides but §2.1 does not
   say which, and the two differ by ~6e-2 in bf16.
