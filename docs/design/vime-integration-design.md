# RL-Kernel to Vime Integration Design

This document defines a non-intrusive integration path for using RL-Kernel
operators under Vime. It builds on the hook inventory in
`docs/design/vime-architecture-and-hook-points.md` and is scoped to issue
#119: define the positioning, contracts, activation shape, change surface, and
fallback policy before broad implementation work starts.

## Goals

- Keep Vime's rollout, training, Ray placement, and weight-sync lifecycle in
  charge.
- Add RL-Kernel as an optional operator-level layer under selected Vime
  callsites.
- Preserve Vime's pure native behavior when RL-Kernel is not enabled, not
  installed, or not usable on the current hardware.
- Make the first PoC small enough to validate with Vime's existing custom
  function hooks.
- Define the contracts that a later production integration must satisfy before
  claiming train-inference consistency.

## Issue Checklist Mapping

This document covers the planned #119 PR items. The exact checklist phrases are
kept here intentionally so reviewers can map the design back to the issue:

- Integration design doc covering insertion points, expected tensor/API contracts, lifecycle, ownership boundaries, and rollout/training-side responsibilities.
- Non-intrusive activation path: custom op, optional backend, and config flag
  shape, with pure Vime inference and native RL paths unaffected when disabled.
- Change-surface estimate: what can live entirely in RL-Kernel, what requires a Vime/slime adapter, and what would need upstream vLLM/vllm-router changes.
- Positioning statement: RL-Kernel provides operator-level consistency beneath
  Vime's framework-level alignment; complementary rather than competing.
- Risk and fallback section: private API reliance, version pinning, missing extension points, unsupported hardware, and how the PoC should fail safely.

## Non-Goals

- No production integration in this issue.
- No Vime, slime, vLLM, or Megatron upstream PR in this issue.
- No claim that this closes real-engine train-inference consistency.
- No vLLM internal kernel replacement from Vime's HTTP layer alone.
- No replacement of Vime's scheduler, rollout buffer, data source, Megatron
  actor, or weight-update path.

## Positioning

Vime is the framework-level RL post-training orchestrator. It owns the macro
dataflow:

```text
prompt/data source
  -> rollout function
  -> vLLM/vllm-router generation
  -> Sample objects
  -> train-data conversion
  -> Megatron logprob/loss/reduction
  -> weight update back to rollout engines
```

RL-Kernel should not compete with that layer. Its role is to provide
operator-level consistency and performance beneath Vime:

- selected-token logprob implementations;
- GRPO/PPO loss or loss sub-primitives;
- deterministic masked reductions;
- rollout-side sampling/logprob kernels only after WS3 exposes an engine-level
  insertion point with the required tensors.

In short: Vime aligns the framework workflow; RL-Kernel aligns and accelerates
the kernels used inside that workflow.

## Activation Model

The integration should be opt-in and fail-safe.

Recommended future flag shape:

| Flag | Values | Default | Purpose |
| --- | --- | --- | --- |
| `--rl-kernel-mode` | `off`, `observe`, `train_ops`, `rollout_ops` | `off` | Top-level enablement gate. |
| `--rl-kernel-logp` | `auto`, `native`, `disabled` | `auto` when enabled | Select selected-logprob backend through RL-Kernel dispatch. |
| `--rl-kernel-loss` | `auto`, `native`, `disabled` | `disabled` initially | Select fused GRPO/PPO loss or sub-primitives. |
| `--rl-kernel-strict` | boolean | `False` | Raise on RL-Kernel import/backend failure instead of falling back. |
| `--rl-kernel-metrics-prefix` | string | `rl_kernel` | Prefix for invocation, fallback, and drift metrics. |

For the minimal PoC in #120, these do not need to be added to Vime. The PoC can
use a Vime custom function path and an environment variable or tiny shim-local
setting. A production-quality integration should expose explicit flags so the
native path remains the default and CI can test both native and enabled modes.

Fallback rules:

1. `off` means Vime runs exactly as it does today.
2. `observe` may import RL-Kernel and emit diagnostics, but must not change
   tensors consumed by training.
3. `train_ops` may replace training-side operator calls only after tensor
   contracts pass validation.
4. `rollout_ops` is blocked for selected-logprob recomputation until vLLM or
   vllm-router exposes logits or an internal custom-op insertion point.
5. In non-strict mode, any import, shape, dtype, device, or backend failure logs
   one warning and falls back to native Vime behavior for that callsite.

## Insertion Points and Ownership

The hook inventory in #118 identified several Vime-level extension points. The
integration should use them in this order.

| Stage | Vime hook or callsite | Integration role | Recommendation |
| --- | --- | --- | --- |
| Rollout invocation proof | `--custom-generate-function-path` in `vime.rollout.vllm_rollout.generate_and_rm` | Prove Vime can call RL-Kernel without replacing rollout orchestration. | Use for #120 PoC. |
| Rollout diagnostics | `--rollout-data-postprocess-path` after rollout data reaches the actor | Add validation, counters, or metadata before training. | Useful in observe mode. |
| Train-data conversion | `--custom-convert-samples-to-train-data-path` in `RolloutManager._convert_samples_to_train_data` | Preserve or add optional RL-Kernel metadata. | Use only if default fields are insufficient. |
| Training selected logprob | `get_log_probs_and_entropy` / `calculate_log_probs_and_entropy` / `compute_log_probs` | Replace selected-token logprob extraction from Megatron logits. | Best production operator target. |
| Training loss | `--custom-loss-function-path` with `--loss-type custom_loss` | Exercise RL-Kernel loss while preserving Vime's actor loop. | Good staged integration for isolated loss tests. |
| PG loss reducer | `--custom-pg-loss-reducer-function-path` | Try deterministic masked reduction or custom normalization. | Low-risk reducer target. |
| Megatron lifecycle hooks | `--custom-megatron-init-path`, `--custom-megatron-before-log-prob-hook-path`, `--custom-megatron-before-train-step-hook-path` | Initialize RL-Kernel, validate device, or emit metrics. | Auxiliary hooks, not enough to replace logprob alone. |
| vLLM internal sampling/logprob | Below vllm-router HTTP generation | Replace rollout-side kernels. | Needs WS3/vLLM insertion point. |

The first real operator integration should be training-side selected logprob or
masked reduction. These are closer to the tensors RL-Kernel already accepts and
do not require changing Vime's macro dataflow.

Ownership boundary:

- Vime owns scheduling, Ray actors, rollout buffering, reward execution,
  train-data conversion, Megatron model forward, optimizer steps, checkpointing,
  and weight sync.
- RL-Kernel owns operator implementations, backend dispatch, local fallback
  behavior, and operator-level reference tests.
- A Vime adapter owns tensor layout conversion between Vime's packed/list
  structures and RL-Kernel's operator contracts.

## Tensor and API Contracts

### Rollout Sample Contract

Vime's rollout path returns `Sample` objects. Any RL-Kernel wrapper that touches
rollout data must preserve these fields:

| Field | Contract |
| --- | --- |
| `tokens` | Prompt plus response token ids as `list[int]`. |
| `response_length` | Number of response tokens. |
| `loss_mask` | Optional `list[int]`; if present, length must equal `response_length`. |
| `rollout_log_probs` | Optional `list[float]`; when present, length must equal `response_length`. |
| `reward` | Must be populated before training unless a later hook intentionally fills it. |
| `status` | Must remain a `Sample.Status` value. |
| `rollout_id` | Shared by sibling samples from one rollout execution when fan-out is used. |

The default vLLM rollout asks vllm-router for `logprobs: 1` and parses selected
response logprobs into `Sample.rollout_log_probs`. That is sufficient for
comparison and mismatch metrics, but not sufficient for recomputing selected
logprobs with RL-Kernel because RL-Kernel's logprob operator requires logits.

### Train-Data Contract

`RolloutManager._convert_samples_to_train_data` produces the dictionary that is
later split by DP rank. RL-Kernel-aware code must preserve the default required
keys:

```text
tokens
response_lengths
rewards
raw_reward
truncated
sample_indices
rollout_ids
loss_masks
rollout_mask_sums
```

Optional fields that matter for train-inference comparison:

```text
rollout_log_probs
teacher_log_probs
rollout_routed_experts
metadata
multimodal_train_inputs
```

`rollout_mask_sums` is important for Vime's per-rollout normalization. A custom
converter must not drop or recompute it with only the current micro-batch, since
Vime may split sibling samples across micro-batches.

### Training Logprob Contract

RL-Kernel's existing logprob operator accepts:

```python
logp_op = kernel_registry.get_op("logp")
selected_logps = logp_op(logits, token_ids)
```

Contract for a Vime adapter:

| Tensor | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `logits` | `[N, V]` or reshapeable to that form | fp32/bf16 depending on backend | Must be contiguous for fused CUDA paths. |
| `token_ids` | `[N]` or same leading shape as `logits` | int64/int32 accepted by wrapper | Target token ids aligned to response logits. |
| output | `[N]` or original leading shape | fp32 preferred for comparison | One selected logprob per response token. |

The adapter must handle Vime's layouts before calling RL-Kernel:

- THD and BSHD packing;
- context parallel slicing;
- tensor parallel vocabulary partitioning;
- optional all-gather-CP mode;
- temperature scaling with `args.rollout_temperature`;
- response-only token alignment where logits at position `t - 1` score token
  `t`.

The existing RL-Kernel logprob op does not by itself replace Vime's
tensor-parallel distributed log-sum-exp semantics. A production integration
must either add a TP-aware RL-Kernel entry point or call RL-Kernel only after
Vime has materialized full-vocab logits for the local response rows.

### Loss and Reduction Contract

RL-Kernel's current GRPO loss operator is shaped around dense `[B, T]` tensors:

```python
loss, policy_loss, kl = grpo_loss(
    current_logps,
    old_logps,
    ref_logps,
    rewards,
    completion_mask,
    clip_eps=...,
    beta=...,
    samples_per_prompt=...,
)
```

Vime's policy loss path is list/packed oriented:

```text
batch["advantages"]       -> list[Tensor], response local
batch["log_probs"]        -> list[Tensor], old train logprobs
batch["rollout_log_probs"] -> list[Tensor], optional rollout logprobs
batch["ref_log_probs"]    -> list[Tensor], optional ref logprobs
batch["loss_masks"]       -> list[Tensor], full response masks
sum_of_sample_mean        -> callable normalizer
```

Therefore a direct fused-loss replacement must first define a packing adapter
that converts Vime's per-sample lists into dense or flat tensors while
preserving:

- variable response lengths;
- masked tokens;
- per-rollout denominators from `rollout_mask_sums`;
- CP local chunks and all-gather behavior;
- GRPO/GSPO/PPO differences;
- optional KL loss, entropy loss, OPSM, TIS/MIS, OPD, and custom reducers.

The safer staged path is to integrate smaller RL-Kernel primitives first:
selected logprob, ratio/KL, or masked reductions. A full `policy_loss_function`
replacement should come later, after parity tests cover the active Vime flags.

## Lifecycle

### Import and Initialization

An adapter should lazily import RL-Kernel at the first enabled callsite:

```python
try:
    from rl_engine.kernels.registry import kernel_registry
except Exception as exc:
    fallback_or_raise(exc)
```

Do not import or initialize CUDA extensions on Vime's driver process unless the
operator will actually run there. Most training-side operators should initialize
inside `MegatronTrainRayActor` processes after the CUDA device and distributed
groups are ready.

The `--custom-megatron-init-path` hook is a reasonable place to validate
installation and log selected RL-Kernel backends. It should not force the
integration to be active unless the explicit RL-Kernel mode is enabled.

### Rollout Loop

The rollout loop should remain:

```text
RolloutManager.generate
  -> configured rollout function
  -> Sample objects
  -> default logging
  -> _convert_samples_to_train_data
  -> _split_train_data_by_dp
```

An observe-mode PoC can wrap `--custom-generate-function-path`, call a cheap
RL-Kernel fallback op on synthetic or available tensors, and record a counter in
`sample.metadata`. It must return a valid `Sample` and leave generated tokens,
logprobs, rewards, and loss masks unchanged.

### Training Loop

The training loop should remain:

```text
MegatronTrainRayActor._get_rollout_data
  -> compute old/ref/current logprobs as configured
  -> compute_advantages_and_returns
  -> optional rollout_data_postprocess
  -> train(...)
  -> loss_function(...)
```

Training-side RL-Kernel work should run only inside actor processes and only on
tensors already owned by the current rank. The adapter must not add hidden
collectives beyond the collectives Vime already performs unless the contract
states them explicitly.

### Weight Sync

RL-Kernel should not participate in Vime's initial weight push or periodic
Megatron-to-vLLM update. It may report diagnostics that compare rollout
logprobs and training logprobs, but the ownership of weight update remains:

```text
actor_model.update_weights()
  -> Megatron weight iterator/updater
  -> VLLMEngine update endpoints
```

## Change Surface

| Area | Can live in RL-Kernel | Needs Vime adapter | Needs upstream vLLM/vllm-router |
| --- | --- | --- | --- |
| Operator dispatch and fallback | Yes | No | No |
| Selected-logprob CUDA/Triton/PyTorch kernels | Yes | Thin callsite adapter | No for training side |
| Vime tensor packing/unpacking | No | Yes | No |
| Vime CLI flags for RL-Kernel enablement | No | Yes | No |
| Training selected-logprob replacement | Mostly | Yes | No |
| Masked reduction helpers | Mostly | Yes | No |
| Custom loss PoC through `--custom-loss-function-path` | Mostly | User/Vime plugin shim | No |
| Rollout invocation proof | Mostly | User/Vime plugin shim | No |
| Rollout-side true logprob recomputation | Partly | Yes | Yes, unless vLLM exposes logits through a supported API |
| vLLM sampling kernel replacement | Partly | Maybe | Yes |
| Weight update | No | Existing Vime path | Existing vLLM endpoints |

This separation keeps the core RL-Kernel repo focused on reusable operators and
small adapters, while any framework-specific lifecycle wiring stays in Vime or a
Vime plugin.

## Recommended PoC for #120

Start with an invocation proof, not a numerical replacement:

1. Use the #117 baseline, preferably
   `examples/fully_async/run-qwen2.5-0.5B-fully_async.sh` or the corresponding
   short CI test.
2. Add a tiny `--custom-generate-function-path` shim that imports RL-Kernel,
   obtains one operator through `kernel_registry`, and increments a structured
   counter.
3. Return the same `Sample` fields Vime would have returned without the shim.
4. Log `rl_kernel/invoked`, `rl_kernel/backend`, and `rl_kernel/fallback` in
   metadata or rollout metrics.
5. Run once with RL-Kernel installed and once with RL-Kernel import disabled to
   prove fallback does not break native Vime.

After that, the first operator-semantic PoC should be training-side selected
logprob on a constrained layout. It should compare RL-Kernel output with Vime's
native `compute_log_probs` on the same fixed logits and target tokens before it
is used for loss.

## Validation Plan

Minimum checks before enabling any real operator path:

- import/fallback test for missing RL-Kernel;
- backend selection test for CPU/PyTorch fallback;
- shape validation for `logits`, `token_ids`, `loss_masks`, and
  `response_lengths`;
- numerical parity with Vime's native selected-logprob reference on fixed
  logits;
- CP/TP layout test or an explicit guard that disables the integration when the
  unsupported layout is active;
- native Vime run with RL-Kernel mode off;
- enabled-mode run that proves the intended callsite was invoked;
- debug dump comparison using Vime's `save_debug_rollout_data` and
  `save_debug_train_data` when available.

For rollout-side claims, add WS3 validation first:

- exact vLLM `logprobs_mode`;
- temperature semantics;
- fixed token replay to remove sampling randomness;
- availability of logits or an internal vLLM custom-op insertion point.

## Risks and Fallbacks

| Risk | Impact | Fallback |
| --- | --- | --- |
| RL-Kernel not installed in Vime workers | Import failure in Ray actor or custom hook | Non-strict mode logs once and uses native Vime path. |
| CUDA extension unavailable or unsupported GPU | Runtime backend failure | Use RL-Kernel PyTorch fallback or native Vime path. |
| Vime TP/CP layout not supported by RL-Kernel op | Incorrect logprobs or shape errors | Disable RL-Kernel for that callsite until adapter coverage exists. |
| vLLM HTTP response lacks logits | Cannot recompute rollout logprobs | Treat rollout path as observe/compare only; wait for WS3. |
| Private vLLM/vllm-router API reliance | Breakage across versions | Pin tested versions and isolate upstream-specific code in a small adapter. |
| Loss replacement misses Vime options | Training behavior regression | Start with sub-primitives; require parity tests for each enabled flag. |
| Silent numerical drift | Hard-to-debug train instability | Emit drift metrics and support strict mode for CI. |

## Open Questions

- Should the production integration live in Vime, in RL-Kernel as a Vime plugin,
  or in a separate adapter package?
- What exact TP-aware logprob contract should RL-Kernel expose for Megatron's
  sharded vocabulary path?
- Which Vime layouts are in scope for the first semantic operator PoC:
  `qkv_format=thd`, `bshd`, CP=1 only, or allgather-CP?
- Can vLLM expose logits or selected-logprob custom-op hooks without expanding
  the rollout HTTP payload?
- What metric threshold should define acceptable train-vs-rollout logprob drift
  for WS5 reporting?
