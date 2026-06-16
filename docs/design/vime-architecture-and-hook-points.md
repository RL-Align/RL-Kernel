# Vime Architecture and RL-Kernel Hook Points

This note maps the Vime code paths that matter for an RL-Kernel integration
exploration. It is scoped to issue #118: identify where rollout-side sampling
and log-prob data are produced, where training-side log-prob/loss/reduction
work runs, and which hook points are plausible for a later PoC.

## Scope

Vime's README describes Vime as an RL post-training framework built on slime. It
keeps slime's training stack and data-generation design, uses Megatron for
training, and uses vLLM plus vllm-router as the default rollout backend. This
matches the WS5 assumption: RL-Kernel should sit underneath Vime as an optional
operator layer, not replace Vime's scheduling or dataflow.

This document does not inject RL-Kernel code, change Vime, or claim real vLLM
and Megatron numerical alignment. It only records the architecture and candidate
hook inventory.

## Top-Level Flow

The synchronous and asynchronous drivers have the same high-level shape:

```text
train.py / train_async.py
  -> parse_args()
  -> create_placement_groups(args)
  -> create_rollout_manager(args, pgs["rollout"])
       -> RolloutManager(args, pg)
       -> start_rollout_servers(args, pg)
       -> VLLMEngine actors + vllm-router
       -> load rollout function from args.rollout_function_path
  -> create_training_models(args, pgs, rollout_manager)
       -> RayTrainGroup
       -> MegatronTrainRayActor
  -> actor_model.update_weights()
  -> rollout_manager.generate(rollout_id)
       -> rollout function
       -> Sample objects
       -> train-data dict
       -> DP split
  -> actor_model.async_train(rollout_id, rollout_data_ref)
       -> Megatron logprob / loss / reductions
  -> actor_model.update_weights()
       -> Megatron -> vLLM weight sync
```

Important source files in Vime:

| Area | Files | Role |
| --- | --- | --- |
| Driver loop | `train.py`, `train_async.py` | Owns rollout/training order and weight updates. |
| Ray placement | `vime/ray/placement_group.py` | Allocates actor, critic, and rollout GPU bundles. |
| Rollout manager | `vime/ray/rollout.py` | Starts vLLM routers/engines, calls rollout functions, converts samples to training data. |
| vLLM engine | `vime/backends/vllm_utils/vllm_engine.py` | Ray actor that launches or attaches to `vllm serve` and forwards vLLM control-plane calls. |
| vLLM args/config | `vime/backends/vllm_utils/arguments.py`, `vime/backends/vllm_utils/vllm_config.py` | Exposes `--vllm-*`, router, multi-model, and PD-disaggregation settings. |
| Rollout functions | `vime/rollout/vllm_rollout.py`, `vime/rollout/fully_async_rollout.py` | Builds sampling requests, calls vllm-router, records token/logprob data, runs reward logic. |
| Training actor | `vime/backends/megatron_utils/actor.py` | Converts rollout refs to GPU tensors, computes old/ref/current logprobs, runs training. |
| Training loss | `vime/backends/megatron_utils/loss.py`, `vime/utils/ppo_utils.py` | Computes selected logprobs, entropy, advantages, GRPO/PPO-style loss, KL, and masked reductions. |
| Train data | `vime/backends/megatron_utils/data.py`, `vime/utils/types.py` | Defines `Sample`, `RolloutBatch`, masks, packed token batches, and logging reductions. |

## Rollout-Side Map

### Engine and Router Lifecycle

`RolloutManager.__init__` loads the configured rollout function and, unless
`--debug-train-only` is set, starts rollout servers. `start_rollout_servers`
resolves a `VllmConfig`, starts one vllm-router per served model, starts
`VLLMEngine` Ray actors for the configured server groups, then exposes
`args.vllm_model_routers` for custom rollout functions.

`VLLMEngine` manages the HTTP serving process and control plane. It can launch
`vllm serve`, register node-0 workers with vllm-router, call vLLM sleep/wake
routes, flush cache, and update weights through vLLM's RLHF weight update
endpoints. The route used for rollout sampling is not called directly from
`VLLMEngine`; generation requests go through the rollout function and the
router address stored in `args.vllm_router_ip` / `args.vllm_router_port`.

### Sampling and Logprob Data

The default rollout entry is `vime.rollout.vllm_rollout.generate_rollout`,
configured by `--rollout-function-path`. `RolloutManager.generate` calls this
function, saves optional debug dumps, converts returned samples to train data,
and splits that data across DP ranks.

In `vime/rollout/vllm_rollout.py`:

- `generate_rollout` calls `generate_rollout_async`.
- `generate_rollout_async` builds prompt groups from the data source and calls
  `generate_and_rm_group`.
- `generate_and_rm_group` schedules one `generate_and_rm` task per sample.
- `generate_and_rm` optionally dispatches a user-provided
  `--custom-generate-function-path`; otherwise it calls the built-in `generate`.
- `generate` posts a token-based request through vllm-router to vLLM and asks
  for `logprobs: 1`.

The output path is token-based. The vLLM response is parsed into:

- `Sample.tokens`
- `Sample.response`
- `Sample.response_length`
- `Sample.loss_mask`
- `Sample.rollout_log_probs`
- `Sample.rollout_routed_experts` when routed-expert metadata is enabled
- trace / prefix-cache / weight-version metadata through `Sample.update_from_meta_info`

`RolloutManager._convert_samples_to_train_data` moves fields from `Sample`
objects into a train-data dictionary. If present, `rollout_log_probs` and
`rollout_routed_experts` are preserved and later passed to Megatron training.

### Rollout Extension Points

Vime already has two useful extension points on the rollout side:

1. `--rollout-function-path` replaces the whole rollout function. This is the
   broadest extension point and owns batching, filtering, reward, and sample
   conversion shape.
2. `--custom-generate-function-path` replaces only the per-sample generation
   function inside the default vLLM rollout loop. This is narrower and keeps
   Vime's default rollout batching, reward, and data-buffer behavior.

For a minimal RL-Kernel PoC, the narrower per-sample hook is safer if the goal
is to prove an operator is invoked without changing Vime's rollout scheduler.
However, replacing logprob computation on the rollout side may require access to
logits or selected-token logprob inputs that vLLM's HTTP response does not
expose. If only processed selected logprobs are returned, RL-Kernel can consume
or validate them but cannot recompute them without a deeper vLLM custom-op or
backend hook.

## Training-Side Map

### Rollout Data Ingestion

`MegatronTrainRayActor._get_rollout_data` receives a Ray `Box` from
`RolloutManager._split_train_data_by_dp`. It converts `tokens`, `loss_masks`,
and optional rollout-side fields to GPU tensors. It also handles context
parallel slicing for `rollout_log_probs` and `teacher_log_probs`, pads variable
length token streams, and carries `rollout_mask_sums` for whole-rollout loss
normalization.

`vime/backends/megatron_utils/data.py` turns those lists into Megatron batches.
It pads/slices tokens and masks, handles context-parallel layout, and logs
rollout metrics through DP/CP-aware gather and reduce helpers.

### Logprob and Loss

Training-side selected logprob is computed in two layers:

- `MegatronTrainRayActor.compute_log_prob` calls `get_log_probs_and_entropy`
  through a forward-only Megatron pass.
- `get_log_probs_and_entropy` runs the model, obtains logits, then calls
  `calculate_log_probs_and_entropy`.
- `calculate_log_probs_and_entropy` calls `compute_log_probs` in
  `vime/utils/ppo_utils.py`, including TP-aware distributed log-sum-exp when a
  tensor-parallel process group is present.

Loss computation is centered in `vime/backends/megatron_utils/loss.py`:

- `compute_advantages_and_returns` consumes rewards, logprobs, ref logprobs,
  values, `loss_masks`, and optional `rollout_log_probs`.
- `policy_loss_function` recomputes current logprobs, builds old logprobs from
  either Megatron or rollout-side values, computes PPO/GRPO-style ratios and
  KL, applies TIS/MIS hooks if configured, and reduces policy-gradient loss.
- `loss_function` dispatches policy, value, SFT, or custom loss logic and owns
  the final loss normalizer behavior.

Useful existing custom hooks:

- `--custom-loss-function-path`
- `--custom-pg-loss-reducer-function-path`
- `--custom-tis-function-path`

These are training-side integration points that can exercise RL-Kernel loss or
reduction helpers without replacing the whole Megatron actor. They are not
enough to replace Megatron's model forward or logits production.

## Hook-Point Matrix

| Operator / concern | Side | Current Vime callsite | Candidate RL-Kernel work | Required tensor contract | Hook stability | Notes / risk |
| --- | --- | --- | --- | --- | --- | --- |
| Per-sample generation wrapper | Rollout | `vime.rollout.vllm_rollout.generate_and_rm` via `--custom-generate-function-path` | Minimal PoC shim that invokes an RL-Kernel op before/after vLLM generation | `Sample`, `sampling_params`, generated tokens/logprobs | High for invocation proof | Best first PoC hook, but not a true vLLM kernel replacement. |
| Whole rollout function | Rollout | `RolloutManager.generate` -> `args.rollout_function_path` | Custom rollout path that records extra operator diagnostics | Data source, rollout id, `Sample` groups | Medium | Broad surface; easy to drift from native Vime behavior. |
| vLLM sampling | Rollout | vllm-router `/inference/v1/generate` request from `vime/rollout/vllm_rollout.py` | RL-Kernel sampling op only if vLLM exposes a custom-op/backend insertion point | token ids, sampling params, RNG/seed, logits/probs | Low from Vime only | Vime sees HTTP responses, not internal vLLM sampling kernels. Needs WS3 vLLM probing. |
| Rollout selected logprob | Rollout | `generate` parses response `logprobs` into `Sample.rollout_log_probs` | Compare, validate, or replace selected logprob if logits/inputs become available | selected tokens, logits or selected logprobs, mask, dtype | Low/Medium | HTTP path returns selected logprobs; recomputation likely needs deeper vLLM integration. |
| Routed experts replay | Rollout -> training | `Sample.rollout_routed_experts`; `MegatronTrainRayActor.fill_routing_replay` | Future routing-consistency helper | per-token expert ids, layer ids, token layout | Medium | Useful for MoE later; out of this month's scope. |
| Training selected logprob | Training | `MegatronTrainRayActor.compute_log_prob` -> `get_log_probs_and_entropy` -> `calculate_log_probs_and_entropy` | RL-Kernel batch-invariant / TP-invariant selected-logprob op | logits, target tokens, TP group, response spans, masks | Medium | Strong technical fit; must respect Megatron tensor/CP/TP layout. |
| Training GRPO/PPO loss | Training | `policy_loss_function` / `loss_function` | RL-Kernel fused GRPO/PPO loss or ratio/KL primitive | current/old/ref logprobs, rewards/advantages, loss masks, normalizer | Medium/High | Existing custom loss/reducer hooks make a staged integration possible. |
| Masked reductions | Training | `sum_of_sample_mean`, `distributed_masked_whiten`, loss reducers | RL-Kernel deterministic masked sum/mean/reduction helpers | per-token values, masks, rollout ids, DP/CP partition info | High | Good early training-side target because it is isolated and testable. |
| Weight sync | Train -> rollout | `actor_model.update_weights()`; `VLLMEngine.update_weights_*` | Usually no RL-Kernel op; only validate it does not introduce mismatch | HF/Megatron weight chunks, update version, vLLM update ids | Medium | Important for WS3 correctness, not a WS5 operator hook. |

## Recommended PoC Path

For #120, the lowest-risk way to prove Vime can invoke RL-Kernel is:

1. Use the native baseline from #117, preferably the Qwen2.5-0.5B
   `examples/fully_async` path or its CI-equivalent short test.
2. Keep Vime's default rollout/training dataflow unchanged.
3. Add a tiny opt-in shim at the per-sample rollout hook or training-side custom
   loss/reducer hook.
4. Emit a structured log or counter proving the RL-Kernel op was called.
5. Preserve Vime fallback behavior when RL-Kernel or its CUDA extension is not
   installed.

If the goal is specifically rollout-side `fused_logp`, first confirm with WS3
whether vLLM can expose logits or an internal custom-op insertion point in the
real rollout path. The Vime HTTP layer alone currently appears to receive
selected logprobs, not enough information for RL-Kernel to recompute logprobs.

## WS3 Handoff

Findings WS3 can reuse:

- Vime defaults to vLLM plus vllm-router for rollout and Megatron for training.
- The rollout request path enters Vime at `RolloutManager.generate`, then
  `vime.rollout.vllm_rollout.generate_rollout`, then HTTP requests through
  vllm-router.
- The Vime-level rollout extension points are `--rollout-function-path` and
  `--custom-generate-function-path`.
- Rollout logprobs are stored as `Sample.rollout_log_probs` and then passed to
  training as `rollout_data["rollout_log_probs"]`.
- Training recomputes logprobs in Megatron through `get_log_probs_and_entropy`
  and `compute_log_probs`.
- GRPO/PPO loss and masked reduction behavior lives in
  `vime/backends/megatron_utils/loss.py`.

Open questions for WS3:

- Which vLLM internal extension point can replace or instrument rollout-side
  selected-logprob computation without modifying Vime's macro dataflow?
- Can vLLM expose enough logits/token metadata for an RL-Kernel rollout-side
  logprob op through a public API, or is a vLLM custom op/backend required?
- Which vLLM `logprobs_mode` and temperature semantics should be considered
  frozen for train-vs-rollout comparison?
- How should the same fixed token sequence be replayed through vLLM and
  Megatron so sampling randomness is not confused with logprob drift?

## Non-Goals

- No RL-Kernel operator injection.
- No Vime, slime, vLLM, or Megatron source changes.
- No production integration path.
- No numerical claim that real vLLM and real Megatron are aligned.
- No performance benchmark.
