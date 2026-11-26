# Companion patches for the native DSV4 recipe

These are changes to other repositories that the recipe needs. Apply each one in a clean
checkout of the pinned commit with `git apply <patch>`.

| Patch | Applies to | Needed for |
| --- | --- | --- |
| `verl-mi35x-w4a16.patch` | verl, pinned commit of the recipe | gfx950 (MI350X/MI355X): selects vLLM's unfused W4A16 MoE implementation while the native rollout loads. See `docs/usage/dsv4-mi35x-validation.md`. |
| `verl-nlayer-pp.patch` | verl `13ffe5472daa092b6482e1d1d9b36909b0d679af` (`dsv4-rocm-multinode`) | Any layer count and pipeline split (`RLK_DSV4_PP_SPLIT`, `RLK_ROLLOUT_PP_SPLIT`). Not needed for the default four-layer 3/1 recipe. |
| `vllm-moe-combine-order.patch` | vLLM `c07d58dc82d75f7d1a947b3264b460384ec3e18d` | gfx950: bit-identical actor/rollout scores after weight updates and beyond 13 layers. |

## `verl-nlayer-pp.patch`

The verl side of the configurable split. `verl/experimental/rl_kernel` assumed a four-layer
checkpoint, two actor ranks and two rollout stages in its preflight, configuration check,
placement check, weight-sync acknowledgments and final stress readback. The patch takes
these counts from the same variables as RL-Kernel:

- `RLK_DSV4_PP_SPLIT="n0,n1,..."`: original layers per actor pipeline stage (default `3,1`).
  The checkpoint must have `sum(split)` layers.
- `RLK_ROLLOUT_PP_SPLIT`: the rollout split when it differs from the actor's, e.g. `43` for
  one rollout GPU. It is also passed to vLLM as `VLLM_PP_LAYER_PARTITION`.

Set them in the environment of the driver and of every Ray worker. The patch also reads
`RLK_ROLLOUT_TP`, but rollout TP > 1 does not work (RL-Kernel rejects it) and should be left
unset. It is independent of `verl-mi35x-w4a16.patch`; both were applied in the runs below.

## `vllm-moe-combine-order.patch`

vLLM's `UnfusedOAITritonExperts` (the MoE implementation selected on gfx950) adds the routed
expert outputs of each token into an FP32 accumulator with `index_add_`. With repeated
indices the GPU adds a token's rows in no defined order. The Megatron actor adds them in
ascending expert id. The two orders give the same BF16 result except when the FP32 sum sits
on a BF16 rounding boundary; then one element of one token differs by one BF16 ulp, and the
difference spreads through the later layers.

The patch sorts the rows by token with a stable sort and adds the top-k rows one by one, so
every token is summed in ascending expert id, the same order as the actor. The scatter-add
is kept when some (token, top-k) slots are invalid.

Measured on MI355X with exclusive GPUs on the recipe's short text task (responses of at most
16 tokens), strict bitwise comparison at every weight version (runs bit-identical through
all updates / runs):

| Original layers | Without the patch | With the patch |
| --- | --- | --- |
| 14 | 0/1 | 2/2 |
| 15 | 2/3 | 1/1 |
| 21 | 0/2 | 5/5 |
| 43 (all layers, no MTP) | not run | 4/4 |

Runs with 4, 7, 11 and 13 layers were bit-identical without the patch. The four-layer
recipe at its fixed weights and audit inputs does not hit a rounding boundary, so it passes
either way.

Not fixed by this patch: one 43-layer run on GSM8K (16 prompts x 2 samples, responses up to
256 tokens, rollout `max_num_seqs=8`) failed the first training-batch comparison at the
original weights. 8 of the 32 sequences differed (630 of 3951 tokens, |diff| up to 0.84);
the other 24 were bit-identical. The cause is not known yet.
