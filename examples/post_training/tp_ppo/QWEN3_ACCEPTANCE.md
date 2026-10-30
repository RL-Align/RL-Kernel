# Qwen3-8B acceptance (in progress)

The previous random-GPT FP32 four-GPU result remains a smoke test. No production
model, BF16 or vLLM TP2 acceptance is claimed yet.

Pinned checkpoint: Qwen/Qwen3-8B at b968826d9c46dd6066d109eabc6255188de91218.
The manifest identifies weights, config and tokenizer files. The native mapping
preserves GQA Q/K/V grouping, replicated Q/K RMSNorm, unscaled RoPE, gate/up
partition stride and vocabulary padding. Scalar heads use seed 144 and scale
0.01 on independently loaded pretrained backbones. This does not validate a
learned reward model or PPO convergence.

BF16 training clips the optimizer's actual FP32 gradient buffers. Megatron uses
FP32 masters and accumulates microbatches into FP32 before clipping; DeepSpeed
uses its BF16 optimizer's FP32 accumulation views. DP=1 and ZeRO=0 are required;
FP16 loss scaling remains rejected. Microbatch loss weights count valid action
tokens, so unequal completion lengths do not bias gradient accumulation.

The historical teacher-forced HF calibration completed in job 1106655. It is
not sampler or PPO evidence. The formal fixed-rollout reference below collects
real vLLM output and independently computes HF PPO targets and updates. Its
artifacts and thresholds must be published before formal candidate execution;
they cannot be widened after observing candidates.

Formal matrix, still awaiting hardware qualification:

| Engine | Training precision | Training TP | Rollout |
|---|---|---|---|
| Megatron and DeepSpeed | FP32/BF16 | 1/2 | Same frozen vLLM TP2 rollout |
| Megatron and DeepSpeed | BF16 | Actor 1, Critic 1 | Actual vLLM TP2, two PPO rounds |
| Megatron and DeepSpeed | BF16 | Actor 2, Critic 1 | Actual vLLM TP1, two PPO rounds |
| Megatron and DeepSpeed | FP32 tiny GPT | Actor 2, Critic 1 | Existing vLLM TP1 regression |

The pretrained four-GPU entry is `tests.distributed.tp.check_qwen3_ppo_rollout`.
It requires frozen FP32-reference thresholds, uses real safetensors and sampler
logprobs, preserves reference/reward weights, and requires versions 0/1/2.
vLLM checks each rank's loaded QKV/gate-up/vocabulary shard against the actor
weights, rejects stale versions/missing rank confirmations, resets prefix cache
and checks cold/replay cache behavior. Rank memory and round timing are recorded.

Remaining: actual native checkpoint load validation; fixed-rollout HF controls;
FP32/BF16 TP1/TP2 fixed-rollout loss/gradient/update comparison; actual two-round
Megatron/DeepSpeed/vLLM TP2 and TP1 runs; GPU negative controls; immutable raw
artifacts and final threshold/acceptance report. CPU mapping and buffer tests
alone do not resolve PR #548's experiment gap.

The fixed-rollout qualification now has two separate jobs. `qualify_reference.sbatch`
uses actual vLLM TP2 sampling and independent HF scoring/GAE/PPO updates. Both HF
precisions repeat on the same immutable rollout and targets. `freeze_fixed_reference`
produces an acceptance manifest and per-tensor error bounds, which must be committed
before submitting `qualify_candidate.sbatch`. The candidate job checks all eight
engine/precision/TP combinations for both roles, then runs both four-GPU topologies
and the tiny-GPT regression. AdamW settings match in every precision. Evidence uses
actual FP32 optimizer gradients, measured clipping norms and parameter deltas.

The rule for max-absolute errors is four times the independent control difference
plus 1e-5 for scores, 1e-6 for losses/gradients/norms, or 1e-7 for parameter updates.
Relative L2 errors are also reported. Same-precision TP comparisons use independent
same-precision repeat bounds. No candidate result may establish or widen a bound.
Large per-parameter evidence stays outside Git; the committed manifests anchor its
hashes. Missing inputs, tampering, invalid tensor inventories and skipped GPU stages
are failures, not completed qualification. All jobs use `overflow`, `test`, 4 B200,
512 GB host memory and a 24-hour limit. Threshold publication is a stage boundary;
no formal PPO GPU acceptance is claimed until that candidate job actually passes.
