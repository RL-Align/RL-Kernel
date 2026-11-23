# Qwen3-8B acceptance (in progress)

Run these tools from the checkout root as modules, for example
`python -m tools.validation.models.qwen3_ppo.compare_fixed`. The batch scripts in
this directory take the source checkout as their first argument.

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
uses its BF16 optimizer's FP32 accumulation views. DP=1 is required, with ZeRO-0 for DeepSpeed FP32 and ZeRO-1 for DeepSpeed BF16 (DeepSpeed 0.19.7 gives `BF16_Optimizer` FP32 gradient buffers only at ZeRO-1, which partitions nothing at DP=1);
FP16 loss scaling remains rejected. Microbatch loss weights count valid action
tokens, so unequal completion lengths do not bias gradient accumulation.

The historical teacher-forced HF calibration completed earlier. It is
not sampler or PPO evidence. The formal fixed-rollout reference below collects
real vLLM output and independently computes HF PPO targets and updates. Its
artifacts and thresholds must be published before formal candidate execution;
they cannot be widened after observing candidates.

Formal matrix, still awaiting hardware qualification:

| Engine | Training precision | Training TP | Rollout |
|---|---|---|---|
| Megatron and DeepSpeed | FP32/BF16 | 2/4 (BF16 actors, DeepSpeed BF16 critic: 4) | Same frozen vLLM TP2 rollout |
| Megatron and DeepSpeed | FP32 tiny GPT | Actor 2, Critic 1 | Existing vLLM TP1 regression |

The matrix is sized for four 80-94 GB GPUs. Qwen3-8B with FP32 AdamW state needs about
131 GB at TP1, so TP1 is not part of it. The Megatron BF16 actor ran out of memory at TP2 on
a 93 GiB H100 (90.0 GiB allocated; 92.9 GiB per rank on B200), and the DeepSpeed BF16 actor
holds the same state; the DeepSpeed BF16 critic peaked at 98.9 GiB per rank at TP2 on B200.
These three rows run at TP4 only and have no TP2-vs-TP4 comparison.
`fixed_matrix.sh` holds the TP sizes of each row for the candidate and the memory probe.
DeepSpeed BF16 runs at ZeRO-1 with DP=1 (see the TP PPO README). The HF reference runs on one
GPU; `hf_ppo_reference` keeps the AdamW moments and the BF16 FP32 masters in host memory
between steps (`--optimizer-state cpu`, the default) and applies the same per-tensor
`torch.optim.AdamW` update on the GPU one parameter at a time, so the values match a
resident optimizer bit for bit. The frozen manifest records where the reference kept it.

The pretrained two-round vLLM topologies (`tests.distributed.tp.check_qwen3_ppo_rollout`,
actor 1 or 2, critic 1, vLLM 2 or 1 on dedicated GPUs) are not part of this matrix: a TP1
trainer does not fit an 80-94 GB GPU, and a TP2 actor plus a TP2 critic plus vLLM needs more
than four GPUs.
It requires frozen FP32-reference thresholds, uses real safetensors and sampler
logprobs, preserves reference/reward weights, and requires versions 0/1/2.
vLLM checks each rank's loaded QKV/gate-up/vocabulary shard against the actor
weights, rejects stale versions/missing rank confirmations, resets prefix cache
and checks cold/replay cache behavior. Rank memory and round timing are recorded.

Remaining: actual native checkpoint load validation; fixed-rollout HF controls;
FP32/BF16 TP2/TP4 fixed-rollout loss/gradient/update comparison; actual two-round
Megatron/DeepSpeed/vLLM TP2 and TP1 runs; GPU negative controls; immutable raw
artifacts and final threshold/acceptance report. CPU mapping and buffer tests
alone do not resolve PR #548's experiment gap.

The fixed-rollout qualification now has two separate jobs. `qualify_reference.sbatch`
uses actual vLLM TP2 sampling and independent HF scoring/GAE/PPO updates. Both HF
precisions repeat on the same immutable rollout and targets. `freeze_fixed_reference`
produces an acceptance manifest and per-tensor error bounds, which must be committed
before submitting `qualify_candidate.sbatch`. The candidate job checks all eight
engine/precision/TP combinations for both roles, then the tiny-GPT regression.
`probe_trainer_memory.sh` measures trainer peak memory on the frozen rollout first; a probe
writes no evidence tensors and needs no acceptance manifest. AdamW settings match in every precision. Evidence uses
actual FP32 optimizer gradients, measured clipping norms and parameter deltas.

The rule for max-absolute errors is four times the largest independent control difference
plus 1e-5 for scores, 1e-6 for losses/gradients/norms, or 1e-7 for parameter updates.
There are two controls. The repeat control runs the same HF update twice; on one GPU the two
runs are usually bit-identical, which alone would demand bitwise agreement between
implementations. The implementation control (`hf_ppo_reference --implementation-control`)
runs the same HF update with `sdpa` attention, every linear layer split into token, output
and input chunks summed in another order, and a vocabulary-chunked log-softmax: the same
math with a different reduction order, without any candidate code. Same-precision bounds
take the larger of the two per tensor. Cross-precision bounds compare the FP32 run with
both the BF16 repeat and the BF16 implementation control.
Relative L2 errors are also reported. Same-precision TP comparisons use independent
same-precision repeat bounds. No candidate result may establish or widen a bound.

FP32 parameter updates have a second way to pass (`compare_fixed.explained_update`). This rule
was added after the first H100 candidate run; see the amendment note below. AdamW's first step
moves a weight by about `lr * g_c / (|g_c| + eps)`, with `g_c` the clipped gradient. Where `|g_c|`
is near `eps`, that is linear in the gradient, with a slope of up to `lr / (eps * grad_norm)`: about
13 at this configuration's grad norm of 77. So a gradient difference that is well inside its own
bound can move one element's update past a bound frozen from controls that barely disturbed that
element.

An FP32 update tensor over its bound therefore still passes if, at every element, the two runs'
update difference equals the AdamW step difference recomputed in FP64 from their own recorded
gradients and grad norms, within two FP32 ulps of the weight. Two ulps covers two roundings per run
per step: decoupled decay, then the step. The recomputation uses the AdamW recurrence over both
updates, with the weights before each step taken from the checkpoint plus the run's own recorded
updates.

Gradients, norms, losses and scores keep their frozen bounds, and they are checked as before. BF16
rows (cross-precision bounds) and parameters missing from the checkpoint, such as the critic's
score head, keep the frozen bound only. Each comparison lists every tensor that used this path
under `explained_updates`.

Weight decay (`lr * weight_decay = 1e-7`) is about one FP32 ulp per step. Neither this rule nor
the 1e-7 update floor can resolve it elementwise.

Amendment note (2026-11-22): the H100 candidate at `19b62661` failed one of 1604 tensors for the
Megatron FP32 actor at TP4. It was `0/update/model.layers.17.post_attention_layernorm.weight`,
at 2.29 times its bound, while that tensor's gradient was at 0.30 times its own bound. A CPU
diagnosis of the frozen tensors explains it:
- Element 3486 has a clipped gradient of 0.46 `eps`.
- At TP4 its update differs by -7.749e-07; AdamW predicts -7.931e-07 from the gradient difference.
- The same element has the largest update difference at TP2 and in the implementation control.

The frozen bounds and the reference are unchanged. Only the candidate stage is rerun.
Large per-parameter evidence stays outside Git; the committed manifests anchor its
hashes. Missing inputs, tampering, invalid tensor inventories and skipped GPU stages
are failures, not completed qualification. The batch scripts also run without Slurm; the
reference needs about 256 GB of free host memory and about 1.4 TB of disk. Threshold publication is a stage boundary;
no formal PPO GPU acceptance is claimed until that candidate job actually passes.
