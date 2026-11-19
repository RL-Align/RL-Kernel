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
| Megatron and DeepSpeed | FP32/BF16 | 2/4 | Same frozen vLLM TP2 rollout |
| Megatron and DeepSpeed | FP32 tiny GPT | Actor 2, Critic 1 | Existing vLLM TP1 regression |

The matrix is sized for four 80-94 GB GPUs. Qwen3-8B with FP32 AdamW state needs about
131 GB at TP1 and about 82 GB at TP2 BF16, so TP1 is not part of it. The HF reference runs
on one GPU; `hf_ppo_reference` keeps the AdamW moments and the BF16 FP32 masters in host
memory between steps (`--optimizer-state cpu`, the default) and applies the same per-tensor
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
Large per-parameter evidence stays outside Git; the committed manifests anchor its
hashes. Missing inputs, tampering, invalid tensor inventories and skipped GPU stages
are failures, not completed qualification. The batch scripts also run without Slurm; the
reference needs about 256 GB of free host memory and about 1.4 TB of disk. Threshold publication is a stage boundary;
no formal PPO GPU acceptance is claimed until that candidate job actually passes.
