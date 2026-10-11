# TP PPO with real Megatron Core, DeepSpeed and vLLM (issue #44 follow-up)

Checks ran on NVIDIA B200 GPUs (driver 580.126.20). Shared-group and
independent-role checks used two GPUs, with processes sharing GPUs across roles;
the vLLM reports below use four dedicated GPUs. The trainer environment was
Python 3.11.15 with PyTorch 2.9.1+cu130, Megatron Core
0.15.3, DeepSpeed 0.19.7 and NumPy 1.26.4. vLLM 0.15.1 ran in a separate Python
3.11 environment with PyTorch 2.9.1+cu128. The optional framework packages were
an isolated `PYTHONPATH` overlay (`examples/post_training/tp_ppo/requirements-*.txt`).
The project's own environment and precision settings were unchanged.

No framework mocks, framework source patches, downloaded checkpoints, Apex or
Transformer Engine were used. Models are randomly initialised two-layer Megatron
`GPTModel`s, so these are correctness checks, not performance measurements.

## 1. Shared-group numerical checks

`tests/distributed/tp/check_tp_ppo_frameworks.py` builds an Actor, a Critic, and
frozen Reward and reference models:
- **Model:** hidden 32, 4 heads, FFN 64, vocabulary 32, FP32, dropout off.
- **Parallelism:** DP=PP=1.
- **DeepSpeed:** ZeRO-0, accumulation 1, SGD with learning rate 0.01.
- **Clipping:** PPO clips the TP-global gradient norm at 1.0, and DeepSpeed's own clipping is disabled.

Each run checks the following, reconstructing full tensors from Megatron's
partition metadata:
- Rewards are equal under left and right padding.
- Initial weights, Critic values, gradients, the clipping norm and an SGD update are compared.
- Actor vocabulary-parallel logprobs and gradients are compared.
- Two `PPOTrainingWorker` updates per trainable model.
- The frozen models are exactly unchanged.

Maximum absolute error against native Megatron TP=1
([reports](frameworks/)):

| Quantity | Megatron TP=2 | DeepSpeed TP=1 | DeepSpeed TP=2 |
| --- | ---: | ---: | ---: |
| Initial parameters | 0 | 0 | 0 |
| Critic values / sequence rewards | 9.313e-10 | 0 | 9.313e-10 |
| Critic gradients | 1.164e-10 | 0 | 1.164e-10 |
| Parameters after initial Critic SGD step | 4.657e-10 | 0 | 4.657e-10 |
| Actor selected-token logprobs | 0 | 0 | 0 |
| Actor gradients in final PPO update | 1.192e-7 | 0 | 1.192e-7 |
| Actor parameters after PPO | 3.725e-9 | 0 | 3.725e-9 |
| Critic parameters after PPO | 4.657e-10 | 0 | 4.657e-10 |
| TP-global gradient norm | 1.649e-11 | 0 | 1.649e-11 |

The assertions use `atol=2e-5, rtol=2e-4` on every reconstructed tensor. These
are numerical, not bitwise, guarantees across TP sizes.

## 2. Independent-role PPO rollout scheduling

`tests/distributed/tp/check_tp_ppo_rollout.py` runs `PPOCoordinator` end to end.
It uses real Megatron GPT sampling and frozen Reward and reference models, then
performs GAE, Actor and Critic training, a full Actor export, and acknowledged
installation of the rollout weights, followed by a second round.

Roles have separate NCCL groups:
- **Actor:** ranks 0 and 1 (TP=2).
- **Critic:** rank 2 (TP=1) or ranks 2 and 3 (TP=2).
- **Rollout workers:** the remaining ranks.

Processes shared two GPUs, and control and leader transport used Gloo groups.

| Run | Ranks | Max behavior-logprob error | Engine steps / role | Megatron vs DeepSpeed final weights |
| --- | ---: | ---: | ---: | ---: |
| Critic TP=1 | 4 | 2.384e-7 | DeepSpeed: 4 | identical (0) |
| Critic TP=2 | 6 | 2.384e-7 | DeepSpeed: 4 | identical (0) |

Every run passes two complete PPO rounds. Each round checks:
- behaviour logprobs against the active Actor;
- PPO targets unchanged during the round;
- identical token and reward packets across roles;
- trainable parameters changed;
- frozen models unchanged.

See the [run summaries](rollout/).

## 3. vLLM rollout with live weight installation

The same coordinator runs with a real vLLM 0.15.1 rollout worker on a dedicated
GPU layout: Actor on GPUs 0–1, Critic on GPU 2, and the rollout callback with the
vLLM subprocess on GPU 3. vLLM runs in FP32 with eager execution, the
`TRITON_ATTN` backend, and untruncated sampling (temperature 1, no top-k/top-p).

| | Megatron | DeepSpeed |
| --- | --- | --- |
| Weight versions installed and acknowledged | 0, 1, 2 | 0, 1, 2 |
| Tensors verified per install | 29 | 29 |
| Cached tokens right after install / on same-version replay | 0 / 16 | 0 / 16 |
| Max behavior-logprob error (vLLM vs Actor recompute) | 4.768e-7 | 4.768e-7 |

See the [summaries and runtime reports](vllm/) for per-rank assertions and
installation/cache observations.

A separate dedicated four-GPU cluster check, job `1095862` (`test`, `overflow`),
completed two rounds with Actor TP=2, Critic TP=1 and vLLM TP=1. Its final Actor
and Critic tensors were exactly equal between the Megatron and DeepSpeed runs;
the maximum behavior-logprob error was `4.7684e-7` in both runs. The
[comparison report](dedicated-four-gpu/comparison.json) also records weight
versions, cache hits and timings. This report comes from the original source
snapshot before the repository-layout migration; it is separate from the
other run summaries above. Its timings are observations for a tiny FP32 model,
not production throughput measurements.

## 4. Unit and multi-process tests on GPU

With `RUN_TP_CUDA=1`, `tests/integrations/common/test_ppo.py`,
`tests/integrations/common/test_tp_scalar.py` and
`tests/integrations/engines/train/test_ppo_adapters.py` report 26 passed
([log](pytest-gpu.txt)).

## Limits

- **Framework configurations not validated:**
  - DeepSpeed AutoTP or inference kernel injection;
  - mixed precision, ZeRO-1/2/3, optimizer offload, gradient accumulation above 1;
  - DP, PP or SP;
  - Transformer Engine fusions.

  The adapters reject unsupported modes.
- **Scope:** multi-node placement and production throughput.
- **Weight conversion:** the vLLM check uses a test-specific Megatron-to-GPT-2 weight converter, not a general checkpoint converter.
