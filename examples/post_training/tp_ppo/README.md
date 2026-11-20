# TP Reward/Critic and synchronous PPO

This example implements the data path requested by issue #44 and a complete
PyTorch PPO loop with **separate Actor and Critic tensor-parallel groups**.
The small feature-sharded bigram models exercise real sampling, autograd,
optimizers, reward return and weight installation. They are not transformer
performance benchmarks. This command uses plain PyTorch; the separate real
DeepSpeed/Megatron adapter checks are described below.

From the repository root, with the project Python environment activated:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 \
  -m examples.post_training.tp_ppo.train --actor-tp 2 --critic-tp 1 --rounds 2
```

Use `--actor-tp 1 --critic-tp 2` for the inverse topology. Both TP sizes must
divide the example's hidden width of 8. For CUDA/ROCm, add `--backend nccl` and
allocate one GPU per process. The production interfaces do not impose the
example's width constraint.

## Integration contracts

- `TensorPacketBridge` in `rl_engine/integrations/common/weights.py` transports an agreed,
  ordered tensor schema with dynamic shapes and dtypes. Its source is a **global
  rank**, and its process group must be explicit. All members call it in the
  same order. Tensors must already be on the configured device. Transported
  tensors are detached snapshots, not distributed autograd edges.
- `TPScalarExecutor` accepts a model adapter returning `[B,S]` state values or
  `[B]` rewards. The adapter accepts `input_ids`, `attention_mask`, `position_ids`
  and `use_cache=False`. `replicated` means that the model already completed
  head communication. `local_sum` means additive rank-local head contributions;
  include the bias on exactly one rank. This layout must be identical throughout
  a TP group. Reward pooling happens before reducing only `[B]` to its leader.
- Critic `values()` keeps autograd enabled. `sum_tp_contributions()` performs
  SUM in forward and identity in backward because **every TP rank evaluates the
  same loss**. It is not a general-purpose distributed-autograd SUM. Adapters
  own any backbone communication and synchronization of replicated parameter
  gradients. No hidden states or vocabulary logits are gathered by the scalar
  executor. Native PyTorch reductions provide numerical, not cross-topology
  bitwise, equivalence.
- `PPORollout` contains full prompt/completion sequences. Action slot `t` scores
  token `t+1` from state `t`; Critic state values include the final state needed
  for bootstrap. Attention and completion spans must be contiguous; both left
  and right padding are accepted. Completion must be a nonempty suffix of valid
  tokens. `terminated=True` requires a genuine terminal state; a length cutoff
  uses `False`. The caller must preserve the sampler's stop semantics.
- Behavior logprobs must come from the actual sampler. V1 requires untruncated
  categorical sampling (`temperature=1`, `top_k=0`, `top_p=1`). There is no
  synthetic-logprob or synthetic-reward fallback. `sample_ids` preserve ordering;
  Actor/Critic version scalars must match the coordinator.
- `PPOTrainingWorker` takes an explicit optimizer, a score callback, and a
  `sharded`/`replicated` layout for every trainable parameter. Replicated gradients
  must already be synchronized by the adapter; they count once in the global
  clipping norm. The optimizer must own exactly those trainable parameters.
  Models run in eval mode with autograd enabled to avoid changing the behavior
  policy through dropout. Caller model modes are restored afterward.
- `PPOProcessGroups` describes three disjoint role groups, their leaders, a
  leaders-only relay group, and a control group containing all participating
  ranks. Create groups in a consistent global order, with finite timeouts.
  Groups must use compatible backends. Actor and Critic TP sizes may differ.
- `PPOServices.reference_model` and Reward parameters are frozen. Reference and
  Reward models must be separate from trainable models. The reference callback
  must use the supplied reference model and return selected `[B,S-1]` logprobs.
- `export` runs on all Actor TP ranks so an adapter can gather weights, returning
  complete tensors on its leader. `install` runs on all rollout ranks and must
  return `True` only after the requested version is active. Weight keys are
  declared in `weight_fields`. Initial installation happens before the first
  sample. Subsequent versions advance only after all installation acknowledgments.

The coordinator relays reference scores and frozen PPO targets through leaders,
then broadcasts within each role group. Returned targets include scalar rewards
on every rollout rank. `last_batch` exposes the scored batch in sample order.
An empty batch skips all model/optimizer work and does not advance versions.

## Objective and defaults

Token reward is `-kl_coefficient * (behavior_logp - reference_logp)`, with the
sequence reward added to the final action. Natural termination bootstraps zero;
length truncation bootstraps the final Critic state. GAE targets are computed in
FP32 and frozen for all PPO epochs. Only valid action advantages are normalized.
The Actor uses the clipped surrogate; the Critic uses half the maximum of
unclipped and clipped squared value error. Minibatches use stable sample order
on both training groups.

`PPOConfig` defaults: gamma 1.0, lambda 0.95, policy/value clipping 0.2,
KL coefficient 0.01, four epochs, minibatch size 8, global gradient norm cap 1.0.
Actor and Critic have independent optimizers. There is no entropy bonus, DP,
PP, asynchronous scheduling, automatic runtime patching or optimizer rollback.

Validation errors are propagated before entering the next model/transport stage.
Failures inside model collectives are bounded by the caller's process-group
timeout. Any failed round poisons the coordinator: restore a checkpoint and
reconstruct it rather than continuing after partial optimizer/installation work.

## Validation and communication accounting

```bash
OMP_NUM_THREADS=1 python -m pytest \
  tests/integrations/common/test_ppo.py tests/integrations/common/test_tp_scalar.py -q
# Additional test requiring two available CUDA/ROCm devices:
RUN_TP_CUDA=1 OMP_NUM_THREADS=1 python -m pytest tests/integrations/common/test_tp_scalar.py -k nccl -q
```

Tests compare TP=1/2/4 scalar outputs, shard gradients and optimizer updates
against an unsharded model. They also cover nonzero leaders, dynamic/empty
packets, left/right padding, GAE, clipped losses, independent TP sizes, two PPO
rounds, frozen models and rejected installation acknowledgments.

`head_metrics` counts only head payload reductions and payload bytes: a local
Reward head reduces `B` scalars; an already-replicated head adds no payload
reduction. Input/output validation uses separate small status collectives.
Bridge metrics report per-rank payload bytes, broadcast calls and host elapsed
time, excluding validation traffic; they are not total network traffic or GPU
kernel timings. Coordinator metrics sum its local bridge observations. No
end-to-end speedup is claimed without a representative model benchmark.

## Real Megatron Core / DeepSpeed checks

`rl_engine/integrations/engines/train/{megatron,deepspeed}/ppo.py` adds explicit adapters for a real
Megatron `GPTModel`, vocabulary-parallel selected logprobs and a feature-sharded
scalar head. `DeepSpeedTPGroups` supplies the legacy MPU methods required by
DeepSpeed using actual TP and singleton DP groups. `deepspeed_ppo_worker()` uses
the engine's `backward()` and `step()` hooks; passing a DeepSpeed engine through
the plain PyTorch optimizer path is not sufficient.

The validated engine configuration is **FP32, ZeRO-0, DP=1, accumulation=1**, with
DeepSpeed gradient clipping disabled because the PPO worker clips the global TP
gradient norm. The adapter rejects unsupported mixed precision, autocast,
partitioned optimizers and mismatched TP groups. Megatron sequence/pipeline
parallelism is not supported by these adapters. The policy's model group must
match Megatron's active TP group used by its vocabulary-parallel cross entropy.

The opt-in test uses two-layer random GPT models, independent frozen Reward and
reference models, and real Actor/Critic backward and optimizer updates. It
reconstructs full tensors using Megatron's partition metadata and compares
TP=2 with TP=1, and DeepSpeed with native Megatron. No pretrained weights,
framework source patches, Apex or Transformer Engine are needed.

In the project's PyTorch environment, the following isolated dependency overlay
reproduces the validation. The base environment must also provide DeepSpeed's
ordinary dependencies (`ninja`, `packaging`, `psutil`, `pydantic>=2`, `tqdm`).
This keeps the optional NumPy 1.26 requirement out of the normal project environment:

```bash
tp_framework_deps=/tmp/rl-kernel-tp-frameworks
DS_BUILD_OPS=0 python -m pip install --target "$tp_framework_deps" --no-deps \
  -r examples/post_training/tp_ppo/requirements-frameworks.txt
export PYTHONPATH="$tp_framework_deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --standalone --nproc-per-node=1 -m tests.distributed.tp.check_tp_ppo_frameworks \
  --framework megatron --tp 1 --output runs/tp_ppo_frameworks/megatron_tp1.pt
torchrun --standalone --nproc-per-node=2 -m tests.distributed.tp.check_tp_ppo_frameworks \
  --framework megatron --tp 2 --baseline runs/tp_ppo_frameworks/megatron_tp1.pt \
  --output runs/tp_ppo_frameworks/megatron_tp2.pt
torchrun --standalone --nproc-per-node=1 -m tests.distributed.tp.check_tp_ppo_frameworks \
  --framework deepspeed --tp 1 --baseline runs/tp_ppo_frameworks/megatron_tp1.pt \
  --output runs/tp_ppo_frameworks/deepspeed_tp1.pt
torchrun --standalone --nproc-per-node=2 -m tests.distributed.tp.check_tp_ppo_frameworks \
  --framework deepspeed --tp 2 --baseline runs/tp_ppo_frameworks/megatron_tp1.pt \
  --output runs/tp_ppo_frameworks/deepspeed_tp2.pt
```

Each command saves a tensor snapshot and a JSON report next to it. See
[the measured validation results](../../../reports/experiments/issue44_tp_ppo/README.md) for versions,
numerical errors and the exact limits of this check.

## Independent real-framework rollout scheduling

The separate `tests.distributed.tp.check_tp_ppo_rollout` harness runs the full
coordinator with real Megatron GPT sampling, frozen Reward/reference models,
GAE, Actor/Critic training, full Actor export and acknowledged rollout weight
installation, followed by a second rollout. It checks behavior logprobs against
the active Actor before each update, unchanged PPO targets, identical token and
reward packets across roles, changed trainable parameters and unchanged frozen
models. DeepSpeed runs additionally assert four actual engine steps per role.

With the dependency overlay and environment above, run on **two physical GPUs**:

```bash
for framework in megatron deepspeed; do
  torchrun --standalone --nproc-per-node=4 -m tests.distributed.tp.check_tp_ppo_rollout \
    --framework "$framework" --critic-tp 1 \
    --output "runs/tp_ppo_rollout/${framework}_c1"
  torchrun --standalone --nproc-per-node=6 -m tests.distributed.tp.check_tp_ppo_rollout \
    --framework "$framework" --critic-tp 2 \
    --output "runs/tp_ppo_rollout/${framework}_c2"
done
```

Actor ranks are `[0,1]`; Critic ranks are `[2]` or `[2,3]`; remaining ranks
are rollout workers. Each role has its own NCCL group. Models explicitly receive
role TP groups and singleton DP/PP/CP groups. Megatron's startup creates extra
nominal groups, but these are not used for DP training. Rollout uses a complete
TP=1 GPT replica on each worker; only its leader samples.

Processes share GPUs across roles (`global_rank % 2`). No NCCL communicator used
for tensor collectives contains two ranks on the same GPU: control and leader
transport use explicit CPU/Gloo groups. `PPOCoordinator(transport_device="cpu",
device="cuda:...")` stages detached tensors between these groups and local CUDA
model groups; staging errors propagate through the control group before the
next collective. Omitting `transport_device` preserves the original same-device
path. Bridge elapsed times exclude staging and are not throughput measurements.

Reports are written to each output directory as `rank<N>.json`, `summary.json`,
`actor.pt` and `critic.pt`. This validates the Megatron rollout callback and
DeepSpeed training engines; it does not validate vLLM installation, DeepSpeed
AutoTP, multi-node placement, or dedicated-GPU performance.

## vLLM rollout and live weight installation

The independent-role harness also supports a real vLLM 0.15.1 rollout worker.
Actor TP=2 and Critic TP=1 remain in their own Megatron groups; both native and
DeepSpeed training paths are supported by this opt-in check. A separate Python
process runs vLLM TP=1 on GPU 1, so its Torch/CUDA dependencies and distributed
runtime are isolated from the trainer.

```bash
uv venv --python python3.11 /tmp/rl-kernel-vllm-env
uv pip install --python /tmp/rl-kernel-vllm-env/bin/python \
  -r examples/post_training/tp_ppo/requirements-vllm.txt
# Run from the trainer environment with the framework overlay described above.
for framework in megatron deepspeed; do
  torchrun --standalone --nproc-per-node=4 -m tests.distributed.tp.check_tp_ppo_rollout \
    --framework "$framework" --critic-tp 1 \
    --vllm-python /tmp/rl-kernel-vllm-env/bin/python \
    --output "runs/tp_ppo_vllm/${framework}_cached"
done
```

This option uses a two-layer, hidden-size-128 GPT, vocabulary 32, four attention
heads, 64 learned positions, and prompts of length 18 and 20. Each prompt
fills at least one KV-cache block. The test-specific converter reorders Megatron's
head-grouped QKV weights and transposes GPT-2 Conv1D weights. It is not a general
checkpoint converter. No pretrained weights or tokenizer downloads are needed.

The vLLM worker extension is invoked by name through `collective_rpc`; it loads
all exported tensors, verifies every loaded parameter, and acknowledges the
new version only after installation and prefix-cache reset. Local tensor files
provide a simple synchronous transport for this check. Sampling uses explicit
EOS token 31, temperature=1, no top-k/top-p truncation, and actual selected-token
vLLM logprobs. The Actor recomputes these logprobs before both PPO updates.

The isolated runtime uses FP32, eager execution, `TRITON_ATTN`, and
`TRITON_F32_DEFAULT=ieee` / `VLLM_FLOAT32_MATMUL_PRECISION=highest` for numerical
validation. The project's training precision settings are unchanged. Prefix
caching is enabled and cleared on every weight update. The test asserts zero
cached tokens immediately after installation, then 16 or more cached tokens
for a same-version replay with equal sampled tokens and logprobs. These settings, tiny
workloads, and shared GPUs are not a production-throughput benchmark.

The runtime log and version/install/sample timing report are saved under each
run's `vllm/` directory. `summary.json` records PPO round wall time and maximum
behavior-logprob error; Actor/Critic snapshots remain available for comparison.

## Cluster allocation with four separate GPUs

`cluster_four_gpu.sbatch` requests one overflow node, four B200 GPUs, 32 CPUs,
128 GiB host memory and one hour under job name **test**. It runs both framework
checks with `--gpu-layout dedicated`: coordinator ranks 0/1 train Actor, rank 2
trains Critic, and rank 3 hosts the vLLM rollout callback. The vLLM subprocess
inherits only rank 3's assigned GPU. This removes cross-role GPU sharing;
other jobs may still occupy the remaining GPUs on the same node.

The batch script takes four absolute paths: a frozen source directory, trainer
Python executable, runtime tar archive and results directory. The archive must
contain `frameworks/` (Megatron/DeepSpeed overlay) and `vllm/` (the isolated
vLLM virtual environment). It is unpacked into private node-local scratch and
removed when the job exits. Preserve a valid Python interpreter symlink in the
virtual environment. The job records allocation details, GPU topology, both
framework reports and a final tensor comparison; a `PASS` file is written only
after both runs and all comparisons succeed. Preempted/requeued attempts use
separate directories through `SLURM_RESTART_COUNT`.

The first submitted allocation is recorded in
`runs/cluster_tp_ppo/submission.json`; logs and results are beneath that same
root. This still uses the numerical-validation model and eager FP32 settings.
A production throughput experiment needs the actual deployment model, precision
and workload distribution in addition to this dedicated-GPU setup.
