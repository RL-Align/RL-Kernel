# DSv4 LoRA GRPO on 8 Hopper GPUs: RL-Kernel + verl (batch-invariant) vs verl #7483

Instructions for an engineer or a coding agent on one node with 8 x H200 141 GB. Follow the
steps in order. Every step says what success looks like and what to send back. Do not change
pins. If a step fails, record the exact command and error, try the listed fixes, and report.
Use the whole RL-Kernel checkout (branch `test-dsv4-cuda`), not only this directory.

## What we want to learn

1. **Bitwise:** with RL-Kernel + verl, is every response token's FP32 log-prob from the
   rollout (vLLM) bit-identical to the actor's recompute, at every training step, on the
   real DeepSeek-V4-Flash checkpoint, on 8 GPUs, while the LoRA adapters change every step?
   The verdict is verl's own `training/rollout_logprobs_mismatch_count` (FP32 `!=` per
   response token): 0 at every step. `actor/old_logprobs_mismatch_count` (the grad-enabled
   update forward against the recompute) should be 0 too.
2. **Speed:** the per-step time and its breakdown (generation, log-prob recompute, actor
   update, weight sync) against verl #7483, the reference DSv4 LoRA recipe (Hopper, not
   batch-invariant), on the same workload.
3. **Control:** with the adapter sync switched off, the same metric must turn nonzero.

Proven so far (design doc 10.1):
- 2 x H200, a 4-layer checkpoint at the real dimensions with random weights: mismatch 0 at
  each of 4 steps; the negative control 4096 of 4096 tokens from step 2.
- 8 x H200, the real checkpoint's 43 layers, the DAPO math reward: mismatch 0 at every one of
  100 steps, twice (E2: 32 x 8, response <= 2048; E3: 32 x 16, response <= 4096). Section 8
  has these runs and the baseline's.

## The two stacks

Both runs use the same verl commit (`8c047f4d`, the head of verl #7483), the same model
view, data, reward, batch and LoRA rank. verl, vLLM and Megatron are not modified in either.

| | RL-Kernel + verl (`rlkernel`) | verl #7483 (`baseline`) |
|---|---|---|
| Actor | verl engine `rlkernel` (RL-Kernel's `Dsv4Model`, batch-invariant kernels), data parallel 8 with the routed experts over EP 8, activation recompute (`RECOMPUTE=1`), no offload | Megatron (Bridge `c7774d44`, Megatron-LM `fd1121b8`), EP 8, parameter and optimizer offload, full recompute, router replay R3 |
| Rollout | vLLM fork `ISEEKYAN/vllm@1a6542c` + RL-Kernel's model plugin, `VLLM_BATCH_INVARIANT=1`, one DP 8 / EP 8 instance (attention data parallel), CUDA graphs with `GRAPH=1`, FP8 KV cache | vLLM 0.32.0 with its own LoRA (`merge=False`), one TP 8 / EP 8 instance, CUDA graphs, FP8 KV cache |
| LoRA targets | `wq_a`, `wkv`, `wq_b`, `wo_b`; routed `fc1` / `fc2` (one adapter shared by every expert); shared expert `fc1` / `fc2`. BF16 adapters, FP32 master copy | the launcher's list: the same projections plus the compressor (`wkv`, `wgate`), the indexer's `weights_proj` and the router; one adapter shared by every expert (`run.sh` drops `experts_shared_outer_loras`, which vLLM 0.32 cannot load); FP32 adapters. The launcher's LoRA+ ratio 16 is read by neither verl nor Bridge |
| Weight sync | BF16 adapters only, into the plugin's layers | verl's LoRA sync into vLLM's LoRA manager |
| Off-policy handling | none (the rollout is the recompute) | token-level rollout importance sampling (0.5-5) |
| Learning rate | `run.sh`: 1e-4, constant | `run.sh`: 5e-6, 3 warmup steps. `experiments.sh` sets both runs' optimizer the same (section 8) |

Read the speed numbers as stack against stack. The LoRA target sets and the rollout
parallelism differ (each run uses its own recipe's choices). The learning rate does not
change the step time. The ablation run (section 5) splits our time into what batch
invariance costs the rollout and the rest.

## 0. Requirements

| Item | Needed |
|---|---|
| GPUs | 8 x H200 141 GB, one node, NVLink. H100 80 GB: see "Model size" |
| Driver | R580+ (CUDA 13). An older driver: NVIDIA's forward-compat libcuda, `CUDA_COMPAT_DIR` in `env.sh`, e.g. on R550 |
| Host RAM | >= 512 GB (the baseline offloads parameters and optimizer state to the host) |
| Disk | >= 400 GB free under `LORA_E2E_ROOT` (default `~/dsv4-lora-e2e`) |
| Software | conda (or mamba), [uv](https://docs.astral.sh/uv/), git, curl, gcc <= 13. CUDA 13.0 toolkit: the setup installs it into the conda env unless you set `SYSTEM_CUDA_HOME` to a local CUDA 13.0 |
| Network | GitHub, PyPI, download.pytorch.org, the verl wheelhouse, Hugging Face (`HF_TOKEN` if needed) |
| Time | setup 3-4 h (the vLLM fork builds from source); then about 3-5 h of runs |

**Model size.** Both runs train on the first `LAYERS` layers of the checkpoint (default 4:
window, window, C4, C128, one of each attention type). Set `LAYERS` before `setup.sh data`
and keep it for every run.
- Small views (sections 2 to 7): without expert parallelism every GPU holds a whole copy of
  the actor and, during generation, of the rollout. Each layer is about 3.2 GB of FP4
  experts plus 6.4 GB of E4M3 expert weights kept expanded for speed. At 4 layers on H200
  the 2-GPU run peaked at 49 GB in the actor, next to the rollout's 35 % of the GPU. On
  H100 80 GB, 4 layers do not fit: use `LAYERS=2` (window layers only, so no compressed
  attention; say so in the report).
- The full model (`LAYERS=43`, section 8): EP 8 in the actor (`ACTOR_EP=8`) and the rollout
  (`ROLLOUT_DP=8`), activation recompute, and `RL_KERNEL_DSV4_PERSIST_W8=0` (the E4M3
  expansion per call instead of kept: same bytes). The actor peaked at 60 GB per H200 at
  8192 tokens per micro-batch.

## 1. Setup (host)

```bash
cd <RL-Kernel checkout>/scripts/verl_e2e_lora
export LORA_E2E_ROOT=~/dsv4-lora-e2e HF_TOKEN=...   # optional overrides (env.sh lists all)
bash setup.sh rlkernel   # conda env: torch 2.13 / CUDA 13, FlashMLA, DeepGEMM, vLLM fork,
                         # verl, RL-Kernel (_C with the MoE kernels + the vLLM plugin)
bash setup.sh baseline   # verl #7483's uv env + the Bridge / Megatron-LM commits
bash setup.sh data       # checkpoint (~160 GB), DAPO-Math-17k + AIME parquet, L$LAYERS view
```

Success: `rlkernel env ok` (with `torch 2.13.0+cu130 ... 13.0 | vllm 20260817.dev33+g1a6542cde
| deep_gemm 2.6.1+e21c821 | flash_mla 1.0.0+a8f794d | gpus 8`), `baseline env ok`, `missing
shards []` and the layer types `[0, 0, 4, 128]`.

- The RL-Kernel env installs the exact package set of the 2-GPU run
  (`rlkernel-env.lock.txt`) and applies one local DeepGEMM fix
  (`deepgemm-sm90-paged-mqa-empty.patch`: the paged-MQA scheduler read past its buffer
  when every context length in a batch was 0).
- If a locked package is not found, install that package's nearest version and report it.
- If the vLLM build fails, the log is `src/vllm-build.log`.
- If `uv sync` fails, report it; do not unpin.
- A host with glibc older than 2.38 (RHEL 8 / 9) builds both envs but cannot load some of
  their wheels (flash-attn in the RL-Kernel env; TransformerEngine, apex, DeepEP in the
  baseline). Pull an Ubuntu 24.04 image with CUDA 13 (`singularity pull cuda13.sif
  docker://nvidia/cuda:13.0.2-devel-ubuntu24.04`) and export `CONTAINER=<the .sif>`:
  `setup.sh baseline` then runs its check inside it and `run.sh` re-runs itself inside it.
  Build the baseline env with `UV_PYTHON_PREFERENCE=only-managed`: a conda Python on `PATH`
  brings an older `libstdc++` that TransformerEngine rejects.

Send back: the preflight block (GPU names, driver, topology, `free -g`) and the env lines.

## 2. Gate: RL-Kernel on the real checkpoint (2 GPUs, ~30-60 min)

The plugin and the engine have only run on random weights so far. These checks compare, byte
for byte: vLLM with the plugin against `Dsv4Model` (layer by layer, with and without LoRA),
the rollout / recompute / training forward per layer, and the verl engine across micro-batch
layouts, after training steps and through a checkpoint round trip.

```bash
bash run.sh gate
```

On H100 80 GB, give vLLM more of the GPU: `GATE_GPU_UTIL=0.8 bash run.sh gate` (0.4, the
default, holds the 4-layer view on an H200 only).

Success: both `vllm_gate*.log` and `layerwise_gate.log` end in a pass with zero differing
tokens, and `verl_engine.log` ends in `VERL ENGINE PASS`. If anything fails, stop here and
send the four logs and JSON reports; the training runs would only repeat the failure.
`vllm_ablation.log` is the ablation's rollout (section 5) on the same check: it must run to
the end, and its differences are expected (report its log-prob line).

## 3. RL-Kernel + verl (8 GPUs, 20 steps, ~1 h)

```bash
tmux new -s rlk
bash run.sh rlkernel
```

Early check after step 3, from another shell:

```bash
python3 compare.py $LORA_E2E_ROOT/outputs/rlkernel_*
```

Every row must show `rollout_logprobs_mismatch_count` 0, `actor/old_logprobs_mismatch_count`
0 and a nonzero `actor/grad_norm`. If not, stop the run (Ctrl-C) and report:
- a mismatch at step 1 (before any update) means a kernel or configuration difference;
- one that starts later means the adapter sync.

Out of memory: retry in this order and note what worked.
1. `GPU_UTIL=0.3`.
2. `TOKENS_PER_GPU=3072 MAX_RESPONSE_LENGTH=1024`. The baseline must then use the same
   values: they are the workload.
3. `LAYERS=3`. Rerun `setup.sh data` first, and redo every run.

Pass them as `GPU_UTIL=0.3 bash run.sh rlkernel`.

Ray gets `num_cpus` = `nproc` (override with `RAY_NUM_CPUS`). Its default counts every
CPU of the node; inside a Slurm allocation it then prestarts that many workers, which can
exceed the per-user thread limit (`ulimit -u`): the workers abort with "Resource
temporarily unavailable" and the run hangs right after `ray.init`, without an error.

## 4. Negative control (8 GPUs, 3 steps, ~15 min)

```bash
bash run.sh control
```

The rollout receives the adapters but keeps the old ones
(`RL_KERNEL_DSV4_SKIP_ADAPTER_SYNC=1`). Expected:
- 0 at step 1, because the initial adapters have zero B and equal the base model;
- nonzero from step 2.

This shows the metric sees a stale rollout.

## 5. Ablation: our stack with the rollout's batch invariance off (8 GPUs, 20 steps, ~1 h)

```bash
bash run.sh ablation
```

The `rlkernel` run with `BI=0`. The rollout plugin then drops what it does only for batch
invariance and byte equality with the actor:
- vLLM's non-invariant attention planning (batched prefill chunks, workspaces sized by the
  batch);
- FlashMLA's paged FP8 decode instead of decode on the sparse prefill kernel;
- vLLM's fused q / kv RMSNorm, query norm / RoPE / KV insert, indexer query and compressor
  store kernels;
- the aux-stream overlap of the indexer and compressor.

It keeps what defines the model the actor trains: RL-Kernel's mHC, MoE, output projection,
LoRA adapters, RoPE tables and FP32 log-prob head. The actor is unchanged; RL-Kernel's
kernels have no non-invariant variant, and their cost per operator against a non-invariant
kernel is in the design doc, section 9.

Expected:
- a nonzero mismatch count: rounding differences, not a different model;
- a shorter `timing_s/gen`.

On the toy checkpoint (random weights, 1 GPU), every token mismatched and the generation
took 0.79x of the BI run's time.

- Against the `rlkernel` run, the difference in `timing_s/gen` is the price of batch
  invariance in the rollout.
- Against the baseline, the remaining gap is the stacks themselves: the engine and the
  parallelism.

## 6. Baseline: verl #7483 (8 GPUs, 20 steps, ~1-2 h)

```bash
bash run.sh baseline
```

This runs a copy of #7483's `examples/tuning/lora/run_deepseek_v4_flash_megatron.sh`, changed
only in its logger (`wandb` -> `file`). Its own environment variables set the workload and
one node: actor EP 8 without PP / CP, rollout TP 8 / EP 8. A nonzero mismatch count is
expected, since this stack is not batch-invariant; report it anyway.

Out of memory: try `ROLLOUT_GPU_MEM_UTIL=0.4 bash run.sh baseline`, then report.

## 7. Compare and report

```bash
cd $LORA_E2E_ROOT/outputs
python3 <checkout>/scripts/verl_e2e_lora/compare.py rlkernel_<stamp> ablation_<stamp> \
  baseline_<stamp> > compare.md
tar czf ~/dsv4-lora-e2e-report.tgz compare.md */summary.md */summary.json */run_info.txt \
  */metrics.jsonl gate_*/*.log gate_*/*.json
```

`compare.py` prints, per run, the per-step table and the means after 2 warmup steps. With two
or more runs it also prints a side-by-side table whose ratios are relative to the first run.
Send the archive plus a short note:

| Item | Report |
|---|---|
| Setup | GPU, driver, host RAM, anything that needed a workaround or a different version |
| Gate | pass / fail of each of the four checks |
| rlkernel | steps completed; bitwise every step (yes / no), mismatched tokens and the first such step; `actor/old_logprobs_mismatch_count`; steps with nonzero grad norm; mean `timing_s/step`, `gen`, `old_log_prob`, `update_actor`, `update_weights`; peak memory; settings changed after OOM |
| control | mismatch count per step |
| ablation | the same as rlkernel |
| baseline | the same as rlkernel |
| Failures | exact command, last 50 lines of `train.log`, what was tried |

## 8. The full model (`experiments.sh`, 8 x H200, about 4-11 h per run)

`experiments.sh` holds the full-model runs as stages, each a `run.sh` call with the
real checkpoint's 43 layers, EP 8 on both sides and the DAPO math reward (its header lists
the settings):

```bash
DRY_RUN=1 bash experiments.sh BL2 E3 A3   # print the run.sh calls
bash experiments.sh BL2 E3 A3             # run them in order; logs in $OUT_DIR/<stage>.log
```

| Stage | Stack | Workload | Steps | Learning rate |
|---|---|---|---|---|
| `E2` | rlkernel, BI, graphs | 32 x 8, response <= 2048 | 100 | 1e-4 constant |
| `BL2` | verl #7483 | W7: 32 x 16, response <= 4096, AIME-2024 every 10 steps | 50 | 1e-4, cosine to 1e-5 over 50 |
| `E3` | rlkernel, BI, graphs | W7 | 100 | 1e-4, cosine to 1e-5 over 100 |
| `A3` | rlkernel, rollout BI off, graphs | W7 | `A3_STEPS` (50) | as E3, over `A3_STEPS` |

W7 sets the same optimizer and PPO knobs in both stacks (the baseline launcher's: weight
decay 0, clip 0.2 / 0.28, clip_c 10, token-mean, 3 warmup steps). BL2's rollout needs
`moe_backend=auto`: verl's default, DeepGEMM, has only an SM100 MXFP4 MoE.

Recorded (2026-11-20 / 21; means over steps 3 to the end, seconds per step):

| Run | Steps | Mismatched tokens per step | step | gen | old_log_prob | update_actor | sync | gen per longest response token | Actor peak |
|---|---|---|---|---|---|---|---|---|---|
| E2 | 100 | 0 | 175 | 150 | 6 | 17 | - | 74 ms | 47 GB |
| BL2 | 50 | 452k to 768k | 239 | 137 | 29 | 61 | 11 | 34 ms | 102 GB (with offload) |
| E3 | 100 | 0 | 388 | 328 | 13 | 43 | 3 | 84 ms | 60 GB |
| A3 | 30 | 156k to 437k | 294 | 229 | 14 | 46 | 3 | 56 ms | 60 GB |

- Bitwise: E2 and E3 have mismatch 0 and `rollout_corr/kl` 0 at every step. BL2 runs
  token-level importance sampling to correct its `rollout_corr/kl` of 0.013 to 0.023;
  A3's is 0.005 (only the rollout's kernels differ from the actor's).
- Speed: our actor side (`old_log_prob` + `update_actor`) is faster than the baseline's,
  our generation slower. Generation time follows the longest response (about 4000 tokens
  in almost every step), so `gen` / longest response is the decode-step latency at small
  batch. Of E3's 50 ms over BL2, the rollout's batch invariance is about 30 ms (E3 against
  A3); the rest is the stack (DP 8 attention with the MoE's static EP exchange against
  vLLM's TP 8, and our modules). Most of those 30 ms turned out to be vLLM's BI mode
  capping NCCL at one channel, which the plugin now undoes (after these runs): in the
  server alone, BI then costs 12-14 % over BI off. The profile is in the design doc,
  section 10.3.
- Learning: AIME-2024 (greedy, 30 problems): BL2 0.608 -> 0.558 over 50 steps; E3 between
  0.633 and 0.767, 0.667 at step 100. The two runs' learning rates differ after step 3
  (50- against 100-step cosine), and one problem is 0.033. A3: 0.649, 0.685, 0.705, 0.671
  at steps 0 to 30. The validation set holds each problem 32 times; with batch invariance
  the 32 greedy answers are identical (scores in steps of 1/30), without it they are not.

## Not covered (open work in the design doc, section 12)

- **Batch invariance off in the actor:** RL-Kernel's kernels have no non-invariant
  variant; the ablation turns it off in the rollout only.
- **The rollout's decode latency:** section 8's gap to vLLM's TP 8 at small batch.

## Files

| File | Role |
|---|---|
| `env.sh` | Pins (commits, packages, model, data) and paths |
| `setup.sh` | Environments, checkpoint, data, the layer view |
| `rlkernel-env.lock.txt` | The RL-Kernel env's packages (from the tested 2-GPU env) |
| `deepgemm-sm90-paged-mqa-empty.patch` | The local DeepGEMM fix |
| `run.sh` | `gate`, `rlkernel`, `control`, `ablation`, `baseline` |
| `experiments.sh` | The full-model runs (section 8) as stages over `run.sh` |
| `profile_rollout.sh`, `prof_client.py` | The rollout's vLLM server alone under nsys, E3 / A3 / BL2 settings (design doc 10.3) |
| `run_dsv4_rlkernel_grpo.sh` | The RL-Kernel + verl launcher `run.sh` calls (also the 2-GPU toy run) |
| `custom_reward.py` | The synthetic dense reward (`REWARD=synthetic`, the default) |
| `compare.py` | Per-run summary and the comparison table |
