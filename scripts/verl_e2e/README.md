# verl + vLLM batch-invariant: DeepSeek-V4-Flash end-to-end on 8 x H100

Instructions for an engineer or a coding agent on one node with 8 x H100 80 GB (H200 also works).
Follow the steps in order. Every step says what success looks like and what to send back. Do not
change pins. If a step fails, record the exact error and the command, try the listed fixes, and
report. This directory is self-contained: copy all of it, not only this file.

## What we want to learn

1. **Bitwise:** does verl + vLLM batch-invariant (BI) mode give *bit-identical* FP32 log-probs
   for every response token between the rollout (vLLM) and the old-policy recompute (Megatron
   Lite actor), at every training step, including after 200 weight updates?
2. **Speed:** what does it cost? Per-step time breakdown in verl, and vLLM rollout throughput with
   BI on vs off on the full model.

Stack under test: verl PR #7688 (`ISEEKYAN/verl@0c849d86`), Megatron-LM
`ISEEKYAN/Megatron-LM@327a2382` (branch `ds4_vllm_align_preview`), runtime image
`iseekyan/verl:ds4_vllm_align.preview-amd64` (vLLM `ISEEKYAN/vllm@168a040e` = vllm#54955
lineage, DeepGEMM grouped FP8 MoE, DeepEP, the BI kernel library). All pins are in `env.sh`.
The recipe trains all parameters (no LoRA), so on 8 GPUs the training tests use the first few
layers of the real checkpoint; the rollout speed test (section 6) uses the full 43-layer model.

Why a patch to verl: verl's own `training/rollout_probs_diff_max` compares `exp(log_prob)` and
reports **0.0 for log-probs that differ by 1 ulp** (checked), and `rollout_corr/k3_kl` rounds tiny
ratios to 0. `patch_verl_bitwise_metric.py` adds `training/rollout_logprob_bitwise_mismatch`: the
number of response tokens whose FP32 bits differ. That number is the verdict.

## 0. Requirements

| Item | Needed |
|---|---|
| GPUs | 8 x H100 80 GB (or H200), one node, NVLink (NV18 in `nvidia-smi topo -m`) |
| Driver | R580+ (the image is CUDA 13) |
| Host RAM | ~700 GB for the 8-layer runs (T3, T4); ~360 GB for 4 layers (T1, T2). The optimizer state (12 bytes per parameter, ~6.6 B parameters per layer) lives in host RAM |
| Disk | >= 600 GB free under `E2E_ROOT` (default `~/dsv4-e2e`) |
| Software | docker with the NVIDIA Container Toolkit, git, curl, python3, tmux (for the long run) |
| Network | Docker Hub, GitHub, Hugging Face (set `HF_TOKEN` if needed) |
| Time | about 1.5 days of machine time in total; T4 alone is roughly 8-20 h |

No docker? The image works with enroot/pyxis (`docker://iseekyan/verl:ds4_vllm_align.preview-amd64`)
or apptainer. Mount `E2E_ROOT` at `/e2e`, give >= 64 GB of shared memory (`--ipc=host`
equivalent), and run the in-container steps the same way.

GB200 / GB300 (arm64) instead of Hopper: stop and report; that needs verl #8403 and another guide.

## 1. Setup (host)

```bash
cp -r <this directory> ~/verl_e2e && cd ~/verl_e2e
export E2E_ROOT=~/dsv4-e2e HF_TOKEN=...      # optional overrides
bash setup.sh            # preflight, docker pull, pinned clones, verl patch, data, model (~160 GB)
bash launch_container.sh # starts container "dsv4-e2e", prints the image manifest and versions
```

Success: `setup.sh` ends with `missing shards []` and `setup done`; `launch_container.sh` prints
`BI kernel lib ok`, a torch version with CUDA 13, `gpus 8`, and the vLLM and DeepGEMM paths.
Send back: the preflight block (GPU names, driver, topology, `free -g`) and the launch output.

All test outputs go to `$E2E_ROOT/outputs/<test>_<stamp>/`: `train.log`, `metrics.jsonl`
(verl's per-step metrics), `summary.md` / `summary.json` (verdict and timing), `run_info.txt`.

## 2. T1: upstream quick alignment test (4 GPUs, 4 layers, 2 steps)

The configuration the recipe authors validated on Hopper; reproduce it before changing anything.

```bash
docker exec -it dsv4-e2e bash /e2e/tools/run_e2e.sh T1
```

Expect 20-60 min (the first run JIT-compiles DeepGEMM kernels).
Success: `summary.md` says **Bitwise on every step: YES** with a nonzero token count.

## 3. T2: 8 GPUs, 4 layers, 5 steps

Actor EP8, rollout DP8 + EP8, 16 prompts x 4 samples, responses up to 2048 tokens, one optimizer
update per step (so the first grad-enabled actor forward must equal the rollout too:
`actor/ppo_kl` exactly 0 at every step).

```bash
docker exec -it dsv4-e2e bash /e2e/tools/run_e2e.sh T2
```

Success: **Bitwise on every step: YES**, `actor/ppo_kl` = 0 and `pg_clipfrac` = 0 at every step.
The `timing_s/*` table in `summary.md` is the speed of this layout.

## 4. T3: 8 layers, 8192-token responses, 5 steps

```bash
docker exec -it dsv4-e2e bash /e2e/tools/run_e2e.sh T3
```

This finds the memory limits before the long run. On OOM (GPU), retry in this order and note
what worked: `ROLLOUT_GPU_MEMORY_UTILIZATION=0.45`, then `MAX_RESPONSE_LENGTH=4096`, then
`LAYERS=6`. On host OOM, use `LAYERS=6` (or 4). Pass variables as
`docker exec -it dsv4-e2e bash -c 'LAYERS=6 bash /e2e/tools/run_e2e.sh T3'`.
Use the settings that worked for T4.

## 5. T4: 200 training steps

The long run: 200 optimizer updates on 8 layers, 32 prompts x 8 samples per step, responses up
to 2048 tokens, one update per step, learning rate 1e-5.

A model cut to 8 of 43 layers cannot solve the math prompts: every sample would score 0, GRPO
advantages would be 0 and the weights would never move, which would make "bitwise after 200
updates" vacuous. T4 therefore uses `custom_reward.py` (fraction of digits in the response) as a
dense synthetic reward, and a learning rate of 1e-5 so that the BF16 / FP8 weights the rollout
receives change every step. The reward measures nothing about quality; this run tests the
alignment machinery over many real updates and weight syncs, and the step time.

Run it inside tmux so it survives a disconnect:

```bash
tmux new -s t4
docker exec -it dsv4-e2e bash /e2e/tools/run_e2e.sh T4
# with T3's working settings, e.g.:
# docker exec -it dsv4-e2e bash -c 'LAYERS=6 ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 bash /e2e/tools/run_e2e.sh T4'
```

Progress while it runs (from the host):

```bash
python3 ~/verl_e2e/summarize_metrics.py $E2E_ROOT/outputs/T4_*/metrics.jsonl --out /tmp/t4_progress
```

Success: **Bitwise on every step: YES** over 200 steps, and `steps_with_nonzero_grad_norm`
close to 200 (in `summary.json`; it proves the weights were updated). `critic/score/mean`
should drift upward (the synthetic reward is learnable), which is a sanity check that updates
reach the rollout.

No checkpoints are written (an 8-layer full-parameter checkpoint with optimizer state is ~700 GB).
If the run dies, report the last step reached and the end of `train.log`, then rerun T4; a
partial run (for example 120 bitwise steps) is still a result worth sending.

## 6. Rollout speed: BI on vs off (full model, 43 layers)

vLLM alone (no verl), DP8 + EP8, FP8 KV cache, random 1024-in / 1024-out prompts, concurrency
1 / 8 / 32 / 128 / 256. Each variant also runs `probe_server_bi.py`: 8 fixed prompts decoded
greedily alone and then mixed into batches of 16 / 64 / 192 random requests; with BI the tokens
and log-probs must be identical.

```bash
docker exec -it dsv4-e2e bash /e2e/tools/bench_rollout.sh bi      # the recipe's BI settings
docker exec -it dsv4-e2e bash /e2e/tools/bench_rollout.sh nonbi   # same FP8 weights, BI off
docker exec -it dsv4-e2e bash /e2e/tools/bench_rollout.sh native  # BI off, native MXFP4 experts
```

Each takes 30-60 min (model load + warmup + sweep). Results in
`$E2E_ROOT/outputs/rollout_<variant>_<stamp>/` (`summary.txt`, `probe.json`, `bench_c*.json`).
Do not run this while T4 is running (it needs all 8 GPUs).

Expected: `bi` probe prints `BATCH-INVARIANT`; `nonbi` / `native` probes usually do not (they
are the control). On H100 the FP8-expert model takes ~36 GB per GPU; if KV-cache allocation
fails, rerun with `GPU_UTIL=0.9 MAX_MODEL_LEN=4096` and say so. If a server fails to start,
check `server.log`; if a flag is rejected, check `vllm serve --help` in the container and report
the flag and message (do not drop BI flags silently).

## 7. Report

Collect into one archive and send it back:

```bash
cd $E2E_ROOT/outputs && tar czf ~/dsv4-e2e-report.tgz \
  */summary.md */summary.json */run_info.txt */metrics.jsonl \
  */probe.json */summary.txt */bench_c*.json */server_cmd.txt */server_env.txt
```

Plus a short note:

| Test | Result to report |
|---|---|
| Setup | GPU model, driver, host RAM, anything that needed a workaround |
| T1 / T2 / T3 | bitwise YES/NO, tokens compared, mismatched tokens, first mismatching step; `actor/ppo_kl`; mean `timing_s/step`, `timing_s/gen`, `timing_s/old_log_prob`, `timing_s/update_actor`, `timing_s/update_weights` (whatever `timing_s/*` keys exist); settings that changed after OOM |
| T4 | steps completed, bitwise YES/NO, mismatched tokens and the first such step, steps with nonzero grad norm, score at step 1 vs the last step, mean step time |
| Rollout | per variant: probe PASS/FAIL; output tok/s, TPOT p50, TTFT p50 at each concurrency |
| Failures | exact command, last 50 lines of the relevant log, what was tried |

If any test shows a mismatch, keep its `train.log` and note whether the mismatch appeared at
step 1 (before any update: a kernel or config difference) or only after updates (weight sync or
reload).

## Files

| File | Role |
|---|---|
| `env.sh` | All pins (image digest, commits, model, data) and paths |
| `setup.sh` | Host preflight, pull, clone, patch, download |
| `patch_verl_bitwise_metric.py` | Adds the FP32-bits mismatch metric to verl (idempotent) |
| `launch_container.sh` | Long-lived container with `/e2e` mounted, sanity print |
| `run_e2e.sh` | T1-T4 through verl #7688's launcher, then `summarize_metrics.py` |
| `custom_reward.py` | T4's synthetic dense reward |
| `summarize_metrics.py` | Per-step bitwise verdict, grad-norm count and timing from verl's JSONL |
| `bench_rollout.sh` | vLLM server BI / non-BI / native: probe + throughput sweep |
| `probe_server_bi.py` | Black-box batch-invariance probe against a vLLM server |
