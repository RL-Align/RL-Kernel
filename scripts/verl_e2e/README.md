# verl + vLLM batch-invariant: DeepSeek-V4-Flash end-to-end on 8 Hopper GPUs

Instructions for an engineer or a coding agent on a machine with 8 x H100 / H200. Follow the
steps in order. Every step says what success looks like and what to send back. Do not change
pins. If a step fails, record the exact error and the command, try the listed fixes, and report.

## What we want to learn

1. **Bitwise:** does verl + vLLM batch-invariant (BI) mode give *bit-identical* FP32 log-probs
   for every response token across the rollout (vLLM) and the old-policy recompute (Megatron
   Lite actor), at every training step, after weight updates?
2. **Speed:** what does it cost? Per-step time breakdown in verl, and vLLM rollout throughput with
   BI on vs off on the full model.

Stack under test: verl PR #7688 (`ISEEKYAN/verl@0c849d86`), Megatron-LM
`ISEEKYAN/Megatron-LM@327a2382` (branch `ds4_vllm_align_preview`), runtime image
`iseekyan/verl:ds4_vllm_align.preview-amd64` (vLLM `ISEEKYAN/vllm@168a040e` = vllm#54955
lineage, DeepGEMM, DeepEP, the BI kernel library). All pins are in `env.sh`.

Why a patch to verl: verl's own `training/rollout_probs_diff_max` compares `exp(log_prob)` and
reports **0.0 for log-probs that differ by 1 ulp** (checked), and `rollout_corr/k3_kl` rounds tiny
ratios to 0. `patch_verl_bitwise_metric.py` adds `training/rollout_logprob_bitwise_mismatch`: the
number of response tokens whose FP32 bits differ. That number is the verdict.

## 0. Requirements

| Item | Needed |
|---|---|
| GPUs | 8 x H100 80 GB or H200, one node, NVLink (NV18 in `nvidia-smi topo -m`) |
| Driver | R580+ (the image is CUDA 13) |
| Host RAM | >= 512 GB for T2 (4 layers); ~80 GB more per extra layer (optimizer offloaded to CPU, see T3) |
| Disk | >= 600 GB free under `E2E_ROOT` (default `~/dsv4-e2e`) |
| Software | docker with the NVIDIA Container Toolkit, git, curl, python3 |
| Network | Docker Hub, GitHub, Hugging Face (set `HF_TOKEN` if needed) |

No docker? The image works with enroot/pyxis (`docker://iseekyan/verl:ds4_vllm_align.preview-amd64`)
or apptainer. Mount `E2E_ROOT` at `/e2e`, add `--ipc=host`-equivalent shared memory (>= 64 GB),
and run the in-container steps below the same way.

GB200 / GB300 (arm64) instead of Hopper: stop and report. The matching recipe there is verl #8403
(DSv4.1 + LoRA, bitwise on 8 GB200) and needs a different guide.

## 1. Setup (host)

```bash
cp -r <this directory> ~/verl_e2e && cd ~/verl_e2e     # or use it from an RL-Kernel checkout
export E2E_ROOT=~/dsv4-e2e HF_TOKEN=...                  # optional overrides
bash setup.sh            # preflight, docker pull, pinned clones, verl patch, data, model (~160 GB)
bash launch_container.sh # starts container "dsv4-e2e", prints the image manifest and versions
```

Success: `setup.sh` ends with `missing shards []` and `setup done`; `launch_container.sh` prints
`BI kernel lib ok`, a torch version with CUDA 13, `gpus 8`, and the vLLM and DeepGEMM paths.
Send back: the preflight block (GPU names, driver, topology, `free -g`) and the launch output.

## 2. T1: upstream quick alignment test (4 GPUs, 4 layers, 2 steps)

The configuration the recipe authors validated on Hopper. It reproduces their claim before we
change anything.

```bash
docker exec -it dsv4-e2e bash /e2e/tools/run_e2e.sh T1
```

Expect 20-60 min (first run JIT-compiles DeepGEMM kernels). Output:
`$E2E_ROOT/outputs/T1_<stamp>/` with `train.log`, `metrics.jsonl`, `summary.md`, `run_info.txt`.

Success: `summary.md` says **Bitwise on every step: YES** with a nonzero token count.

## 3. T2: 8 GPUs, 4 layers, 5 steps

Actor EP8, rollout DP8 + EP8, 16 prompts x 4 samples, responses up to 2048 tokens, one optimizer
update per step (so the grad-enabled actor forward must equal the rollout too: `actor/ppo_kl`
exactly 0 at every step).

```bash
docker exec -it dsv4-e2e bash /e2e/tools/run_e2e.sh T2
```

Success: **Bitwise on every step: YES**, `actor/ppo_kl` = 0 and `pg_clipfrac` = 0 at every step.
Also look at `timing_s/*` in `summary.md`: these are the speed numbers for this layout.

## 4. T3: more layers, longer responses

```bash
docker exec -it dsv4-e2e bash -c 'LAYERS=8 bash /e2e/tools/run_e2e.sh T3'
```

Memory guide (DSv4-Flash: ~6.6 B parameters per layer, ~2 B for embedding + head): the optimizer
(FP32 master + Adam moments, 12 bytes per parameter) is offloaded to host RAM, so host RAM needs
about `80 GB x LAYERS + 30 GB` plus the Ray/dataloader overhead. GPU memory per rank grows by
roughly 4-6 GB per layer at EP8 (BF16 weights and gradients, plus the colocated FP8 rollout copy). Start at `LAYERS=8`; if it fits, try 12 or 16. On OOM, lower
`ROLLOUT_GPU_MEMORY_UTILIZATION` (0.45), `MAX_RESPONSE_LENGTH` (4096) or `LAYERS`.

Optional, if time allows: `TOTAL_TRAINING_STEPS=20` on the largest `LAYERS` that fits, to show the
bitwise property holds over many updates.

## 5. Rollout speed: BI on vs off (full model, 43 layers)

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

Expected: `bi` probe prints `BATCH-INVARIANT`; `nonbi` / `native` probes usually do not (they
are the control). If a server fails to start, check `server.log`; if a flag is rejected, check
`vllm serve --help` in the container and report the flag and message (do not drop BI flags
silently). If the full model does not fit (H100), rerun with `GPU_UTIL=0.9 MAX_MODEL_LEN=4096` or
`MODEL=/e2e/models/DeepSeek-V4-Flash-L8` (created by T3) and say so.

## 6. Report

Collect into one archive and send it back:

```bash
cd $E2E_ROOT/outputs && tar czf ~/dsv4-e2e-report.tgz \
  */summary.md */summary.json */run_info.txt */probe.json */summary.txt */bench_c*.json \
  */server_cmd.txt */server_env.txt
```

Plus, in a short note:

| Test | Result to report |
|---|---|
| Setup | GPU model, driver, host RAM, anything that needed a workaround |
| T1 / T2 / T3 | bitwise YES/NO, tokens compared, mismatched tokens, first mismatching step; `actor/ppo_kl`; mean `timing_s/step`, `timing_s/gen`, `timing_s/old_log_prob`, `timing_s/update_actor`, `timing_s/update_weights` (whatever `timing_s/*` keys exist) |
| Rollout | per variant: probe PASS/FAIL; output tok/s, TPOT p50, TTFT p50 at each concurrency |
| Failures | exact command, last 50 lines of the relevant log, what was tried |

If any test shows a mismatch, also keep `train.log` of that run and note whether the mismatch
appeared at step 1 (before any update: a kernel or config difference) or only after an update
(weight sync / reload).

## Files

| File | Role |
|---|---|
| `env.sh` | All pins (image digest, commits, model, data) and paths |
| `setup.sh` | Host preflight, pull, clone, patch, download |
| `patch_verl_bitwise_metric.py` | Adds the FP32-bits mismatch metric to verl (idempotent) |
| `launch_container.sh` | Long-lived container with `/e2e` mounted, sanity print |
| `run_e2e.sh` | T1 / T2 / T3 through verl #7688's launcher, then `summarize_metrics.py` |
| `summarize_metrics.py` | Per-step bitwise verdict and timing from verl's JSONL logger |
| `bench_rollout.sh` | vLLM server BI / non-BI / native: probe + throughput sweep |
| `probe_server_bi.py` | Black-box batch-invariance probe against a vLLM server |
