# Qwen3-Next: the strict path against stock vLLM and stock VIME, end to end (B200, H100)

This is RFC #428 §7's report: training, prefill and decode against the qualified native path.

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-e2e-speed-b200-h100-v2.zip).

ZIP SHA-256: `010259830406de60dd1977927b007cad786895b76971ee03fae2b873ab47b013`.

Commits (every report has `tracked_tree_dirty: false`):
- **Replay and training** were measured at `37e04258`, which is this PR's tree without #622's strict-provider files.
- **Rollout** was re-measured at `3e06906f`, which runs the stock vLLM arms at CUDA's default connection limit (see Arms). The strict arm's environment is the same at both commits. Its code differs by #622's strict-provider files, merged into the base since, and by the bot-round guards of #740 and #744. Its rollout times are 1.00-1.03× those at `37e04258`.

| | B200 | H100 |
|---|---|---|
| Hardware | 4 × B200, driver 580.126.20 | 4 × H100, driver 580.173.02 |
| Phases | rollout, replay, training | rollout and replay; the strict TP4 actor needs about 120 GB per GPU to train |

- **Software:** torch 2.13.0+cu130, vLLM 0.30.0, FlashInfer 0.6.18.post1, Megatron-core 0.16.0rc0, flash-linear-attention 0.4.2.
- **Checkpoint:** `Qwen/Qwen3-Next-80B-A3B-Instruct` revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`.
- The same job first passed the TP4 chain (`preflight` → `l2`, `bi` 97 / 97, gate rc 0), so the strict arm is the certified one.

```bash
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 --checkpoint <ckpt> ...
python benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py --run <l2 run> --out <dir> --phases rollout,replay,train
python benchmarks/models/qwen3_next/plot_qwen3_next_e2e_bench.py <dir>/report.json [<rollout rerun>/report.json]
```

## Arms

Each arm runs in its own processes on the same four GPUs, with the same synthetic token inputs. Medians are over the repeats of 2 rounds; the per-repeat values are in each report.

| Phase | RL-Kernel strict | Native baseline |
|---|---|---|
| rollout (`generate`: prefill + decode) | the certified strict vLLM engine, TP4 | stock vLLM 0.30.0 with CUDA graphs, and eager. `VLLM_BATCH_INVARIANT=1` is refused by vLLM for GDN attention, so it is recorded as unsupported |
| replay (forward + selected-token logprob) | RL-Kernel model and logprob in VIME, TP4 | VIME's own Qwen3-Next model at its recipe layout, TP2 + sequence parallel / EP4 / ETP1 / DP2 |
| training (one `train_one_step`) | same as replay | same as replay |

The stock arms drop the strict runtime knobs, so vLLM's and Triton's numerics are their defaults. All arms keep the stage's host settings `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 16 and `VLLM_PLUGINS=""` (keeps RL-Kernel's own vLLM plugin out of the stock engine). `CUDA_DEVICE_MAX_CONNECTIONS=1` stays for the strict arm and for stock VIME, which needs it at TP > 1. The stock vLLM arms run at CUDA's default since `3e06906f`; vLLM does not need the limit.

Stock VIME cannot run Qwen3-Next at TP4, for two reasons:
- Megatron's attention output gate is not narrowed when TP exceeds the 2 KV heads.
- VIME's loader shards grouped experts only at ETP1.

So the stock arm uses `scripts/run-qwen3-next-80B-A3B.sh`'s layout on the same four GPUs. Each report records both layouts under `phases.replay.layouts`.

Replay and training therefore compare the two setups as each runs on four GPUs, not provider speed per rank. Each strict rank processes every sequence of a batch (TP4, DP1). Each stock rank processes half of them (DP2), with their tokens split again by sequence parallelism. In training, a strict rank runs all 4 micro-batches of the step, and a stock replica runs 2.

Two workloads:
- **certified:** the chain's shapes, 64 / 256 / 1024-token prompts, batch 1 and 4, 16 new tokens;
- **long:** 512-token prompts with 512 new tokens.

![B200](https://github.com/fusheng-ji/RL-Kernel/blob/f295151738653180bf686a48aa86df85834561bf/pending/qwen3-next-e2e-speed/qwen3-next-e2e-speed-b200-v2.png?raw=1)

![H100](https://github.com/fusheng-ji/RL-Kernel/blob/f295151738653180bf686a48aa86df85834561bf/pending/qwen3-next-e2e-speed/qwen3-next-e2e-speed-h100-v2.png?raw=1)

## Rollout (prefill + decode), total seconds, at `3e06906f`

| Workload | Shape | B200 strict | B200 graphs | B200 eager | H100 strict | H100 graphs | H100 eager |
|---|---|---:|---:|---:|---:|---:|---:|
| certified | p64 b1 | 2.92 | 0.07 | 1.02 | 4.54 | 0.10 | 1.77 |
| certified | p256 b4 | 3.35 | 0.20 | 1.04 | 5.23 | 0.31 | 2.02 |
| certified | p1024 b4 | 3.65 | 0.19 | 1.17 | 5.61 | 0.31 | 2.01 |
| long | p512 b1 | 102.9 | 1.73 | 32.02 | 163.0 | 2.27 | 55.66 |
| long | p512 b16 | 137.1 | 2.15 | 32.38 | 217.0 | 3.50 | 56.90 |
| long | p512 b64 | 234.0 | 3.19 | 32.92 | 373.4 | 5.93 | 57.80 |

- **Prefill** is no longer the cost. Time to the first token (one `generate` with `max_tokens=1`: the prompt's forward and the first sample, engine overhead included) is 0.18-0.71 s strict on B200 for the certified prompts.
- **Decode dominates:** at batch 1, strict decodes 5.0-5.5 tokens/s on B200, against stock eager's 15.9-16.0 and CUDA graphs' 298-308; on H100 the rates are 3.1-3.5, 9.2 and 231-238.
- Over all shapes, strict's rollout takes 16.6-73.5× (B200) and 16.8-71.9× (H100) as long as stock with CUDA graphs, and 2.8-7.1× (B200) and 2.5-6.5× (H100) as long as eager.
- **The connection limit:** against the same arms at `37e04258`, where they kept `CUDA_DEVICE_MAX_CONNECTIONS=1`:
  - H100: stock vLLM with CUDA graphs is 0.98-1.00×, and eager 1.01-1.02×;
  - B200: CUDA graphs are 1.00-1.02×, except p256 b4 and p1024 b4, which are 19-27% slower in both B200 runs (for example p256 b4, 0.166 → 0.201 s); eager is 0.90-1.02×;
  - strict, whose environment did not change, is 1.00-1.03×.
- **One B200 arm has 1 round, not 2:** stock CUDA graphs on the certified workload. In round 0, two TP ranks failed with Triton `KeyError: 'cubin'` while compiling into the cold shared compile cache, and the other ranks timed out. A rerun of that arm hit the same failure in round 0, and its round 1 agrees with this one within 5% on every shape. Both runs are in the ZIP.
- **Consistency:**
  - strict generates the same tokens across repeats and rounds on every shape, and row 0 of a batch equals the same prompt alone;
  - stock vLLM does not, on either GPU. At the larger batches, with CUDA graphs or eager, its tokens differ between repeats or rounds of the same request: on B200 at p256 b4, p1024 b4, long b16 and long b64; on H100 at p1024 b4 (eager), long b16, and long b64 (eager).

## Replay, forward + selected-token logprob, median seconds

| Workload | Shape | B200 strict | B200 stock VIME | strict / stock | H100 strict | H100 stock VIME | strict / stock |
|---|---|---:|---:|---:|---:|---:|---:|
| certified | p64 b1 | 0.200 | 0.265 | 0.75 | 0.282 | 0.426 | 0.66 |
| certified | p64 b4 | 0.201 | 0.292 | 0.69 | 0.310 | 0.471 | 0.66 |
| certified | p256 b1 | 0.202 | 0.286 | 0.71 | 0.290 | 0.463 | 0.63 |
| certified | p256 b4 | 0.211 | 0.302 | 0.70 | 0.319 | 0.491 | 0.65 |
| certified | p1024 b1 | 0.293 | 0.296 | 0.99 | 0.420 | 0.476 | 0.88 |
| certified | p1024 b4 | 0.432 | 0.308 | 1.40 | 0.573 | 0.502 | 1.14 |
| long | p512+512 b1 | 0.297 | 0.287 | 1.03 | 0.423 | 0.456 | 0.93 |
| long | p512+512 b4 | 0.437 | 0.312 | 1.40 | 0.573 | 0.491 | 1.17 |
| long | p512+512 b16 | 1.146 | 0.338 | 3.39 | 1.448 | 0.535 | 2.71 |

- **Small inputs:** strict replay is faster than stock VIME. Stock pays for its two-replica layout: with one sample, the second data-parallel replica runs a filler copy for the MoE all-to-all.
- **Large inputs:** strict falls behind as tokens grow (1024-token prompts at batch 4, and the long workload at batch 4 and 16). The cause has not been profiled.

## Training, one step, median seconds (B200)

| Workload | Strict step | of which optimizer | of which forward + backward | Stock VIME step | Optimizer | Forward + backward | Strict / stock |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified, p64 b4 | 75.1 | 25.7 | 49.4 | 29.0 | 27.0 | 2.0 | 2.6 |
| long, p512+512 b4 | 570.2 | 26.0 | 544.3 | 30.1 | 27.9 | 2.2 | 18.9 |

- **The optimizer** (CPU-offloaded Adam) costs about the same in both arms.
- **Strict forward + backward** is the gap: the strict GDN backward still differentiates the per-token recurrence, and the strict MoE backward loops over experts.
- These are the next targets; this PR does not change them.
