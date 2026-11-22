# Qwen3-Next: the strict path against stock vLLM and stock VIME, end to end (B200, H100)

This is RFC #428 §7's report: training, prefill and decode against the qualified native path.

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-e2e-speed-b200-h100.zip).

ZIP SHA-256: `223f78fd770be3ceeb91afd19066635b0eca9a9a5771de4ae3d4feb28c21135e`.

Measured at commit `37e04258`, which is this PR's tree without #622's strict-provider files. Every report has `tracked_tree_dirty: false`.

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
python benchmarks/models/qwen3_next/plot_qwen3_next_e2e_bench.py <dir>/report.json
```

## Arms

Each arm runs in its own processes on the same four GPUs, with the same synthetic token inputs. Medians are over the repeats of 2 rounds; the per-repeat values are in each report.

| Phase | RL-Kernel strict | Native baseline |
|---|---|---|
| rollout (`generate`: prefill + decode) | the certified strict vLLM engine, TP4 | stock vLLM 0.30.0 with CUDA graphs, and eager. `VLLM_BATCH_INVARIANT=1` is refused by vLLM for GDN attention, so it is recorded as unsupported |
| replay (forward + selected-token logprob) | RL-Kernel model and logprob in VIME, TP4 | VIME's own Qwen3-Next model at its recipe layout, TP2 + sequence parallel / EP4 / ETP1 / DP2 |
| training (one `train_one_step`) | same as replay | same as replay |

Stock VIME cannot run Qwen3-Next at TP4, for two reasons:
- Megatron's attention output gate is not narrowed when TP exceeds the 2 KV heads.
- VIME's loader shards grouped experts only at ETP1.

So the stock arm uses `scripts/run-qwen3-next-80B-A3B.sh`'s layout on the same four GPUs. Each report records both layouts under `phases.replay.layouts`.

Two workloads:
- **certified:** the chain's shapes, 64 / 256 / 1024-token prompts, batch 1 and 4, 16 new tokens;
- **long:** 512-token prompts with 512 new tokens.

![B200](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-e2e-speed-b200.png)

![H100](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-e2e-speed-h100.png)

## Rollout (prefill + decode), total seconds

| Workload | Shape | B200 strict | B200 graphs | B200 eager | H100 strict | H100 graphs | H100 eager |
|---|---|---:|---:|---:|---:|---:|---:|
| certified | p64 b1 | 2.88 | 0.07 | 1.00 | 4.52 | 0.10 | 1.75 |
| certified | p256 b4 | 3.31 | 0.17 | 1.15 | 5.21 | 0.32 | 1.99 |
| certified | p1024 b4 | 3.63 | 0.16 | 1.15 | 5.57 | 0.31 | 1.97 |
| long | p512 b1 | 99.7 | 1.72 | 31.64 | 163.2 | 2.26 | 54.85 |
| long | p512 b16 | 132.7 | 2.14 | 32.09 | 217.2 | 3.50 | 56.25 |
| long | p512 b64 | 226.5 | 3.17 | 32.62 | 370.5 | 5.93 | 57.06 |

- **Prefill** is no longer the cost: 0.18-0.71 s strict on B200 for the certified prompts.
- **Decode dominates:** at batch 1, strict decodes 5.1-5.6 tokens/s on B200, against stock eager's 16.2 and CUDA graphs' 300; on H100 the rates are 3.1-3.6, 9.3 and 232-238.
- Over all shapes, strict's rollout takes 19.9-71.4× (B200) and 16.4-72.1× (H100) as long as stock with CUDA graphs, and 2.8-6.9× (B200) and 2.6-6.5× (H100) as long as eager.
- **Consistency:**
  - strict generates the same tokens across repeats and rounds on every shape, and row 0 of a batch equals the same prompt alone;
  - stock vLLM does not, on either GPU. With CUDA graphs or eager, at the larger batches (p1024 b4, long b16 and b64), its tokens differ between repeats or rounds of the same request.

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
