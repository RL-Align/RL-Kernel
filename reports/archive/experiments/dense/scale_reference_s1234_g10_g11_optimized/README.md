# Version-aligned G10 vs optimized G11 (200 steps)

Both runs use the same Qwen3-8B workload: one 8×H100 node, TP4/CP2, 200 steps,
seed and rollout seed 1234, rollout batch 8 prompts × 16 samples, global batch
128, maximum response length 7,168, and maximum 4,096 tokens/GPU.

| Metric | G10 | Optimized G11 | Result |
|---|---:|---:|---|
| Rollout time (s) | 130.22 | 82.75 | G11 36.5% faster |
| Rollout tokens/GPU/s | 672.39 | 1134.00 | G11 68.7% higher |
| Reference logp time (s) | 20.90 | 20.92 | approximately equal |
| Actor train time (s) | 80.51 | 107.18 | G11 33.1% slower |
| Total step time (s) | 251.99 | 231.27 | G11 8.2% faster |
| Mean raw reward | 0.528555 | 0.491445 | G10−G11 +0.037109 |

G11 has exactly zero mismatch count and zero maximum absolute difference at all
200 steps. G10 is the production P/P comparison and has non-zero mismatch at
all 200 steps.

The paired mean reward difference (G10−G11) has a 95% bootstrap interval of
[+0.030664, +0.043633], using seed 1234 and
20,000 paired-step resamples. This is a single-training-seed result, not a
multi-seed generalization interval.

`rounds.csv` contains every paired step. `summary.json` records formulas,
distribution summaries, and bootstrap details. `plot_report.py` regenerates all
five PNG figures from the authoritative Ray logs and the sealed G10 CSV.
