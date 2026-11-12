# ROCm G10 P/P vs G11 R/R (200 steps)

Both runs use the same Qwen3-8B workload: one 8×AMD Instinct MI300X 192GB node,
TP4/CP2/PP1, 200 steps, seed and rollout seed 1234, rollout batch 1 prompt ×
8 samples, global batch 8, maximum response length 7,168, and maximum 4,096
tokens/GPU. The reference model and KL loss coefficient 0.001 are enabled.

| Metric | G10 P/P | G11 R/R | Result |
|---|---:|---:|---|
| Rollout time (s) | 73.77 | 81.43 | G11 10.4% slower |
| Rollout tokens/GPU/s | 93.79 | 72.60 | G11 22.6% lower |
| Reference logp time (s) | 1.56 | 3.33 | G11 113.4% slower |
| Actor train time (s) | 4.88 | 8.75 | G11 79.3% slower |
| Total step time (s) | 84.52 | 99.21 | G11 17.4% slower |
| Mean raw reward | 0.163750 | 0.453125 | G10−G11 -0.289375 |

G11 passes strict validation with 0 mismatches over 9,400,614 compared elements,
zero maximum absolute difference, and `torch.equal == true`. G10 completed all
200 training steps; its validator failure is limited to RL-Kernel operator
readbacks that the native P/P route intentionally does not emit.

The paired mean reward difference (G10−G11) has a 95% bootstrap interval of
[-0.345000, -0.235625], using seed 1234 and
20,000 paired-step resamples. This is a single-training-seed result, not a
multi-seed generalization interval.

For the fairest throughput comparison, excluding only rollout 0 warmup and
pooling actual generated tokens across steps 1–199, end-to-end throughput is
59.28 tok/GPU/s for G11 and
80.78 tok/GPU/s for G10.

`rounds.csv` contains every scalar RL, training, and performance field for all
200 paired steps. `summary.json` records formulas, distribution summaries,
bootstrap details, and the warmup-excluded token-normalized cross-check.
`plot_report.py` regenerates the consistency and mean-logp-diff figures from
the two authoritative launcher logs. Pass `--include-performance-plots` only
when preparing external W&B artifacts. `wandb_upload.py` uploads per-step
metrics, raw logs, validation JSON, and this result bundle to W&B.
