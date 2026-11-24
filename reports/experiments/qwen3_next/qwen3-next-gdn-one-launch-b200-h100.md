# Qwen3-Next: the strict GDN recurrence in one launch per layer (B200, H100)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-gdn-one-launch-b200-h100.zip).

ZIP SHA-256: `004df9ff369680300d0e39adbbcbfe910b333db289a625d9aba5d1fb1045aa97`.

## Runs

| Run | Commit |
|---|---|
| Real-model chain and timings | `213a9bcd` (the fix `63d46013` merged with the speed test's first commits; their benchmark files do not touch the strict path) |
| EP4 chain | `213a9bcd` merged locally with `9eea76a2`, a commit that changes only the speed test's stock arms (tree `11834e36`; the run directories are named after that local merge) |
| Local B200 bit check | this PR's head `ae913959` |
| Timings before the fix | `f5100979` |

- **Hardware:** 4 × B200 (driver 580.126.20) and 4 × H100 (driver 580.173.02).
- **Software:** torch 2.13.0+cu130, vLLM 0.30.0, FlashInfer 0.6.18.post1.
- **Checkpoint:** `Qwen/Qwen3-Next-80B-A3B-Instruct` revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`.
- Every report has `tracked_tree_dirty: false`.

```bash
# bits and per-layer time of one GDN layer (one GPU), the ZIP's probe/gdn_one_launch_probe.py
TRITON_F32_DEFAULT=ieee PYTHONPATH=<src> python gdn_one_launch_probe.py <src>
python -m pytest tests/models/qwen3_next/check_qwen3_next_gdn_sequence.py   # fused == loop, values and gradients
# real-weight chain and batch invariance (four GPUs)
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 --checkpoint <ckpt> ...
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage bi --bi-suite full --previous <l2 run> ...
# timings: the end-to-end speed test, which follows as its own PR
python benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py --run <l2 run> --out <dir> --phases rollout,replay,train
```

## Bits

One GDN layer, the old per-token decode loop against the new one-launch forward, on the same inputs and state bank. All four cases are bit-identical on both GPUs: output, final state and untouched cache slots, with `output_max_abs` 0.0.

| Sequences | Tokens | Cache slots | B200 one launch | H100 one launch | H100 decode loop |
|---|---:|---:|---:|---:|---:|
| 1 × 64 | 64 | 65 | 1.14 ms | 1.06 ms | 29.7 ms |
| 1 × 1024 | 1024 | 65 | 3.19 ms | 4.58 ms | 470.3 ms |
| 1024 + 777 + 1500 + 64 | 3365 | 65 | 4.49 ms | 6.40 ms | 692.7 ms |
| 16 × 512 | 8192 | 362 | 2.15 ms | 2.89 ms | 265.3 ms |

- The decode loop is Python, one kernel launch per token, so its time is set by the host.
- The B200 bit check ran on a shared host, where it took 286-2588 ms; it is in the ZIP, but the H100 column comes from an idle node.
- `check_qwen3_next_gdn_sequence.py` and the other GDN, convolution and core checks give 43 passed on both GPUs. They compare the fused forward with the loop bit for bit, mixed and zero lengths included, and the gradients.

## Real-weight chain

| | B200 | H100 |
|---|---|---|
| TP4: `preflight`, `layers`, `l2`, `bi` (full) | passed; `bi-matrix.json` 97 / 97; chain report `passed=True`, 16 cells; gate 17 passed | passed; 97 / 97; `passed=True`, 16 cells; gate rc 0 |
| EP4: `preflight`, `layers`, `l2` | passed | passed |
| `tests/models/qwen3_next/` (CPU) | | 658 passed, 15 failed (the training environment's known `Invalid device id` group in `test_qwen3_next_training_checkpoint.py`) |

`l2` compares vLLM rollout with VIME recompute, so the fused forward keeps rollout and replay bit-identical with real weights.

## Time, strict path (TP4, B200)

Medians of the end-to-end speed test, before the fix (`f5100979`) and after it (`213a9bcd`):

| | before | after | H100 after |
|---|---:|---:|---:|
| rollout prefill, 64 / 256 / 1024-token prompt, batch 1 | 0.89 / 3.03 / 11.59 s | 0.18 / 0.20 / 0.30 s | 0.29 / 0.31 / 0.46 s |
| rollout prefill, 512-token prompts, batch 1 / 16 / 64 | 7.85 / 16.01 / 25.51 s | 0.25 / 0.79 / 2.15 s | 0.38 / 1.14 / 2.99 s |
| replay logprob, 64 / 256 / 1024 tokens, batch 1 | 0.98 / 2.80 / 9.65 s | 0.19 / 0.20 / 0.29 s | 0.29 / 0.29 / 0.42 s |
| replay logprob, 512 + 512 tokens, batch 1 / 4 / 16 | 9.78 / 9.77 / 10.89 s | 0.31 / 0.44 / 1.15 s | 0.42 / 0.58 / 1.45 s |
| decode, tokens/s, batch 1 | 5.55 | 5.69 | 3.51 |
| one training step, 64-token prompts, batch 4 | 80.5 s | 73.8 s | (does not fit) |
| one training step, 512 + 512 tokens, batch 4 | 596.6 s | 568.1 s | (does not fit) |

- Decode is unchanged: it was already one step per launch.
- Training barely moves, because the backward still differentiates the per-token loop.
- The strict TP4 actor needs about 120 GB per GPU for training, which an H100 does not have.
