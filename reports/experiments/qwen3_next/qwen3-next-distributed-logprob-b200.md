# Qwen3-Next distributed logprob parity, TP4 (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-pr549-evidence/qwen3-next-distributed-logprob-b200.zip).

ZIP SHA-256: `a48bf29d19e40d22178185de81dcfacdd9138ca03f5dc347d23eed4a4fbd2cd9`.

One node with four B200 GPUs, torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7, the
official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`, 1024 tokens. Every run used a clean
clone with `rl_engine._C` built in place.

| Directory | Measured at | Produced by |
| --- | --- | --- |
| `parity/runs/<setting>/rank-{0..3}.json` | `99c244a` | `<NCCL env> torchrun --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_logprob_parity.py --checkpoint <ckpt> --output parity/runs/<setting>` |
| `parity/report.json` | `99c244a` | `python tools/validation/models/qwen3_next/qwen3_next_logprob_parity_compare.py --runs parity/runs/* --out parity/report.json` |

The 13 NCCL settings are the defaults plus `NCCL_ALGO` Ring and all-reduce Tree
(`allreduce:tree`; NCCL 2.29 has no Tree all-gather) × `NCCL_PROTO` LL, LL128,
Simple × one channel (`NCCL_MAX_NCHANNELS=1`) or the default channel count.
The archive preserves the published JSON bytes and their `SHA256SUMS`, including
the historical prior-art results used by the separate benchmark PR. The TP
parity measurements were collected at `99c244a`; prior-art data at `c5cea58`.

## Results

- Every TP4 cross-rank point -- embedding forward and weight gradient; GDN
  layer 0, full-attention layer 3 and MoE layer 0 forward and input gradient;
  selected logprob, LSE, entropy and logits gradient; the gathered sampling
  logits -- has one digest per rank across all 13 settings
  (`parity/report.json`: `deterministic_points_invariant: true`).
- TP4 selected logprob and LSE equal the TP1 computation on the same logits
  (`tp4_equals_tp1: true`); entropy does not (`entropy_tp4_equals_tp1: false`),
  as documented.
- Negative control: the same real attention `o_proj` partials summed with NCCL
  `all_reduce` give 5 different results across the 13 settings
  (`nccl_all_reduce_varies: true`).

Commands above use the current layout; at the measured commits, runners lived
directly under `tools/validation/models/`. To package a new run without adding raw JSON to the repository:

```bash
python tools/validation/models/qwen3_next/qwen3_next_publish_parity_evidence.py \
    --parity parity/runs --parity-report parity/report.json \
    --out artifacts/qwen3-next-distributed-logprob-b200
cd artifacts
zip -r qwen3-next-distributed-logprob-b200.zip qwen3-next-distributed-logprob-b200
```

## Post-review rerun (2026-10-12)

The relocated runners were executed from clean commit `1106e461e30d724c700eeffa5df7550fc1389556`
on four B200 GPUs with torch 2.13.0+cu130, vLLM 0.30.0, Triton 3.7.1,
FlashInfer 0.6.18.post1 and NCCL 2.29.7, using the checkpoint revision above.
The commands above reproduce this run with the relocated paths.

[Download new raw results, logs and provenance (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-pr549-evidence/qwen3-next-pr549-gpu-1106e461.zip).
ZIP SHA-256: `5fb49bc151fc0224dc92128c0279fb58998435cfa033b62565d9a62957be0e57`.
All 153 internal checksums were verified. Source paths are redacted; measured
values, source commits and the imported extension's SHA-256 are retained.

- All 233 mandatory operator cases passed with no skips; all four norm
  forward/gradient gates passed. The TP reduction guard also passed.
- Vocabulary-parallel logprob tests: 31 passed; four ROCm-only cases skipped.
  Two multiprocessing semaphore-cleanup warnings were recorded in the log.
- Across all 13 NCCL settings, all 17 deterministic points had one digest per
  rank. TP4 selected-token logprob and LSE equaled TP1, and full-vocabulary
  logits equaled gathered shards. The raw-NCCL negative control produced five
  distinct digests per rank. Entropy remains non-invariant across TP degrees.

The archive also contains the separate benchmark PR's rerun at `67a45169`.
This is operator and TP parity evidence; the full-model L2 and 97-cell batch
invariance runs were not repeated. Later commit `a9d0ba8f` changes artifact
hashing for Python 3.10 compatibility and has CPU/CI coverage, not a separate
GPU measurement.
