# Qwen3-Next full-model chain (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-chain-b200.zip).

ZIP SHA-256: `76028dfea0a2b8698aa8d14d1f82e8da47df4ea11f9cd5240ebfb33094e527c1`.

Measured at commit `92e93a4` (`measured-commit.txt`) from clean clones, one
stage after another on nodes with four B200 GPUs, in one environment
(`environment_sha256` `118384371a88…` in every `run.json`), on the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. The archive holds every stage's
redacted `run.json`, `result.json` and evidence files, `chain-report.json`,
`figure.png`, `redaction.json`, the original README and `SHA256SUMS`, byte for
byte as published.

```bash
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> --python <venv>/bin/python
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <runs>/<...-train-...> --out chain-report.json
bash ci/scripts/run_qwen3_next_chain_gate.sh chain-report.json
python tools/validation/models/qwen3_next/qwen3_next_publish_evidence.py --last <runs>/<...-train-...> \
    --chain-report chain-report.json --out artifacts/qwen3-next-chain-b200
python benchmarks/models/qwen3_next/plot_qwen3_next_chain.py artifacts/qwen3-next-chain-b200
```

At `92e93a4` the runners lived in `scripts/` and `ci/`; the commands use the
current layout (the plot script is in the Qwen3-Next benchmark PR). The
stages ran the code of `92e93a4`; the published copy was made by the publisher
of `f431bad`, which only adds redaction rules and changes no measured byte.

| Stage | Checks | Result |
| --- | --- | --- |
| `preflight` | operator gates, device guards, TP4 NCCL probe, checkpoint identity, rollout smoke | passed |
| `layers` | per-layer TP4 contracts on real weights (GDN, attention, MoE, adapters, head) | passed |
| `l2` | vLLM rollout vs VIME recompute: tokens, BF16 logits, FP32 logprobs, layer and sub-node digests | 16 / 16 raw-bit |
| `train1`, `train2` | one RL update each, then L2 on the updated weights | 16 / 16 raw-bit each |
| `train` | checkpoint reload, L2 on the reloaded weights; reload == update 2; continuous == resumed step 3; checkpoint fault injection | 16 / 16 raw-bit, all sections passed |
| chain-gate report | `qwen3-next-chain-v1`, a superset of `ws1-c10-c11-v5` | 64 cells, `passed: true`, `first_drift: null` |

The 16 cases are prompt length 8 / 64 / 256 / 1024 × batch 1 / 4 × full /
chunked prefill, 16 response tokens. `train` was preempted and restarted;
before the published run the journal's earlier reload and controls were
retired into `recovery` (`train/reload.json`), so this run repeated all three
itself (see the known issue in the PR).

Site paths, host names, users and the scheduler are replaced by placeholders
(`<CHECKPOINT>`, `<RUN>`, `<PYTHON>`, `<SITE>`, `<HOME>`); `redaction.json`
counts them.

## Rerun on the stack top (`f05aaef6`)

After the stack was aligned with the reviewed layout (runners under
`tools/validation/models/qwen3_next/`, code under `rl_engine/models/qwen3_next/`),
the whole TP4 x EP1 chain was run again from clean clones of `f05aaef6`
(commit and tree in `measured-commit.txt` and every `run.json`), same environment
and checkpoint as above: [raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-chain-stacktop-b200.zip),
SHA-256 `bde7e27c5db8d92d1171f08d7858231e4f8dee4e4831f60c3103c95505245f09`
(`figure.png` is the chain figure, `figure-bi.png` the batch-invariance matrix
of the same commit).

| Stage | Result |
| --- | --- |
| `preflight`, `layers` | passed |
| `l2` | 16 / 16 raw-bit; every artifact bitwise identical to the `92e93a4` run |
| `train1`, `train2` | passed, 16 / 16 raw-bit on the updated weights each |
| `train` | passed: checkpoint reload, reload == update 2, continuous == resumed step 3, checkpoint fault injection; 16 / 16 raw-bit |
| chain-gate report | `qwen3-next-chain-v1`, 64 cells, `passed: true`, `first_drift: null` |

`train1` and `train2` were preempted and restarted several times; each stage is
published only as a complete, passed attempt, and before `train` the training
journal was checked to hold exactly the events of the two updates (the known
resume issue, see the PR). `train` ran in one attempt without a restart. The
published chain is a copy of the stage outputs made by the committed publisher
(`redaction.json` counts the placeholders; the ZIP contains no site detail).
