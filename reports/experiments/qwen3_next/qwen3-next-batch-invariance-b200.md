# Qwen3-Next batch invariance (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-pr549-evidence/qwen3-next-batch-invariance-b200.zip).

ZIP SHA-256: `a6f7245c83d4e2fbbc949ba915abaf05a9b71d5fbeaaf241f846eab5addeb878`.

Measured at commit `759755e` (`measured-commit.txt`) from clean clones, one
stage after another on one node with four B200 GPUs, in one environment
(`environment_sha256` `118384371a88…` in every `run.json`):

```bash
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> --python <venv>/bin/python
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage bi --bi-suite full --previous <runs>/<...-l2-...> \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> --python <venv>/bin/python
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <...-l2-...> --bi <...-bi-...> --out chain-report.json
python tools/validation/models/qwen3_next/qwen3_next_publish_evidence.py --last <...-l2-...> --also <...-bi-...> \
    --chain-report chain-report.json --out artifacts/qwen3-next-batch-invariance-b200
python benchmarks/models/qwen3_next/plot_qwen3_next_batch_invariance.py artifacts/qwen3-next-batch-invariance-b200
```

Commands use the current layout; at `759755e` the runners lived in `scripts/`, and the plot script is now in the Qwen3-Next benchmark PR.
The archive preserves the original evidence bytes and `SHA256SUMS`; the publisher
at `3025497` only added site redactions. Extract the ZIP before inspecting its JSON.

| Stage | Result |
| --- | --- |
| `preflight` | passed: environment, mandatory operator cases and norm gates, device guards, TP4 probe, checkpoint hashes, rollout smoke |
| `layers` | passed: GDN, attention, MoE, GDN-adapter and embedding/head TP4 gates on real weights; native scheduler on 16 cases |
| `l2` | passed: 16/16 cases bitwise rollout vs replay, every layer, sub-node and state digest equal |
| `bi` (full) | passed: **97/97 cells bitwise** |

| `bi` family | Cells | Passed | Compared with |
| --- | --- | --- | --- |
| concurrency (1-64 requests, target first or last) | 13 | 13 | the target alone, same chunk plan |
| interference (content, lengths, arrival, generation length) | 10 | 10 | the target alone, same chunk plan |
| chunk (prompts 63-1024, chunk 64/128, first chunk 1/63/65) | 66 | 66 | the replay engine's recompute |
| packing (replay THD batch with 1-7 neighbours) | 8 | 8 | the replay of the target alone |

Each comparison is raw bits of the target's 16 response tokens, its BF16
logits and FP32 logprobs over the full vocabulary, the embedding rows and, for
all 48 layers, the five sub-node outputs, the layer output and the
convolution/recurrent or KV state. `bi/bi_matrix.json` lists every cell;
`chain-report.json` is the chain-gate report with its `batch_invariance`
section.

Site paths, host names and the scheduler are replaced by placeholders
(`<CHECKPOINT>`, `<RUN>`, `<PYTHON>`, `<SITE>`, `<PATH>/name`, `<CGROUP>/file`).

## Rerun of the full suite on the stack top (`90c09164`)

The full 97-cell suite was run again from a clean clone of `90c09164` (TP4 x EP1,
the same checkpoint and environment; `preflight`, `layers` and `l2` passed first
and `l2` is bitwise identical to the `c67ec14` run). Result: **97 / 97 cells
bitwise** (concurrency 13 / 13, interference 10 / 10, chunk 66 / 66, packing 8 / 8),
no failed cell, `first_failure: null`. The published copy is inside the chain
archive of the full-model chain,
[`qwen3-next-chain-stacktop-b200.zip`](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-chain-stacktop-b200.zip)
(SHA-256 `bde7e27c5db8d92d1171f08d7858231e4f8dee4e4831f60c3103c95505245f09`,
`bi/bi_matrix.json`, `figure-bi.png`).
