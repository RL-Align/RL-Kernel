# Qwen3-Next distributed logprob parity, EP4 (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-distributed-logprob-ep4-b200.zip).

ZIP SHA-256: `c7530d29f2495d19d43b5424215af960795f62d9676b223fe41c27aa2616ba53`.

Measured at commit `1c4ae265` (`measured-commit.txt`) from clean clones, one
stage after another on one node with four B200 GPUs, torch 2.13.0+cu130,
vLLM 0.30.0, the official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`, in one environment
(`environment_sha256` `118384371a88…` in every `run.json`, the same as the
TP4 x EP1 chain). Topology TP4 x EP4 x ETP1 (`expert_model_parallel_size` 4,
`expert_tensor_parallel_size` 1): rank r owns experts `[128r, 128r + 128)` at
full width in both vLLM rollout and VIME replay.

```bash
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 --ep 4 \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> --python <venv>/bin/python
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <...-l2-...> --out chain-report.json
python tools/validation/models/qwen3_next/qwen3_next_publish_evidence.py --last <...-l2-...> \
    --chain-report chain-report.json --out artifacts/qwen3-next-distributed-logprob-ep4-b200
cd artifacts
zip -X -r qwen3-next-distributed-logprob-ep4-b200.zip qwen3-next-distributed-logprob-ep4-b200
```

The chain report and the published copy were written at `2d1fc2f3`, which only
adds the EP size to the chain report's disclaimer and backend profile. Extract
the ZIP before inspecting its JSON; `SHA256SUMS` covers every file.

| Stage | Result |
| --- | --- |
| `preflight` | passed: environment, mandatory operator cases and norm gates, device guards, TP4 probe, checkpoint hashes, rollout smoke (vLLM's own MoE) |
| `layers` | passed: GDN, attention, GDN-adapter and embedding/head TP4 gates and the EP4 MoE gate (`moe_ep`) on real weights; native scheduler on 16 cases |
| `l2` | passed: **16/16 cases bitwise** rollout vs replay, every layer, sub-node and state digest equal |

The `l2` comparison is raw bits of the 16 response tokens, BF16 logits and
FP32 logprobs over the full vocabulary, the embedding rows and, for all 48
layers, the five sub-node outputs, the layer output and the
convolution/recurrent or KV state. `chain-report.json` is the chain-gate report
(`config_fingerprint.ep` 4, 16 cells, `first_drift: null`).

## EP4 MoE gate (layer 0, real weights)

`layers/layers.json` pins the four rank reports by SHA-256. Every rank passed:

| Case | Checks (bitwise) |
| --- | --- |
| `hf-load-export-each-expert-once` | each of the 512 experts is loaded and exported by exactly one rank, equal to the checkpoint |
| `ep4-vs-ep1-chunk-reorder-train-{8,64,256,1024}` | routes identical on every rank; EP4 routed rows, input gradient and every owned expert's weight gradients equal all 512 experts on one GPU (EP1); full == chunked, reordered tokens == reordered output, first row alone; training forward == no-grad; output replicated |
| `empty-rank-participation` | a batch in which rank 3 owns no route: rows and gradients equal EP1 |

The EP4 block output differs from the TP4 (ETP4) block by at most 2.4e-4
(8 tokens) and 4.9e-4 (64-1024 tokens): one K = 512 GEMM per expert instead of
four K = 128 partials and a fixed-order sum. EP4 is therefore its own certified
profile (`qwen3-next-moe-ep4-v1`); parity is EP4 rollout == EP4 replay.

## EP4 vs TP4 x EP1, end to end

Not part of the acceptance contract. Against the TP4 x EP1 L2 run of the
full-model chain (#543, measured at `c67ec14`), all 16 cases generate the same
16 response tokens. Logits and logprobs are not bitwise equal: the largest
selected-logprob difference per case is 1.3e-3 to 0.32 and the largest logit
difference 0.66 to 7.2. The embedding and layer 0's recurrent state are equal
and layer 0's output is not, i.e. the first difference is after the layer-0
mixer, where the MoE block runs.

Site paths, host names and the scheduler are replaced by placeholders
(`<CHECKPOINT>`, `<RUN>`, `<PYTHON>`, `<SITE>`, `<HOME>`).

## Batch invariance at EP4 (stack top `90c09164`)

Beyond the `l2` acceptance above, the full batch-invariance suite of the
full-model chain (97 cells, see #544) was run at TP4 x EP4 x ETP1 from a clean
clone of `90c09164`, after a passed EP4 `preflight`, `layers` and `l2` at that
commit (the quick suite, 5 cells, passed first). Result: **97 / 97 cells
bitwise** (concurrency 13 / 13, interference 10 / 10, chunk 66 / 66, packing
8 / 8), `first_failure: null`, with the same cell definitions as the EP1 run
(`cells_sha256` equal). [Raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-batch-invariance-ep4-b200.zip),
SHA-256 `b3012ab60e42fe372a843b02c2aebbf1dcf3320e54c1cacb57df58d40782fe4d`
(`bi/bi_matrix.json`, `figure.png`). EP4 training and weight export stay
fail-closed (follow-up), so no EP4 training stage is claimed.
