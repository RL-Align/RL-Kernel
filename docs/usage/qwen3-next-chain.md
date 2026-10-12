# Qwen3-Next full-model chain (RFC #428 `full_model_chain`)

The chain checks, bitwise, that Qwen3-Next-80B-A3B-Instruct gives the same
response tokens, BF16 logits and FP32 selected logprobs in rollout (native vLLM
prefill and decode) and in replay (VIME/Megatron recompute), on the pinned
weights and again after two optimizer updates and a checkpoint reload. When a
case differs, the report names the first layer, the sub-node inside it and its
layer family.

## Requirements

* One node with four B200 GPUs, TP4 x EP1 (all 512 experts on every rank).
* The audited runtime: CUDA 13.0.88, torch 2.13.0+cu130, vLLM 0.30.0, Triton
  3.7.1, transformers 5.17.0 (`pip install -e ".[qwen3-next]"`), plus VIME and
  its patched Megatron-LM in the same virtual environment for the replay and
  training stages. The environment is fingerprinted at the start of every run;
  installing anything into it between stages invalidates the chain.
* The official checkpoint `Qwen/Qwen3-Next-80B-A3B-Instruct`, revision
  `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`; every shard is hashed before the
  first model load.
* Host memory: 256 GB for preflight, 512 GB for `layers`/`l2`, 1.28 TB for the
  training stages (`--memory` declares what the allocation grants), and shared
  storage for the training checkpoints (about 1.1 TB each) and HF exports.

## Running

Inside an allocation with four GPUs, from a clean checkout of the commit to measure:

```bash
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> \
    --python <venv>/bin/python --until l2
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --from train1 --until train2 \
    --previous <runs>/<...-l2-...> \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> --python <venv>/bin/python
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage train --previous <runs>/<...-train2-...> \
    --runs-root <runs> --checkpoint <checkpoint> --cuda-home <cuda-13.0> --python <venv>/bin/python
```

Each stage gets its own run directory with a clean clone of the commit, an
immutable `run.json` (identified by a fresh `execution_id`) and, once it
passes, a validated `result.json`; a later stage refuses a predecessor that
did not pass or that pins a different source, environment, checkpoint,
provider or topology identity.

| Stage | What it checks | Typical time |
| --- | --- | --- |
| `preflight` | environment and versions, the mandatory operator cases (`rl_engine/validation/models/qwen3_next/qwen3_next_operator_cases.json`) and norm C3/C4 gates, multi-GPU device guards, a four-rank TP probe, checkpoint hashes, a vLLM rollout smoke | 15 min |
| `layers` | the real-weight TP4 gates (GDN, attention, MoE, GDN engine adapters, embedding/LM head) and the native scheduler on 16 cases; certifies the shared provider profile | 25 min |
| `l2` | 16 cases (prompt 8/64/256/1024 x batch 1/4 x full/chunked prefill, 16 response tokens): rollout vs replay, raw bits, plus every layer and sub-node digest | 30 min |
| `train1`, `train2` | one optimizer update each, checkpoint, HF export, L2 on the exported weights | 3-4 h each |
| `train` | reload, L2, continuous and resumed step-3 controls, checkpoint fault injection | 4 h |

## Reports

```bash
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <runs>/<...-train-...> --out chain-report.json
bash ci/scripts/run_qwen3_next_chain_gate.sh chain-report.json
python tools/validation/models/qwen3_next/qwen3_next_publish_evidence.py --last <runs>/<...-train-...> \
    --chain-report chain-report.json --out docs/usage/evidence/qwen3-next-chain-<date>
```

`chain-report.json` has every field of the Qwen3-8B chain gate's report
(`rl_engine.validation.models.chain_gate.ChainGateReport`, schema `ws1-c10-c11-v5`)
with `schema_version: qwen3-next-chain-v1`; each case at each weight version is
a cell, and rollout/replay logits and logprobs are compared with
`atol = rtol = 0`. A failing chain reports `first_drift` in the dense form
`layer23.router:rollout->recompute`, plus `first_drift_layer`,
`first_drift_family` (`embedding`, `norm`, `gdn`, `full_attention`, `moe` or
`lm_head_logprob`) and `first_drift_cell`.

The sub-nodes are observed at the same five points of every decoder layer in
both engines (`rl_engine/models/qwen3_next/qwen3_next_subnode.py`): the normed
mixer input, the mixer output, the normed MLP input, the ten selected expert
ids and the MoE output, plus the embedding once per model.

`tools/validation/models/qwen3_next/qwen3_next_publish_evidence.py` copies the stage records and the
report, replaces site paths, host and user names with placeholders, refuses to
publish if any survive, and writes `SHA256SUMS`.

## Checks outside the default CPU suite

`tests/models/qwen3_next/check_qwen3_next_*.py` import vLLM and run in the self-hosted
`Qwen3-Next-provider-GPU` workflow and in `preflight`.
`tests/models/qwen3_next/check_qwen3_next_vime_weight_hooks.py`,
`tests/models/qwen3_next/test_qwen3_next_training_checkpoint.py` and
`tests/models/qwen3_next/test_qwen3_next_vllm_attention_metadata.py` need the training
environment (VIME, Megatron) and run there.
