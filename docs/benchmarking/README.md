# Benchmarking

RL-Kernel benchmarks track operator latency, memory behavior, and dispatch overhead.

Current benchmark entry points:

```bash
python scripts/run_profile_suite.py --smoke
python scripts/run_profile_suite.py --output reports/profile.csv
python benchmarks/profiler.py --format json --output reports/profile.json
python benchmarks/logprob_cross_engine.py --smoke --output-dir reports/logprob_smoke
python benchmarks/benchmark_sampling.py
python benchmarks/benchmark_grpo_op.py
python scripts/run_perf.py
```

The automated profiler records one row per workload shape with:

- `tokens_per_sec`: active tokens divided by median latency.
- `tflops`: estimated operator FLOPs divided by median latency.
- `peak_vram_gb`: CUDA peak allocated memory during the measured run.
- `gpu_*`: detected device name, architecture, backend, driver, and memory.
- `status`: `pass`, `blocked`, or `oom`.

Useful presets:

```bash
# CPU-friendly validation for CI or local development.
python scripts/run_profile_suite.py --smoke --workloads logp-native

# CUDA logprob profiling with native and fused candidates.
python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 8,16,32 \
  --seq-lens 128,512 \
  --vocab-sizes 4096,128256 \
  --workloads logp-native,logp-fused \
  --output reports/logp_profile.csv

# Sampling baseline profiling.
python scripts/run_profile_suite.py \
  --workloads sampling-native \
  --batch-sizes 64,128,256 \
  --vocab-sizes 128256 \
  --top-k 50 \
  --top-p 0.9
```

## Train-Inference Logprob Cross-Benchmark

`benchmarks/logprob_cross_engine.py` validates the P0.3 identity that the same
model, prompt/completion tokens, and policy state produce aligned selected
logprobs across rollout and training engines.

The CI-friendly smoke path uses a deterministic tiny causal LM. It performs a
real greedy rollout and then teacher-forces the same tokens through the training
replay path:

```bash
python benchmarks/logprob_cross_engine.py \
  --smoke \
  --output-dir reports/logprob_cross_engine/smoke
```

For a local Hugging Face model, run rollout capture and training replay against
the same model path:

```bash
python benchmarks/logprob_cross_engine.py \
  --rollout-engine hf \
  --training-engine torch \
  --model /models/policy \
  --old-model /models/old-policy \
  --reference-model /models/reference \
  --tokenizer /models/policy \
  --prompts fixtures/prompts.jsonl \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --rollout-batch-size 8 \
  --training-micro-batch-size 4 \
  --output-dir reports/logprob_cross_engine/hf_vs_torch
```

When rollout happens in production vLLM/sglang infrastructure, export or convert
the rollout into the fixture schema and replay it offline:

```json
{
  "sequence_id": "run-42-sample-0",
  "prompt_token_ids": [1, 320, 42],
  "completion_token_ids": [934, 18],
  "rollout_logprobs": {
    "policy": [-0.12, -1.34],
    "old": [-0.13, -1.30],
    "ref": [-0.22, -1.41]
  },
  "completion_mask": [true, true],
  "metadata": {"weight_version": 42}
}
```

Each line can be one JSONL sequence, or the file can be a full JSON fixture with
a top-level `sequences` list. Replay it with:

```bash
python benchmarks/logprob_cross_engine.py \
  --rollout-engine fixture \
  --training-engine torch \
  --rollout-fixture artifacts/rollout_logprobs.jsonl \
  --model /models/policy \
  --old-model /models/old-policy \
  --reference-model /models/reference \
  --tokenizer /models/policy \
  --device cuda \
  --dtype bfloat16 \
  --output-dir reports/logprob_cross_engine/production_replay
```

The comparator maps rollout channels named `policy` or `current` to `--model`,
channels named `old` or `old_policy` to `--old-model`, and channels named `ref`
or `reference` to `--reference-model`. Channels without a configured replay
model are listed in `skipped_channels` rather than silently compared against the
wrong policy state.

The output directory contains:

- `rollout_fixture.json`: normalized prompt/completion tokens and rollout logprobs.
- `report.json`: pass/fail status, thresholds, metadata, and worst drift.
- `token_drifts.jsonl`: one row per compared active completion token.
- `summary.md`: compact benchmark summary for CI artifacts or issue comments.

Failures identify the sequence id, completion token index, absolute token
position, target token id, engine pair, dtype, and per-token drift so they can be
triaged against batch/cache/layout invariance, TP reductions, GRPO reduction
semantics, or fused logprob kernels.

When adding a new operator, document the benchmark command on the operator page and keep
the tested shapes close to the target RL workload.

## Adding Workloads

Profiler workloads are registered in `benchmarks/profiler.py` through
`WORKLOAD_REGISTRY`. To add an operator benchmark:

1. Add a small workload runner that builds deterministic inputs and calls the relevant
   `PerformanceProfiler.profile_*` method.
2. Register the runner under a stable CLI name in `WORKLOAD_REGISTRY`.
3. Add a focused test in `tests/test_profiler.py` that verifies the workload can be
   selected through `--workloads`.
4. Document a representative command and shape preset for the operator.

Workloads that require CUDA should report `status=blocked` with a clear note when the
requested device cannot run them, so CPU smoke validation still produces useful reports.
