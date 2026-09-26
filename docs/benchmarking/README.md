# Benchmarking

RL-Kernel benchmarks track operator latency, memory behavior, and dispatch overhead.

See the [Hardware Benchmark Dashboard](hardware-dashboard.md) for the
cross-hardware reporting template and result-status rules.

Current benchmark entry points:

```bash
python scripts/run_profile_suite.py --smoke
python scripts/run_profile_suite.py --output reports/profile.csv
python benchmarks/profiler.py --format json --output reports/profile.json
python benchmarks/benchmark_sampling.py
python benchmarks/benchmark_grpo_op.py
python benchmarks/benchmark_pack.py --smoke
python benchmarks/benchmark_final_logit_softcap.py --output-dir reports/final-logit-softcap
python benchmarks/benchmark_rocm_ffn.py --help
python scripts/run_perf.py
```

The [final logit softcap benchmark](../operators/final-logit-softcap.md#benchmark)
compares eager PyTorch and Triton forward, backward, and forward+backward on the
same GPU, on CUDA or ROCm. It reuses the profiler's timer and writes JSON/Markdown
reports; it is a standalone entry point, not a `--workloads` selection.

## Unified ROCm deterministic FFN benchmark

The RCCL/FFN benchmark has one entry point. It measures the deterministic
Triton FFN across TP, CP, and sequence-parallel layouts, checks bitwise
exactness against deterministic TP=1, and compares four distributed paths at
the same topology: H100 official/deterministic and MI300X
official/deterministic. TP=1 latency is not part of the distributed figure.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_IB_DISABLE=1 PYTHONPATH=. \
  python benchmarks/benchmark_rocm_ffn.py \
  --warmup 5 --samples 20 --training-samples 10 \
  --output-dir benchmarks/results/pr325_rocm_mi300x
```

The checked-in report is
`benchmarks/results/pr325_rocm_mi300x/report.md`, with the machine-readable
data in `results.json` and the updated topology figure in
`distributed_ffn_overhead.png`.

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
