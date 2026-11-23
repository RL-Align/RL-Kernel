# Benchmarks

Implementations are grouped by the measured component: `operators`, `layers`,
`backends`, `distributed`, `models`, or `e2e`. Shared profiling code lives in
`common`/`profiling`; tuning experiments belong in `tuning`.

Examples, from the repository root:

```bash
python benchmarks/operators/gemm/benchmark_det_gemm.py --help
python benchmarks/distributed/benchmark_rocm_collectives.py --help
python benchmarks/e2e/benchmark_stateless_executor.py --help
python -m benchmarks.backends.shared.triton.benchmark_ordered_reduction --dry-run
```

Some benchmark entry points require their target accelerator libraries even for
help output. Install the matching CUDA/ROCm environment before using them.
Existing top-level benchmark filenames remain compatibility launchers.

Write new local outputs to `artifacts/benchmarks/`; reviewed measurements belong
in `reports/experiments`, release evidence in `reports/releases`. Historic output
is preserved in `reports/archive`. Keep weights, workload, warmup, repetitions,
clocks, compiler flags, library versions and backend provenance identical for
before/after comparisons. Report both raw samples and median latency; report
throughput and peak memory separately. The refactor has no automatic performance
regression percentage gate.

The [ordered reduction benchmark](../docs/architecture/ordered-reduction.md#performance-comparison)
compares `torch.sum`, native `tl.sum`, and fixed-order Triton row/parallel callers.
`--include-gluon` adds shared Gluon callers using the selected target's warp/wave
width. Only H100 has measured results so far. Each run uses one
explicit execution configuration per schedule; parameter sweeps and diagnostic
candidates are excluded. Independent accuracy and same-tree checks, eager/graph
timing and hard per-case timeouts remain. Its CPU-safe dry run prints the exact
workload before allocating GPU time.
