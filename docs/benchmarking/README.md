# Benchmarking

RL-Kernel benchmarks track operator latency, memory behavior, and dispatch overhead.

Current benchmark entry points:

```bash
python benchmarks/benchmark_sampling.py
python benchmarks/benchmark_grpo_op.py
python scripts/run_perf.py
```

When adding a new operator, document the benchmark command on the operator page and keep
the tested shapes close to the target RL workload.
